"""Shared CPG / BLIF network helpers for the CPG-SNN training and inference scripts."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.set_float32_matmul_precision('high')
import snntorch as snn
from snntorch import surrogate
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin
from scipy.integrate import solve_ivp

class CPG_SNN(nn.Module):
    """
    Single-network SNN with gait flag in input, gated by LayerNorm.

    The original multi-gait design (gait one-hot concatenated per event)
    was correct in principle but failed because a constant per-step input
    saturates the LIF membrane: for any fc weight w > threshold*(1-beta)
    the neuron fires at every timestep regardless of which gait is active,
    making all four flags produce identical spike patterns.

    The fix is LayerNorm applied to the fc output BEFORE thresholding.
    LN zero-centres activations across the hidden dimension at each step.
    The four one-hot gait flags produce four different fc projections;
    after LN each is normalised relative to the others in that timestep,
    landing at different pre-threshold values and producing different
    firing patterns.  At inference (batch=1) LN uses per-sample stats
    over the hidden dimension — still effective, unlike BatchNorm which
    would collapse to running stats at batch=1.

    Architecture
    ------------
    Input  : (seq_len, B, n_in)
             n_in = N + 4 + n_gaits
                  = 4  one-hot neuron identity
                  + 2  sin/cos absolute phase
                  + 2  sin/cos relative (ISI) phase
                  + 4  one-hot gait flag   ← LN prevents saturation
    Layer 1: Linear(n_in, hidden)    → LayerNorm → Leaky LIF
    Layer 2: Linear(hidden, hidden)  → LayerNorm → Leaky LIF
    Layer 3: Linear(hidden, hidden)  → LayerNorm → Leaky LIF
    Readout: Linear(hidden, hidden)  → Leaky (threshold=1e9, analog)
             read at LAST timestep  → fc_out → (B, n_joints)

    Parameters
    ----------
    n_in    : int   N + 4 + n_gaits  (from build_dataset)
    hidden  : int   hidden layer width
    n_out   : int   number of joints
    n_gaits : int   kept for API / ONNX compatibility; not used in forward
    beta    : float LIF membrane decay
    """

    def __init__(self, n_in, hidden=128, n_out=8, n_gaits=4,
                 beta=0.9, spike_grad=None):
        super().__init__()
        spike_grad   = spike_grad or surrogate.fast_sigmoid(slope=25)

        self.fc1     = nn.Linear(n_in,   hidden)
        self.ln1     = nn.LayerNorm(hidden)
        self.lif1    = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2     = nn.Linear(hidden, hidden)
        self.ln2     = nn.LayerNorm(hidden)
        self.lif2    = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc3     = nn.Linear(hidden, hidden)
        self.ln3     = nn.LayerNorm(hidden)
        self.lif3    = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc_read = nn.Linear(hidden, hidden)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad,
                                  threshold=1e9)
        self.fc_out  = nn.Linear(hidden, n_out)

    def forward(self, x, gait_idx=None, return_recordings=False):
        """
        x        : (seq_len, B, n_in)  includes gait one-hot flag
        gait_idx : (B,) LongTensor  kept for API/ONNX compatibility;
                   the gait info is already inside x via the one-hot flag
        """
        m1 = self.lif1.init_leaky()
        m2 = self.lif2.init_leaky()
        m3 = self.lif3.init_leaky()
        mo = self.lif_out.init_leaky()
        spk1_rec, spk2_rec = [], []

        for t in range(x.shape[0]):
            s1, m1 = self.lif1(self.ln1(self.fc1(x[t])),  m1)
            s2, m2 = self.lif2(self.ln2(self.fc2(s1)),     m2)
            s3, m3 = self.lif3(self.ln3(self.fc3(s2)),     m3)
            _,  mo = self.lif_out(self.fc_read(s3),         mo)
            if return_recordings:
                spk1_rec.append(s1.detach())
                spk2_rec.append(s2.detach())

        output = self.fc_out(mo)

        if return_recordings:
            return output, torch.stack(spk1_rec), torch.stack(spk2_rec)
        return output


class LIFGeneralArray:
    def __init__(self, N, vth, du, dv, bias, u=0, v=0, ufloor=0, vfloor=0, refractory_period=0):
        self.vth = vth
        self.du = du
        self.dv = dv
        self.bias = bias
        self.u = np.ones(N) * u
        self.v = np.ones(N) * v
        self.ufloor = ufloor
        self.vfloor = vfloor
        self.refractory_period = refractory_period
        self.time_since_last_spike = np.zeros(N)
        self.N = N

    def next_step(self, current):
        self.u = self.u * (1 - self.du) + current
        self.v = self.v * (1 - self.dv) + self.u + self.bias

        refractory_mask = self.time_since_last_spike > 0
        self.v[refractory_mask] = 0
        self.time_since_last_spike = np.clip(self.time_since_last_spike - 1, 0, None)

        spike = self.v >= self.vth
        self.v[spike] = 0
        self.time_since_last_spike[spike] = self.refractory_period
        return spike.astype(np.float32)

    def reset(self, u=0, v=0):
        self.u.fill(u)
        self.v.fill(v)


class BurstingLIF:
    def __init__(self, N, vth_main, du_main, dv_main, refrac_main,
                 vth_fb, du_fb, dv_fb, refrac_fb, from_fb_weight, to_fb_weight):
        self.n_main = LIFGeneralArray(N, vth_main, du_main, dv_main, bias=0, u=0, v=0,
                                      ufloor=-vth_main * 100, vfloor=-vth_main * 100,
                                      refractory_period=refrac_main)
        self.n_fb = LIFGeneralArray(N, vth_fb, du_fb, dv_fb, bias=0, u=0, v=0,
                                    ufloor=-vth_fb * 100, vfloor=-vth_fb * 100,
                                    refractory_period=refrac_fb)

        self.input_2_feedback_neuron_weight = to_fb_weight
        self.feedback_2_input_neuron_weight = from_fb_weight
        self.fb_current = np.zeros(N)

    def forward(self, current):
        input_neuron_current = current + self.fb_current
        input_neuron_spike = self.n_main.next_step(input_neuron_current)

        to_fb_current = input_neuron_spike * self.input_2_feedback_neuron_weight
        fb_neuron_spike = self.n_fb.next_step(to_fb_current)
        self.fb_current = fb_neuron_spike * self.feedback_2_input_neuron_weight
        return input_neuron_spike, fb_neuron_spike

    def reset(self):
        self.n_main.reset()
        self.n_fb.reset()


class BLIF_CPG:
    def __init__(self, N=4, t_max=2000):
        vth_main = 100
        du_main = 0.1
        dv_main = 0.3
        refrac_main = 1

        vth_fb = 100
        du_fb = 1.0
        dv_fb = 0.
        refrac_fb = 1

        from_fb_weight = -1000000
        to_fb_weight = 10

        self.burstingNeuron1 = BurstingLIF(N, vth_main, du_main, dv_main, refrac_main,
                                           vth_fb, du_fb, dv_fb, refrac_fb,
                                           from_fb_weight, to_fb_weight)
        self.weight_matrix = []

        if N == 3:
            self.weight_matrix = np.asarray([
                [0.0, -523.65135942, -593.28982051],
                [-696.81822016, 0.0, -632.34680962],
                [-687.56816569, -577.5693762, 0.0],
            ])
        elif N == 4:
            self.weight_matrix = np.asarray([
                [0.0, -648.52905924, -449.60304695, -413.48426163],
                [-369.91504928, 0.0, -592.29635234, -568.0712858],
                [-412.08729881, -391.54918498, 0.0, -618.03381552],
                [-498.16458351, -655.01105883, -345.38277449, 0.0],
            ])
        elif N == 6:
            self.weight_matrix = np.asarray([
                [0.0, -375.86210512, -518.18703523, -371.82375498, -399.74231244, -487.45119873],
                [-531.99480471, 0.0, -489.1139223, -128.33470562, -404.33117771, -628.03347932],
                [-529.89653583, -418.34662835, 0.0, -543.37143674, -336.83773596, -679.12224243],
                [-674.09562904, -130.56007131, -297.35360394, 0.0, -363.1208234, -425.10847629],
                [-486.03391005, -386.7920052, -412.91478912, -437.7646991, 0.0, -288.47748806],
                [-112.97808475, -510.59115452, -367.63412082, -374.83106147, -393.86103887, 0.0],
            ])

        i_scale = 8.0
        self.i_app = np.ones((N, t_max)) * i_scale
        self.bn_spikes = np.zeros((N, t_max))
        self.inter_neuron_current = np.zeros(N)
        self.currents = np.zeros((N, t_max))
        self.t = 0

    def step(self):
        c_in = self.inter_neuron_current + self.i_app[:, self.t]
        self.currents[:, self.t] = c_in
        n_main, _ = self.burstingNeuron1.forward(c_in)
        self.bn_spikes[:, self.t] = n_main
        self.inter_neuron_current = self.weight_matrix @ n_main
        self.t += 1
        return n_main, self.burstingNeuron1.n_main.v, self.t


def run_blif_cpg(t_max=2000, N=4, cpg_start_time=100):
    network = BLIF_CPG(N=N, t_max=t_max)
    for _ in range(cpg_start_time):
        network.step()

    bn_spikes = []
    v_ms = []
    for _ in range(cpg_start_time, t_max):
        spikes, v_m, _ = network.step()
        bn_spikes.append(spikes)
        v_ms.append(v_m)

    bn_spikes = np.array(bn_spikes).T
    spike_times = []
    spike_neurons = []
    for t in range(cpg_start_time, t_max):
        for i in range(N):
            if bn_spikes[i][t - cpg_start_time]:
                spike_times.append(t)
                spike_neurons.append(i)
    return np.array(spike_times), np.array(spike_neurons)


def sigmoid(x, b=5.0, dsyn=-1.0):
    return 1.0 / (1.0 + np.exp(-b * (x - dsyn)))


def neuron_eqs(S, I, alpha, delta, Tf, Ts, Tus):
    vm, vf, vs, vus = S
    dvm = (-vm
           - alpha[0] * np.tanh(vf - delta[0])
           - alpha[1] * np.tanh(vs - delta[1])
           - alpha[2] * np.tanh(vs - delta[2])
           - alpha[3] * np.tanh(vus - delta[3])
           + I)
    dvf = (vm - vf) / Tf
    dvs = (vm - vs) / Ts
    dvus = (vm - vus) / Tus
    return [dvm, dvf, dvs, dvus]


def make_network(N, alpha, delta, g_inh, Iapp):
    asyn = g_inh * np.ones((N, N))
    np.fill_diagonal(asyn, 0.0)

    def network(t, S):
        dS = []
        Vs = np.array([S[i * 4 + 2] for i in range(N)])
        Isyn = asyn @ sigmoid(Vs)
        for i in range(N):
            dS.extend(neuron_eqs(
                S[i * 4:(i + 1) * 4], Iapp + Isyn[i],
                alpha, delta, 1.0, 50.0, 2500.0))
        return dS

    return network

def detect_burst_threshold(spike_times_n0, out_dir, bw_method=0.3):
    """
    Find the ISI threshold that separates within-burst spikes from
    between-burst gaps using the antimode of the log-ISI KDE.
    """
    isis     = np.diff(spike_times_n0)
    log_isis = np.log(isis + 1e-6)

    kde      = gaussian_kde(log_isis, bw_method=bw_method)
    x_eval   = np.linspace(log_isis.min(), log_isis.max(), 2000)
    density  = kde(x_eval)

    local_min_idx = argrelmin(density, order=20)[0]

    if len(local_min_idx) == 0:
        threshold = float(np.exp(np.median(log_isis)))
        print(f"  WARNING: no antimode found in log-ISI KDE; "
              f"using fallback threshold = {threshold:.1f}")
    else:
        mid      = (log_isis.min() + log_isis.max()) / 2.0
        best_idx = local_min_idx[np.argmin(np.abs(x_eval[local_min_idx] - mid))]
        threshold = float(np.exp(x_eval[best_idx]))
        print(f"  Burst ISI threshold (antimode) : {threshold:.1f} steps"
              f"  (log-ISI = {x_eval[best_idx]:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(log_isis, bins=80, density=True,
                 color="#457b9d", alpha=0.6, label="log-ISI histogram")
    axes[0].plot(x_eval, density, color="#e63946", lw=2, label="KDE")
    axes[0].axvline(np.log(threshold), color="#f4a261", lw=2, ls="--",
                    label=f"threshold = {threshold:.1f}")
    axes[0].set_xlabel("log(ISI)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Neuron 0 — log-ISI distribution & burst threshold")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(isis, bins=100, density=True,
                 color="#2a9d8f", alpha=0.6, label="ISI histogram")
    axes[1].axvline(threshold, color="#f4a261", lw=2, ls="--",
                    label=f"threshold = {threshold:.1f}")
    axes[1].set_xlabel("ISI (steps)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Neuron 0 — raw ISI distribution")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, np.percentile(isis, 99))

    plt.tight_layout()
    p = out_dir / "burst_threshold.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  [saved] {p}")

    return threshold


def get_burst_first_spikes(spike_times_n0, threshold, burnin_bursts=5):
    burst_starts = [spike_times_n0[0]]
    for i in range(1, len(spike_times_n0)):
        if spike_times_n0[i] - spike_times_n0[i - 1] > threshold:
            burst_starts.append(spike_times_n0[i])

    burst_starts = np.array(burst_starts, dtype=np.float32)
    print(f"  Neuron-0 bursts detected : {len(burst_starts)}"
          f"  (skipping first {burnin_bursts} for burn-in)")

    if len(burst_starts) <= burnin_bursts + 1:
        raise ValueError(
            f"Only {len(burst_starts)} bursts detected — increase tmax "
            f"or reduce burnin_bursts ({burnin_bursts}).")

    return burst_starts[burnin_bursts:]


def estimate_gait_period(spike_times, spike_neurons, out_dir,
                          N=4, burnin_bursts=5, kde_bw=0.3):
    t0        = spike_times[spike_neurons == 0]
    threshold = detect_burst_threshold(t0, out_dir, bw_method=kde_bw)
    burst_starts = get_burst_first_spikes(t0, threshold,
                                           burnin_bursts=burnin_bursts)
    inter_burst = np.diff(burst_starts)
    gait_period = float(np.median(inter_burst)) * 1.0

    print(f"  Inter-burst intervals  : {len(inter_burst)}")
    print(f"  Median gait period     : {gait_period:.1f} steps"
          f"  (= median inter-burst interval of neuron 0)")
    return gait_period, threshold


# ═══════════════════════════════════════════════════════════════════
# 4.  Phase encoding
# ═══════════════════════════════════════════════════════════════════

def encode_spike_events(spike_times, spike_neurons, gait_period, N=4, use_phase=True):
    """
    Per event → [one_hot_neuron(N), sin(φ_abs), cos(φ_abs),
                                    sin(φ_rel), cos(φ_rel)]  (N+4 dims).

    φ_abs = 2π · (t mod gait_period) / gait_period   — absolute phase
    φ_rel = 2π · ISI / gait_period                    — relative (ISI) phase

    Why two phase channels?
    -----------------------
    Within a burst, consecutive spikes arrive ~50 steps apart.
    Over the full gait_period (~4500 steps), 50 steps is only ~1% of a
    cycle, so φ_abs barely changes within a burst and gives the SNN
    almost no discriminative information between consecutive windows.

    φ_rel (ISI-based) varies strongly between within-burst events (~4°)
    and between-burst events (~56°), giving the SNN a clear signal for
    where in the burst structure each event falls — independent of when
    the trajectory started.  The combination of both channels gives full
    context: φ_abs says where we are in the global cycle; φ_rel says
    how densely spikes are arriving.

    The first event's ISI is set to gait_period (one full cycle) so its
    φ_rel = 2π, which maps to sin=0, cos=1 — a neutral initialisation.

    Gait flags are NOT added here — injected per-event during dataset
    construction so transition windows can mix flags freely.
    """
    # ── Absolute phase ────────────────────────────────────────────
    abs_phase = (2.0 * np.pi
                 * (spike_times % gait_period) / gait_period
                 ).astype(np.float32)

    # ── Relative phase (ISI-based) ────────────────────────────────
    isis = np.empty(len(spike_times), dtype=np.float32)
    isis[0]  = gait_period          # neutral: first event has no predecessor
    isis[1:] = np.diff(spike_times)
    rel_phase = (2.0 * np.pi * isis / gait_period).astype(np.float32)

    one_hot = np.zeros((len(spike_times), N), dtype=np.float32)
    one_hot[np.arange(len(spike_times)), spike_neurons] = 1.0

    if use_phase:
        base_feats = np.concatenate(
            [one_hot,
            np.sin(abs_phase)[:, None],
            np.cos(abs_phase)[:, None],
            #np.sin(rel_phase)[:, None],
            #np.cos(rel_phase)[:, None]
            ], axis=1)   # (E, N+4)
    else:
        base_feats = one_hot

    print(f"  Base feature matrix : {base_feats.shape}"
          f"  ({N} one-hot + sin/cos abs-phase + sin/cos rel-phase)"
          f"  [gait flag added per-event in build_dataset]")
    return base_feats, abs_phase
