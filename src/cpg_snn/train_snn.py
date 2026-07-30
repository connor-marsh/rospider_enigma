"""
CPG Spike Train → SNN → Robust Multi-Gait Joint Angle Prediction
=================================================================

Key design decisions
--------------------
1.  Bursting CPG period estimation.
    The CPG neurons fire in bursts (not single spikes per cycle).
    One gait cycle = one complete rotation through all four neuron
    bursts (neuron 0 burst → 1 → 2 → 3 → neuron 0 again).
    The gait period is therefore the inter-BURST interval of neuron 0:
    time from the first spike of one neuron-0 burst to the first spike
    of the next neuron-0 burst.

    The within-burst vs between-burst ISI threshold is found
    automatically from the antimode of the log-ISI kernel density
    estimate of neuron 0 — no manual tuning required.  A diagnostic
    plot of the ISI distribution and detected threshold is saved so
    the split can be visually verified.

    A second diagnostic plot overlays gait-table joint angles against
    detected burst boundaries so the phase→row correspondence can be
    verified before committing to training.

2.  Per-event gait flag.
    Every spike event carries its own 4-dim one-hot gait flag.
    During a gait transition the sliding window naturally contains a
    mix of old- and new-flag events as the buffer fills — matching
    inference exactly.

3.  Two window types (mixed at build time):
    a) Pure-gait windows   — all seq_len events share the same flag.
       Stride-1, then randomly subsampled to hit transition_frac.
    b) Transition windows  — single A→B switch at a point sampled
       from the last quarter of the window.  All 12 ordered pairs.
    Target = new-gait row at phase of last spike in both cases.

4.  No temporal-position cue.
    Pure windows subsampled in random order.  Transition switch points
    randomised per window.  Global shuffle of final dataset.

5.  Separate val tracking: val_pure MSE and val_trans MSE logged
    independently every epoch.

6.  Optuna sweep over seq_len, hidden, beta, transition_frac, lr.

Input feature vector per spike event  (length = N + 2 + n_gaits = 10):
    [one_hot_neuron(4),  sin(phase)(1),  cos(phase)(1),  gait_flag(4)]

Phase
-----
φ = 2π · (t mod gait_period) / gait_period
where gait_period = median inter-burst interval of neuron 0 (post burn-in).

CPG Integration
---------------
Both training data generation and deployment use the same chunk-based
BDF integrator (CPGChunkStepper).  Spike events are detected inline
during integration — identical to what the Raspberry Pi will run —
eliminating any train/deploy mismatch from batch vs streaming integration.

The warm-up phase (cpg_start_time steps) is run in chunks before
data collection begins, matching the deployment boot sequence exactly.
"""

import argparse
import itertools
import os
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


class LIFGeneralArray:
	def __init__(self,N,vth,du,dv,bias,u=0,v=0,ufloor=0,vfloor=0,refractory_period=0):
		# To get the SW CFG Model Comment out the left shift and multiplications
		self.vth = vth
		self.du = du
		self.dv = dv
		self.bias = bias
		self.u = np.ones(N)*u
		self.v = np.ones(N)*v
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

		# self.u = np.clip(self.u,self.ufloor,None)
		# self.v = np.clip(self.v,self.vfloor,None)

		spike = self.v >= self.vth
		self.v[spike] = 0
		self.time_since_last_spike[spike] = self.refractory_period
		return spike.astype(np.float32)

	def reset(self, u=0, v=0):
		self.u.fill(u)
		self.v.fill(v)

class BurstingLIF:
	def __init__(self,N,vth_main,du_main,dv_main,refrac_main,
			  vth_fb,du_fb,dv_fb,refrac_fb,from_fb_weight,to_fb_weight):
		self.n_main = LIFGeneralArray(N,vth_main,du_main,dv_main,bias=0,u=0,v=0,ufloor=-vth_main*100,vfloor=-vth_main*100,refractory_period=refrac_main)
		self.n_fb = LIFGeneralArray(N,vth_fb,du_fb,dv_fb,bias=0,u=0,v=0,ufloor=-vth_fb*100,vfloor=-vth_fb*100,refractory_period=refrac_fb)
		
		self.input_2_feedback_neuron_weight = to_fb_weight
		self.feedback_2_input_neuron_weight = from_fb_weight

		self.fb_current = np.zeros(N)

	def forward(self,current):
		input_neuron_current = current + self.fb_current
		input_neuron_spike = self.n_main.next_step(input_neuron_current)

		to_fb_current = input_neuron_spike*self.input_2_feedback_neuron_weight

		fb_neuron_spike = self.n_fb.next_step(to_fb_current)

		self.fb_current = fb_neuron_spike*self.feedback_2_input_neuron_weight

		return input_neuron_spike, fb_neuron_spike
	
	def reset(self):
		self.n_main.reset()
		self.n_fb.reset()

class BLIF_CPG:
    def __init__(self, N=4, t_max=2000):
        # Input Neuron Params
        vth_main = 100
        du_main = 0.1
        dv_main = 0.3
        refrac_main = 1

        # Feedback Neuron Params
        vth_fb = 100
        du_fb = 1.0
        dv_fb = 0.
        refrac_fb = 1

        # Weight Params
        from_fb_weight = -1000000
        to_fb_weight = 10

        # Number of CPG Neurons

        self.burstingNeuron1 = BurstingLIF(N,vth_main,du_main,dv_main,refrac_main,vth_fb,du_fb,dv_fb,refrac_fb,from_fb_weight,to_fb_weight)
        self.weight_matrix = []
        # Weight initialization

        ############# This configuration works best for 3 neuron CPG
        if N == 3:
            self.weight_matrix = np.asarray([[   0.         ,-523.65135942 ,-593.28982051],
                                        [-696.81822016  ,  0.         ,-632.34680962],
                                        [-687.56816569 ,-577.5693762   ,  0.        ]])

        ############# This configuration works best for 4 neuron CPG
        elif N==4:
            self.weight_matrix = np.asarray([[   0.         ,-648.52905924 ,-449.60304695 ,-413.48426163],
                                        [-369.91504928 ,   0.         ,-592.29635234 ,-568.0712858 ],
                                        [-412.08729881 ,-391.54918498 ,   0.         ,-618.03381552],
                                        [-498.16458351 ,-655.01105883 ,-345.38277449 ,   0.        ]])

        ############# This configuration works best for 6 neuron CPG
        elif N==6:
            self.weight_matrix = np.asarray([[   0.,         -375.86210512, -518.18703523, -371.82375498, -399.74231244,
                                        -487.45119873],
                                        [-531.99480471,    0.,         -489.1139223,  -128.33470562, -404.33117771,
                                        -628.03347932],
                                        [-529.89653583, -418.34662835,    0.,         -543.37143674, -336.83773596,
                                        -679.12224243],
                                        [-674.09562904, -130.56007131, -297.35360394,    0.,         -363.1208234,
                                        -425.10847629],
                                        [-486.03391005, -386.7920052,  -412.91478912, -437.7646991,     0.,
                                        -288.47748806],
                                        [-112.97808475, -510.59115452, -367.63412082, -374.83106147, -393.86103887,
                                            0.        ]])

        # Run the system

        i_scale = 8.

        self.i_app = np.ones((N,t_max))*i_scale

        self.bn_spikes = np.zeros((N,t_max))

        self.inter_neuron_current = np.zeros(N)

        self.currents = np.zeros((N,t_max))

        self.t = 0

    def step(self):
        c_in = self.inter_neuron_current + self.i_app[:,self.t]

        self.currents[:,self.t] = c_in

        n_main,_ = self.burstingNeuron1.forward(c_in)

        self.bn_spikes[:,self.t] = n_main

        self.inter_neuron_current = self.weight_matrix @ n_main

        self.t += 1

        return n_main, self.burstingNeuron1.n_main.v, self.t
    
def run_blif_cpg(t_max=2000, N=4, cpg_start_time=100):
    network = BLIF_CPG(N=N, t_max=t_max)

    for t in range(cpg_start_time):
        _, _, _ = network.step()

    bn_spikes = []
    v_ms = []
    for t in range(cpg_start_time, t_max):
        spikes, v_m, _ = network.step()
        bn_spikes.append(spikes)
        v_ms.append(v_m)

    
    bn_spikes = np.array(bn_spikes).T
    print(bn_spikes.shape)

    fig, axes = plt.subplots(1, 1)
    for i in range(N):
        axes.plot(bn_spikes[i], label=f"Neuron {i+1} spikes")
    axes.legend()
    plt.tight_layout()

    fig2, axes2 = plt.subplots(N, 2)
    for i in range(N):
        axes2[i,0].plot(v_ms[i])
        axes2[i,0].set_title(f"Input Current to Neuron {i+1}")

        axes2[i,1].plot(bn_spikes[i])
        axes2[i,1].set_title(f"Neuron {i+1} spikes")

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    plt.close(fig2)

    spike_times = []
    spike_neurons = []
    for t in range(cpg_start_time, t_max):
        for i in range(N):
            if bn_spikes[i][t-cpg_start_time]:
                spike_times.append(t)
                spike_neurons.append(i)
    return np.array(spike_times), np.array(spike_neurons)

# ═══════════════════════════════════════════════════════════════════
# 1.  CPG Dynamics (shared by both integrators)
# ═══════════════════════════════════════════════════════════════════

def sigmoid(x, b=5.0, dsyn=-1.0):
    return 1.0 / (1.0 + np.exp(-b * (x - dsyn)))


def neuron_eqs(S, I, alpha, delta, Tf, Ts, Tus):
    vm, vf, vs, vus = S
    dvm = (-vm
           - alpha[0] * np.tanh(vf  - delta[0])
           - alpha[1] * np.tanh(vs  - delta[1])
           - alpha[2] * np.tanh(vs  - delta[2])
           - alpha[3] * np.tanh(vus - delta[3])
           + I)
    dvf  = (vm - vf)  / Tf
    dvs  = (vm - vs)  / Ts
    dvus = (vm - vus) / Tus
    return [dvm, dvf, dvs, dvus]


def make_network(N, alpha, delta, g_inh, Iapp):
    asyn = g_inh * np.ones((N, N))
    np.fill_diagonal(asyn, 0.0)

    def network(t, S):
        dS   = []
        Vs   = np.array([S[i * 4 + 2] for i in range(N)])
        Isyn = asyn @ sigmoid(Vs)
        for i in range(N):
            dS.extend(neuron_eqs(
                S[i * 4:(i + 1) * 4], Iapp + Isyn[i],
                alpha, delta, 1.0, 50.0, 2500.0))
        return dS

    return network




# ═══════════════════════════════════════════════════════════════════
# 3.  Burst detection + gait period estimation
# ═══════════════════════════════════════════════════════════════════

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

def encode_spike_events(spike_times, spike_neurons, gait_period, N=4):
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

    base_feats = np.concatenate(
        [one_hot,
         np.sin(abs_phase)[:, None],
         np.cos(abs_phase)[:, None],
         #np.sin(rel_phase)[:, None],
         #np.cos(rel_phase)[:, None]
         ], axis=1)   # (E, N+4)

    print(f"  Base feature matrix : {base_feats.shape}"
          f"  ({N} one-hot + sin/cos abs-phase + sin/cos rel-phase)"
          f"  [gait flag added per-event in build_dataset]")
    return base_feats, abs_phase


# ═══════════════════════════════════════════════════════════════════
# 4b.  Gait table upsampling
# ═══════════════════════════════════════════════════════════════════

def upsample_gait_tables(gait_tables, gait_names, target_rows=None):
    """
    Upsample all gait tables to the same number of rows via cubic
    interpolation along the phase axis.

    With different row counts (wkF=54, bk=22, wkL=39, wkR=39), the
    phase → target mapping has different angular resolution per gait.
    Shorter tables produce coarser targets (larger quantisation error)
    that inflate the apparent loss for those gaits and make it harder
    for the SNN to learn smooth joint trajectories.

    Upsampling to a common row count (default: max across all tables)
    equalises target resolution without changing the gait shape —
    cubic interpolation preserves the continuous joint trajectory.

    The original gait tables are stored in the config so that the
    inference script can also upsample identically for GT comparison.

    Parameters
    ----------
    gait_tables  : list of (rows_i, J) float32 arrays
    gait_names   : list of str
    target_rows  : int or None  (None → use max row count)

    Returns
    -------
    upsampled    : list of (target_rows, J) float32 arrays
    target_rows  : int  (stored in config for inference)
    """
    from scipy.interpolate import interp1d

    if target_rows is None:
        target_rows = max(g.shape[0] for g in gait_tables)

    upsampled = []
    for gt, name in zip(gait_tables, gait_names):
        n_orig = gt.shape[0]
        if n_orig == target_rows:
            upsampled.append(gt.copy())
            print(f"      {name:>4s} : {n_orig} rows (unchanged)")
        else:
            x_orig = np.linspace(0.0, 1.0, n_orig)
            x_new  = np.linspace(0.0, 1.0, target_rows)
            interp = interp1d(x_orig, gt, axis=0, kind='cubic',
                              fill_value='extrapolate')
            gt_up  = interp(x_new).astype(np.float32)
            upsampled.append(gt_up)
            print(f"      {name:>4s} : {n_orig} → {target_rows} rows "
                  f"(cubic upsampled)")

    return upsampled, target_rows


# ═══════════════════════════════════════════════════════════════════
# 5.  Diagnostic: burst boundaries vs gait table
# ═══════════════════════════════════════════════════════════════════

def plot_burst_gait_overlay(spike_times, spike_neurons, gait_period,
                             threshold, gait_tables, gait_names, out_dir,
                             n_cycles=6, N=4):
    t0           = spike_times[spike_neurons == 0]
    isis_n0      = np.diff(t0)
    burst_starts = [t0[0]]
    for i in range(1, len(t0)):
        if isis_n0[i - 1] > threshold:
            burst_starts.append(t0[i])
    burst_starts = np.array(burst_starts)

    if len(burst_starts) < n_cycles + 1:
        n_cycles = len(burst_starts) - 1
    bs = burst_starts[:n_cycles + 1]

    colors_n  = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a0572", "#8ecae6"]
    colors_gt = ["#e63946", "#f4a261", "#2a9d8f", "#6a0572", "#8ecae6", "#ffb703"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)

    t_end = bs[-1] + gait_period * 0.1
    ax0   = axes[0]
    for i in range(N):
        mask = (spike_times >= bs[0]) & (spike_times <= t_end) & (spike_neurons == i)
        ax0.scatter(spike_times[mask], np.full(mask.sum(), i),
                    marker="|", s=120, lw=1.6, color=colors_n[i],
                    label=f"Neuron {i}")
    for b in bs:
        ax0.axvline(b, color="k", lw=1.0, alpha=0.5, ls="--")
    ax0.set_yticks(range(N))
    ax0.set_yticklabels([f"N{i}" for i in range(N)])
    ax0.set_title(f"Spike raster — first {n_cycles} gait cycles"
                  f"  (dashed = neuron-0 burst start)")
    ax0.legend(loc="upper right", fontsize=8, ncol=4)
    ax0.grid(True, axis="x", alpha=0.2)

    ax1 = axes[1]
    t_fine  = np.linspace(bs[0], bs[-1], 500)
    phase_f = (2.0 * np.pi * (t_fine % gait_period) / gait_period)

    for g_idx, (gt, name) in enumerate(zip(gait_tables, gait_names)):
        n_rows  = gt.shape[0]
        row_idx = (phase_f / (2.0 * np.pi) * n_rows).astype(int) % n_rows
        ax1.plot(t_fine, gt[row_idx, 0],
                 color=colors_gt[g_idx % len(colors_gt)],
                 lw=1.5, label=f"{name} J1")
    for b in bs:
        ax1.axvline(b, color="k", lw=1.0, alpha=0.5, ls="--")
    ax1.set_xlabel("Simulation time (steps)")
    ax1.set_ylabel("Joint 1 angle (°)")
    ax1.set_title("Gait-table joint 1 angle phase-indexed over gait cycles")
    ax1.legend(fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.25)

    plt.tight_layout()
    p = out_dir / "burst_gait_overlay.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {p}")


# ═══════════════════════════════════════════════════════════════════
# 6.  Helper: phase → gait-table row
# ═══════════════════════════════════════════════════════════════════

def phase_to_row(phase_rad, n_rows):
    return (phase_rad / (2.0 * np.pi) * n_rows).astype(int) % n_rows


# ═══════════════════════════════════════════════════════════════════
# 7.  Dataset builder
# ═══════════════════════════════════════════════════════════════════

def build_dataset(base_feats, event_phases, gait_tables,
                  seq_len=32, transition_frac=0.30, rng=None):
    """
    Build the full training dataset with pure-gait and transition windows.

    FiLM architecture change
    ------------------------
    The gait flag is NO LONGER concatenated into the per-event feature
    vector.  Instead the gait index is stored in `labels` and passed
    separately to the FiLM conditioning layer at forward time.

    Why: LIF neurons cannot discriminate a static, constant-per-window
    gait flag.  Any non-zero fc1 weight from the flag drives the LIF
    membrane to a saturated firing rate that encodes *magnitude* (always
    the same for a given gait) rather than gait *identity*.  The hidden
    spike pattern becomes identical for all gaits, making multi-gait
    decoding impossible regardless of depth or hidden size.

    With FiLM, the gait index conditions the analog readout membrane via
    learned per-gait scale (gamma) and shift (beta), completely bypassing
    the spike-discretisation bottleneck.

    Input feature per event: [one_hot_neuron(N), sin_abs, cos_abs,
                               sin_rel, cos_rel]  — N+4 dims, NO gait flag.

    Window label = target gait index (int).  For transition windows this
    is the *new* gait (the one whose table provides the target angles),
    matching inference where gait_idx switches atomically on gesture input.

    Parameters
    ----------
    base_feats      : (E, N+4)  float32  from encode_spike_events
    event_phases    : (E,)      float32  absolute phase rad per event
    gait_tables     : list of (target_rows, J) float32  (already upsampled)
    seq_len         : int
    transition_frac : float  fraction of windows that are transition type
    rng             : np.random.Generator

    Returns
    -------
    X         : (N_total, seq_len, N+4)   float32  — no gait flag
    y         : (N_total, J)              float32  normalised targets
    tgt_range : (min, max)
    pure_mask : (N_total,)  bool
    labels    : (N_total,)  int32         gait index for FiLM conditioning
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_gaits = len(gait_tables)
    E       = len(base_feats)
    J       = gait_tables[0].shape[1]
    N_win   = E - seq_len

    if N_win <= 0:
        raise ValueError(
            f"seq_len ({seq_len}) >= spike events ({E}). "
            "Increase tmax or reduce seq_len.")

    all_vals         = np.concatenate([g.flatten() for g in gait_tables])
    tgt_min, tgt_max = float(all_vals.min()), float(all_vals.max())

    def normalise(arr):
        return ((arr - tgt_min) / (tgt_max - tgt_min + 1e-8) * 2 - 1
                ).astype(np.float32)

    gait_norms  = [normalise(g) for g in gait_tables]
    last_phases = event_phases[seq_len - 1: seq_len - 1 + N_win]

    # ── Pure-gait windows ────────────────────────────────────────
    pure_X_parts, pure_y_parts, pure_lbl = [], [], []
    gait_onehot = np.eye(n_gaits, dtype=np.float32)
    for g in range(n_gaits):
        flag     = gait_onehot[g]
        n_rows_g = gait_norms[g].shape[0]
        flag_col = np.tile(flag, (seq_len, 1))          # (seq_len, n_gaits)
        windows  = np.stack(
            [np.concatenate([base_feats[s: s + seq_len], flag_col], axis=1)
             for s in range(N_win)])                    # (N_win, seq_len, N+4+n_gaits)
        row_idx  = phase_to_row(last_phases, n_rows_g)
        targets  = gait_norms[g][row_idx]
        pure_X_parts.append(windows)
        pure_y_parts.append(targets)
        pure_lbl.append(np.full(N_win, g, dtype=np.int32))

    pure_X   = np.concatenate(pure_X_parts, axis=0)
    pure_y   = np.concatenate(pure_y_parts, axis=0)
    pure_lbl = np.concatenate(pure_lbl,     axis=0)

    # ── Transition windows ────────────────────────────────────────
    # Per-event gait flag switches from flag_a to flag_b at a random
    # point in the last quarter of the window — matches inference where
    # the gesture sensor fires mid-stride.
    sw_low  = (3 * seq_len) // 4
    sw_high = seq_len - 1
    pairs = [(a, b) for a, b in itertools.product(range(n_gaits), repeat=2)
             if a != b]

    trans_X_parts, trans_y_parts, trans_lbl = [], [], []
    for (ga, gb) in pairs:
        flag_a   = gait_onehot[ga]
        flag_b   = gait_onehot[gb]
        n_rows_b = gait_norms[gb].shape[0]
        switch_pts = rng.integers(sw_low, sw_high + 1, size=N_win)
        windows = []
        for k in range(N_win):
            p        = int(switch_pts[k])
            flag_col = np.empty((seq_len, n_gaits), dtype=np.float32)
            flag_col[:p]  = flag_a
            flag_col[p:]  = flag_b
            windows.append(
                np.concatenate([base_feats[k: k + seq_len], flag_col], axis=1))
        windows = np.stack(windows)
        row_idx = phase_to_row(last_phases, n_rows_b)
        targets = gait_norms[gb][row_idx]
        trans_X_parts.append(windows)
        trans_y_parts.append(targets)
        trans_lbl.append(np.full(N_win, gb, dtype=np.int32))

    trans_X   = np.concatenate(trans_X_parts, axis=0)
    trans_y   = np.concatenate(trans_y_parts, axis=0)
    trans_lbl = np.concatenate(trans_lbl,     axis=0)

    # ── Subsample pure to hit transition_frac ────────────────────
    n_trans       = len(trans_X)
    n_pure_target = max(1, int(round(n_trans * (1.0 - transition_frac)
                                     / transition_frac)))
    n_pure_target = min(n_pure_target, len(pure_X))
    idx      = rng.permutation(len(pure_X))[:n_pure_target]
    pure_X   = pure_X[idx]
    pure_y   = pure_y[idx]
    pure_lbl = pure_lbl[idx]

    # ── Merge + shuffle ───────────────────────────────────────────
    X         = np.concatenate([pure_X,   trans_X],   axis=0).astype(np.float32)
    y         = np.concatenate([pure_y,   trans_y],   axis=0).astype(np.float32)
    labels    = np.concatenate([pure_lbl, trans_lbl], axis=0)
    pure_mask = np.concatenate(
        [np.ones(len(pure_X),  dtype=bool),
         np.zeros(len(trans_X), dtype=bool)], axis=0)
    shuf = rng.permutation(len(X))
    X, y, labels, pure_mask = (X[shuf], y[shuf], labels[shuf], pure_mask[shuf])

    actual_frac = (~pure_mask).sum() / len(pure_mask)
    print(f"  Pure windows       : {pure_mask.sum():>8,}")
    print(f"  Transition windows : {(~pure_mask).sum():>8,}"
          f"  (actual frac = {actual_frac:.2f})")
    print(f"  Total              : {len(X):>8,}")
    print(f"  Feature dim        : {X.shape[2]}  (one-hot + 2 phase + {n_gaits} gait flag)")

    return X, y, (tgt_min, tgt_max), pure_mask, labels


# ═══════════════════════════════════════════════════════════════════
# 8.  Dataset / DataLoader
# ═══════════════════════════════════════════════════════════════════

class GaitDataset(Dataset):
    def __init__(self, X, y, labels=None):
        self.X      = torch.tensor(X, dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.float32)
        # labels: gait index per window, used for weighted loss
        self.labels = (torch.tensor(labels, dtype=torch.long)
                       if labels is not None
                       else torch.zeros(len(X), dtype=torch.long))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.labels[idx]


def snn_collate(batch):
    X, y, lbl = zip(*batch)
    return torch.stack(X).permute(1, 0, 2), torch.stack(y), torch.stack(lbl)


def make_loader(ds, batch_size, shuffle):
    if len(ds) == 0:
        return []
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=snn_collate, num_workers=0)


def train_val_test_split(X, y, pure_mask, labels,
                          val_frac=0.15, test_frac=0.10, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n   = len(X)
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val - n_test
    tr_idx   = idx[:n_train]
    val_idx  = idx[n_train: n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def split_mask(indices):
        pm = pure_mask[indices]
        return indices[pm], indices[~pm]

    vp_idx, vt_idx = split_mask(val_idx)
    tp_idx, tt_idx = split_mask(test_idx)

    train_ds      = GaitDataset(X[tr_idx],  y[tr_idx],  labels[tr_idx])
    val_pure_ds   = GaitDataset(X[vp_idx],  y[vp_idx],  labels[vp_idx])
    val_trans_ds  = GaitDataset(X[vt_idx],  y[vt_idx],  labels[vt_idx])
    test_pure_ds  = GaitDataset(X[tp_idx],  y[tp_idx],  labels[tp_idx])
    test_trans_ds = GaitDataset(X[tt_idx],  y[tt_idx],  labels[tt_idx])

    print(f"  Train           : {len(train_ds):>8,}")
    print(f"  Val   pure      : {len(val_pure_ds):>8,}")
    print(f"  Val   trans     : {len(val_trans_ds):>8,}")
    print(f"  Test  pure      : {len(test_pure_ds):>8,}")
    print(f"  Test  trans     : {len(test_trans_ds):>8,}")

    return train_ds, val_pure_ds, val_trans_ds, test_pure_ds, test_trans_ds


# ═══════════════════════════════════════════════════════════════════
# 9.  SNN Model
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# 10.  Training
# ═══════════════════════════════════════════════════════════════════

def make_gait_weighted_criterion(gait_tables_orig, device):
    """
    Per-gait MSE weight inversely proportional to angular range.

    Gaits with a larger joint-angle range (wkF, wkL, wkR) produce
    larger absolute errors for the same fractional error, but the SNN
    is trained on NORMALISED targets [−1, 1] so MSE is already scale-
    equalised.  The residual imbalance comes from the fact that gaits
    with FEWER original table rows have coarser phase targets, making
    them harder to fit precisely.

    Weight = target_rows / row_count_i
    This upweights gaits that were upsampled more aggressively (bk: 22→54
    gets weight 2.45×) and keeps wkF at 1.0, so the loss gradient is
    proportional to the difficulty rather than the raw row count.

    Returns a callable loss(pred, target, gait_labels) that computes
    per-sample MSE weighted by the label's gait weight.
    """
    max_rows = max(g.shape[0] for g in gait_tables_orig)
    weights  = torch.tensor(
        [max_rows / g.shape[0] for g in gait_tables_orig],
        dtype=torch.float32, device=device)
    print(f"  Gait loss weights: "
          + "  ".join(f"g{i}={weights[i].item():.2f}"
                      for i in range(len(gait_tables_orig))))

    def weighted_criterion(pred, target, gait_labels):
        # pred, target : (B, J)
        # gait_labels  : (B,) int  — new gait index for each window
        w   = weights[gait_labels]           # (B,)
        mse = ((pred - target) ** 2).mean(dim=1)  # (B,)
        return (w * mse).mean()

    return weighted_criterion


def train_epoch(model, loader, optimizer, criterion, device,
                weighted=False):
    model.train()
    total = 0.0
    for batch in loader:
        X, y, glbl = batch
        X, y, glbl = X.to(device), y.to(device), glbl.to(device)
        optimizer.zero_grad()
        pred = model(X, glbl)                  # FiLM: pass gait index
        loss = criterion(pred, y, glbl) if weighted else criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, weighted=False):
    model.eval()
    if not loader:
        return float("nan")
    total = 0.0
    for batch in loader:
        X, y, glbl = batch
        X, y, glbl = X.to(device), y.to(device), glbl.to(device)
        pred  = model(X, glbl)                 # FiLM: pass gait index
        loss  = criterion(pred, y, glbl) if weighted else criterion(pred, y)
        total += loss.item()
    return total / len(loader)


def run_training(model, train_loader, val_pure_loader, val_trans_loader,
                 optimizer, scheduler, criterion, device,
                 epochs, out_dir, weighted=False):
    best_val  = float("inf")
    best_path = out_dir / "best_model.pt"
    history   = {"train": [], "val_pure": [], "val_trans": []}

    def run_epoch(epoch, best_val):
        tl = train_epoch(model, train_loader, optimizer, criterion,
                         device, weighted=weighted)
        vp = eval_epoch(model, val_pure_loader,  criterion, device,
                        weighted=weighted)
        vt = eval_epoch(model, val_trans_loader, criterion, device,
                        weighted=weighted)
        scheduler.step()

        history["train"].append(tl)
        history["val_pure"].append(vp)
        history["val_trans"].append(vt)

        valid_vals   = [v for v in (vp, vt) if not np.isnan(v)]
        val_combined = float(np.mean(valid_vals)) if valid_vals else float("inf")

        flag = ""
        if val_combined < best_val:
            best_val = val_combined
            torch.save(model.state_dict(), best_path)
            flag = " ✓"

        if epoch % 10 == 0 or epoch == 1:
            vp_s = f"{vp:.6f}" if not np.isnan(vp) else "       nan"
            vt_s = f"{vt:.6f}" if not np.isnan(vt) else "       nan"
            print(f"  {epoch:>6}  {tl:>10.6f}  {vp_s:>10}"
                  f"  {vt_s:>10}"
                  f"  {optimizer.param_groups[0]['lr']:>8.2e}{flag}")
            
        return best_val

    print(f"\n  {'Epoch':>6}  {'Train':>10}  {'Val-Pure':>10}"
          f"  {'Val-Trans':>10}  {'LR':>8}")
    print("  " + "-" * 58)

    try:
        for epoch in range(1, epochs + 1):
            best_val = run_epoch(epoch)
    except KeyboardInterrupt:
        print("\n  [interrupt] Ctrl+C received; stopping.")

    print("  " + "-" * 58)
    return best_val, history


# ═══════════════════════════════════════════════════════════════════
# 11.  ONNX export
# ═══════════════════════════════════════════════════════════════════

def export_to_onnx(model, seq_len, n_in, out_dir, device,
                    inference_config=None):
    import json

    model.eval()
    dummy_x   = torch.zeros(seq_len, 1, n_in, device=device)
    onnx_path = out_dir / "cpg_snn.onnx"
    torch.onnx.export(
        model, dummy_x, str(onnx_path),
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=["spike_window"],
        output_names=["joint_angles"],
        dynamic_axes={"spike_window": {1: "batch_size"},
                      "joint_angles": {0: "batch_size"}})
    print(f"  [saved] ONNX → {onnx_path}")

    try:
        import onnxruntime as ort
        sess    = ort.InferenceSession(str(onnx_path),
                                       providers=["CPUExecutionProvider"])
        pt_out  = model(dummy_x).squeeze(0).detach().cpu().numpy()
        ort_out = sess.run(["joint_angles"],
                           {"spike_window": dummy_x.cpu().numpy()})[0].squeeze(0)
        diff    = float(np.abs(pt_out - ort_out).max())
        print(f"  PyTorch vs ONNX max diff : {diff:.2e}"
              f"  ({'OK' if diff < 1e-4 else 'WARNING'})")
    except ImportError:
        print("  onnxruntime not installed — skipping sanity check.")

    if inference_config is not None:
        cfg_path = out_dir / "cpg_snn_config.json"
        clean = {}
        for k, v in inference_config.items():
            if isinstance(v, list):
                clean[k] = v
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = float(v) if isinstance(v, (int, float)) else v
        # Also save chunk_size so the deployment stepper uses the same value
        clean["chunk_size"] = int(inference_config.get("chunk_size", 50))
        with open(cfg_path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"  [saved] config → {cfg_path}")
        print(f"          gait_period     = {clean.get('gait_period', 0):.1f}")
        print(f"          burst_threshold = {clean.get('burst_threshold', 0):.1f}")
        print(f"          chunk_size      = {clean.get('chunk_size', 0)}")
        print(f"          global_min/max  = "
              f"{clean.get('global_min', 0):.1f} / "
              f"{clean.get('global_max', 0):.1f}")

    return onnx_path


# ═══════════════════════════════════════════════════════════════════
# 12.  Visualisation
# ═══════════════════════════════════════════════════════════════════

def plot_cpg_vm(vm_record, out_dir, n_show=30_000):
    """
    Plot CPG membrane potentials.

    Accepts either a scipy OdeSolution object (legacy) or a dict with
    keys 't' and 'y' as returned by run_cpg_chunked.
    """
    # Support both scipy sol objects and our vm_record dict
    t_axis = vm_record["t"] if isinstance(vm_record, dict) else vm_record.t
    y_mat  = vm_record["y"] if isinstance(vm_record, dict) else vm_record.y
    N      = y_mat.shape[0] // 4

    # Clip to n_show points
    n_pts  = min(n_show, t_axis.shape[0])
    t_plot = t_axis[:n_pts]
    y_plot = y_mat[:, :n_pts]

    fig, axes = plt.subplots(N, 1, figsize=(14, 8), sharex=True)
    colors    = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a0572", "#8ecae6"]
    if N == 1:
        axes = [axes]
    for i in range(N):
        axes[i].plot(t_plot, y_plot[i * 4, :],
                     color=colors[i], lw=0.9)
        axes[i].set_ylabel(f"CPG {i}\n$v_m$", fontsize=9)
        axes[i].axhline(-2.0, color="k", ls="--", lw=0.7, alpha=0.5,
                         label="threshold")
        axes[i].grid(True, alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel(f"Time (first {n_pts} steps)")
    plt.suptitle("CPG Membrane Potentials — chunk-based integrator", fontsize=12)
    plt.tight_layout()
    p = out_dir / "cpg_vm.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  [saved] {p}")


def plot_spike_events(spike_times, spike_neurons, gait_period,
                      out_dir, n_show=3_000, N=4):
    mask   = spike_times <= spike_times[0] + n_show
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a0572", "#8ecae6"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                    sharex=True, height_ratios=[3, 1])
    for i in range(N):
        idx = np.where((spike_neurons == i) & mask)[0]
        ax1.scatter(spike_times[idx], np.full(len(idx), i),
                    marker="|", s=150, lw=1.8,
                    color=colors[i], label=f"Neuron {i}")
    ax1.set_yticks(range(N))
    ax1.set_yticklabels([f"CPG {i}" for i in range(N)])
    ax1.set_title(f"Spike Events  (gait_period ≈ {gait_period:.0f} steps)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, axis="x", alpha=0.2)
    t_show = spike_times[mask]
    phase  = np.degrees(2.0 * np.pi * (t_show % gait_period) / gait_period)
    ax2.plot(t_show, phase, color="#6a0572", lw=1.2)
    ax2.set_ylabel("Phase (°)"); ax2.set_xlabel("Time")
    ax2.set_title("Gait phase at each spike event")
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    p = out_dir / "spike_events.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  [saved] {p}")


def plot_training_curves(history, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs  = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"],     label="Train",       lw=2,
            color="#457b9d")
    ax.plot(epochs, history["val_pure"],  label="Val (pure)",  lw=2,
            color="#2a9d8f", ls="--")
    ax.plot(epochs, history["val_trans"], label="Val (trans)", lw=2,
            color="#e63946", ls=":")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training — pure vs transition validation loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "training_curves.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  [saved] {p}")


def plot_inference(model, dataset, device, out_dir, n_joints, n_gaits,
                   sample_idx=0):
    model.eval()
    X_np, y_np, lbl = dataset[sample_idx]
    gait_idx_t = lbl.unsqueeze(0).to(device)          # (1,)
    X_in       = X_np.unsqueeze(1).to(device)          # (seq_len, 1, n_in)
    with torch.no_grad():
        pred, spk1, spk2 = model(X_in, gait_idx_t,
                                  return_recordings=True)
    pred_np = pred.squeeze(0).cpu().numpy()
    y_np    = y_np.numpy()
    spk1_np = spk1[:, 0, :].cpu().numpy()
    spk2_np = spk2[:, 0, :].cpu().numpy()
    N       = 4
    onehot  = X_np[:, :N].numpy()
    sin_ph  = X_np[:, N].numpy()
    cos_ph  = X_np[:, N + 1].numpy()
    #sin_rel = X_np[:, N + 2].numpy()
    #cos_rel = X_np[:, N + 3].numpy()
    gait_name = f"Gait {lbl.item()}"

    T, hidden = spk1_np.shape
    n_show    = min(24, hidden)
    show_idx  = np.random.choice(hidden, n_show, replace=False)
    colors    = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a0572", "#8ecae6"]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    for i in range(N):
        t_ev = np.where(onehot[:, i] > 0.5)[0]
        ax0.scatter(t_ev, np.full_like(t_ev, i),
                    marker="|", s=120, color=colors[i % len(colors)])
    ax0.set_yticks(range(N))
    ax0.set_yticklabels([f"CPG {i}" for i in range(N)])
    ax0.set_title(f"Input spike-event window (seq_len={T})  [{gait_name}]")
    ax0.grid(True, axis="x", alpha=0.2)
    ax0b = ax0.twinx()
    ax0b.plot(sin_ph,  color="purple", lw=1.2, alpha=0.6, ls="--", label="sin(φ_abs)")
    ax0b.plot(cos_ph,  color="gray",   lw=1.2, alpha=0.6, ls=":",  label="cos(φ_abs)")
    #ax0b.plot(sin_rel, color="orange", lw=1.2, alpha=0.6, ls="-",  label="sin(φ_rel)")
    ax0b.set_ylabel("Phase channels", fontsize=8)
    ax0b.legend(fontsize=7, loc="upper right")

    # FiLM weights for this gait
    ax_gait = fig.add_subplot(gs[0, 1])
    N_feat  = N + 2 #Its 2 now caused rmeoved relative phase # 4   # one-hot + 4 phase channels
    gait_fl = X_np[:, N_feat:].numpy()  # (seq_len, n_gaits)
    for g in range(n_gaits):
        ax_gait.plot(gait_fl[:, g], label=f"Gait {g}", lw=1.5,
                     color=colors[g % len(colors)])
    ax_gait.set_title(f"Per-event gait flag — {gait_name}")
    ax_gait.set_xlabel("Event index"); ax_gait.set_ylabel("Flag value")
    ax_gait.set_ylim(-0.1, 1.1); ax_gait.legend(fontsize=7)
    ax_gait.grid(True, alpha=0.2)

    ax1 = fig.add_subplot(gs[1, 0])
    for row, nid in enumerate(show_idx):
        t_spk = np.where(spk1_np[:, nid] > 0.5)[0]
        ax1.scatter(t_spk, np.full_like(t_spk, row),
                    marker="|", s=60, color="#457b9d")
    ax1.set_title(f"Hidden Layer 1 ({n_show}/{hidden} neurons)")
    ax1.set_xlabel("Event index"); ax1.set_ylabel("Neuron")
    ax1.grid(True, axis="x", alpha=0.2)

    ax2 = fig.add_subplot(gs[1, 1])
    for row, nid in enumerate(show_idx):
        t_spk = np.where(spk2_np[:, nid] > 0.5)[0]
        ax2.scatter(t_spk, np.full_like(t_spk, row),
                    marker="|", s=60, color="#f4a261")
    ax2.set_title(f"Hidden Layer 2 ({n_show}/{hidden} neurons)")
    ax2.set_xlabel("Event index"); ax2.set_ylabel("Neuron")
    ax2.grid(True, axis="x", alpha=0.2)

    ax3 = fig.add_subplot(gs[2, :])
    x = np.arange(n_joints); w = 0.35
    ax3.bar(x - w / 2, y_np,    w, label="True",      color="#457b9d", alpha=0.85)
    ax3.bar(x + w / 2, pred_np, w, label="Predicted", color="#f4a261", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"J{i+1}" for i in range(n_joints)],
                         rotation=45, ha="right")
    ax3.set_ylabel("Angle (normalised)")
    ax3.set_title("True vs Predicted Joint Angles")
    ax3.legend(); ax3.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"SNN Inference — Sample #{sample_idx}  ({gait_name})",
                 fontsize=13)
    p = out_dir / "snn_inference.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {p}")


def plot_gait_reconstruction(model, X, y, pure_mask, labels,
                              device, out_dir, n_joints, tgt_range,
                              gait_names, n_samples=300):
    model.eval()
    tgt_min, tgt_max = tgt_range
    scale   = (tgt_max - tgt_min) / 2.0
    shift   = (tgt_max + tgt_min) / 2.0
    n_gaits = len(gait_names)
    colors  = ["#e63946", "#f4a261", "#2a9d8f", "#6a0572", "#8ecae6", "#ffb703"]
    TRUE_C  = "#457b9d"

    def predict_batch(indices):
        X_t   = torch.tensor(X[indices]).permute(1, 0, 2).to(device)
        lbl_t = torch.tensor(labels[indices], dtype=torch.long).to(device)
        with torch.no_grad():
            pred = model(X_t, lbl_t).cpu().numpy()
        return y[indices] * scale + shift, pred * scale + shift

    rmse_pure  = np.full((n_gaits, n_joints), np.nan)
    rmse_trans = np.full((n_gaits, n_joints), np.nan)

    for g, name in enumerate(gait_names):
        for wtype, mask_cond, suffix, color, rmse_arr in [
            ("pure",  pure_mask,  "pure",  TRUE_C,        rmse_pure),
            ("trans", ~pure_mask, "trans", colors[g % len(colors)], rmse_trans),
        ]:
            idx = np.where((labels == g) & mask_cond)[0]
            if len(idx) == 0:
                continue
            idx = idx[:n_samples]
            true_arr, pred_arr = predict_batch(idx)

            cols = min(4, n_joints)
            rows = int(np.ceil(n_joints / cols))
            fig, axes = plt.subplots(rows, cols,
                                      figsize=(5 * cols, 3 * rows),
                                      squeeze=False)
            for j in range(n_joints):
                ax   = axes[j // cols][j % cols]
                rmse = np.sqrt(np.mean((pred_arr[:, j] - true_arr[:, j]) ** 2))
                rmse_arr[g, j] = rmse
                ax.plot(true_arr[:, j], label="GT",   color=TRUE_C, lw=1.8)
                ax.plot(pred_arr[:, j], label="Pred", color=color,
                        lw=1.5, ls="--", alpha=0.9)
                err = np.abs(pred_arr[:, j] - true_arr[:, j])
                ax.fill_between(range(len(idx)),
                                pred_arr[:, j] - err,
                                pred_arr[:, j] + err,
                                color=color, alpha=0.12)
                ax.set_title(f"J{j+1}  RMSE={rmse:.2f}°", fontsize=9)
                ax.set_xlabel("Window", fontsize=8)
                ax.set_ylabel("Angle (°)", fontsize=8)
                ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
            for j in range(n_joints, rows * cols):
                axes[j // cols][j % cols].set_visible(False)
            plt.suptitle(f"{name} — {wtype} ({len(idx)} samples)",
                         fontsize=11, fontweight="bold")
            plt.tight_layout()
            p = out_dir / f"/recons/recon_{name}_{suffix}.png"
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            print(f"  [saved] {p}")

    vmax = np.nanmax(np.stack([rmse_pure, rmse_trans]))
    fig, axes = plt.subplots(1, 2,
                              figsize=(max(8, n_joints * 1.2),
                                       n_gaits + 2.0))
    for ax, rmse_mat, title in zip(
            axes,
            [rmse_pure, rmse_trans],
            ["RMSE — Pure windows (°)", "RMSE — Transition windows (°)"]):
        im = ax.imshow(rmse_mat, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="RMSE (°)")
        ax.set_xticks(range(n_joints))
        ax.set_xticklabels([f"J{j+1}" for j in range(n_joints)], fontsize=9)
        ax.set_yticks(range(n_gaits))
        ax.set_yticklabels(gait_names, fontsize=9)
        ax.set_title(title, fontsize=10)
        for g in range(n_gaits):
            for j in range(n_joints):
                v = rmse_mat[g, j]
                if not np.isnan(v):
                    ax.text(j, g, f"{v:.1f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if v > vmax * 0.6 else "black")
    plt.suptitle("Per-Joint RMSE: Pure vs Transition Windows",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = out_dir / "rmse_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {p}")




# ═══════════════════════════════════════════════════════════════════
# 15.  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CPG-SNN robust multi-gait controller — chunk-based CPG")

    # ── CPG ─────────────────────────────────────────────────────
    parser.add_argument("--tmax",            type=int,   default=10000)
    parser.add_argument("--cpg_start_time",  type=int,   default=90)
    parser.add_argument("--chunk_size",      type=int,   default=1,
                        help="Steps per solve_ivp call in CPGChunkStepper. "
                             "Must match the value used at deployment on RPi.")
    parser.add_argument("--spike_thresh",    type=float, default=-2.0,
                        help="Upward vm crossing threshold for spike detection")

    # ── Network ──────────────────────────────────────────────────
    parser.add_argument("--seq_len",         type=int,   default=3,
                        help="Spike events per input window. "
                             "Needs to span ~1 full gait cycle (~32 spikes) "
                             "for reliable phase tracking.")
    parser.add_argument("--hidden",          type=int,   default=128)
    parser.add_argument("--beta",            type=float, default=0.9)

    # ── Training ─────────────────────────────────────────────────
    parser.add_argument("--epochs",          type=int,   default=400)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch",           type=int,   default=256)
    parser.add_argument("--val",             type=float, default=0.15)
    parser.add_argument("--test",            type=float, default=0.10)
    parser.add_argument("--transition_frac", type=float, default=0.30)

    # ── Period estimation ────────────────────────────────────────
    parser.add_argument("--burnin_bursts",   type=int,   default=5)
    parser.add_argument("--kde_bw",          type=float, default=0.3)

    # ── Misc ─────────────────────────────────────────────────────
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--out_dir",         type=str,   default="outputs")

    N = 6

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = Path(this_file_dir + "/" + args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device : {device}\n")

    # ── 1. CPG — chunk-based (same integrator as deployment) ────
    print("[1/6] Running CPG via CPGChunkStepper ...")
    print(f"      chunk_size={args.chunk_size}  spike_thresh={args.spike_thresh}")
    
    # spike_times, spike_neurons, vm_record = run_cpg_chunked(
    #     N=N,
    #     tmax=args.tmax,
    #     cpg_start_time=args.cpg_start_time,
    #     chunk_size=args.chunk_size,
    #     spike_thresh=args.spike_thresh,
    # )
    print(os.getcwd())
    spike_times, spike_neurons = run_blif_cpg(N=N, t_max = args.tmax, cpg_start_time=args.cpg_start_time)
    
    print(f"      Collected {len(spike_times)} spike events "
          f"over t=[{spike_times[0]:.0f}, {spike_times[-1]:.0f}]")
    # plot_cpg_vm(vm_record, out_dir)

    # ── 2. Burst-based gait period ───────────────────────────────
    print("\n[2/6] Estimating gait period from burst structure ...")
    gait_period, burst_thresh = estimate_gait_period(
        spike_times, spike_neurons, out_dir,
        N=N, burnin_bursts=args.burnin_bursts, kde_bw=args.kde_bw)

    base_feats, event_phases = encode_spike_events(
        spike_times, spike_neurons, gait_period, N=N)
    plot_spike_events(spike_times, spike_neurons, gait_period, out_dir, N=N)

    # ── 3. Gait tables ──────────────────────────────────────────
    print("\n[3/6] Loading and upsampling gait tables ...")

    base_gait_names = [
        "tripod", "tripod_huge", "tripod_right", "tripod_huge_right",
        "ripple", "ripple_tiny", "ripple_right", "ripple_tiny_right",
    ]
    mirrored_gait_names = [
        "tripod_backwards", "tripod_huge_backwards", "tripod_left", "tripod_huge_left",
        "ripple_backwards", "ripple_tiny_backwards", "ripple_left", "ripple_tiny_left",
    ]
    gait_names = base_gait_names + mirrored_gait_names
    gait_names = gait_names[0:4]

    gait_tables_orig = []
    for name in base_gait_names:
        gait_table = np.loadtxt(f"{this_file_dir}/gaits/{name}.csv",
                                delimiter=",", dtype=np.float32)
        gait_tables_orig.append(gait_table)

    for name in mirrored_gait_names:
        base_name = name.replace("_backwards", "").replace("_left", "_right")
        gait_table = np.loadtxt(f"{this_file_dir}/gaits/{base_name}.csv",
                                delimiter=",", dtype=np.float32)
        gait_tables_orig.append(np.flip(gait_table, axis=0).copy())

    # gait_tables_orig = [wkF, bk, wkL, wkR]
    # gait_names       = ["wkF", "bk", "wkL", "wkR"]

    for name, g in zip(gait_names, gait_tables_orig):
        print(f"      {name:>4s} : {g.shape[0]} rows × {g.shape[1]} joints (original)")

    # # Upsample to equal row count — equalises phase target resolution
    gait_tables, target_rows = upsample_gait_tables(
        gait_tables_orig, gait_names)
    n_joints = gait_tables[0].shape[1]

    print("\n      Generating burst/gait overlay diagnostic ...")
    plot_burst_gait_overlay(
        spike_times, spike_neurons, gait_period, burst_thresh,
        gait_tables_orig, gait_names, out_dir, n_cycles=6, N=N)
    
    # ── 4. Dataset ──────────────────────────────────────────────
    print("\n[4/6] Building dataset ...")
    print(f"      seq_len={args.seq_len}  "
          f"transition_frac={args.transition_frac:.2f}  "
          f"hidden={args.hidden}")
    rng = np.random.default_rng(args.seed)
    X, y, tgt_range, pure_mask, labels = build_dataset(
        base_feats, event_phases, gait_tables,
        seq_len=args.seq_len,
        transition_frac=args.transition_frac,
        rng=rng)
    n_in = X.shape[2]
    print(f"\n      X : {X.shape}   y : {y.shape}")
    print(f"      tgt_range : [{tgt_range[0]:.1f}, {tgt_range[1]:.1f}]")

    (train_ds, val_pure_ds, val_trans_ds,
     test_pure_ds, test_trans_ds) = train_val_test_split(
        X, y, pure_mask, labels,
        val_frac=args.val, test_frac=args.test, seed=args.seed)

    kw           = dict(collate_fn=snn_collate, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, **kw)
    vp_loader    = make_loader(val_pure_ds,   args.batch, False)
    vt_loader    = make_loader(val_trans_ds,  args.batch, False)
    tp_loader    = make_loader(test_pure_ds,  args.batch, False)
    tt_loader    = make_loader(test_trans_ds, args.batch, False)

    # ── 5. Train ────────────────────────────────────────────────
    print("\n[5/6] Training SNN ...")
    model = CPG_SNN(n_in=n_in, hidden=args.hidden,
                    n_out=n_joints, n_gaits=len(gait_tables),
                    beta=args.beta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters : {n_params:,}")
    print(f"      n_in={n_in}  hidden={args.hidden}  "
          f"beta={args.beta:.3f}  lr={args.lr:.2e}")

    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    # Weighted criterion upweights gaits with fewer original rows (bk)
    criterion  = make_gait_weighted_criterion(gait_tables_orig, device)

    best_val, history = run_training(
        model, train_loader, vp_loader, vt_loader,
        optimizer, scheduler, criterion, device,
        epochs=args.epochs, out_dir=out_dir, weighted=True)

    model.load_state_dict(torch.load(out_dir / "best_model.pt",
                                     map_location=device))

    # Use plain MSE for test reporting so numbers are in comparable units
    plain_mse  = nn.MSELoss()
    # Wrap plain MSE to accept 3-tuple interface
    plain_crit = lambda p, t, _: plain_mse(p, t)
    tp_mse = eval_epoch(model, tp_loader, plain_crit, device, weighted=True)
    tt_mse = eval_epoch(model, tt_loader, plain_crit, device, weighted=True)
    print(f"\n  Test MSE (plain)  pure       : {tp_mse:.6f}")
    print(f"  Test MSE (plain)  transition : {tt_mse:.6f}")

    # ── 6. Plots + export ────────────────────────────────────────
    print("\n[6/6] Generating plots and exporting ...")
    plot_training_curves(history, out_dir)
    full_ds = GaitDataset(X, y, labels)
    plot_inference(model, full_ds, device, out_dir,
                   n_joints=n_joints, n_gaits=len(gait_tables))
    plot_gait_reconstruction(
        model, X, y, pure_mask, labels, device, out_dir,
        n_joints=n_joints, tgt_range=tgt_range,
        gait_names=gait_names, n_samples=300)

    inference_config = {
        "gait_period":     gait_period,
        "burst_threshold": burst_thresh,
        "global_min":      tgt_range[0],
        "global_max":      tgt_range[1],
        "seq_len":         args.seq_len,
        "n_gaits":         len(gait_tables),
        "n_joints":        n_joints,
        "gait_names":      gait_names,
        # Stepper parameters — deployment must use these exact values
        "chunk_size":      args.chunk_size,
        "spike_thresh":    args.spike_thresh,
        "cpg_start_time":  args.cpg_start_time,
        # Upsampling — inference must upsample gait tables identically
        "target_rows":     target_rows,
        # Feature dim — used to verify ONNX input shape
        "n_in":            n_in,
    }
    export_to_onnx(model, seq_len=args.seq_len, n_in=n_in,
                   out_dir=out_dir, device=device,
                   inference_config=inference_config)

    print(f"\nDone — outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()