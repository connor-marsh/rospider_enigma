"""
The bursting-LIF CPG: tunable parameters and coupling matrices, the real and fake
oscillators, running and replaying them, and burst/phase analysis (analyse_cpg also draws
the burst-structure figure).
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import cfg_get, out_path


# ═══════════════════════════════════════════════════════════════════
# Bursting-LIF CPG
# ═══════════════════════════════════════════════════════════════════

# All-to-all inhibitory coupling, one matrix per CPG size (add a row and
# --n_cpg_neurons accepts it); N=3/6 were ported from cpg_utils.py::BLIF_CPG.
# from_fb_weight (the burst-terminating kick) is fixed at CPG_FROM_FB_WEIGHT for every
# N, so there is no per-N regime to configure. analyse_cpg warns if burst phase
# offsets are far from evenly spaced.
CPG_FROM_FB_WEIGHT = -1_000_000.0

# ── tunable CPG parameters: name -> (default, help) ─────────
# One source of truth for argparse (--cpg_<name>), LIFCPGStepper's construction and
# the config dump (replay_cpg rebuilds the oscillator from it).
# from_fb_weight is deliberately absent (fixed, see CPG_FROM_FB_WEIGHT). The two that
# shape the burst:
#   vth_fb       spikes per burst: the feedback neuron integrates to_fb_weight per
#                main spike with no leak, ending the burst after vth_fb/to_fb_weight.
#   refrac_main  steps between spikes within a burst (ISI = refrac_main + 1).
# Both change the fake-CPG period, (vth_fb/to_fb_weight) * (refrac_main+1) * N, and
# every period-derived default with it; run_cpg prints the resulting structure.
CPG_PARAMS = {
    "vth_main":     (100.0, "Main-neuron spike threshold."),
    "du_main":      (0.1,   "Main-neuron current decay."),
    "dv_main":      (0.3,   "Main-neuron voltage decay."),
    "refrac_main":  (1,     "Main-neuron refractory period, in steps. Sets "
                            "the inter-spike interval WITHIN a burst: "
                            "ISI = refrac_main + 1. Raise to spread bursts "
                            "out in time."),
    "vth_fb":       (100.0, "Feedback-neuron threshold. Sets SPIKES PER "
                            "BURST: the feedback neuron integrates "
                            "to_fb_weight per main spike with no leak, so it "
                            "terminates the burst after vth_fb/to_fb_weight "
                            "spikes. Lower for shorter bursts."),
    "du_fb":        (1.0,   "Feedback-neuron current decay."),
    "dv_fb":        (0.0,   "Feedback-neuron voltage decay. 0 = pure "
                            "integrator, which is what makes the burst "
                            "length exactly vth_fb/to_fb_weight."),
    "refrac_fb":    (1,     "Feedback-neuron refractory period, in steps."),
    "to_fb_weight": (10.0,  "Main->feedback weight. Together with vth_fb "
                            "this sets spikes per burst; prefer changing "
                            "vth_fb and leaving this alone, so the ratio "
                            "stays easy to read."),
}

CPG_W_BY_N = {
    1: np.asarray([[0]]),

    3: np.asarray([
        [    0.0      , -523.65135942, -593.28982051],
        [-696.81822016,     0.0      , -632.34680962],
        [-687.56816569, -577.5693762 ,     0.0      ],
    ], dtype=np.float64),

    4: np.asarray([
        [    0.0      , -648.52905924, -449.60304695, -413.48426163],
        [-369.91504928,     0.0      , -592.29635234, -568.0712858 ],
        [-412.08729881, -391.54918498,     0.0      , -618.03381552],
        [-498.16458351, -655.01105883, -345.38277449,     0.0      ],
    ], dtype=np.float64),

    6: np.asarray([
        [    0.0      , -375.86210512, -518.18703523, -371.82375498, -399.74231244, -487.45119873],
        [-531.99480471,     0.0      , -489.1139223 , -128.33470562, -404.33117771, -628.03347932],
        [-529.89653583, -418.34662835,     0.0      , -543.37143674, -336.83773596, -679.12224243],
        [-674.09562904, -130.56007131, -297.35360394,     0.0      , -363.1208234 , -425.10847629],
        [-486.03391005, -386.7920052 , -412.91478912, -437.7646991 ,     0.0      , -288.47748806],
        [-112.97808475, -510.59115452, -367.63412082, -374.83106147, -393.86103887,     0.0      ],
    ], dtype=np.float64),

    # PLACEHOLDER, --fake_cpg ONLY. N=18 gives one CPG neuron per joint, shrinking the window
    # a timing unit is confined to from 1/6 to 1/18 of a cycle. Zeros is NOT a working
    # oscillator (no inhibition, so all 18 would fire in phase); it is safe only because
    # fake_step_chunk synthesiizes its spike train arithmetically and never reads W.
    # Guarded in run_cpg. Use a tuned matrix before running N=18 on the real oscillator.
    18: np.zeros((18, 18), dtype=np.float64),
}

# Sizes whose CPG_W_BY_N entry is a placeholder and so require --fake_cpg.
CPG_W_FAKE_ONLY = {18}


def cpg_weight_matrix(N):
    """Coupling matrix for an N-neuron CPG.  Raises rather than falling back,
    so a typo'd --n_cpg_neurons fails at startup instead of silently
    training against the wrong oscillator."""
    if N not in CPG_W_BY_N:
        raise ValueError(
            f"No CPG weight matrix for N={N}; available: "
            f"{sorted(CPG_W_BY_N)}.  Add one to CPG_W_BY_N.")
    return CPG_W_BY_N[N].copy()


class LIFGeneralArray:
    """Vectorized current-based LIF with 2-stage filtering and refractoriness."""

    def __init__(self, num_neurons, vth, du, dv, bias=0.0, u=0.0, v=0.0,
                 refractory_period=0):
        self.vth  = vth
        self.du   = du
        self.dv   = dv
        self.bias = bias
        self.u    = np.full(num_neurons, float(u))
        self.v    = np.full(num_neurons, float(v))
        self.refractory_period    = refractory_period
        self.time_since_last_spike = np.zeros(num_neurons)
        self.num_neurons = num_neurons

    def next_step(self, current):
        self.u = self.u * (1 - self.du) + current
        self.v = self.v * (1 - self.dv) + self.u + self.bias

        refractory_mask = self.time_since_last_spike > 0
        self.v[refractory_mask] = 0
        self.time_since_last_spike = np.clip(
            self.time_since_last_spike - 1, 0, None)

        spike = self.v >= self.vth
        self.v[spike] = 0
        self.time_since_last_spike[spike] = self.refractory_period
        return spike.astype(np.float32)

    def reset(self, u=0.0, v=0.0):
        self.u.fill(u)
        self.v.fill(v)
        self.time_since_last_spike.fill(0)


class BurstingLIF:
    """Main neuron + fast feedback neuron. The feedback neuron integrates `to_fb_weight`
    per main spike with no leak, so after ~vth_fb/to_fb_weight spikes it dumps
    `from_fb_weight` (large negative) into the main neuron, ending the burst (100/10
    = 10 spikes by default).
    """

    def __init__(self, num_neurons, vth_main, du_main, dv_main, refrac_main,
                 vth_fb, du_fb, dv_fb, refrac_fb, from_fb_weight, to_fb_weight):
        self.n_main = LIFGeneralArray(num_neurons, vth_main, du_main, dv_main,
                                      refractory_period=refrac_main)
        self.n_fb   = LIFGeneralArray(num_neurons, vth_fb, du_fb, dv_fb,
                                      refractory_period=refrac_fb)
        self.input_2_feedback_neuron_weight = to_fb_weight
        self.feedback_2_input_neuron_weight = from_fb_weight
        self.fb_current = np.zeros(num_neurons)

    def forward(self, current):
        main_spike = self.n_main.next_step(current + self.fb_current)
        fb_spike   = self.n_fb.next_step(
            main_spike * self.input_2_feedback_neuron_weight)
        self.fb_current = fb_spike * self.feedback_2_input_neuron_weight
        return main_spike, fb_spike

    def reset(self):
        self.n_main.reset()
        self.n_fb.reset()
        self.fb_current.fill(0.0)


class LIFCPGStepper:
    """Canonical spike generator, used for both training data and deployment. `step()`
    advances one timestep and returns the (N,) binary spike vector; no integrator
    state differs between batch and streaming use.
    """

    def __init__(self, N, W=None, i_app=8.0,
                 vth_main=100.0, du_main=0.1, dv_main=0.3, refrac_main=1,
                 vth_fb=100.0, du_fb=1.0, dv_fb=0.0, refrac_fb=1,
                 from_fb_weight=CPG_FROM_FB_WEIGHT, to_fb_weight=10.0):
        self.N     = int(N)
        self.W     = (cpg_weight_matrix(self.N) if W is None
                      else np.asarray(W, dtype=np.float64))
        if self.W.shape != (self.N, self.N):
            raise ValueError(
                f"CPG weight matrix is {self.W.shape}, expected "
                f"({self.N}, {self.N}).")
        self.i_app = float(i_app)
        # Kept so fake_step_chunk reuses the same constants as self.core.
        self.refrac_main  = int(refrac_main)
        self.vth_fb       = float(vth_fb)
        self.to_fb_weight = float(to_fb_weight)
        self.core  = BurstingLIF(N, vth_main, du_main, dv_main, refrac_main,
                                 vth_fb, du_fb, dv_fb, refrac_fb,
                                 from_fb_weight, to_fb_weight)
        self.inter_neuron_current = np.zeros(N)
        self.t = 0

    def step(self):
        spk = self.core.forward(self.inter_neuron_current + self.i_app)[0]
        self.inter_neuron_current = self.W @ spk
        self.t += 1
        return spk

    def step_chunk(self, n_steps):
        out = np.zeros((n_steps, self.N), dtype=np.float32)
        for k in range(n_steps):
            out[k] = self.step()
        return out

    def fake_step_chunk(self, n_steps):
        """Synthetic stand-in for step_chunk: N evenly-spaced, BACK-TO-BACK bursts per cycle
        (no inter-burst gap). Temporary, for testing whether a continuous-activity CPG
        helps before the real oscillator is retuned (see architecture_change_todo.md).
        Uses this instance's refrac_main / vth_fb / to_fb_weight, so the burst width
        matches step_chunk's.
        """
        n_spikes = int((self.vth_fb // self.to_fb_weight)
                       * (self.refrac_main + 1))
        period = n_spikes * self.N
        pattern = np.zeros((self.N, n_steps), dtype=np.float32)
        for i in range(self.N):
            start = i * n_spikes
            for t in range(start, n_steps, period):
                pattern[i, t:t + n_spikes:self.refrac_main + 1] = 1.0
        if self.N == 1:
            pattern = np.zeros((self.N, n_steps), dtype=np.float32)
            for t in range(0, n_steps, period):
                pattern[0, t:t + n_spikes:self.refrac_main + 1] = 1.0
        return pattern.T

    def reset(self):
        self.core.reset()
        self.inter_neuron_current.fill(0.0)
        self.t = 0


def run_cpg(N, tmax=120_000, warmup=2_000, i_app=8.0, fake_cpg=False,
            cpg_params=None):
    """Warm up, then collect the spike train used for training.

    `N` has no default on purpose: it selects the coupling matrix, and a wrong value
    changes the oscillator silently. `cpg_params` is a dict over CPG_PARAMS' keys
    (None = defaults); callers pass the same dict that goes into the config.
    fake_cpg=True substitutes fake_step_chunk's back-to-back bursts for the real
    oscillator.
    """
    cpg_params = ({k: v[0] for k, v in CPG_PARAMS.items()}
                  if cpg_params is None else dict(cpg_params))
    unknown = set(cpg_params) - set(CPG_PARAMS)
    if unknown:
        raise ValueError(f"unknown cpg_params key(s) {sorted(unknown)}; "
                         f"expected a subset of {sorted(CPG_PARAMS)}")
    if N in CPG_W_FAKE_ONLY and not fake_cpg:
        raise ValueError(
            f"--n_cpg_neurons {N} has only a PLACEHOLDER (zeros) coupling matrix, so it "
            f"requires --fake_cpg 1 (the real oscillator would fire in phase). "
            f"Add a tuned matrix for N={N} first.")
    cpg = LIFCPGStepper(N=N, i_app=i_app, **cpg_params)
    print(f"  N={N}  i_app={i_app}  from_fb_weight={CPG_FROM_FB_WEIGHT:g}")
    if N in CPG_W_FAKE_ONLY:
        print(f"  (N={N} coupling matrix is a --fake_cpg-only placeholder; "
              f"W is unused)")

    # Burst structure implied by the parameters. Exact under --fake_cpg; under the real
    # oscillator it is the design point (coupling and i_app also move where bursts land;
    # analyse_cpg measures that).
    spb     = cpg_params["vth_fb"] / cpg_params["to_fb_weight"]
    isi     = int(cpg_params["refrac_main"]) + 1
    n_slots = int(spb) * isi
    print(f"  burst: {int(spb)} spk/burst (vth_fb/to_fb_weight), "
          f"ISI {isi} step(s) (refrac_main+1), width {n_slots} steps "
          f"-> {'exact' if fake_cpg else 'nominal'} period {n_slots * N} "
          f"steps, cpg_rate {int(spb)} spk/cyc/neuron, "
          f"ceiling {int(spb) * N} spk/cyc")
    if abs(spb - round(spb)) > 1e-9:
        print(f"  WARNING vth_fb/to_fb_weight = {spb:.4f} is not an integer: bursts floor "
        f"to {int(spb)} spikes and carry the remainder forward. Use a vth_fb that is "
        f"a multiple of to_fb_weight ({cpg_params['to_fb_weight']:g}).")
    if int(spb) < 1:
        raise ValueError(
            f"vth_fb ({cpg_params['vth_fb']:g}) is below to_fb_weight "
            f"({cpg_params['to_fb_weight']:g}), so the burst length is zero. Raise "
            f"vth_fb or lower to_fb_weight.")
    if fake_cpg:
        print(f"  FAKE CPG: back-to-back bursts, no inter-burst gap "
              f"(see fake_step_chunk)")
    print(f"  Warming up CPG ({warmup} steps) ...")
    cpg.step_chunk(warmup)
    print(f"  Collecting {tmax} steps ...")
    spikes = cpg.fake_step_chunk(tmax) if fake_cpg else cpg.step_chunk(tmax)

    counts = spikes.sum(0).astype(int)
    print(f"  Spikes per neuron : {counts.tolist()}")
    if counts.min() == 0:
        raise RuntimeError("A CPG neuron never fired — check W / i_app.")
    return spikes


# ═══════════════════════════════════════════════════════════════════
# Burst detection  →  phase
# ═══════════════════════════════════════════════════════════════════

def detect_burst_threshold(spike_steps, bw_method=0.3):
    """ISI threshold separating within-burst from between-burst gaps: the antimode of
    the log-ISI KDE.
    """
    isi = np.diff(spike_steps).astype(np.float64)
    if len(isi) < 4 or isi.max() - isi.min() < 1e-9:
        return float(np.median(isi)) if len(isi) else 1.0

    log_isi = np.log(isi + 1e-6)
    kde     = gaussian_kde(log_isi, bw_method=bw_method)
    x_eval  = np.linspace(log_isi.min(), log_isi.max(), 2000)
    density = kde(x_eval)
    minima  = argrelmin(density, order=20)[0]

    if len(minima) == 0:
        return float(np.median(isi))
    mid  = 0.5 * (log_isi.min() + log_isi.max())
    best = minima[np.argmin(np.abs(x_eval[minima] - mid))]
    return float(np.exp(x_eval[best]))


def burst_onsets(spike_steps, threshold):
    """First spike of every burst."""
    keep = np.concatenate([[True], np.diff(spike_steps) > threshold])
    return spike_steps[keep]


def analyse_cpg(spikes, out_dir):
    """
    Returns
    -------
    onsets  : list of (n_bursts_i,) int arrays — burst onsets per neuron
    period  : float — median inter-burst interval of neuron 0
    offsets : (N,) float — per-neuron burst phase offset vs neuron 0, in [0,1)
    """
    N = spikes.shape[1]
    onsets, thresholds = [], []
    for i in range(N):
        ts  = np.where(spikes[:, i] > 0)[0]
        thr = detect_burst_threshold(ts)
        on  = burst_onsets(ts, thr)
        onsets.append(on)
        thresholds.append(thr)
        print(f"    N{i}: ISI thr={thr:6.2f}  bursts={len(on):4d}  "
              f"spk/burst={len(ts)/max(1,len(on)):5.2f}  "
              f"period={np.median(np.diff(on)):7.1f}")

    period = float(np.median(np.diff(onsets[0])))
    ref    = onsets[0][len(onsets[0]) // 2]
    offsets = np.array([
        float((onsets[i][np.searchsorted(onsets[i], ref)] - ref) % period) / period
        for i in range(N)], dtype=np.float64)
    print(f"    period = {period:.1f} steps   "
          f"neuron phase offsets = {np.round(offsets, 3).tolist()}")

    # A healthy N-neuron ring puts the bursts at ~i/N; gaps far from 1/N mean the coupling matrix
    # and i_app disagree (the run still "works", but not on the oscillator it was tuned for).
    gaps = np.diff(np.sort(np.concatenate([offsets % 1.0, [1.0]])))
    if gaps.size and (gaps.min() < 0.5 / N or gaps.max() > 2.0 / N):
        print(f"    WARNING: burst phase offsets are far from evenly spaced "
              f"(sorted gaps {np.round(gaps, 3).tolist()}, ideal {1.0/N:.3f}).")
        print(f"             The N={N} coupling matrix may not agree with "
              f"the current i_app — inspect cpg_raster.png before trusting "
              f"this run.")

    # diagnostic: log-ISI split for neuron 0
    ts  = np.where(spikes[:, 0] > 0)[0]
    isi = np.diff(ts)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(np.log(isi + 1e-6), bins=80, density=True,
               color="#457b9d", alpha=0.65)
    ax[0].axvline(np.log(thresholds[0]), color="#f4a261", lw=2, ls="--",
                  label=f"threshold = {thresholds[0]:.1f}")
    ax[0].set_xlabel("log(ISI)"); ax[0].set_ylabel("density")
    ax[0].set_title("Neuron 0 — log-ISI split"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].hist(isi, bins=100, color="#2a9d8f", alpha=0.65)
    ax[1].axvline(thresholds[0], color="#f4a261", lw=2, ls="--")
    ax[1].set_yscale("log"); ax[1].set_xlabel("ISI (steps)")
    ax[1].set_title("Neuron 0 — raw ISI"); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    p = out_path(out_dir, "burst_threshold.png")
    plt.savefig(p, dpi=140); plt.close()
    print(f"    [saved] {p}")

    return onsets, period, offsets, thresholds


def cycle_phase(T, onsets):
    """
    Phase in [0,1) that ramps linearly from one burst onset to the next.
    Steps outside [first_onset, last_onset) are NaN and get trimmed away.
    """
    phase = np.full(T, np.nan, dtype=np.float32)
    for a, b in zip(onsets[:-1], onsets[1:]):
        phase[a:b] = np.arange(b - a, dtype=np.float32) / float(b - a)
    return phase


def cpg_spike_stats(spikes, phase, period):
    """The CPG's own firing statistics, per neuron then averaged: spikes per cycle and
    circular concentration R of the spike phases. Measured, not configured ("fire like
    the CPG does" needs no hyperparameter; R=1.0 is unsatisfiable). The spike count
    also sets the calibration ceiling (n_cpg * cpg_rate); the band itself is a
    fraction of the period.
    """
    ok = ~np.isnan(phase)
    n_cycles = max(ok.sum() / float(period), 1e-9)
    rates, Rs = [], []
    for i in range(spikes.shape[1]):
        m = (spikes[:, i] > 0) & ok
        rates.append(m.sum() / n_cycles)
        if m.sum() == 0:
            Rs.append(0.0)
            continue
        z = np.mean(np.exp(1j * 2.0 * np.pi * phase[m]))
        Rs.append(abs(z))
    return float(np.mean(rates)), float(np.mean(Rs))


# ═══════════════════════════════════════════════════════════════════
# Replay the CPG from a run's config
# ═══════════════════════════════════════════════════════════════════

def replay_cpg(cfg, n_steps):
    """Re-run the exact CPG the model was trained against, from cfg["cpg"] (including the
    weight matrix, so it is reproducible even if train.py's defaults change). The warm-up
    is replayed too, since burst phase depends on it. cfg["fake_cpg"] selects
    fake_step_chunk's back-to-back bursts (default False for older configs).
    """
    c = dict(cfg.get("cpg", {}))
    N = int(c.get("N") or cfg_get(cfg, "n_cpg_neurons", 4))
    W = np.asarray(c["W"], dtype=np.float64) if "W" in c else cpg_weight_matrix(N)

    cpg = LIFCPGStepper(
        N=N, W=W,
        i_app          = float(c.get("i_app", 8.0)),
        vth_main       = float(c.get("vth_main", 100.0)),
        du_main        = float(c.get("du_main", 0.1)),
        dv_main        = float(c.get("dv_main", 0.3)),
        refrac_main    = int(c.get("refrac_main", 1)),
        vth_fb         = float(c.get("vth_fb", 100.0)),
        du_fb          = float(c.get("du_fb", 1.0)),
        dv_fb          = float(c.get("dv_fb", 0.0)),
        refrac_fb      = int(c.get("refrac_fb", 1)),
        from_fb_weight = float(c.get("from_fb_weight", CPG_FROM_FB_WEIGHT)),
        to_fb_weight   = float(c.get("to_fb_weight", 10.0)))

    fake = bool(cfg_get(cfg, "fake_cpg", False))
    warmup = int(c.get("warmup", 2000))
    cpg.step_chunk(warmup)
    spikes = cpg.fake_step_chunk(n_steps) if fake else cpg.step_chunk(n_steps)

    counts = spikes.sum(0).astype(int)
    print(f"  CPG replayed: N={N}  warmup={warmup}  steps={n_steps}"
          f"{'  FAKE (back-to-back bursts)' if fake else ''}  "
          f"spikes/neuron={counts.tolist()}")
    if counts.min() == 0:
        print("  WARNING: a CPG neuron never fired during the replay window.")
    return spikes


def cpg_phase(spikes):
    """Burst onsets of neuron 0, the median period, and a per-step phase ramp
    in [0,1) between consecutive onsets (NaN outside)."""
    ts  = np.where(spikes[:, 0] > 0)[0]
    thr = detect_burst_threshold(ts)
    on  = burst_onsets(ts, thr)
    if len(on) < 3:
        raise RuntimeError(
            f"Only {len(on)} neuron-0 bursts in the replay window; "
            f"raise --n_cycles.")
    period = float(np.median(np.diff(on)))

    phase = np.full(len(spikes), np.nan, dtype=np.float64)
    for a, b in zip(on[:-1], on[1:]):
        phase[a:b] = np.arange(b - a, dtype=np.float64) / float(b - a)
    return on, period, phase, thr


def spike_bursts(spike_steps, thr):
    """Split a spike-time array into bursts using an ISI threshold; returns a
    list of arrays.  Used for the timing layer, where 'when did this unit
    start firing' is more legible than every individual spike."""
    if len(spike_steps) == 0:
        return []
    cuts = np.where(np.diff(spike_steps) > thr)[0] + 1
    return np.split(spike_steps, cuts)
