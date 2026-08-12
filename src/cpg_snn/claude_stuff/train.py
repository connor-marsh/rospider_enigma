"""
Bursting-LIF CPG  →  Leg-Grouped Stateful SNN  →  Multi-Gait Joint Angles
=========================================================================

What changed vs. the previous (conductance-based CPG + event-window) version
---------------------------------------------------------------------------
1.  CPG replaced.
    The 4-neuron BDF/solve_ivp conductance CPG is gone.  It is replaced by
    the discrete bursting-LIF CPG (LIFGeneralArray + BurstingLIF + an
    all-to-all inhibitory weight matrix).  This is a pure integer-step
    integrator: no ODE solver, no chunk/batch mismatch, no ARM/x86 FPU
    divergence.  One `step()` == one timestep.

    Measured behaviour of the supplied 4-neuron weight matrix @ i_app=8:
        period            ~254 steps
        spikes per burst  10
        burst duration    ~29 steps  (duty ~11%)
        burst order       N0 -> N1 -> N2 -> N3
        phase offsets     0.000 / 0.248 / 0.500 / 0.748   (clean quarters)

2.  Sin/cos phase encoding removed.
    The SNN input is now ONLY the 4 raw CPG spikes at each timestep.
    There is no absolute-phase channel, no ISI channel, no hand-computed
    gait_period fed to the network.  Phase is something the network has
    to *hold in its state*.

3.  Stateful SNN.
    Because the input carries no phase, the network needs memory that
    spans a full gait cycle (~254 steps).  This comes entirely from
    heterogeneous, learnable membrane time constants: each hidden unit's
    tau is initialised log-uniformly over [tau_min, tau_max] (default
    2..256 steps) so the population tiles timescales from within-burst to
    full-cycle, forming a temporal basis over "time since last burst"
    that the readout reads off linearly.
    Training is truncated BPTT over contiguous chunks with state carried
    (and detached) across chunk boundaries -- B parallel streams walking
    the spike train, exactly like deployment.

    Within-group recurrent connections on the spike outputs were also
    tried (rec1/rec2, 32,768 params = 45% of the model).  Ablating them
    left free-run reconstruction RMSE and Val(post-switch) unchanged
    while slightly improving step time and memory, so they were removed.
    See git history before the removal commit to reproduce that test.

4.  `--arch dense` (StatefulSNN): fully connected.
    Every CPG spike reaches every hidden unit, both hidden layers are
    dense, and the readout maps the whole hidden state to all 8 joints.

    The design BEFORE that split the hidden layer into 4 leg groups, drove
    group l with a single CPG neuron chosen by a per-gait permutation, and
    read leg l's two joints out of group l only.  Both parts were removed:

      - The grouping made layers 2+ block diagonal (4 x Hg x Hg), i.e.
        four independent sub-networks, with `cross_gain * w_cross` in
        layer 1 as the only path between them.
      - The routing permutation matched each leg's swing onset to the CPG
        neuron whose burst was closest in phase, but the residuals were
        0.116-0.133 cycle for 3 of the 4 gaits -- about 34 steps at period
        254, longer than a burst (~29 steps) -- so the alignment only
        really held for wkF.  And since `w_cross` already spanned all four
        neurons, routing was an initialisation prior rather than a
        capability; the network had to learn per-gait timing corrections
        regardless, which is what FiLM is for.

    Servo routing itself is out of scope for this file -- see
    inference.py -- but the gait-table COLUMN ordering it depends on is
    fixed: column j always means the same joint, so a deployment script
    can map columns to servo channels however its harness needs.

    >>> CHECK THIS <<<  see LEG_COLS below.

5.  `--arch timing_grouped` (TimingGroupedSNN): grouping, reintroduced.
    CPG spikes -> a small TIMING layer of n_timing LIF neurons (densely
    driven, so n_timing need NOT equal n_cpg_neurons) -> n_timing fully
    disconnected sub-networks, one per timing neuron, each two spiking
    layers of `--hidden` units plus a block-diagonal analog readout.  Group
    g writes only its own gait-table columns (see build_group_cols).

    The split of labour: the timing layer learns the RHYTHM in a few
    hundred parameters; the sub-networks learn ANGLES given a clean phase
    reference, instead of four copies of the network each re-deriving phase
    from raw CPG spikes.

    What makes the grouping viable this time is a free per-gait CPG->timing
    weight matrix.  The old routing was a fixed permutation solved offline,
    and its phase alignment held for only 1 of 4 gaits; here each gait learns
    its own routing.  A shared-weights alternative (one weight pair plus a
    per-gait FiLM gate on a small LIF hidden layer) was built and reverted --
    it produced near-identical timing phases across gaits.  See the
    TimingGroupedSNN docstring for the diagnosis and todo item 11.

    This is todo 3a (no cross talk).  3b -- every sub-network sees all
    n_timing timing spikes -- is a one-line change to sub-net layer 1 and
    is deliberately not wired up yet.

6.  CPG size is an argument.
    `--n_cpg_neurons {3,4,6}` selects a coupling matrix from CPG_W_BY_N and
    sizes the SNN input; nothing downstream assumes 4.  from_fb_weight is
    fixed at CPG_FROM_FB_WEIGHT for every N (confirmed to work for both
    N=4 and the ported N=3/N=6), so there is no regime to configure here.

Leg / joint layout  (LEG_COLS)
------------------------------
LEG_COLS is presentation only -- it groups the 8 output columns per leg for
the diagnostic plots.  It no longer constrains the network architecture,
and this file has no notion of servo channels at all: that mapping is
inference.py's problem, not training's.

The 8 gait-table columns are two joints x four legs, laid out as
    columns 0..3 = joint A of legs 0..3
    columns 4..7 = joint B of legs 0..3
so leg l == columns (l, l+4).  This was verified numerically: for every
gait, col j and col j+4 share the same circular phase offset (5/54 cycle
in wkF, 5/39 in wkL/wkR, 15..17/22 in bk) while cols 0..3 are the same
waveform shifted, and cols 4..7 are a different waveform shifted by the
same per-leg amounts.  Change LEG_COLS if your gait tables disagree.

LEG_COLS / N_LEGS / N_JOINTS above are the QUADRUPED values (n_cpg_neurons=4,
n_joints=8).  HEXAPOD_LEG_COLS / HEXAPOD_N_LEGS / HEXAPOD_N_JOINTS
(n_cpg_neurons=6, n_joints=18; 3 servos/leg, legs in LF/LM/LR/RF/RM/RR order)
are the hexapod equivalent.  A run resolves n_legs/leg_cols/n_joints from
whichever matches --n_cpg_neurons via `default_leg_layout()`, or from
--leg_cols if that's given explicitly.

Usage
-----
    python train.py --epochs 300 --hidden 256
    python train.py --dry_run                       # data + plots, no training

    # timing layer + per-leg sub-networks (todo 3a)
    python train.py --arch timing_grouped --hidden 256

    # matched-parameter comparison against dense --hidden 256
    python train.py --arch timing_grouped --hidden 128

    # sanity-check the timing layer's firing regime before committing
    python train.py --arch timing_grouped --dry_run

    # 6-neuron CPG (see CPG_W_BY_N)
    python train.py --n_cpg_neurons 6

    # hexapod: n_cpg_neurons=6 auto-selects the 16 tripod/ripple gait CSVs
    # from --gaits_dir (default ../gaits) and HEXAPOD_LEG_COLS for grouping
    python train.py --arch timing_grouped --n_cpg_neurons 6 --dry_run
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
torch.set_float32_matmul_precision('high')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
# 0.  Gait-table column layout  —  EDIT HERE IF YOUR GAIT TABLES DIFFER
# ═══════════════════════════════════════════════════════════════════
# Servo channel mapping is deliberately NOT here: that is inference.py's
# concern, not training's.  This file only groups output columns by leg.

# leg l -> (gait-table column for joint A, column for joint B)
LEG_COLS   = [(0, 4), (1, 5), (2, 6), (3, 7)]
N_LEGS     = 4
N_JOINTS   = 8

# Hexapod: 6 legs x 3 servos (coxa, femur, tibia), columns grouped
# consecutively per leg in that order — leg0 = cols [0,1,2], leg1 = [3,4,5],
# etc.  Leg order in the CSV is LF, LM, LR, RF, RM, RR.
HEXAPOD_LEG_COLS  = [[3 * l, 3 * l + 1, 3 * l + 2] for l in range(6)]
HEXAPOD_LEG_NAMES = ["LF", "LM", "LR", "RF", "RM", "RR"]
HEXAPOD_N_LEGS    = 6
HEXAPOD_N_JOINTS  = 18

# Long enough for a 6-neuron CPG; indexed modulo its own length everywhere.
CPG_PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261",
               "#6a0572", "#8ecae6"]


def build_group_cols(n_timing, leg_cols=LEG_COLS, n_joints=N_JOINTS):
    """
    Map sub-network index -> the gait-table output columns it owns.

    `n_timing` is the number of timing neurons, and there is exactly one
    disconnected sub-network per timing neuron, so this also fixes how the
    8 output columns are partitioned:

        n_timing == n_legs   -> group l owns LEG_COLS[l], e.g. (l, l+4)
        n_timing == n_joints -> group j owns column j alone

    Anything else is rejected: a partition into unequal or overlapping
    groups would make `w_out` ragged and silently mis-route servos, so
    there is no third case to guess at.
    """
    n_legs = len(leg_cols)
    if n_timing == n_legs:
        groups = [list(c) for c in leg_cols]
    elif n_timing == n_joints:
        groups = [[j] for j in range(n_joints)]
    else:
        raise ValueError(
            f"n_timing must be n_legs ({n_legs}) or n_joints ({n_joints}), "
            f"got {n_timing}.")

    flat = [c for grp in groups for c in grp]
    if sorted(flat) != list(range(n_joints)):
        raise ValueError(
            f"group columns {groups} are not a partition of "
            f"0..{n_joints - 1} (flattened: {sorted(flat)}).")
    if len({len(g) for g in groups}) != 1:
        raise ValueError(f"groups must be equal size, got {groups}.")
    return groups


# ═══════════════════════════════════════════════════════════════════
# 0b.  Small utilities
# ═══════════════════════════════════════════════════════════════════

def json_safe(obj):
    """
    Recursively coerce numpy scalars/arrays, Paths and tuples into
    JSON-serialisable types.

    Worth having: the config is written at the very END of a training run,
    so a bare json.dump choking on a np.float32 would throw away the whole
    run's artifacts at the last possible moment.
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None      # NaN/inf are not valid JSON
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def git_info():
    """Short commit hash + dirty flag of the repo this file lives in."""
    here = os.path.dirname(os.path.abspath(__file__))
    def run(cmd):
        return subprocess.check_output(
            cmd, cwd=here, stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
    try:
        commit = run(["git", "rev-parse", "--short", "HEAD"])
        dirty  = bool(run(["git", "status", "--porcelain"]))
        return {"commit": commit, "dirty": dirty}
    except Exception:                                    # noqa: BLE001
        return {"commit": None, "dirty": None}


# ═══════════════════════════════════════════════════════════════════
# 1.  Bursting-LIF CPG
# ═══════════════════════════════════════════════════════════════════

# All-to-all inhibitory coupling, one matrix per CPG size.  Keyed by N so
# nothing downstream has to know which sizes exist -- add a row here and
# `--n_cpg_neurons` accepts it.
#
# PROVENANCE: the N=3 and N=6 matrices were ported from the older
# `cpg_utils.py::BLIF_CPG`.  from_fb_weight (the burst-terminating kick, see
# BurstingLIF below) is fixed at CPG_FROM_FB_WEIGHT = -1e6 for every N --
# confirmed to work for both the original N=4 and the ported N=3/N=6, so
# there is no per-N regime to configure or mismatch.  `analyse_cpg` still
# warns if burst phase offsets come out far from evenly spaced, which would
# now point at the coupling matrix or i_app rather than this weight.
CPG_FROM_FB_WEIGHT = -1_000_000.0

CPG_W_BY_N = {
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
}


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
    """Vectorised current-based LIF with 2-stage filtering and refractoriness."""

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
    """
    Main neuron + fast feedback neuron.

    The feedback neuron integrates `to_fb_weight` per main spike with no
    leak (dv_fb=0), so after ~vth_fb/to_fb_weight main spikes it fires and
    dumps `from_fb_weight` (large negative) into the main neuron, killing
    the burst.  With the supplied params that is 100/10 = 10 spikes/burst.
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
    """
    Canonical spike generator — identical object is used for training data
    generation and for deployment on the Raspberry Pi.

    One `step()` advances exactly one timestep and returns the (N,) binary
    spike vector.  There is no integrator state that differs between batch
    and streaming use, so there is no train/deploy mismatch to reason about.
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

    def reset(self):
        self.core.reset()
        self.inter_neuron_current.fill(0.0)
        self.t = 0


def run_cpg(N, tmax=120_000, warmup=2_000, i_app=8.0):
    """Warm up, then collect the spike train used for training.

    `N` is deliberately positional-with-no-default: it selects the coupling
    matrix, and a wrong value changes the oscillator rather than raising, so
    the caller is made to say it.

    from_fb_weight is not a parameter here: it is fixed at
    CPG_FROM_FB_WEIGHT (see LIFCPGStepper) for every N, so there is nothing
    for a caller to get wrong by omission.
    """
    cpg = LIFCPGStepper(N=N, i_app=i_app)
    print(f"  N={N}  i_app={i_app}  from_fb_weight={CPG_FROM_FB_WEIGHT:g}")
    print(f"  Warming up CPG ({warmup} steps) ...")
    cpg.step_chunk(warmup)
    print(f"  Collecting {tmax} steps ...")
    spikes = cpg.step_chunk(tmax)

    counts = spikes.sum(0).astype(int)
    print(f"  Spikes per neuron : {counts.tolist()}")
    if counts.min() == 0:
        raise RuntimeError("A CPG neuron never fired — check W / i_app.")
    return spikes


# ═══════════════════════════════════════════════════════════════════
# 2.  Burst detection  →  phase
# ═══════════════════════════════════════════════════════════════════

def detect_burst_threshold(spike_steps, bw_method=0.3):
    """
    ISI threshold separating within-burst from between-burst gaps, taken as
    the antimode of the log-ISI KDE.  Unchanged in spirit from the previous
    pipeline; with this CPG the two modes are ~3.5 and ~226 steps so the
    split is unambiguous.
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

    # A healthy N-neuron ring puts the bursts at ~i/N.  Gaps far from 1/N mean
    # the coupling matrix and i_app don't agree (from_fb_weight is fixed at
    # CPG_FROM_FB_WEIGHT and confirmed to work across N, so it's not a
    # suspect here) -- the run will still "work", it just will not be the
    # oscillator the matrix was tuned for.
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
    p = out_dir / "burst_threshold.png"
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


# ═══════════════════════════════════════════════════════════════════
# 3.  Gait tables
# ═══════════════════════════════════════════════════════════════════
#
# Loaded from CSV, ported from train_snn.py's loader: np.loadtxt on
# gaits_dir/{name}.csv.  The file list is picked from --n_cpg_neurons
# (4 -> quadruped, 6 -> hexapod) unless overridden with --gaits.
#
# train_snn.py's own list of hexapod file stems (below) was itself dead
# code there -- it got built, sliced, and then immediately overwritten by
# the quadruped list before the load loop ran -- but the 16 names are real
# CSV files, presumably still sitting in the gaits folder, so they are
# preserved here verbatim rather than re-derived.
QUADRUPED_GAIT_FILES = ["bittle_wkF", "bittle_bk", "bittle_wkL", "bittle_wkR"]
HEXAPOD_GAIT_FILES = [
    "tripod", "tripod_huge", "tripod_right", "tripod_huge_right",
    "ripple", "ripple_tiny", "ripple_right", "ripple_tiny_right",
    "tripod_backwards", "tripod_huge_backwards", "tripod_left", "tripod_huge_left",
    "ripple_backwards", "ripple_tiny_backwards", "ripple_left", "ripple_tiny_left",
]
GAIT_FILES_BY_N = {4: QUADRUPED_GAIT_FILES, 6: HEXAPOD_GAIT_FILES}


def outputs_path(this_file_dir, rel=""):
    """
    this_file_dir/outputs[/rel].

    Used for --out_dir (train.py, visualize.py) and --model_dir
    (visualize.py) so a bare name like "test1" always lands at
    outputs/test1 instead of needing the "outputs/" prefix typed out every
    time.  rel="" (the default for all three args) resolves to
    this_file_dir/outputs itself, unchanged from the old default.
    """
    return Path(this_file_dir, "outputs", rel) if rel else Path(this_file_dir, "outputs")


def load_gait_tables(names, gaits_dir):
    """
    Load one CSV per name from gaits_dir/{name}.csv.  Identical loader to
    train_snn.py's (np.loadtxt, comma-delimited, float32); the only change
    is that the caller decides the name list instead of it being
    hardcoded, so the same function serves quadruped, hexapod, or a custom
    --gaits list.

    Validates that every table has the same column count: upsample_gait_tables
    interpolates per column, and a width mismatch there fails confusingly far
    from its actual cause.
    """
    gaits_dir = Path(gaits_dir)
    tables = []
    for nm in names:
        p = gaits_dir / f"{nm}.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"Gait file not found: {p}\n"
                f"  Looked for {len(names)} file(s) in {gaits_dir.resolve()}: "
                f"{names}\n"
                f"  Pass --gaits_dir to point at the right folder, or "
                f"--gaits to load a different set of names.")
        tables.append(np.loadtxt(p, delimiter=",", dtype=np.float32))

    widths = {t.shape[1] for t in tables}
    if len(widths) > 1:
        detail = "  ".join(f"{nm}={t.shape[1]}" for nm, t in zip(names, tables))
        raise ValueError(
            f"Gait tables have mismatched column counts, cannot form one "
            f"target array: {detail}")
    return tables, list(names)


def default_leg_layout(n_cpg_neurons, n_joints):
    """
    (n_legs, leg_cols) for a known (n_cpg_neurons, n_joints) combination.

    Quadruped (4, 8): column j and column j+4 share a circular phase offset,
    confirmed numerically against the four quadruped tables — see the module
    docstring.  Hexapod (6, 18): rows are ordered leg-major, 3 columns per
    leg (coxa, femur, tibia), legs in LF/LM/LR/RF/RM/RR order.

    Raises for anything else rather than guessing — pass --leg_cols for a
    layout this hasn't seen.
    """
    if n_cpg_neurons == 4 and n_joints == 8:
        return N_LEGS, [list(c) for c in LEG_COLS]
    if n_cpg_neurons == 6 and n_joints == 18:
        return HEXAPOD_N_LEGS, [list(c) for c in HEXAPOD_LEG_COLS]
    raise ValueError(
        f"No known leg layout for n_cpg_neurons={n_cpg_neurons}, "
        f"n_joints={n_joints}. Pass --leg_cols explicitly.")


def upsample_gait_tables(tables, names, target_rows=None, verbose=True):
    """Cubic-interpolate every table to a common row count (equal phase
    resolution across gaits, so per-gait loss isn't skewed by quantisation)."""
    if target_rows is None:
        target_rows = max(t.shape[0] for t in tables)
    out = []
    for t, nm in zip(tables, names):
        if t.shape[0] == target_rows:
            out.append(t.copy())
            if verbose:
                print(f"      {nm:>4s} : {t.shape[0]} rows (unchanged)")
        else:
            x0 = np.linspace(0.0, 1.0, t.shape[0])
            x1 = np.linspace(0.0, 1.0, target_rows)
            f  = interp1d(x0, t, axis=0, kind="cubic", fill_value="extrapolate")
            out.append(f(x1).astype(np.float32))
            if verbose:
                print(f"      {nm:>4s} : {t.shape[0]} -> {target_rows} rows (cubic)")
    return out, int(target_rows)



# ═══════════════════════════════════════════════════════════════════
# 5.  Targets
# ═══════════════════════════════════════════════════════════════════

def build_targets(phase, gait_tables, phase_zero=0.0):
    """
    phase       : (T,) float in [0,1), NaN where undefined
    gait_tables : list of (R, 8) upsampled tables

    Returns
    -------
    targets : (G, T, 8) float32, normalised to [-1, 1] with a single global
              min/max across all gaits (so one denormalisation on the robot)
    valid   : (T,) bool
    (lo,hi) : the global min/max used
    """
    T     = len(phase)
    valid = ~np.isnan(phase)
    R     = gait_tables[0].shape[0]

    ph  = np.where(valid, phase, 0.0).astype(np.float64)
    row = (((ph + phase_zero) % 1.0) * R).astype(np.int64) % R

    allv   = np.concatenate([t.ravel() for t in gait_tables])
    lo, hi = float(allv.min()), float(allv.max())

    targets = np.stack([
        ((t[row] - lo) / (hi - lo + 1e-8) * 2.0 - 1.0).astype(np.float32)
        for t in gait_tables])
    targets[:, ~valid] = 0.0
    return targets, valid, (lo, hi)


# ═══════════════════════════════════════════════════════════════════
# 6.  Stream sampler  (truncated BPTT with carried state)
# ═══════════════════════════════════════════════════════════════════

class StreamSampler:
    """
    B independent read heads walking the CPG spike train.

    Each head holds its own position, its own active gait, and its own
    countdown to the next gait switch.  `next_chunk(L)` returns L
    consecutive timesteps for every head; the caller carries (and detaches)
    the SNN state across calls, so the network is trained exactly the way it
    runs on the robot: one continuous stream, gait changing underneath it.

    A head that runs off the end of its time range is rewound to a random
    start and flagged in `reset_mask` so the caller can zero its state.

    Device residency
    ----------------
    The whole "dataset" is tiny -- spikes (T,4), targets (G,T,J) and valid
    (T,) come to well under 10 MB at tmax=50k -- so all three live on the
    training device permanently and batches are gathered on-device.  That
    removes a Python loop over B heads doing numpy slicing plus a blocking
    host->device copy per chunk.  It is worth only ~0.1% of epoch time (the
    model, not the data path, is the cost) and is done mainly so there is no
    data pipeline left to reason about.

    The per-head gait timeline stays in numpy: it is branchy control flow,
    and switches are rare (every switch_min..switch_max steps), so the
    `while` loop almost never fires.
    """

    def __init__(self, spikes, targets, valid, t_lo, t_hi, batch,
                 switch_min=600, switch_max=3000, rng=None, n_gaits=4,
                 device=None, phase=None, warm_steps=0):
        self.t_lo    = int(t_lo)
        self.t_hi    = int(t_hi)
        self.B       = int(batch)
        self.smin    = int(switch_min)
        self.smax    = int(switch_max)
        self.n_gaits = int(n_gaits)
        self.rng     = rng or np.random.default_rng(0)

        # One code path for CPU and GPU: torch indexing is identical, so
        # `device=None` simply keeps everything on the CPU.
        self.device = torch.device(device) if device is not None \
            else torch.device("cpu")
        self.spikes  = torch.as_tensor(np.ascontiguousarray(spikes),
                                       dtype=torch.float32,
                                       device=self.device)
        self.targets = torch.as_tensor(np.ascontiguousarray(targets),
                                       dtype=torch.float32,
                                       device=self.device)
        self.valid   = torch.as_tensor(
            np.ascontiguousarray(valid).astype(np.float32),
            dtype=torch.float32, device=self.device)
        # NaNs zeroed here rather than downstream: `valid` is literally
        # ~isnan(phase), so the mask already excludes exactly those steps, and
        # carrying NaNs into the loss would poison it via 0*NaN = NaN.
        ph = (np.zeros(len(spikes), dtype=np.float32) if phase is None
              else np.nan_to_num(np.asarray(phase, dtype=np.float32)))
        self.phase = torch.as_tensor(np.ascontiguousarray(ph),
                                     dtype=torch.float32, device=self.device)
        # Steps each head has advanced since its state was last zeroed.
        # Starts at 0 because init_state() really does hand out zero state.
        self.warm_steps  = int(warm_steps)
        self.since_reset = np.zeros(self.B, dtype=np.int64)

        self.pos   = self.rng.integers(t_lo, t_hi, size=self.B)
        self.gait  = self.rng.integers(0, n_gaits, size=self.B)
        self.count = self.rng.integers(self.smin, self.smax, size=self.B)
        self._off  = {}                     # cached arange(L) per bptt

    def _offsets(self, L):
        off = self._off.get(L)
        if off is None:
            off = torch.arange(L, device=self.device, dtype=torch.long)
            self._off[L] = off
        return off

    def _rewind(self, b, L):
        hi = self.t_hi - L
        if hi <= self.t_lo:
            raise ValueError(
                f"stream range [{self.t_lo}, {self.t_hi}) is shorter than "
                f"bptt={L}; raise --tmax or lower --bptt/--val_frac.")
        self.pos[b]   = self.rng.integers(self.t_lo, hi)
        self.gait[b]  = self.rng.integers(0, self.n_gaits)
        self.count[b] = self.rng.integers(self.smin, self.smax)

    def next_chunk(self, L):
        B = self.B
        g   = np.zeros((L, B), dtype=np.int64)
        sw  = np.zeros((L, B), dtype=np.float32)   # 1 on the switch step
        reset_mask = np.zeros(B, dtype=np.float32)

        # ── per-head bookkeeping: rewind + gait timeline (CPU) ────
        for b in range(B):
            if self.pos[b] + L > self.t_hi:
                self._rewind(b, L)
                reset_mask[b] = 1.0
                self.since_reset[b] = 0

            gs = np.full(L, self.gait[b], dtype=np.int64)
            c  = self.count[b]
            while c < L:
                new_g = self.rng.integers(0, self.n_gaits)
                while new_g == gs[c] and self.n_gaits > 1:
                    new_g = self.rng.integers(0, self.n_gaits)
                gs[c:] = new_g
                sw[c, b] = 1.0
                c += self.rng.integers(self.smin, self.smax)
            self.gait[b]  = int(gs[-1])
            self.count[b] = int(c - L)
            g[:, b] = gs

        # Capture the starts AFTER any rewind, then advance.
        starts = self.pos.copy()
        self.pos = self.pos + L

        # ── on-device gather ─────────────────────────────────────
        # idx[t, b] = which absolute timestep position t of head b refers to.
        # pos[None,:] is (1,B) and off[:,None] is (L,1); broadcasting gives
        # (L,B). Indexing a (T,4) tensor with an (L,B) index tensor replaces
        # the indexed dim and keeps trailing dims -> (L,B,4), already in the
        # layout the training loop wants, so no permutes.
        dev   = self.device
        pos_t = torch.as_tensor(starts, dtype=torch.long, device=dev)
        idx   = pos_t[None, :] + self._offsets(L)[:, None]      # (L, B)
        g_t   = torch.as_tensor(g, dtype=torch.long, device=dev)

        x = self.spikes[idx]                        # (L, B, n_cpg_neurons)
        y = self.targets[g_t, idx]                              # (L, B, J)
        m = self.valid[idx]                                     # (L, B)
        ph = self.phase[idx]                                    # (L, B)

        # ── post-reset warm-up mask ──────────────────────────────
        # A head whose state was just zeroed cannot know where it is in the
        # cycle yet, so its first `warm_steps` outputs are not something to
        # train against.  The FORWARD pass still runs on them -- that is how
        # the state builds -- only the loss is suppressed.  This bites for
        # roughly one head per ~(range/L) chunks, i.e. ~1% of a batch of 128,
        # so it is a small correctness fix rather than a big win.
        warm = np.ones((L, B), dtype=np.float32)
        if self.warm_steps > 0:
            for b in range(B):
                k = max(0, min(L, self.warm_steps - int(self.since_reset[b])))
                if k > 0:
                    warm[:k, b] = 0.0
        self.since_reset += L

        return (x, g_t, y, m,
                torch.as_tensor(sw, dtype=torch.float32, device=dev),
                torch.as_tensor(reset_mask, dtype=torch.float32, device=dev),
                ph,
                torch.as_tensor(warm, dtype=torch.float32, device=dev))


# ═══════════════════════════════════════════════════════════════════
# 7.  Stateful SNN
# ═══════════════════════════════════════════════════════════════════

def spike_fn(x, slope=25.0):
    """
    Heaviside forward, fast-sigmoid surrogate backward.

    Mathematically identical to the previous FastSigmoidSpike
    (torch.autograd.Function), but built from plain tensor ops so
    TorchDynamo can trace straight through it.  Custom autograd.Function
    subclasses are a common graph-break source, and even when Dynamo does
    trace past them the hand-written backward stays an opaque box the
    fuser cannot touch -- which is where most of the ops in this model are.

    Straight-through construction:

        surr = x / (slope*|x| + 1)      d(surr)/dx = 1 / (slope*|x| + 1)^2
        hard = (x > 0)                  the forward value we want

        hard + (surr - surr.detach())

    Forward:  the parentheses matter.  `surr - surr.detach()` is a value
              minus a bit-identical copy of itself, so it is EXACTLY 0.0
              in IEEE-754 for any finite input, and hard + 0.0 == hard
              bit-for-bit.  Written unparenthesised as
              `hard.detach() + surr - surr.detach()` it evaluates as
              (hard + surr) - surr, and float addition is not associative,
              so spikes come back as 1.0 +/- 1 ulp (~1.2e-7 in float32)
              instead of exactly 1.0.  Numerically irrelevant next to TF32
              matmul error, but the grouped form makes the parity test a
              hard == 0.0 gate instead of a judgement call.
    Backward: only the tracked `surr` term carries gradient (the .detach()
              term contributes none), so the gradient is
              d(surr)/dx = 1/(slope*|x| + 1)^2, character-for-character
              what FastSigmoidSpike.backward returned.

    Verify with verify_spike_fn_parity.py before trusting a training run:
    both the forward and gradient diffs against the old version should
    come out at exactly 0.0.
    """
    surr = x / (slope * x.abs() + 1.0)
    hard = (x > 0).to(x.dtype)
    return hard + (surr - surr.detach())


# Init scales that used to be the single --timing_w_scale arg.  Split and
# made internal because the arg was doing two unrelated jobs and neither
# needed user input:
#   _W_IN_INIT : CPG->timing weights.  calibrate_gains rescales these before
#                training, so this only sets relative shape and the sign of
#                the mean -- there is nothing for a user to tune.
#   _W1_INIT   : timing->sub-net layer 1.  Deliberately uncalibrated (see the
#                note at self.w1), and learnable, so this is a starting point
#                rather than a setting.
# Historical note: under the older sub_ln="both" default, layer 1 had
# LayerNorm and LN is exactly scale-invariant with b1 init at zero, so the
# shared arg had LITERALLY no effect there.  It only became live when the
# default changed to "l2".
_W_IN_INIT = 0.5
_W1_INIT   = 0.5


def init_beta_logit(shape, tau_min, tau_max, generator=None):
    """
    Log-uniform membrane time constants -> beta = exp(-1/tau), stored as a
    logit so training keeps beta in (0,1) without clamping.

    This is the single most important choice in the whole model: with no
    phase channel, the ONLY way the network can know where it is in a
    254-step cycle is that some units decay slowly enough to still carry
    the last burst.  A homogeneous beta=0.9 (tau~10) cannot do it.
    """
    u   = torch.rand(shape, generator=generator)
    tau = torch.exp(math.log(tau_min) + u * (math.log(tau_max) - math.log(tau_min)))
    beta = torch.exp(-1.0 / tau).clamp(1e-4, 1 - 1e-6)
    return torch.log(beta / (1.0 - beta))


class StatefulSNN(nn.Module):
    """
    Input  : n_neurons binary CPG spikes per timestep  (nothing else)
    Gait   : integer index, used only for FiLM conditioning
    Output : n_joints joint angles per timestep

    Fully connected: every CPG spike reaches every hidden unit, and both
    hidden layers and the readout are dense.  There is no leg grouping and
    no per-gait input routing.

    Why not leg-grouped (the previous design):
      - Layers 2+ were block diagonal (4 x Hg x Hg), so the network was
        four independent sub-networks and `cross_gain * w_cross` in layer 1
        was the ONLY path between them.
      - The routing permutation aligned each leg group with the CPG neuron
        whose burst matched that leg's swing onset, but the residuals were
        0.116-0.133 cycle for 3 of 4 gaits -- ~34 steps at period 254,
        longer than a burst (~29 steps) -- so the alignment only held for
        wkF and the network had to learn per-gait corrections anyway.
      - `w_cross` already spanned all neurons, so routing was an
        initialisation prior, not a capability. Removed.

    Purely feedforward within a timestep; all cross-timestep memory lives
    in the three leaky membranes, whose per-unit time constants are
    learnable and initialised log-uniformly over [tau_min, tau_max].  With
    no phase input and no recurrence, those taus are the ONLY mechanism
    that can hold position within a ~254-step gait cycle, so tau_max must
    comfortably exceed one period.

    FiLM conditioning
    -----------------
    `film1`/`film2` are nn.Embedding LOOKUP TABLES, not linear layers:
    they map an integer gait index to a (gamma, beta) pair per hidden
    unit, applied AFTER LayerNorm.

    After LayerNorm is the whole point.  A gait flag concatenated onto the
    input contributes an additive term to the pre-activation, and
    LayerNorm then subtracts the mean across the hidden dimension and
    divides by the std -- removing the uniform part of that offset
    outright and normalising away its scale.  Input concatenation is
    therefore fought by the very layer that keeps the LIF from
    saturating.  FiLM is applied after LN, so it survives, and gamma is
    multiplicative, which lets a gait gate which units contribute rather
    than merely bias them.

    The table is OVER-ALLOCATED to `max_gaits` rows and only the first
    `n_gaits` are used.  Unused rows initialise to identity modulation
    (gamma=1, beta=0) and receive no gradient, so they are inert.  This
    keeps every parameter shape independent of the gait count, so a
    checkpoint trained on 4 gaits loads into a run using 8.  Embedding
    lookup is O(1) in table size, so spare rows cost file size only.

    State (all (B, H)): mem1, mem2, memo.
    """

    # Consumed by export_onnx / config so neither has to branch on isinstance.
    arch            = "dense"
    state_names_in  = ("mem1_in",  "mem2_in",  "memo_in")
    state_names_out = ("mem1_out", "mem2_out", "memo_out")

    def __init__(self, hidden=128, n_gaits=4, max_gaits=16,
                 tau_min=2.0, tau_max=256.0,
                 slope=25.0, thresh=1.0, n_neurons=4, n_joints=N_JOINTS):
        super().__init__()
        if n_gaits > max_gaits:
            raise ValueError(
                f"n_gaits ({n_gaits}) > max_gaits ({max_gaits}); raise "
                f"--max_gaits. Note that changing max_gaits changes the FiLM "
                f"parameter shape and so invalidates old checkpoints.")
        self.H         = hidden
        self.n_gaits   = n_gaits
        self.max_gaits = max_gaits
        self.slope     = slope
        self.thresh    = thresh
        # Stored so export_onnx can size the dummy spike input from the model
        # rather than assuming a 4-neuron CPG.
        self.n_neurons = n_neurons
        self.n_joints  = n_joints

        # ── layer 1: all CPG spikes -> all hidden units ───────────
        # Scale 0.8 carries over from the old per-group self-drive. Only one
        # CPG neuron fires per timestep, so cur1 is a single row of w_in and
        # the magnitude matches the old design; LayerNorm follows anyway, so
        # this mostly sets the gradient scale.
        self.w_in   = nn.Parameter(torch.randn(n_neurons, hidden) * 0.8)
        self.b1     = nn.Parameter(torch.zeros(hidden))
        self.ln1    = nn.LayerNorm(hidden)

        # ── layer 2 ───────────────────────────────────────────────
        self.w2     = nn.Parameter(torch.randn(hidden, hidden) / math.sqrt(hidden))
        self.b2     = nn.Parameter(torch.zeros(hidden))
        self.ln2    = nn.LayerNorm(hidden)

        # ── readout: non-spiking leaky membrane, then joint angles ─
        self.w_read = nn.Parameter(torch.randn(hidden, hidden) / math.sqrt(hidden))
        self.b_read = nn.Parameter(torch.zeros(hidden))
        self.w_out  = nn.Parameter(torch.randn(hidden, n_joints) / math.sqrt(hidden))
        self.b_out  = nn.Parameter(torch.zeros(n_joints))

        # ── heterogeneous, learnable time constants ───────────────
        self.beta1_logit = nn.Parameter(init_beta_logit((hidden,), tau_min, tau_max))
        self.beta2_logit = nn.Parameter(init_beta_logit((hidden,), tau_min, tau_max))
        self.betao_logit = nn.Parameter(init_beta_logit((hidden,), 2.0, 40.0))

        # ── FiLM: per-gait scale/shift, over-allocated ────────────
        self.film1 = nn.Embedding(max_gaits, 2 * hidden)
        self.film2 = nn.Embedding(max_gaits, 2 * hidden)
        for e in (self.film1, self.film2):
            nn.init.zeros_(e.weight)
            e.weight.data[:, :hidden] = 1.0     # gamma := 1, beta := 0

    # ---------------------------------------------------------------
    def init_state(self, batch, device, dtype=torch.float32):
        """State is (mem1, mem2, memo) -- the three leaky membranes."""
        z = lambda: torch.zeros(batch, self.H, device=device, dtype=dtype)
        return (z(), z(), z())

    def step(self, x, gait, state):
        """
        x     : (B, n_neurons) float — CPG spikes this timestep
        gait  : (B,) int64
        state : 3-tuple (mem1, mem2, memo), each (B, H)

        Feedforward within a timestep: spk1/spk2 are local, consumed by the
        next layer in the same step and never carried across steps.
        `addmm` folds each bias into its matmul, one kernel per layer.
        """
        mem1, mem2, memo = state

        v1 = self.film1(gait)                              # (B, 2H)
        v2 = self.film2(gait)
        g1, f1 = v1[:, :self.H], v1[:, self.H:]
        g2, f2 = v2[:, :self.H], v2[:, self.H:]

        # ---- layer 1 -------------------------------------------------
        cur1  = torch.addmm(self.b1, x, self.w_in)
        cur1  = self.ln1(cur1) * g1 + f1
        beta1 = torch.sigmoid(self.beta1_logit)
        mem1  = beta1 * mem1 + cur1
        spk1  = spike_fn(mem1 - self.thresh, self.slope)
        mem1  = mem1 - self.thresh * spk1

        # ---- layer 2 -------------------------------------------------
        cur2  = torch.addmm(self.b2, spk1, self.w2)
        cur2  = self.ln2(cur2) * g2 + f2
        beta2 = torch.sigmoid(self.beta2_logit)
        mem2  = beta2 * mem2 + cur2
        spk2  = spike_fn(mem2 - self.thresh, self.slope)
        mem2  = mem2 - self.thresh * spk2

        # ---- analog readout -----------------------------------------
        curo  = torch.addmm(self.b_read, spk2, self.w_read)
        betao = torch.sigmoid(self.betao_logit)
        memo  = betao * memo + curo

        y = torch.addmm(self.b_out, memo, self.w_out)       # (B, n_joints)
        # aux is None here (no timing layer to report); the 3-tuple keeps
        # step()'s contract identical across both architectures so callers
        # never branch on type.
        return y, (mem1, mem2, memo), None

    def forward(self, x_seq, gait_seq, state=None, return_aux=False):
        """
        x_seq    : (L, B, n_neurons)
        gait_seq : (L, B)

        Returns (y_seq, state), or (y_seq, state, None) when return_aux=True
        -- this arch has no timing layer, so there are no spikes to report,
        but the signature matches TimingGroupedSNN so run_training does not
        have to know which model it has.
        """
        L, B = x_seq.shape[0], x_seq.shape[1]
        if state is None:
            state = self.init_state(B, x_seq.device, x_seq.dtype)
        ys = []
        for t in range(L):
            y, state, _ = self.step(x_seq[t], gait_seq[t], state)
            ys.append(y)
        if return_aux:
            return torch.stack(ys), state, None
        return torch.stack(ys), state


class TimingGroupedSNN(nn.Module):
    """
    CPG spikes -> TIMING layer -> G disconnected sub-networks.

    Shape of the thing
    ------------------
        x            (B, n_neurons)   CPG spikes, at most one fires per step
        timing       (B, n_timing)    LIF, per-gait weights from CPG spikes
        sub-net g    (B, Hg) x2 + memo   driven by timing neuron g ALONE
        y            (B, n_joints)    group g writes its own columns only

    G == n_timing: exactly one sub-network per timing neuron, and
    `group_cols[g]` says which gait-table columns that sub-network owns
    (see build_group_cols).

    Why split it this way
    ---------------------
    The dense model has to solve two problems in one set of weights: work
    out where in the cycle it is, and turn that into joint angles.  The
    first is shared across all joints and is cheap -- a handful of
    phase-shifted oscillations.  The second is per-joint and needs
    capacity.  Giving the timing layer n_timing units and no other job
    means the rhythm is learned in a few hundred parameters, and the
    sub-networks get a clean phase reference instead of re-deriving it
    once per leg.

    This is todo 3a: NO cross talk between sub-networks.  Sub-network g
    sees exactly one binary channel.  Layers 2+ are block diagonal, the
    readout is block diagonal, and there is no path between groups
    anywhere after the timing layer.

    Per-gait input weights, and why the shared router was reverted
    -------------------------------------------------------------
    `w_in_gait` is an Embedding(max_gaits, n_neurons * n_timing) reshaped to
    a per-gait (n_neurons, n_timing) matrix, and `b_t` is a matching
    Embedding(max_gaits, n_timing) per-gait bias.  Each gait gets its own free
    routing matrix and its own bias; nothing is shared between them.  There is
    no FiLM on this layer -- see the init block for why its gamma was provably
    redundant against a free per-gait matrix, and why its beta survives as the
    per-gait bias.

    A shared alternative was built and measured: one pair of weight matrices
    with a per-gait FiLM gate on a 16-unit LIF hidden layer in between, so
    that gaits would express their routings in a shared vocabulary
    (`W_eff(g) = W2 diag(gamma_g) W1`).  It is strictly expressive enough --
    randomised gates over a shared router reach hundreds of distinct
    routings, including the many-to-one patterns a tripod needs.  It still
    failed: the learned timing phases came out essentially IDENTICAL across
    gaits, and the measured routing matrices were near-copies.

    Why, most likely.  Two mechanisms, both pointing the same way:

      1. Competing capacity.  The sub-network FiLM tables carry ~3,072
         parameters per gait against the router gate's ~44, and sat 1-2
         spiking layers from the loss rather than 3-4.  Each spiking layer
         attenuates gradient by the surrogate derivative
         1/(slope*|x|+1)^2 -- order 1e-3 at slope 25 -- so the sub-network
         gate's gradient is orders of magnitude larger.  Gradient descent
         put the gait knowledge where it was cheapest to put it, and the
         timing layer collapsed to a gait-independent clock.

      2. Gradient averaging.  With W1/W2 shared, gradients from gaits that
         want DIFFERENT routings land in the same weights and partially
         cancel, so the tug-of-war resolves toward one compromise routing.
         A per-gait table removes the averaging entirely: gait g's weights
         only ever see gait g's gradient.

    So the embedding is the version that demonstrably separates gaits, and
    it is what is here.  It is not the aesthetically preferred answer --
    n_gaits appears in a parameter shape, nothing transfers between gaits,
    and per-gait capacity grows linearly in the gait count.  Finding a
    conditioning scheme that separates gaits WITHOUT a per-gait weight
    table is tracked in architecture_change_todo.md; the shared-router
    attempt is in git history and should be reproducible from it.

    Many-to-one routing is the normal case, not an edge case.  A tripod
    needs 3 legs at one phase and 3 at the opposite phase; ripple needs 3
    pairs; wave needs 6 singletons.  Nothing here constrains the routing to
    be a permutation, which is why a hard permutation parameterisation would
    have been the wrong choice -- it cannot collapse three legs onto one
    phase.

    No LayerNorm on the timing layer
    --------------------------------
    LN subtracts the mean across the normalised dimension.  Over 6-16 units
    that mean IS the signal: "some CPG neuron fired this timestep" is almost
    entirely common mode, and LN deletes it.  Worse, during CPG silence
    cur = bias, and LN(bias) is a FIXED NONZERO vector once bias trains
    away from uniform -- so every unit would receive tonic drive and
    free-run during the silent gap instead of staying quiet, which is the
    opposite of what a rhythm layer should do.  A 256-wide layer tolerates
    this (the dense model has the same property and works); a 6-wide one
    will not.

    Instead, the firing regime is set explicitly: `calibrate_gains` bisects
    a per-unit multiplier into FiLM's gamma before training so every unit
    starts inside a target spikes-per-cycle band, and the chosen
    SpikeObjective then shapes it from there.  See those.

    Inside the sub-networks LN is optional (`sub_ln`) and uses
    elementwise_affine=False: a shared gamma/beta over (G, Hg) would be a
    parameter tied ACROSS groups, which breaks "fully disconnected".  FiLM
    follows and supplies per-group per-gait affine anyway.

    State (4 tensors, mixed rank):
        mem_timing (B, n_timing)
        mem1, mem2 (B, G, Hg)
        memo       (B, G, H_o)   -- narrower; see the readout note in __init__
    """

    arch = "timing_grouped"
    state_names_in  = ("mem_timing_in",  "mem1_in",  "mem2_in",  "memo_in")
    state_names_out = ("mem_timing_out", "mem1_out", "mem2_out", "memo_out")

    def __init__(self, hidden_per_group=128, n_gaits=4, max_gaits=16,
                 n_neurons=4, n_timing=N_LEGS, group_cols=None,
                 n_joints=N_JOINTS, readout_hidden=32,
                 tau_min=2.0, tau_max=256.0,
                 tau_timing_min=2.0, tau_timing_max=64.0,
                 tau_readout_max=40.0,
                 sub_ln="l2", sub_film="both",
                 timing_reset="zero", event_gated=True,
                 slope=25.0, timing_slope=None, thresh=1.0):
        super().__init__()
        if n_gaits > max_gaits:
            raise ValueError(
                f"n_gaits ({n_gaits}) > max_gaits ({max_gaits}); raise "
                f"--max_gaits. Note that changing max_gaits changes the FiLM "
                f"parameter shapes and so invalidates old checkpoints.")
        if sub_ln not in ("none", "l1", "l2", "both"):
            raise ValueError(f"sub_ln must be none|l1|l2|both, got {sub_ln!r}")
        if sub_film not in ("none", "l1", "l2", "both"):
            raise ValueError(f"sub_film must be none|l1|l2|both, got {sub_film!r}")
        if timing_reset not in ("zero", "subtract"):
            raise ValueError(f"timing_reset must be zero|subtract, got "
                             f"{timing_reset!r}")

        group_cols = (build_group_cols(n_timing, n_joints=n_joints)
                      if group_cols is None else
                      [list(g) for g in group_cols])
        if len(group_cols) != n_timing:
            raise ValueError(
                f"group_cols has {len(group_cols)} groups but n_timing="
                f"{n_timing}; there is exactly one sub-network per timing "
                f"neuron.")

        G   = int(n_timing)
        Hg  = int(hidden_per_group)
        Ho  = int(readout_hidden)
        C   = len(group_cols[0])            # output columns per group

        self.G          = G
        self.Hg         = Hg
        self.Ho         = Ho
        self.C          = C
        self.n_timing   = G
        self.n_neurons  = int(n_neurons)
        self.n_joints   = int(n_joints)
        self.n_gaits    = int(n_gaits)
        self.max_gaits  = int(max_gaits)
        self.slope      = slope
        # A wider surrogate on the small spiking layers: the gradient through
        # spike_fn is 1/(slope*|x|+1)^2, so at slope=25 a unit sitting a few
        # units below threshold is nearly invisible to gradients.  These two
        # layers are the ones where a dead unit is catastrophic rather than
        # merely wasteful, so they get a gentler slope by default.
        self.timing_slope = float(slope if timing_slope is None
                                  else timing_slope)
        self.thresh     = thresh
        self.sub_ln     = sub_ln
        self.sub_film   = sub_film
        self.timing_reset = timing_reset
        self.event_gated  = bool(event_gated)
        self.group_cols = group_cols
        self.H          = Hg          # alias for generic callers

        # ── output column routing ─────────────────────────────────
        flat = [c for grp in group_cols for c in grp]
        inv  = np.empty(n_joints, dtype=np.int64)
        for pos, col in enumerate(flat):
            inv[col] = pos
        self.register_buffer("out_perm", torch.from_numpy(inv), persistent=False)

        # ── timing: per-gait CPG->timing weight matrix ────────────
        # Embedding(max_gaits, n_neurons * n_timing) reshaped per gait, so
        # gait g gets its own free (n_neurons, n_timing) routing matrix with
        # nothing shared.  See the class docstring for why the shared-router
        # alternative was tried and reverted.
        #
        # There is NO FiLM on this layer.  There used to be, and its gamma was
        # provably redundant: with W already a free per-gait matrix,
        #     gamma_g * (x W_g + b) + beta_g  ==  x (W_g gamma_g) + (gamma_g b + beta_g)
        # so gamma is exactly absorbable into W_g and adds no expressiveness
        # (verified numerically to 1e-16).  FiLM's beta was NOT redundant --
        # it supplied a PER-GAIT bias where b_t was gait-shared -- so that
        # capability is kept, expressed directly as a per-gait bias embedding
        # below rather than hidden inside a gate.  If the per-gait weight
        # table is ever replaced by a shared scheme, gamma becomes
        # load-bearing again and FiLM should come back here.
        #
        # INIT SCALE is arbitrary: calibrate_gains rescales these columns to
        # hit a target firing rate before training starts, so whatever is set
        # here only fixes the relative SHAPE.  A positive mean matters though,
        # and is not cosmetic: calibration multiplies the current, and
        # multiplying a net-negative current by a larger positive gain moves it
        # FURTHER from threshold, so a unit born net-negative cannot be
        # rescued by calibration at all.  Sign is fixed here; magnitude is
        # calibration's job.
        self.w_in_gait = nn.Embedding(max_gaits, self.n_neurons * G)
        nn.init.normal_(self.w_in_gait.weight, mean=_W_IN_INIT, std=_W_IN_INIT)

        # Per-gait bias.  Note this is tonic drive -- present on silent steps
        # too -- which is exactly why it is wanted here: with
        # timing_reset="zero" a timing unit can only fire when input arrives,
        # so bias is the ONLY mechanism by which it can spike BETWEEN CPG
        # bursts.  Under event gating that maintenance firing is what the
        # sub-networks need, so tonic drive is load-bearing rather than the
        # liability it would be for a CPG-locked burster.
        self.b_t = nn.Embedding(max_gaits, G)
        nn.init.zeros_(self.b_t.weight)
        self.beta_t_logit = nn.Parameter(
            init_beta_logit((G,), tau_timing_min, tau_timing_max))

        # ── sub-network layer 1: ONE binary spike -> Hg units ─────
        # Fixed init scale, deliberately NOT calibrated.  A dead timing
        # neuron silences a whole sub-network, which is why that layer gets
        # calibration; a few dead units out of Hg here are noise, and w1 plus
        # film1's gamma are both learnable so the init only sets the
        # optimisation path, not what is reachable.
        self.w1 = nn.Parameter(torch.randn(G, Hg) * _W1_INIT)
        self.b1 = nn.Parameter(torch.zeros(G, Hg))

        # ── sub-network layer 2: block diagonal (G, Hg, Hg) ───────
        self.w2 = nn.Parameter(torch.randn(G, Hg, Hg) / math.sqrt(Hg))
        self.b2 = nn.Parameter(torch.zeros(G, Hg))

        # ── block-diagonal analog readout ─────────────────────────
        # Ho, not Hg: memo is a BANK OF LOW-PASS FILTERS (one tau per unit,
        # see betao_logit) whose weighted sum is the output, so its width is
        # the size of the temporal basis available for synthesising the output
        # waveform -- not the output width.  Ho == C would give exactly one
        # basis function per output column and no redundancy, and because
        # filter-then-combine differs from combine-then-filter when the taus
        # differ, projecting down to C BEFORE the membrane would leave only C
        # distinct (projection, tau) pairs for the whole group.  Ho == Hg (the
        # original) is the other extreme and was ~40% of the model's params.
        self.w_read = nn.Parameter(torch.randn(G, Hg, Ho) / math.sqrt(Hg))
        self.b_read = nn.Parameter(torch.zeros(G, Ho))
        self.w_out  = nn.Parameter(torch.randn(G, Ho, C) / math.sqrt(Ho))
        self.b_out  = nn.Parameter(torch.zeros(G, C))

        self.ln1 = nn.LayerNorm(Hg, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(Hg, elementwise_affine=False)

        self.beta1_logit = nn.Parameter(init_beta_logit((G, Hg), tau_min, tau_max))
        self.beta2_logit = nn.Parameter(init_beta_logit((G, Hg), tau_min, tau_max))
        self.betao_logit = nn.Parameter(
            init_beta_logit((G, Ho), 2.0, tau_readout_max))

        self.film1 = nn.Embedding(max_gaits, 2 * G * Hg)
        self.film2 = nn.Embedding(max_gaits, 2 * G * Hg)
        for e in (self.film1, self.film2):
            nn.init.zeros_(e.weight)
            e.weight.data[:, :G * Hg] = 1.0

    # ---------------------------------------------------------------
    def init_state(self, batch, device, dtype=torch.float32):
        z = lambda w: torch.zeros(batch, self.G, w, device=device, dtype=dtype)
        mem_t = torch.zeros(batch, self.G, device=device, dtype=dtype)
        # memo is Ho wide, not Hg (see the readout comment in __init__).
        return (mem_t, z(self.Hg), z(self.Hg), z(self.Ho))

    # ---------------------------------------------------------------
    def _timing(self, x, gait, mem_t):
        """
        One timestep of the timing layer.  x (B, n_neurons) -> spk_t (B, G).

        Factored out so calibration and diagnostics can run this layer without
        the sub-networks, and without going through the compiled `step` (which
        would add a Dynamo guard set per new shape).

        RESET: `timing_reset="zero"` by default, matching the CPG.  This is not
        a stylistic choice -- it is what makes the burst structure match.  With
        subtractive reset, firing leaves `mem - thresh` behind, and if that
        residual is still above threshold the unit fires again on the NEXT
        step even with no input.  Since CPG spikes arrive every 2nd step
        (refrac_main=1), the result is ISIs of 2,1,1,2,1,1,... -- consecutive
        spikes filling the gaps -- and ~50% more spikes per burst than the CPG
        has.  Reset-to-zero discards the residual, so firing tracks the input's
        spacing and the burst comes out at the CPG's own width and count.
        `LIFGeneralArray` (the CPG) does exactly this: `v[spike] = 0`.
        """
        W   = self.w_in_gait(gait).view(-1, self.n_neurons, self.G)
        cur = torch.bmm(x.unsqueeze(1), W).squeeze(1) + self.b_t(gait)
        mem_t = torch.sigmoid(self.beta_t_logit) * mem_t + cur
        spk_t = spike_fn(mem_t - self.thresh, self.timing_slope)
        if self.timing_reset == "zero":
            mem_t = mem_t * (1.0 - spk_t)
        else:
            mem_t = mem_t - self.thresh * spk_t
        return spk_t, mem_t

    def step(self, x, gait, state):
        """
        x     : (B, n_neurons) float — CPG spikes this timestep
        gait  : (B,) int64
        state : (mem_timing, mem1, mem2, memo)

        Returns (y, state, aux) where aux = (spk_timing,).
        `aux` exists so the spike-statistics penalty can see the spikes without
        a second forward pass; the ONNX wrappers unpack and discard it, and
        since spk_timing already feeds `y` it is in the graph regardless.  Kept
        as a 1-tuple so callers iterate over it uniformly.
        """
        mem_t, mem1, mem2, memo = state
        G, Hg = self.G, self.Hg

        spk_t, mem_t = self._timing(x, gait, mem_t)          # (B, G)

        # ---- event gate -------------------------------------------
        # With event_gated, sub-network g advances ONLY on steps where its
        # timing neuron fired.  Membranes still decay every step (that is
        # deliberate -- it is what makes the OUTPUT change between spikes and
        # therefore what makes spike PLACEMENT matter to the task loss), but
        # no current is injected and no sub-network spike is emitted.
        #
        # Why this is the point rather than an efficiency tweak: without it,
        # the task loss is EXACTLY invariant to the timing layer's burst
        # phase.  A sub-network whose taus span a cycle has a complete
        # "time since burst" basis, so shifting the burst by any amount is
        # absorbed by relearning the waveform offset.  Gating removes that
        # freedom -- the output genuinely cannot track the target between
        # spikes -- so alignment becomes something the loss can see.
        #
        # NOTE the gate multiplies both `cur` and `spk` on each layer, so
        # gradient reaching spk_t through those paths is scaled up.  Forward
        # is exact (gate is 0/1, so gate*gate == gate).  Left uncorrected: it
        # is a scale factor rather than a bug, and a stronger gradient into
        # the timing layer is desirable here, not a problem.
        gate = spk_t.unsqueeze(-1) if self.event_gated else None

        # ---- sub-net layer 1 ---------------------------------------
        # w1 and b1 are now redundant (only their sum enters) because the
        # gate has replaced the spk_t multiplication that used to separate
        # them.  Both kept so checkpoint shapes are unchanged.
        if self.event_gated:
            cur1 = self.w1 + self.b1                          # (G, Hg)
        else:
            cur1 = spk_t.unsqueeze(-1) * self.w1 + self.b1    # (B, G, Hg)
        if self.sub_ln in ("l1", "both"):
            cur1 = self.ln1(cur1)
        if self.sub_film in ("l1", "both"):
            v1 = self.film1(gait).view(-1, 2, G, Hg)
            cur1 = cur1 * v1[:, 0] + v1[:, 1]
        if gate is not None:
            cur1 = gate * cur1
        mem1  = torch.sigmoid(self.beta1_logit) * mem1 + cur1
        spk1  = spike_fn(mem1 - self.thresh, self.slope)
        if gate is not None:
            spk1 = gate * spk1
        mem1  = mem1 - self.thresh * spk1

        # ---- sub-net layer 2: block diagonal ------------------------
        cur2 = torch.einsum("bgh,ghk->bgk", spk1, self.w2) + self.b2
        if self.sub_ln in ("l2", "both"):
            cur2 = self.ln2(cur2)
        if self.sub_film in ("l2", "both"):
            v2 = self.film2(gait).view(-1, 2, G, Hg)
            cur2 = cur2 * v2[:, 0] + v2[:, 1]
        if gate is not None:
            cur2 = gate * cur2
        mem2  = torch.sigmoid(self.beta2_logit) * mem2 + cur2
        spk2  = spike_fn(mem2 - self.thresh, self.slope)
        if gate is not None:
            spk2 = gate * spk2
        mem2  = mem2 - self.thresh * spk2

        # ---- block-diagonal analog readout -------------------------
        # memo is NOT gated on the decay side: it leaks every timestep, so
        # between timing spikes the output relaxes toward b_out.  That decay
        # is the mechanism that penalises spikes placed where the waveform
        # does not need them.
        curo  = torch.einsum("bgh,ghk->bgk", spk2, self.w_read) + self.b_read
        if gate is not None:
            curo = gate * curo
        memo  = torch.sigmoid(self.betao_logit) * memo + curo

        y_grp = torch.einsum("bgh,ghc->bgc", memo, self.w_out) + self.b_out
        y     = y_grp.flatten(1).index_select(1, self.out_perm)
        return y, (mem_t, mem1, mem2, memo), (spk_t,)

    def forward(self, x_seq, gait_seq, state=None, return_aux=False):
        """
        x_seq    : (L, B, n_neurons)
        gait_seq : (L, B)

        Returns (y_seq, state), or (y_seq, state, (spk_t_seq,)) when
        return_aux=True.
        """
        B = x_seq.shape[1]
        if state is None:
            state = self.init_state(B, x_seq.device, x_seq.dtype)
        ys, at = [], []
        for t in range(x_seq.shape[0]):
            y, state, aux = self.step(x_seq[t], gait_seq[t], state)
            ys.append(y)
            if return_aux:
                at.append(aux[0])
        if return_aux:
            return torch.stack(ys), state, (torch.stack(at),)
        return torch.stack(ys), state

    # ---------------------------------------------------------------
    @torch.no_grad()
    def timing_only(self, x_seq, gait_seq, mem_t=None):
        """
        (L, B, n_timing) timing spikes straight from CPG spikes.

        Runs the eager `_timing`, so this is exact -- the same spikes the
        sub-networks saw -- and costs nothing next to the sub-networks it
        skips.
        """
        B = x_seq.shape[1]
        if mem_t is None:
            mem_t = torch.zeros(B, self.G, device=x_seq.device,
                                dtype=x_seq.dtype)
        out = []
        for t in range(x_seq.shape[0]):
            spk, mem_t = self._timing(x_seq[t], gait_seq[t], mem_t)
            out.append(spk)
        return torch.stack(out)

    # ---------------------------------------------------------------
    def param_breakdown(self):
        """Grouped parameter counts, for the startup print."""
        n = lambda *ps: int(sum(p.numel() for p in ps))
        return {
            "timing":  n(self.w_in_gait.weight, self.b_t.weight,
                         self.beta_t_logit),
            "sub_l1":  n(self.w1, self.b1, self.beta1_logit),
            "sub_l2":  n(self.w2, self.b2, self.beta2_logit),
            "readout": n(self.w_read, self.b_read, self.w_out, self.b_out,
                         self.betao_logit),
            "sub_film": n(self.film1.weight, self.film2.weight),
        }



class SingleStepONNX(nn.Module):
    """Flat-signature wrapper so the exported graph is one timestep with
    explicit state in/out — the robot calls this once per CPG step."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spikes, gait, mem1, mem2, memo):
        y, (m1, m2, mo), _ = self.model.step(
            spikes, gait, (mem1, mem2, memo))
        return y, m1, m2, mo


class SingleStepONNXTiming(nn.Module):
    """
    As SingleStepONNX but for TimingGroupedSNN's 4-tensor state.

    Written as an explicit signature rather than *state: torch.onnx.export
    traces varargs unreliably, and the input names have to line up
    positionally with `model.state_names_in` anyway.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spikes, gait, mem_timing, mem1, mem2, memo):
        y, (mt, m1, m2, mo), _ = self.model.step(
            spikes, gait, (mem_timing, mem1, mem2, memo))
        return y, mt, m1, m2, mo


# ═══════════════════════════════════════════════════════════════════
# 7a.  Keeping the small spiking layers alive
# ═══════════════════════════════════════════════════════════════════
#
# The router and timing layers have no LayerNorm (see TimingGroupedSNN's
# docstring for why), so nothing automatically holds their input current in
# the range where a threshold-1.0 LIF actually fires.  Three mechanisms,
# acting at different times:
#
#   calibrate_gains      once, before training  -- never start dead
#   SpikeObjective       every gradient step    -- shape the spike train
#   reinit_dead_units    on detection           -- rescue what died anyway
#
# A dead timing unit is not merely a wasted unit: its sub-network's input
# weights `w1[l]` have gradient exactly proportional to that unit's spike
# output, so while it is silent w1[l] receives EXACTLY zero gradient and
# stays at random init, while everything downstream of it trains happily on
# the constant b1-driven activity.  If the unit revives late, the one matrix
# that consumes its input has learned nothing and the LR has already decayed.
# That is why reinit_dead_units re-rolls w1 too, not just the router path.


@torch.no_grad()
def measure_rates(model, spikes, n_gaits, device,
                  t0=0, n_steps=None, period=254.0):
    """
    Spikes-per-cycle for every timing unit, for every gait, measured by
    replaying the CPG spike train at batch 1.

    Returns (n_gaits, n_timing) float array.  Forward passes only.
    """
    if n_steps is None:
        n_steps = int(min(20 * period, len(spikes) - t0))
    n_steps = int(min(n_steps, len(spikes) - t0))
    x = torch.as_tensor(spikes[t0:t0 + n_steps], dtype=torch.float32,
                        device=device).unsqueeze(1)
    n_cycles = max(n_steps / float(period), 1e-9)

    out = []
    for g in range(n_gaits):
        gg = torch.full((n_steps, 1), g, dtype=torch.long, device=device)
        spk = model.timing_only(x, gg)
        out.append((spk[:, 0].sum(0) / n_cycles).cpu().numpy())
    return np.stack(out)


@torch.no_grad()
def calibrate_gains(model, spikes, n_gaits, device, period,
                    lo=1.0, hi=5.0, iters=18, g_lo=1e-3, g_hi=1e3,
                    verbose=True):
    """
    Scale each timing unit's CPG->timing weight column so its firing rate
    starts inside [lo, hi] spikes per cycle.

    This is what `--timing_w_scale` used to be for, done by measurement
    instead of by a user-supplied number.  It writes directly into
    `w_in_gait` (it used to go through FiLM's gamma, which was only ever a
    convenient handle -- and that gamma is gone now, being provably
    absorbable into a free per-gait matrix).

    HOW.  Rate is monotone non-decreasing in a unit's own gain: more input
    current can only add threshold crossings, never remove them.  So a
    bisection per unit finds a gain landing in the band.  Rate is a STEP
    function of gain (it jumps by whole spikes, with flat stretches between),
    which is exactly why the target is a band rather than an exact value.

    Units are independent: timing unit l's current depends only on column l of
    each gait's weight matrix and on its own bias.  So all units bisect in
    parallel, and scaling column l cannot disturb any other unit.

    WHY WEIGHTS AND NOT BIAS.  Adding a positive constant to the bias would
    also raise the rate, monotonically and with no sign caveat.  But bias is
    tonic drive, present during CPG silence too, so raising it manufactures
    activity that is unrelated to the CPG rhythm -- alive and uninformative.
    Scaling the weights amplifies whatever phase preference the unit already
    has.  (The per-gait bias is still free to LEARN tonic drive, which under
    event gating is how a unit fires between bursts at all; the point is only
    that calibration should not be the thing that sets it.)

    WHY IT CAN STILL FAIL.  Gain cannot fix a wrong sign.  If a unit's net
    input current is negative, multiplying by a larger positive gain moves it
    FURTHER from threshold, and the bisection rides to g_hi and gives up.  The
    positive-mean init on `w_in_gait` exists to prevent that.  Units that
    cannot be brought into band are reported, not silently left.

    WHICH GAIT.  One gain per unit, applied to that unit's column in EVERY
    gait's matrix, so the relative shape across gaits is preserved.  The rate
    targeted is the MINIMUM over gaits, because the failure that matters is
    dead-for-one-particular-gait, not dead-on-average.
    """
    if not hasattr(model, "w_in_gait"):
        return {}

    MG, n_cpg, G = model.max_gaits, model.n_neurons, model.G
    # (max_gaits, n_cpg, n_timing) view; column l is timing unit l's inputs.
    W = model.w_in_gait.weight.data.view(MG, n_cpg, G)
    base = W.clone()

    lo_g = torch.full((G,), g_lo, device=device)
    hi_g = torch.full((G,), g_hi, device=device)

    def rates_at(gain):
        # gain is (G,); broadcast over (max_gaits, n_cpg, G) so each timing
        # unit's whole column is scaled in every gait at once.
        W.copy_(base * gain.view(1, 1, G))
        r = measure_rates(model, spikes, n_gaits, device, period=period)
        return torch.as_tensor(r, device=device).min(dim=0).values

    for _ in range(iters):
        mid = torch.sqrt(lo_g * hi_g)                   # geometric midpoint
        r   = rates_at(mid)
        too_quiet = r < lo
        too_loud  = r > hi
        lo_g = torch.where(too_quiet, mid, lo_g)
        hi_g = torch.where(too_loud,  mid, hi_g)
        if not (too_quiet | too_loud).any():
            break

    # sqrt(lo*hi) is safe to return rather than tracking the last verified
    # gain separately: once a unit is in band NEITHER bound updates again, so
    # its midpoint is frozen at exactly the value that was measured in band.
    # (Checked -- tracking it separately gives bit-identical gains.)
    final_gain  = torch.sqrt(lo_g * hi_g)
    final_rates = rates_at(final_gain)
    ok = ((final_rates >= lo * 0.5) & (final_rates <= hi * 2.0))
    report = {"timing": {
        "gain":     [float(v) for v in final_gain.cpu()],
        "min_rate": [float(v) for v in final_rates.cpu()],
        "in_band":  [bool(v) for v in ok.cpu()],
    }}
    if verbose:
        fmt = lambda v: " ".join(f"{x:6.2f}" for x in v)
        print(f"      calibrated timing: gain [{fmt(report['timing']['gain'])}]")
        print(f"                         min spk/cyc across gaits "
              f"[{fmt(report['timing']['min_rate'])}]")
        bad = [i for i, v in enumerate(ok.cpu()) if not v]
        if bad:
            print(f"      WARNING: timing unit(s) {bad} could not be brought "
                  f"into [{lo}, {hi}] spk/cyc by scaling alone — likely "
                  f"net-negative input current, which a positive gain cannot "
                  f"fix. The spike objective will keep pushing; if they stay "
                  f"dead, the _W_IN_INIT positive mean is the thing to look "
                  f"at.")
    return report


def cpg_spike_stats(spikes, phase, period):
    """
    The CPG's own firing statistics, per neuron then averaged: spikes per
    cycle, and circular concentration R of the spike phases.

    These become the TARGETS for the timing layer, which is why they are
    MEASURED rather than configured -- "fire like the CPG does" needs no
    hyperparameter.  Hardcoding a value would also be wrong: R=1.0 is
    unsatisfiable, since 10 spikes at 2-step spacing inside a 352-step cycle
    top out near 0.995, so a target of 1.0 would apply permanent pressure that
    can never be met.

    Also used to size the calibration band, so the timing layer starts in the
    right neighbourhood before any training happens.
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


# ── Spike-statistics objectives (strategy pattern) ───────────────
#
# What the timing layer's spike train SHOULD look like is an open research
# question, and the answer changes with the architecture, so the objective is
# swappable rather than hardcoded.  Add a subclass, give it a `name`, and it
# becomes available as `--spike_objective <name>` automatically.
#
# All objectives share one signature so run_training never branches:
#
#     penalty(spk, gait, phase, mask) -> scalar tensor
#
#     spk   : (L, B, U) surrogate spikes from step()'s aux
#     gait  : (L, B) int64 gait label per stream per timestep
#     phase : (L, B) cycle phase in [0,1), NaNs already zeroed
#     mask  : (L, B) 1 where the sample counts (phase valid AND past the
#             post-reset warm-up)
#
# Statistics are grouped per (gait, unit), never averaged over the batch:
# streams in a chunk carry different gaits, so a batch average would hide a
# unit misbehaving for exactly one gait.  Gaits absent from a chunk are masked
# out rather than counted as silent.


class SpikeObjective:
    """Base class.  Subclasses implement `penalty`; `lam == 0` disables."""

    name = "base"

    def __init__(self, lam=0.0, period=254.0, n_gaits=4,
                 target_rate=None, target_R=None):
        self.lam        = float(lam)
        self.period     = float(period)
        self.n_gaits    = int(n_gaits)
        self.target_rate = target_rate     # CPG spikes per cycle
        self.target_R    = target_R        # CPG circular concentration

    # -- shared helper ------------------------------------------------
    def _grouped(self, spk, gait, mask):
        """(rate, present, cnt) per (gait, unit).  rate is per-timestep."""
        oh   = torch.nn.functional.one_hot(gait, self.n_gaits).to(spk.dtype)
        oh   = oh * mask.unsqueeze(-1)                        # (L,B,G)
        cnt  = torch.einsum("lbg,lbu->gu", oh, spk)
        den  = oh.sum(dim=(0, 1))
        rate = cnt / den.clamp(min=1.0).unsqueeze(1)
        present = (den > 0).to(spk.dtype).unsqueeze(1)
        return rate, present, cnt, oh

    @property
    def enabled(self):
        return self.lam > 0.0

    def penalty(self, spk, gait, phase, mask):
        raise NotImplementedError

    def describe(self):
        return f"{self.name} (lam={self.lam:g})"


class NoSpikeObjective(SpikeObjective):
    """No constraint on the timing layer's spike statistics."""

    name = "none"

    @property
    def enabled(self):
        return False

    def penalty(self, spk, gait, phase, mask):
        return spk.sum() * 0.0

    def describe(self):
        return "none (timing spike statistics unconstrained)"


class CPGMatchSpikeObjective(SpikeObjective):
    """
    Fire LIKE THE CPG: match its spikes-per-cycle and its burst tightness.

    Two terms, because rate alone is not enough -- 10 spikes per cycle is
    satisfied equally well by one tight burst and by one lone spike every 35
    steps, so rate constrains the COUNT but not the CLUSTERING:

      rate  ((rate - target)/target)^2, two-sided, relative so lam is
            scale-free.
      conc  relu(target_R - R)^2, one-sided, where R is the circular
            concentration of the unit's spike phases,
                R = |sum_t spk_t exp(i 2pi phase_t)| / sum_t spk_t
            R near 1 means every spike lands at the same cycle phase, which
            IS "one burst per cycle".  `phase` is constant and `spk` carries
            the surrogate gradient, so this is differentiable as written.
            One-sided because a burst tighter than the CPG's is not a problem.

    KNOWN LIMITATION (todo item 10): R uses the fundamental only, so two
    bursts at opposite phases cancel to R ~ 0 and are punished as hard as
    spikes smeared uniformly.  Correct for gaits where each leg swings once
    per cycle; wrong for a genuinely two-burst gait.

    NOT COMPATIBLE with event_gated sub-networks: this objective wants every
    spike at one phase, whereas a gated sub-network needs spikes wherever its
    output must change.  Kept because it is the objective that produced the
    CPG-matched bursts, and because the comparison is worth being able to
    re-run.
    """

    name = "cpg_match"

    def penalty(self, spk, gait, phase, mask):
        rate, present, cnt, oh = self._grouped(spk, gait, mask)
        tgt = self.target_rate / self.period

        rate_err = ((rate - tgt) / tgt) ** 2 * present

        ang = 2.0 * math.pi * phase
        C = torch.einsum("lbg,lbu,lb->gu", oh, spk, torch.cos(ang))
        S = torch.einsum("lbg,lbu,lb->gu", oh, spk, torch.sin(ang))
        R = torch.sqrt(C ** 2 + S ** 2 + 1e-8) / cnt.clamp(min=1e-2)
        # Only shape units that are already firing: R is meaningless for a
        # silent unit and the rate term is what should push it up.  Detached
        # hard mask, so no gradient flows through the gate itself.
        alive = (cnt > 1.0).to(spk.dtype).detach()
        conc_err = torch.relu(self.target_R - R) ** 2 * present * alive

        denom = present.sum().clamp(min=1.0) * spk.shape[2]
        return self.lam * (rate_err.sum() + conc_err.sum()) / denom

    def describe(self):
        return (f"cpg_match (lam={self.lam:g}, target "
                f"{self.target_rate:.2f} spk/cyc, R={self.target_R:.3f})")


class MinCountSpikeObjective(SpikeObjective):
    """
    Spend as few spikes as possible.  Pay a flat cost per spike and let the
    TASK loss decide where they are worth spending.

    Only meaningful with event_gated sub-networks, and then it is the whole
    idea: a gated sub-network's output can only change on a timing spike, so
    the task loss already forces spikes wherever the waveform must move.
    Adding a uniform per-spike cost on top means the cheapest solution is to
    place spikes densely where the target changes fast and sparsely where it
    creeps -- i.e. an adaptive sampling clock, derived from the data rather
    than supervised.  Nothing has to be told which phase is "swing".

    LINEAR in the rate, not squared, on purpose: a flat marginal cost per
    spike is the L1-sparsity form and drives genuine sparsity, whereas a
    squared cost pushes hard at high rates and then gives up as the rate
    falls, which is the opposite of what is wanted.

    Reported as DUTY CYCLE (fraction of timesteps with a spike), so `lam`
    reads directly as "loss cost of a unit spiking every single timestep".

    NO FLOOR TERM, deliberately: the task loss is the floor.  With gating, a
    silent timing unit starves its sub-network completely -- memo decays,
    y collapses to b_out -- so silence is heavily punished by the task loss
    itself.  The risk is the transient: early in training the task gradient
    through a barely-firing unit is weak, so this penalty can prune a unit to
    silence before the task loss has learned to need it, and a fully silent
    unit passes ZERO gradient to w1/w2/w_read/w_out (only b_out still learns),
    making it unrecoverable.  Mitigations: `--spike_lambda_warmup` ramps lam
    from 0 so the network learns to use spikes before being charged for them,
    and `--reinit_dead_after` revives anything that dies anyway.
    """

    name = "min_count"

    def penalty(self, spk, gait, phase, mask):
        rate, present, cnt, oh = self._grouped(spk, gait, mask)
        denom = present.sum().clamp(min=1.0) * spk.shape[2]
        return self.lam * (rate * present).sum() / denom

    def describe(self):
        return (f"min_count (lam={self.lam:g}, linear/L1 in duty cycle; "
                f"the task loss supplies the floor)")


SPIKE_OBJECTIVES = {cls.name: cls for cls in (
    NoSpikeObjective, CPGMatchSpikeObjective, MinCountSpikeObjective)}


def make_spike_objective(name, **ctx):
    if name not in SPIKE_OBJECTIVES:
        raise ValueError(f"Unknown --spike_objective {name!r}; "
                         f"available: {sorted(SPIKE_OBJECTIVES)}")
    return SPIKE_OBJECTIVES[name](**ctx)


@torch.no_grad()
def reinit_dead_units(model, dead, generator=None, verbose=True):
    """
    Re-roll the parameters feeding and consuming a dead TIMING unit.

    `dead` : list of timing-unit indices to rescue.

    Three things get re-rolled per unit, and the third is the one that
    matters most:
      - w_in_gait  : that unit's column of every gait's routing matrix, with
                     positive mean restored, so it gets net positive drive
      - b_t        : that unit's per-gait bias, back to zero
      - w1[l]      : THE SUB-NETWORK'S INPUT WEIGHTS.  d(cur1)/d(w1) is
                     proportional to the timing spike, so w1[l] received
                     exactly zero gradient for the whole dead period and is
                     still at its original random init while the rest of
                     sub-network l trained around it.  Re-rolling it costs
                     nothing (it learned nothing) and starting it fresh
                     alongside a freshly-driven input is strictly better
                     than leaving stale init.

    Deliberately does NOT touch w2/w_read/w_out for that group: those DID
    train (on the constant b1-driven activity) and may hold something useful
    about the group's output range.
    """
    if not dead or not hasattr(model, "w_in_gait"):
        return
    dev = model.w_in_gait.weight.device
    MG, nn_, G = model.max_gaits, model.n_neurons, model.G
    W = model.w_in_gait.weight.view(MG, nn_, G)
    for l in dead:
        W[:, :, l] = (torch.randn(MG, nn_, generator=generator).to(dev)
                      * _W_IN_INIT + _W_IN_INIT)
        model.b_t.weight[:, l] = 0.0
        model.w1[l] = (torch.randn(model.Hg, generator=generator).to(dev)
                       * _W1_INIT)
    if verbose:
        print(f"      [reinit] timing unit(s) {dead}: re-rolled w_in_gait "
              f"column (all gaits), zeroed its per-gait bias, and re-rolled "
              f"sub-net w1 (which had received zero gradient while dead)")


# ═══════════════════════════════════════════════════════════════════
# 7b.  Timing-layer diagnostics
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def timing_report(model, spikes, phase, period, n_gaits, device,
                  t0, n_steps=1500, gait_names=None, indent="    "):
    """
    Per-gait firing statistics for the timing layer.  Returns a list of
    formatted lines (also returned as raw dicts) so the caller can print
    them every N epochs.

    Three numbers per timing neuron:

      spk/cyc  Spikes per CPG cycle.  0.00 means the unit is DEAD -- its
               sub-network then receives a constant-zero input and its
               joints are frozen at whatever the decaying membranes settle
               to, and the surrogate gradient through a never-crossing
               membrane is weak, so it tends not to recover on its own.
               This is a much sharper failure than a dead unit in a
               256-wide dense layer, which is why it is checked every few
               epochs rather than at the end.
               Very large values (>> spikes/burst of the CPG) mean the unit
               is saturated and carries no timing information either.

      phase    Circular MEAN of the cycle phase at which the unit fires,
               in [0,1).  This is the direct successor to the old routing
               residual measurement: it says which part of the cycle each
               sub-network is being told about.

      R        Circular concentration in [0,1].  R near 1 means the unit is
               phase-locked (fires in a tight burst at a consistent point
               in the cycle) -- what a timing neuron is for.  R near 0
               means its spikes are smeared around the cycle, so `phase` is
               meaningless and the unit is not providing a phase reference
               regardless of its rate.
    """
    if not hasattr(model, "timing_only"):
        return [], []

    n_steps = int(min(n_steps, len(spikes) - t0))
    if n_steps <= 0:
        return [], []

    x  = torch.as_tensor(spikes[t0:t0 + n_steps], dtype=torch.float32,
                         device=device).unsqueeze(1)          # (L, 1, Nn)
    ph = np.asarray(phase[t0:t0 + n_steps], dtype=np.float64)
    ok = ~np.isnan(ph)
    n_cycles = max(n_steps / float(period), 1e-9)

    lines, stats = [], []
    for g in range(n_gaits):
        gg  = torch.full((n_steps, 1), g, dtype=torch.long, device=device)
        spk = model.timing_only(x, gg)[:, 0].cpu().numpy()

        rate, mu, R = [], [], []
        for j in range(spk.shape[1]):
            m = (spk[:, j] > 0.5) & ok
            rate.append(spk[:, j].sum() / n_cycles)
            if m.sum() == 0:
                mu.append(np.nan); R.append(0.0)
                continue
            z = np.mean(np.exp(1j * 2.0 * np.pi * ph[m]))
            mu.append((np.angle(z) / (2.0 * np.pi)) % 1.0)
            R.append(abs(z))

        name = (gait_names[g] if gait_names is not None else f"g{g}")
        fmt  = lambda v, w=5, p=2: " ".join(
            ("  nan" if not np.isfinite(x_) else f"{x_:{w}.{p}f}") for x_ in v)
        lines.append(f"{indent}timing {name:>5s} : "
                     f"spk/cyc [{fmt(rate)}]  "
                     f"phase [{fmt(mu)}]  R [{fmt(R)}]")
        dead = [j for j, r in enumerate(rate) if r < 1e-6]
        if dead:
            lines.append(f"{indent}       {'':>5s}   "
                         f"WARNING: timing neuron(s) {dead} DEAD for "
                         f"{name} — sub-network(s) {dead} get no input.")
        stats.append({"gait": name, "rate": [float(v) for v in rate],
                      "phase": [None if not np.isfinite(v) else float(v)
                                for v in mu],
                      "R": [float(v) for v in R]})

    # THE headline diagnostic for gait separation. If the PHASE VECTORS are
    # near-identical across gaits, the timing layer has collapsed to a
    # gait-independent clock and the sub-networks are carrying all the gait
    # knowledge -- which is exactly what killed the shared-router version.
    # Rate and R being identical across gaits is EXPECTED: the
    # spike-statistics penalty pushes every (gait, unit) pair to the CPG's
    # values, so phase is the only statistic here left free to differentiate.
    if len(stats) > 1:
        ph = np.array([[np.nan if v is None else v for v in st["phase"]]
                       for st in stats], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            z = np.exp(1j * 2.0 * np.pi * ph)
            spread = np.nanmax(np.abs(z - np.nanmean(z, axis=0, keepdims=True)))
        if np.isfinite(spread):
            lines.append(f"{indent}phase separation across gaits: "
                         f"{spread:.3f}  (max deviation on the unit circle; "
                         f"below ~0.1 means the gaits are NOT being "
                         f"distinguished)")
    return lines, stats


# ═══════════════════════════════════════════════════════════════════
# 8.  Training
# ═══════════════════════════════════════════════════════════════════

def make_gait_weights(tables_orig, names, device):
    """Upweight gaits whose original table was coarsest (bk: 22 rows).

    `names` is for the print only, but is a required (not defaulted)
    parameter: the caller resolved a specific gait set for this run, and a
    silent fallback here could print the wrong species' labels next to the
    right numbers.
    """
    R = max(t.shape[0] for t in tables_orig)
    w = torch.tensor([R / t.shape[0] for t in tables_orig],
                     dtype=torch.float32, device=device)
    print("      gait loss weights: " +
          "  ".join(f"{names[i]}={w[i].item():.2f}" for i in range(len(w))))
    return w


def masked_loss(pred, targ, mask, gait, gait_w=None):
    """MSE over (L,B,8), masked by (L,B), optionally weighted per gait."""
    err = ((pred - targ) ** 2).mean(dim=2)          # (L,B)
    if gait_w is not None:
        err = err * gait_w[gait]
    denom = mask.sum().clamp(min=1.0)
    return (err * mask).sum() / denom


def detach_state(state):
    return tuple(s.detach() for s in state)


def apply_reset(state, reset_mask):
    """
    Zero the state of any stream that was rewound.

    `reset_mask` is (B,); each state tensor is (B, ...). The mask is
    reshaped to (B, 1, 1, ...) matching that tensor's rank rather than a
    hardcoded rank, because getting this wrong broadcasts instead of
    failing: with 2-D state (B,H) a (B,1,1) mask silently produces
    (B,B,H). That is how the leg-grouping removal first broke -- the old
    view(-1,1,1) was written for the (B,G,Hg) state.

    TimingGroupedSNN makes this load-bearing again: its state MIXES ranks
    -- mem_timing is (B, n_timing) while mem1/mem2/memo are (B, G, Hg) --
    so a single hardcoded rank cannot be right for all four tensors.
    """
    if reset_mask.sum() == 0:
        return state
    keep = 1.0 - reset_mask
    return tuple(s * keep.view(-1, *([1] * (s.dim() - 1))) for s in state)


def grad_blocks(model):
    """
    Group parameters into coarse blocks for per-block gradient reporting.

    Derived from parameter NAMES rather than hardcoded lists, so it works for
    both architectures without either one having to know about the other.
    Order of the checks matters: "w_in_gait" also startswith "w_in", so the
    timing block is tested before the dense arch's input block.

    Why per-block at all: the single |grad| scalar hides which parts of the
    model are actually receiving signal.  A near-zero norm on the timing block
    specifically would mean the timing layer is not learning, which is a
    completely different problem from a uniformly small gradient (and under
    Adam a uniformly small gradient is not a problem at all -- the update is
    scale-invariant in the gradient).
    """
    rules = (
        ("timing",    ("w_in_gait", "b_t", "beta_t_logit")),
        ("input",     ("w_in",)),                    # dense arch only
        ("sub_l1",    ("w1", "b1", "beta1_logit")),
        ("sub_l2",    ("w2", "b2", "beta2_logit")),
        ("readout",   ("w_read", "b_read", "w_out", "b_out", "betao_logit")),
        ("sub_film",  ("film1", "film2")),
        ("layernorm", ("ln1", "ln2")),
    )
    blocks = {}
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        block = "other"
        for bname, prefixes in rules:
            if pname.startswith(prefixes):
                block = bname
                break
        blocks.setdefault(block, []).append(p)
    return blocks


class MetricsWriter:
    """
    Append one row per epoch to out_dir/metrics.csv.

    Appended and flushed every epoch rather than written at the end, because
    Ctrl+C is a normal way to finish a run here -- the config's `history` only
    lands on a clean(ish) exit, so a partially-interrupted run would otherwise
    leave nothing comparable behind.  Columns are fixed from the first row, so
    a key appearing later is dropped rather than shifting the table.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.cols = None
        self.fh = None

    def write(self, row):
        if self.fh is None:
            self.cols = list(row.keys())
            self.fh = open(self.path, "w")
            self.fh.write(",".join(self.cols) + "\n")
        vals = []
        for c in self.cols:
            v = row.get(c, "")
            vals.append("" if v is None else
                        (f"{v:.6g}" if isinstance(v, float) else str(v)))
        self.fh.write(",".join(vals) + "\n")
        self.fh.flush()

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def run_training(model, tr_sampler, va_sampler, opt, sched, device, args,
                 gait_w, out_dir, timing_diag=None, n_gaits=4, period=254.0,
                 spike_obj=None):
    """
    `timing_diag` : optional zero-arg callable returning (lines, stats).
                    Called every args.timing_log_every epochs for the
                    timing-grouped arch; None for the dense arch.  Its last
                    return value is handed back so the config can record it.
    `n_gaits`,
    `period`      : reporting and diagnostics.
    `spike_obj`   : a SpikeObjective (strategy). Its `penalty` is added to the
                    task loss; NoSpikeObjective disables it. Swapping the
                    objective needs no change here.
    """
    best = float("inf")
    best_path = out_dir / "best_model.pt"
    hist = {"train": [], "val": [], "val_sw": [], "gnorm": [], "sec": [],
            "floor": [], "upd": []}
    last_timing_stats = []

    # Per-block gradient norms and the per-epoch parameter update norm.
    #
    # |upd| is the metric |grad| cannot be: under Adam the update is
    # lr*m/(sqrt(v)+eps), which is scale-invariant in the gradient, so a small
    # |grad| says nothing about whether the model is moving. |upd| measures the
    # movement directly. Expected scale is ~lr*sqrt(N) per gradient step, so
    # roughly lr*sqrt(N)*sqrt(chunks) per epoch if steps are uncorrelated.
    blocks = grad_blocks(model)
    for b in blocks:
        hist[f"g_{b}"] = []
        hist[f"u_{b}"] = []
    n_par = sum(p.numel() for p in model.parameters())
    exp_upd = args.lr * math.sqrt(n_par) * math.sqrt(args.chunks_per_epoch)
    metrics = MetricsWriter(out_dir / "metrics.csv")
    print(f"\n  Per-epoch metrics -> {out_dir / 'metrics.csv'}")
    print(f"  Gradient blocks: {', '.join(sorted(blocks))}")
    print(f"  |upd| = per-epoch ||delta theta||. Order-of-magnitude "
          f"expectation at lr={args.lr:g} is ~{exp_upd:.2f}")
    print(f"  (Adam's step is scale-invariant in the gradient, so |upd| -- not "
          f"|grad| -- is what says whether the model is moving.)")
    print(f"  u/g per block is the diagnostic: Adam normalises PER PARAMETER, so")
    print(f"  a block moves at ~lr per weight no matter how small its gradient.")
    print(f"  u/g >> 1 means a block is moving far more than its share of the")
    print(f"  signal warrants -- i.e. random-walking on noise, which perturbs")
    print(f"  the representation the downstream layers are trying to use.")

    # Spike statistics are only meaningful for the arch with a timing layer.
    use_stats = (spike_obj is not None and spike_obj.enabled
                 and hasattr(model, "n_timing"))
    if use_stats:
        print(f"\n  Spike objective: {spike_obj.describe()}")
        if args.spike_lambda_warmup > 0:
            print(f"  lambda ramps linearly from 0 over the first "
                  f"{args.spike_lambda_warmup} epoch(s), so the network learns "
                  f"to USE spikes before it is charged for them.")
        print(f"  Judge it on free-run RMSE, not on the spike count.")

    # Consecutive epochs each timing unit has been observed dead, for
    # reinit_dead_units.  Only advanced on epochs where the diagnostic runs.
    dead_streak = {}

    print(f"\n  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  "
          f"{'Val(post-sw)':>13}  {'LR':>9}  {'|grad|':>8}  {'|upd|':>8}"
          f"  {'sec':>6}")
    print("  " + "-" * 88)
    print(f"  (Ctrl+C stops training and exports.  |grad| is the PRE-clip "
          f"norm, clip={args.clip:g};")
    print(f"   routinely much larger than clip means most updates are being "
          f"truncated and the LR is too hot.)")

    try:
        for epoch in range(1, args.epochs + 1):
            # Measures train + validate, i.e. the cost that recurs every
            # epoch.  Read before the timing report below, so an occasional
            # diagnostic epoch does not show up as a spike in this column.
            t_epoch = time.perf_counter()
            # Per-block snapshot (one full copy of the params, same cost as
            # the old single flat concat) so the update norm can be attributed
            # per block rather than only reported in aggregate.
            theta0 = {b: [p.detach().clone() for p in plist]
                      for b, plist in blocks.items()}
            bacc = {b: torch.zeros((), device=device) for b in blocks}

            # ---- train -------------------------------------------------
            model.train()
            state = model.init_state(args.batch, device)
            # Linear lambda warm-up.  See MinCountSpikeObjective's
            # docstring: an L1 spike cost applied before the task loss has
            # learned to need the spikes can prune a unit into unrecoverable
            # silence, because a gated sub-network with no input passes zero
            # gradient to everything except b_out.
            lam_scale = (min(1.0, epoch / float(args.spike_lambda_warmup))
                         if args.spike_lambda_warmup > 0 else 1.0)
            tot, gtot, nb, ftot = 0.0, 0.0, 0, 0.0
            for _ in range(args.chunks_per_epoch):
                (x, g, y, m, sw, rst,
                 ph, warm) = tr_sampler.next_chunk(args.bptt)
                x, g, y, m = (x.to(device), g.to(device),
                              y.to(device), m.to(device))
                ph, warm = ph.to(device), warm.to(device)
                state = apply_reset(detach_state(state), rst.to(device))

                # `warm` zeroes steps whose head had its state wiped less than
                # one cycle ago: the forward pass still runs (that is how state
                # builds) but those outputs are not trained against, because a
                # network with zero state cannot know where in the cycle it is.
                m_eff = m * warm

                if use_stats:
                    pred, state, aux = model(x, g, state, return_aux=True)
                else:
                    pred, state = model(x, g, state)
                    aux = None
                loss = masked_loss(pred, y, m_eff, g, gait_w)

                if aux is not None:
                    # aux = (timing spikes,) each (L, B, U).  Same warm mask as
                    # the task loss: a unit whose state was just zeroed
                    # legitimately fires less and should not be charged for it.
                    pen = lam_scale * sum(
                        spike_obj.penalty(a, g, ph, m_eff) for a in aux)
                    loss = loss + pen
                    ftot += float(pen.detach())

                opt.zero_grad()
                loss.backward()
                # Returns the total norm BEFORE clipping — free to read,
                # and the only way to tell whether clip=1.0 is quietly
                # truncating nearly every update (i.e. masking a too-hot LR).
                # Before clip_grad_norm_, which rescales grads IN PLACE --
                # taken here so these are comparable to the pre-clip |grad|.
                # Accumulated as device tensors and synced once per epoch
                # rather than per chunk.
                with torch.no_grad():
                    for b, plist in blocks.items():
                        gs = [p.grad.detach() for p in plist
                              if p.grad is not None]
                        if gs:
                            bacc[b] += torch.linalg.vector_norm(
                                torch.stack([torch.linalg.vector_norm(g)
                                             for g in gs]))
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                # Stepped per GRADIENT STEP, not per epoch: the LR schedule
                # must not depend on chunks_per_epoch, which is only a
                # logging/validation boundary. T_max is set to
                # epochs * chunks_per_epoch to match this call count.
                sched.step()
                tot += loss.item(); gtot += float(gnorm); nb += 1
            tr_loss  = tot / max(nb, 1)
            tr_gnorm = gtot / max(nb, 1)
            tr_floor = ftot / max(nb, 1)

            # ---- validate ----------------------------------------------
            model.eval()
            vstate = model.init_state(args.batch, device)
            vtot, vsw_tot, vn, vsw_n = 0.0, 0.0, 0, 0
            with torch.no_grad():
                for _ in range(args.val_chunks):
                    (x, g, y, m, sw, rst,
                     ph, warm) = va_sampler.next_chunk(args.bptt)
                    x, g, y, m = (x.to(device), g.to(device),
                                  y.to(device), m.to(device))
                    sw, warm = sw.to(device), warm.to(device)
                    vstate = apply_reset(vstate, rst.to(device))
                    pred, vstate = model(x, g, vstate)
                    m = m * warm            # same warm-up exclusion as train

                    vtot += masked_loss(pred, y, m, g).item(); vn += 1

                    # "post-switch" window: args.settle steps after each switch
                    post = torch.zeros_like(sw)
                    idx = sw.nonzero(as_tuple=False)
                    for t_i, b_i in idx:
                        post[t_i:min(t_i + args.settle, sw.shape[0]), b_i] = 1.0
                    if post.sum() > 0:
                        vsw_tot += masked_loss(pred, y, m * post, g).item()
                        vsw_n   += 1

            va_loss = vtot / max(vn, 1)
            vsw     = vsw_tot / max(vsw_n, 1) if vsw_n else float("nan")
            epoch_s = time.perf_counter() - t_epoch

            # Appended together, so an interrupt can never leave these
            # lists at different lengths.
            hist["train"].append(tr_loss)
            hist["val"].append(va_loss)
            hist["val_sw"].append(vsw)
            hist["gnorm"].append(tr_gnorm)
            hist["sec"].append(epoch_s)
            hist["floor"].append(tr_floor)

            with torch.no_grad():
                ublk = {}
                for b, plist in blocks.items():
                    ublk[b] = float(torch.linalg.vector_norm(torch.stack([
                        torch.linalg.vector_norm(p.detach() - p0)
                        for p, p0 in zip(plist, theta0[b])])))
            # grad_blocks partitions every trainable parameter into exactly one
            # block, so the Euclidean total over blocks IS the full update norm
            # -- no separate whole-model pass needed.
            upd = math.sqrt(sum(v * v for v in ublk.values()))
            hist["upd"].append(upd)
            gblk = {b: float(bacc[b]) / max(nb, 1) for b in blocks}
            for b in blocks:
                hist[f"g_{b}"].append(gblk[b])
                hist[f"u_{b}"].append(ublk[b])

            flag = ""
            if va_loss < best:
                best = va_loss
                torch.save(model.state_dict(), best_path)
                flag = " *"

            if epoch % args.log_every == 0 or epoch == 1:
                print(f"  {epoch:>6}  {tr_loss:>10.6f}  {va_loss:>10.6f}  "
                      f"{vsw:>13.6f}  {opt.param_groups[0]['lr']:>9.2e}"
                      f"  {tr_gnorm:>8.3f}  {upd:>8.3f}  {epoch_s:>6.1f}{flag}")

            metrics.write({
                "epoch": epoch, "train": tr_loss, "val": va_loss,
                "val_post_switch": vsw, "lr": opt.param_groups[0]["lr"],
                "grad_norm": tr_gnorm, "update_norm": upd,
                "spike_penalty": tr_floor, "sec": epoch_s,
                "best": int(flag.strip() == "*"),
                **{f"grad_{b}": gblk[b] for b in sorted(blocks)},
                **{f"upd_{b}": ublk[b] for b in sorted(blocks)},
            })

            # Per-block breakdown, on the diagnostic cadence so the main
            # table stays scannable.  Read u/g: a block with 1% of the gradient
            # and 60% of the movement is diffusing, not learning.
            if epoch % args.timing_log_every == 0 or epoch == 1:
                gtot = sum(gblk.values()) or 1.0
                utot = sum(ublk.values()) or 1.0
                print(f"      {'block':<10}{'|grad|':>11}{'g%':>7}"
                      f"{'|upd|':>10}{'u%':>7}{'u/g':>8}{'upd/param':>11}")
                for b in sorted(blocks, key=lambda k: -ublk[k]):
                    gs, us = 100*gblk[b]/gtot, 100*ublk[b]/utot
                    npar = sum(p.numel() for p in blocks[b])
                    print(f"      {b:<10}{gblk[b]:>11.3g}{gs:>6.1f}%"
                          f"{ublk[b]:>10.4g}{us:>6.1f}%"
                          f"{(us/gs if gs > 1e-9 else float('inf')):>8.1f}"
                          f"{ublk[b]/math.sqrt(max(npar,1)):>11.2e}")

            # Timing layer: cheap (timing units only, batch 1) but it prints
            # n_gaits lines, so it runs on its own slower cadence.
            if timing_diag is not None and (
                    epoch % args.timing_log_every == 0 or epoch == 1):
                lines, last_timing_stats = timing_diag()
                for ln in lines:
                    print(ln)

                # A unit counts as dead only if it is silent for EVERY gait:
                # dead-for-one-gait is the rate floor's job, and re-rolling a
                # unit that works for 15 of 16 gaits would throw away more
                # than it fixes.
                if args.reinit_dead_after > 0 and last_timing_stats:
                    n_u  = len(last_timing_stats[0]["rate"])
                    dead = [u for u in range(n_u)
                            if all(st["rate"][u] < 1e-6
                                   for st in last_timing_stats)]
                    for u in range(n_u):
                        dead_streak[u] = (dead_streak.get(u, 0) + 1
                                          if u in dead else 0)
                    due = [u for u in dead
                           if dead_streak[u] >= args.reinit_dead_after]
                    if due:
                        reinit_dead_units(model, due)
                        for u in due:
                            dead_streak[u] = 0

    except KeyboardInterrupt:
        # Return normally rather than propagating: main() then falls through
        # to plots + config + ONNX export using the best checkpoint so far,
        # so an aborted run still produces deployable artifacts.
        done = len(hist["train"])
        print()
        print("  " + "-" * 88)
        print(f"  [INTERRUPT] Ctrl+C received during epoch {done + 1}.")
        print(f"              {done} epoch(s) completed and recorded; the "
              f"partial epoch is discarded.")
        print(f"              Best val MSE so far : {best:.6f}")
        print( "              Stopping training and proceeding to export.")

    metrics.close()
    print("  " + "-" * 88)
    if hist["sec"]:
        tot = sum(hist["sec"])
        print(f"  {len(hist['sec'])} epoch(s) in {tot:.1f}s  "
              f"(mean {tot / len(hist['sec']):.2f}s/epoch, train+val only)")
    return best, hist, last_timing_stats


# ═══════════════════════════════════════════════════════════════════
# 9.  Plots
# ═══════════════════════════════════════════════════════════════════

def plot_cpg_raster(spikes, onsets, out_dir, n_show=1200):
    N = spikes.shape[1]
    colors = CPG_PALETTE
    fig, ax = plt.subplots(figsize=(15, 3.6))
    for i in range(N):
        t = np.where(spikes[:n_show, i] > 0)[0]
        ax.scatter(t, np.full_like(t, i), marker="|", s=130, lw=1.6,
                   color=colors[i % len(colors)], label=f"N{i}")
    for b in onsets[0][onsets[0] < n_show]:
        ax.axvline(b, color="k", lw=0.9, alpha=0.45, ls="--")
    ax.set_yticks(range(N)); ax.set_yticklabels([f"CPG {i}" for i in range(N)])
    ax.set_xlabel("timestep"); ax.legend(fontsize=8, ncol=N, loc="upper right")
    ax.set_title("Bursting-LIF CPG raster (dashed = neuron-0 burst onset)")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    p = out_dir / "cpg_raster.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"    [saved] {p}")



def plot_training_curves(hist, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    e = range(1, len(hist["train"]) + 1)
    ax.plot(e, hist["train"],  lw=2, color="#457b9d", label="train")
    ax.plot(e, hist["val"],    lw=2, color="#2a9d8f", ls="--", label="val")
    ax.plot(e, hist["val_sw"], lw=2, color="#e63946", ls=":",
            label="val (post-switch)")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked MSE"); ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Stateful TBPTT training")
    plt.tight_layout()
    p = out_dir / "training_curves.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"    [saved] {p}")


@torch.no_grad()
def plot_reconstruction(model, spikes, targets, valid, device,
                        out_dir, tgt_range, t0, gait_names, leg_cols, n_joints,
                        n_steps=1200, warm=600):
    """
    Free-run the network on held-out steps, one plot per gait.

    `gait_names`, `leg_cols`, `n_joints` are required, explicit parameters —
    this used to read GAIT_NAMES / LEG_COLS / N_JOINTS off the module, which
    broke the moment a run could be quadruped OR hexapod: the module
    constants are the quadruped layout only (see default_leg_layout), and a
    hexapod run silently plotted under the wrong legend would be worse than
    an import error.

    `leg_cols` may have any number of equal-size groups (not just 4 groups
    of 2) -- the subplot grid below sizes itself from len(leg_cols) and the
    per-group column count, the same squeeze=False + hide-unused pattern
    plotting_utils.py already uses elsewhere in this repo.
    """
    lo, hi = tgt_range
    scale, shift = (hi - lo) / 2.0, (hi + lo) / 2.0
    model.eval()
    rmse = np.zeros((len(gait_names), n_joints))

    n_legs   = len(leg_cols)
    cols_per = len(leg_cols[0])

    for g in range(len(gait_names)):
        x = torch.tensor(spikes[t0 - warm:t0 + n_steps]).unsqueeze(1).to(device)
        gg = torch.full((x.shape[0], 1), g, dtype=torch.long, device=device)
        pred, _ = model(x, gg)
        pred = pred[warm:, 0].cpu().numpy() * scale + shift
        true = targets[g, t0:t0 + n_steps] * scale + shift
        v    = valid[t0:t0 + n_steps]

        fig, axes = plt.subplots(n_legs, cols_per, figsize=(5 * cols_per, 2.5 * n_legs),
                                 sharex=True, squeeze=False)
        for l in range(n_legs):
            for k, col in enumerate(leg_cols[l]):
                ax = axes[l][k]
                r = float(np.sqrt(np.mean((pred[v, col] - true[v, col]) ** 2)))
                rmse[g, col] = r
                ax.plot(true[:, col], color="#457b9d", lw=1.8, label="GT")
                ax.plot(pred[:, col], color="#e63946", lw=1.4, ls="--",
                        label="pred")
                ax.set_title(f"leg{l}  col{col}   RMSE={r:.2f}°", fontsize=9)
                ax.grid(alpha=0.25); ax.legend(fontsize=7)
        for l in range(n_legs):
            axes[l][0].set_ylabel(f"leg{l}", fontsize=8)
        for k in range(cols_per):
            axes[-1][k].set_xlabel("timestep")
        plt.suptitle(f"{gait_names[g]} — free-run reconstruction "
                     f"({warm}-step warm-up discarded)", fontweight="bold")
        plt.tight_layout()
        p = out_dir / f"recon_{gait_names[g]}.png"
        plt.savefig(p, dpi=140); plt.close()
        print(f"    [saved] {p}  mean RMSE = {rmse[g].mean():.2f}°")

    fig, ax = plt.subplots(figsize=(max(6, n_joints * 0.7), n_legs * 0.9 + 2.0))
    im = ax.imshow(rmse, aspect="auto", cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, label="RMSE (deg)")
    col_to_leg = {c: l for l, grp in enumerate(leg_cols) for c in grp}
    ax.set_xticks(range(n_joints))
    ax.set_xticklabels([f"c{j}\nleg{col_to_leg.get(j, '?')}"
                        for j in range(n_joints)], fontsize=8)
    ax.set_yticks(range(len(gait_names))); ax.set_yticklabels(gait_names)
    for g in range(len(gait_names)):
        for j in range(n_joints):
            ax.text(j, g, f"{rmse[g, j]:.1f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if rmse[g, j] > rmse.max() * 0.6 else "black")
    ax.set_title("Per-joint RMSE (degrees)", fontweight="bold")
    plt.tight_layout()
    p = out_dir / "rmse_heatmap.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"    [saved] {p}")
    return rmse


@torch.no_grad()
def plot_transition(model, spikes, targets, device, out_dir, tgt_range,
                    t0, gait_names, leg_cols, g_from=0, g_to=1, warm=600,
                    n_steps=1400, switch_at=600):
    """See plot_reconstruction's docstring for why gait_names/leg_cols are
    required parameters rather than module globals."""
    lo, hi = tgt_range
    scale, shift = (hi - lo) / 2.0, (hi + lo) / 2.0
    model.eval()
    L = warm + n_steps
    x = torch.tensor(spikes[t0 - warm:t0 + n_steps]).unsqueeze(1).to(device)
    gg = torch.full((L, 1), g_from, dtype=torch.long, device=device)
    gg[warm + switch_at:] = g_to
    pred, _ = model(x, gg)
    pred = pred[warm:, 0].cpu().numpy() * scale + shift

    true = np.where(
        (np.arange(n_steps) < switch_at)[:, None],
        targets[g_from, t0:t0 + n_steps],
        targets[g_to,   t0:t0 + n_steps]) * scale + shift

    n_legs = len(leg_cols)
    fig, axes = plt.subplots(n_legs, 1, figsize=(14, 2.2 * n_legs), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    for l in range(n_legs):
        col = leg_cols[l][0]
        axes[l].plot(true[:, col], color="#457b9d", lw=1.8, label="GT")
        axes[l].plot(pred[:, col], color="#e63946", lw=1.4, ls="--", label="pred")
        axes[l].axvline(switch_at, color="k", lw=1.5, ls="-.")
        axes[l].set_ylabel(f"leg{l} c{col} (deg)"); axes[l].grid(alpha=0.25)
        axes[l].legend(fontsize=7)
    axes[-1].set_xlabel("timestep")
    plt.suptitle(f"Gait switch {gait_names[g_from]} -> {gait_names[g_to]} "
                 f"at t={switch_at}", fontweight="bold")
    plt.tight_layout()
    p = out_dir / "transition.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"    [saved] {p}")


# ═══════════════════════════════════════════════════════════════════
# 10.  ONNX export
# ═══════════════════════════════════════════════════════════════════

def export_onnx(model, out_dir, device, cfg):
    model.eval()

    # If main() compiled model.step, swap the eager version back in for the
    # duration of the export: torch.onnx.export does not trace reliably
    # through a torch.compile'd callable, and SingleStepONNX calls
    # self.model.step().  Restored in the finally block so the caller's
    # model is left exactly as it was found.
    compiled_step = None
    if hasattr(model, "_step_eager") and model.step is not model._step_eager:
        compiled_step = model.step
        model.step = model._step_eager
        print("    [onnx] using eager step() for export")

    try:
        # Wrapper, spike width and state shapes all come from the model, so
        # neither the CPG size nor the state layout is written twice.
        wrap_cls = (SingleStepONNXTiming if model.arch == "timing_grouped"
                    else SingleStepONNX)
        wrapper  = wrap_cls(model).to(device).eval()

        dummy = (torch.zeros(1, model.n_neurons, device=device),
                 torch.zeros(1, dtype=torch.long, device=device),
                 *model.init_state(1, device))

        in_names  = ["spikes", "gait"] + list(model.state_names_in)
        out_names = ["angles"]         + list(model.state_names_out)

        path = out_dir / "cpg_lif_snn_step.onnx"
        torch.onnx.export(
            wrapper, dummy, str(path),
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=in_names, output_names=out_names,
            dynamic_axes={n: {0: "batch"} for n in in_names + out_names})
        print(f"    [saved] ONNX -> {path}")

        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(path),
                                        providers=["CPUExecutionProvider"])
            feed = {n: d.cpu().numpy() for n, d in zip(in_names, dummy)}
            ort_out = sess.run(None, feed)
            pt_out  = [t.detach().cpu().numpy() for t in wrapper(*dummy)]
            diff = max(float(np.abs(a - b).max())
                       for a, b in zip(pt_out, ort_out))
            print(f"    PyTorch vs ONNX max diff : {diff:.2e} "
                  f"({'OK' if diff < 1e-4 else 'WARNING'})")
        except ImportError:
            print("    onnxruntime not installed — skipping parity check.")
    finally:
        if compiled_step is not None:
            model.step = compiled_step

    cfg_path = out_dir / "cpg_lif_snn_config.json"
    with open(cfg_path, "w") as f:
        json.dump(json_safe(cfg), f, indent=2, default=str)
    print(f"    [saved] config -> {cfg_path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 11.  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Bursting-LIF CPG -> leg-grouped stateful SNN")

    # CPG
    ap.add_argument("--tmax",   type=int,   default=50_000,
                    help="Steps of CPG spike train to collect. Lowered from "
                         "150k: the CPG is exactly periodic after warmup, so "
                         "150k was ~590 duplicate copies of the same ~254-step "
                         "cycle. Phase alignment and switch timing are "
                         "randomised by StreamSampler independently of this.")
    ap.add_argument("--warmup", type=int,   default=2_000)
    ap.add_argument("--i_app",  type=float, default=8.0)
    ap.add_argument("--n_cpg_neurons", type=int, default=4,
                    choices=sorted(CPG_W_BY_N),
                    help="CPG size. Selects the coupling matrix from "
                         "CPG_W_BY_N and sets the SNN's input width; nothing "
                         "downstream assumes 4.")

    # gait tables
    ap.add_argument("--gaits_dir", type=str, default="../gaits",
                    help="Folder of {name}.csv gait tables, resolved as "
                         "this_file_dir/<gaits_dir> (same join convention as "
                         "--out_dir). Default assumes the folder is a "
                         "sibling of this script's own directory, not a "
                         "child of it — differs from train_snn.py's "
                         "'{this_file_dir}/gaits'.")
    ap.add_argument("--gaits", type=str, nargs="*", default=None,
                    help="CSV file stems (no extension) to load as gaits, "
                         "overriding the --n_cpg_neurons default. Use this "
                         "to run a subset, a different naming, or a species "
                         "GAIT_FILES_BY_N has no entry for. Named to match "
                         "visualize.py's --gaits, though that one filters "
                         "which loaded gaits to plot rather than which "
                         "files to load — same name, different job.")
    ap.add_argument("--leg_cols", type=str, default=None,
                    help="JSON list of equal-size column-index groups, one "
                         "group per leg. Overrides the built-in default for "
                         "--n_cpg_neurons (4->quadruped LEG_COLS, "
                         "6->HEXAPOD_LEG_COLS). Use this for a robot neither "
                         "of those matches.")

    # architecture
    ap.add_argument("--arch", type=str, default="timing_grouped",
                    choices=["dense", "timing_grouped"],
                    help="dense: StatefulSNN, one fully connected network. "
                         "timing_grouped: TimingGroupedSNN — CPG -> small "
                         "timing layer -> n_timing disconnected sub-networks, "
                         "one per timing neuron. Keep 'dense' runnable so A/B "
                         "at matched gradient steps stays possible.")
    ap.add_argument("--n_timing", type=int, default=None,
                    help="[timing_grouped] Number of timing-layer LIF "
                         "neurons, and therefore the number of sub-networks. "
                         "Must equal n_legs or n_joints OF THE RESOLVED GAIT "
                         "SET (printed at startup as 'gait layout'), not a "
                         f"fixed number — e.g. n_legs={N_LEGS}/n_joints="
                         f"{N_JOINTS} for the verified quadruped layout, but "
                         "whatever --leg_cols implies otherwise. Defaults to "
                         "n_legs when the layout is known, else n_joints (see "
                         "--leg_cols). Independent of --n_cpg_neurons: the "
                         "timing layer is densely driven by all CPG spikes, "
                         "so the two counts need not match.")
    ap.add_argument("--readout_hidden", type=int, default=32,
                    help="[timing_grouped] Width of the analog readout "
                         "membrane per group. Was implicitly --hidden, i.e. a "
                         "full (G, Hg, Hg) w_read, which at hidden=128 was 40% "
                         "of the whole model to produce a handful of angles "
                         "per leg. Nothing requires memo to be as wide as "
                         "spk2. NOT set to the output width (3): memo is where "
                         "the temporal filtering happens and each unit has its "
                         "own tau, so its width is the size of the temporal "
                         "BASIS available to synthesise the output waveform. "
                         "Filter-then-combine differs from "
                         "combine-then-filter when the taus differ, so "
                         "projecting to 3 before filtering leaves only 3 "
                         "distinct (projection, tau) pairs for the whole "
                         "group -- one basis function per output, no "
                         "redundancy. Sweep 3/8/32/128 to find where it "
                         "actually binds.")
    ap.add_argument("--tau_readout_max", type=float, default=None,
                    help="[timing_grouped] Tau init ceiling for the analog "
                         "READOUT membrane. Default None = period/(2*pi), the "
                         "corner frequency of a leaky integrator at the gait "
                         "fundamental. Deliberately MUCH shorter than "
                         "--tau_max, and not a shortcut: memo's job is to "
                         "RENDER the current joint angle, which is a local "
                         "operation, whereas mem1/mem2's job is to REMEMBER "
                         "where in the cycle we are. A leaky integrator with "
                         "tau near the period passes only ~14%% of the gait "
                         "fundamental relative to DC -- it measures the cycle "
                         "MEAN and blurs the waveform instead of resolving "
                         "it, so long tau here would waste most of the "
                         "temporal basis. At the corner (tau = period/2*pi) "
                         "passage is 0.71, and the bank spans [2, that], "
                         "giving units from near-perfect passage down to the "
                         "corner. Reproduces the value hardcoded in the "
                         "original StatefulSNN almost exactly at the "
                         "quadruped period (254/2*pi = 40.4 vs 40.0), which "
                         "is why that number worked; it just did not "
                         "generalise to 352, where it should be ~56.")
    ap.add_argument("--timing_reset", type=str, default="zero",
                    choices=["zero", "subtract"],
                    help="[timing_grouped] Membrane reset for the timing "
                         "layer. 'zero' matches the CPG (LIFGeneralArray does "
                         "v[spike]=0). With 'subtract' the residual "
                         "mem-thresh can still exceed threshold, so the unit "
                         "re-fires on the SILENT steps between CPG spikes: "
                         "measured ISIs of 2,1,1,2,1,1,... and 50%% more "
                         "spikes per burst than the CPG has. That was the "
                         "cause of the observed over-firing AND the "
                         "too-tight-burst R values. Kept only for A/B.")
    ap.add_argument("--timing_slope", type=float, default=5.0,
                    help="[timing_grouped] Surrogate-gradient slope for the "
                         "TIMING layer only; --slope still applies to the "
                         "sub-networks. The surrogate "
                         "derivative is 1/(slope*|x|+1)^2, so at slope=25 a "
                         "unit a few units below threshold is nearly "
                         "invisible to gradients. This is the layer where "
                         "a dead unit is catastrophic rather than merely "
                         "wasteful (it cuts off a whole sub-network), so it "
                         "gets a wider (gentler) surrogate by default.")
    ap.add_argument("--tau_timing_min", type=float, default=2.0)
    ap.add_argument("--tau_timing_max", type=float, default=None,
                    help="[timing_grouped] Tau init ceiling for the TIMING "
                         "layer only. Default None = period/n_cpg_neurons, "
                         "i.e. ONE INTER-BURST GAP (59 steps at N=6/P=352, "
                         "63.5 at N=4/P=254). "
                         "Why a gap and not a cycle, which is the intuitive "
                         "answer: the CPG already bursts at every k/n_cpg "
                         "phase, so a timing unit that must fire at phase 0.5 "
                         "does not have to REMEMBER anything from phase 0 -- "
                         "it listens to the CPG neuron that bursts at 0.5. "
                         "The per-gait weight matrix selects which. Memory is "
                         "only needed to bridge BETWEEN adjacent CPG phases, "
                         "which is one gap. "
                         "Long tau is actively harmful here, twice over: the "
                         "input has energy at n_cpg/period (six bursts per "
                         "cycle at N=6), so a slow filter smooths the burst "
                         "structure away -- measured within-cycle modulation "
                         "depth falls from 1.18 at tau=10 to 0.068 at "
                         "tau=352, a 17x loss of phase information -- and as "
                         "tau approaches the period the membrane stops "
                         "decaying between cycles, so the unit accumulates "
                         "and fires continuously instead of once per cycle. "
                         "Taus stay learnable, so this is an init range, not "
                         "a cap; raise it if you want to test the "
                         "hold-across-the-cycle hypothesis directly.")
    ap.add_argument("--calibrate_gains", type=int, default=1,
                    help="[timing_grouped] 1 = before training, bisect each "
                         "timing unit's CPG->timing weight column until its "
                         "firing rate lands within a factor of 2 of the CPG's "
                         "own spikes-per-cycle (measured at startup, so there "
                         "is no band to configure). Forward passes only, a "
                         "second of compute, and it is why there is no "
                         "--timing_w_scale to guess at. 0 = skip.")
    ap.add_argument("--spike_objective", type=str, default="min_count",
                    choices=sorted(SPIKE_OBJECTIVES),
                    help="[timing_grouped] Which objective shapes the timing "
                         "layer's spike train. "
                         "'min_count' (default): pay a flat L1 cost per spike "
                         "and let the TASK loss decide where spikes are worth "
                         "spending — with --event_gated this yields an "
                         "adaptive sampling clock (dense through swing, sparse "
                         "through stance) with no supervision. "
                         "'cpg_match': match the CPG's spikes-per-cycle and "
                         "burst tightness; produces clean CPG-like bursts but "
                         "conflicts with event gating, since it wants every "
                         "spike at one phase. "
                         "'none': unconstrained. "
                         "New strategies: subclass SpikeObjective, set a "
                         "`name`, and it appears here automatically.")
    ap.add_argument("--spike_stats_lambda", type=float, default=0.005,
                    help="[timing_grouped] Weight on the chosen spike "
                         "objective. For 'min_count' the penalty IS the duty "
                         "cycle, so this reads directly as the loss cost of a "
                         "unit spiking on every single timestep. Sizing it: at "
                         "~17%% duty (≈60 spk/cyc at period 352, roughly what "
                         "a 1.5-degree hold error needs) the penalty is "
                         "8.5e-4, comparable to a converged task loss of "
                         "~1e-3 — i.e. deliberately balanced so the task loss "
                         "wins wherever spikes genuinely matter. Raise it if "
                         "the spike rate stays stubbornly high, LOWER it if "
                         "units collapse toward silence. 0 disables.")
    ap.add_argument("--spike_lambda_warmup", type=int, default=10,
                    help="[timing_grouped] Epochs over which the spike "
                         "penalty's lambda ramps linearly from 0. Insurance "
                         "against a specific failure: with --event_gated, a "
                         "silent timing unit starves its sub-network entirely "
                         "(memo decays, y collapses to b_out) and passes ZERO "
                         "gradient to w1/w2/w_read/w_out, so an L1 spike cost "
                         "applied before the task loss needs the spikes can "
                         "prune a unit into a state it cannot recover from. "
                         "0 disables the ramp.")
    ap.add_argument("--reinit_dead_after", type=int, default=2,
                    help="[timing_grouped] Re-roll a timing unit after it has "
                         "been silent for EVERY gait across this many "
                         "consecutive timing reports (so the wall-clock "
                         "trigger scales with --timing_log_every). Re-rolls "
                         "that unit's w_in_gait column, its bias, AND the "
                         "sub-network's w1 — that last one is the point, "
                         "since w1's gradient is proportional to the timing "
                         "spike and so was exactly zero the whole time it was "
                         "dead. 0 disables.")
    ap.add_argument("--event_gated", type=int, default=1,
                    help="[timing_grouped] 1 = a sub-network advances ONLY on "
                         "timesteps where its own timing neuron fired; no "
                         "current is injected (bias included) and no "
                         "sub-network spike is emitted otherwise. Membranes "
                         "still decay every step, so the OUTPUT relaxes toward "
                         "b_out between spikes. "
                         "This is the mechanism that makes spike PLACEMENT "
                         "matter: without it the task loss is EXACTLY "
                         "invariant to the timing layer's burst phase, since "
                         "a sub-network whose taus span a cycle can absorb any "
                         "shift by relearning the waveform offset — which is "
                         "why alignment never emerged from end-to-end "
                         "training. It is also the event-driven form the "
                         "hardware target wants, so alignment and efficiency "
                         "turn out to be the same requirement. 0 restores the "
                         "always-on behaviour.")
    ap.add_argument("--sub_film", type=str, default="both",
                    choices=["none", "l1", "l2", "both"],
                    help="[timing_grouped] Which sub-network layers get "
                         "per-gait FiLM conditioning. Default 'both' = "
                         "unchanged behaviour. Set 'none' to force the "
                         "sub-networks to infer the gait SOLELY from their "
                         "timing neuron's spike train, which is the clean "
                         "test of whether the timing layer is really encoding "
                         "gait. Two warnings: (a) the film1/film2 tables are "
                         "still allocated (19%% of params) but receive no "
                         "gradient, so 'none' wastes them rather than saving "
                         "them; (b) it may be over-constrained together with "
                         "--spike_stats_lambda > 0, which pins rate and burst "
                         "concentration to the CPG's values for every gait "
                         "and so leaves PHASE as the only axis able to carry "
                         "gait — run it with --spike_stats_lambda 0 first.")
    ap.add_argument("--sub_ln", type=str, default="l2",
                    choices=["none", "l1", "l2", "both"],
                    help="[timing_grouped] Which sub-network layers get "
                         "LayerNorm (elementwise_affine=False; FiLM supplies "
                         "the affine). Never applies to the timing layer. "
                         "Default l2 (NOT both): sub-net layer 1's only "
                         "input is one binary channel, so its pre-activation "
                         "has exactly two possible values and LN normalises "
                         "away the amplitude of the sole drive it gets, "
                         "leaving FiLM gamma to put it back. Dropping it also "
                         "saves LN's retained tensors per timestep. Set "
                         "'both' to restore the old behaviour.")

    ap.add_argument("--hidden",     type=int,   default=128,
                    help="Hidden width. For --arch dense this is the TOTAL "
                         "width (dense H x H layers). For --arch "
                         "timing_grouped it is PER GROUP, so w2/w_read are "
                         "(G, H, H) — at G=4, hidden=128 lands near the dense "
                         "hidden=256 parameter count and is the matched-"
                         "parameter baseline; hidden=256 is ~4x that.")
    ap.add_argument("--max_gaits",  type=int,   default=16,
                    help="Rows allocated in the FiLM embedding tables. "
                         "Only the first n_gaits are used. Fixing this keeps "
                         "every parameter shape independent of the gait "
                         "count, so checkpoints transfer between runs with "
                         "different numbers of gaits. Changing it does NOT. "
                         "Every per-gait table is now a FiLM table whose "
                         "unused rows are near-identity, so an added gait "
                         "starts from a sensible routing — an improvement on "
                         "the old per-gait weight matrix, whose unused rows "
                         "were random noise.")
    ap.add_argument("--tau_min",    type=float, default=2.0)
    ap.add_argument("--tau_max",    type=float, default=None,
                    help="Longest SUB-NETWORK membrane time constant, in "
                         "steps. Default None = the measured CPG period "
                         "rounded up to a multiple of 64, which reproduces "
                         "the old fixed 256 at N=4 (period 254) and gives 384 "
                         "at N=6 (period 352). The sub-networks are the layers "
                         "that must hold position across a whole cycle, so "
                         "this has to track the period; the timing "
                         "layer does not.")
    ap.add_argument("--slope",      type=float, default=25.0)

    # training
    ap.add_argument("--epochs",           type=int,   default=100)
    ap.add_argument("--chunks_per_epoch", type=int,   default=40)
    ap.add_argument("--val_chunks",       type=int,   default=2)
    ap.add_argument("--bptt",             type=int,   default=None,
                    help="Gradient truncation horizon. NOT the network's "
                         "receptive field -- state is carried and detached "
                         "across chunks, so the forward pass sees unbounded "
                         "history. Default None = the measured CPG period "
                         "rounded up to a multiple of 64, i.e. ~one full "
                         "cycle: 256 at N=4, 384 at N=6. Sweep 128/256/512 at "
                         "fixed batch*bptt. NOTE batch*bptt is the compute "
                         "budget, so a longer bptt at N=6 costs proportionally "
                         "more unless batch drops.")
    ap.add_argument("--batch",            type=int,   default=128,
                    help="Stream heads per gradient step. Activation memory "
                         "is roughly 12 * batch * bptt * n_timing * hidden * "
                         "4 bytes (TBPTT retains ~12 (B,G,Hg) tensors per "
                         "timestep), so at hexapod n_timing=6 / bptt=384 / "
                         "hidden=128 this is ~1.8 GiB at 128 and ~3.6 GiB at "
                         "256. batch*bptt is the compute budget: if you raise "
                         "bptt with the period, drop batch to match. "
                         "Historical note: at "
                         "these sizes the timestep loop is kernel-launch "
                         "bound, so a bigger batch is nearly free in "
                         "wall-clock. If raising further, consider an LR "
                         "rescale (sqrt rule for Adam).")
    ap.add_argument("--lr",               type=float, default=2e-3,
                    help="Kept at 2e-3 after raising batch 32->128 so the "
                         "first benchmark was a clean comparison. Adam's "
                         "sqrt-scaling rule suggests ~4e-3 at batch 128 — "
                         "worth sweeping, judged on the |grad| column and "
                         "free-run RMSE rather than train loss.")
    ap.add_argument("--clip",             type=float, default=1.0,
                    help="Gradient-norm clip. Watch the |grad| column: if the "
                         "pre-clip norm sits far above this, clipping is "
                         "truncating most updates and stability is NOT "
                         "evidence the LR is well chosen.")
    ap.add_argument("--switch_min",       type=int,   default=600)
    ap.add_argument("--switch_max",       type=int,   default=3000)
    ap.add_argument("--settle",           type=int,   default=100,
                    help="Steps after a gait switch counted as 'post-switch'.")
    ap.add_argument("--val_frac",         type=float, default=0.15)
    ap.add_argument("--phase_zero",       type=float, default=0.0,
                    help="Global rotation of gait-table row 0 relative to "
                         "neuron-0 burst onset, in cycles.")

    # misc
    ap.add_argument("--freeze_blocks", type=str, default="",
                    help="Comma-separated grad_blocks names to freeze "
                         "(requires_grad=False), e.g. "
                         "'sub_l1,sub_l2,sub_film'. Empty = train everything.")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--timing_log_every", type=int, default=10,
                    help="[timing_grouped] Epoch cadence for the timing-layer "
                         "firing report. Slower than --log_every because it "
                         "prints one line per gait; the compute is "
                         "negligible (timing units only, batch 1).")
    ap.add_argument("--dry_run",   action="store_true",
                    help="Build data + diagnostics, skip training.")
    ap.add_argument("--out_dir",   type=str, default="",
                    help="Resolved as outputs/<out_dir> — e.g. --out_dir "
                         "test1 writes to outputs/test1. Default '' means "
                         "outputs/ itself.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = outputs_path(this_file_dir, args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device : {device}\nOutput : {out_dir.resolve()}")
    print(f"Arch   : {args.arch}\n")

    # ── 0. Gait tables + leg layout ───────────────────────────────
    # Resolved before the CPG run and before --n_timing's default, since
    # both depend on n_joints / n_legs, which depend on what got loaded.
    print("[0/6] Gait tables ...")
    gaits_dir = Path(this_file_dir + "/" + args.gaits_dir)
    if args.gaits is not None:
        gait_files = list(args.gaits)
        print(f"      --gaits override: {gait_files}")
    else:
        gait_files = GAIT_FILES_BY_N.get(args.n_cpg_neurons)
        if gait_files is None:
            raise ValueError(
                f"No default gait file list for n_cpg_neurons="
                f"{args.n_cpg_neurons} (have entries for "
                f"{sorted(GAIT_FILES_BY_N)}). Pass --gaits explicitly.")
        species = {4: "quadruped", 6: "hexapod"}.get(args.n_cpg_neurons, "?")
        print(f"      n_cpg_neurons={args.n_cpg_neurons} -> {species} "
              f"gait set: {gait_files}")
    gait_tables_orig, gait_names = load_gait_tables(gait_files, gaits_dir)
    n_joints = gait_tables_orig[0].shape[1]
    for nm, g in zip(gait_names, gait_tables_orig):
        print(f"      {nm:>18s} : {g.shape[0]} rows x {g.shape[1]} "
              f"joints (original)")
    if len(gait_tables_orig) == args.max_gaits:
        print(f"      NOTE: n_gaits ({len(gait_tables_orig)}) == max_gaits "
              f"({args.max_gaits}) — zero headroom to add a gait later "
              f"without invalidating this checkpoint (todo item 1).")

    if args.leg_cols is not None:
        leg_cols = [list(c) for c in json.loads(args.leg_cols)]
        flat = sorted(c for grp in leg_cols for c in grp)
        if flat != list(range(n_joints)):
            raise ValueError(
                f"--leg_cols {leg_cols} is not a partition of "
                f"0..{n_joints - 1} (flattened, sorted: {flat}).")
        if len({len(g) for g in leg_cols}) != 1:
            raise ValueError(f"--leg_cols groups must be equal size, got "
                             f"{leg_cols}.")
        n_legs, layout_src = len(leg_cols), "user"
    else:
        n_legs, leg_cols = default_leg_layout(args.n_cpg_neurons, n_joints)
        layout_src = "default"
    print(f"      gait layout: n_legs={n_legs}  n_joints={n_joints}  "
          f"source={layout_src}  leg_cols={leg_cols}")

    if args.n_timing is None:
        args.n_timing = n_legs
    group_cols = (build_group_cols(args.n_timing, leg_cols=leg_cols, n_joints=n_joints)
                  if args.arch == "timing_grouped" else None)

    # ── 1. CPG ──────────────────────────────────────────────────
    print("\n[1/6] Bursting-LIF CPG ...")
    spikes = run_cpg(N=args.n_cpg_neurons, tmax=args.tmax, warmup=args.warmup,
                     i_app=args.i_app)

    print("\n[2/6] Burst structure & phase ...")
    onsets, period, neuron_offsets, burst_thresholds = analyse_cpg(spikes, out_dir)
    plot_cpg_raster(spikes, onsets, out_dir)

    phase = cycle_phase(len(spikes), onsets[0])

    # ── Period-derived defaults ─────────────────────────────────
    # Resolved here, after the period is MEASURED, rather than as argparse
    # constants: the N=6 CPG runs at ~352 steps against N=4's ~254, and both
    # tau_max (the sub-networks must hold position across a full cycle) and
    # bptt (chosen as "about one cycle") are meaningless as fixed numbers
    # once the period can change. Rounding up to a multiple of 64 reproduces
    # the previous hardcoded 256 at N=4 and gives 384 at N=6.
    round64 = lambda v: int(64 * math.ceil(v / 64.0))
    if args.tau_max is None:
        args.tau_max = float(round64(period))
        print(f"      tau_max        : {args.tau_max:6.1f}  (sub-networks; "
              f"period {period:.0f} rounded up to a multiple of 64 — they "
              f"must hold position across a whole cycle)")
    if args.bptt is None:
        args.bptt = round64(period)
        print(f"      bptt           : {args.bptt:6d}  (~one cycle)")
    if args.tau_timing_max is None:
        # One inter-burst gap.  NOT one cycle: the CPG already bursts at every
        # k/n_cpg phase, so a timing unit needing phase 0.5 listens to the
        # neuron that bursts there rather than remembering from phase 0.
        # Memory is only needed to bridge between ADJACENT CPG phases.
        args.tau_timing_max = float(period) / float(args.n_cpg_neurons)
        print(f"      tau_timing_max : {args.tau_timing_max:6.1f}  (timing "
              f"layer; one inter-burst gap = period/{args.n_cpg_neurons})")
    if args.tau_readout_max is None:
        # Corner frequency of a leaky integrator at the gait fundamental.
        # Much shorter than tau_max on purpose: memo RENDERS the current
        # value (local), it does not REMEMBER the phase (global).  A tau near
        # the period passes ~14% of the fundamental and just measures the
        # cycle mean.
        args.tau_readout_max = float(period) / (2.0 * math.pi)
        print(f"      tau_readout_max: {args.tau_readout_max:6.1f}  (readout "
              f"membrane; period/2pi, the low-pass corner at the gait "
              f"fundamental)")

    # The CPG's own firing statistics become the targets for the router and
    # timing layers -- "fire like the CPG does" needs no hyperparameter.
    cpg_rate, cpg_R = cpg_spike_stats(spikes, phase, period)
    print(f"      CPG stats   : {cpg_rate:.2f} spikes/cycle per neuron, "
          f"phase concentration R={cpg_R:.3f}")

    # ── 3. Upsample ──────────────────────────────────────────────
    print("\n[3/6] Upsampling gait tables ...")
    gait_tables, target_rows = upsample_gait_tables(gait_tables_orig, gait_names)

    targets, valid, tgt_range = build_targets(phase, gait_tables,
                                              phase_zero=args.phase_zero)
    print(f"      targets {targets.shape}   valid coverage "
          f"{valid.mean()*100:.2f}%   range [{tgt_range[0]:.1f}, "
          f"{tgt_range[1]:.1f}] deg")

    # ── 4. Samplers ─────────────────────────────────────────────
    print("\n[4/6] Stream samplers (truncated BPTT, state carried) ...")
    T       = len(spikes)
    t_lo    = int(onsets[0][2])
    t_split = int(T * (1.0 - args.val_frac))
    t_hi    = int(onsets[0][-2])
    print(f"      train steps [{t_lo}, {t_split})   "
          f"val steps [{t_split}, {t_hi})")
    if t_split - t_lo < 4 * args.bptt or t_hi - t_split < 2 * args.bptt:
        raise ValueError("Not enough timesteps — raise --tmax or lower --bptt.")

    # Data lives on the training device; batches are gathered there.
    # warm_steps = one full CPG cycle: a head whose state was just zeroed
    # gets that long to build state before its outputs count toward the loss.
    warm_steps = int(round(period))
    print(f"      post-reset warm-up: {warm_steps} steps "
          f"(one cycle) excluded from the loss")
    tr_sampler = StreamSampler(spikes, targets, valid, t_lo, t_split,
                               args.batch, args.switch_min, args.switch_max,
                               rng, n_gaits=len(gait_tables), device=device,
                               phase=phase, warm_steps=warm_steps)
    va_sampler = StreamSampler(spikes, targets, valid, t_split, t_hi,
                               args.batch, args.switch_min, args.switch_max,
                               np.random.default_rng(args.seed + 1),
                               n_gaits=len(gait_tables), device=device,
                               phase=phase, warm_steps=warm_steps)

    # ── 5. Model + training ─────────────────────────────────────
    print("\n[5/6] Model ...")
    if args.arch == "timing_grouped":
        model = TimingGroupedSNN(
            hidden_per_group=args.hidden, n_gaits=len(gait_tables),
            max_gaits=args.max_gaits, n_neurons=args.n_cpg_neurons,
            n_timing=args.n_timing, group_cols=group_cols,
            n_joints=n_joints, readout_hidden=args.readout_hidden,
            tau_min=args.tau_min, tau_max=args.tau_max,
            tau_timing_min=args.tau_timing_min,
            tau_timing_max=args.tau_timing_max,
            tau_readout_max=args.tau_readout_max,
            sub_ln=args.sub_ln, sub_film=args.sub_film,
            timing_reset=args.timing_reset,
            event_gated=bool(args.event_gated),
            slope=args.slope, timing_slope=args.timing_slope).to(device)
    else:
        model = StatefulSNN(hidden=args.hidden, n_gaits=len(gait_tables),
                            max_gaits=args.max_gaits,
                            n_neurons=args.n_cpg_neurons,
                            tau_min=args.tau_min, tau_max=args.tau_max,
                            slope=args.slope, n_joints=n_joints).to(device)

    # ── Compile the single timestep, NOT forward() ────────────────
    # forward() loops over L (= --bptt, 256-512) timesteps in Python.
    # Compiling forward() would make Dynamo trace-unroll that entire loop
    # into one enormous graph: minutes of compile time, and no reuse.
    # step() is one timestep -- compiled once, then reused L times per
    # chunk, which is what actually removes the per-timestep kernel-launch
    # overhead that dominates wall-clock at these batch sizes.
    #
    # `model.step` is set as an instance attribute, which shadows the class
    # method, so forward()'s `self.step(...)` picks up the compiled version
    # with no other changes needed.  The eager version is stashed on the
    # instance because torch.onnx.export does not trace reliably through a
    # compiled callable -- export_onnx() swaps it back in (see there).
    #
    # dynamic=False pins static shapes.  Train and val both use args.batch
    # so they share one graph; the batch=1 plotting/eval passes later will
    # compile a second graph, and toggling .train()/.eval() may add one
    # more.  That is 2-3 graphs total, comfortably under the 8-recompile
    # limit past which Dynamo silently falls back to eager.
    if args.freeze_blocks:
        want = {b.strip() for b in args.freeze_blocks.split(",") if b.strip()}
        all_blocks = grad_blocks(model)
        unknown = want - set(all_blocks)
        if unknown:
            raise ValueError(f"--freeze_blocks: unknown {sorted(unknown)}; "
                             f"available {sorted(all_blocks)}")
        n_frozen = 0
        for b in want:
            for p in all_blocks[b]:
                p.requires_grad_(False)
                n_frozen += p.numel()
        n_tot = sum(p.numel() for p in model.parameters())
        print(f"      FROZEN {sorted(want)}: {n_frozen:,} params "
              f"({100*n_frozen/n_tot:.0f}% of model) held at init")

    model._step_eager = model.step
    if device.type == "cuda":
        model.step = torch.compile(model.step, dynamic=False)
        print("      torch.compile: step() compiled (dynamic=False)")
    else:
        print(f"      torch.compile: SKIPPED (device={device.type}, not cuda)")

    n_par = sum(p.numel() for p in model.parameters())
    if args.arch == "timing_grouped":
        print(f"      hidden={args.hidden} PER GROUP  "
              f"n_timing={args.n_timing}  params={n_par:,}")
        print(f"      CPG({args.n_cpg_neurons}) -> timing"
              f"({args.n_timing}, LIF, per-gait weights) "
              f"-> {args.n_timing} x [{args.hidden} -> {args.hidden} -> "
              f"readout({args.readout_hidden})], no cross talk (todo 3a)")
        print(f"      timing reset={args.timing_reset}  "
              f"sub_film={args.sub_film}  sub_ln={args.sub_ln}  "
              f"event_gated={bool(args.event_gated)}")
        if args.event_gated:
            print(f"      event gating ON: each sub-network advances only on "
                  f"its own timing spike; membranes still decay, so the "
                  f"output relaxes toward b_out in between — which is what "
                  f"makes spike placement visible to the task loss.")
        if args.event_gated and args.spike_objective == "cpg_match":
            print(f"      WARNING: --spike_objective cpg_match wants every "
                  f"spike at ONE cycle phase, while event gating needs spikes "
                  f"wherever the output must change. These two pull in "
                  f"opposite directions; 'min_count' is the matching "
                  f"objective for a gated network.")
        print(f"      group -> gait-table cols : " +
              "  ".join(f"g{i}={grp}" for i, grp in enumerate(group_cols)))
        for k, v in model.param_breakdown().items():
            print(f"        {k:<8s}: {v:>9,}  ({100.0 * v / n_par:4.1f}%)")
        print(f"      tau init ranges: timing [2, "
              f"{args.tau_timing_max:.0f}]  readout [2, "
              f"{args.tau_readout_max:.0f}]  sub-net [2, "
              f"{args.tau_max:.0f}]  (period {period:.0f})")
        print(f"      timing tau init range "
              f"[{args.tau_timing_min:.0f}, {args.tau_timing_max:.0f}] steps; "
              f"sub-net [{args.tau_min:.0f}, {args.tau_max:.0f}]")
        print(f"      sub_ln={args.sub_ln} (affine=False)  "
              f"timing LayerNorm: never (see TimingGroupedSNN docstring)")
    else:
        print(f"      hidden={args.hidden}  params={n_par:,}  "
              f"(fully connected, no leg grouping)")
    n_film = model.film1.weight.numel() + model.film2.weight.numel()
    print(f"      FiLM table : {n_film:,} params for max_gaits="
          f"{args.max_gaits}, of which {len(gait_tables)} row(s) in use; "
          f"unused rows are identity modulation and get no gradient")
    if args.arch == "timing_grouped":
        n_route = model.w_in_gait.weight.numel()
        print(f"      routing: {n_route:,} params, a FREE per-gait "
              f"({args.n_cpg_neurons} x {args.n_timing}) matrix per gait row, "
              f"max_gaits={args.max_gaits}")
        print(f"               (the shared-router alternative produced "
              f"near-identical timing phases across gaits and was reverted — "
              f"see the TimingGroupedSNN docstring and todo item 11)")
    print(f"      tau range [{args.tau_min:.0f}, {args.tau_max:.0f}] steps "
          f"vs CPG period {period:.0f}")
    if args.tau_max < period:
        print(f"      WARNING: tau_max={args.tau_max:.0f} < CPG period "
              f"{period:.0f}. The leaky membranes are the ONLY long-timescale "
              f"memory in this model — raise tau_max to >= one period or the "
              f"network cannot hold phase.")
    gait_w = make_gait_weights(gait_tables_orig, gait_names, device)

    # ── Timing-layer diagnostic hook ────────────────────────────
    # Closure so run_training does not need the spike train / phase array.
    # t_eval sits in the val region so the report is about held-out steps.
    if args.arch == "timing_grouped":
        t_diag = max(t_split, t_lo) + 200
        timing_diag = lambda: timing_report(
            model, spikes, phase, period, len(gait_tables), device,
            t0=t_diag, n_steps=int(6 * period), gait_names=gait_names,
            indent="      ")
    else:
        timing_diag = None

    # ── Gain calibration ────────────────────────────────────────
    # Before ANY training: bisect each router/timing unit's FiLM gamma until
    # it fires inside the target band. Must come after the CPG run (it needs
    # a real spike train) and before the optimiser is built (it mutates
    # parameters in place).
    calib = {}
    if args.arch == "timing_grouped" and args.calibrate_gains:
        print("\n      Calibrating timing-layer gains ...")
        # Band derived from the CPG, but scaled to suit the objective.
        # cpg_match wants the CPG's own rate, so calibrate around it.
        # An EVENT-GATED network with min_count needs far more spikes than the
        # CPG emits -- a sub-network can only update when its timing neuron
        # fires, and ~10 spikes per 352-step cycle cannot render a moving
        # waveform (a zero-order-hold analysis on a gait-shaped waveform puts
        # the requirement nearer 40-60/cycle for ~1.5 degrees of error).
        # Starting at the CPG's rate would begin in a starved regime where the
        # task gradient is weak, exactly when L1 pressure is arriving.  So
        # start high and let the L1 term prune downward.
        if args.event_gated and args.spike_objective == "min_count":
            band_lo, band_hi = 2.0 * cpg_rate, 8.0 * cpg_rate
            print(f"      (event-gated + min_count: calibrating to "
                  f"{band_lo:.0f}-{band_hi:.0f} spk/cyc, well above the CPG's "
                  f"{cpg_rate:.1f}, so the network starts able to render the "
                  f"waveform and prunes from there)")
        else:
            band_lo, band_hi = 0.5 * cpg_rate, 2.0 * cpg_rate
        calib = calibrate_gains(
            model, spikes, len(gait_tables), device, period,
            lo=band_lo, hi=band_hi)

    # ── Spike objective (strategy) ──────────────────────────────
    spike_obj = make_spike_objective(
        args.spike_objective,
        lam=args.spike_stats_lambda, period=period,
        n_gaits=len(gait_tables),
        target_rate=cpg_rate, target_R=cpg_R)

    # Defined for both branches so the config can always report them.
    best          = float("nan")
    hist          = {"train": [], "val": [], "val_sw": [],
                     "gnorm": [], "sec": [], "floor": [], "upd": []}
    final_lr      = float(args.lr)
    timing_stats  = []

    if not args.dry_run:
        opt   = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr)
        # bite #3: T_max must equal the number of sched.step() calls, and
        # sched.step() now fires once per GRADIENT STEP (inside the chunk
        # loop) rather than once per epoch. With T_max=epochs the cosine
        # would finish after the first chunks_per_epoch steps and the rest
        # of training would run at eta_min. Counting in gradient steps also
        # makes the schedule independent of chunks_per_epoch, which is only
        # a logging/validation boundary.
        total_steps = args.epochs * args.chunks_per_epoch
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=1e-5)
        best, hist, timing_stats = run_training(
            model, tr_sampler, va_sampler, opt, sched,
            device, args, gait_w, out_dir, timing_diag=timing_diag,
            n_gaits=len(gait_tables), period=period,
            spike_obj=spike_obj)
        final_lr = float(opt.param_groups[0]["lr"])
        model.load_state_dict(torch.load(out_dir / "best_model.pt",
                                         map_location=device))
        print(f"\n  best val MSE : {best:.6f}")
        plot_training_curves(hist, out_dir)

        # Re-report on the RESTORED best checkpoint: the last in-loop report
        # may be several epochs stale and describes different weights.
        if timing_diag is not None:
            print("\n  Timing layer at best checkpoint:")
            lines, timing_stats = timing_diag()
            for ln in lines:
                print(ln)
    else:
        print("      --dry_run: skipping training.")
        # Still worth seeing: at init this says whether calibration put the
        # layer in a firing regime at all before you spend an hour.
        if timing_diag is not None:
            print("      Timing layer at initialisation:")
            lines, timing_stats = timing_diag()
            for ln in lines:
                print(ln)

    # ── 6. Eval + export ────────────────────────────────────────
    print("\n[6/6] Evaluation & export ...")
    t_eval = max(t_split + 800, t_lo + 800)
    rmse = plot_reconstruction(model, spikes, targets, valid, device,
                               out_dir, tgt_range, t0=t_eval,
                               gait_names=gait_names, leg_cols=leg_cols,
                               n_joints=n_joints)
    plot_transition(model, spikes, targets, device, out_dir, tgt_range,
                    t0=t_eval, gait_names=gait_names, leg_cols=leg_cols,
                    g_from=0, g_to=1)

    epochs_done = len(hist["train"])
    grad_steps  = epochs_done * args.chunks_per_epoch

    cfg = {
        # ── identity ──────────────────────────────────────────────
        "model":            ("cpg_lif_timing_grouped"
                             if args.arch == "timing_grouped"
                             else "cpg_lif_dense_stateful"),
        "arch":             args.arch,
        "config_version":   3,
        "created_utc":      datetime.now(timezone.utc).isoformat(
                                timespec="seconds"),

        # ── deployment-critical: inference.py reads these by name at
        #    the top level.  Do not move or rename them. ───────────
        # NOTE: for arch=timing_grouped, `hidden` is PER GROUP and the state
        # is (mem_timing (B,n_timing), mem1/mem2/memo (B,n_timing,hidden)).
        "hidden":           args.hidden,
        "hidden_is_per_group": args.arch == "timing_grouped",
        "max_gaits":        int(args.max_gaits),
        "n_gaits":          len(gait_tables),
        "n_legs":           int(n_legs),
        "n_joints":         int(n_joints),
        "n_cpg_neurons":    int(args.n_cpg_neurons),
        "n_timing":         (int(args.n_timing)
                             if args.arch == "timing_grouped" else None),
        "readout_hidden":   (int(args.readout_hidden)
                             if args.arch == "timing_grouped" else None),
        "group_cols":       ([list(g) for g in group_cols]
                             if group_cols is not None else None),
        "gait_names":       gait_names,
        # Same list as gait_names for CSV-loaded gaits (file stem == display
        # name, matching train_snn.py's convention) — kept as a separate key
        # anyway, so a future remap of display names doesn't have to also
        # change what visualize.py loads from disk.
        "gait_files":       gait_files,
        "gaits_dir":        str(gaits_dir.resolve()),
        "leg_cols":         [list(c) for c in leg_cols],
        "leg_layout_source": layout_src,
        "global_min":       float(tgt_range[0]),
        "global_max":       float(tgt_range[1]),
        "target_rows":      int(target_rows),
        "phase_zero":       float(args.phase_zero),
        "cpg_period_steps": float(period),
        "cpg": {
            "i_app": args.i_app, "vth_main": 100.0, "du_main": 0.1,
            "dv_main": 0.3, "refrac_main": 1, "vth_fb": 100.0,
            "du_fb": 1.0, "dv_fb": 0.0, "refrac_fb": 1,
            "from_fb_weight": CPG_FROM_FB_WEIGHT,
            "to_fb_weight": 10.0,
            "N": int(args.n_cpg_neurons),
            "W": cpg_weight_matrix(args.n_cpg_neurons).tolist(),
            "warmup": args.warmup,
        },
        "per_joint_rmse_deg": rmse.tolist(),
        "timing_layer_stats": timing_stats,
        "gain_calibration":   calib,
        "cpg_spike_stats":    {"spikes_per_cycle": cpg_rate,
                               "concentration_R": cpg_R},

        # ── full argparse namespace, verbatim ──────────────────────
        "args": vars(args),

        # ── run provenance ────────────────────────────────────────
        "run": {
            "git":            git_info(),
            "argv":           sys.argv,
            "cwd":            os.getcwd(),
            "script":         os.path.abspath(__file__),
            "out_dir":        str(out_dir.resolve()),
            "hostname":       platform.node(),
            "platform":       platform.platform(),
            "processor":      platform.processor(),
            "python":         sys.version.split()[0],
            "torch":          torch.__version__,
            "numpy":          np.__version__,
            "device":         str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device":    (torch.cuda.get_device_name(0)
                               if torch.cuda.is_available() else None),
            "torch_compile_step":       device.type == "cuda",
            "float32_matmul_precision": "high",
            "seed":           args.seed,
        },

        # ── model detail (not needed to run, useful to reproduce) ──
        # Names and shapes are read off the live model so this section
        # cannot drift from what was actually exported.
        "model_detail": {
            "class":            type(model).__name__,
            "arch":             args.arch,
            "fully_connected":  args.arch == "dense",
            "leg_grouped":      args.arch == "timing_grouped",
            "cross_talk":       False,
            "timing_layer":     (args.arch == "timing_grouped"),
            "input_routing":    (args.arch == "timing_grouped"),
            "input_routing_kind": ("per-gait learned w_in_gait embedding"
                                   if args.arch == "timing_grouped" else None),
            "n_params":         int(n_par),
            "param_breakdown":  (model.param_breakdown()
                                 if hasattr(model, "param_breakdown") else None),
            "recurrent":        False,
            "n_cpg_neurons":    int(args.n_cpg_neurons),
            "n_timing":         (int(args.n_timing)
                                 if args.arch == "timing_grouped" else None),
            "hidden_per_group": (int(args.hidden)
                                 if args.arch == "timing_grouped" else None),
            "tau_min":          float(args.tau_min),
            "tau_max":          float(args.tau_max),
            "tau_timing_min":   (float(args.tau_timing_min)
                                 if args.arch == "timing_grouped" else None),
            "tau_timing_max":   (float(args.tau_timing_max)
                                 if args.arch == "timing_grouped" else None),
            "sub_ln":           (args.sub_ln
                                 if args.arch == "timing_grouped" else None),
            "timing_layernorm": False,
            "readout_hidden":   (int(args.readout_hidden)
                                 if args.arch == "timing_grouped" else None),
            "tau_readout_max":  (float(args.tau_readout_max)
                                 if args.arch == "timing_grouped" else None),
            "timing_reset":     (args.timing_reset
                                 if args.arch == "timing_grouped" else None),
            "freeze_blocks":    (args.freeze_blocks or None),
            "event_gated":      (bool(args.event_gated)
                                 if args.arch == "timing_grouped" else None),
            "spike_objective":  (spike_obj.describe()
                                 if args.arch == "timing_grouped" else None),
            "sub_film":         (args.sub_film
                                 if args.arch == "timing_grouped" else None),
            "warm_steps":       warm_steps,


            "timing_slope":     (float(args.timing_slope)
                                 if args.arch == "timing_grouped" else None),
            "input_conditioning": ("free per-gait CPG->timing weight matrix "
                                   "(w_in_gait embedding)"
                                   if args.arch == "timing_grouped" else None),
            "slope":            float(args.slope),
            "thresh":           1.0,
            "surrogate":        "fast-sigmoid straight-through "
                                "(plain ops, bit-exact forward)",
            "state_tensors":    [n.replace("_in", "")
                                 for n in model.state_names_in],
            "state_shapes":     [list(s.shape)
                                 for s in model.init_state(1, "cpu")],
            "onnx_inputs":      ["spikes", "gait"] + list(model.state_names_in),
            "onnx_outputs":     ["angles"] + list(model.state_names_out),
            "weights_file":     "best_model.pt",
        },

        # ── CPG analysis ──────────────────────────────────────────
        "cpg_analysis": {
            "period_steps":           float(period),
            "neuron_burst_offsets":   neuron_offsets.tolist(),
            "burst_isi_thresholds":   [float(t) for t in burst_thresholds],
            "bursts_per_neuron":      [int(len(o)) for o in onsets],
            "spikes_per_neuron":      [int(c) for c in spikes.sum(0)],
            "phase_valid_coverage":   float(valid.mean()),
        },

        # ── data / split ──────────────────────────────────────────
        "data": {
            "tmax":                 int(args.tmax),
            "warmup":               int(args.warmup),
            "t_lo":                 int(t_lo),
            "t_split":              int(t_split),
            "t_hi":                 int(t_hi),
            "train_steps":          int(t_split - t_lo),
            "val_steps":            int(t_hi - t_split),
            "gait_table_rows_orig": [int(g.shape[0])
                                     for g in gait_tables_orig],
            "target_rows":          int(target_rows),
            "target_range_deg":     [float(tgt_range[0]), float(tgt_range[1])],
        },

        # ── training outcome ──────────────────────────────────────
        "training": {
            "dry_run":            bool(args.dry_run),
            "epochs_requested":   int(args.epochs),
            "epochs_completed":   int(epochs_done),
            "chunks_per_epoch":   int(args.chunks_per_epoch),
            "gradient_steps":     int(grad_steps),
            "sample_timesteps":   int(grad_steps * args.bptt * args.batch),
            "batch":              int(args.batch),
            "bptt":               int(args.bptt),
            "lr_initial":         float(args.lr),
            "lr_final":           final_lr,
            "lr_schedule":        "CosineAnnealingLR (per gradient step)",
            "lr_T_max_steps":     int(args.epochs * args.chunks_per_epoch),
            "lr_eta_min":         1e-5,
            "optimizer":          "Adam",
            "grad_clip":          float(args.clip),
            "best_val_mse":       best,
            "history":            hist,
        },
    }
    export_onnx(model, out_dir, device, cfg)

    print(f"\nDone — {out_dir.resolve()}")


if __name__ == "__main__":
    main()