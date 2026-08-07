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

    What makes the grouping viable this time is per-gait input weights.
    The old routing was a fixed permutation solved offline, and its phase
    alignment held for only 1 of 4 gaits.  Here CPG->timing is a per-gait
    learned matrix (`w_in_gait`, an Embedding lookup), which is what lets
    footfall ORDER change between gaits.  FiLM alone cannot do that -- it
    is one scale and one shift per unit, so it cannot reorder which CPG
    neuron drives which timing neuron.  See the TimingGroupedSNN docstring.

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
                 device=None):
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

        return (x, g_t, y, m,
                torch.as_tensor(sw, dtype=torch.float32, device=dev),
                torch.as_tensor(reset_mask, dtype=torch.float32, device=dev))


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
        return y, (mem1, mem2, memo)

    def forward(self, x_seq, gait_seq, state=None):
        """
        x_seq    : (L, B, n_neurons)
        gait_seq : (L, B)

        Returns (y_seq, state) with y_seq (L, B, n_joints).
        """
        L, B = x_seq.shape[0], x_seq.shape[1]
        if state is None:
            state = self.init_state(B, x_seq.device, x_seq.dtype)
        ys = []
        for t in range(L):
            y, state = self.step(x_seq[t], gait_seq[t], state)
            ys.append(y)
        return torch.stack(ys), state


class TimingGroupedSNN(nn.Module):
    """
    CPG spikes -> small TIMING layer -> G fully disconnected sub-networks.

    Shape of the thing
    ------------------
        x            (B, n_neurons)      CPG spikes, one neuron fires at most
        timing layer (B, n_timing)       LIF, densely driven by all CPG spikes
        sub-net g    (B, Hg) x2 + memo   driven by timing neuron g ALONE
        y            (B, n_joints)       group g writes its own columns only

    G == n_timing: exactly one sub-network per timing neuron, and
    `group_cols[g]` says which gait-table columns that sub-network owns
    (see build_group_cols).

    Why split it this way
    ---------------------
    The dense model has to solve two problems in one set of weights: work
    out where in the ~254-step cycle it is, and turn that into 8 angles.
    The first problem is shared across all joints and is cheap -- it is a
    handful of phase-shifted oscillations.  The second is per-joint and
    needs capacity.  Giving the timing layer n_timing units and no other
    job means the rhythm is learned in a few hundred parameters, and the
    sub-networks get a clean phase reference instead of re-deriving it
    four times over.

    This is todo item 3a: NO cross talk.  Sub-network g sees exactly one
    binary channel.  Layers 2+ are block diagonal, the readout is block
    diagonal, and there is no path between groups anywhere after the
    timing layer.  (3b -- every sub-network sees all n_timing spikes -- is
    a one-line change to layer 1, deliberately not wired up yet so the
    comparison is between two committed variants rather than a flag.)

    Per-gait input weights
    ----------------------
    `w_in_gait` is an Embedding(max_gaits, n_neurons * n_timing) reshaped
    to a per-gait (n_neurons, n_timing) matrix.  This is the learned,
    continuous successor to the deleted `solve_leg_routing` permutation.

    It exists because FiLM CANNOT express a per-gait re-assignment of legs
    to CPG neurons.  With a single static `w_in`, timing neuron j's input
    current takes one of n_neurons+1 discrete values fixed by column j of
    w_in.  FiLM contributes one scale and one shift per unit, so the
    ORDERING of those values across CPG neurons is identical for every
    gait: timing neuron j is always driven hardest by the same CPG neuron.
    Gaits whose footfall ORDER differs (not just amplitude or phase
    offset) are therefore unreachable, and the sub-networks would have to
    re-derive phase internally -- exactly the thing the timing layer is
    supposed to remove.

    Consequences, noted rather than hidden:
      - FiLM gamma on the timing layer is now largely redundant with
        w_in_gait (both per-gait and multiplicative).  beta is not -- it is
        an additive tonic drive that shifts threshold-crossing time.  Kept
        because the layer is specified to have FiLM, and because gamma
        becomes load-bearing again if w_in_gait is ever made gait-shared.
      - Unused embedding rows (n_gaits..max_gaits) hold random values and
        receive no gradient.  Unlike the FiLM tables, they are NOT inert
        identities, so loading a 4-gait checkpoint into an 8-gait run gives
        the 4 new gaits random routings.  That is the sensible default -- a
        genuinely new gait needs a new routing -- but it does mean the new
        gaits start untrained rather than as copies.

    No LayerNorm on the timing layer
    --------------------------------
    LN subtracts the mean across the normalised dimension.  Over 4-8 units
    that mean IS the signal: "some CPG neuron fired this timestep" is
    almost entirely common mode, and LN deletes it.  Worse, during CPG
    silence cur = b_timing, and LN(b) is a FIXED NONZERO vector once b
    trains away from uniform -- so every timing neuron would receive tonic
    drive and free-run during the silent gap instead of staying quiet,
    which is the opposite of what a rhythm layer should do.  A 256-wide
    layer tolerates this (the dense model has the same property and works);
    an 8-wide one will not.  `timing_w_scale` against thresh=1.0 sets the
    firing regime instead, and `timing_report` prints the rates so a dead
    or saturated timing layer is visible from epoch 1.

    Inside the sub-networks LN is optional (`sub_ln`) and uses
    elementwise_affine=False: a shared gamma/beta over (G, Hg) would be a
    parameter tied ACROSS groups, which breaks "fully disconnected".  FiLM
    follows and supplies per-group per-gait affine anyway.

    State: (mem_timing (B, n_timing), mem1, mem2, memo -- each (B, G, Hg)).
    """

    arch            = "timing_grouped"
    state_names_in  = ("mem_timing_in",  "mem1_in",  "mem2_in",  "memo_in")
    state_names_out = ("mem_timing_out", "mem1_out", "mem2_out", "memo_out")

    def __init__(self, hidden_per_group=256, n_gaits=4, max_gaits=16,
                 n_neurons=4, n_timing=N_LEGS, group_cols=None,
                 n_joints=N_JOINTS,
                 tau_min=2.0, tau_max=256.0,
                 tau_timing_min=2.0, tau_timing_max=64.0,
                 timing_w_scale=0.5, sub_ln="both",
                 slope=25.0, thresh=1.0):
        super().__init__()
        if n_gaits > max_gaits:
            raise ValueError(
                f"n_gaits ({n_gaits}) > max_gaits ({max_gaits}); raise "
                f"--max_gaits. Note that changing max_gaits changes the FiLM "
                f"and w_in_gait parameter shapes and so invalidates old "
                f"checkpoints.")
        if sub_ln not in ("none", "l1", "l2", "both"):
            raise ValueError(f"sub_ln must be none|l1|l2|both, got {sub_ln!r}")

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
        C   = len(group_cols[0])            # output columns per group

        self.G          = G
        self.Hg         = Hg
        self.C          = C
        self.n_timing   = G
        self.n_neurons  = int(n_neurons)
        self.n_joints   = int(n_joints)
        self.n_gaits    = int(n_gaits)
        self.max_gaits  = int(max_gaits)
        self.slope      = slope
        self.thresh     = thresh
        self.sub_ln     = sub_ln
        self.group_cols = group_cols
        # `H` kept as an alias for the dense model's attribute name so any
        # generic caller sizing a state tensor still works.
        self.H          = Hg

        # ── output column routing ─────────────────────────────────
        # Groups emit (B, G, C); flattening gives columns in group order,
        # e.g. [0,4, 1,5, 2,6, 3,7] for leg grouping.  out_perm reorders
        # that into gait-table column order in one index_select, so no
        # scatter and nothing for ONNX to choke on.
        flat = [c for grp in group_cols for c in grp]
        inv  = np.empty(n_joints, dtype=np.int64)
        for pos, col in enumerate(flat):
            inv[col] = pos
        self.register_buffer("out_perm", torch.from_numpy(inv), persistent=False)

        # ── timing layer ──────────────────────────────────────────
        # Per-gait (n_neurons, n_timing) matrix, stored flat in an Embedding.
        self.w_in_gait = nn.Embedding(max_gaits, self.n_neurons * G)
        nn.init.normal_(self.w_in_gait.weight, mean=0.0, std=timing_w_scale)
        self.b_timing  = nn.Parameter(torch.zeros(G))
        self.film_t    = nn.Embedding(max_gaits, 2 * G)
        nn.init.zeros_(self.film_t.weight)
        self.film_t.weight.data[:, :G] = 1.0        # gamma := 1, beta := 0
        self.beta_t_logit = nn.Parameter(
            init_beta_logit((G,), tau_timing_min, tau_timing_max))

        # ── sub-network layer 1: ONE binary spike -> Hg units ─────
        # The only input is spk_timing[g], so this is an outer product, not
        # a matmul: cur1 = spk[..., None] * w1 + b1.
        self.w1 = nn.Parameter(torch.randn(G, Hg) * timing_w_scale)
        self.b1 = nn.Parameter(torch.zeros(G, Hg))

        # ── sub-network layer 2: block diagonal (G, Hg, Hg) ───────
        self.w2 = nn.Parameter(torch.randn(G, Hg, Hg) / math.sqrt(Hg))
        self.b2 = nn.Parameter(torch.zeros(G, Hg))

        # ── block-diagonal analog readout ─────────────────────────
        self.w_read = nn.Parameter(torch.randn(G, Hg, Hg) / math.sqrt(Hg))
        self.b_read = nn.Parameter(torch.zeros(G, Hg))
        self.w_out  = nn.Parameter(torch.randn(G, Hg, C) / math.sqrt(Hg))
        self.b_out  = nn.Parameter(torch.zeros(G, C))

        # affine=False: a shared gamma/beta would tie parameters across
        # groups and break the disconnection.  FiLM supplies the affine.
        self.ln1 = nn.LayerNorm(Hg, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(Hg, elementwise_affine=False)

        # ── heterogeneous learnable taus, per group per unit ──────
        self.beta1_logit = nn.Parameter(init_beta_logit((G, Hg), tau_min, tau_max))
        self.beta2_logit = nn.Parameter(init_beta_logit((G, Hg), tau_min, tau_max))
        self.betao_logit = nn.Parameter(init_beta_logit((G, Hg), 2.0, 40.0))

        # ── FiLM, per group per unit, over-allocated to max_gaits ─
        self.film1 = nn.Embedding(max_gaits, 2 * G * Hg)
        self.film2 = nn.Embedding(max_gaits, 2 * G * Hg)
        for e in (self.film1, self.film2):
            nn.init.zeros_(e.weight)
            e.weight.data[:, :G * Hg] = 1.0          # gamma := 1, beta := 0

    # ---------------------------------------------------------------
    def init_state(self, batch, device, dtype=torch.float32):
        z = lambda: torch.zeros(batch, self.G, self.Hg,
                                device=device, dtype=dtype)
        mem_t = torch.zeros(batch, self.n_timing, device=device, dtype=dtype)
        return (mem_t, z(), z(), z())

    # ---------------------------------------------------------------
    def _timing(self, x, gait, mem_t):
        """
        One timestep of the timing layer alone.

        x     : (B, n_neurons)
        gait  : (B,) int64
        mem_t : (B, n_timing)

        Factored out so `timing_only` can run it for diagnostics without
        touching the sub-networks (and without going through the compiled
        `step`, which would add a Dynamo guard set).
        """
        W  = self.w_in_gait(gait).view(-1, self.n_neurons, self.n_timing)
        cur = torch.bmm(x.unsqueeze(1), W).squeeze(1) + self.b_timing
        v   = self.film_t(gait)
        cur = cur * v[:, :self.n_timing] + v[:, self.n_timing:]
        beta_t = torch.sigmoid(self.beta_t_logit)
        mem_t  = beta_t * mem_t + cur
        spk_t  = spike_fn(mem_t - self.thresh, self.slope)
        mem_t  = mem_t - self.thresh * spk_t
        return spk_t, mem_t

    def step(self, x, gait, state):
        """
        x     : (B, n_neurons) float — CPG spikes this timestep
        gait  : (B,) int64
        state : (mem_timing, mem1, mem2, memo)
        """
        mem_t, mem1, mem2, memo = state
        G, Hg = self.G, self.Hg

        # ---- timing layer (no LayerNorm — see class docstring) -------
        spk_t, mem_t = self._timing(x, gait, mem_t)          # (B, G)

        v1 = self.film1(gait).view(-1, 2, G, Hg)
        v2 = self.film2(gait).view(-1, 2, G, Hg)
        g1, f1 = v1[:, 0], v1[:, 1]
        g2, f2 = v2[:, 0], v2[:, 1]

        # ---- sub-net layer 1: outer product, no cross-group path -----
        cur1 = spk_t.unsqueeze(-1) * self.w1 + self.b1        # (B, G, Hg)
        if self.sub_ln in ("l1", "both"):
            cur1 = self.ln1(cur1)
        cur1  = cur1 * g1 + f1
        beta1 = torch.sigmoid(self.beta1_logit)
        mem1  = beta1 * mem1 + cur1
        spk1  = spike_fn(mem1 - self.thresh, self.slope)
        mem1  = mem1 - self.thresh * spk1

        # ---- sub-net layer 2: block diagonal ------------------------
        cur2 = torch.einsum("bgh,ghk->bgk", spk1, self.w2) + self.b2
        if self.sub_ln in ("l2", "both"):
            cur2 = self.ln2(cur2)
        cur2  = cur2 * g2 + f2
        beta2 = torch.sigmoid(self.beta2_logit)
        mem2  = beta2 * mem2 + cur2
        spk2  = spike_fn(mem2 - self.thresh, self.slope)
        mem2  = mem2 - self.thresh * spk2

        # ---- block-diagonal analog readout -------------------------
        curo  = torch.einsum("bgh,ghk->bgk", spk2, self.w_read) + self.b_read
        betao = torch.sigmoid(self.betao_logit)
        memo  = betao * memo + curo

        y_grp = torch.einsum("bgh,ghc->bgc", memo, self.w_out) + self.b_out
        y     = y_grp.flatten(1).index_select(1, self.out_perm)
        return y, (mem_t, mem1, mem2, memo)

    def forward(self, x_seq, gait_seq, state=None):
        """
        x_seq    : (L, B, n_neurons)
        gait_seq : (L, B)
        Returns (y_seq, state) with y_seq (L, B, n_joints).
        """
        B = x_seq.shape[1]
        if state is None:
            state = self.init_state(B, x_seq.device, x_seq.dtype)
        ys = []
        for t in range(x_seq.shape[0]):
            y, state = self.step(x_seq[t], gait_seq[t], state)
            ys.append(y)
        return torch.stack(ys), state

    # ---------------------------------------------------------------
    @torch.no_grad()
    def timing_only(self, x_seq, gait_seq, mem_t=None):
        """
        Run ONLY the timing layer over a sequence.  Diagnostics path.

        Deliberately calls `self._timing` rather than `self.step`: the
        sub-networks are ~99% of the FLOPs and are not needed to ask
        "when does each timing neuron fire", and going through the
        compiled `step` at batch=1 would add a Dynamo guard set per call
        shape.

        Returns (L, B, n_timing) spikes.
        """
        B = x_seq.shape[1]
        if mem_t is None:
            mem_t = torch.zeros(B, self.n_timing, device=x_seq.device,
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
            "timing":  n(self.w_in_gait.weight, self.b_timing,
                         self.film_t.weight, self.beta_t_logit),
            "sub_l1":  n(self.w1, self.b1, self.beta1_logit),
            "sub_l2":  n(self.w2, self.b2, self.beta2_logit),
            "readout": n(self.w_read, self.b_read, self.w_out, self.b_out,
                         self.betao_logit),
            "film":    n(self.film1.weight, self.film2.weight),
        }


class SingleStepONNX(nn.Module):
    """Flat-signature wrapper so the exported graph is one timestep with
    explicit state in/out — the robot calls this once per CPG step."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spikes, gait, mem1, mem2, memo):
        y, (m1, m2, mo) = self.model.step(
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
        y, (mt, m1, m2, mo) = self.model.step(
            spikes, gait, (mem_timing, mem1, mem2, memo))
        return y, mt, m1, m2, mo


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
        spk = model.timing_only(x, gg)[:, 0].cpu().numpy()    # (L, n_timing)

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


def run_training(model, tr_sampler, va_sampler, opt, sched, device, args,
                 gait_w, out_dir, timing_diag=None):
    """
    `timing_diag` : optional zero-arg callable returning (lines, stats).
                    Called every args.timing_log_every epochs for the
                    timing-grouped arch; None for the dense arch.  Its last
                    return value is handed back so the config can record it.
    """
    best = float("inf")
    best_path = out_dir / "best_model.pt"
    hist = {"train": [], "val": [], "val_sw": [], "gnorm": [], "sec": []}
    last_timing_stats = []

    print(f"\n  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  "
          f"{'Val(post-sw)':>13}  {'LR':>9}  {'|grad|':>8}  {'sec':>6}")
    print("  " + "-" * 78)
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

            # ---- train -------------------------------------------------
            model.train()
            state = model.init_state(args.batch, device)
            tot, gtot, nb = 0.0, 0.0, 0
            for _ in range(args.chunks_per_epoch):
                x, g, y, m, sw, rst = tr_sampler.next_chunk(args.bptt)
                x, g, y, m = (x.to(device), g.to(device),
                              y.to(device), m.to(device))
                state = apply_reset(detach_state(state), rst.to(device))

                pred, state = model(x, g, state)
                loss = masked_loss(pred, y, m, g, gait_w)

                opt.zero_grad()
                loss.backward()
                # Returns the total norm BEFORE clipping — free to read,
                # and the only way to tell whether clip=1.0 is quietly
                # truncating nearly every update (i.e. masking a too-hot LR).
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

            # ---- validate ----------------------------------------------
            model.eval()
            vstate = model.init_state(args.batch, device)
            vtot, vsw_tot, vn, vsw_n = 0.0, 0.0, 0, 0
            with torch.no_grad():
                for _ in range(args.val_chunks):
                    x, g, y, m, sw, rst = va_sampler.next_chunk(args.bptt)
                    x, g, y, m = (x.to(device), g.to(device),
                                  y.to(device), m.to(device))
                    sw = sw.to(device)
                    vstate = apply_reset(vstate, rst.to(device))
                    pred, vstate = model(x, g, vstate)

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

            flag = ""
            if va_loss < best:
                best = va_loss
                torch.save(model.state_dict(), best_path)
                flag = " *"

            if epoch % args.log_every == 0 or epoch == 1:
                print(f"  {epoch:>6}  {tr_loss:>10.6f}  {va_loss:>10.6f}  "
                      f"{vsw:>13.6f}  {opt.param_groups[0]['lr']:>9.2e}"
                      f"  {tr_gnorm:>8.2f}  {epoch_s:>6.1f}{flag}")

            # Timing layer: cheap (timing units only, batch 1) but it prints
            # n_gaits lines, so it runs on its own slower cadence.
            if timing_diag is not None and (
                    epoch % args.timing_log_every == 0 or epoch == 1):
                lines, last_timing_stats = timing_diag()
                for ln in lines:
                    print(ln)

    except KeyboardInterrupt:
        # Return normally rather than propagating: main() then falls through
        # to plots + config + ONNX export using the best checkpoint so far,
        # so an aborted run still produces deployable artifacts.
        done = len(hist["train"])
        print()
        print("  " + "-" * 78)
        print(f"  [INTERRUPT] Ctrl+C received during epoch {done + 1}.")
        print(f"              {done} epoch(s) completed and recorded; the "
              f"partial epoch is discarded.")
        print(f"              Best val MSE so far : {best:.6f}")
        print( "              Stopping training and proceeding to export.")

    print("  " + "-" * 78)
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
    ap.add_argument("--arch", type=str, default="dense",
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
    ap.add_argument("--tau_timing_min", type=float, default=2.0)
    ap.add_argument("--tau_timing_max", type=float, default=64.0,
                    help="[timing_grouped] Tau ceiling for the TIMING layer "
                         "only. Separate from --tau_max because the sample is "
                         "tiny: with n_timing=4 units, a log-uniform draw "
                         "over [2, 256] puts ~1 unit above tau=64, so which "
                         "timescales get covered is decided by the seed "
                         "rather than by tiling. The timing layer also does "
                         "not need to HOLD a cycle -- the sub-networks do "
                         "that -- it only needs to re-time bursts, which "
                         "needs memory of order the inter-burst onset gap "
                         "(~63 steps at period 254). Taus stay learnable, so "
                         "this is an init range, not a cap.")
    ap.add_argument("--timing_w_scale", type=float, default=0.5,
                    help="[timing_grouped] Init std for the per-gait CPG-> "
                         "timing weights and for sub-net layer 1. There is NO "
                         "LayerNorm on the timing layer, so this sets the "
                         "firing regime directly against thresh=1.0 — check "
                         "the spk/cyc column of the timing report and raise "
                         "it if units are dead, lower it if saturated.")
    ap.add_argument("--sub_ln", type=str, default="both",
                    choices=["none", "l1", "l2", "both"],
                    help="[timing_grouped] Which sub-network layers get "
                         "LayerNorm (elementwise_affine=False; FiLM supplies "
                         "the affine). Never applies to the timing layer. "
                         "Worth ablating on l1: that layer's only input is "
                         "one binary channel, so LN normalises away the "
                         "amplitude of the sole drive and leaves FiLM gamma "
                         "to restore it.")

    ap.add_argument("--hidden",     type=int,   default=256,
                    help="Hidden width. For --arch dense this is the TOTAL "
                         "width (dense H x H layers). For --arch "
                         "timing_grouped it is PER GROUP, so w2/w_read are "
                         "(G, H, H) — at G=4, hidden=128 lands near the dense "
                         "hidden=256 parameter count and is the matched-"
                         "parameter baseline; hidden=256 is ~4x that.")
    ap.add_argument("--max_gaits",  type=int,   default=16,
                    help="Rows allocated in the FiLM embedding tables (and, "
                         "for timing_grouped, in the per-gait CPG->timing "
                         "table). Only the first n_gaits are used. Fixing "
                         "this keeps every parameter shape independent of the "
                         "gait count, so checkpoints transfer between runs "
                         "with different numbers of gaits. Changing it does "
                         "NOT. Note the unused FiLM rows are inert identities "
                         "but the unused w_in_gait rows are random, so a new "
                         "gait starts with a random routing rather than a "
                         "trained one.")
    ap.add_argument("--tau_min",    type=float, default=2.0)
    ap.add_argument("--tau_max",    type=float, default=256.0,
                    help="Longest membrane time constant, in steps. Lowered "
                         "from 500: what the network actually has to hold is "
                         "'which neuron burst last, and how long ago', and "
                         "inter-burst onset spacing is only ~63 steps "
                         "(silent gap ~34). One full CPG period (~254) is "
                         "the natural ceiling; 500 was overkill. Sweep "
                         "150-256.")
    ap.add_argument("--slope",      type=float, default=25.0)

    # training
    ap.add_argument("--epochs",           type=int,   default=100)
    ap.add_argument("--chunks_per_epoch", type=int,   default=40)
    ap.add_argument("--val_chunks",       type=int,   default=2)
    ap.add_argument("--bptt",             type=int,   default=256,
                    help="Gradient truncation horizon. NOT the network's "
                         "receptive field -- state is carried and detached "
                         "across chunks, so the forward pass sees unbounded "
                         "history. 256 ~= one full CPG period. Sweep "
                         "128/256/512 at fixed batch*bptt.")
    ap.add_argument("--batch",            type=int,   default=256,
                    help="Stream heads per gradient step. Raised from 32: at "
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
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--timing_log_every", type=int, default=10,
                    help="[timing_grouped] Epoch cadence for the timing-layer "
                         "firing report. Slower than --log_every because it "
                         "prints one line per gait; the compute is "
                         "negligible (timing units only, batch 1).")
    ap.add_argument("--dry_run",   action="store_true",
                    help="Build data + diagnostics, skip training.")
    ap.add_argument("--out_dir",   type=str, default="outputs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = Path(this_file_dir + "/" + args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
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
    tr_sampler = StreamSampler(spikes, targets, valid, t_lo, t_split,
                               args.batch, args.switch_min, args.switch_max,
                               rng, n_gaits=len(gait_tables), device=device)
    va_sampler = StreamSampler(spikes, targets, valid, t_split, t_hi,
                               args.batch, args.switch_min, args.switch_max,
                               np.random.default_rng(args.seed + 1),
                               n_gaits=len(gait_tables), device=device)

    # ── 5. Model + training ─────────────────────────────────────
    print("\n[5/6] Model ...")
    if args.arch == "timing_grouped":
        model = TimingGroupedSNN(
            hidden_per_group=args.hidden, n_gaits=len(gait_tables),
            max_gaits=args.max_gaits, n_neurons=args.n_cpg_neurons,
            n_timing=args.n_timing, group_cols=group_cols,
            n_joints=n_joints,
            tau_min=args.tau_min, tau_max=args.tau_max,
            tau_timing_min=args.tau_timing_min,
            tau_timing_max=args.tau_timing_max,
            timing_w_scale=args.timing_w_scale, sub_ln=args.sub_ln,
            slope=args.slope).to(device)
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
        print(f"      CPG({args.n_cpg_neurons}) -> timing({args.n_timing}) "
              f"-> {args.n_timing} x [{args.hidden} -> {args.hidden} -> "
              f"readout], no cross talk (todo 3a)")
        print(f"      group -> gait-table cols : " +
              "  ".join(f"g{i}={grp}" for i, grp in enumerate(group_cols)))
        for k, v in model.param_breakdown().items():
            print(f"        {k:<8s}: {v:>9,}  ({100.0 * v / n_par:4.1f}%)")
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
        print(f"      per-gait CPG->timing routing : {n_route:,} params "
              f"(unused rows are RANDOM, not identity — a new gait starts "
              f"with an untrained routing)")
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

    # Defined for both branches so the config can always report them.
    best          = float("nan")
    hist          = {"train": [], "val": [], "val_sw": [],
                     "gnorm": [], "sec": []}
    final_lr      = float(args.lr)
    timing_stats  = []

    if not args.dry_run:
        opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
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
            device, args, gait_w, out_dir, timing_diag=timing_diag)
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
        # Still worth seeing: at init this says whether timing_w_scale has
        # put the layer in a firing regime at all before you spend an hour.
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
            "timing_w_scale":   (float(args.timing_w_scale)
                                 if args.arch == "timing_grouped" else None),
            "sub_ln":           (args.sub_ln
                                 if args.arch == "timing_grouped" else None),
            "timing_layernorm": False,
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