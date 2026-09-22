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

4.  `--arch dense` (DenseSNN): fully connected. Kept as the ablation
    baseline, and it accepts the same feature flags as the grouped model
    so the comparison isolates the factorisation.
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
    TimingGroupedSNN docstring for the diagnosis.

    Sub-networks never see each other's timing spikes. A sub-network's fan-in
    is whichever timing units own its output columns (--timing_shape /
    --decoder_shape); full cross talk, where every sub-network sees all
    n_timing spikes, is not wired up.

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

    # timing layer + per-leg sub-networks
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


STRUCTURE_SHAPES = ("per_leg", "per_joint")


def build_partition(shape, leg_cols, n_joints, what="structure"):
    """
    Output columns owned by each unit of a "per_leg" or "per_joint" structure.

    This is the single primitive behind both --timing_shape and
    --decoder_shape.  Saying a layer is shaped "per_leg" means it has one unit
    per leg and that unit is responsible for that leg's columns; "per_joint"
    means one unit per column.  Expressing both layers in the same currency --
    which output columns does each unit own -- is what lets `build_timing_map`
    connect them without any special-casing per combination.

    `build_group_cols` is the older form of this, keyed on a neuron COUNT
    rather than a name.  It is kept because configs written before
    --timing_shape existed record only `n_timing`, and reconstructing those
    checkpoints has to keep working.
    """
    if shape == "per_leg":
        groups = [list(c) for c in leg_cols]
    elif shape == "per_joint":
        groups = [[j] for j in range(n_joints)]
    else:
        raise ValueError(f"{what} must be one of "
                         f"{'|'.join(STRUCTURE_SHAPES)}, got {shape!r}")

    flat = [c for grp in groups for c in grp]
    if sorted(flat) != list(range(n_joints)):
        raise ValueError(
            f"{what}={shape!r} gives {groups}, not a partition of "
            f"0..{n_joints - 1} (flattened: {sorted(flat)}). Check --leg_cols.")
    if len({len(g) for g in groups}) != 1:
        raise ValueError(f"{what}={shape!r} gives unequal groups {groups}; "
                         f"w1/w_out would be ragged. Check --leg_cols.")
    return groups


def build_timing_map(timing_cols, group_cols):
    """
    Which timing units feed each sub-network: a (G, K) list of indices.

    THE RULE: sub-network g takes every timing unit whose output columns
    overlap g's own.  One rule covers every combination of the two shapes,
    including the two where the counts differ:

      timing per_joint + decoder per_leg   K=3, disjoint.  18 timing units,
          6 decoders, each decoder fed by the 3 timing units for its leg.
          The interesting case: a decoder sees three independent phase
          references and can learn inter-joint structure within its leg.

      timing per_leg + decoder per_joint   K=1, SHARED.  6 timing units, 18
          decoders; the 3 decoders of a leg are all fed the same spike train,
          so they differ only in their own weights.  Legal but nearly
          pointless -- it spends 3x the sub-network parameters on 1x the
          timing information.

      matched shapes                       K=1, disjoint.  What this model did
          before either arg existed.

    K must come out equal across groups, because `w1` is a dense (G, K, Hg)
    tensor and a ragged fan-in has nowhere to live.  Both shapes partition the
    same columns, so equality is automatic when one shape refines the other;
    it is validated anyway because --leg_cols is user-supplied.
    """
    G = len(group_cols)
    tmap = []
    for gcols in group_cols:
        own = set(gcols)
        tmap.append([t for t, tcols in enumerate(timing_cols)
                     if own & set(tcols)])

    if any(not m for m in tmap):
        bad = [g for g, m in enumerate(tmap) if not m]
        raise ValueError(
            f"sub-network(s) {bad} have no timing input: group_cols "
            f"{group_cols} and timing_cols {timing_cols} do not overlap. A "
            f"sub-network with no input is permanently silent.")
    sizes = {len(m) for m in tmap}
    if len(sizes) != 1:
        raise ValueError(
            f"timing fan-in must be equal for every sub-network (w1 is a "
            f"dense (G, K, Hg) tensor); got sizes {sorted(sizes)} from "
            f"group_cols {group_cols} and timing_cols {timing_cols}.")

    used = {t for m in tmap for t in m}
    missing = sorted(set(range(len(timing_cols))) - used)
    if missing:
        raise ValueError(
            f"timing unit(s) {missing} feed no sub-network, so their "
            f"CPG->timing weights would receive zero gradient forever. "
            f"group_cols {group_cols}, timing_cols {timing_cols}.")

    assert len(tmap) == G
    return tmap


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
        # Stored so fake_step_chunk (below) can reuse the exact constants
        # that configure self.core, instead of redeclaring its own copy of
        # every default.
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
        """
        Synthetic stand-in for step_chunk: N evenly-spaced, BACK-TO-BACK
        bursts per cycle -- no inter-burst gap -- instead of the real CPG's
        bursts separated by silence.

        Temporary.  For testing whether training benefits from a
        continuous-activity CPG ahead of the real oscillator being retuned to
        produce this directly (see architecture_change_todo.md). Uses the
        same refrac_main / vth_fb / to_fb_weight this instance's self.core
        was actually built with, so the burst width matches step_chunk's even
        though the gap between bursts does not.
        """
        n_spikes = int((self.vth_fb // self.to_fb_weight)
                       * (self.refrac_main + 1))
        period = n_spikes * self.N
        pattern = np.zeros((self.N, n_steps), dtype=np.float32)
        for i in range(self.N):
            start = i * n_spikes
            for t in range(start, n_steps, period):
                pattern[i, t:t + n_spikes:self.refrac_main + 1] = 1.0
        return pattern.T

    def reset(self):
        self.core.reset()
        self.inter_neuron_current.fill(0.0)
        self.t = 0


def run_cpg(N, tmax=120_000, warmup=2_000, i_app=8.0, fake_cpg=False):
    """Warm up, then collect the spike train used for training.

    `N` is deliberately positional-with-no-default: it selects the coupling
    matrix, and a wrong value changes the oscillator rather than raising, so
    the caller is made to say it.

    from_fb_weight is not a parameter here: it is fixed at
    CPG_FROM_FB_WEIGHT (see LIFCPGStepper) for every N, so there is nothing
    for a caller to get wrong by omission.

    fake_cpg=True substitutes fake_step_chunk's back-to-back, no-gap bursts
    for the real oscillator's output -- see that method's docstring.
    """
    cpg = LIFCPGStepper(N=N, i_app=i_app)
    print(f"  N={N}  i_app={i_app}  from_fb_weight={CPG_FROM_FB_WEIGHT:g}")
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
    "tripod", "tripod_backwards",
    "tripod_right", "tripod_left",
    "ripple", "ripple_backwards",
    "ripple_right", "ripple_left"
]
GAIT_FILES_BY_N = {4: QUADRUPED_GAIT_FILES, 6: HEXAPOD_GAIT_FILES}


# Output files are grouped into per-category subfolders of the run dir.
# Routing is by FILENAME so every call site can keep passing a bare name and
# the layout lives in exactly one place: train.py, visualize_timing.py and
# reorganize_outputs.py all consult this table and so cannot disagree.
#
# Left LOOSE in the run directory (deliberately -- these are the ones looked at
# first): metrics.csv, training_curves.png, transition.png, rmse_heatmap.png.
#
# The model/ entries are EXACT filenames rather than ".pt"/".onnx" suffixes, so
# nothing unexpected gets swept in -- including .onnx.data, which
# torch.onnx.export writes alongside the graph when weights are stored
# externally and which is useless separated from it.
OUT_ROUTES = (
    # (kind, pattern, subfolder);  kind is "prefix" | "exact"
    ("prefix", "recon_",                      "recons"),
    ("prefix", "timing_alignment_",           "timing_alignments"),
    ("prefix", "membranes_",                  "membrane_waveforms"),
    ("prefix", "phase_fold_",                 "phase_folds"),
    ("exact",  "best_model.pt",               "model"),
    ("exact",  "cpg_lif_snn_config.json",     "model"),
    ("exact",  "cpg_lif_snn_step.onnx",       "model"),
    ("exact",  "cpg_lif_snn_step.onnx.data",  "model"),
    ("exact",  "alignment_summary.png",       "misc_info"),
    ("exact",  "burst_threshold.png",         "misc_info"),
    ("exact",  "cpg_raster.png",              "misc_info"),
    ("exact",  "routing_matrices.png",        "misc_info"),
    ("exact",  "tau_distributions.png",       "misc_info"),
    ("exact",  "timing_summary.json",         "misc_info"),
)
OUT_CATEGORY_DIRS = tuple(dict.fromkeys(d for _, _, d in OUT_ROUTES))


def route_subdir(name):
    """Category subfolder for a filename, or "" to leave it loose."""
    for kind, pat, sub in OUT_ROUTES:
        if ((kind == "prefix" and name.startswith(pat))
                or (kind == "exact" and name == pat)):
            return sub
    return ""


def out_path(out_dir, name):
    """Routed path for WRITING `name`, with its parent directory created."""
    p = Path(out_dir, route_subdir(name), name)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def in_path(in_dir, name):
    """
    Routed path for READING `name`, falling back to the loose location.

    The fallback is what keeps runs made before this layout loadable: their
    best_model.pt sits at the top level, not in model/. If neither exists the
    routed path is returned so the caller's error message points at where the
    file is now supposed to live.
    """
    routed = Path(in_dir, route_subdir(name), name)
    if routed.exists():
        return routed
    loose = Path(in_dir, name)
    return loose if loose.exists() else routed

# Anatomical name for the k-th joint within a leg, keyed by joints-per-leg.
# Used for axis labels wherever joints are grouped by type.
JOINT_TYPE_NAMES = {2: ["shoulder", "knee"],
                    3: ["coxa", "femur", "tibia"]}


def joint_type_names(k):
    """Names for k joints-per-leg, falling back to generic labels."""
    return JOINT_TYPE_NAMES.get(k, [f"joint {i}" for i in range(k)])

def outputs_path(this_file_dir, rel=""):
    """
    this_file_dir/outputs[/rel].

    Used for --out_dir (train.py, visualize_timing.py) and --model_dir
    (visualize_timing.py) so a bare name like "test1" always lands at
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


# Weight init scales.  Internal rather than args because neither needs user
# input:
#   _W_IN_INIT : CPG->timing weights.  calibrate_gains rescales these before
#                training, so this only sets relative shape and the sign of
#                the mean -- there is nothing to tune.
#   _W1_INIT   : timing->sub-net layer 1.  Deliberately uncalibrated (see the
#                note at self.w1), and learnable, so this is a starting point
#                rather than a setting.  Note it is inert whenever layer 1 has
#                LayerNorm (--sub_ln l1/both), since LN is exactly
#                scale-invariant with b1 init at zero.
_W_IN_INIT = 0.5
_W1_INIT   = 0.5

# Highest fraction of the phase-blind ceiling that gain calibration is allowed
# to target.  The ceiling is n_cpg * cpg_rate spikes per cycle -- one timing
# spike per CPG spike -- at which point the unit's spike train is the OR of
# the CPG's, identical for every saturated unit, and carries no phase.  A unit
# already at 0.6 of that is firing on most CPG spikes and is close to useless,
# so the calibration band is capped here rather than at 1.0.
_SAT_FRAC = 0.6

# Smallest gap kept between a voltage-mode bias and `thresh`, so the
# effective threshold `thresh - v` stays strictly positive.  See
# TimingGroupedSNN.clamp_bias_voltage.
_V_EPS = 1e-6

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


class DenseSNN(nn.Module):
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

    State: mem1, mem2 are (B, H); memo is (B, Ho), which equals H unless
    --readout_hidden narrows it.

    Shared options with TimingGroupedSNN
    ------------------------------------
    readout_hidden, tau_readout_max, bias_mode, hidden_reset, sub_ln and
    sub_film all mean exactly what they mean there, and are accepted here so
    the dense ABLATION differs from the grouped model in the factorisation
    and in nothing else. An ablation whose baseline also differs in
    normalisation placement, bias parameterisation, readout width and reset
    convention cannot isolate what it claims to test.

    The defaults are the values this class used BEFORE the options existed --
    readout_hidden=None meaning "as wide as hidden", tau_readout_max=40,
    bias currents, subtractive reset, LayerNorm and FiLM on both layers -- so
    an old checkpoint still reconstructs as what it actually ran.
    """

    # Consumed by export_onnx / config so neither has to branch on isinstance.
    arch            = "dense"
    state_names_in  = ("mem1_in",  "mem2_in",  "memo_in")
    state_names_out = ("mem1_out", "mem2_out", "memo_out")

    def __init__(self, hidden=128, n_gaits=4, max_gaits=16,
                 tau_min=2.0, tau_max=256.0,
                 slope=25.0, thresh=1.0, n_neurons=4, n_joints=N_JOINTS,
                 readout_hidden=None, tau_readout_max=40.0,
                 bias_mode="current", hidden_reset="subtract",
                 sub_ln="both", sub_film="both", film_mode="gamma_only"):
        super().__init__()
        if n_gaits > max_gaits:
            raise ValueError(
                f"n_gaits ({n_gaits}) > max_gaits ({max_gaits}); raise "
                f"--max_gaits. Note that changing max_gaits changes the FiLM "
                f"parameter shape and so invalidates old checkpoints.")
        for nm, val, ok in (("bias_mode", bias_mode,
                             ("current", "voltage", "none")),
                            ("hidden_reset", hidden_reset,
                             ("subtract", "zero")),
                            ("sub_ln", sub_ln, ("none", "l1", "l2", "both")),
                            ("sub_film", sub_film,
                             ("none", "l1", "l2", "both"))):
            if val not in ok:
                raise ValueError(f"{nm} must be one of {ok}, got {val!r}")
        self.H         = hidden
        # None = as wide as the hidden layers, which is what this class did
        # before the option existed.
        self.Ho        = int(readout_hidden) if readout_hidden else hidden
        self.bias_mode    = bias_mode
        self.hidden_reset = hidden_reset
        self.sub_ln       = sub_ln
        self.sub_film     = sub_film
        self.film_mode    = film_mode
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
        self.ln1    = nn.LayerNorm(hidden)

        # ── layer 2 ───────────────────────────────────────────────
        self.w2     = nn.Parameter(torch.randn(hidden, hidden) / math.sqrt(hidden))
        self.ln2    = nn.LayerNorm(hidden)

        # ── readout: non-spiking leaky membrane, then joint angles ─
        Ho = self.Ho
        self.w_read = nn.Parameter(torch.randn(hidden, Ho) / math.sqrt(hidden))
        self.w_out  = nn.Parameter(torch.randn(Ho, n_joints) / math.sqrt(Ho))
        self.b_out  = nn.Parameter(torch.zeros(n_joints))

        # ── bias: additive current, or an offset to the threshold ─
        # b_read has no voltage equivalent (memo is analog and never spikes,
        # so there is no threshold to offset); it is dropped outside "current"
        # mode and b_out carries the output offset, which is added to y rather
        # than to a membrane and so is not a bias current in the sense that
        # matters.
        if bias_mode == "current":
            self.b1     = nn.Parameter(torch.zeros(hidden))
            self.b2     = nn.Parameter(torch.zeros(hidden))
            self.b_read = nn.Parameter(torch.zeros(Ho))
        elif bias_mode == "voltage":
            self.v1 = nn.Parameter(torch.zeros(hidden))
            self.v2 = nn.Parameter(torch.zeros(hidden))

        # ── heterogeneous, learnable time constants ───────────────
        self.beta1_logit = nn.Parameter(init_beta_logit((hidden,), tau_min, tau_max))
        self.beta2_logit = nn.Parameter(init_beta_logit((hidden,), tau_min, tau_max))
        self.betao_logit = nn.Parameter(
            init_beta_logit((self.Ho,), 2.0, float(tau_readout_max)))

        # ── FiLM: per-gait scale/shift, over-allocated ────────────
        if sub_film in ("l1", "both"):
            self.film1 = nn.Embedding(max_gaits, 2 * hidden)
        if sub_film in ("l2", "both"):
            self.film2 = nn.Embedding(max_gaits, 2 * hidden)
        for e in (getattr(self, "film1", None), getattr(self, "film2", None)):
            if e is not None:
                nn.init.zeros_(e.weight)
                e.weight.data[:, :hidden] = 1.0   # gamma := 1, beta := 0

    # ---------------------------------------------------------------
    def init_state(self, batch, device, dtype=torch.float32):
        """(mem1, mem2, memo). memo is Ho wide, which is H unless narrowed."""
        z = lambda w: torch.zeros(batch, w, device=device, dtype=dtype)
        return (z(self.H), z(self.H), z(self.Ho))

    def step(self, x, gait, state):
        """
        x     : (B, n_neurons) float — CPG spikes this timestep
        gait  : (B,) int64
        state : 3-tuple (mem1, mem2, memo); mem1/mem2 are (B, H) and
                memo is (B, Ho)

        Feedforward within a timestep: spk1/spk2 are local, consumed by the
        next layer in the same step and never carried across steps.
        `addmm` folds each bias into its matmul, one kernel per layer.
        """
        mem1, mem2, memo = state
        cur_bias = self.bias_mode == "current"

        # ---- layer 1 -------------------------------------------------
        cur1 = (torch.addmm(self.b1, x, self.w_in) if cur_bias
                else x @ self.w_in)
        if self.sub_ln in ("l1", "both"):
            cur1 = self.ln1(cur1)
        if self.sub_film in ("l1", "both"):
            v1 = self.film1(gait)                          # (B, 2H)
            # See TimingGroupedSNN's film_mode note: gamma is multiplicative
            # and vanishes with the injection, beta is tonic drive.
            cur1 = cur1 * v1[:, :self.H]
            if self.film_mode == "gamma_beta":
                cur1 = cur1 + v1[:, self.H:]
        beta1 = torch.sigmoid(self.beta1_logit)
        mem1  = beta1 * mem1 + cur1
        # Comparison threshold only: the reset below subtracts the fixed
        # self.thresh, matching TimingGroupedSNN. Resetting by the offset
        # threshold instead lets a unit that learns a low threshold fire
        # easily AND reset by almost nothing, which runs away.
        th1  = self.thresh - self.v1 if self.bias_mode == "voltage" else self.thresh
        spk1 = spike_fn(mem1 - th1, self.slope)
        mem1 = (mem1 * (1.0 - spk1) if self.hidden_reset == "zero"
                else mem1 - self.thresh * spk1)

        # ---- layer 2 -------------------------------------------------
        cur2 = (torch.addmm(self.b2, spk1, self.w2) if cur_bias
                else spk1 @ self.w2)
        if self.sub_ln in ("l2", "both"):
            cur2 = self.ln2(cur2)
        if self.sub_film in ("l2", "both"):
            v2 = self.film2(gait)
            cur2 = cur2 * v2[:, :self.H]
            if self.film_mode == "gamma_beta":
                cur2 = cur2 + v2[:, self.H:]
        beta2 = torch.sigmoid(self.beta2_logit)
        mem2  = beta2 * mem2 + cur2
        th2  = self.thresh - self.v2 if self.bias_mode == "voltage" else self.thresh
        spk2 = spike_fn(mem2 - th2, self.slope)
        mem2 = (mem2 * (1.0 - spk2) if self.hidden_reset == "zero"
                else mem2 - self.thresh * spk2)

        # ---- analog readout -----------------------------------------
        curo  = (torch.addmm(self.b_read, spk2, self.w_read) if cur_bias
                 else spk2 @ self.w_read)
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
        sub-net g    (B, Hg) x2 + memo   driven by ITS OWN K timing units
        y            (B, n_joints)    group g writes its own columns only

    G and n_timing are INDEPENDENT. `group_cols[g]` says which gait-table
    columns sub-network g owns, and `timing_map[g]` says which K timing units
    feed it; both come from --decoder_shape / --timing_shape via
    `build_partition` and `build_timing_map`. K=1 with G == n_timing is the
    matched-shape case and is what this class did before those existed.

    Sub-networks stay fully disconnected from each other in either case: a
    shared timing unit (per_leg timing, per_joint decoders) is a shared INPUT,
    not a shared parameter, so the block-diagonal structure is untouched.

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

    NO cross talk between sub-networks.  Sub-network g sees only the K
    binary channels of the timing units that own its output columns.  Layers
    2+ are block diagonal, the readout is block diagonal, and there is no
    path between groups anywhere after the timing layer.  A timing unit shared
    by several sub-networks is a shared INPUT, not a shared parameter, so
    block diagonality holds either way.

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

    BIAS MODES
    ----------
    `bias_mode` decides how a neuron's excitability is parameterised.

      "current"  the classic form: an additive term in the input current.
      "voltage"  (default) a per-unit offset to the SPIKING THRESHOLD instead.
                 No bias current anywhere in the spiking path.
      "none"     no bias at all.

    Why voltage is the default. For an UNGATED layer with subtractive reset the
    two are EXACTLY equivalent -- substituting m' = mem - b/(1-beta) turns
    mem[t] = beta*mem[t-1] + w*s[t] + b into the unbiased recurrence with the
    spike test shifted to m' > thresh - b/(1-beta) (verified numerically:
    identical spike trains). So nothing is given up.

    Where they differ, voltage is preferable:
      - A bias CURRENT is tonic drive: it flows on every timestep, gate or no
        gate, so a sub-network keeps firing during timing-layer silence and
        the timing spikes stop being the only input. A threshold offset never
        touches the membrane, so it cannot manufacture activity. This is what
        makes NATURAL GATING possible (below) and is the main reason voltage
        is the default.
      - It also removes w1/b1 redundancy under explicit gating, where only
        their sum enters. Injection magnitude is what produces the jerk on
        each spike, so moving the bias out of the injection and into a
        persistent threshold attacks that directly.

    The RESET still subtracts the fixed `thresh`, not the offset threshold.
    Resetting by the offset would let a unit that learns a low threshold fire
    easily AND reset by almost nothing, which runs away; and it is not the
    variant that is equivalent to a bias current (measured 60 vs 75 spikes on
    the same input).

    b_read has no voltage equivalent and is dropped outside "current" mode:
    memo is analog and never spikes, so there is no threshold to offset. b_out
    still carries the output offset, and being added to y rather than to a
    membrane it is not a bias current in the sense that matters.

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

    Instead, the firing regime is set explicitly: `calibrate_gains` bisects a
    multiplier into each (gait, unit) column of `w_in_gait` before training so
    every gait of every unit starts inside a target spikes-per-cycle band.
    Calibrating per gait rather than once per unit matters: a single gain
    bisected against a unit's MINIMUM rate across gaits silently drove its
    loud gaits toward the ceiling of one spike per CPG spike, at which point
    the spike train is the OR of the CPG's and carries no phase at all. See
    `calibrate_gains`.

    NATURAL GATING
    --------------
    The operating setup. `gate_mode="none"` with `bias_mode="voltage"`: there
    is no explicit gate, but the sub-networks are nonetheless driven only by
    timing spikes, because in voltage mode layer 1's injection is
    `spk_t * w1` and there is no bias current to flow during silence. A
    positive membrane can then only decay toward zero, and `spike_fn` uses a
    strict `>`, so a unit with no input cannot cross threshold.

    Three things can break that and let a hidden unit fire on a timing-silent
    step.  All three are closed by the current defaults:

      bias current      closed by bias_mode="voltage" -- a bias CURRENT flows
                        every step, gate or no gate, so it is tonic drive; a
                        threshold offset never touches the membrane.
      film beta         closed by film_mode="gamma_only" -- FiLM's additive
                        term is applied every step regardless of spikes,
                        whereas gamma is multiplicative and vanishes wherever
                        the injection is zero. Gamma alone still carries full
                        per-gait conditioning, which is why this is preferable
                        to sub_film="none": w1/w2 have no gait axis, so FiLM
                        is layer 1's only per-gait handle.
      subtractive reset closed by hidden_reset="zero" -- under subtraction a
                        unit left above threshold keeps firing on subsequent
                        silent steps until the residual is spent.

    Each of the three costs something, and the reset is the expensive one:
    subtractive reset is what converts injection MAGNITUDE into a spike COUNT
    spread over the following steps, a real expansion of what layer 1 can
    represent, and it measured slightly better on RMSE.  The trade taken here
    is RMSE against a loss surface on which alignment is determined rather
    than lucky: with any of these leaks open the sub-networks receive some
    drive during timing silence, which makes the task loss partly insensitive
    to WHEN the timing spikes land, and alignment then comes out
    seed-dependent even while RMSE stays good.

    `--bias_mode current`, `--film_mode gamma_beta` and
    `--hidden_reset subtract` each reopen one leak, for A/B.

    `check_bias_voltage.py` decides, from a checkpoint alone, exactly which
    units can fire without a timing spike and by which mechanism.

    The known cost of relying on natural gating rather than the explicit gate:
    the gate multiplies `cur` and `spk` at every layer, which scales up the
    gradient reaching `spk_t`, so ungated runs get a weaker "put the spike
    here" signal. Improving on this probably needs a second-order synaptic
    filter so a single injection produces a smooth response rather than a
    sharp one; not implemented.

    Inside the sub-networks LN is optional (`sub_ln`) and uses
    elementwise_affine=False: a shared gamma/beta over (G, Hg) would be a
    parameter tied ACROSS groups, which breaks "fully disconnected".  FiLM
    follows and supplies per-group per-gait affine anyway.

    State (5 tensors, mixed rank):
        mem_timing (B, n_timing) -- the TIMING layer, T units
        since_upd  (B, G)        vestigial, never read; kept only to hold the
                                 exported ONNX signature fixed
        mem1, mem2 (B, G, Hg)
        memo       (B, G, H_o)   -- narrower; see the readout note in __init__
    """

    arch = "timing_grouped"
    # syn1/syn2/syno are the second-order synaptic currents. Allocated
    # unconditionally, like the alpha logits and like the vestigial
    # since_upd, so neither the state arity nor the exported ONNX signature
    # depends on --synaptic. They stay exactly zero when synaptic="none".
    state_names_in  = ("mem_timing_in", "since_upd_in",
                       "mem1_in",  "mem2_in",  "memo_in",
                       "syn1_in",  "syn2_in",  "syno_in")
    state_names_out = ("mem_timing_out", "since_upd_out",
                       "mem1_out", "mem2_out", "memo_out",
                       "syn1_out", "syn2_out", "syno_out")

    def __init__(self, hidden_per_group=128, n_gaits=4, max_gaits=16,
                 n_neurons=4, n_timing=N_LEGS, group_cols=None,
                 timing_map=None,
                 n_joints=N_JOINTS, readout_hidden=32,
                 tau_min=2.0, tau_max=256.0,
                 tau_timing_min=2.0, tau_timing_max=64.0,
                 tau_readout_max=40.0,
                 sub_ln="l2", sub_film="both", film_mode="gamma_only",
                 timing_reset="zero", hidden_reset="subtract", gate_mode="none",
                 bias_mode="voltage", readout_gait_bias=True,
                 synaptic="none", tau_syn_min=2.0, tau_syn_max=20.0,
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
        if film_mode not in ("gamma_only", "gamma_beta"):
            raise ValueError(f"film_mode must be gamma_only|gamma_beta, got "
                             f"{film_mode!r}")
        if synaptic not in ("none", "hidden", "all"):
            raise ValueError(f"synaptic must be none|hidden|all, got "
                             f"{synaptic!r}")

        group_cols = (build_group_cols(n_timing, n_joints=n_joints)
                      if group_cols is None else
                      [list(g) for g in group_cols])

        G = len(group_cols)
        T = int(n_timing)
        # timing_map[g] lists the timing units feeding sub-network g. The
        # None fallback is EXACTLY what this class did before timing_map
        # existed -- one timing unit per sub-network, in order -- so a config
        # predating the arg reconstructs as the model it actually was.
        if timing_map is None:
            if T != G:
                raise ValueError(
                    f"group_cols has {G} groups but n_timing={T}, and no "
                    f"timing_map was given. Without a map there is exactly "
                    f"one sub-network per timing neuron, so the two counts "
                    f"must match. Pass timing_map (see build_timing_map) for "
                    f"a fan-in other than 1.")
            timing_map = [[g] for g in range(G)]
        else:
            timing_map = [list(m) for m in timing_map]
            if len(timing_map) != G:
                raise ValueError(
                    f"timing_map has {len(timing_map)} entries but there are "
                    f"{G} sub-networks.")
        K = len(timing_map[0])
        if any(len(m) != K for m in timing_map):
            raise ValueError(
                f"timing_map fan-in must be uniform; got sizes "
                f"{sorted({len(m) for m in timing_map})}.")
        bad = [i for m in timing_map for i in m if not 0 <= i < T]
        if bad:
            raise ValueError(f"timing_map indexes units {sorted(set(bad))} "
                             f"outside 0..{T - 1}.")

        Hg  = int(hidden_per_group)
        Ho  = int(readout_hidden)
        C   = len(group_cols[0])            # output columns per group

        self.G          = G
        self.Hg         = Hg
        self.Ho         = Ho
        self.C          = C
        self.K          = K           # timing units feeding each sub-network
        self.n_timing   = T           # NOT G: the two are independent now
        self.timing_map_list = [list(m) for m in timing_map]
        self.register_buffer(
            "timing_map",
            torch.tensor(timing_map, dtype=torch.long).view(G, K),
            persistent=False)
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
        self.film_mode  = film_mode
        self.synaptic   = synaptic
        self.timing_reset = timing_reset
        self.hidden_reset = hidden_reset
        if gate_mode not in ("none", "decay"):
            raise ValueError(f"gate_mode must be none|decay, got "
                             f"{gate_mode!r}")
        self.gate_mode = gate_mode
        if bias_mode not in ("current", "voltage", "none"):
            raise ValueError(f"bias_mode must be current|voltage|none, got "
                             f"{bias_mode!r}")
        self.bias_mode = bias_mode
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
        # There is NO FiLM on this layer: its gamma would be provably
        # redundant, since with W already a free per-gait matrix,
        #     gamma_g * (x W_g + b) + beta_g  ==  x (W_g gamma_g) + (gamma_g b + beta_g)
        # so gamma is exactly absorbable into W_g and adds no expressiveness
        # (verified numerically to 1e-16).  A per-gait BIAS is not redundant,
        # and is supplied directly by the embedding below rather than hidden
        # inside a FiLM gate.  If the per-gait weight table is ever replaced
        # by a shared scheme, gamma becomes load-bearing and FiLM should come
        # back here.
        #
        # INIT SCALE is arbitrary: calibrate_gains rescales these columns to
        # hit a target firing rate before training starts, so whatever is set
        # here only fixes the relative SHAPE.  A positive mean matters though,
        # and is not cosmetic: calibration multiplies the current, and
        # multiplying a net-negative current by a larger positive gain moves it
        # FURTHER from threshold, so a unit born net-negative cannot be
        # rescued by calibration at all.  Sign is fixed here; magnitude is
        # calibration's job.
        self.w_in_gait = nn.Embedding(max_gaits, self.n_neurons * T)
        nn.init.normal_(self.w_in_gait.weight, mean=_W_IN_INIT, std=_W_IN_INIT)

        # Per-gait excitability.  See BIAS MODES in the class docstring.
        #   "current": an additive current, i.e. tonic drive on every step it
        #              is applied.  With timing_reset="zero" that is the only
        #              way a unit fires BETWEEN CPG bursts, so it is
        #              load-bearing rather than a liability here.
        #   "voltage": a per-gait threshold offset instead. Never touches the
        #              membrane, so it cannot manufacture a spike on a step
        #              where no input arrived.
        if bias_mode == "current":
            self.b_t = nn.Embedding(max_gaits, T)
            nn.init.zeros_(self.b_t.weight)
        elif bias_mode == "voltage":
            self.v_t = nn.Embedding(max_gaits, T)
            nn.init.zeros_(self.v_t.weight)
        self.beta_t_logit = nn.Parameter(
            init_beta_logit((T,), tau_timing_min, tau_timing_max))

        # ── sub-network layer 1: K binary spikes -> Hg units ──────
        # Fixed init scale, deliberately NOT calibrated.  A dead timing
        # neuron silences a whole sub-network, which is why that layer gets
        # calibration; a few dead units out of Hg here are noise, and w1 plus
        # film1's gamma are both learnable so the init only sets the
        # optimisation path, not what is reachable.
        #
        # Shape (G, K, Hg): K is the number of timing units feeding this
        # sub-network, so this is K independent (1 -> Hg) weight vectors per
        # group, contracted over K in `step`.  The leading G is the
        # sub-network index, not an input axis -- same role it plays in w2's
        # (G, Hg, Hg).
        #
        # The 1/sqrt(K) keeps the injection variance independent of fan-in:
        # with K=3 timing units able to fire on the same timestep, an
        # uncorrected init would deliver up to 3x the layer-1 current that
        # K=1 was tuned around, and layer 1 is not calibrated. At K=1 the
        # factor is exactly 1.0, so this is bit-identical to the old
        # (G, Hg) init for a matched-shape run.
        self.w1 = nn.Parameter(
            torch.randn(G, K, Hg) * (_W1_INIT / math.sqrt(K)))

        # ── sub-network layer 2: block diagonal (G, Hg, Hg) ───────
        self.w2 = nn.Parameter(torch.randn(G, Hg, Hg) / math.sqrt(Hg))

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
        self.w_out  = nn.Parameter(torch.randn(G, Ho, C) / math.sqrt(Ho))
        self.b_out  = nn.Parameter(torch.zeros(G, C))

        # Bias parameters, whichever form is in use.  b_read has no voltage
        # equivalent -- memo is ANALOG and never spikes, so there is no
        # threshold to offset -- so in voltage/none mode it is simply dropped
        # and b_out carries the output offset.  b_out is added to y, not to a
        # membrane, so it is not a "bias current" in the sense that matters.
        if bias_mode == "current":
            self.b1     = nn.Parameter(torch.zeros(G, Hg))
            self.b2     = nn.Parameter(torch.zeros(G, Hg))
            self.b_read = nn.Parameter(torch.zeros(G, Ho))
        elif bias_mode == "voltage":
            self.v1 = nn.Parameter(torch.zeros(G, Hg))
            self.v2 = nn.Parameter(torch.zeros(G, Hg))

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

        # ── per-gait readout bias CURRENT ─────────────────────────
        # A per-gait constant DC offset on the output, injected into `memo`
        # rather than added to `y`.
        #
        # WHY IT IS NEEDED.  w_read, w_out, b_out, b1, b2 and b_read all have
        # NO gait axis, so with film_mode="gamma_only" there is no per-gait
        # constant anywhere in the sub-networks: every gait relaxes toward the
        # same shared b_out between spikes.  Gaits that differ in MEAN joint
        # angle rather than in waveform shape -- turning gaits, where the left
        # and right legs sit at different average positions -- then have to
        # express that offset through the spike RATE, since gamma can only
        # scale the response to spikes.  That works (memo is an integrator, so
        # level is proportional to rate) but it couples three things: raising
        # the mean costs spikes, favours the positive-weight hidden units and
        # so shrinks that gait's downward range, and leaves visible ripple,
        # because `memo` LEAKS between spikes -- a level held at ~15 spk/cyc
        # with tau_o 60 sags ~12% between consecutive spikes.
        #
        # WHY INTO memo AND NOT ONTO y.  Added to `y` it would step
        # instantaneously when the gait index changes.  Injected as a current
        # it is filtered by the readout's own tau, so the offset fades in over
        # tau_o at a gait switch instead of jumping.
        #
        # WHY THIS DOES NOT REINTRODUCE THE PHASE DEGENERACY.  `memo` is
        # analog and never spikes, so this cannot manufacture spike activity
        # the way a bias current into a spiking layer can, and a CONSTANT
        # contributes no time-varying basis -- it cannot absorb a shift in
        # timing-spike phase the way tonic drive into mem1/mem2 could.  There
        # is also no surrogate gradient on this path, so it cannot create the
        # rate-dependent gradient that made film beta self-reinforcing.
        #
        # Created only when enabled, so a config written before this option
        # existed reconstructs with no such parameter and no missing key.
        self.readout_gait_bias = bool(readout_gait_bias)
        if self.readout_gait_bias:
            self.b_read_gait = nn.Embedding(max_gaits, G * Ho)
            nn.init.zeros_(self.b_read_gait.weight)

        # ── second-order synaptic filter (snnTorch `Synaptic`) ────
        # An extra decaying state between the injection and the membrane:
        #
        #     syn = alpha * syn + injection
        #     mem = beta  * mem + syn
        #
        # exactly snnTorch's Synaptic ordering (mem integrates the NEW syn).
        # This is also Loihi's native CUBA neuron, which carries a synaptic
        # current and a membrane voltage with separate decays -- so the
        # second-order form is the standard neuromorphic model rather than an
        # extension of it, and one instantaneous-injection LIF is the less
        # standard choice.
        #
        # WHAT IT BUYS.  A single injection is spread over many timesteps
        # instead of landing in one, which (a) reduces the size of the output
        # step at each spike and (b) keeps hidden membranes near threshold for
        # a WINDOW after a spike rather than a single step, so the surrogate
        # gradient stays alive on silent steps and firing rates become
        # learnable upward as well as downward. Both were things film beta was
        # doing via tonic drive, except the filter decays to zero during
        # genuine silence, so spike placement still matters.
        #
        # NOT a zero-slope start: because mem integrates the NEW syn, the
        # membrane still moves on the spike step itself. The step is smaller,
        # by the ratio of the filter's peak response to its initial one (~3.9x
        # at alpha = beta = 0.9), not zero. Making mem integrate the PREVIOUS
        # syn would give a true zero-slope start but would no longer match
        # snnTorch.
        #
        # alpha logits are always allocated so the parameter set does not
        # depend on the mode, matching how beta logits are handled; they
        # receive no gradient when synaptic="none".
        self.alpha1_logit = nn.Parameter(
            init_beta_logit((G, Hg), tau_syn_min, tau_syn_max))
        self.alpha2_logit = nn.Parameter(
            init_beta_logit((G, Hg), tau_syn_min, tau_syn_max))
        self.alphao_logit = nn.Parameter(
            init_beta_logit((G, Ho), tau_syn_min, tau_syn_max))

        # ---------------------------------------------------------------
    @torch.no_grad()
    def clamp_bias_voltage(self):
        """
        Project v_t / v1 / v2 into [-thresh, thresh - _V_EPS].

        The UPPER bound is the load-bearing one.  A voltage-mode unit
        compares against `thresh - v`, so v >= thresh makes the effective
        threshold non-positive, and such a unit fires on EVERY timestep
        with no way back: reset-to-zero lands at 0, which is still above a
        non-positive threshold, and subtractive reset is the same failure
        one step slower.  It stops carrying information, and in an ungated
        run it spikes on timesteps where its timing neuron did not, which
        is the opposite of what the layer is for.

        The LOWER bound is mild by comparison: it caps the effective
        threshold at 2*thresh so a unit cannot be driven permanently silent
        by the bias alone.

        Clamped on `.data` after the optimiser step, NOT at the use site in
        `step()`: `(thresh - v.clamp(...))` gives a parameter sitting
        outside the range exactly zero gradient, so it could never come
        back.  Clamping the data leaves the true gradient of the unclamped
        expression, which points inward.  This is projected gradient
        descent, and it is a no-op on any run that never left the range.

        Deliberately NOT called from build_model_from_cfg / load_run: a
        trained checkpoint must reconstruct as whatever it actually was,
        out-of-range biases included.  Use check_bias_voltage.py to find
        those.
        """
        if self.bias_mode != "voltage":
            return
        lo, hi = -self.thresh, self.thresh - _V_EPS
        for p in (self.v_t.weight, self.v1, self.v2):
            p.data.clamp_(lo, hi)

    # ---------------------------------------------------------------
    def init_state(self, batch, device, dtype=torch.float32):
        z = lambda w: torch.zeros(batch, self.G, w, device=device, dtype=dtype)
        mem_t = torch.zeros(batch, self.n_timing, device=device, dtype=dtype)
        # since_upd: vestigial. It was the update counter for the removed
        # gate_mode="freeze"; nothing reads it now. Kept, and kept G-wide,
        # only so the exported ONNX signature keeps its arity and names.
        since = torch.zeros(batch, self.G, device=device, dtype=dtype)
        # memo is Ho wide, not Hg (see the readout comment in __init__).
        # syn1/syn2/syno mirror mem1/mem2/memo: same shapes, one per layer.
        return (mem_t, since, z(self.Hg), z(self.Hg), z(self.Ho),
                z(self.Hg), z(self.Hg), z(self.Ho))

    # ---------------------------------------------------------------
    def _timing(self, x, gait, mem_t):
        """
        One timestep of the timing layer.  x (B, n_neurons) -> spk_t (B, T),
        T = n_timing.  Independent of the number of sub-networks: `step` maps
        these spikes onto sub-networks through `timing_map`.

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
        W   = self.w_in_gait(gait).view(-1, self.n_neurons, self.n_timing)
        cur = torch.bmm(x.unsqueeze(1), W).squeeze(1)
        if self.bias_mode == "current":
            cur = cur + self.b_t(gait)
        # Comparison threshold only; the RESET below still subtracts the fixed
        # self.thresh. Resetting by the offset threshold instead would let a
        # unit that learns a low threshold fire easily AND reset by almost
        # nothing, which runs away.
        th = (self.thresh - self.v_t(gait) if self.bias_mode == "voltage"
              else self.thresh)
        mem_t = torch.sigmoid(self.beta_t_logit) * mem_t + cur
        spk_t = spike_fn(mem_t - th, self.timing_slope)
        if self.timing_reset == "zero":
            mem_t = mem_t * (1.0 - spk_t)
        else:
            mem_t = mem_t - self.thresh * spk_t
        return spk_t, mem_t

    def step(self, x, gait, state):
        """
        x     : (B, n_neurons) float — CPG spikes this timestep
        gait  : (B,) int64
        state : (mem_timing, since_upd, mem1, mem2, memo,
                 syn1, syn2, syno)

        Returns (y, state, aux) where aux = (spk_timing,).
        `aux` exists so the spike-statistics penalty can see the spikes without
        a second forward pass; the ONNX wrappers unpack and discard it, and
        since spk_timing already feeds `y` it is in the graph regardless.  Kept
        as a 1-tuple so callers iterate over it uniformly.
        """
        mem_t, since, mem1, mem2, memo, syn1, syn2, syno = state
        G, Hg = self.G, self.Hg

        spk_t, mem_t = self._timing(x, gait, mem_t)          # (B, T)

        # ---- timing -> sub-network routing -------------------------
        # timing_map is (G, K), so this gathers each sub-network's own K
        # timing spikes. At K=1 with a matched-shape map this is exactly
        # spk_t.unsqueeze(-1), i.e. the old behaviour.
        spk_sel = spk_t[:, self.timing_map]                  # (B, G, K)

        # ---- event gate -------------------------------------------
        # gate_mode:
        #   "none"   (default) ungated. Sub-net spikes fire freely and
        #            membranes decay every step. This is the NATURAL GATING
        #            setup -- see the class docstring. Under bias_mode
        #            "voltage" the sub-networks receive current only on a
        #            timing spike anyway, so most hidden units cannot fire
        #            without one and the explicit gate has little left to do.
        #   "decay"  input and sub-net spikes explicitly gated; membranes
        #            still leak every step. Kept for A/B: it forces the
        #            spike-train-only property rather than relying on it. Not
        #            the default -- it destroys the magnitude-to-spike-count
        #            expansion that subtractive reset provides (a gated unit
        #            can emit at most one spike per timing spike), and it
        #            reshapes the gradient into the timing layer in a way that
        #            is NOT simply "stronger" (see the NOTE).
        #
        # NOTE on gradients.  Under the fully naturally-gated defaults the two
        # modes have an IDENTICAL forward pass -- gate is 0/1 and everything it
        # multiplies is already zero wherever gate is 0 -- but they do NOT
        # train the same, because `gate` is spk_t and carries its surrogate
        # gradient, so gating makes spk_t appear multiplicatively five extra
        # times (cur1, spk1, cur2, spk2, curo).  Treating spk_t as the
        # continuous variable the surrogate pretends it is, a path with k gate
        # factors goes as s^(1+k), whose derivative is (1+k)*s^k.  So gating:
        #
        #   on a SPIKING step  multiplies the gradient by (1+k), up to 6x --
        #                      amplifying "this spike was badly placed"
        #   on a SILENT step   makes it EXACTLY ZERO, because both factors of
        #                      the product vanish there, whereas ungated
        #                      d(cur1)/d(spk_t) = w1 at every step
        #
        # That second line is the one that matters.  The silent-step term is
        # the only route by which "this unit should have fired HERE" can reach
        # the timing layer, and gating removes it outright.  So "decay" biases
        # spike counts downward twice over, and dead timing units should be
        # MORE likely under it than under full natural gating, not less.
        gated = self.gate_mode != "none"
        # A sub-network updates when ANY of its K timing units fires, so the
        # gate is the OR over that group's selected spikes. At K=1 amax is a
        # no-op and this is spk_t.unsqueeze(-1) exactly. Using amax rather
        # than a sum keeps the gate strictly 0/1, which the forward relies on
        # (gate*gate == gate).
        gate = spk_sel.amax(dim=-1, keepdim=True) if gated else None
        dec1 = torch.sigmoid(self.beta1_logit)
        dec2 = torch.sigmoid(self.beta2_logit)
        deco = torch.sigmoid(self.betao_logit)

        # Injection -> membrane, either first order (inject straight into the
        # membrane) or second order (through a decaying synaptic current).
        # Second order is snnTorch's `Synaptic`, whose ordering is
        # syn = alpha*syn + cur  THEN  mem = beta*mem + syn, i.e. the membrane
        # integrates the NEW syn. Written once here so the three layers cannot
        # drift apart. Returns (mem, syn); syn is passed through unchanged in
        # first-order mode so the state slot stays exactly zero.
        def integrate(mem, syn, cur, dec, alpha_logit, second_order):
            if not second_order:
                return dec * mem + cur, syn
            syn = torch.sigmoid(alpha_logit) * syn + cur
            return dec * mem + syn, syn

        syn_hidden  = self.synaptic in ("hidden", "all")
        syn_readout = self.synaptic == "all"

        # ---- sub-net layer 1 ---------------------------------------
        # In "current" mode and gated, w1 and b1 are redundant (only their sum
        # enters). In "voltage" mode the bias has moved out of the injection
        # entirely, which is the point: injection magnitude is what drives the
        # jerk, and a threshold offset does not inflate it.
        #
        # The contraction over K is done explicitly in both gate modes. Under
        # gating it is tempting to write cur1 = w1 + b1 and let the gate
        # reintroduce the spike, but that is only valid at K=1: with K>1 the
        # injection is sum_k spk_k * w1[g,k], which is NOT gate * sum_k
        # w1[g,k] unless all K units fire together. The gate below is then a
        # no-op on spiking steps (gate=1 wherever any spk_sel is 1).
        b1 = self.b1 if self.bias_mode == "current" else 0.0
        cur1 = torch.einsum("bgk,gkh->bgh", spk_sel, self.w1) + b1
        if self.sub_ln in ("l1", "both"):
            cur1 = self.ln1(cur1)
        if self.sub_film in ("l1", "both"):
            v1 = self.film1(gait).view(-1, 2, G, Hg)
            # gamma always; beta only in "gamma_beta". Gamma is multiplicative
            # so it vanishes wherever the injection is zero, which is what
            # keeps the layer naturally gated; beta is added on EVERY step
            # regardless of spikes and is therefore tonic drive. The beta half
            # of the table stays allocated either way so checkpoint shapes do
            # not depend on the mode -- it simply receives no gradient in
            # "gamma_only".
            cur1 = cur1 * v1[:, 0]
            if self.film_mode == "gamma_beta":
                cur1 = cur1 + v1[:, 1]
        if gated:
            cur1 = gate * cur1
        mem1, syn1 = integrate(mem1, syn1, cur1, dec1,
                               self.alpha1_logit, syn_hidden)
        th1 = (self.thresh - self.v1 if self.bias_mode == "voltage"
               else self.thresh)
        spk1 = spike_fn(mem1 - th1, self.slope)
        if gated:
            spk1 = gate * spk1
        if self.hidden_reset == "zero":
            mem1 = mem1 * (1.0 - spk1)
        else:
            mem1 = mem1 - self.thresh * spk1

        # ---- sub-net layer 2: block diagonal ------------------------
        cur2 = torch.einsum("bgh,ghk->bgk", spk1, self.w2)
        if self.bias_mode == "current":
            cur2 = cur2 + self.b2
        if self.sub_ln in ("l2", "both"):
            cur2 = self.ln2(cur2)
        if self.sub_film in ("l2", "both"):
            v2 = self.film2(gait).view(-1, 2, G, Hg)
            cur2 = cur2 * v2[:, 0]
            if self.film_mode == "gamma_beta":
                cur2 = cur2 + v2[:, 1]
        if gated:
            cur2 = gate * cur2
        mem2, syn2 = integrate(mem2, syn2, cur2, dec2,
                               self.alpha2_logit, syn_hidden)
        th2 = (self.thresh - self.v2 if self.bias_mode == "voltage"
               else self.thresh)
        spk2 = spike_fn(mem2 - th2, self.slope)
        if gated:
            spk2 = gate * spk2
        if self.hidden_reset == "zero":
            mem2 = mem2 * (1.0 - spk2)
        else:
            mem2 = mem2 - self.thresh * spk2

        # ---- block-diagonal analog readout -------------------------
        curo = torch.einsum("bgh,ghk->bgk", spk2, self.w_read)
        if self.bias_mode == "current":
            curo = curo + self.b_read
        if gated:
            curo = gate * curo
        if self.readout_gait_bias:
            # AFTER the gate, deliberately: this is a DC offset, not an
            # injection. Gating it would make the offset itself appear only on
            # spike steps, which is the opposite of the point -- see the
            # b_read_gait comment in __init__.
            curo = curo + self.b_read_gait(gait).view(-1, G, self.Ho)
        memo, syno = integrate(memo, syno, curo, deco,
                               self.alphao_logit, syn_readout)

        # `since` is passed through untouched. It was the update counter for
        # the removed gate_mode="freeze"; the slot is retained only so the
        # exported ONNX signature keeps its input/output arity and names, and
        # nothing reads it. Remove it from state_names_in/out if the signature
        # is ever allowed to change.

        y_grp = torch.einsum("bgh,ghc->bgc", memo, self.w_out) + self.b_out
        y     = y_grp.flatten(1).index_select(1, self.out_perm)
        return (y, (mem_t, since, mem1, mem2, memo, syn1, syn2, syno),
                (spk_t,))

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
            # n_timing, NOT G: this is the timing layer's own membrane, and
            # the two counts differ whenever the shapes differ.
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
        n = lambda *ps: int(sum(p.numel() for p in ps
                                 if p is not None))
        g = lambda name: getattr(self, name, None)
        return {
            "timing":  n(self.w_in_gait.weight, self.beta_t_logit,
                         *(w.weight for w in (g("b_t"), g("v_t"))
                           if w is not None)),
            "sub_l1":  n(self.w1, self.beta1_logit, self.alpha1_logit,
                         g("b1"), g("v1")),
            "sub_l2":  n(self.w2, self.beta2_logit, self.alpha2_logit,
                         g("b2"), g("v2")),
            "readout": n(self.w_read, self.w_out, self.b_out,
                         self.betao_logit, self.alphao_logit, g("b_read"),
                         *( [self.b_read_gait.weight]
                            if self.readout_gait_bias else [] )),
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
    As SingleStepONNX but for TimingGroupedSNN's 8-tensor state.

    Written as an explicit signature rather than *state: torch.onnx.export
    traces varargs unreliably, and the input names have to line up
    positionally with `model.state_names_in` anyway.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spikes, gait, mem_timing, since_upd, mem1, mem2, memo,
                syn1, syn2, syno):
        y, (mt, su, m1, m2, mo, s1, s2, so), _ = self.model.step(
            spikes, gait, (mem_timing, since_upd, mem1, mem2, memo,
                           syn1, syn2, syno))
        return y, mt, su, m1, m2, mo, s1, s2, so


# ═══════════════════════════════════════════════════════════════════
# 7a.  Keeping the small spiking layers alive
# ═══════════════════════════════════════════════════════════════════
#
# The router and timing layers have no LayerNorm (see TimingGroupedSNN's
# docstring for why), so nothing automatically holds their input current in
# the range where a threshold-1.0 LIF actually fires.  Three mechanisms,
# acting at different times:
#
#   calibrate_gains      once, before training  -- never start dead OR loud
#   SpikeObjective       every gradient step    -- shape the spike train
#   reinit_timing_units  on detection           -- rescue what failed anyway
#
# Both extremes are fatal and neither self-corrects.  A dead timing unit is
# not merely a wasted unit: the `w1` slots it feeds have gradient exactly
# proportional to its spike output, so while it is silent those slots receive
# EXACTLY zero gradient and stay at random init while everything downstream
# trains around them -- and if the unit revives late, the matrix consuming its
# input has learned nothing and the LR has already decayed.  A SATURATED unit
# fires on every CPG spike, so its spike train is the OR of the CPG's and
# carries no phase at all, and it sits far above threshold where spike_fn's
# surrogate is down 1e2-1e3x, so the gradient reaching it is ~1e-6 and Adam's
# scale-invariance means raising --spike_stats_lambda cannot compensate.
#
# Hence the division of labour: calibration prevents both at init,
# --min_count_floor keeps the spike penalty from pruning into silence, and
# reinit_timing_units repairs per (gait, unit) whatever still fails.


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
                    per_gait=True, cpg_rate_hint=None, verbose=True):
    """
    Scale each timing unit's CPG->timing weight column so its firing rate
    starts inside [lo, hi] spikes per cycle.

    Set by measurement rather than by a user-supplied scale factor.  Writes
    directly into `w_in_gait`.

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
    has.  (The per-gait bias is still free to LEARN tonic drive if the task
    wants it; the point is only that calibration should not be the thing that
    sets it.)

    WHY IT CAN STILL FAIL.  Gain cannot fix a wrong sign.  If a unit's net
    input current is negative, multiplying by a larger positive gain moves it
    FURTHER from threshold, and the bisection rides to g_hi and gives up.  The
    positive-mean init on `w_in_gait` exists to prevent that.  Units that
    cannot be brought into band are reported, not silently left.

    WHICH GAIT.  `per_gait=True` (the default) bisects a SEPARATE gain for
    every (gait, unit) pair, so every pair lands in the band.  `per_gait=False`
    is the old behaviour: one gain per unit, applied to that unit's column in
    every gait, bisected against the MINIMUM rate across gaits.

    Per-gait is the default because one-gain-per-unit is a saturation
    generator.  Only the minimum was constrained, nothing bounded the maximum,
    so a unit weakly driven in ONE gait had its gain raised until that gait
    fired -- dragging every other gait up with it.  Measured over 24 draws of
    the real init (w ~ N(_W_IN_INIT, _W_IN_INIT), 6 CPG inputs, 4 gaits), the
    min-over-gaits gain left max-over-gaits rates of 5 to 48 spk/cyc against a
    target band of [1, 5]: typically 3-10x over, and 20-48x when one gait's
    column happened to sum near zero.

    That inflation is not recoverable by training.  A unit pinned at the
    ceiling (n_cpg * cpg_rate spikes per cycle, one per CPG spike) sits far
    above threshold, where spike_fn's surrogate 1/(slope*|x|+1)^2 is down by
    1e2 to 1e3, so the spike penalty's gradient is order 1e-6.  Adam is
    scale-invariant -- lr * m/sqrt(v) is unchanged if every gradient is
    multiplied by a constant -- so raising --spike_stats_lambda does NOT speed
    the descent up: a 10,000x increase moved the gain 1.5% in a 4000-step
    simulation, leaving the rate at 39 spk/cyc.  The gain has to travel ~40x
    multiplicatively, and gradient descent will not walk that far.  Hence:
    start in the band, per gait, rather than trying to get back to it.

    The old behaviour justified one gain per unit as preserving "the relative
    shape across gaits".  There is no such shape at init -- `w_in_gait` is a
    single `normal_` over the whole (max_gaits, n_cpg * n_timing) tensor, so
    every gait's matrix is already independent -- and per-gait gains cost no
    extra forward passes, because `measure_rates` measures every gait on every
    call either way.

    Rows of `w_in_gait` beyond `n_gaits` are left unscaled (gain 1.0) in
    per-gait mode: they are never indexed during this run, so there is no rate
    to measure for them.

    `cpg_rate_hint` is the CPG's own spikes-per-cycle from `cpg_spike_stats`.
    Only used to print the saturation warning: the ceiling a timing unit can
    reach is n_cpg * cpg_rate, one spike per CPG spike, because the BLIF CPG
    never fires two neurons in the same timestep. Passed in rather than
    measured again so there is one source of truth for that number.
    """
    if not hasattr(model, "w_in_gait"):
        return {}

    # G here is the TIMING-unit count, which is model.n_timing and is no
    # longer the same as model.G (the sub-network count) -- w_in_gait is
    # indexed by timing unit. Named G only because the bisection below is
    # written against it.
    MG, n_cpg, G = model.max_gaits, model.n_neurons, model.n_timing
    # (max_gaits, n_cpg, n_timing) view; column l is timing unit l's inputs.
    W = model.w_in_gait.weight.data.view(MG, n_cpg, G)
    base = W.clone()

    # Bisection brackets. Per-gait mode carries one bracket per (gait, unit);
    # the old mode carries one per unit. Everything below is written against
    # `shape` so the two paths share a single bisection loop.
    shape = (n_gaits, G) if per_gait else (G,)
    lo_g = torch.full(shape, g_lo, device=device)
    hi_g = torch.full(shape, g_hi, device=device)

    def rates_at(gain):
        """
        Write `gain` into w_in_gait and measure. Returns a tensor shaped like
        `gain`, so the comparisons against lo/hi are elementwise in both modes.
        """
        if per_gait:
            # (n_gaits, G) -> (max_gaits, 1, G). Rows past n_gaits are never
            # indexed this run and have no measured rate, so they stay at 1.
            full = torch.ones(MG, G, device=device, dtype=base.dtype)
            full[:n_gaits] = gain
            W.copy_(base * full.unsqueeze(1))
        else:
            # gain is (G,); broadcast over (max_gaits, n_cpg, G) so each timing
            # unit's whole column is scaled in every gait at once.
            W.copy_(base * gain.view(1, 1, G))
        r = torch.as_tensor(
            measure_rates(model, spikes, n_gaits, device, period=period),
            device=device)                                  # (n_gaits, G)
        return r if per_gait else r.min(dim=0).values

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

    # `rate_by_gait` is always the full (n_gaits, G) measurement, whichever
    # mode ran, so the two modes stay directly comparable in the config dump.
    # In per-gait mode that IS final_rates; in the old mode final_rates is
    # only the per-unit minimum, so re-read the full matrix.
    rate_by_gait = final_rates if per_gait else torch.as_tensor(
        measure_rates(model, spikes, n_gaits, device, period=period),
        device=device)

    tolist = lambda t: t.cpu().tolist()
    report = {"timing": {
        "per_gait":     bool(per_gait),
        "band":         [float(lo), float(hi)],
        "gain":         tolist(final_gain),      # (n_gaits, G) or (G,) by mode
        "rate_by_gait": tolist(rate_by_gait),    # always (n_gaits, G)
        # Kept under their original names and meanings so anything reading an
        # older config still finds them; max_rate is the number the old mode
        # never constrained and is the one to watch.
        "min_rate":     tolist(rate_by_gait.min(dim=0).values),
        "max_rate":     tolist(rate_by_gait.max(dim=0).values),
        "in_band":      tolist(ok),
    }}

    if verbose:
        fmt  = lambda v: " ".join(f"{x:6.2f}" for x in v)
        tim  = report["timing"]
        mode = "per (gait, unit)" if per_gait else "per unit, min over gaits"
        print(f"      calibrated timing [{mode}], band [{lo}, {hi}] spk/cyc")
        if per_gait:
            for g in range(n_gaits):
                print(f"        gait {g}: gain [{fmt(tim['gain'][g])}]")
                print(f"                 rate [{fmt(tim['rate_by_gait'][g])}]")
        else:
            print(f"        gain     [{fmt(tim['gain'])}]")
        print(f"        min spk/cyc across gaits [{fmt(tim['min_rate'])}]")
        print(f"        MAX spk/cyc across gaits [{fmt(tim['max_rate'])}]")

        # A unit at the ceiling fires once per CPG spike, so its spike train is
        # the OR of the CPG's and carries no phase information at all. Flagged
        # separately from the band check because it is the failure that kills a
        # sub-network outright, and because training cannot undo it.
        #
        # Threshold is _SAT_FRAC, the same constant the band cap uses, NOT
        # something looser like 0.9. Measured: min-over-gaits calibration on
        # the ungated band [1.5, 2.0] x cpg_rate left a (gait, unit) pair at
        # 51.5 of a 60 ceiling -- 86%, thoroughly phase-blind, and a 0.9
        # threshold would have said nothing about it.
        ceiling = float(n_cpg) * float(cpg_rate_hint) if cpg_rate_hint else None
        if ceiling:
            sat = (rate_by_gait >= _SAT_FRAC * ceiling).nonzero().cpu().tolist()
            if sat:
                print(f"      WARNING: (gait, unit) {sat} are at >="
                      f"{_SAT_FRAC:g} x the {ceiling:.0f} spk/cyc ceiling "
                      f"(one spike per CPG spike), so they carry little or no "
                      f"phase. Training will not fix this — the spike "
                      f"penalty's gradient is ~1e-6 there and Adam is "
                      f"scale-invariant.")

        bad = ok.logical_not().nonzero().cpu().tolist()
        if bad:
            label = "(gait, unit)" if per_gait else "unit"
            print(f"      WARNING: timing {label} {bad} could not be brought "
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
                 target_rate=None, target_R=None,
                 min_count_floor=0.0, floor_weight=1.0):
        self.lam        = float(lam)
        self.period     = float(period)
        self.n_gaits    = int(n_gaits)
        self.target_rate = target_rate     # CPG spikes per cycle
        self.target_R    = target_R        # CPG circular concentration
        self.min_count_floor = float(min_count_floor)   # spikes/cycle
        self.floor_weight    = float(floor_weight)

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

    KNOWN LIMITATION: R uses the fundamental only, so two
    bursts at opposite phases cancel to R ~ 0 and are punished as hard as
    spikes smeared uniformly.  Correct for gaits where each leg swings once
    per cycle; wrong for a genuinely two-burst gait.

    NOT COMPATIBLE with gated sub-networks: this objective wants every
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
    Spend as few spikes as possible.  Pay a flat cost per spike BEYOND a free
    allowance and let the TASK loss decide where the rest are worth spending.

    Needs the output to depend on the spike train alone to mean anything --
    whether by explicit gating or by natural gating, see NATURAL GATING in
    TimingGroupedSNN's docstring.  Given that, it is the whole idea: the task
    loss already forces spikes wherever the waveform must move, so a uniform
    per-spike cost on top makes the cheapest solution place spikes densely
    where the target changes fast and sparsely where it creeps -- an adaptive
    sampling clock, derived from the data rather than supervised.  Nothing has
    to be told which phase is "swing".

    LINEAR in the rate, not squared, on purpose: a flat marginal cost per
    spike is the L1-sparsity form and drives genuine sparsity, whereas a
    squared cost pushes hard at high rates and then gives up as the rate
    falls, which is the opposite of what is wanted.

    FLOOR (`min_count_floor`, in spikes per CYCLE).  The penalty is a
    two-sided hinge around it:

        penalty = lam * (       relu(duty - floor/period)
                  + floor_weight * relu(floor/period - duty) )

    ABOVE the floor this is the L1 sparsity term, marginal cost lam per spike.
    BELOW it the second hinge takes over with the OPPOSITE sign of gradient,
    so descent increases the rate -- a reward for spiking, which is what stops
    a unit being pruned into silence.

    WHY A HINGE AND NOT A LINEAR FUNCTION.  Writing the term as
    lam*(duty - floor/period) looks like it should reward spiking below the
    floor, and it does make the penalty negative there -- but a linear
    function has a CONSTANT derivative, so subtracting the floor changes the
    penalty's value and not its gradient.  lam*(duty - floor) and lam*duty are
    gradient-identical at every rate, i.e. the linear version is exactly the
    un-floored penalty.  The nonlinearity is the whole mechanism; only a kink
    at the floor can make the gradient change sign there.

    WHY THIS IS NEEDED AT ALL.  With no floor the marginal cost per spike is
    identical at 40 spk/cyc and at 1, so the penalty keeps pushing all the way
    to zero, and a unit that reaches silence is UNRECOVERABLE through the task
    loss: it passes exactly zero gradient to the w1 slots it feeds, so the
    task can no longer object.  Observed directly -- a run pruned several
    sub-networks to an excellent 2-5 well-placed spikes per cycle and pruned a
    handful of others to 0.  The floor term does still reach a silent unit,
    because it acts on the timing membrane through spike_fn's surrogate, which
    is small but non-zero below threshold (~0.03 at timing_slope=5), rather
    than through the sub-network like the task loss.

    WHAT IT COSTS.  A two-sided hinge makes the floor an ATTRACTOR, not just a
    floor: a unit the task would rather run at 1 spk/cyc gets pushed to the
    floor.  That is a real departure from "spend as few spikes as possible",
    which is why the floor should be set to the fewest spikes a sub-network
    can do anything with, not to a comfortable number.  `floor_weight=0`
    recovers the pure one-sided version (pruning stops at the floor, but
    nothing pushes back up), and `min_count_floor=0` recovers the original
    un-floored L1 exactly.

    Note this does NOT address the opposite failure.  A unit that ends up
    saturated sits far above threshold where spike_fn's surrogate is down
    1e2-1e3x, so the penalty's gradient there is ~1e-6, and Adam's
    scale-invariance means raising lam cannot compensate.  Saturation is
    handled by `reinit_timing_units`, which scales the offending column down.

    Other mitigations: `--spike_lambda_warmup` ramps lam from 0 so the network
    learns to use spikes before being charged for them, and
    `--reinit_dead_after` repairs (gait, unit) pairs that fail anyway.
    """

    name = "min_count"

    def penalty(self, spk, gait, phase, mask):
        rate, present, cnt, oh = self._grouped(spk, gait, mask)
        # rate is per-TIMESTEP duty; the floor is per CYCLE, so convert.
        floor_duty = self.min_count_floor / self.period
        over  = torch.relu(rate - floor_duty)
        under = torch.relu(floor_duty - rate)
        term  = over + self.floor_weight * under
        denom = present.sum().clamp(min=1.0) * spk.shape[2]
        return self.lam * (term * present).sum() / denom

    def describe(self):
        return (f"min_count (lam={self.lam:g}, linear/L1 in duty cycle above "
                f"a floor of {self.min_count_floor:g} spk/cyc, "
                f"floor_weight={self.floor_weight:g})")


SPIKE_OBJECTIVES = {cls.name: cls for cls in (
    NoSpikeObjective, CPGMatchSpikeObjective, MinCountSpikeObjective)}


def make_spike_objective(name, **ctx):
    if name not in SPIKE_OBJECTIVES:
        raise ValueError(f"Unknown --spike_objective {name!r}; "
                         f"available: {sorted(SPIKE_OBJECTIVES)}")
    return SPIKE_OBJECTIVES[name](**ctx)


@torch.no_grad()
def reinit_timing_units(model, dead_pairs, sat_pairs, n_gaits,
                        generator=None, sat_scale=0.5, verbose=True):
    """
    Rescue timing units that have failed, PER (gait, unit).

    `dead_pairs` / `sat_pairs` : lists of (gait, unit) index tuples.

    Per-gait is the right granularity because the CPG->timing weights already
    are: `w_in_gait` is (max_gaits, n_cpg, n_timing), so gait g's column for
    unit u is an independent slice and fixing it cannot disturb any other
    gait, any other unit, or anything downstream.  The previous version only
    acted when a unit was silent for EVERY gait, which missed the common case
    of one unit failing for one gait.

    The two failures get OPPOSITE treatment, because they are not the same
    illness:

      DEAD (0 spk/cyc)  Re-roll that (gait, unit) column of w_in_gait with the
                        positive mean restored, and zero its per-gait bias.  A
                        silent column carries no information, so its direction
                        is worthless and there is nothing to preserve.

      SATURATED         SCALE that column down by `sat_scale` instead.  A
                        saturated unit is firing on every CPG spike, so its
                        spike train is the OR of the CPG's and carries no
                        phase -- but that is a MAGNITUDE failure, not a
                        direction one: whatever phase preference the column
                        encodes is still in there, just swamped.  Re-rolling
                        would throw that away to fix a scale problem.  Rate is
                        monotone in the column's gain, so halving strictly
                        reduces it; a badly saturated unit may need several
                        triggers to walk down, which is fine and self-limiting
                        since it stops as soon as the unit leaves the band.

    Neither case can be left to training.  A silent unit passes exactly zero
    gradient to the w1 slots it feeds, and a saturated one sits far above
    threshold where spike_fn's surrogate is down 1e2-1e3x, so the gradient
    reaching it is ~1e-6 and Adam's scale-invariance means no amount of
    --spike_stats_lambda compensates.

    `w1` IS re-rolled, but only for units dead in EVERY gait.  The
    justification is specific to that case: d(cur1)/d(w1) is proportional to
    the timing spike, so a unit silent in all gaits left its w1 slots at
    random init for the whole dead period while the rest of the sub-network
    trained around them.  A unit dead in only some gaits still trained those
    slots on the gaits where it fired, so re-rolling would destroy working
    weights.

    `w1` is indexed by SUB-NETWORK, not by timing unit, so the slots re-rolled
    are the (g, k) pairs where timing_map[g][k] == u -- possibly several
    sub-networks (per_leg timing with per_joint decoders) or one slot of
    several in one sub-network (per_joint timing with per_leg decoders).  The
    other K-1 slots of a shared sub-network were driven by live units and are
    left alone.

    Deliberately does NOT touch w2/w_read/w_out for any group: those DID
    train and may hold something useful about the group's output range.
    """
    if not (dead_pairs or sat_pairs) or not hasattr(model, "w_in_gait"):
        return
    dev = model.w_in_gait.weight.device
    MG, nn_, T = model.max_gaits, model.n_neurons, model.n_timing
    W = model.w_in_gait.weight.view(MG, nn_, T)

    for g, u in dead_pairs:
        W[g, :, u] = (torch.randn(nn_, generator=generator).to(dev)
                      * _W_IN_INIT + _W_IN_INIT)
        # Per-gait excitability back to neutral, in whichever form is in use.
        # Guarded because "voltage" mode has v_t and no b_t at all, and
        # "none" mode has neither.
        for name in ("b_t", "v_t"):
            emb = getattr(model, name, None)
            if emb is not None:
                emb.weight[g, u] = 0.0

    for g, u in sat_pairs:
        W[g, :, u] *= sat_scale

    # w1 only for units dead in every gait -- see the docstring.
    dead_by_unit = {}
    for g, u in dead_pairs:
        dead_by_unit.setdefault(u, set()).add(g)
    everywhere = sorted(u for u, gs in dead_by_unit.items()
                        if len(gs) >= n_gaits)
    slots = {}
    if everywhere:
        K = model.K
        scale = _W1_INIT / math.sqrt(K)      # same as the constructor's init
        for u in everywhere:
            hit = [(g, k) for g, m in enumerate(model.timing_map_list)
                   for k, t in enumerate(m) if t == u]
            for g, k in hit:
                model.w1[g, k] = (torch.randn(model.Hg, generator=generator)
                                  .to(dev) * scale)
            slots[u] = hit

    if verbose:
        if dead_pairs:
            print(f"      [reinit] DEAD (gait, unit) {sorted(dead_pairs)}: "
                  f"re-rolled that gait's w_in_gait column and zeroed its "
                  f"per-gait bias")
        if everywhere:
            print(f"      [reinit] unit(s) {everywhere} were dead for ALL "
                  f"{n_gaits} gaits, so sub-net w1 slot(s) {slots} were "
                  f"re-rolled too (they had received zero gradient throughout)")
        if sat_pairs:
            print(f"      [reinit] SATURATED (gait, unit) "
                  f"{sorted(sat_pairs)}: scaled that gait's w_in_gait column "
                  f"by {sat_scale:g} (magnitude failure, so the column's "
                  f"direction is kept)")


# ═══════════════════════════════════════════════════════════════════
# 7b.  Timing-layer diagnostics
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def timing_report(model, spikes, phase, period, n_gaits, device,
                  t0, n_steps=1500, gait_names=None, indent="    ",
                  sat_rate=None):
    """
    Per-gait firing statistics for the timing layer.  Returns a list of
    formatted lines (also returned as raw dicts) so the caller can print
    them every N epochs.

    PRINTS spk/cyc per gait, plus warnings for the two failure modes.  Phase
    and circular concentration R are still COMPUTED and returned in the stats
    dicts (the config records them, and the cross-gait phase separation line
    below is derived from them) but are no longer printed -- they made the
    every-N-epochs block three times longer than the number actually watched.

      spk/cyc  Spikes per CPG cycle.  0.00 means the unit is DEAD -- its
               sub-network then receives a constant-zero input, its joints
               freeze at whatever the decaying membranes settle to, and the
               w1 slots it feeds get exactly zero gradient, so it does not
               recover on its own.  A much sharper failure than a dead unit
               in a 256-wide dense layer, which is why it is checked every
               few epochs rather than at the end.

               `sat_rate` spk/cyc or above is the opposite failure: the unit
               is firing on essentially every CPG spike, so its spike train
               approaches the OR of the CPG's, is near-identical for every
               saturated unit, and carries no phase.  Just as fatal as
               silence and NOT self-correcting either -- a saturated unit
               sits far above threshold where the surrogate gradient is down
               1e2-1e3x, so neither the task loss nor the spike penalty can
               move it (and Adam's scale-invariance means raising lam does
               not help).  Pass the CEILING, n_cpg * cpg_rate, less a little
               tolerance -- NOT _SAT_FRAC of it.  _SAT_FRAC is a calibration
               cap whose job is to start units well below the ceiling so they
               do not drift into it; a unit at 0.6 of the ceiling is busy but
               still carries phase.

      phase    Circular MEAN of the cycle phase at which the unit fires, in
               [0,1): which part of the cycle each sub-network is told about.

      R        Circular concentration in [0,1].  R near 1 means phase-locked;
               R near 0 means the spikes are smeared and `phase` is
               meaningless.
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
        lines.append(f"{indent}timing {name:>5s} : spk/cyc [{fmt(rate)}]")
        dead = [j for j, r in enumerate(rate) if r < 1e-6]
        if dead:
            lines.append(f"{indent}       {'':>5s}   "
                         f"WARNING: timing neuron(s) {dead} DEAD for "
                         f"{name} — sub-network(s) they feed get no input.")
        if sat_rate:
            sat = [j for j, r in enumerate(rate) if r >= sat_rate]
            if sat:
                lines.append(f"{indent}       {'':>5s}   "
                             f"WARNING: timing neuron(s) {sat} SATURATED for "
                             f"{name} (>= {sat_rate:.0f} spk/cyc) — firing on "
                             f"most CPG spikes, so they carry no phase.")
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
        ("sub_l1",    ("w1", "b1", "beta1_logit", "alpha1_logit")),
        ("sub_l2",    ("w2", "b2", "beta2_logit", "alpha2_logit")),
        # "b_read" also prefix-matches "b_read_gait", which belongs here too.
        ("readout",   ("w_read", "b_read", "w_out", "b_out", "betao_logit",
                       "alphao_logit")),
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
                 spike_obj=None, sat_rate=None):
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
    `sat_rate`    : spk/cyc at or above which a timing unit counts as
                    saturated, for the reinit trigger. The SAME value passed
                    to timing_report, so the warning printed and the action
                    taken cannot disagree. None disables saturation rescue.
    """
    best = float("inf")
    best_path = out_path(out_dir, "best_model.pt")
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
    metrics = MetricsWriter(out_path(out_dir, "metrics.csv"))
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

    # Consecutive timing reports each (gait, unit) pair has been observed
    # dead or saturated, for reinit_timing_units.  Keyed per PAIR, not per
    # unit, because the rescue is per-gait: w_in_gait has a gait axis, so one
    # gait failing is independently fixable.  Only advanced on epochs where
    # the diagnostic runs.
    dead_streak, sat_streak = {}, {}

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
                # Keep the voltage-mode thresholds strictly positive.  See
                # TimingGroupedSNN.clamp_bias_voltage.  getattr because
                # --arch dense has no such parameters; `model` is the raw
                # module (compile swaps `step`, not the module), so this
                # reaches the real method.
                clamp = getattr(model, "clamp_bias_voltage", None)
                if clamp is not None:
                    clamp()
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

                # Failures are tracked per (gait, unit), because the fix is:
                # w_in_gait has a gait axis, so a unit that is fine for five
                # gaits and dead for the sixth gets that one column repaired
                # and nothing else touched.
                if args.reinit_dead_after > 0 and last_timing_stats:
                    n_u = len(last_timing_stats[0]["rate"])
                    pairs = [(gi, u) for gi in range(len(last_timing_stats))
                             for u in range(n_u)]
                    is_dead = {p: last_timing_stats[p[0]]["rate"][p[1]] < 1e-6
                               for p in pairs}
                    is_sat  = {p: (sat_rate is not None and
                                   last_timing_stats[p[0]]["rate"][p[1]]
                                   >= sat_rate)
                               for p in pairs}
                    for p in pairs:
                        dead_streak[p] = (dead_streak.get(p, 0) + 1
                                          if is_dead[p] else 0)
                        sat_streak[p]  = (sat_streak.get(p, 0) + 1
                                          if is_sat[p] else 0)
                    dead_due = [p for p in pairs
                                if dead_streak[p] >= args.reinit_dead_after]
                    sat_due  = [p for p in pairs
                                if sat_streak[p] >= args.reinit_dead_after]
                    if dead_due or sat_due:
                        reinit_timing_units(
                            model, dead_due, sat_due, n_gaits,
                            sat_scale=args.sat_scale)
                        for p in dead_due:
                            dead_streak[p] = 0
                        for p in sat_due:
                            sat_streak[p] = 0

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
    p = out_path(out_dir, "cpg_raster.png")
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
    ax.set_title("TBPTT training")
    plt.tight_layout()
    p = out_path(out_dir, "training_curves.png")
    plt.savefig(p, dpi=140); plt.close()
    print(f"    [saved] {p}")


@torch.no_grad()
def plot_reconstruction(model, spikes, targets, valid, device,
                        out_dir, tgt_range, t0, gait_names, leg_cols, n_joints,
                        period=None, n_cycles=4.0, warm_cycles=2.0,
                        n_steps=None, warm=None):
    """
    Free-run the network on held-out steps, one plot per gait.

    `gait_names`, `leg_cols`, `n_joints` are required, explicit parameters —
    the layout is passed in rather than read off the module constants, which
    are the quadruped layout only (see default_leg_layout). A hexapod run
    silently plotted under the wrong legend would be worse than an error.

    `leg_cols` may have any number of equal-size groups (not just 4 groups
    of 2) -- the subplot grid below sizes itself from len(leg_cols) and the
    per-group column count, the same squeeze=False + hide-unused pattern
    plotting_utils.py already uses elsewhere in this repo.

    The window is set in CPG CYCLES, not raw timesteps. The old fixed
    n_steps=1200 was ~3.4 cycles at the real CPG's period of 352 but ~10 at
    fake_cpg's 120, which squashed the traces to the point of being
    unreadable. n_steps/warm still override for a caller wanting exact counts.
    """
    if n_steps is None:
        n_steps = int(round(n_cycles * float(period))) if period else 1200
    if warm is None:
        warm = int(round(warm_cycles * float(period))) if period else 600

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
        p = out_path(out_dir, f"recon_{gait_names[g]}.png")
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
    p = out_path(out_dir, "rmse_heatmap.png")
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
    p = out_path(out_dir, "transition.png")
    plt.savefig(p, dpi=140); plt.close()
    print(f"    [saved] {p}")


# ═══════════════════════════════════════════════════════════════════
# 9b. Config / checkpoint loading  (used by visualize_timing.py, run_inference.py)
# ═══════════════════════════════════════════════════════════════════

def cfg_get(cfg, key, default=None):
    """
    Look a key up at the config top level, then in `model_detail`, then in the
    verbatim `args` dump.

    Three places because the config records the same fact more than once on
    purpose: the top level is the deployment contract, `model_detail` is the
    reproduction record, and `args` is whatever was actually typed.  Older
    configs (config_version 2, pre-timing-layer) are missing the newer keys
    entirely, hence the default.
    """
    for src in (cfg, cfg.get("model_detail", {}), cfg.get("args", {})):
        if isinstance(src, dict) and src.get(key) is not None:
            return src[key]
    return default


def build_model_from_cfg(cfg, device):
    """Reconstruct the trained architecture.  Shapes must match the
    checkpoint exactly; init-only values (tau ranges, weight scales) are
    irrelevant to the loaded weights but are passed through anyway so the
    printed summary is honest about what the run used."""
    arch = cfg_get(cfg, "arch", "dense")

    # Only reached for configs missing these keys (they are recorded now).
    # Derived the same way train.py derives them, so an old config still gets
    # self-consistent values rather than quadruped-specific constants.
    _P = float(cfg_get(cfg, "cpg_period_steps", 254.0))
    _n_cpg = int(cfg_get(cfg, "n_cpg_neurons", 4))

    common = dict(
        n_gaits   = int(cfg_get(cfg, "n_gaits", 4)),
        max_gaits = int(cfg_get(cfg, "max_gaits", 16)),
        n_neurons = int(cfg_get(cfg, "n_cpg_neurons", 4)),
        n_joints  = int(cfg_get(cfg, "n_joints", N_JOINTS)),
        tau_min   = float(cfg_get(cfg, "tau_min", 2.0)),
        tau_max   = float(cfg_get(cfg, "tau_max", 256.0)),
        slope     = float(cfg_get(cfg, "slope", 25.0)),
    )

    if arch == "timing_grouped":
        n_timing = int(cfg_get(cfg, "n_timing", N_LEGS))
        cfg_leg_cols = cfg_get(cfg, "leg_cols")
        gc_kwargs = {"n_joints": common["n_joints"]}
        if cfg_leg_cols is not None:
            gc_kwargs["leg_cols"] = cfg_leg_cols
        group_cols = cfg_get(cfg, "group_cols") or build_group_cols(
            n_timing, **gc_kwargs)
        # timing_map fallback is None, which the constructor turns into the
        # one-timing-unit-per-sub-network identity map -- i.e. exactly what
        # this model was before --timing_shape/--decoder_shape existed. A
        # config predating those args therefore reconstructs as the model it
        # actually was, not as the current default.
        timing_map = cfg_get(cfg, "timing_map")
        # A config from the (reverted) shared-router era has router_hidden
        # set and a checkpoint containing w_r/w_t, neither of which exists in
        # this class any more -- load_run's strict=False report will flag it.
        #
        # readout_hidden / tau_readout_max fall back to what was ACTUALLY
        # true before those became separate options, not to their current
        # smarter defaults: readout_hidden didn't exist yet, and the readout
        # membrane was simply Hg wide (see the "Ho == Hg (the original)"
        # comment in TimingGroupedSNN.__init__), so its fallback here is
        # "hidden", not 32. tau_readout_max was hardcoded 40.0 (see the
        # module's own note on this in --tau_readout_max's help text), not
        # period/(2*pi) -- that derivation is newer than the option.
        # timing_slope similarly falls back to "slope": the constructor's own
        # timing_slope=None branch does exactly this, so a config predating
        # the separate --timing_slope arg had the timing layer using --slope.
        _hidden = int(cfg_get(cfg, "hidden", 256))
        model = TimingGroupedSNN(
            hidden_per_group = _hidden,
            n_timing         = n_timing,
            group_cols       = group_cols,
            timing_map       = timing_map,
            tau_timing_min   = float(cfg_get(cfg, "tau_timing_min", 2.0)),
            tau_timing_max   = float(cfg_get(cfg, "tau_timing_max",
                                               _P / _n_cpg)),
            readout_hidden   = int(cfg_get(cfg, "readout_hidden", _hidden)),
            tau_readout_max  = float(cfg_get(cfg, "tau_readout_max", 40.0)),
            timing_slope     = float(cfg_get(cfg, "timing_slope",
                                               cfg_get(cfg, "slope", 25.0))),
            timing_reset     = str(cfg_get(cfg, "timing_reset", "subtract")),
            hidden_reset     = str(cfg_get(cfg, "hidden_reset", "subtract")),
            sub_film         = str(cfg_get(cfg, "sub_film", "both")),
            # Each fallback is what the code did BEFORE the option existed,
            # not its current default: film beta WAS applied, there was no
            # per-gait readout bias (so the parameter is not created and no
            # key goes missing), and the synapse was first order.
            film_mode        = str(cfg_get(cfg, "film_mode", "gamma_beta")),
            readout_gait_bias = bool(cfg_get(cfg, "readout_gait_bias", 0)),
            synaptic         = str(cfg_get(cfg, "synaptic", "none")),
            tau_syn_min      = float(cfg_get(cfg, "tau_syn_min", 2.0)),
            tau_syn_max      = float(cfg_get(cfg, "tau_syn_max", 20.0)),
            # gate_mode replaced the old boolean event_gated. Map the old
            # key when only it is present: True was the "decay" behaviour
            # (membranes leak every step), and absent entirely means ungated.
            # "current" was the only behaviour before bias_mode existed.
            bias_mode        = str(cfg_get(cfg, "bias_mode", "current")),
            gate_mode        = str(cfg_get(
                cfg, "gate_mode",
                "decay" if cfg_get(cfg, "event_gated", False) else "none")),
            # sub_ln changes FORWARD BEHAVIOUR, not shapes, so a wrong value
            # loads cleanly and then quietly computes something else.
            sub_ln           = str(cfg_get(cfg, "sub_ln", "l2")),
            **common)
    else:
        # Fallbacks are what this class did BEFORE each option existed, so an
        # old dense checkpoint reconstructs as what it actually ran:
        # readout as wide as hidden, tau_readout_max 40, bias currents,
        # subtractive reset, LayerNorm and FiLM on both layers.
        model = DenseSNN(
            hidden          = int(cfg_get(cfg, "hidden", 256)),
            readout_hidden  = cfg_get(cfg, "readout_hidden"),
            tau_readout_max = float(cfg_get(cfg, "tau_readout_max", 40.0)),
            bias_mode       = str(cfg_get(cfg, "bias_mode", "current")),
            hidden_reset    = str(cfg_get(cfg, "hidden_reset", "subtract")),
            sub_ln          = str(cfg_get(cfg, "sub_ln", "both")),
            sub_film        = str(cfg_get(cfg, "sub_film", "both")),
            film_mode       = str(cfg_get(cfg, "film_mode", "gamma_beta")),
            **common)

    return model.to(device), arch


def load_run(model_dir, ckpt_name, cfg_name, device):
    model_dir = Path(model_dir)
    cfg_path  = in_path(model_dir, cfg_name)
    ckpt_path = in_path(model_dir, ckpt_name)
    for p in (cfg_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    cfg = json.loads(cfg_path.read_text())
    model, arch = build_model_from_cfg(cfg, device)

    sd = torch.load(ckpt_path, map_location=device)
    # torch.compile in train.py is an instance-attribute swap on `step`, not a
    # module wrapper, so keys should be unprefixed -- but strip the wrapper
    # prefix anyway in case a future run compiles the module itself.
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  NOTE  missing keys: {list(missing)}")
        print(f"        unexpected  : {list(unexpected)}")
        print( "        A non-empty list here means the config and the "
               "checkpoint disagree — treat every figure as suspect.")
    model.eval()

    n_par = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {ckpt_path.name}  ({type(model).__name__}, "
          f"arch={arch}, {n_par:,} params)")
    print(f"  Config {cfg_path.name}  (version "
          f"{cfg.get('config_version', '?')})")
    return cfg, model, arch


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

        path = out_path(out_dir, "cpg_lif_snn_step.onnx")
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

    cfg_path = out_path(out_dir, "cpg_lif_snn_config.json")
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
    ap.add_argument("--fake_cpg", type=int, default=1,
                    help="1 = substitute LIFCPGStepper.fake_step_chunk's "
                         "back-to-back, no-inter-burst-gap spike pattern for "
                         "the real oscillator. Temporary, for testing whether "
                         "training benefits from a continuous-activity CPG "
                         "ahead of the real one being retuned to produce this "
                         "directly.")
    ap.add_argument("--n_cpg_neurons", type=int, default=6,
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
                         "visualize_timing.py's --gaits, though that one filters "
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
                    help="dense: DenseSNN, one fully connected network. "
                         "timing_grouped: TimingGroupedSNN — CPG -> small "
                         "timing layer -> n_timing disconnected sub-networks, "
                         "one per timing neuron. Keep 'dense' runnable so A/B "
                         "at matched gradient steps stays possible.")
    ap.add_argument("--timing_shape", type=str, default="per_joint",
                    choices=list(STRUCTURE_SHAPES),
                    help="[timing_grouped] Structure of the timing layer. "
                         "per_leg: one timing LIF per leg (n_legs units). "
                         "per_joint: one per output column (n_joints units). "
                         "Independent of --n_cpg_neurons — the timing layer "
                         "is densely driven by all CPG spikes, so the counts "
                         "need not match.")
    ap.add_argument("--decoder_shape", type=str, default=None,
                    choices=list(STRUCTURE_SHAPES),
                    help="[timing_grouped] Structure of the angle decoders, "
                         "i.e. the disconnected sub-networks. Default None = "
                         "match --timing_shape, which gives one decoder per "
                         "timing unit and a fan-in of K=1. per_leg: one "
                         "sub-network per leg, each emitting that leg's "
                         "columns (3 joints on the hexapod). per_joint: one "
                         "per column, emitting a single angle. Combined with "
                         "--timing_shape this fixes the fan-in K: a "
                         "sub-network is fed by every timing unit whose "
                         "columns overlap its own. per_joint timing + per_leg "
                         "decoders gives K=3, so each decoder sees three "
                         "independent phase references for its own leg. The "
                         "reverse (per_leg timing + per_joint decoders) is "
                         "legal but wasteful: K=1 and the three decoders of a "
                         "leg are handed identical spike trains.")
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
                         "original DenseSNN almost exactly at the "
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
    ap.add_argument("--recon_cycles", type=float, default=4.0,
                    help="CPG cycles shown in each recon_<gait>.png. In cycles "
                         "rather than timesteps so the plot stays readable "
                         "whether the period is 352 (real CPG) or 120 "
                         "(--fake_cpg); the old fixed 1200-step window was 10 "
                         "cycles at the latter and unreadably dense.")
    ap.add_argument("--hidden_reset", type=str, default="zero",
                        choices=["zero", "subtract"],
                        help="[timing_grouped] Membrane reset for the hidden "
                             "layer. 'zero' (default) matches the CPG "
                             "(LIFGeneralArray does v[spike]=0) and caps each "
                             "unit at one spike per timing spike. Together "
                             "with --bias_mode voltage and --film_mode "
                             "gamma_only it closes the last path by which a "
                             "hidden unit can fire on a timing-silent step, so "
                             "the sub-networks are FULLY naturally gated — see "
                             "NATURAL GATING in TimingGroupedSNN's docstring. "
                             "'subtract' leaves the residual mem-thresh, which "
                             "can still exceed threshold, so a strongly-driven "
                             "unit re-fires on the silent steps after a timing "
                             "spike. That converts injection MAGNITUDE into a "
                             "spike COUNT spread over the following steps, a "
                             "real expansion of what layer 1 can represent, "
                             "and it measured slightly better on RMSE — but it "
                             "also leaves the task loss partly insensitive to "
                             "when the timing spikes land, which is what makes "
                             "alignment seed-dependent. The trade is RMSE "
                             "against a loss surface where alignment is "
                             "determined rather than lucky.")
    ap.add_argument("--timing_slope", type=float, default=5.0,
                    help="[timing_grouped] Surrogate-gradient slope for the "
                         "TIMING layer only; --slope still applies to the "
                         "sub-networks. The surrogate "
                         "derivative is 1/(slope*|x|+1)^2, so at the "
                         "sub-networks' default of 25 a unit a few units below "
                         "threshold is nearly invisible to gradients. This is "
                         "the layer where "
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
                         "a cap; raise it to test whether a sub-network can "
                         "usefully hold its output across a whole cycle.")
    ap.add_argument("--calibrate_gains", type=int, default=1,
                    help="[timing_grouped] 1 = before training, bisect each "
                         "timing unit's CPG->timing weight column until its "
                         "firing rate lands within a factor of 2 of the CPG's "
                         "own spikes-per-cycle (measured at startup, so there "
                         "is no band to configure). Forward passes only, a "
                         "second of compute, so there is no scale factor to "
                         "guess at. 0 = skip.")
    ap.add_argument("--calibrate_per_gait", type=int, default=1,
                    help="[timing_grouped, needs --calibrate_gains 1] 1 = "
                         "bisect a separate gain for every (gait, unit) pair, "
                         "so every pair lands in the band. 0 = the old "
                         "behaviour: one gain per unit, bisected against that "
                         "unit's MINIMUM rate across gaits. 0 is a saturation "
                         "generator and is kept only for A/B: nothing bounded "
                         "the maximum, so a unit weakly driven in one gait had "
                         "its gain raised until that gait fired, dragging "
                         "every other gait up with it -- measured at 3-10x "
                         "over the band typically and 20-48x when a column "
                         "summed near zero. Training cannot undo it: a "
                         "saturated unit sits far above threshold where "
                         "spike_fn's surrogate is down 1e2-1e3x, so the spike "
                         "penalty's gradient is ~1e-6, and Adam is "
                         "scale-invariant, so raising --spike_stats_lambda "
                         "10,000x moves the gain 1.5%%.")
    ap.add_argument("--spike_objective", type=str, default="min_count",
                    choices=sorted(SPIKE_OBJECTIVES),
                    help="[timing_grouped] Which objective shapes the timing "
                         "layer's spike train. "
                         "'min_count' (default): pay a flat L1 cost per spike "
                         "and let the TASK loss decide where spikes are worth "
                         "spending. The intended result is an adaptive "
                         "sampling clock (dense through swing, sparse through "
                         "stance) with no supervision. Note it needs the "
                         "output to depend on the spike train ALONE to mean "
                         "anything -- see NATURAL GATING in TimingGroupedSNN's "
                         "docstring. "
                         "'cpg_match': match the CPG's spikes-per-cycle and "
                         "burst tightness; produces clean CPG-like bursts but "
                         "conflicts with min_count's premise, since it wants "
                         "every spike at one phase rather than wherever the "
                         "output must change. "
                         "'none': unconstrained. "
                         "New strategies: subclass SpikeObjective, set a "
                         "`name`, and it appears here automatically.")
    ap.add_argument("--spike_stats_lambda", type=float, default=0.0,
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
    ap.add_argument("--min_count_floor", type=float, default=2.0,
                    help="[--spike_objective min_count] Spikes per cycle, per "
                         "(gait, timing unit), that the penalty aims at as a "
                         "lower bound. The penalty is a TWO-SIDED HINGE: "
                         "lam * (relu(duty - floor/period) + "
                         "floor_weight * relu(floor/period - duty)). Above "
                         "the floor it is the usual L1 sparsity cost; below "
                         "it the gradient reverses sign, so descent pushes "
                         "the rate back UP. That is what stops a unit being "
                         "pruned into silence, which is unrecoverable through "
                         "the task loss (a silent unit passes exactly zero "
                         "gradient to the w1 slots it feeds). Observed: a run "
                         "pruned several sub-networks to an excellent 2-5 "
                         "well-placed spikes/cycle and pruned others to 0. "
                         "Note a linear term would NOT do this — a constant "
                         "derivative means subtracting the floor changes the "
                         "penalty's value but not its gradient, so the linear "
                         "form is identical to no floor at all; only a kink "
                         "can flip the sign. Set to the fewest spikes a "
                         "sub-network can do anything with, not a comfortable "
                         "number, since the hinge makes the floor an "
                         "attractor. 0 recovers the original un-floored L1.")
    ap.add_argument("--min_count_floor_weight", type=float, default=1.0,
                    help="[--spike_objective min_count] Weight on the "
                         "BELOW-floor half of the hinge, relative to lam. "
                         "1.0 (default) is symmetric: equal pressure per "
                         "spike in both directions. 0.0 makes the floor "
                         "one-sided — pruning stops there but nothing pushes "
                         "back up, so units can still drift to silence via "
                         "the task gradient. Raise above 1.0 if units still "
                         "die; lower it if the floor is dragging units up "
                         "that the task would rather run sparser.")
    ap.add_argument("--spike_lambda_warmup", type=int, default=10,
                    help="[timing_grouped] Epochs over which the spike "
                         "penalty's lambda ramps linearly from 0. Insurance "
                         "against a specific failure: when gated, a "
                         "silent timing unit starves its sub-network entirely "
                         "(memo decays, y collapses to b_out) and passes ZERO "
                         "gradient to w1/w2/w_read/w_out, so an L1 spike cost "
                         "applied before the task loss needs the spikes can "
                         "prune a unit into a state it cannot recover from. "
                         "0 disables the ramp.")
    ap.add_argument("--reinit_dead_after", type=int, default=2,
                    help="[timing_grouped] Rescue a failed (gait, timing unit) "
                         "pair after this many consecutive timing reports in "
                         "the same state, so the wall-clock trigger scales "
                         "with --timing_log_every. PER GAIT, matching "
                         "w_in_gait's own gait axis: a unit that is fine for "
                         "five gaits and dead for the sixth gets only that "
                         "column touched. Dead pairs get the column re-rolled "
                         "and the per-gait bias zeroed; saturated pairs get "
                         "the column scaled by --sat_scale instead, since "
                         "saturation is a magnitude failure and the column's "
                         "direction is worth keeping. A unit dead for EVERY "
                         "gait additionally gets its sub-net w1 slots "
                         "re-rolled, since w1's gradient is proportional to "
                         "the timing spike and so was exactly zero throughout. "
                         "Neither failure self-corrects: a silent unit passes "
                         "zero gradient, and a saturated one sits where the "
                         "surrogate is down 1e2-1e3x. 0 disables.")
    ap.add_argument("--sat_scale", type=float, default=0.5,
                    help="[timing_grouped] Factor applied to a saturated "
                         "(gait, unit) column of w_in_gait when "
                         "--reinit_dead_after triggers. Rate is monotone in "
                         "the column's gain, so this strictly reduces it; a "
                         "badly saturated unit may take several triggers to "
                         "walk down, which is self-limiting since it stops as "
                         "soon as the unit leaves the saturation band. 1.0 "
                         "leaves saturated units alone.")
    ap.add_argument("--bias_mode", type=str, default="voltage",
                    choices=["current", "voltage", "none"],
                    help="[timing_grouped] How neuron excitability is "
                         "parameterised. 'current': an additive term in the "
                         "input current. 'voltage' (default): a per-unit "
                         "offset to the spiking THRESHOLD instead, with no "
                         "bias current in the spiking path. 'none': no bias. "
                         "For an ungated layer with subtractive reset the "
                         "first two are exactly equivalent, so voltage gives "
                         "up nothing; where they differ voltage is preferable, "
                         "because a threshold never touches the membrane (so a "
                         "frozen sub-network provably cannot have spiked) and "
                         "because it keeps the bias out of the per-spike "
                         "injection, whose magnitude is what causes the jerk. "
                         "See BIAS MODES in TimingGroupedSNN's docstring.")
    ap.add_argument("--gate_mode", type=str, default="none",
                    choices=["none", "decay"],
                    help="[timing_grouped] Whether to gate the sub-networks "
                         "on their own timing spikes explicitly. "
                         "'none' (default): no explicit gate. With "
                         "--bias_mode voltage the sub-networks receive current "
                         "only on a timing spike anyway, so most hidden units "
                         "cannot fire without one -- see NATURAL GATING in "
                         "TimingGroupedSNN's docstring. "
                         "'decay': input and sub-net spikes explicitly gated, "
                         "membranes still leak every step. Forces the "
                         "spike-train-only property instead of relying on it. "
                         "Under the fully naturally-gated defaults its FORWARD "
                         "pass is identical to 'none', but it does not train "
                         "the same: `gate` is spk_t and carries its surrogate "
                         "gradient, so gating amplifies the gradient on "
                         "SPIKING steps (up to 6x, from five extra "
                         "multiplicative appearances) and makes it EXACTLY "
                         "ZERO on silent steps -- removing the only route by "
                         "which 'this unit should have fired here' reaches the "
                         "timing layer. So it biases spike counts downward "
                         "twice over. It also caps each hidden unit at one "
                         "spike per timing spike, discarding the "
                         "magnitude-to-count expansion subtractive reset "
                         "provides.")
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
    ap.add_argument("--film_mode", type=str, default="gamma_only",
                    choices=["gamma_only", "gamma_beta"],
                    help="Which halves of the FiLM tables are applied. "
                         "'gamma_only' (default): multiplicative modulation "
                         "only. Gamma vanishes wherever the injection is "
                         "zero, so per-gait conditioning is retained WITHOUT "
                         "any tonic drive -- this is what makes the "
                         "sub-networks fully naturally gated (see NATURAL "
                         "GATING in TimingGroupedSNN's docstring), and it is "
                         "strictly better than --sub_film none, which would "
                         "drop per-gait conditioning of layer 1 entirely "
                         "(w1/w2 have no gait axis, so FiLM is the only "
                         "per-gait handle there). "
                         "'gamma_beta': also apply the additive beta, the "
                         "behaviour before this option existed. Beta is added "
                         "on every timestep whether or not anything spiked, "
                         "so it is tonic drive: it smooths the output and "
                         "keeps hidden membranes near threshold, but it also "
                         "makes the task loss insensitive to timing-spike "
                         "phase. The beta half of the table is allocated in "
                         "both modes, so checkpoint shapes are unaffected; in "
                         "gamma_only it simply gets no gradient.")
    ap.add_argument("--readout_gait_bias", type=int, default=1,
                    help="[timing_grouped] 1 = a learned per-gait constant "
                         "CURRENT into the readout membrane. Everything else "
                         "in the sub-networks (w_read, w_out, b_out, b1, b2, "
                         "b_read) is gait-shared, so with --film_mode "
                         "gamma_only there is otherwise no per-gait DC "
                         "anywhere and every gait relaxes toward the same "
                         "output offset. Gaits that differ in MEAN joint "
                         "angle rather than waveform shape would then have to "
                         "buy that offset with spike RATE, which costs spikes, "
                         "shrinks the gait's downward range, and leaves "
                         "inter-spike ripple. Injected as a current rather "
                         "than added to y so it is filtered by the readout's "
                         "own tau and fades in over a gait switch instead of "
                         "stepping. Does not reintroduce phase invariance: "
                         "memo is analog, never spikes, and a constant "
                         "supplies no time-varying basis. 0 = off, which is "
                         "also what configs predating this option "
                         "reconstruct as.")
    ap.add_argument("--synaptic", type=str, default="none",
                    choices=["none", "hidden", "all"],
                    help="Second-order synaptic filter, matching snnTorch's "
                         "`Synaptic` neuron: syn = alpha*syn + injection, then "
                         "mem = beta*mem + syn. 'none' (default): first-order, "
                         "injection goes straight into the membrane. 'hidden': "
                         "sub-net layers 1 and 2. 'all': those plus the analog "
                         "readout membrane, which is where the output jolt "
                         "actually lands. Spreading one injection over many "
                         "timesteps shrinks the output step at each spike and "
                         "keeps hidden membranes near threshold for a WINDOW "
                         "after a spike, so firing rates stay learnable "
                         "upward -- both things film beta was providing via "
                         "tonic drive, except the filter decays to zero during "
                         "real silence, so spike placement still matters. This "
                         "is Loihi's native CUBA neuron (synaptic current plus "
                         "membrane voltage, separate decays), so it is the "
                         "standard neuromorphic model rather than a departure "
                         "from it. Costs one extra state tensor per layer.")
    ap.add_argument("--tau_syn_min", type=float, default=2.0,
                    help="[--synaptic] Tau init floor for the synaptic "
                         "current.")
    ap.add_argument("--tau_syn_max", type=float, default=None,
                    help="[--synaptic] Tau init ceiling for the synaptic "
                         "current. Default None = period/16 (7.5 steps at the "
                         "fake CPG's 120, 22 at the real 352), chosen so one "
                         "spike's effect is spread over a short window "
                         "relative to the cycle rather than smeared across it "
                         "-- the filter is meant to smooth the injection, not "
                         "to become the memory, which is mem1/mem2's job at "
                         "--tau_max. A GUESS, not a derived value; taus stay "
                         "learnable so this is only an init range, and it is "
                         "worth sweeping.")
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
                         "saves LN's retained tensors per timestep.")

    ap.add_argument("--hidden",     type=int,   default=128,
                    help="Hidden width. For --arch dense this is the TOTAL "
                         "width (dense H x H layers). For --arch "
                         "timing_grouped it is PER GROUP, so w2/w_read are "
                         "(G, H, H) — at G=4, hidden=128 lands near the dense "
                         "hidden=256 parameter count and is the matched-"
                         "parameter baseline; hidden=256 is ~4x that.")
    ap.add_argument("--max_gaits",  type=int,   default=8,
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
    ap.add_argument("--val_chunks",       type=int,   default=8)
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
    ap.add_argument("--lr",               type=float, default=4e-3,
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
    ap.add_argument("--visualize", type=int, default=1,
                    help="1 = run visualize_timing.py's timing-layer analysis at the "
                         "end of training, writing into this run's own output "
                         "folders. 0 = skip (run visualize_timing.py by hand later).")
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
    # Resolved before the CPG run and before the timing/decoder shapes, since
    # those depend on n_joints / n_legs, which depend on what got loaded.
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
              f"without invalidating this checkpoint.")

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

    if args.arch == "timing_grouped":
        # None means "same as the timing layer", i.e. one decoder per timing
        # unit and K=1. Resolved onto args itself so the config dump records
        # the shape that actually ran rather than a null.
        if args.decoder_shape is None:
            args.decoder_shape = args.timing_shape
        timing_cols = build_partition(args.timing_shape, leg_cols, n_joints,
                                      what="--timing_shape")
        group_cols  = build_partition(args.decoder_shape, leg_cols, n_joints,
                                      what="--decoder_shape")
        timing_map  = build_timing_map(timing_cols, group_cols)
        args.n_timing = len(timing_cols)
        print(f"      structure: timing {args.timing_shape} "
              f"({len(timing_cols)} units) -> decoders {args.decoder_shape} "
              f"({len(group_cols)} sub-nets), fan-in K="
              f"{len(timing_map[0])}, {len(group_cols[0])} column(s) out per "
              f"sub-net")
        if len(timing_cols) < len(group_cols):
            print(f"      NOTE every sub-network of a leg is fed the SAME "
                  f"timing unit, so they receive identical input and differ "
                  f"only in their own weights. Consider --timing_shape "
                  f"per_joint.")
    else:
        timing_cols = group_cols = timing_map = None
        args.n_timing = None

    # ── 1. CPG ──────────────────────────────────────────────────
    print("\n[1/6] Bursting-LIF CPG ...")
    spikes = run_cpg(N=args.n_cpg_neurons, tmax=args.tmax, warmup=args.warmup,
                     i_app=args.i_app, fake_cpg=bool(args.fake_cpg))

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
    if args.tau_syn_max is None:
        # A short window relative to the cycle: the synaptic filter is meant
        # to smooth the injection, not to hold memory (that is mem1/mem2 at
        # tau_max). Learnable, so this is an init range only.
        args.tau_syn_max = float(period) / 16.0
        print(f"      tau_syn_max    : {args.tau_syn_max:6.1f}  (synaptic "
              f"current; period/16 — a guess, worth sweeping)")
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
            timing_map=timing_map,
            n_joints=n_joints, readout_hidden=args.readout_hidden,
            tau_min=args.tau_min, tau_max=args.tau_max,
            tau_timing_min=args.tau_timing_min,
            tau_timing_max=args.tau_timing_max,
            tau_readout_max=args.tau_readout_max,
            sub_ln=args.sub_ln, sub_film=args.sub_film,
            film_mode=args.film_mode,
            timing_reset=args.timing_reset, hidden_reset=args.hidden_reset,
            gate_mode=args.gate_mode, bias_mode=args.bias_mode,
            readout_gait_bias=bool(args.readout_gait_bias),
            synaptic=args.synaptic,
            tau_syn_min=args.tau_syn_min, tau_syn_max=args.tau_syn_max,
            slope=args.slope, timing_slope=args.timing_slope).to(device)
    else:
        # The dense ABLATION. Every option below is shared with
        # TimingGroupedSNN so the two differ in the factorisation and nothing
        # else. NOTE on matched capacity: the dense hidden layer is H x H where
        # the grouped model has G blocks of Hg x Hg, so at the same --hidden
        # the dense model is G times smaller; scale --hidden by sqrt(G) to
        # compare at equal parameter count.
        model = DenseSNN(hidden=args.hidden, n_gaits=len(gait_tables),
                            max_gaits=args.max_gaits,
                            n_neurons=args.n_cpg_neurons,
                         tau_min=args.tau_min, tau_max=args.tau_max,
                         slope=args.slope, n_joints=n_joints,
                         readout_hidden=args.readout_hidden,
                         tau_readout_max=args.tau_readout_max,
                         bias_mode=args.bias_mode,
                         hidden_reset=args.hidden_reset,
                         sub_ln=args.sub_ln, sub_film=args.sub_film,
                         film_mode=args.film_mode).to(device)

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
        n_dec = len(group_cols)
        fan_in = len(timing_map[0])
        print(f"      hidden={args.hidden} PER GROUP  "
              f"n_timing={args.n_timing}  n_decoders={n_dec}  "
              f"params={n_par:,}")
        print(f"      CPG({args.n_cpg_neurons}) -> timing"
              f"({args.n_timing}, {args.timing_shape}, LIF, per-gait weights) "
              f"-> {n_dec} x ({args.decoder_shape}) "
              f"[{fan_in} -> {args.hidden} -> {args.hidden} -> "
              f"readout({args.readout_hidden}) -> {len(group_cols[0])}], "
              f"no cross talk between sub-networks")
        print(f"      timing reset={args.timing_reset}  "
              f"sub_film={args.sub_film}  sub_ln={args.sub_ln}  "
              f"gate_mode={args.gate_mode}  bias_mode={args.bias_mode}")
        print(f"      film_mode={args.film_mode}  "
              f"readout_gait_bias={bool(args.readout_gait_bias)}  "
              f"synaptic={args.synaptic}"
              + (f" (tau_syn init [{args.tau_syn_min:.0f}, "
                 f"{args.tau_syn_max:.0f}])" if args.synaptic != "none"
                 else ""))
        if args.gate_mode == "decay":
            print(f"      explicit gating ON: each sub-network is injected "
                  f"only on its own timing spike, and its spikes are "
                  f"suppressed in between; membranes still decay.")
        elif args.bias_mode == "voltage":
            leaks = []
            if args.film_mode != "gamma_only":
                leaks.append("film beta")
            if args.hidden_reset != "zero":
                leaks.append("subtractive reset")
            if leaks:
                print(f"      natural gating PARTIAL: voltage-mode bias means "
                      f"the sub-networks are driven only on timing spikes, but "
                      f"{' and '.join(leaks)} can still fire a hidden unit "
                      f"without one. Run check_bias_voltage.py to see which.")
            else:
                print(f"      natural gating FULL: no explicit gate, and no "
                      f"path by which a hidden unit can fire on a "
                      f"timing-silent step (voltage bias, gamma-only FiLM, "
                      f"zero reset). gate_mode=decay should now give a "
                      f"bit-identical FORWARD pass — but not identical "
                      f"training, since the gate carries spk_t's surrogate "
                      f"gradient (see the NOTE in step()).")
        if args.spike_objective == "cpg_match":
            print(f"      NOTE --spike_objective cpg_match wants every spike "
                  f"at ONE cycle phase, whereas the sub-networks need spikes "
                  f"wherever the output must change. 'min_count' is the "
                  f"objective that lets the task loss place them.")
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
              f"see the TimingGroupedSNN docstring)")
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
        # Actually saturated means firing on essentially EVERY CPG spike,
        # i.e. at the ceiling of n_cpg * cpg_rate -- not at _SAT_FRAC of it.
        # _SAT_FRAC is only a calibration cap, whose whole job is to start
        # units well BELOW the ceiling so they do not drift into it; a unit
        # sitting at 0.6 of the ceiling is firing a lot but still carries
        # phase.
        #
        # The 0.95 is tolerance, not a lower target: cpg_rate is measured
        # over the whole spike train while timing_report measures over ~6
        # cycles, so a genuinely saturated unit can come out a hair under the
        # product of the two. Computed once and passed to BOTH timing_report
        # and run_training, so the number warned about and the number acted
        # on are the same by construction.
        sat_rate = 0.95 * float(args.n_cpg_neurons) * float(cpg_rate)
        timing_diag = lambda: timing_report(
            model, spikes, phase, period, len(gait_tables), device,
            t0=t_diag, n_steps=int(6 * period), gait_names=gait_names,
            indent="      ", sat_rate=sat_rate)
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
        # min_count needs far more spikes than the CPG emits: under natural
        # gating the sub-networks are driven only on timing spikes, and ~10
        # spikes per 352-step cycle cannot render a moving waveform (a
        # zero-order-hold analysis on a gait-shaped waveform puts the
        # requirement nearer 40-60/cycle for ~1.5 degrees of error).  Starting
        # at the CPG's rate would begin in a starved regime where the task
        # gradient is weak, exactly when L1 pressure is arriving.  So start
        # high and let the L1 term prune downward.
        #
        # Conditioned on the objective being ACTIVE (lambda > 0), not on
        # --gate_mode: natural gating gives the same spike-train dependence
        # that explicit gating does, so gate_mode is the wrong thing to test.
        # And with lambda at 0 nothing prunes, so calibrating high would just
        # leave the layer firing high for the whole run.
        if args.spike_objective == "min_count" and args.spike_stats_lambda > 0.0:
            band_lo, band_hi = 2.0 * cpg_rate, 4.0 * cpg_rate
            print(f"      (min_count: calibrating to "
                  f"{band_lo:.0f}-{band_hi:.0f} spk/cyc, well above the CPG's "
                  f"{cpg_rate:.1f}, so the network starts able to render the "
                  f"waveform and prunes from there)")
        else:
            band_lo, band_hi = 1.5 * cpg_rate, 2.0 * cpg_rate
            print(f"      (no active min_count: calibrating to "
                  f"{band_lo:.0f}-{band_hi:.0f} spk/cyc)")

        # A timing unit cannot exceed one spike per CPG spike: the BLIF CPG
        # never fires two neurons in the same timestep, so there are exactly
        # n_cpg * cpg_rate opportunities per cycle, and at calibration time
        # b_t is zero and the reset is to zero, so nothing fires between them.
        # AT that ceiling a unit's spike train is the OR of the CPG's and
        # identical for every such unit -- zero phase information, sub-network
        # effectively dead.
        #
        # This has to be enforced HERE, not inside calibrate_gains, because the
        # band above can exceed the ceiling. The min_count branch asks for
        # 4 * cpg_rate = 40 spk/cyc against a ceiling of 60 and a cap of 36
        # for a 6-neuron CPG, and a band whose top is above the CEILING makes
        # the `too_loud` test unreachable -- the bisection would drive units TO
        # saturation and then report them as in-band. Inert for the other
        # branch, whose band tops out at 2 * cpg_rate.
        ceiling = float(args.n_cpg_neurons) * float(cpg_rate)
        cap     = _SAT_FRAC * ceiling
        if band_hi > cap:
            print(f"      NOTE band top {band_hi:.0f} exceeds "
                  f"{_SAT_FRAC:g} x the {ceiling:.0f} spk/cyc ceiling "
                  f"(n_cpg {args.n_cpg_neurons} x cpg_rate {cpg_rate:.1f}); "
                  f"capping to {cap:.0f} so calibration cannot target a "
                  f"phase-blind unit.")
            band_hi = cap
            band_lo = min(band_lo, 0.5 * band_hi)
        calib = calibrate_gains(
            model, spikes, len(gait_tables), device, period,
            lo=band_lo, hi=band_hi, per_gait=bool(args.calibrate_per_gait),
            cpg_rate_hint=cpg_rate)

    # ── Spike objective (strategy) ──────────────────────────────
    spike_obj = make_spike_objective(
        args.spike_objective,
        lam=args.spike_stats_lambda, period=period,
        n_gaits=len(gait_tables),
        target_rate=cpg_rate, target_R=cpg_R,
        min_count_floor=args.min_count_floor,
        floor_weight=args.min_count_floor_weight)

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
            spike_obj=spike_obj, sat_rate=sat_rate)
        final_lr = float(opt.param_groups[0]["lr"])
        model.load_state_dict(torch.load(in_path(out_dir, "best_model.pt"),
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
                               n_joints=n_joints, period=period,
                               n_cycles=args.recon_cycles)
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
        "fake_cpg":         bool(args.fake_cpg),
        "n_timing":         (int(args.n_timing)
                             if args.arch == "timing_grouped" else None),
        "readout_hidden":   (int(args.readout_hidden)
                             if args.arch == "timing_grouped" else None),
        "group_cols":       ([list(g) for g in group_cols]
                             if group_cols is not None else None),
        # The structure is recorded three ways on purpose: the two shape
        # NAMES are the deployment contract, timing_map is what the model
        # actually wired up, and timing_cols/group_cols are what each unit
        # owns. build_model_from_cfg reads timing_map, so a hand-edited shape
        # name cannot silently change the architecture out from under a
        # checkpoint.
        "timing_shape":     (str(args.timing_shape)
                             if args.arch == "timing_grouped" else None),
        "decoder_shape":    (str(args.decoder_shape)
                             if args.arch == "timing_grouped" else None),
        "timing_cols":      ([list(t) for t in timing_cols]
                             if timing_cols is not None else None),
        "timing_map":       ([list(m) for m in timing_map]
                             if timing_map is not None else None),
        "timing_fan_in":    (len(timing_map[0])
                             if timing_map is not None else None),
        "gait_names":       gait_names,
        # Same list as gait_names for CSV-loaded gaits (file stem == display
        # name, matching train_snn.py's convention) — kept as a separate key
        # anyway, so a future remap of display names doesn't have to also
        # change what visualize_timing.py loads from disk.
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
            "n_decoders":       (len(group_cols)
                                 if group_cols is not None else None),
            "timing_shape":     (str(args.timing_shape)
                                 if args.arch == "timing_grouped" else None),
            "decoder_shape":    (str(args.decoder_shape)
                                 if args.arch == "timing_grouped" else None),
            "timing_map":       ([list(m) for m in timing_map]
                                 if timing_map is not None else None),
            "hidden_per_group": (int(args.hidden)
                                 if args.arch == "timing_grouped" else None),
            "tau_min":          float(args.tau_min),
            "tau_max":          float(args.tau_max),
            "tau_timing_min":   (float(args.tau_timing_min)
                                 if args.arch == "timing_grouped" else None),
            "tau_timing_max":   (float(args.tau_timing_max)
                                 if args.arch == "timing_grouped" else None),
            # The six below apply to BOTH arches now that DenseSNN accepts
            # them, so they are recorded unconditionally: a None here would
            # make build_model_from_cfg fall back to a default and rebuild a
            # dense checkpoint as something it never was.
            "sub_ln":           str(args.sub_ln),
            "timing_layernorm": False,
            "readout_hidden":   int(args.readout_hidden),
            "tau_readout_max":  float(args.tau_readout_max),
            "timing_reset":     (args.timing_reset
                                 if args.arch == "timing_grouped" else None),
            "freeze_blocks":    (args.freeze_blocks or None),
            "bias_mode":        str(args.bias_mode),
            "gate_mode":        (args.gate_mode
                                 if args.arch == "timing_grouped" else None),
            "spike_objective":  (spike_obj.describe()
                                 if args.arch == "timing_grouped" else None),
            "min_count_floor":  float(args.min_count_floor),
            "min_count_floor_weight": float(args.min_count_floor_weight),
            "sub_film":         str(args.sub_film),
            "film_mode":        str(args.film_mode),
            "readout_gait_bias": (bool(args.readout_gait_bias)
                                  if args.arch == "timing_grouped" else None),
            "synaptic":         (str(args.synaptic)
                                 if args.arch == "timing_grouped" else None),
            "tau_syn_min":      float(args.tau_syn_min),
            "tau_syn_max":      float(args.tau_syn_max),
            # Was read by build_model_from_cfg but never written, so a
            # --hidden_reset zero checkpoint silently reconstructed as
            # "subtract" for EITHER architecture. Pre-existing bug.
            "hidden_reset":     str(args.hidden_reset),
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
            "weights_file":     "model/best_model.pt",
            "config_file":      "model/cpg_lif_snn_config.json",
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

    # ── 7. Timing-layer visualisation ───────────────────────────
    # Imported HERE, not at module scope: visualize_timing.py imports from this file,
    # so a top-level import would be circular.
    #
    # out_dir is passed as both the model dir and the output dir, so the
    # figures land in this run's own category folders rather than a nested
    # outputs/<run>/visualize/. It re-reads best_model.pt and the config from
    # disk, which doubles as an end-of-run check that those artifacts load and
    # run.
    if args.visualize and not args.dry_run:
        print("\n[7/7] Timing-layer visualisation ...")
        try:
            from visualize_timing import default_args, run_visualization
            # recon=0: phase 6 above has already written recon_*.png and
            # transition.png with this run's --recon_cycles, so regenerating
            # them here would be duplicate work. Run visualize_timing.py by
            # hand with --recon 1 to re-plot an existing checkpoint.
            run_visualization(out_dir, out_dir,
                              default_args(gaits_dir=args.gaits_dir, recon=0))
        except Exception as e:
            # Never let plotting lose a finished run: the checkpoint, config
            # and metrics are already on disk by this point.
            print(f"  visualisation FAILED ({type(e).__name__}: {e})")
            print(f"  training output is intact; rerun with "
                  f"'python visualize_timing.py --model_dir {args.out_dir or ''}'")

    print(f"\nDone — {out_dir.resolve()}")


if __name__ == "__main__":
    main()