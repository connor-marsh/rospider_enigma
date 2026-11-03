"""Bursting-LIF CPG -> stateful SNN -> multi-gait joint angles
===========================================================

A discrete bursting-LIF CPG (LIFGeneralArray + BurstingLIF + an inhibitory
coupling matrix; one step() == one timestep) emits raw spikes. They are the
network's ONLY input -- no phase channel, no gait period -- so phase has to be
held in state: hidden units get learnable membrane taus, initialiized
log-uniformly over [tau_min, tau_max], forming a temporal basis over "time since
last burst". Training is truncated BPTT over contiguous chunks with state carried
(detached) across chunk boundaries, like deployment.

--arch dense           one fully connected network; the ablation baseline.
--arch timing_grouped  CPG -> (timing layer | direct CPG drive) -> G disconnected
                       sub-networks, each two spiking layers plus a block-diagonal
                       analog readout; group g writes only its own gait-table
                       columns. --drive_source picks the drive: timing (learned
                       timing LIF layer, per-gait CPG->timing weights), cpg
                       (per-gait gain on each CPG channel) or cpg_lif (the same
                       gains through a fixed-threshold LIF). See the
                       TimingGroupedSNN docstring and --help.

CPG size is an argument: --n_cpg_neurons selects a coupling matrix from
CPG_W_BY_N and sizes the SNN input; nothing downstream assumes a particular N.

Leg / joint layout: LEG_COLS (quadruped, 8 joints: columns j and j+4 are one
leg's two joints) and HEXAPOD_LEG_COLS (18 joints, 3 per leg, LF/LM/LR/RF/RM/RR)
only group columns for the diagnostic plots. A run resolves them from
--n_cpg_neurons via default_leg_layout(), or from --leg_cols. Servo routing is
inference.py's concern; the gait-table column order is fixed.

Usage
    python train.py --epochs 300 --hidden 256
    python train.py --dry_run                         # data + plots, no training
    python train.py --arch timing_grouped --hidden 128
    python train.py --n_cpg_neurons 18 --drive_source cpg_lif --drive_lambda 1e-3
"""

import argparse
import json
import math
import os
import platform
import re
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
# 0. Gait-table column layout — EDIT HERE IF YOUR GAIT TABLES DIFFER
# ═══════════════════════════════════════════════════════════════════

# leg l -> (gait-table column for joint A, column for joint B)
LEG_COLS   = [(0, 4), (1, 5), (2, 6), (3, 7)]
N_LEGS     = 4
N_JOINTS   = 8

# Hexapod: 6 legs x 3 joints (coxa, femur, tibia), consecutive per leg; CSV leg order is
# LF, LM, LR, RF, RM, RR.
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
    """Which timing units feed each sub-network: a (G, K) list of indices.

    The rule: sub-network g takes every timing unit whose output columns overlap g's.
      timing per_joint + decoder per_leg   K=3, disjoint: 18 timing units, 6 decoders,
          each fed by its leg's 3 units (a decoder sees three phase references).
      timing per_leg + decoder per_joint   K=1, SHARED: 6 timing units, 18 decoders;
          a leg's 3 decoders get the same spike train. Legal but nearly pointless.
      matched shapes                       K=1, disjoint (the original model).

    K must be equal across groups because w1 is a dense (G, K, Hg) tensor; it is
    validated because --leg_cols is user-supplied.
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
    """Recursively coerce numpy scalars/arrays, Paths and tuples to JSON types (a bare
    json.dump choking on a np.float32 would abort a run or lose its results).
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

# All-to-all inhibitory coupling, one matrix per CPG size (add a row and
# --n_cpg_neurons accepts it); N=3/6 were ported from cpg_utils.py::BLIF_CPG.
# from_fb_weight (the burst-terminating kick) is fixed at CPG_FROM_FB_WEIGHT for every
# N, so there is no per-N regime to configure. analyse_cpg warns if burst phase
# offsets are far from evenly spaced.
CPG_FROM_FB_WEIGHT = -1_000_000.0

# ── tunable CPG parameters: name -> (default, help) ─────────
# One source of truth for argparse (--cpg_<name>), LIFCPGStepper's construction and
# the config dump (visualize_timing.replay_cpg rebuilds the oscillator from it).
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
# 2.  Burst detection  →  phase
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


# ═══════════════════════════════════════════════════════════════════
# 3. Gait tables
# ═══════════════════════════════════════════════════════════════════
# Loaded from CSV (np.loadtxt on gaits_dir/{name}.csv); the file list is picked from
# --n_cpg_neurons (4 -> quadruped, 6 -> hexapod) unless overridden with --gaits.
QUADRUPED_GAIT_FILES = ["bittle_wkF", "bittle_bk", "bittle_wkL", "bittle_wkR"]
HEXAPOD_GAIT_FILES = [
    "tripod", "tripod_backwards",
    "tripod_right", "tripod_left",
    "ripple", "ripple_backwards",
    "ripple_right", "ripple_left"
]
GAIT_FILES_BY_N = {4: QUADRUPED_GAIT_FILES, 6: HEXAPOD_GAIT_FILES}


# Output files are grouped into per-category subfolders of the run dir, routed by FILENAME so
# every call site passes a bare name and the layout lives here (train.py,
# visualize_timing.py and reorganize_outputs.py all consult this table). Left LOOSE:
# metrics.csv, training_curves.png, transition.png, rmse_heatmap.png. model/ entries are
# exact names (plus the epoch<N>.pt pattern) so nothing unexpected is swept in,
# including .onnx.data (only written with --export_onnx 1).
OUT_ROUTES = (
    # (kind, pattern, subfolder);  kind is "prefix" | "exact" | "regex"
    ("prefix", "recon_",                      "recons"),
    ("prefix", "timing_alignment_",           "timing_alignments"),
    ("prefix", "membranes_",                  "membrane_waveforms"),
    ("prefix", "phase_fold_",                 "phase_folds"),
    ("exact",  "best_model.pt",               "model"),
    ("regex",  r"epoch\d+\.pt",               "model"),
    ("exact",  "cpg_lif_snn_config.json",     "model"),
    ("exact",  "cpg_lif_snn_step.onnx",       "model"),
    ("exact",  "cpg_lif_snn_step.onnx.data",  "model"),
    ("exact",  "alignment_summary.png",       "misc_info"),
    ("exact",  "burst_threshold.png",         "misc_info"),
    ("exact",  "cpg_raster.png",              "misc_info"),
    ("exact",  "routing_matrices.png",        "misc_info"),
    ("exact",  "tau_distributions.png",       "misc_info"),
    # visualize_timing suffixes these with the checkpoint name for any non-best checkpoint,
    # so they never overwrite best_model's.
    ("prefix", "timing_summary",              "misc_info"),
    ("prefix", "alignment_scores",            "misc_info"),
)
OUT_CATEGORY_DIRS = tuple(dict.fromkeys(d for _, _, d in OUT_ROUTES))


def route_subdir(name):
    """Category subfolder for a filename, or "" to leave it loose."""
    for kind, pat, sub in OUT_ROUTES:
        if ((kind == "prefix" and name.startswith(pat))
                or (kind == "exact" and name == pat)
                or (kind == "regex" and re.fullmatch(pat, name))):
            return sub
    return ""


def out_path(out_dir, name):
    """Routed path for WRITING `name`, with its parent directory created."""
    p = Path(out_dir, route_subdir(name), name)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def in_path(in_dir, name):
    """Routed path for READING `name`, falling back to the loose location (older runs
    keep best_model.pt at the top level). If neither exists the routed path is
    returned, so errors point at where the file should now be.
    """
    routed = Path(in_dir, route_subdir(name), name)
    if routed.exists():
        return routed
    loose = Path(in_dir, name)
    return loose if loose.exists() else routed

# Anatomical name of the k-th joint in a leg, keyed by joints per leg.
JOINT_TYPE_NAMES = {2: ["shoulder", "knee"],
                    3: ["coxa", "femur", "tibia"]}


def joint_type_names(k):
    """Names for k joints-per-leg, falling back to generic labels."""
    return JOINT_TYPE_NAMES.get(k, [f"joint {i}" for i in range(k)])

def outputs_path(this_file_dir, rel=""):
    """this_file_dir/outputs[/rel], so a bare name like "test1" lands at outputs/test1.
    rel="" resolves to outputs itself.
    """
    return Path(this_file_dir, "outputs", rel) if rel else Path(this_file_dir, "outputs")


def load_gait_tables(names, gaits_dir):
    """Load gaits_dir/{name}.csv for each name (np.loadtxt, comma-delimited, float32).
    Validates equal column counts: upsample_gait_tables interpolates per column, and a
    mismatch there fails far from its cause.
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
    """(n_legs, leg_cols) for a known n_joints, keyed on n_joints alone (leg_cols is
    robot anatomy, not a function of the CPG size). n_joints=8 (quadruped): column j
    and j+4 share a circular phase offset. n_joints=18 (hexapod): leg-major, 3
    columns per leg (coxa, femur, tibia), legs LF/LM/LR/RF/RM/RR. Raises otherwise:
    pass --leg_cols.
    """
    # Keyed on n_joints alone: leg_cols is robot anatomy, which does not change when
    # --n_cpg_neurons goes from 6 to 18. n_cpg_neurons stays in the signature (and the
    # error message) for context.
    if n_joints == 8:
        return N_LEGS, [list(c) for c in LEG_COLS]
    if n_joints == 18:
        return HEXAPOD_N_LEGS, [list(c) for c in HEXAPOD_LEG_COLS]
    raise ValueError(
        f"No known leg layout for n_cpg_neurons={n_cpg_neurons}, "
        f"n_joints={n_joints}. Pass --leg_cols explicitly.")


def upsample_gait_tables(tables, names, target_rows=None, verbose=True):
    """Cubic-interpolate every table to a common row count (equal phase
    resolution across gaits, so per-gait loss isn't skewed by quantiization)."""
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
    targets : (G, T, 8) float32, normaliized to [-1, 1] with a single global
              min/max across all gaits (so one denormalization on the robot)
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
    """B independent read heads walking the CPG spike train. Each holds its own position,
    active gait and countdown to the next gait switch. `next_chunk(L)` returns L
    consecutive steps per head; the caller carries (and detaches) the SNN state across
    calls, so training matches deployment: one continuous stream with the gait changing
    underneath. A head that runs off the end is rewound to a random start and flagged
    in `reset_mask` so the caller can zero its state.

    The whole dataset is tiny, so spikes, targets and valid live on the training device
    and batches are gathered on-device; the per-head gait timeline stays in numpy
    (branchy, and switches are rare).
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

        # One code path for CPU and GPU (device=None keeps everything on the CPU).
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
        # NaNs zeroed here: `valid` is ~isnan(phase), so the mask already excludes those steps,
        # and NaNs in the loss would poison it via 0*NaN.
        ph = (np.zeros(len(spikes), dtype=np.float32) if phase is None
              else np.nan_to_num(np.asarray(phase, dtype=np.float32)))
        self.phase = torch.as_tensor(np.ascontiguousarray(ph),
                                     dtype=torch.float32, device=self.device)
        # Steps each head has advanced since its state was last zeroed (init_state() hands out zeros).
        self.warm_steps  = int(warm_steps)
        self.since_reset = np.zeros(self.B, dtype=np.int64)

        self.pos   = self.rng.integers(t_lo, t_hi, size=self.B)
        self.gait  = self.rng.integers(0, n_gaits, size=self.B)
        self.count = self.rng.integers(self.smin, self.smax, size=self.B)
        self._off  = {}                     # cached arange(L) per bptt

    def mark_all_reset(self):
        """Record that the network state of every head was just zeroed. Call whenever the
        caller re-initialiizes the model state outside the sampler (run_training does, at the
        start of every train and val pass): the warm-up mask is driven by `since_reset`,
        which the sampler otherwise only zeroes on its own rewinds. Without it the
        cold-start transient was scored every epoch, ~1/8 of an 8-chunk val, which is how
        val loss came out ~10x high under --synaptic.
        """
        self.since_reset[:] = 0

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

        # ── on-device gather ──
        # idx[t, b] = absolute timestep head b refers to at position t; (1,B) + (L,1) broadcasts
        # to (L,B), and indexing a (T,4) tensor with it gives (L,B,4), the layout the loop wants.
        dev   = self.device
        pos_t = torch.as_tensor(starts, dtype=torch.long, device=dev)
        idx   = pos_t[None, :] + self._offsets(L)[:, None]      # (L, B)
        g_t   = torch.as_tensor(g, dtype=torch.long, device=dev)

        x = self.spikes[idx]                        # (L, B, n_cpg_neurons)
        y = self.targets[g_t, idx]                              # (L, B, J)
        m = self.valid[idx]                                     # (L, B)
        ph = self.phase[idx]                                    # (L, B)

        # ── post-reset warm-up mask ──
        # A head whose state was just zeroed cannot know where it is in the cycle, so its first
        # `warm_steps` outputs are not trained on (the forward pass still runs; only the loss is
        # masked). Heads reset two ways: the sampler's own rewinds, and mark_all_reset, which
        # covers EVERY head at the start of each train and val pass (without it the cold-start
        # transient was scored every epoch).
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
    """Heaviside forward, fast-sigmoid surrogate backward, from plain tensor ops (a custom
    autograd.Function is a Dynamo graph-break source and an opaque box to the fuser).

        surr = x / (slope*|x| + 1)      d(surr)/dx = 1 / (slope*|x| + 1)^2
        return hard + (surr - surr.detach())          hard = (x > 0)

    The parentheses matter: `surr - surr.detach()` is exactly 0.0 in IEEE-754, so the
    forward is bit-exactly `hard`; unparenthesized, float non-associativity gives
    1.0 +/- 1 ulp. Backward: only the tracked `surr` term carries gradient.
    """
    surr = x / (slope * x.abs() + 1.0)
    hard = (x > 0).to(x.dtype)
    return hard + (surr - surr.detach())


# Weight init scales (internal: neither needs user input). _W_IN_INIT: CPG->timing weights;
# calibrate_gains rescales them, so only the shape and the sign of the mean matter.
# _W1_INIT: layer 1; uncalibrated and learnable, so a starting point (inert under
# --sub_ln l1/both, where LN is scale-invariant).
_W_IN_INIT = 0.5
_W1_INIT   = 0.5

# Calibration target bands, as FRACTIONS OF THE PERIOD (timesteps per cycle), not of the
# spike-count ceiling (n_cpg * cpg_rate): what matters is how much of the cycle, in time,
# carries input, and --cpg_refrac_main lengthens the period without changing the ceiling.
# The saturation cap stays on the ceiling (a hard count limit); see the main() calibration
# block. At the default refrac_main=1 (ISI 2) period == 2 * ceiling, giving 15-20 spk/cyc
# (plain) and 20-40 (min_count) at n_cpg=6.
_BAND_FRAC_PLAIN     = (0.125, 1.0 / 6.0)       # 15-20 spk/cyc at n_cpg=6, ISI=2
_BAND_FRAC_MIN_COUNT = (1.0 / 6.0, 1.0 / 3.0)   # 20-40 spk/cyc at n_cpg=6, ISI=2

# Highest fraction of the phase-blind ceiling (n_cpg * cpg_rate, where a unit's train is
# the OR of the CPG's and carries no phase) that calibration may target.
_SAT_FRAC = 0.6

# Drive-LIF membrane tau for cpg_lif, in burst widths. Fixed because both ends are pinned:
# it must forget a full cycle of silence (a unit sees one burst per cycle) yet not within
# a burst (so a low gain can integrate several spikes, giving sub-burst placement). At 2
# burst widths (burst_width = period / n_cpg) the residual after a cycle is exp(-n_cpg/2),
# 1.2e-4 at n_cpg 18, independent of burst width, while retention across one ISI is ~0.9;
# the gain to fire on spike 1..5 of a 5-spike burst is 1.00/0.52/0.37/0.29/0.24.
_TAU_DRIVE_BURSTS = 2.0

# Initial s_gate under cpg_lif: mean 1.1, std 0.01. The spread breaks an exact tie: all-ones
# makes the drive exactly uniform, PR's stationary maximum, so dPR/ds is exactly 0 and the
# penalty does nothing (under cpg, u's row-norm spread breaks it by accident). Mean 1.1,
# not 1.0: spike_fn uses a strict >, so s=1.0 fires on spikes 2 and 4 only (2/5) while
# s=1.01 fires on all 5, and 1.1 starts every unit cleanly "all on". Small spread because
# PR is rich-get-richer: in a toy, a wide N(0.8, 0.4) init let the init's strongest
# channel win 90% of the time against 52% for a tight one.
_S_GATE_INIT_MEAN   = 1.1
_S_GATE_INIT_SPREAD = 0.01

# Smallest gap kept between a voltage-mode bias and `thresh` (see clamp_bias_voltage).
_V_EPS = 1e-6

def init_beta_logit(shape, tau_min, tau_max, generator=None):
    """Log-uniform membrane time constants -> beta = exp(-1/tau), stored as a logit so
    training keeps beta in (0,1). The most important choice in the model: with no
    phase channel, only units that decay slowly enough to still carry the last burst
    can know where in the cycle they are; a homogeneous beta=0.9 cannot.
    """
    u   = torch.rand(shape, generator=generator)
    tau = torch.exp(math.log(tau_min) + u * (math.log(tau_max) - math.log(tau_min)))
    beta = torch.exp(-1.0 / tau).clamp(1e-4, 1 - 1e-6)
    return torch.log(beta / (1.0 - beta))


class DenseSNN(nn.Module):
    """Input: n_neurons binary CPG spikes per step (nothing else). Gait: integer index,
    used only for FiLM. Output: n_joints joint angles per step.

    Fully connected: every CPG spike reaches every hidden unit, both hidden layers and
    the readout are dense, with no grouping or per-gait input routing. Purely
    feedforward within a step; all memory is in the three leaky membranes, whose
    learnable per-unit taus (log-uniform over [tau_min, tau_max]) are the only way to
    hold position within a gait cycle, so tau_max must comfortably exceed one period.

    FiLM: `film1`/`film2` are nn.Embedding lookup tables mapping a gait index to a
    (gamma, beta) per hidden unit, applied AFTER LayerNorm (a gait flag concatenated
    onto the input would have its uniform part removed and its scale normaliized away
    by LN). The tables are over-allocated to `max_gaits` rows (unused rows are identity
    modulation and get no gradient), so parameter shapes do not depend on the gait
    count.

    State: mem1, mem2 (B, H); memo (B, Ho), which equals H unless --readout_hidden
    narrows it.

    readout_hidden, tau_readout_max, bias_mode, hidden_reset, sub_ln and sub_film mean
    what they mean in TimingGroupedSNN, so this ablation differs from the grouped model
    only in the factorization. Defaults are the values this class used before the
    options existed, so old checkpoints reconstruct as what they ran.
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
        # None = as wide as the hidden layers (what this class did before the option existed).
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
        # Stored so export_onnx can size the dummy input.
        self.n_neurons = n_neurons
        self.n_joints  = n_joints

        # ── layer 1: all CPG spikes -> all hidden units ─────────────
        # Scale 0.8: one CPG neuron fires per step, so cur1 is a single row of w_in; LayerNorm
        # follows, so this mostly sets gradient scale.
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
        # b_read has no voltage equivalent (memo is analog, never spikes), so it exists only in
        # "current" mode; b_out carries the output offset.
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
    """CPG spikes -> TIMING layer -> G disconnected sub-networks.

        x          (B, n_neurons)  CPG spikes, at most one fires per step
        timing     (B, n_timing)   LIF; per-gait weights from the CPG spikes
        sub-net g  (B, Hg) x2 + memo, driven by ITS OWN K timing units
        y          (B, n_joints)   group g writes only its own columns

    G and n_timing are independent: `group_cols[g]` is the output columns
    sub-network g owns, `timing_map[g]` the K timing units feeding it (from
    --decoder_shape / --timing_shape via build_partition / build_timing_map).
    Sub-networks never interact: a timing unit shared by several is a shared
    INPUT, not a shared parameter, so layers 2+ and the readout stay block diagonal.

    Per-gait routing: `w_in_gait` is Embedding(max_gaits, n_neurons * n_timing) and
    `b_t` a per-gait bias, so each gait has its own free routing matrix. A shared
    router with a per-gait FiLM gate was tried and reverted (the learned timing
    phases came out identical across gaits; see git history). Routing is not
    constrained to a permutation: a tripod needs three legs on one phase.

    bias_mode: "current" = additive bias current; "voltage" (default) = per-unit
    offset to the spiking threshold, with no bias current anywhere in the spiking
    path (what makes natural gating possible); "none". The reset still subtracts
    the fixed `thresh`, not the offset threshold (resetting by the offset runs
    away). b_read exists only in "current" mode (memo is analog and never spikes).

    No LayerNorm on the timing layer: over a handful of units it removes the
    common-mode "some CPG neuron fired" signal and turns the bias into tonic drive.
    `calibrate_gains` sets the firing regime instead.

    NATURAL GATING (the default setup): gate_mode="none", bias_mode="voltage",
    film_mode="gamma_only" and hidden_reset="zero" leave no way for a hidden unit
    to fire on a timing-silent step (no bias current, no FiLM beta, no residual
    after a subtractive reset). --bias_mode current, --film_mode gamma_beta and
    --hidden_reset subtract each reopen one leak, for A/B; check_bias_voltage.py
    reports which units can fire unprompted. With a leak open, the task loss is
    partly insensitive to WHEN the timing spikes land.

    Inside the sub-networks LN (`sub_ln`) uses elementwise_affine=False so no
    parameter is tied across groups; FiLM supplies the per-group per-gait affine.

    State (9 tensors), in this order:
        mem_timing (B, n_timing)  timing layer; unused under cpg / cpg_lif
        since_upd  (B, G)         vestigial; kept only so the ONNX signature is fixed
        mem1, mem2 (B, G, Hg)
        memo       (B, G, H_o)    narrower than Hg; see the readout note in __init__
        syn1, syn2 (B, G, Hg)     second-order synaptic currents (--synaptic)
        syno       (B, G, H_o)    zero when synaptic="none"
        mem_drive  (B, G, n_cpg)  drive LIF membrane (cpg_lif only), else zero
    """

    arch = "timing_grouped"
    # syn1/syn2/syno are the second-order synaptic currents. Allocated
    # unconditionally, like the alpha logits and like the vestigial
    # since_upd, so neither the state arity nor the exported ONNX signature
    # depends on --synaptic. They stay exactly zero when synaptic="none".
    # mem_drive is the drive LIF's membrane, (B, G, C). Allocated
    # unconditionally like syn*/since_upd so neither the state arity nor the
    # exported ONNX signature depends on --drive_source; it stays exactly
    # zero except under cpg_lif.
    state_names_in  = ("mem_timing_in", "since_upd_in",
                       "mem1_in",  "mem2_in",  "memo_in",
                       "syn1_in",  "syn2_in",  "syno_in",
                       "mem_drive_in")
    state_names_out = ("mem_timing_out", "since_upd_out",
                       "mem1_out", "mem2_out", "memo_out",
                       "syn1_out", "syn2_out", "syno_out",
                       "mem_drive_out")

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
                 drive_source="timing", tau_drive=None,
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
        if drive_source not in ("timing", "cpg", "cpg_lif"):
            raise ValueError(f"drive_source must be timing|cpg|cpg_lif, got "
                             f"{drive_source!r}")
        if drive_source in ("cpg", "cpg_lif") and gate_mode != "none":
            raise ValueError(
                "drive_source cpg / cpg_lif requires gate_mode='none' (there is no timing "
                "unit to gate on).")

        group_cols = (build_group_cols(n_timing, n_joints=n_joints)
                      if group_cols is None else
                      [list(g) for g in group_cols])

        G = len(group_cols)
        T = int(n_timing)
        # timing_map[g] = timing units feeding sub-network g; None means one per
        # sub-network, in order (what configs predating the arg were).
        if timing_map is None:
            if T != G:
                raise ValueError(
                    f"group_cols has {G} groups but n_timing={T} and no timing_map was given; "
                    f"pass timing_map (see build_timing_map) for a fan-in other than 1.")
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
        # Under cpg / cpg_lif the drive channels are the CPG's own, so n_timing is
        # n_neurons (timing_only returns the CPG train there; diagnostics key off this).
        self.n_timing   = T if drive_source == "timing" else int(n_neurons)
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
        # Gentler surrogate on the timing layer, where a dead unit is catastrophic.
        self.timing_slope = float(slope if timing_slope is None
                                  else timing_slope)
        self.thresh     = thresh
        self.sub_ln     = sub_ln
        self.sub_film   = sub_film
        self.film_mode  = film_mode
        self.synaptic   = synaptic
        self.drive_source = drive_source
        self.uses_timing  = (drive_source == "timing")
        # cpg_lif shares every parameter with cpg and adds the drive LIF.
        self.drive_lif    = (drive_source == "cpg_lif")
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

        # ── timing: per-gait CPG->timing weights ─────────────────
        # Each gait has its own free (n_neurons, n_timing) matrix and bias. No FiLM here:
        # with a free per-gait W, gamma is absorbable into W (the per-gait bias is not,
        # so it comes from the embedding). Init scale is arbitrary (calibrate_gains
        # rescales it) but the mean must be positive: calibration cannot rescue a
        # net-negative unit.
        # Timing layer skipped under cpg / cpg_lif, which also makes calibrate_gains /
        # reinit_timing_units (keyed on hasattr w_in_gait) no-ops.
        if self.uses_timing:
            self.w_in_gait = nn.Embedding(max_gaits, self.n_neurons * T)
            nn.init.normal_(self.w_in_gait.weight,
                            mean=_W_IN_INIT, std=_W_IN_INIT)

            # Per-gait excitability (see BIAS MODES). "current" = tonic drive, the only way a
            # unit fires BETWEEN bursts under timing_reset=zero; "voltage" = threshold offset.
        if self.uses_timing:
            if bias_mode == "current":
                self.b_t = nn.Embedding(max_gaits, T)
                nn.init.zeros_(self.b_t.weight)
            elif bias_mode == "voltage":
                self.v_t = nn.Embedding(max_gaits, T)
                nn.init.zeros_(self.v_t.weight)
            self.beta_t_logit = nn.Parameter(
                init_beta_logit((T,), tau_timing_min, tau_timing_max))

        # ── sub-network layer 1: K binary spikes -> Hg units ──────
        # Fixed init, not calibrated (a few dead units out of Hg are noise). Shape
        # (G, K, Hg), contracted over K in `step`; 1/sqrt(K) keeps the injection
        # variance independent of fan-in (exactly 1.0 at K=1). Created only when
        # uses_timing; the cpg paths have s_gate/u instead.
        if self.uses_timing:
            self.w1 = nn.Parameter(
                torch.randn(G, K, Hg) * (_W1_INIT / math.sqrt(K)))
        else:
        # ── drive_source cpg / cpg_lif: skip the timing layer ─────
        # Sub-networks are driven straight off the CPG channels, whose phases are fixed
        # and evenly tiled. The per-gait selection w_in_gait provided is factorized:
        #   s_gate (max_gaits, G*n_ch)  gain of channel c on sub-network g, per gait,
        #       applied BEFORE the sum over channels, so 0 removes that channel from that
        #       (gait, sub-network) exactly.
        #   u (G, n_ch, Hg)  the layer-1 weights (w1's counterpart), gait-SHARED: a gait
        #       can scale a channel's hidden-space direction but not rotate it (rank 1
        #       over gait; ~7.5x fewer parameters than a full per-gait w1).
        #   cur1[b,g,h] = sum_c x[b,c] * s_gate[gait,g,c] * u[g,c,h]
        # n_ch, not C: C is the OUTPUT-column count and sizes w_out / b_out below
        # (rebinding it here once routed every joint through sub-network 0).
            n_ch = self.n_neurons
            self.s_gate = nn.Embedding(max_gaits, G * n_ch)
            nn.init.ones_(self.s_gate.weight)     # every channel on
            if self.drive_lif:
                # See _S_GATE_INIT_*. cpg_lif only: under cpg, u's row norms already break the tie.
                with torch.no_grad():
                    self.s_gate.weight.fill_(_S_GATE_INIT_MEAN)
                    self.s_gate.weight.add_(
                        torch.randn_like(self.s_gate.weight)
                        * _S_GATE_INIT_SPREAD)
                    self.s_gate.weight.clamp_(min=1e-3)

            # Dense at _W1_INIT with no 1/sqrt(C) fan-in factor: CPG channels never co-fire, so
            # a hidden unit receives ONE weight per spiking step, not a sum. Dense (not
            # sparse) because a unit needs some positive weight to ever fire (fan-in 2
            # leaves ~25% dead) and argmax_c u[g,c,h] already tiles phase across units.
            self.u = nn.Parameter(torch.randn(G, n_ch, Hg) * _W1_INIT)

            if self.drive_lif:
                # ── drive_source cpg_lif ─────────────────────────────────
                # A LIF between the gain and u: one unit per (g, c), each seeing exactly one
                # channel, so it can thin or delay that channel's burst but never move it.
                # Why: under cpg, s_gate and u are multiplicatively redundant (halve s, double u
                # = identical forward), so a penalty on s is meaningless. The threshold bounds
                # that escape: dropped spikes lose information u cannot restore (timing within
                # the burst, spike count), and below s ~0.24 a channel is silent. Spike count is
                # a staircase in s, so the penalty acts on |s_gate| instead (smooth).
                # FIXED threshold: a learnable one reopens the escape.
                # Zero reset: makes a sub-network's C units mutually exclusive (one channel each,
                # CPG channels never coincide), so they collapse to one effective timing neuron.
                # FIXED tau (buffer): both ends are pinned by the burst structure.
                if tau_drive is None:
                    raise ValueError(
                        "drive_source='cpg_lif' needs tau_drive (main() passes "
                        "_TAU_DRIVE_BURSTS * period / n_cpg_neurons; the constructor does not know "
                        "the period).")
                self.tau_drive = float(tau_drive)
                self.register_buffer(
                    "drive_decay",
                    torch.tensor(math.exp(-1.0 / max(self.tau_drive, 1e-6)),
                                 dtype=torch.float32),
                    persistent=False)

        # ── sub-network layer 2: block diagonal (G, Hg, Hg) ───────
        self.w2 = nn.Parameter(torch.randn(G, Hg, Hg) / math.sqrt(Hg))

        # ── block-diagonal analog readout ─────────────────────────
        # Ho, not Hg: memo is a bank of low-pass filters (one tau per unit), so its width
        # is the temporal basis for the output waveform, not the output width. Ho == C
        # would leave only C distinct (projection, tau) pairs per group.
        # C must be the OUTPUT-column count (asserted: a wrong C still trains, with 17
        # of 18 sub-networks disconnected).
        assert C == len(group_cols[0]), (
            f"C was rebound: expected len(group_cols[0])="
            f"{len(group_cols[0])}, got {C}. w_out/b_out would be sized "
            f"wrong and out_perm would select only sub-network 0.")
        self.w_read = nn.Parameter(torch.randn(G, Hg, Ho) / math.sqrt(Hg))
        self.w_out  = nn.Parameter(torch.randn(G, Ho, C) / math.sqrt(Ho))
        self.b_out  = nn.Parameter(torch.zeros(G, C))

        # b_read has no voltage equivalent (memo is analog and never spikes), so it exists
        # only in "current" mode; b_out carries the output offset.
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

        # ── per-gait readout bias current ────────────────────────
        # A per-gait DC current into `memo` (not added to y, so it fades in over the
        # readout tau at a gait switch instead of stepping). Needed because nothing else
        # in the sub-networks has a gait-dependent constant under film_mode=gamma_only,
        # so gaits with different MEAN angles would have to express it through spike
        # rate. memo is analog and never spikes, so this adds no tonic drive to a
        # spiking layer. Created only when enabled.
        self.readout_gait_bias = bool(readout_gait_bias)
        if self.readout_gait_bias:
            self.b_read_gait = nn.Embedding(max_gaits, G * Ho)
            nn.init.zeros_(self.b_read_gait.weight)

        # ── second-order synaptic filter (snnTorch Synaptic / Loihi CUBA) ──
        #     syn = alpha * syn + injection;   mem = beta * mem + syn
        # (mem integrates the NEW syn, so the membrane still moves on the spike step,
        # just by less.) One injection is spread over many steps: a smaller output step
        # per spike, and hidden membranes stay near threshold for a window, so the
        # surrogate gradient stays alive on silent steps. It decays to zero in true
        # silence, so spike placement still matters. alpha logits are always allocated
        # (no gradient when synaptic="none").
        self.alpha1_logit = nn.Parameter(
            init_beta_logit((G, Hg), tau_syn_min, tau_syn_max))
        self.alpha2_logit = nn.Parameter(
            init_beta_logit((G, Hg), tau_syn_min, tau_syn_max))
        self.alphao_logit = nn.Parameter(
            init_beta_logit((G, Ho), tau_syn_min, tau_syn_max))

    def drive_strength(self):
        """(n_gaits, G, C) drive of CPG channel c on sub-network g, per gait.

        cpg:      |s_gate| * ||u[g,c,:]||_1. s_gate and u enter the forward pass only as
                  a product, so neither alone says how much current a channel delivers.
        cpg_lif:  |s_gate|. A fixed-threshold LIF sits between s_gate and u, so u gets
                  binary spikes and the penaliized quantity is the gain itself.

        Shared by `drive_penalty` and the reporters, so the penaliized and reported
        numbers are the same by construction. Only the first n_gaits rows are used.
        """
        S = self.s_gate.weight[:self.n_gaits].abs().view(
            self.n_gaits, self.G, self.n_neurons)               # (n_gaits,G,C)
        if self.drive_lif:
            return S
        unorm = self.u.abs().sum(dim=2)                         # (G, C)
        return S * unorm.unsqueeze(0)

    @torch.no_grad()
    def drive_spikes_only(self, x_seq, gait_seq, mem_d=None):
        """Replay just the drive LIF layer: (L, B, G, C) binary spikes, or None outside
        cpg_lif. The cpg_lif counterpart of `timing_only`; "channel c delivers nothing to
        sub-network g" is the exact statement "unit (g,c) emitted zero spikes".
        """
        if not self.drive_lif:
            return None
        B = x_seq.shape[1]
        dev, dt = x_seq.device, x_seq.dtype
        if mem_d is None:
            mem_d = torch.zeros(B, self.G, self.n_neurons,
                                device=dev, dtype=dt)
        dec = self.drive_decay
        out = []
        for t in range(x_seq.shape[0]):
            s_g = self.s_gate(gait_seq[t]).view(-1, self.G, self.n_neurons)
            mem_d = dec * mem_d + x_seq[t].unsqueeze(1) * s_g
            spk = (mem_d > self.thresh).to(dt)
            mem_d = mem_d * (1.0 - spk)
            out.append(spk)
        return torch.stack(out)

    def drive_penalty(self, kind="pr"):
        """Sparsity pressure on the drive (cpg / cpg_lif; see drive_strength for what the
        drive is), per (gait, sub-network) over that row's C channels.

        kind="pr" (default): participation ratio (sum d)^2 / sum d^2, the effective
            number of channels used (1 = one channel, C = all equal); the same quantity
            drive_report prints as fanin. Scale-invariant, so shrinking every drive and
            compensating elsewhere cannot lower it. Rich-get-richer: it amplifies the
            leading channel, and exactly-uniform is its (unstable) maximum, broken by u's
            row-norm spread under cpg and _S_GATE_INIT_SPREAD under cpg_lif. A weight
            that dominates the task gradient lets init pick the winner, so start small
            and rely on --lambda_hold / --lambda_warmup.
        kind="l1": mean of the drive. Under cpg it FAILED (the network shrank every drive
            together and compensated via u and the hidden thresholds: max channel share
            0.08 at fanin ~15.8), and its gradient ||u[g,c,:]||_1 barely varies across
            channels. Under cpg_lif the drive is |s_gate| and it is UNTESTED; the escape
            is bounded: s_gate can fall from 1.1 to ~0.25 (one late-burst spike) while u,
            FiLM gamma and v1 make up the charge, but dropped spikes lose timing within
            the burst and spike count.

        Returns a 0-d tensor; zero when there is no s_gate.
        """
        if self.uses_timing:
            return torch.zeros((), device=self.w2.device, dtype=self.w2.dtype)
        d = self.drive_strength()                       # (n_gaits, G, C)
        if kind == "l1":
            return d.mean()
        if kind != "pr":
            raise ValueError(f"drive penalty kind must be pr|l1, got {kind!r}")
        s1 = d.sum(dim=2)
        s2 = d.pow(2).sum(dim=2).clamp(min=1e-12)
        # Mean over (gait, sub-network), divided by C, so the weight needs no rescaling
        # when n_gaits, G or n_cpg_neurons change.
        return (s1.pow(2) / s2).mean() / float(self.n_neurons)

    @torch.no_grad()
    def clamp_bias_voltage(self):
        """Project v_t / v1 / v2 into [-thresh, thresh - _V_EPS].

        The upper bound matters: a voltage-mode unit compares against `thresh - v`, so
        v >= thresh gives a non-positive threshold and the unit fires every step with no
        way back. The lower bound only caps the threshold at 2*thresh.

        Applied to `.data` after the optimiizer step, not in `step()`, where a clamp
        would give out-of-range parameters exactly zero gradient and trap them. Not
        called from load_run: a checkpoint must reconstruct as it was
        (check_bias_voltage.py finds out-of-range biases).
        """
        if self.bias_mode != "voltage":
            return
        lo, hi = -self.thresh, self.thresh - _V_EPS
        # v_t exists only when there IS a timing layer (drive_source="timing").
        v_t = getattr(self, "v_t", None)
        tensors = [self.v1, self.v2]
        if v_t is not None:
            tensors.append(v_t.weight)
        for p in tensors:
            p.data.clamp_(lo, hi)

    def init_state(self, batch, device, dtype=torch.float32):
        z = lambda w: torch.zeros(batch, self.G, w, device=device, dtype=dtype)
        mem_t = torch.zeros(batch, self.n_timing, device=device, dtype=dtype)
        # since_upd: vestigial (the update counter of the removed gate_mode="freeze"); kept
        # only so the ONNX signature keeps its arity and names.
        since = torch.zeros(batch, self.G, device=device, dtype=dtype)
        # memo is Ho wide, not Hg; syn1/syn2/syno mirror mem1/mem2/memo.
        return (mem_t, since, z(self.Hg), z(self.Hg), z(self.Ho),
                z(self.Hg), z(self.Hg), z(self.Ho), z(self.n_neurons))

    def _timing(self, x, gait, mem_t):
        """One timestep of the timing layer: x (B, n_neurons) -> spk_t (B, T). `step` maps
        these onto sub-networks through `timing_map`. Factored out so calibration and
        diagnostics can run it without the sub-networks or the compiled `step`.

        timing_reset="zero" matches the CPG (LIFGeneralArray does v[spike]=0). A
        subtractive reset leaves mem - thresh behind, which can re-fire on the next
        silent step: ISIs of 2,1,1,2,... and ~50% more spikes per burst than the CPG.
        """
        W   = self.w_in_gait(gait).view(-1, self.n_neurons, self.n_timing)
        cur = torch.bmm(x.unsqueeze(1), W).squeeze(1)
        if self.bias_mode == "current":
            cur = cur + self.b_t(gait)
        # Comparison threshold only; the reset still subtracts the fixed self.thresh
        # (resetting by the offset runs away).
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
        """x : (B, n_neurons) CPG spikes this step;  gait : (B,) int64
        state : (mem_timing, since_upd, mem1, mem2, memo, syn1, syn2, syno, mem_drive)

        Returns (y, state, aux), aux = (spk_timing,): lets the spike penalty see the
        spikes without a second forward pass (the ONNX wrappers discard it).
        """
        (mem_t, since, mem1, mem2, memo,
         syn1, syn2, syno, mem_d) = state
        G, Hg = self.G, self.Hg

        if self.uses_timing:
            spk_t, mem_t = self._timing(x, gait, mem_t)      # (B, T)
            # ---- timing -> sub-network routing: timing_map is (G, K), gathering each
            # sub-network's K timing spikes ----
            spk_sel = spk_t[:, self.timing_map]              # (B, G, K)
        else:
            # drive_source cpg / cpg_lif: no timing layer to run. mem_t passes through (all-zero;
            # it only holds the state/ONNX signature fixed). spk_t is the CPG vector itself (an
            # input, no gradient), reported in aux for diagnostics. main() forces the spike
            # objective to none in these modes.
            spk_t   = x
            spk_sel = None

        # ---- event gate ----
        # gate_mode "none" (default) is NATURAL GATING (see the class docstring). "decay" gates
        # the input and sub-net spikes explicitly, kept for A/B. Not the default: it destroys
        # the magnitude-to-spike-count expansion of subtractive reset, and it changes the
        # timing layer's gradient. The gate is spk_t, so it multiplies spk_t in five extra
        # places and a path with k gate factors has gradient (1+k)*s^k: up to 6x on a spiking
        # step, but EXACTLY ZERO on a silent step, which removes the "fire HERE" signal. Dead
        # timing units are therefore MORE likely under it.
        gated = self.gate_mode != "none"
        # A sub-network updates when ANY of its K timing units fires: the OR over its selected
        # spikes (amax keeps the gate strictly 0/1; a no-op at K=1).
        gate = spk_sel.amax(dim=-1, keepdim=True) if gated else None
        dec1 = torch.sigmoid(self.beta1_logit)
        dec2 = torch.sigmoid(self.beta2_logit)
        deco = torch.sigmoid(self.betao_logit)

        # Injection -> membrane, first order (straight in) or second order (via a decaying
        # synaptic current; snnTorch ordering, mem integrates the NEW syn). Written once so
        # the three layers cannot drift; syn passes through unchanged (zero) in first order.
        def integrate(mem, syn, cur, dec, alpha_logit, second_order):
            if not second_order:
                return dec * mem + cur, syn
            syn = torch.sigmoid(alpha_logit) * syn + cur
            return dec * mem + syn, syn

        syn_hidden  = self.synaptic in ("hidden", "all")
        syn_readout = self.synaptic == "all"

        # ---- sub-net layer 1 ----
        # In "voltage" mode the bias is not in the injection, so it cannot inflate it. The
        # contraction over K is explicit in both gate modes: with K>1 the injection is
        # sum_k spk_k * w1[g,k], which is NOT gate * sum_k w1[g,k] unless all K units fire.
        b1 = self.b1 if self.bias_mode == "current" else 0.0
        if self.uses_timing:
            cur1 = torch.einsum("bgk,gkh->bgh", spk_sel, self.w1) + b1
        else:
            # Gate each CPG channel per (gait, sub-network) BEFORE the sum over channels, so
            # s_gate[gait,g,c]=0 removes exactly that channel from that sub-network and gait.
            s_g    = self.s_gate(gait).view(-1, G, self.n_neurons)  # (B,G,C)
            scaled = x.unsqueeze(1) * s_g                           # (B,G,C)
            if self.drive_lif:
                # One LIF per (g, c): fixed threshold, zero reset. Its binary OUTPUT is what reaches u,
                # which bounds how far a penalty on s_gate can be escaped (see __init__).
                mem_d = self.drive_decay * mem_d + scaled
                spk_d = spike_fn(mem_d - self.thresh, self.slope)
                mem_d = mem_d * (1.0 - spk_d)            # zero reset, always
                drive = spk_d
            else:
                drive = scaled
            cur1   = torch.einsum("bgc,gch->bgh", drive, self.u) + b1
        if self.sub_ln in ("l1", "both"):
            cur1 = self.ln1(cur1)
        if self.sub_film in ("l1", "both"):
            v1 = self.film1(gait).view(-1, 2, G, Hg)
            # gamma always; beta only in gamma_beta. Gamma is multiplicative, so it vanishes where
            # the injection is zero (natural gating); beta is added every step, i.e. tonic drive.
            # The beta half stays allocated either way so checkpoint shapes do not depend on mode.
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
            # AFTER the gate, deliberately: a DC offset, not an injection (gating it would make it
            # appear only on spike steps).
            curo = curo + self.b_read_gait(gait).view(-1, G, self.Ho)
        memo, syno = integrate(memo, syno, curo, deco,
                               self.alphao_logit, syn_readout)

        # `since` passes through untouched (kept only for the ONNX signature; see init_state).

        y_grp = torch.einsum("bgh,ghc->bgc", memo, self.w_out) + self.b_out
        y     = y_grp.flatten(1).index_select(1, self.out_perm)
        return (y, (mem_t, since, mem1, mem2, memo,
                    syn1, syn2, syno, mem_d),
                (spk_t,))

    def forward(self, x_seq, gait_seq, state=None, return_aux=False):
        """x_seq (L, B, n_neurons), gait_seq (L, B). Returns (y_seq, state), plus
        (spk_t_seq,) when return_aux=True.
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

    @torch.no_grad()
    def timing_only(self, x_seq, gait_seq, mem_t=None):
        """(L, B, n_timing) timing spikes from CPG spikes via the eager `_timing`: exact,
        and cheap next to the sub-networks it skips.
        """
        if not self.uses_timing:
            # cpg / cpg_lif: the sub-networks are driven off the CPG's own channels (through the
            # drive LIF under cpg_lif), so return the CPG train unchanged. This keeps
            # timing_report and measure_rates working without a timing layer.
            return x_seq
        B = x_seq.shape[1]
        if mem_t is None:
            # n_timing, NOT G: this is the timing layer's own membrane.
            mem_t = torch.zeros(B, self.n_timing, device=x_seq.device,
                                dtype=x_seq.dtype)
        out = []
        for t in range(x_seq.shape[0]):
            spk, mem_t = self._timing(x_seq[t], gait_seq[t], mem_t)
            out.append(spk)
        return torch.stack(out)

    def param_breakdown(self):
        """Grouped parameter counts, for the startup print."""
        n = lambda *ps: int(sum(p.numel() for p in ps
                                 if p is not None))
        g = lambda name: getattr(self, name, None)
        # Under cpg / cpg_lif w1 is replaced by s_gate + u. s_gate is counted under "timing"
        # (it took over the per-gait routing job), keeping block totals comparable.
        return {
            "timing":  (n(self.w_in_gait.weight, self.beta_t_logit,
                          *(w.weight for w in (g("b_t"), g("v_t"))
                            if w is not None))
                        if self.uses_timing else n(self.s_gate.weight)),
            "sub_l1":  n(self.w1 if self.uses_timing else self.u,
                         self.beta1_logit, self.alpha1_logit,
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
    As SingleStepONNX but for TimingGroupedSNN's 9-tensor state.

    Written as an explicit signature rather than *state: torch.onnx.export
    traces varargs unreliably, and the input names have to line up
    positionally with `model.state_names_in` anyway.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spikes, gait, mem_timing, since_upd, mem1, mem2, memo,
                syn1, syn2, syno, mem_drive):
        y, (mt, su, m1, m2, mo, s1, s2, so, md), _ = self.model.step(
            spikes, gait, (mem_timing, since_upd, mem1, mem2, memo,
                           syn1, syn2, syno, mem_drive))
        return y, mt, su, m1, m2, mo, s1, s2, so, md


# ═══════════════════════════════════════════════════════════════════
# 7a. Keeping the small spiking layers alive
# ═══════════════════════════════════════════════════════════════════
# The timing layer has no LayerNorm, so nothing holds its input current where a threshold-1.0
# LIF fires. Three mechanisms: calibrate_gains (once, before training: never start dead or
# loud), SpikeObjective (every step: shapes the spike train) and reinit_timing_units (on
# detection: rescues what failed). Both extremes are fatal: a dead timing unit gives the w1
# slots it feeds EXACTLY zero gradient, and a saturated one fires on every CPG spike (its
# train is the CPG's OR, no phase) while sitting where the surrogate gradient is ~1e-6, which
# Adam's scale-invariance means a larger --spike_stats_lambda cannot fix.


@torch.no_grad()
def measure_rates(model, spikes, n_gaits, device,
                  t0=0, n_steps=None, period=254.0):
    """Spikes per cycle for every (gait, timing unit), replaying the CPG at batch 1.
    Returns (n_gaits, n_timing); forward passes only.
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
    """Scale each timing unit's CPG->timing weight column so its firing rate starts
    inside [lo, hi] spikes per cycle, by bisection on measured rates (no user scale
    factor). Writes into `w_in_gait`.

    Rate is monotone non-decreasing in a unit's own gain (a step function: it jumps
    by whole spikes), hence a band rather than an exact target. Units are independent
    (unit l's current depends only on column l of each gait's matrix), so all bisect
    in parallel.

    Weights, not bias: a bias is tonic drive that also exists during CPG silence, so
    raising it makes a unit alive but unrelated to the rhythm; scaling weights
    amplifies the phase preference it already has. Gain cannot fix a wrong sign (a
    net-negative unit rides to g_hi and gives up; the positive-mean init prevents
    that); such units are reported.

    per_gait=True (default) bisects a separate gain per (gait, unit). per_gait=False
    is the old one-gain-per-unit behavior, bisected against the MINIMUM rate across
    gaits, which is a saturation generator: nothing bounded the maximum, so one
    weakly driven gait dragged the others up (max rates of 5-48 spk/cyc against a
    [1, 5] band over 24 draws). Training cannot undo that: a unit at the ceiling sits
    where the surrogate gradient is ~1e-6, and Adam's scale-invariance means raising
    --spike_stats_lambda 10,000x moved the gain 1.5%. Rows of w_in_gait past n_gaits
    are left at gain 1.

    `cpg_rate_hint` (the CPG's spikes per cycle, from cpg_spike_stats) only feeds the
    saturation warning: a unit can reach at most n_cpg * cpg_rate, one spike per CPG
    spike.
    """
    if not hasattr(model, "w_in_gait"):
        return {}

    # G here is the timing-unit count (model.n_timing, not model.G); w_in_gait is indexed
    # by timing unit.
    MG, n_cpg, G = model.max_gaits, model.n_neurons, model.n_timing
    # (max_gaits, n_cpg, n_timing) view; column l is timing unit l's inputs.
    W = model.w_in_gait.weight.data.view(MG, n_cpg, G)
    base = W.clone()

    # Bisection brackets: one per (gait, unit) in per-gait mode, one per unit otherwise;
    # everything below is written against `shape` so both modes share one loop.
    shape = (n_gaits, G) if per_gait else (G,)
    lo_g = torch.full(shape, g_lo, device=device)
    hi_g = torch.full(shape, g_hi, device=device)

    def rates_at(gain):
        """Write `gain` into w_in_gait and measure; returns a tensor shaped like `gain`."""
        if per_gait:
            # (n_gaits, G) -> (max_gaits, 1, G); rows past n_gaits are never indexed, so stay at 1.
            full = torch.ones(MG, G, device=device, dtype=base.dtype)
            full[:n_gaits] = gain
            W.copy_(base * full.unsqueeze(1))
        else:
            # gain (G,) broadcasts over (max_gaits, n_cpg, G): each unit's whole column, every gait.
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

    # sqrt(lo*hi) is safe to return: once a unit is in band neither bound updates, so its
    # midpoint is frozen at the value measured in band.
    final_gain  = torch.sqrt(lo_g * hi_g)
    final_rates = rates_at(final_gain)
    ok = ((final_rates >= lo * 0.5) & (final_rates <= hi * 2.0))

    # rate_by_gait is always the full (n_gaits, G) measurement, so both modes stay
    # comparable in the config dump.
    rate_by_gait = final_rates if per_gait else torch.as_tensor(
        measure_rates(model, spikes, n_gaits, device, period=period),
        device=device)

    tolist = lambda t: t.cpu().tolist()
    report = {"timing": {
        "per_gait":     bool(per_gait),
        "band":         [float(lo), float(hi)],
        "gain":         tolist(final_gain),      # (n_gaits, G) or (G,) by mode
        "rate_by_gait": tolist(rate_by_gait),    # always (n_gaits, G)
        # Original names and meanings kept so older configs still read; max_rate is the one to watch.
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

        # A unit at the ceiling fires once per CPG spike (the CPG's OR, no phase). The threshold is
        # _SAT_FRAC, not looser: 0.9 would have missed a (gait, unit) pair at 86% of the ceiling.
        ceiling = float(n_cpg) * float(cpg_rate_hint) if cpg_rate_hint else None
        if ceiling:
            sat = (rate_by_gait >= _SAT_FRAC * ceiling).nonzero().cpu().tolist()
            if sat:
                print(f"      WARNING: (gait, unit) {sat} are at >={_SAT_FRAC:g} x the "
                f"{ceiling:.0f} spk/cyc ceiling, so they carry little or no phase, and "
                f"training will not fix it (surrogate gradient ~1e-6).")

        bad = ok.logical_not().nonzero().cpu().tolist()
        if bad:
            label = "(gait, unit)" if per_gait else "unit"
            print(f"      WARNING: timing {label} {bad} could not be brought into "
            f"[{lo}, {hi}] spk/cyc by scaling alone (likely net-negative input "
            f"current); if they stay dead, check the _W_IN_INIT positive mean.")
    return report


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


# ── Spike-statistics objectives (strategy pattern) ─────────
# What the spike train SHOULD look like is open, so the objective is swappable: add a
# subclass with a `name` and it becomes --spike_objective <name>. All share one signature,
# so run_training never branches:
#     penalty(spk, gait, phase, mask) -> scalar
#     spk (L, B, U) surrogate spikes from step()'s aux;  gait (L, B) int64;
#     phase (L, B) in [0,1), NaNs zeroed;  mask (L, B) 1 where the sample counts
#     (valid phase AND past the post-reset warm-up).
# Statistics are grouped per (gait, unit), never averaged over the batch (that would hide a
# unit misbehaving for one gait); gaits absent from a chunk are masked out, not counted silent.


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
    """Fire LIKE THE CPG: match its spikes per cycle and burst tightness. Two terms,
    because rate alone cannot tell one tight burst from a lone spike every 35 steps:
      rate  ((rate - target)/target)^2, two-sided, relative so lam is scale-free.
      conc  relu(target_R - R)^2, one-sided; R = |sum_t spk_t exp(i 2pi phase_t)| /
            sum_t spk_t is the circular concentration of the unit's spike phases (near
            1 = one burst per cycle). `phase` is constant and `spk` carries the
            surrogate gradient, so it is differentiable as written.
    Limitation: R uses the fundamental only, so two bursts at opposite phases cancel to
    R ~ 0 (wrong for a genuinely two-burst gait). NOT compatible with gated
    sub-networks, which need spikes wherever the output changes. Kept for comparison.
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
        # Only shape units that are already firing (R is meaningless for a silent unit; the rate
        # term pushes it up). Detached hard mask.
        alive = (cnt > 1.0).to(spk.dtype).detach()
        conc_err = torch.relu(self.target_R - R) ** 2 * present * alive

        denom = present.sum().clamp(min=1.0) * spk.shape[2]
        return self.lam * (rate_err.sum() + conc_err.sum()) / denom

    def describe(self):
        return (f"cpg_match (lam={self.lam:g}, target "
                f"{self.target_rate:.2f} spk/cyc, R={self.target_R:.3f})")


class MinCountSpikeObjective(SpikeObjective):
    """Spend as few spikes as possible: a flat cost per spike beyond a free allowance,
    with the TASK loss deciding where the rest are worth spending. Needs the output to
    depend on the spike train alone (natural or explicit gating; see TimingGroupedSNN).
    Then the cheapest solution places spikes densely where the target changes fast and
    sparsely where it creeps: an adaptive sampling clock derived from the data.

    Linear in the rate, not squared: a flat marginal cost per spike is the L1-sparsity
    form, while a squared cost gives up as the rate falls.

    FLOOR (`min_count_floor`, spikes per CYCLE), a two-sided hinge:
        penalty = lam * (relu(duty - floor/period)
                         + floor_weight * relu(floor/period - duty))
    Above the floor it is L1 (marginal cost lam per spike); below it the gradient
    reverses, so descent raises the rate. It has to be a hinge: lam*(duty - floor/period)
    has the same gradient as lam*duty (the derivative is constant), so only a kink can
    change the gradient's sign.

    Why a floor: without one the penalty keeps pushing to zero, and a silent unit is
    UNRECOVERABLE through the task loss (it passes exactly zero gradient to the w1 slots
    it feeds); one run pruned several sub-networks to an excellent 2-5 spikes per cycle
    and others to 0. The floor term still reaches a silent unit because it acts on the
    timing membrane through spike_fn's small non-zero surrogate (~0.03 at
    timing_slope=5). Cost: the floor is an attractor (a unit the task would run at 1
    spk/cyc is pushed up to it), so set it to the fewest spikes a sub-network can use.
    floor_weight=0 gives the one-sided version; min_count_floor=0 the original L1.

    It does NOT fix saturation (surrogate gradient ~1e-6, and Adam's scale-invariance
    means a larger lam cannot compensate); reinit_timing_units handles that. See also
    --lambda_hold / --lambda_warmup and --reinit_dead_after.
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
    """Rescue failed timing units, PER (gait, unit). `dead_pairs` / `sat_pairs` are lists
    of (gait, unit) tuples. Per-gait because w_in_gait already is: gait g's column for
    unit u is an independent slice, so fixing it cannot disturb any other gait, unit or
    anything downstream.

    DEAD (0 spk/cyc): re-roll that (gait, unit) column with the positive mean restored
        and zero its per-gait bias (a silent column carries no direction worth keeping).
    SATURATED: SCALE the column down by `sat_scale` instead. This is a magnitude
        failure, not a direction one: the phase preference is still in the column, just
        swamped, and rate is monotone in gain, so halving strictly reduces it (a badly
        saturated unit may take several triggers).
    Neither can be left to training (zero gradient when silent, ~1e-6 when saturated).

    `w1` is re-rolled only for units dead in EVERY gait: d(cur1)/d(w1) is proportional
    to the timing spike, so those slots stayed at random init while the sub-network
    trained around them, whereas a unit dead in only some gaits trained its slots on the
    others. w1 is indexed by sub-network, so the slots re-rolled are the (g, k) with
    timing_map[g][k] == u; that sub-network's other K-1 slots are left alone.
    w2/w_read/w_out are never touched (they trained and may hold useful output range).
    """
    if not (dead_pairs or sat_pairs) or not hasattr(model, "w_in_gait"):
        return
    dev = model.w_in_gait.weight.device
    MG, nn_, T = model.max_gaits, model.n_neurons, model.n_timing
    W = model.w_in_gait.weight.view(MG, nn_, T)

    for g, u in dead_pairs:
        W[g, :, u] = (torch.randn(nn_, generator=generator).to(dev)
                      * _W_IN_INIT + _W_IN_INIT)
        # Per-gait excitability back to neutral, in whichever form is in use ("voltage" mode has
        # v_t and no b_t; "none" has neither).
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
@torch.no_grad()
def drive_report(model, n_gaits, gait_names=None, indent="    "):
    """The --drive_source cpg counterpart of `timing_report`, with the same (lines, stats)
    contract so run_training can call any of them. (cpg_lif has `drive_spike_report`.)

    Reports DRIVE SELECTION, not spike rates: under cpg the spikes are the CPG's own at a
    fixed rate, so rates would print a constant; what moves is which channels each
    sub-network listens to, in `s_gate`. Needs no CPG replay: it reads the parameters
    directly, which is sound because s_gate and u sit before the sum over channels (no
    threshold or membrane in between). The quantity is drive_strength: |s_gate| *
    ||u[g,c,:]||_1 under cpg (reached under cpg_lif only through visualize_timing's
    --drive_metric gain/product, where it is |s_gate|).

    Per (gait, sub-network):
      dom      the channel supplying the most drive; since channel c bursts at phase
               ~c/n_cpg this is the dominant input PHASE. One that stops moving means
               selection has converged. How well it matches the joint's motion is
               scored by alignment_pair.
      fanin90  channels carrying 90% of the drive: what --drive_lambda is meant to
               reduce (a temporal claim, since channels are phase-localiized).
    """
    # drive_strength, not |s_gate|: under cpg the drive is |s_gate|*||u||_1 (reporting |s_gate|
    # alone made every sub-network look dominated by channel 0: all-ones init, argmax ties
    # go to index 0).
    S = model.drive_strength().detach().float().cpu()        # (n_gaits, G, C)
    C = model.n_neurons

    # Share per sub-network; overall magnitude is discarded.
    share = S / S.sum(dim=2, keepdim=True).clamp(min=1e-12)
    dom      = share.argmax(dim=2)                       # (n_gaits, G)
    domshare = share.max(dim=2).values
    srt      = share.sort(dim=2, descending=True).values
    # +1: the entries strictly under 90% are the channels BEFORE the one that crosses it.
    k90 = (srt.cumsum(dim=2) < 0.90).sum(dim=2) + 1      # (n_gaits, G)

    names = gait_names or [f"g{i}" for i in range(n_gaits)]
    fmt_i = lambda v: " ".join(f"{int(x):>2d}" for x in v)
    lines, stats = [], []
    for gi in range(n_gaits):
        nm = names[gi] if gi < len(names) else f"g{gi}"
        k  = k90[gi]
        lines.append(
            f"{indent}drive {nm:>5s} : dom ch [{fmt_i(dom[gi])}]  "
            f"fanin90 mean {float(k.float().mean()):.1f} "
            f"(min {int(k.min())} max {int(k.max())} of {C})")
        stats.append({
            "gait":              nm,
            "dominant_channel":  [int(v) for v in dom[gi]],
            "dominant_share":    [float(v) for v in domshare[gi]],
            "fanin90":           [int(v) for v in k],
        })
    return lines, stats


@torch.no_grad()
def drive_spike_report(model, spikes, n_gaits, device, period,
                       t0=0, n_steps=None, gait_names=None, indent="    ",
                       align_ctx=None):
    """The --drive_source cpg_lif reporter, same (lines, stats) contract as timing_report
    and drive_report. Reports exact SPIKE COUNTS: with a fixed threshold there is no scale
    ambiguity, and "channel c delivers nothing" is the integer statement "unit (g,c)
    emitted zero spikes", where fanin90 needed an arbitrary 90% cutoff.

    Per (gait, sub-network):
      spk/cyc  spikes per cycle of the OR over the sub-network's C units. Well defined
               because the units are mutually exclusive (one channel each, CPG channels
               never coincide, zero reset), so it is how many input events the
               sub-network receives per cycle; comparable to the timing layer's spk/cyc.
      active   how many of the C channels emitted ANY spike.
      dom ch   the channel contributing the most spikes (dominant input phase).
    """
    L = int(n_steps if n_steps is not None else 6 * period)
    sl = slice(int(t0), int(t0) + L)
    x = torch.as_tensor(spikes[sl], dtype=torch.float32, device=device)
    x = x.unsqueeze(1)                                      # (L, 1, C)
    G, C = model.G, model.n_neurons
    names = gait_names or [f"g{i}" for i in range(n_gaits)]
    n_cyc = max(L / float(period), 1e-9)

    # fanin90 of the GAIN |s_gate|, the quantity the penalty acts on. `active` is exact but
    # coarse (a channel goes silent only below gain ~0.24, so it sits at C for most of
    # training); fanin90 is continuous, the early-warning signal.
    Sg = model.drive_strength().detach().float().cpu()[:n_gaits]  # (ng,G,C)
    sh = Sg / Sg.sum(dim=2, keepdim=True).clamp(min=1e-12)
    srt = sh.sort(dim=2, descending=True).values
    k90 = (srt.cumsum(dim=2) < 0.90).sum(dim=2) + 1                 # (ng,G)

    align_rows = []
    lines, stats = [], []
    for gi in range(n_gaits):
        gt = torch.full((L, 1), gi, dtype=torch.long, device=device)
        spk = model.drive_spikes_only(x, gt)                # (L,1,G,C)
        if spk is None:
            return [f"{indent}(drive_spikes_only unavailable)"], []
        per_ch = spk[:, 0].sum(dim=0).cpu().numpy()         # (G, C) counts
        # Collapsed train: at most one unit per sub-network fires per step, so sum and OR agree
        # (asserted: the collapse rests on it).
        both = spk[:, 0].sum(dim=2)                         # (L, G)
        if float(both.max()) > 1.0 + 1e-6:
            raise AssertionError(
                f"two drive units of one sub-network fired on the same timestep "
                f"(max {float(both.max())}); the collapsed spk/cyc needs zero reset and "
                f"non-overlapping CPG channels.")
        coll = per_ch.sum(axis=1) / n_cyc                   # (G,) spk/cyc
        active = (per_ch > 0).sum(axis=1)                   # (G,)
        dom = per_ch.argmax(axis=1)
        nm = names[gi] if gi < len(names) else f"g{gi}"
        fi = lambda v: " ".join(f"{int(z):>2d}" for z in v)
        ff = lambda v: " ".join(f"{z:>5.1f}" for z in v)
        f90 = k90[gi].numpy()
        # Velocity and corner alignment of each sub-network's collapsed input train, from spikes
        # already in hand.
        av_, ac_ = [float("nan")] * G, [float("nan")] * G
        if align_ctx is not None:
            tr_all = spk[:, 0].cpu().numpy().sum(axis=2)           # (L, G)
            ph_w = np.asarray(align_ctx["phase"][sl], dtype=np.float64)
            for g in range(G):
                av_[g], ac_[g] = alignment_pair(
                    tr_all[:, g], align_ctx["tables"][gi],
                    model.group_cols[g], ph_w, align_ctx["phase_zero"],
                    np.isfinite(ph_w), align_ctx["vel_tau"],
                    align_ctx["cor_tau"])
            align_rows.append((nm, av_, ac_))
        lines.append(f"{indent}drive {nm:>5s} : spk/cyc [{ff(coll)}]")
        lines.append(f"{indent}      {'':>5s}   active  [{fi(active)}] of {C}"
                     f"   dom ch [{fi(dom)}]")
        lines.append(f"{indent}      {'':>5s}   fanin90 [{fi(f90)}] of {C}"
                     f"   (of |s_gate|; mean {f90.mean():.1f})")
        stats.append({
            "gait":     nm,
            "spk_cyc":  [float(v) for v in coll],
            "n_active": [int(v) for v in active],
            # Same key as drive_report, so metrics.csv gets the same fanin_<gait>_g<N> columns.
            "fanin90":  [int(v) for v in f90],
            "dom_ch":   [int(v) for v in dom],
            "align_vel": [float(v) for v in av_],
            "align_cor": [float(v) for v in ac_],
        })
    if align_rows:
        lines += alignment_section(align_rows, indent=indent)
    return lines, stats


def timing_report(model, spikes, phase, period, n_gaits, device,
                  t0, n_steps=1500, gait_names=None, indent="    ",
                  sat_rate=None, align_ctx=None):
    """Per-gait firing statistics for the timing layer: returns formatted lines (and raw
    dicts) for the caller to print every N epochs. Only spk/cyc and the failure warnings
    are printed; phase and R are still computed and returned (the config records them,
    and the cross-gait separation line uses them).

    spk/cyc  spikes per CPG cycle. 0.00 = DEAD: the sub-network gets constant-zero input
             and the w1 slots it feeds get exactly zero gradient, so it does not recover.
             `sat_rate` or above = SATURATED: the unit fires on essentially every CPG
             spike, so its train is the CPG's OR and carries no phase, and it sits where
             the surrogate gradient is ~1e-6. Pass the CEILING (n_cpg * cpg_rate, less a
             little), not _SAT_FRAC of it (a calibration cap; a unit at 0.6 of the
             ceiling still carries phase).
    phase    circular mean of the cycle phase at which the unit fires, in [0,1).
    R        circular concentration in [0,1]; near 0 the spikes are smeared and `phase`
             is meaningless.
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
    align_rows = []
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
        # Velocity and corner alignment per SUB-NETWORK, from the timing units feeding it
        # (timing_map).
        av_, ac_ = [], []
        if align_ctx is not None and hasattr(model, "timing_map_list"):
            ph_w = np.asarray(ph, dtype=np.float64)
            for gs, units in enumerate(model.timing_map_list):
                a1, a2 = alignment_pair(
                    spk[:, list(units)].sum(axis=1), align_ctx["tables"][g],
                    model.group_cols[gs], ph_w, align_ctx["phase_zero"],
                    np.isfinite(ph_w), align_ctx["vel_tau"],
                    align_ctx["cor_tau"])
                av_.append(a1); ac_.append(a2)
            align_rows.append((name, av_, ac_))
        stats.append({"gait": name, "rate": [float(v) for v in rate],
                      "phase": [None if not np.isfinite(v) else float(v)
                                for v in mu],
                      "R": [float(v) for v in R],
                      "align_vel": [float(v) for v in av_],
                      "align_cor": [float(v) for v in ac_]})

    # THE headline diagnostic for gait separation: near-identical phase vectors across gaits mean
    # the timing layer collapsed to a gait-independent clock and the sub-networks carry all
    # the gait knowledge. Identical rate and R across gaits is EXPECTED (the spike penalty
    # pulls every pair to the CPG's values), so phase is the only statistic left to differ.
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
    if align_rows:
        lines += alignment_section(align_rows, indent=indent)
    return lines, stats


# ═══════════════════════════════════════════════════════════════════
# 8.  Training
# ═══════════════════════════════════════════════════════════════════

def make_gait_weights(tables_orig, names, device):
    """Upweight gaits whose original table was coarsest (bk: 22 rows). `names` is required
    (not defaulted) so a silent fallback cannot print the wrong labels next to the right
    numbers.
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
    """Zero the state of any stream that was rewound. `reset_mask` is (B,); it is reshaped
    to match each state tensor's own rank, not a hardcoded one: a wrong rank broadcasts
    instead of failing (a (B,1,1) mask on (B,H) state silently gives (B,B,H)), and
    TimingGroupedSNN's state mixes ranks (mem_timing is (B, n_timing); mem1/mem2/memo are
    (B, G, Hg)).
    """
    if reset_mask.sum() == 0:
        return state
    keep = 1.0 - reset_mask
    return tuple(s * keep.view(-1, *([1] * (s.dim() - 1))) for s in state)


def grad_blocks(model):
    """Group parameters into coarse blocks for per-block gradient reporting, by parameter
    NAME so it works for both architectures. Order matters: "w_in_gait" also startswith
    "w_in", so the timing block is tested first. Per block because a single |grad| hides
    which parts of the model get signal: a near-zero timing block means the timing layer
    is not learning, which a uniformly small gradient (harmless under Adam) would not.
    """
    rules = (
        # s_gate counts under "timing": under cpg / cpg_lif it took over the timing layer's
        # per-gait routing job, keeping block diagnostics comparable across modes.
        ("timing",    ("w_in_gait", "b_t", "beta_t_logit", "s_gate")),
        ("input",     ("w_in",)),                    # dense arch only
        ("sub_l1",    ("w1", "b1", "beta1_logit", "alpha1_logit", "u")),
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
    """Append one row per epoch to out_dir/metrics.csv, flushed every epoch (Ctrl+C is a
    normal way to finish a run, and the config's `history` only lands on a clean exit).
    Columns are fixed from the first row; a key appearing later is dropped rather than
    shifting the table.
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


def _tau_from_logit(logit):
    """Membrane/synapse time constant from a stored decay logit.
    Decays are stored as logit(beta) with beta = exp(-1/tau)."""
    d = torch.sigmoid(logit.detach().float()).clamp(1e-6, 1.0 - 1e-9)
    return (-1.0 / torch.log(d)).cpu().numpy()


# Alignment credit horizons, in steps. FIXED (not derived from the learned taus) so the
# yardstick is the same across runs, readout-tau experiments included. velocity: ~a drive
# spike's propagation delay through the two hidden layers. corner: ~one step, i.e. credit
# for being AT a corner, not for anticipating it (the cubic upsampling already spreads
# |theta''| over a few steps).
_ALIGN_VEL_TAU = 3.0
_ALIGN_COR_TAU = 1.0


def joint_demand(table, cols, phase, phase_zero=0.0, order=1):
    """Per-step |d^order theta / dphase^order| for a sub-network's joint column(s), from the
    SMOOTH gait table at each step's exact phase. Not from the per-step target:
    build_targets looks up the lower table row per step, a staircase whose derivatives
    measured 8.4% (1st) and 143% (2nd) error, against 0.1% / 2.1% from the table (512
    rows, 180 steps/cycle). Units are per unit phase (dphase/dstep is constant under the
    fake CPG and the score is chance-anchored, so the conversion would cancel).

    table (R, J): one gait's upsampled table. cols: the sub-network's output columns
    (several are each scaled to their own mean, then RMS-combined, so no joint dominates
    by amplitude). phase (L,): per-step phase, NaN where undefined (demand 0).
    order: 1 = speed, 2 = curvature (the corner signal).
    """
    T = np.asarray(table, dtype=np.float64)[:, list(cols)]
    R = T.shape[0]
    if order == 1:
        d = (np.roll(T, -1, axis=0) - np.roll(T, 1, axis=0)) * (R / 2.0)
    elif order == 2:
        d = (np.roll(T, -1, axis=0) - 2.0 * T
             + np.roll(T, 1, axis=0)) * float(R * R)
    else:
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    d = np.abs(d)
    if d.shape[1] > 1:
        d = d / np.maximum(d.mean(axis=0, keepdims=True), 1e-12)
        d = np.sqrt((d ** 2).mean(axis=1))
    else:
        d = d[:, 0]
    ph = np.asarray(phase, dtype=np.float64)
    ok = np.isfinite(ph)
    x = ((np.where(ok, ph, 0.0) + phase_zero) % 1.0) * R
    i0 = np.floor(x).astype(np.int64) % R
    f = x - np.floor(x)
    out = (1.0 - f) * d[i0] + f * d[(i0 + 1) % R]
    out[~ok] = 0.0
    return out


def alignment_score(train, demand, tau, valid=None):
    """How well a sub-network's input spikes coincide with a per-step demand signal,
    chance-anchored so it is comparable across joints. Used with demand = |speed|
    (velocity score) and |curvature| (corner score), both from joint_demand.

    Per-step credit is a DECAYED MAXIMUM of the demand v:
        w(t) = max over j >= 0 of  r^j * v(t + j),     r = exp(-1/tau)
    so a spike gets the demand at its own step in full, partial credit for higher demand
    shortly AHEAD, and none for demand behind it (which it cannot have caused). (A decayed
    sum placed the best spike before fast motion under --synaptic.)
        S = (spike-weighted mean of w - mean of w over all steps) / (max of w - mean of w)
    S = 0 is random placement, 1 = every spike at peak demand, < 0 = spikes where demand
    is low (unbounded below and waveform-dependent, so compare only positive scores
    across joints). NaN if there are no spikes. Anchored to the MEAN: under max-only
    normaliization random placement scores ~0.64 on a sinusoid but ~0.25 on a sharp one.

    train (L,) non-negative spike weights; demand (L,) from joint_demand; tau credit
    horizon in steps; valid (L,) bool. The window must be a whole number of cycles of ONE
    gait so the circular wrap lands on the same phase.
    """
    v = np.asarray(demand, dtype=np.float64)
    L = v.shape[0]
    r = math.exp(-1.0 / max(float(tau), 1e-6))
    # w(t) = max(v(t), r * w(t+1)), wrapped circularly; repeated passes
    # converge because r < 1.
    w = v.copy()
    for _ in range(200):
        prev = w.copy()
        nxt = w[0]
        for t in range(L - 1, -1, -1):
            cand = r * nxt
            if cand > w[t]:
                w[t] = cand
            nxt = w[t]
        if np.array_equal(w, prev):
            break
    ok = (np.ones(L, dtype=bool) if valid is None
          else np.asarray(valid, dtype=bool)[:L])
    s = np.asarray(train, dtype=np.float64)[:L] * ok
    if s.sum() <= 0 or not ok.any():
        return float("nan")
    base, top = float(w[ok].mean()), float(w[ok].max())
    if top - base <= 1e-12:
        return float("nan")
    return float(((s * w).sum() / s.sum() - base) / (top - base))


def alignment_pair(train, table, cols, phase, phase_zero, valid=None,
                   vel_tau=_ALIGN_VEL_TAU, cor_tau=_ALIGN_COR_TAU):
    """(velocity score, corner score) for one sub-network over one window."""
    return (alignment_score(train, joint_demand(table, cols, phase,
                                                phase_zero, 1),
                            vel_tau, valid),
            alignment_score(train, joint_demand(table, cols, phase,
                                                phase_zero, 2),
                            cor_tau, valid))


def alignment_section(per_gait, indent="      "):
    """The printed ALIGNMENT block: one line per gait, grouped by statistic with the two
    scores side by side (max, min, mean, each as (vel, cor)).
    per_gait: list of (gait_name, vel_scores, cor_scores).
    """
    def st(a):
        a = np.asarray(a, dtype=np.float64)
        f = a[np.isfinite(a)]
        if f.size == 0:
            return (float("nan"),) * 3
        return float(f.max()), float(f.min()), float(f.mean())
    fmt = lambda x: "  n/a" if not np.isfinite(x) else f"{x:+.2f}"
    out = [f"{indent}alignment   (velocity, corner)   "
           f"0 = random, 1 = ideal"]
    for nm, vel, cor in per_gait:
        mv, nv, av = st(vel)
        mc, nc, ac = st(cor)
        dead = int((~np.isfinite(np.asarray(vel, dtype=np.float64))).sum())
        out.append(f"{indent}  {nm:>10s} : max ({fmt(mv)}, {fmt(mc)})   "
                   f"min ({fmt(nv)}, {fmt(nc)})   "
                   f"mean ({fmt(av)}, {fmt(ac)})"
                   + (f"   [{dead} with no input]" if dead else ""))
    return out


def lambda_schedule(epoch, hold, warmup):
    """Sparsity-penalty multiplier for a 1-based epoch: 0.0 through `hold`, then
    (epoch - hold) / warmup, then 1.0. hold=0 is the old ramp-from-epoch-1; warmup=0 is a
    step at hold+1.
    """
    if epoch <= hold:
        return 0.0
    if warmup <= 0:
        return 1.0
    return min(1.0, (epoch - hold) / float(warmup))


def lambda_schedule_text(args):
    """One-line human description of the schedule, for startup prints."""
    h, w = int(args.lambda_hold), int(args.lambda_warmup)
    parts = []
    if h > 0:
        parts.append(f"0 for epochs 1-{h}")
    if w > 0:
        parts.append(f"linear ramp over epochs {h + 1}-{h + w}")
    parts.append(f"full strength from epoch {h + w + 1}")
    return ", ".join(parts)


def run_training(model, tr_sampler, va_sampler, opt, sched, device, args,
                 gait_w, out_dir, timing_diag=None, n_gaits=4, period=254.0,
                 spike_obj=None, sat_rate=None, tgt_range=None,
                 gait_names=None):
    """`timing_diag`: optional zero-arg callable returning (lines, stats), called every
        args.timing_log_every epochs for timing_grouped (None for dense); its last return
        value is handed back so the config can record it.
    `n_gaits`, `period`: reporting and diagnostics.
    `spike_obj`: a SpikeObjective whose `penalty` is added to the task loss
        (NoSpikeObjective disables it).
    `sat_rate`: spk/cyc at or above which a timing unit counts as saturated, for the
        reinit trigger; the same value passed to timing_report, so the warning and the
        action cannot disagree. None disables saturation rescue.
    """
    best = float("inf")
    best_path = out_path(out_dir, "best_model.pt")
    hist = {"train": [], "val": [], "val_sw": [], "gnorm": [], "sec": [],
            "floor": [], "upd": []}
    last_timing_stats = []

    # Per-block gradient norms and the per-epoch update norm. |upd| measures movement directly:
    # under Adam the update is scale-invariant in the gradient, so a small |grad| says nothing
    # about whether the model is moving. Expected ~lr*sqrt(N) per gradient step.
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
    print(f"  u/g per block is the diagnostic: Adam normalizes PER PARAMETER, so")
    print(f"  a block moves at ~lr per weight no matter how small its gradient.")
    print(f"  u/g >> 1 means a block is moving far more than its share of the")
    print(f"  signal warrants -- i.e. random-walking on noise, which perturbs")
    print(f"  the representation the downstream layers are trying to use.")

    # Spike statistics are only meaningful for the arch with a timing layer.
    use_stats = (spike_obj is not None and spike_obj.enabled
                 and hasattr(model, "n_timing"))
    if use_stats:
        print(f"\n  Spike objective: {spike_obj.describe()}")
        if args.lambda_hold > 0 or args.lambda_warmup > 0:
            print(f"  lambda: {lambda_schedule_text(args)}, so the network "
                  f"learns to USE spikes before it is charged for them.")
        print(f"  Judge it on free-run RMSE, not on the spike count.")
    # Announced separately: under cpg / cpg_lif the spike objective is forced off, yet the
    # same schedule applies to the drive penalty.
    if args.drive_lambda > 0.0 and (args.lambda_hold > 0
                                    or args.lambda_warmup > 0):
        print(f"\n  Drive penalty ({args.drive_penalty}, lambda "
              f"{args.drive_lambda:g}): {lambda_schedule_text(args)}, so the "
              f"task can show which channels matter before the penalty "
              f"starts pruning.")

    # Consecutive timing reports each (gait, unit) pair has been dead or saturated, for
    # reinit_timing_units. Keyed per pair (the rescue is per-gait); advanced only on diagnostic epochs.
    dead_streak, sat_streak = {}, {}

    print(f"\n  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  "
          f"{'Val(post-sw)':>13}  {'LR':>9}  {'|grad|':>8}  {'|upd|':>8}"
          f"  {'sec':>6}")
    print("  " + "-" * 88)
    print(f"  (Ctrl+C stops training and exports.  |grad| is the PRE-clip "
          f"norm, clip={args.clip:g};")
    print(f"   routinely much larger than clip means most updates are being "
          f"truncated and the LR is too hot.)")

    # epoch0.pt: the model before the first update (after gain calibration), the only exact
    # record of the initial weights and taus.
    if args.save_model_every > 0:
        torch.save(model.state_dict(), out_path(out_dir, "epoch0.pt"))
        print("  [saved] epoch0.pt (initial weights, before any update)")

    try:
        for epoch in range(1, args.epochs + 1):
            # Times train + validate, the cost that recurs every epoch; read before the timing report so
            # a diagnostic epoch does not spike this column.
            t_epoch = time.perf_counter()
            # Per-block snapshot of the params, so the update norm can be attributed per block.
            theta0 = {b: [p.detach().clone() for p in plist]
                      for b, plist in blocks.items()}
            bacc = {b: torch.zeros((), device=device) for b in blocks}

            # ---- train -------------------------------------------------
            model.train()
            state = model.init_state(args.batch, device)
            tr_sampler.mark_all_reset()     # so the warm-up mask covers it
            # ONE schedule for every sparsity term (see lambda_schedule): both the spike objective and
            # the drive penalty can lock in a structure before the task has said what it needs.
            lam_scale = lambda_schedule(epoch, args.lambda_hold,
                                        args.lambda_warmup)
            tot, gtot, nb, ftot = 0.0, 0.0, 0, 0.0
            for _ in range(args.chunks_per_epoch):
                (x, g, y, m, sw, rst,
                 ph, warm) = tr_sampler.next_chunk(args.bptt)
                x, g, y, m = (x.to(device), g.to(device),
                              y.to(device), m.to(device))
                ph, warm = ph.to(device), warm.to(device)
                state = apply_reset(detach_state(state), rst.to(device))

                # `warm` zeroes steps whose head was reset less than one cycle ago: the forward pass still
                # runs (that is how state builds), but a zero-state network cannot know where it is in the
                # cycle, so those outputs are not trained against.
                m_eff = m * warm

                if use_stats:
                    pred, state, aux = model(x, g, state, return_aux=True)
                else:
                    pred, state = model(x, g, state)
                    aux = None
                loss = masked_loss(pred, y, m_eff, g, gait_w)

                if aux is not None:
                    # aux = (timing spikes,), each (L, B, U). Same warm mask as the task loss: a just-reset unit
                    # legitimately fires less.
                    pen = lam_scale * sum(
                        spike_obj.penalty(a, g, ph, m_eff) for a in aux)
                    loss = loss + pen
                    ftot += float(pen.detach())

                # Drive sparsity penalty (cpg / cpg_lif; participation ratio, or L1 with --drive_penalty l1) on
                # the shared hold/ramp schedule. Separate from the spike objective: it penaliizes the drive's
                # GAINS (under cpg the spikes are the CPG's own, an input; under cpg_lif the spike count is a
                # staircase in the gain). Folded into the same `floor` metric, so one column tracks whichever
                # pressure is active.
                if (args.drive_lambda > 0.0
                        and hasattr(model, "drive_penalty")):
                    dpen = (lam_scale * args.drive_lambda
                            * model.drive_penalty(args.drive_penalty))
                    loss = loss + dpen
                    ftot += float(dpen.detach())

                opt.zero_grad()
                loss.backward()
                # Returns the total norm BEFORE clipping, the only way to see whether clip=1.0 is quietly
                # truncating nearly every update (masking a too-hot LR). Taken before clip_grad_norm_, which
                # rescales in place; accumulated on device and synced once per epoch.
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
                # Keep voltage-mode thresholds strictly positive (see clamp_bias_voltage). getattr because
                # dense has no such parameters; `model` is the raw module (compile swaps `step`, not it).
                clamp = getattr(model, "clamp_bias_voltage", None)
                if clamp is not None:
                    clamp()
                # Stepped per GRADIENT STEP so the LR schedule does not depend on chunks_per_epoch
                # (T_max = epochs * chunks_per_epoch).
                sched.step()
                tot += loss.item(); gtot += float(gnorm); nb += 1
            tr_loss  = tot / max(nb, 1)
            tr_gnorm = gtot / max(nb, 1)
            tr_floor = ftot / max(nb, 1)

            # ---- validate ----------------------------------------------
            model.eval()
            vstate = model.init_state(args.batch, device)
            va_sampler.mark_all_reset()     # so the warm-up mask covers it
            # Accumulated as (masked error sum, valid-step count), not an average of chunk means: after the
            # per-epoch reset the first chunk is almost entirely masked, so an equal-weight mean would
            # count its few valid steps as a full chunk (and a fully masked chunk would drag val down).
            # masked_loss * mask.sum() recovers the exact masked sum.
            vtot, vsw_tot, vn, vsw_n = 0.0, 0.0, 0.0, 0.0
            # Per (gait, joint) squared error and step count for an RMSE in degrees, taken from this
            # pass's predictions (no extra forward pass).
            rm_sse = rm_cnt = None
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

                    nv = float(m.sum())
                    if nv > 0:
                        vtot += masked_loss(pred, y, m, g).item() * nv
                        vn += nv
                        if rm_sse is None:
                            rm_sse = torch.zeros(n_gaits, pred.shape[2],
                                                 device=pred.device)
                            rm_cnt = torch.zeros(n_gaits, device=pred.device)
                        e2 = ((pred - y) ** 2) * m.unsqueeze(-1)   # (L,B,J)
                        gi_ = g.reshape(-1).long()
                        rm_sse.index_add_(0, gi_, e2.reshape(-1, e2.shape[2]))
                        rm_cnt.index_add_(0, gi_, m.reshape(-1))

                    # "post-switch" window: args.settle steps after each switch
                    post = torch.zeros_like(sw)
                    idx = sw.nonzero(as_tuple=False)
                    for t_i, b_i in idx:
                        post[t_i:min(t_i + args.settle, sw.shape[0]), b_i] = 1.0
                    mp = m * post
                    npv = float(mp.sum())
                    if npv > 0:
                        vsw_tot += masked_loss(pred, y, mp, g).item() * npv
                        vsw_n   += npv

            if vn <= 0:
                raise RuntimeError(
                    f"validation scored ZERO steps: the post-reset warm-up "
                    f"({args.val_chunks} chunks x {args.bptt} steps) masked "
                    f"everything. Raise --val_chunks.")
            va_loss = vtot / vn
            vsw     = vsw_tot / vsw_n if vsw_n > 0 else float("nan")

            # Validation RMSE in degrees per gait (mean over joints of each joint's RMSE); targets use one
            # global (lo, hi), so the conversion is (hi-lo)/2. Measured on the validation STREAM, so gait
            # switches make it run above a steady single-gait RMSE.
            rmse_g = np.full(n_gaits, np.nan)
            if rm_sse is not None and tgt_range is not None:
                scale = (float(tgt_range[1]) - float(tgt_range[0])) / 2.0
                cnt = rm_cnt.clamp(min=1.0).unsqueeze(1)
                per = (torch.sqrt(rm_sse / cnt) * scale).cpu().numpy()
                seen = rm_cnt.cpu().numpy() > 0
                rmse_g = np.where(seen, per.mean(axis=1), np.nan)
            gnames = (list(gait_names) if gait_names is not None
                      else [f"g{i}" for i in range(n_gaits)])
            epoch_s = time.perf_counter() - t_epoch

            # Appended together so an interrupt cannot leave these lists at different lengths.
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
            # grad_blocks partitions every trainable parameter into one block, so the Euclidean total over
            # blocks is the full update norm.
            upd = math.sqrt(sum(v * v for v in ublk.values()))
            hist["upd"].append(upd)
            gblk = {b: float(bacc[b]) / max(nb, 1) for b in blocks}
            for b in blocks:
                hist[f"g_{b}"].append(gblk[b])
                hist[f"u_{b}"].append(ublk[b])

            flag = ""
            # A real boolean for metrics.csv: `flag` is a display string that also carries the snapshot
            # tag, so parsing `best` from it would log 0 on an epoch that is both.
            is_best = False
            if va_loss < best:
                best = va_loss
                torch.save(model.state_dict(), best_path)
                flag = " *"
                is_best = True

            # Periodic snapshot, saved REGARDLESS of val and independent of best_model.pt: val picks the
            # lowest-scoring epoch, which under --lambda_hold can be an unpenalized one from before any
            # pruning. Gives a fixed trajectory to choose from; load with --ckpt epochN.pt.
            if (args.save_model_every > 0
                    and epoch % args.save_model_every == 0):
                torch.save(model.state_dict(),
                           out_path(out_dir, f"epoch{epoch}.pt"))
                flag += " [saved epoch]"

            if epoch % args.log_every == 0 or epoch == 1:
                print(f"  {epoch:>6}  {tr_loss:>10.6f}  {va_loss:>10.6f}  "
                      f"{vsw:>13.6f}  {opt.param_groups[0]['lr']:>9.2e}"
                      f"  {tr_gnorm:>8.3f}  {upd:>8.3f}  {epoch_s:>6.1f}{flag}")

            # Run the timing diagnostic BEFORE metrics.write so this epoch's rates land in THIS epoch's
            # row (printing is deferred below). Epoch 1 always runs it, so the rate columns exist from
            # the first row, which fixes MetricsWriter's columns.
            run_timing_diag = (timing_diag is not None and
                               (epoch % args.timing_log_every == 0
                                or epoch == 1))
            timing_lines = None
            if run_timing_diag:
                timing_lines, last_timing_stats = timing_diag()

            # One column per (gait, unit) for whichever per-unit series the diagnostic produced, so
            # metrics.csv carries the numbers the printed report does: timing_report 'rate' ->
            # spk_<gait>_t<unit>; drive_report 'fanin90' -> fanin_<gait>_g<sub-network> (driven off which
            # key is present, so the reports stay interchangeable). Names use the gait's NAME, so they
            # survive --gaits reordering. Blank on non-eval epochs (None renders as ""), which separates
            # "not measured" from "measured at zero". dominant_channel is not logged per epoch (it lands
            # in timing_layer_stats at the end).
            spike_rate_cols = {}
            if last_timing_stats:
                for key, prefix, unit in (("rate", "spk", "t"),
                                          ("fanin90", "fanin", "g"),
                                          ("spk_cyc", "dspk", "g"),
                                          ("n_active", "nact", "g"),
                                          ("align_vel", "avel", "g"),
                                          ("align_cor", "acor", "g")):
                    if key not in last_timing_stats[0]:
                        continue
                    for st in last_timing_stats:
                        for u, r in enumerate(st[key]):
                            spike_rate_cols[f"{prefix}_{st['gait']}_{unit}{u}"] = (
                                r if run_timing_diag else None)

            metrics.write({
                "epoch": epoch, "train": tr_loss, "val": va_loss,
                "val_post_switch": vsw, "lr": opt.param_groups[0]["lr"],
                "grad_norm": tr_gnorm, "update_norm": upd,
                "spike_penalty": tr_floor, "sec": epoch_s,
                "best": int(is_best),
                "rmse_mean": (float(np.nanmean(rmse_g))
                              if np.isfinite(rmse_g).any() else None),
                **{f"rmse_{gnames[i]}": (float(rmse_g[i])
                                         if np.isfinite(rmse_g[i]) else None)
                   for i in range(n_gaits)},
                **{f"grad_{b}": gblk[b] for b in sorted(blocks)},
                **{f"upd_{b}": ublk[b] for b in sorted(blocks)},
                **spike_rate_cols,
            })

            # Per-block breakdown on the diagnostic cadence. Read u/g: a block with 1% of the gradient and
            # 60% of the movement is diffusing, not learning.
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
                # Validation RMSE is not a timing quantity: it prints in this every-N block so dense runs see
                # it too (rmse_g is computed every epoch for every arch and always logged to metrics.csv).
                if np.isfinite(rmse_g).any():
                    print("      val RMSE (deg, mean over joints): "
                          + "  ".join(f"{gnames[i]} {rmse_g[i]:.2f}"
                                      for i in range(n_gaits)
                                      if np.isfinite(rmse_g[i]))
                          + f"   | mean {np.nanmean(rmse_g):.2f}"
                          + "   (validation stream, incl. gait switches)")

            # Timing layer: cheap (timing units only, batch 1) but prints n_gaits lines, so it runs on a
            # slower cadence. The call happened earlier (before metrics.write); this is the print plus
            # the reinit logic.
            if run_timing_diag:
                for ln in timing_lines:
                    print(ln)

                # Failures are tracked per (gait, unit): w_in_gait has a gait axis, so a unit fine for five
                # gaits and dead for the sixth gets only that column repaired. "rate" in the guard because
                # drive_report / drive_spike_report produce no rates, so the rescue does not apply under
                # cpg / cpg_lif (and would KeyError).
                if (args.reinit_dead_after > 0 and last_timing_stats
                        and "rate" in last_timing_stats[0]):
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
        # Return normally rather than propagating, so main() still writes the final config and plots
        # (and ONNX with --export_onnx 1) from the best checkpoint so far.
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
    """Free-run the network on held-out steps, one plot per gait. gait_names, leg_cols and
    n_joints are required (not module constants, which are the quadruped layout): a hexapod
    plotted under the wrong legend would be worse than an error. leg_cols may have any
    number of equal-size groups; the grid sizes itself from it. The window is in CPG
    CYCLES, not timesteps (a fixed 1200 steps was ~3.4 cycles at period 352 but ~10 at
    fake_cpg's 120); n_steps/warm override for exact counts.
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
# 9b. Config / checkpoint loading  (used by visualize_timing.py, inference.py)
# ═══════════════════════════════════════════════════════════════════

def cfg_get(cfg, key, default=None):
    """Look a key up at the config top level, then `model_detail`, then the verbatim `args`
    dump (the config records facts more than once on purpose: deployment contract,
    reproduction record, what was typed). Older configs lack newer keys, hence the default.
    """
    for src in (cfg, cfg.get("model_detail", {}), cfg.get("args", {})):
        if isinstance(src, dict) and src.get(key) is not None:
            return src[key]
    return default


def build_model_from_cfg(cfg, device):
    """Reconstruct the trained architecture; shapes must match the checkpoint exactly. Every
    fallback for a missing key is what the code did BEFORE that option existed, not its
    current default, so an old checkpoint reconstructs as what it ran. Init-only values
    (tau ranges, weight scales) do not affect loaded weights but are passed through so the
    printed summary is honest.
    """
    arch = cfg_get(cfg, "arch", "dense")

    # Only reached for configs missing these keys; derived as train.py derives them, so an old
    # config gets self-consistent values rather than quadruped constants.
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
        # timing_map fallback None -> identity (one timing unit per sub-network), what the model
        # was before --timing_shape/--decoder_shape.
        timing_map = cfg_get(cfg, "timing_map")
        # A shared-router-era config has router_hidden and w_r/w_t, which no longer exist (load_run's
        # strict=False report flags it). Fallbacks are what was TRUE before each option existed, not
        # today's defaults: readout_hidden = hidden (Ho == Hg), tau_readout_max = 40.0, timing_slope = slope.
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
            # Fallbacks = what the code did before the option existed: film beta was applied, there was
            # no per-gait readout bias, and the synapse was first order.
            film_mode        = str(cfg_get(cfg, "film_mode", "gamma_beta")),
            readout_gait_bias = bool(cfg_get(cfg, "readout_gait_bias", 0)),
            synaptic         = str(cfg_get(cfg, "synaptic", "none")),
            # "timing" is what the model did before drive_source existed.
            drive_source     = str(cfg_get(cfg, "drive_source", "timing")),
            tau_drive        = cfg_get(cfg, "tau_drive"),
            tau_syn_min      = float(cfg_get(cfg, "tau_syn_min", 2.0)),
            tau_syn_max      = float(cfg_get(cfg, "tau_syn_max", 20.0)),
            # gate_mode replaced the boolean event_gated: map the old key when only it is present (True =
            # "decay", absent = ungated). bias_mode was "current" before it existed.
            bias_mode        = str(cfg_get(cfg, "bias_mode", "current")),
            gate_mode        = str(cfg_get(
                cfg, "gate_mode",
                "decay" if cfg_get(cfg, "event_gated", False) else "none")),
            # sub_ln changes forward behavior, not shapes: a wrong value loads cleanly and computes
            # something else.
            sub_ln           = str(cfg_get(cfg, "sub_ln", "l2")),
            **common)
    else:
        # Fallbacks are what this class did before each option existed (readout as wide as hidden,
        # tau_readout_max 40, bias currents, subtractive reset, LN and FiLM on both layers).
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
            have = sorted({q.name for d in (Path(model_dir),
                                             Path(model_dir, "model"))
                           if d.is_dir() for q in d.glob("*.pt")},
                          key=lambda n: (n != "best_model.pt",
                                         int(re.sub(r"\D", "", n) or 0), n))
            hint = (f"  Available checkpoints: {', '.join(have)}"
                    if have else "  No .pt checkpoints found in this run.")
            if p == cfg_path:
                # The config is written before training starts, so a run that has snapshots has one; missing
                # means a wrong folder, a run that died in setup, or an older run.
                hint += ("\n  The config is written just before training "
                         "starts, so it should exist for any run that saved "
                         "a checkpoint. Check --model_dir; if it is right, "
                         "this run predates that change (older versions only "
                         "wrote the config when training finished).")
            raise FileNotFoundError(f"Not found: {p}\n{hint}")

    cfg = json.loads(cfg_path.read_text())
    model, arch = build_model_from_cfg(cfg, device)

    sd = torch.load(ckpt_path, map_location=device)
    # torch.compile here swaps `step` on the instance, so keys are unprefixed; strip a module
    # wrapper prefix anyway.
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

def build_run_config(*, args, model, device, out_dir,
                     n_par, calib, spike_obj, gait_tables,
                     gait_tables_orig, gait_names, gait_files, gaits_dir,
                     n_legs, n_joints, group_cols, timing_cols,
                     timing_map, leg_cols, layout_src, tgt_range,
                     target_rows, period, cpg_params, cpg_rate,
                     cpg_R, warm_steps, neuron_offsets, burst_thresholds,
                     onsets, spikes, valid, t_lo,
                     t_split, t_hi):
    """The run's config dict, built from everything known BEFORE training, so every
    epoch<N>.pt snapshot is loadable while the run is going (build_model_from_cfg reads only
    architecture/shape/tau keys, all fixed at init). Written at the start (write_run_config)
    and again, completed, at the end: the outcome fields (per_joint_rmse_deg,
    timing_layer_stats, and under "training" epochs_completed, gradient_steps,
    sample_timesteps, lr_final, best_val_mse, history) are None here and filled by
    record_training_outcome; "status" is "training" until then. Keyword-only so a missing
    input fails at the call, by name.
    """
    cfg = {
        # ── identity ──────────────────────────────────────────────
        "model":            ("cpg_lif_timing_grouped"
                             if args.arch == "timing_grouped"
                             else "cpg_lif_dense_stateful"),
        "arch":             args.arch,
        "config_version":   3,
        # "training" until record_training_outcome runs, then "complete" / "dry_run".
        "status":           "training",
        "created_utc":      datetime.now(timezone.utc).isoformat(
                                timespec="seconds"),

        # ── deployment-critical: inference.py reads these by name at the top level; do not move
        # or rename them. NOTE: for timing_grouped, `hidden` is PER GROUP. ─────────────
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
        # Structure is recorded three ways: the two shape NAMES (deployment contract), timing_map (what
        # was actually wired; build_model_from_cfg reads it, so a hand-edited name cannot change the
        # architecture) and timing_cols/group_cols (what each unit owns).
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
        # Same as gait_names for CSV-loaded gaits; a separate key so remapping display names does not
        # change what visualize_timing loads.
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
            # Every tunable comes from the SAME dict that built the oscillator, so the config describes
            # what ran.
            **cpg_params,
            "i_app": args.i_app,
            "from_fb_weight": CPG_FROM_FB_WEIGHT,
            "N": int(args.n_cpg_neurons),
            "W": cpg_weight_matrix(args.n_cpg_neurons).tolist(),
            "warmup": args.warmup,
        },
        "per_joint_rmse_deg": None,      # set by record_training_outcome
        "timing_layer_stats": None,      # set by record_training_outcome
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

        # ── model detail (not needed to run; useful to reproduce) ── read off the live model,
        # so it cannot drift.
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
            # These six apply to BOTH arches, so are recorded unconditionally (a None would make
            # build_model_from_cfg rebuild a dense checkpoint as something it never was).
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
            "tau_drive":        (float(getattr(model, "tau_drive", 0.0))
                                 if getattr(model, "drive_lif", False)
                                 else None),
            "drive_source":     (str(args.drive_source)
                                 if args.arch == "timing_grouped" else None),
            "drive_lambda":     float(args.drive_lambda),
            "drive_penalty":    str(args.drive_penalty),
            "tau_syn_min":      float(args.tau_syn_min),
            "tau_syn_max":      float(args.tau_syn_max),
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
            # Whether cpg_lif_snn_step.onnx was written (the signatures below describe the interface
            # regardless).
            "onnx_exported":    bool(args.export_onnx),
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
            "epochs_completed":   0,
            "chunks_per_epoch":   int(args.chunks_per_epoch),
            "gradient_steps":     0,
            "sample_timesteps":   0,
            "batch":              int(args.batch),
            "bptt":               int(args.bptt),
            "lr_initial":         float(args.lr),
            "lr_final":           None,
            "lr_schedule":        "CosineAnnealingLR (per gradient step)",
            "lr_T_max_steps":     int(args.epochs * args.chunks_per_epoch),
            "lr_eta_min":         1e-5,
            "optimizer":          "Adam",
            "grad_clip":          float(args.clip),
            "best_val_mse":       None,
            "history":            None,
        },
    }
    return cfg


def record_training_outcome(cfg, *, args, best, hist, final_lr, timing_stats,
                            rmse):
    """Fill the config's end-of-training fields in place and mark it complete."""
    epochs_done = len(hist["train"])
    grad_steps  = epochs_done * args.chunks_per_epoch
    cfg["per_joint_rmse_deg"] = rmse.tolist()
    cfg["timing_layer_stats"] = timing_stats
    tr = cfg["training"]
    tr["epochs_completed"] = int(epochs_done)
    tr["gradient_steps"]   = int(grad_steps)
    tr["sample_timesteps"] = int(grad_steps * args.bptt * args.batch)
    tr["lr_final"]         = final_lr
    tr["best_val_mse"]     = best
    tr["history"]          = hist
    cfg["status"] = "dry_run" if args.dry_run else "complete"
    cfg["completed_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    return cfg


def write_run_config(cfg, out_dir):
    """Write the config to the run's config file; return its path."""
    cfg_path = out_path(out_dir, "cpg_lif_snn_config.json")
    with open(cfg_path, "w") as f:
        json.dump(json_safe(cfg), f, indent=2, default=str)
    print(f"    [saved] config ({cfg.get('status', '?')}) -> {cfg_path}")
    return cfg_path


def export_onnx(model, out_dir, device):
    model.eval()

    # Swap the eager step back in for the export: torch.onnx.export does not trace reliably
    # through a torch.compile'd callable. Restored in the finally block.
    compiled_step = None
    if hasattr(model, "_step_eager") and model.step is not model._step_eager:
        compiled_step = model.step
        model.step = model._step_eager
        print("    [onnx] using eager step() for export")

    try:
        # Wrapper, spike width and state shapes all come from the model.
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

    return path


# ═══════════════════════════════════════════════════════════════════
# 11.  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Bursting-LIF CPG -> leg-grouped stateful SNN")

    # CPG
    ap.add_argument("--tmax",   type=int,   default=50_000,
                    help="Steps of CPG spike train to collect (the CPG is periodic "
                         "after warmup, so more only duplicates the cycle). Phase and "
                         "switch timing are randomiized independently by StreamSampler.")
    ap.add_argument("--warmup", type=int,   default=2_000)
    ap.add_argument("--i_app",  type=float, default=8.0)
    ap.add_argument("--fake_cpg", type=int, default=1,
                    help="1 = use fake_step_chunk's back-to-back, no-gap bursts "
                         "instead of the real oscillator (temporary, to test a "
                         "continuous-activity CPG). Required for --n_cpg_neurons 18.")
    ap.add_argument("--n_cpg_neurons", type=int, default=6,
                    choices=sorted(CPG_W_BY_N),
                    help="CPG size: selects the coupling matrix from CPG_W_BY_N and "
                         "sets the input width. 6 = one neuron per leg; 18 = one per "
                         "joint (needs --fake_cpg 1, its matrix is a placeholder), "
                         "which narrows the window a timing unit is confined to from "
                         "1/6 to 1/18 of a cycle and lengthens the fake-CPG period, so "
                         "every period-derived default moves.")

    # Tunable CPG parameters, generated from CPG_PARAMS (see its comment); the two to reach for
    # are --cpg_vth_fb and --cpg_refrac_main.
    for _name, (_default, _help) in CPG_PARAMS.items():
        ap.add_argument(f"--cpg_{_name}", type=type(_default),
                        default=_default,
                        help=f"[CPG] {_help} Default {_default!r}.")

    # gait tables
    ap.add_argument("--gaits_dir", type=str, default="../gaits",
                    help="Folder of {name}.csv gait tables, resolved as "
                         "this_file_dir/<gaits_dir> (a sibling of this script's folder "
                         "by default).")
    ap.add_argument("--gaits", type=str, nargs="*", default=None,
                    help="CSV stems (no extension) to load, overriding the "
                         "--n_cpg_neurons default. (visualize_timing's --gaits instead "
                         "filters which loaded gaits to plot.)")
    ap.add_argument("--leg_cols", type=str, default=None,
                    help="JSON list of equal-size column groups, one per leg; "
                         "overrides the default for --n_cpg_neurons (quadruped "
                         "LEG_COLS / HEXAPOD_LEG_COLS).")

    # architecture
    ap.add_argument("--arch", type=str, default="timing_grouped",
                    choices=["dense", "timing_grouped"],
                    help="dense: one fully connected network (the ablation baseline). "
                         "timing_grouped: CPG -> timing layer or direct CPG drive -> "
                         "disconnected sub-networks.")
    ap.add_argument("--timing_shape", type=str, default="per_joint",
                    choices=list(STRUCTURE_SHAPES),
                    help="[timing_grouped] Timing layer structure: per_leg = one LIF "
                         "per leg, per_joint = one per output column. Independent of "
                         "--n_cpg_neurons. Ignored under cpg / cpg_lif.")
    ap.add_argument("--decoder_shape", type=str, default=None,
                    choices=list(STRUCTURE_SHAPES),
                    help="[timing_grouped] Sub-network structure: per_leg (one per "
                         "leg, emitting its columns) or per_joint (one per column); "
                         "None = match --timing_shape. Together they fix the fan-in K: "
                         "a sub-network is fed by every timing unit whose columns "
                         "overlap its own (per_joint timing + per_leg decoders gives "
                         "K=3).")
    ap.add_argument("--readout_hidden", type=int, default=32,
                    help="[timing_grouped] Width of the analog readout membrane per "
                         "group. Not the output width: each unit has its own tau, so "
                         "the width is the temporal basis available to synthesiize the "
                         "waveform (a full Hg-wide w_read was 40%% of the model). "
                         "Sweep 3/8/32/128 to find where it binds.")
    ap.add_argument("--tau_readout_max", type=float, default=None,
                    help="[timing_grouped] Tau init ceiling for the readout membrane; "
                         "None = period/(2*pi), the corner frequency at the gait "
                         "fundamental. Much shorter than --tau_max on purpose: memo "
                         "RENDERS the current angle (local) rather than remembering "
                         "the phase (global), and a tau near the period passes only "
                         "~14%% of the fundamental, i.e. measures the cycle mean.")
    ap.add_argument("--timing_reset", type=str, default="zero",
                    choices=["zero", "subtract"],
                    help="[timing_grouped] Timing-layer reset. 'zero' matches the CPG "
                         "(v[spike]=0); 'subtract' leaves mem-thresh, which re-fires "
                         "on silent steps (ISIs 2,1,1,2,..., ~50%% more spikes per "
                         "burst than the CPG). A/B only. Ignored under cpg / cpg_lif.")
    ap.add_argument("--recon_cycles", type=float, default=4.0,
                    help="CPG cycles shown in each recon_<gait>.png (in cycles, so it "
                         "reads the same at any period).")
    ap.add_argument("--hidden_reset", type=str, default="zero",
                        choices=["zero", "subtract"],
                        help="[timing_grouped] Hidden-layer reset. 'zero' (default) "
                             "matches the CPG and caps each unit at one spike per timing "
                             "spike; with --bias_mode voltage and --film_mode gamma_only "
                             "it makes the sub-networks FULLY naturally gated (see "
                             "TimingGroupedSNN). 'subtract' turns injection magnitude into "
                             "a spike count and measured slightly better RMSE, but leaves "
                             "the loss partly insensitive to when timing spikes land, "
                             "which makes alignment seed-dependent.")
    ap.add_argument("--timing_slope", type=float, default=5.0,
                    help="[timing_grouped] Surrogate slope for the timing layer only "
                         "(--slope applies to the sub-networks). Gentler by default "
                         "because a dead timing unit cuts off a whole sub-network. "
                         "Ignored under cpg / cpg_lif.")
    ap.add_argument("--tau_timing_min", type=float, default=2.0)
    ap.add_argument("--tau_timing_max", type=float, default=None,
                    help="[timing_grouped] Tau init ceiling for the timing layer; None "
                         "= period/n_cpg_neurons, one inter-burst gap. A gap, not a "
                         "cycle: the CPG already bursts at every k/n_cpg phase, so a "
                         "unit only needs memory to bridge adjacent phases, and a long "
                         "tau smooths the burst structure away (modulation depth fell "
                         "17x from tau 10 to 352) and stops decaying between cycles. "
                         "Init range only (learnable). Ignored under cpg / cpg_lif.")
    ap.add_argument("--calibrate_gains", type=int, default=1,
                    help="[timing_grouped, --drive_source timing] 1 = before training, "
                         "bisect each timing unit's CPG->timing weight column until "
                         "its rate lands in a band set as a fraction of the CPG period "
                         "(1/8-1/6, or 1/6-1/3 with min_count), capped below the spike "
                         "ceiling. Forward passes only. 0 = skip.")
    ap.add_argument("--calibrate_per_gait", type=int, default=1,
                    help="[timing_grouped, needs --calibrate_gains 1] 1 = one gain per "
                         "(gait, unit) pair. 0 = the old one gain per unit, bisected "
                         "against the MINIMUM rate across gaits, which is a saturation "
                         "generator (one weak gait dragged the others 3-48x over band) "
                         "that training cannot undo; kept for A/B. Ignored under cpg / "
                         "cpg_lif.")
    ap.add_argument("--spike_objective", type=str, default="min_count",
                    choices=sorted(SPIKE_OBJECTIVES),
                    help="[timing_grouped] Shapes the timing layer's spike train. "
                         "'min_count' (default): a flat L1 cost per spike, with the "
                         "task loss deciding where spikes are worth spending (an "
                         "adaptive sampling clock); needs the output to depend on the "
                         "spike train alone (see NATURAL GATING). 'cpg_match': match "
                         "the CPG's rate and burst tightness; conflicts with "
                         "min_count's premise. 'none'. New strategies: subclass "
                         "SpikeObjective. Forced to 'none' under cpg / cpg_lif.")
    ap.add_argument("--spike_stats_lambda", type=float, default=0.0,
                    help="[timing_grouped] Weight on the spike objective. For "
                         "min_count it is the loss cost of a unit spiking every step "
                         "(at ~17%% duty, ~60 spk/cyc, the penalty is 8.5e-4, "
                         "comparable to a converged task loss). Raise if rates stay "
                         "high, lower if units collapse to silence. 0 disables. "
                         "Ignored under cpg / cpg_lif.")
    ap.add_argument("--min_count_floor", type=float, default=2.0,
                    help="[--spike_objective min_count] Spikes per cycle per (gait, "
                         "unit) below which the penalty reverses (a two-sided hinge), "
                         "so descent pushes the rate back UP instead of pruning to "
                         "silence, which the task loss cannot undo. Set it to the "
                         "fewest spikes a sub-network can use: the floor is an "
                         "attractor. 0 = original un-floored L1. Ignored under cpg / "
                         "cpg_lif.")
    ap.add_argument("--min_count_floor_weight", type=float, default=1.0,
                    help="[--spike_objective min_count] Weight of the below-floor half "
                         "of the hinge, relative to lam. 1.0 = symmetric; 0.0 = "
                         "one-sided (pruning stops at the floor but nothing pushes "
                         "back). Raise if units still die. Ignored under cpg / "
                         "cpg_lif.")
    ap.add_argument("--align_vel_tau", type=float, default=_ALIGN_VEL_TAU,
                    help="Credit horizon (steps) of the VELOCITY alignment score: a "
                         "spike gets the joint speed at its own step plus decaying "
                         "credit for faster motion up to ~this far ahead. Fixed, not "
                         "derived from learned taus, so runs are scored on one "
                         "yardstick.")
    ap.add_argument("--align_cor_tau", type=float, default=_ALIGN_COR_TAU,
                    help="Credit horizon (steps) of the CORNER (|theta''|) alignment "
                         "score: ~1 = credit for being at a corner, not anticipating "
                         "one.")
    ap.add_argument("--save_model_every", type=int, default=50,
                    help="Save epoch<N>.pt every N epochs regardless of val, alongside "
                         "best_model.pt (0 = off). Load with --ckpt epoch<N>.pt "
                         "downstream. The final epoch is saved only if a multiple of "
                         "N.")
    ap.add_argument("--lambda_hold", type=int, default=50,
                    help="[timing_grouped] Epochs at the start with every sparsity "
                         "penalty at EXACTLY zero, before --lambda_warmup's ramp: with "
                         "the s_gate init spread the PR penalty has a gradient from "
                         "step 1, which would amplify init noise while the task is "
                         "still finding which channels matter. 0 = ramp from epoch 1.")
    ap.add_argument("--lambda_warmup", type=int, default=20,
                    help="[timing_grouped] Epochs of linear ramp from 0 to full "
                         "strength, after --lambda_hold, for EVERY sparsity penalty "
                         "(--spike_stats_lambda and --drive_lambda). Both can lock in "
                         "structure before the task has said what it needs: PR is "
                         "rich-get-richer (the init leader wins), and an L1 spike cost "
                         "can prune a gated unit into silence it cannot recover from. "
                         "0 = no ramp.")
    ap.add_argument("--reinit_dead_after", type=int, default=2,
                    help="[timing_grouped] Rescue a failed (gait, timing unit) pair "
                         "after this many consecutive timing reports in the same "
                         "state. Per gait, matching w_in_gait: dead pairs get the "
                         "column re-rolled and the bias zeroed; saturated pairs get it "
                         "scaled by --sat_scale (a magnitude failure, so the direction "
                         "is kept); a unit dead in EVERY gait also gets its w1 slots "
                         "re-rolled. Neither failure self-corrects. 0 disables. "
                         "Ignored under cpg / cpg_lif.")
    ap.add_argument("--sat_scale", type=float, default=0.5,
                    help="[timing_grouped] Factor applied to a saturated (gait, unit) "
                         "column when --reinit_dead_after triggers; a badly saturated "
                         "unit may take several triggers. 1.0 leaves saturated units "
                         "alone. Ignored under cpg / cpg_lif.")
    ap.add_argument("--bias_mode", type=str, default="voltage",
                    choices=["current", "voltage", "none"],
                    help="[timing_grouped] Neuron excitability: 'current' = additive "
                         "input current; 'voltage' (default) = per-unit offset to the "
                         "spiking threshold, with no bias current in the spiking path; "
                         "'none'. Equivalent for an ungated layer with subtractive "
                         "reset; otherwise voltage is preferable since a threshold "
                         "never touches the membrane (so a silent sub-network provably "
                         "cannot spike) and stays out of the per-spike injection. See "
                         "BIAS MODES in TimingGroupedSNN.")
    ap.add_argument("--gate_mode", type=str, default="none",
                    choices=["none", "decay"],
                    help="[timing_grouped] 'none' (default): no explicit gate; with "
                         "--bias_mode voltage the sub-networks only get current on a "
                         "timing spike anyway (see NATURAL GATING in "
                         "TimingGroupedSNN). 'decay': input and sub-net spikes "
                         "explicitly gated (membranes still leak), for A/B. The "
                         "forward pass equals 'none' under the natural-gating "
                         "defaults, but the gradient differs: the gate is spk_t, so it "
                         "amplifies the gradient on spiking steps (up to 6x) and "
                         "zeroes it on silent ones, removing the 'fire HERE' signal; "
                         "it also caps each unit at one spike per timing spike.")
    ap.add_argument("--sub_film", type=str, default="both",
                    choices=["none", "l1", "l2", "both"],
                    help="[timing_grouped] Which sub-network layers get per-gait FiLM. "
                         "'both' (default). 'none' forces the sub-networks to infer "
                         "the gait only from their timing spikes (a clean test of "
                         "whether the timing layer encodes gait; the tables stay "
                         "allocated, ~19%% of params, but get no gradient). Run it "
                         "with --spike_stats_lambda 0 first: that penalty pins rate "
                         "and burst concentration to the CPG's, leaving phase the only "
                         "axis able to carry gait.")
    ap.add_argument("--film_mode", type=str, default="gamma_only",
                    choices=["gamma_only", "gamma_beta"],
                    help="Which FiLM halves apply. 'gamma_only' (default): "
                         "multiplicative only; gamma vanishes where the injection is "
                         "zero, so per-gait conditioning survives WITHOUT tonic drive "
                         "(what makes the sub-networks fully naturally gated; see "
                         "NATURAL GATING) and beats --sub_film none, since w1/w2 have "
                         "no gait axis. 'gamma_beta': also the additive beta, which is "
                         "tonic drive: it smooths the output but makes the loss "
                         "insensitive to timing-spike phase. The beta half is "
                         "allocated either way (checkpoint shapes unchanged).")
    ap.add_argument("--readout_gait_bias", type=int, default=1,
                    help="[timing_grouped] 1 = a learned per-gait constant CURRENT "
                         "into the readout membrane. Nothing else in the sub-networks "
                         "has a gait-dependent DC under --film_mode gamma_only, so "
                         "gaits differing in MEAN angle would otherwise buy the offset "
                         "with spike rate (costly, shrinks downward range, leaves "
                         "ripple). A current, not added to y, so it fades in over a "
                         "gait switch; memo never spikes, so it adds no tonic drive to "
                         "a spiking layer. 0 = off (what older configs reconstruct "
                         "as).")
    ap.add_argument("--drive_source", type=str, default="timing",
                    choices=["timing", "cpg", "cpg_lif"],
                    help="[timing_grouped] What drives the sub-networks. 'timing' "
                         "(default): the learned timing layer. 'cpg': skip it and "
                         "drive each sub-network straight off the CPG's own channels, "
                         "a fixed, evenly tiled phase basis (a timing unit cannot move "
                         "its phase freely: swapping channels needs a two-step weight "
                         "move that fires at both phases, which gradient descent will "
                         "not traverse). Per-gait selection is s_gate, sparsified by "
                         "--drive_lambda; needs --gate_mode none and forces "
                         "--spike_objective none. 'cpg_lif': as cpg plus a "
                         "fixed-threshold LIF per (sub-network, channel) between the "
                         "gain and the expansion, so u receives binary SPIKES: halving "
                         "s then drops spikes, which u can only partly make up for "
                         "(see --drive_penalty), so --drive_lambda acts on |s_gate| "
                         "alone, reporters print exact spike counts instead of "
                         "fanin90, and a low gain restores sub-burst placement.")
    ap.add_argument("--drive_lambda", type=float, default=0.0,
                    help="[--drive_source cpg / cpg_lif] Weight on the drive sparsity "
                         "term (--drive_penalty), on the --lambda_hold / "
                         "--lambda_warmup schedule. Because CPG channels are "
                         "phase-localiized, concentrating a sub-network's drive is a "
                         "temporal statement, not just a structural one. 'pr' is "
                         "rich-get-richer: a weight that dominates the task gradient "
                         "lets init pick the winner, so start small. 0 = off. (Was "
                         "--drive_l1.)")
    ap.add_argument("--drive_penalty", type=str, default="pr",
                    choices=["pr", "l1"],
                    help="[--drive_source cpg / cpg_lif] Sparsity term. 'pr' "
                         "(default): participation ratio, the effective number of "
                         "channels per sub-network (what drive_report prints as "
                         "fanin); scale-invariant. 'l1': mean of the drive. Under cpg "
                         "(drive = |s_gate|*||u||_1) it FAILED: the network shrank "
                         "every drive together and compensated via u and the hidden "
                         "thresholds (max channel share 0.08 at fanin ~15.8). Under "
                         "cpg_lif (drive = |s_gate|) it is UNTESTED and the escape is "
                         "bounded: s_gate can fall from 1.1 to ~0.25 (one late-burst "
                         "spike, penalty ~23%%) while u, FiLM gamma and the hidden "
                         "thresholds make up the charge, though the dropped spikes "
                         "lose within-burst timing and count; only below ~0.24 does a "
                         "channel go silent.")
    ap.add_argument("--synaptic", type=str, default="none",
                    choices=["none", "hidden", "all"],
                    help="Second-order synaptic filter (snnTorch Synaptic / Loihi "
                         "CUBA): syn = alpha*syn + injection, mem = beta*mem + syn. "
                         "'none' (default): injection goes straight into the membrane. "
                         "'hidden': sub-net layers 1 and 2. 'all': also the analog "
                         "readout. Spreading one injection over many steps shrinks the "
                         "output step per spike and keeps hidden membranes near "
                         "threshold for a window, so rates stay learnable upward, and "
                         "it decays to zero in true silence, so spike placement still "
                         "matters. One extra state tensor per layer.")
    ap.add_argument("--tau_syn_min", type=float, default=2.0,
                    help="[--synaptic] Tau init floor for the synaptic "
                         "current.")
    ap.add_argument("--tau_syn_max", type=float, default=None,
                    help="[--synaptic] Tau init ceiling for the synaptic current; None "
                         "= period/16 (a short window: the filter smooths the "
                         "injection, it is not the memory). A guess worth sweeping; "
                         "taus stay learnable.")
    ap.add_argument("--sub_ln", type=str, default="l2",
                    choices=["none", "l1", "l2", "both"],
                    help="[timing_grouped] Which sub-network layers get LayerNorm (no "
                         "affine; FiLM supplies it); never the timing layer. Default "
                         "l2, not both: layer 1's only input is one binary channel, so "
                         "its pre-activation has two possible values and LN would "
                         "normaliize the amplitude away, leaving FiLM gamma to restore "
                         "it.")

    ap.add_argument("--hidden",     type=int,   default=128,
                    help="Hidden width: the TOTAL for --arch dense (H x H layers), PER "
                         "GROUP for timing_grouped (w2/w_read are (G, H, H)); at G=4, "
                         "hidden=128 is near dense hidden=256's parameter count.")
    ap.add_argument("--max_gaits",  type=int,   default=8,
                    help="Rows allocated in the FiLM embedding tables (only the first "
                         "n_gaits are used), so parameter shapes do not depend on the "
                         "gait count and checkpoints transfer between runs with "
                         "different numbers of gaits. Changing it does NOT; unused "
                         "rows are near-identity, so an added gait starts from a "
                         "sensible routing.")
    ap.add_argument("--tau_min",    type=float, default=2.0)
    ap.add_argument("--tau_max",    type=float, default=None,
                    help="Longest SUB-NETWORK membrane tau, in steps; None = the "
                         "measured CPG period rounded up to a multiple of 64 (256 at "
                         "N=4, 384 at N=6). The sub-networks must hold position across "
                         "a cycle, so this tracks the period; the timing layer's does "
                         "not.")
    ap.add_argument("--slope",      type=float, default=25.0)

    # training
    ap.add_argument("--epochs",           type=int,   default=100)
    ap.add_argument("--chunks_per_epoch", type=int,   default=40)
    ap.add_argument("--val_chunks",       type=int,   default=8)
    ap.add_argument("--bptt",             type=int,   default=None,
                    help="Gradient truncation horizon (not the receptive field: state "
                         "is carried and detached across chunks). None = the period "
                         "rounded up to a multiple of 64, ~one cycle. Sweep "
                         "128/256/512 at fixed batch*bptt; batch*bptt is the compute "
                         "budget, so a longer bptt costs proportionally more unless "
                         "batch drops.")
    ap.add_argument("--batch",            type=int,   default=128,
                    help="Stream heads per gradient step. Activation memory is ~12 * "
                         "batch * bptt * n_timing * hidden * 4 bytes (TBPTT keeps ~12 "
                         "(B,G,Hg) tensors per step): ~1.8 GiB at 128 and ~3.6 GiB at "
                         "256 for hexapod n_timing=6, bptt=384, hidden=128. batch*bptt "
                         "is the compute budget. The timestep loop is kernel-launch "
                         "bound, so a bigger batch is nearly free in wall-clock; "
                         "consider an LR rescale (sqrt rule) if raising it.")
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
    ap.add_argument("--export_onnx", type=int, default=0,
                    help="1 = export a single-step ONNX graph (cpg_lif_snn_step.onnx) "
                         "and check it against onnxruntime if installed. Default 0: no "
                         "ONNX code runs or is imported, so environments without onnx "
                         "/ onnxscript / onnxruntime work.")
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
    # Resolved before the CPG run and the timing/decoder shapes, which depend on n_joints/n_legs.
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
        # None means "same as the timing layer" (one decoder per timing unit, K=1). Resolved onto args
        # itself so the config records the shape that ran, not a null.
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
            print(f"      NOTE every sub-network of a leg shares ONE timing unit, so they "
            f"differ only in their own weights; consider --timing_shape per_joint.")
    else:
        timing_cols = group_cols = timing_map = None
        args.n_timing = None

    # ── 1. CPG ──────────────────────────────────────────────────
    print("\n[1/6] Bursting-LIF CPG ...")
    # Resolved once from CPG_PARAMS' keys and used for BOTH the oscillator and the config dump, so
    # they cannot describe different CPGs (visualize_timing.replay_cpg rebuilds the oscillator
    # from the dump).
    cpg_params = {name: getattr(args, f"cpg_{name}") for name in CPG_PARAMS}
    spikes = run_cpg(N=args.n_cpg_neurons, tmax=args.tmax, warmup=args.warmup,
                     i_app=args.i_app, fake_cpg=bool(args.fake_cpg),
                     cpg_params=cpg_params)

    print("\n[2/6] Burst structure & phase ...")
    onsets, period, neuron_offsets, burst_thresholds = analyse_cpg(spikes, out_dir)
    plot_cpg_raster(spikes, onsets, out_dir)

    phase = cycle_phase(len(spikes), onsets[0])

    # ── Period-derived defaults ─────────────────────────────────
    # Resolved after the period is MEASURED, not as argparse constants: the period varies with N
    # (~352 at N=6, ~254 at N=4), and tau_max (hold position across a cycle) and bptt (about one
    # cycle) are meaningless as fixed numbers. Rounding up to a multiple of 64 gives 256 at N=4
    # and 384 at N=6.
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
        # One inter-burst gap (see --tau_timing_max).
        args.tau_timing_max = float(period) / float(args.n_cpg_neurons)
        print(f"      tau_timing_max : {args.tau_timing_max:6.1f}  (timing "
              f"layer; one inter-burst gap = period/{args.n_cpg_neurons})")
    if args.tau_syn_max is None:
        # Short relative to the cycle: the synaptic filter smooths the injection, it does not hold
        # memory (that is mem1/mem2 at tau_max). Init range only (learnable).
        args.tau_syn_max = float(period) / 16.0
        print(f"      tau_syn_max    : {args.tau_syn_max:6.1f}  (synaptic "
              f"current; period/16 — a guess, worth sweeping)")
    if args.tau_readout_max is None:
        # Corner frequency at the gait fundamental (see --tau_readout_max).
        args.tau_readout_max = float(period) / (2.0 * math.pi)
        print(f"      tau_readout_max: {args.tau_readout_max:6.1f}  (readout "
              f"membrane; period/2pi, the low-pass corner at the gait "
              f"fundamental)")

    # The CPG's own firing statistics become the targets for the timing layer.
    cpg_rate, cpg_R = cpg_spike_stats(spikes, phase, period)
    print(f"      CPG stats   : {cpg_rate:.2f} spikes/cycle per neuron, "
          f"phase concentration R={cpg_R:.3f}")

    # ── 3. Upsample ──────────────────────────────────────────────
    print("\n[3/6] Upsampling gait tables ...")
    gait_tables, target_rows = upsample_gait_tables(gait_tables_orig, gait_names)

    targets, valid, tgt_range = build_targets(phase, gait_tables,
                                              phase_zero=args.phase_zero)
    # What the alignment scores need: the SMOOTH tables (derivatives come from them, not from the
    # staircase targets above), the per-step phase, and the two credit horizons.
    align_ctx = {"tables": gait_tables, "phase": phase,
                 "phase_zero": float(args.phase_zero),
                 "vel_tau": float(args.align_vel_tau),
                 "cor_tau": float(args.align_cor_tau)}
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

    # Data lives on the training device. warm_steps = one CPG cycle: a head whose state was just
    # zeroed gets that long to build state before its outputs count toward the loss.
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
            drive_source=args.drive_source,
            # Fixed, derived: bursts are back-to-back, so period/n_cpg is one burst width (see _TAU_DRIVE_BURSTS).
            tau_drive=(_TAU_DRIVE_BURSTS * float(period)
                       / float(args.n_cpg_neurons)),
            slope=args.slope, timing_slope=args.timing_slope).to(device)
    else:
        # The dense ABLATION: every option is shared with TimingGroupedSNN, so the two differ only in
        # the factorization. Matched capacity: the dense hidden layer is H x H where the grouped model
        # has G blocks of Hg x Hg, so scale --hidden by sqrt(G) to compare at equal parameter count.
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

    # ── Compile the single timestep, NOT forward() ───────────────
    # forward() loops over L (--bptt) timesteps in Python, so compiling it would trace-unroll the
    # whole loop (minutes of compile, no reuse). step() is one timestep, compiled once and reused L
    # times per chunk, which removes the per-step kernel-launch overhead that dominates here.
    # `model.step` is an instance attribute shadowing the method, so forward() picks it up; the
    # eager version is stashed because torch.onnx.export cannot trace a compiled callable
    # (export_onnx swaps it back). dynamic=False pins static shapes. Train and val share one
    # graph (args.batch); batch-1 eval passes and .train()/.eval() add a few more, well under the
    # 8-recompile limit past which Dynamo silently falls back to eager.
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
        print(f"      drive_source={args.drive_source}"
              + (f"  drive_lambda={args.drive_lambda:g} "
                 f"({args.drive_penalty})"
                 if args.drive_source in ("cpg", "cpg_lif") else ""))
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
                print(f"      natural gating PARTIAL: voltage bias means the sub-networks are "
                f"driven only on timing spikes, but {' and '.join(leaks)} can still fire a "
                f"hidden unit without one (see check_bias_voltage.py).")
            else:
                print(f"      natural gating FULL: no explicit gate and no path for a hidden "
                f"unit to fire on a timing-silent step (voltage bias, gamma-only FiLM, "
                f"zero reset).")
        if args.spike_objective == "cpg_match":
            print(f"      NOTE --spike_objective cpg_match wants every spike at ONE phase, "
            f"but the sub-networks need spikes wherever the output changes; "
            f"'min_count' lets the task place them.")
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
    if args.arch == "timing_grouped" and args.drive_source == "timing":
        n_route = model.w_in_gait.weight.numel()
        print(f"      routing: {n_route:,} params, a FREE per-gait "
              f"({args.n_cpg_neurons} x {args.n_timing}) matrix per gait row, "
              f"max_gaits={args.max_gaits}")
        print(f"               (the shared-router alternative produced "
              f"near-identical timing phases across gaits and was reverted — "
              f"see the TimingGroupedSNN docstring)")
    elif (args.arch == "timing_grouped"
          and args.drive_source in ("cpg", "cpg_lif")):
        n_sgate = model.s_gate.weight.numel()
        n_u     = model.u.numel()
        print(f"      drive: s_gate {n_sgate:,} params (per-gait "
              f"{args.n_cpg_neurons} x {model.G} channel selection) + "
              f"u {n_u:,} params (gait-shared {args.n_cpg_neurons} x "
              f"{model.G} x {args.hidden} expansion)")
        if args.drive_source == "cpg_lif":
            print(f"             + {model.G * args.n_cpg_neurons:,} drive LIF "
                  f"units ({model.G} sub-nets x {args.n_cpg_neurons} "
                  f"channels, one input each), 0 learnable params: "
                  f"FIXED threshold {model.thresh:g}, FIXED tau "
                  f"{model.tau_drive:.1f}, zero reset")
            print(f"             penalty acts on |s_gate| alone; spike counts "
                  f"are reported, not penalized")
    print(f"      tau range [{args.tau_min:.0f}, {args.tau_max:.0f}] steps "
          f"vs CPG period {period:.0f}")
    if args.tau_max < period:
        print(f"      WARNING: tau_max={args.tau_max:.0f} < CPG period {period:.0f}: the "
        f"leaky membranes are the only long-timescale memory, so the network "
        f"cannot hold phase.")
    gait_w = make_gait_weights(gait_tables_orig, gait_names, device)

    # ── Timing-layer diagnostic hook ────────────────────────────
    # A closure, so run_training needs no spike train / phase array. t_eval is in the val
    # region, so the report is about held-out steps.
    if args.arch == "timing_grouped":
        t_diag = max(t_split, t_lo) + 200
        # Saturated means firing on essentially EVERY CPG spike, i.e. at the ceiling n_cpg * cpg_rate,
        # not at _SAT_FRAC of it (a calibration cap that starts units below the ceiling; a unit at 0.6
        # of it is busy but still carries phase). The 0.95 is tolerance: cpg_rate is measured over the
        # whole train and timing_report over ~6 cycles. Computed once and passed to BOTH timing_report
        # and run_training, so the warning and the action use the same number.
        sat_rate = 0.95 * float(args.n_cpg_neurons) * float(cpg_rate)
        if args.drive_source == "cpg_lif":
            # Spike counts, not drive weights: the fixed-threshold LIF makes "zero spikes" exact.
            timing_diag = lambda: drive_spike_report(
                model, spikes, len(gait_tables), device, period,
                t0=t_diag, n_steps=int(6 * period), gait_names=gait_names,
                indent="      ", align_ctx=align_ctx)
        elif args.drive_source == "cpg":
            # No timing layer to measure and the CPG's rates are constant, so report drive SELECTION
            # (same (lines, stats) contract).
            timing_diag = lambda: drive_report(
                model, len(gait_tables), gait_names=gait_names,
                indent="      ")
        else:
            timing_diag = lambda: timing_report(
                model, spikes, phase, period, len(gait_tables), device,
                t0=t_diag, n_steps=int(6 * period), gait_names=gait_names,
                indent="      ", sat_rate=sat_rate,
                align_ctx=align_ctx)
    else:
        # Both hook values are defined for every arch, so run_training always gets defined arguments
        # (dense used to fail here on an unbound sat_rate).
        timing_diag = None
        sat_rate    = None

    # ── Gain calibration ────────────────────────────────────────
    # Before ANY training: bisect a gain on each (gait, unit) column of w_in_gait until that unit
    # fires inside the target band. After the CPG run (needs a real spike train) and before the
    # optimiizer (it mutates parameters in place). Only with --drive_source timing.
    calib = {}
    # Calibration tunes w_in_gait, which exists only with a timing layer; under cpg / cpg_lif the
    # input is the CPG's own spike train, whose rate is fixed by --cpg_vth_fb.
    if (args.arch == "timing_grouped" and args.calibrate_gains
            and args.drive_source == "timing"):
        print("\n      Calibrating timing-layer gains ...")
        # ── the band, as a FRACTION OF THE PERIOD ───────────────────
        # Two denominators answer two questions:
        #   true_ceiling = n_cpg * cpg_rate: the spike-COUNT ceiling (one spike per CPG spike, since the
        #       BLIF CPG never fires two neurons at once). The un-exceedable limit; the saturation cap
        #       below is measured against it.
        #   period: total TIMESTEPS per cycle, the band's denominator. --cpg_refrac_main lengthens the
        #       period without touching true_ceiling, so a fixed fraction of the ceiling would smear
        #       the same spike count over more steps (sparser input) without the band reflecting it;
        #       a fraction of the PERIOD tracks how much of the cycle, in time, carries input.
        # At the default (refrac_main 1, ISI 2) period == 2 * true_ceiling, so these fractions
        # reproduce the old absolute 15 / 20 / 20 / 40 spk/cyc; they diverge from that ratio when
        # refrac_main changes the period, which is intended.
        true_ceiling = float(args.n_cpg_neurons) * float(cpg_rate)

        # min_count needs far more spikes than the CPG emits per neuron: under natural gating the
        # sub-networks are driven only on timing spikes, and too few cannot render a moving waveform
        # (a zero-order-hold analysis puts it around two thirds of the ceiling for ~1.5 degrees of
        # error). Starting low would begin starved, exactly when L1 pressure arrives, so start high
        # and let the penalty prune. Conditioned on the objective being ACTIVE (lambda > 0), not on
        # --gate_mode (natural gating gives the same spike dependence); at lambda 0 nothing prunes.
        if args.spike_objective == "min_count" and args.spike_stats_lambda > 0.0:
            f_lo, f_hi = _BAND_FRAC_MIN_COUNT
            label = "min_count"
        else:
            f_lo, f_hi = _BAND_FRAC_PLAIN
            label = "no active min_count"
        band_lo, band_hi = f_lo * period, f_hi * period
        print(f"      ({label}: calibrating to {band_lo:.0f}-{band_hi:.0f} "
              f"spk/cyc = {100*f_lo:.0f}-{100*f_hi:.0f}% of the "
              f"{period:.0f}-step period "
              f"[true ceiling {true_ceiling:.0f} spk/cyc = n_cpg "
              f"{args.n_cpg_neurons} x cpg_rate {cpg_rate:.1f}])")

        # The cap stays on true_ceiling, NOT period: it stops calibration targeting something
        # physically unreachable, and it must bite exactly when band_hi (which scales with the
        # period) exceeds the ceiling. It has to be enforced HERE, not in calibrate_gains: a band
        # above the ceiling makes calibrate_gains' `too_loud` test unreachable, so the bisection would
        # drive units to saturation and report them in-band.
        cap = _SAT_FRAC * true_ceiling
        if band_hi > cap:
            print(f"      NOTE band top {band_hi:.0f} exceeds "
                  f"{_SAT_FRAC:g} x the {true_ceiling:.0f} spk/cyc true "
                  f"ceiling; capping to {cap:.0f} so calibration cannot "
                  f"target a phase-blind unit.")
            band_hi = cap
            band_lo = min(band_lo, 0.5 * band_hi)
        calib = calibrate_gains(
            model, spikes, len(gait_tables), device, period,
            lo=band_lo, hi=band_hi, per_gait=bool(args.calibrate_per_gait),
            cpg_rate_hint=cpg_rate)

    # Forced off under cpg / cpg_lif; the reason is printed below.
    if args.drive_source in ("cpg", "cpg_lif") and args.spike_objective != "none":
        why = ("The CPG's spikes are an input, not a learnable train, so a "
               "spike penalty on them has zero gradient."
               if args.drive_source == "cpg" else
               "The drive LIF's spike count is a STAIRCASE in s_gate -- flat "
               "over ~94% of its range -- so a penalty on the count has zero "
               "true gradient almost everywhere. --drive_lambda on |s_gate| "
               "is the smooth, monotonically equivalent version, and the "
               "spike count is REPORTED rather than penalized.")
        print(f"      NOTE --drive_source {args.drive_source}: forcing "
              f"--spike_objective none (was {args.spike_objective!r}). {why}")
        args.spike_objective = "none"
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

    # Write the config NOW so every epoch<N>.pt snapshot is loadable while the run is going;
    # it is rewritten, completed, at the end.
    cfg = build_run_config(
        args=args, model=model, device=device, out_dir=out_dir, n_par=n_par,
        calib=calib, spike_obj=spike_obj, gait_tables=gait_tables,
        gait_tables_orig=gait_tables_orig, gait_names=gait_names,
        gait_files=gait_files, gaits_dir=gaits_dir, n_legs=n_legs,
        n_joints=n_joints, group_cols=group_cols, timing_cols=timing_cols,
        timing_map=timing_map, leg_cols=leg_cols, layout_src=layout_src,
        tgt_range=tgt_range, target_rows=target_rows, period=period,
        cpg_params=cpg_params, cpg_rate=cpg_rate, cpg_R=cpg_R,
        warm_steps=warm_steps, neuron_offsets=neuron_offsets,
        burst_thresholds=burst_thresholds, onsets=onsets, spikes=spikes,
        valid=valid, t_lo=t_lo, t_split=t_split, t_hi=t_hi)
    write_run_config(cfg, out_dir)

    if not args.dry_run:
        opt   = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr)
        # T_max must equal the number of sched.step() calls, which fire once per GRADIENT STEP (inside
        # the chunk loop). With T_max=epochs the cosine would finish after the first chunks_per_epoch
        # steps and the rest of training would run at eta_min.
        total_steps = args.epochs * args.chunks_per_epoch
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=1e-5)
        best, hist, timing_stats = run_training(
            model, tr_sampler, va_sampler, opt, sched,
            device, args, gait_w, out_dir, timing_diag=timing_diag,
            n_gaits=len(gait_tables), period=period,
            spike_obj=spike_obj, sat_rate=sat_rate,
            tgt_range=tgt_range, gait_names=gait_names)
        final_lr = float(opt.param_groups[0]["lr"])
        model.load_state_dict(torch.load(in_path(out_dir, "best_model.pt"),
                                         map_location=device))
        print(f"\n  best val MSE : {best:.6f}")
        plot_training_curves(hist, out_dir)

        # Re-report on the RESTORED best checkpoint: the last in-loop report may be several epochs stale.
        if timing_diag is not None:
            print("\n  Timing layer at best checkpoint:")
            lines, timing_stats = timing_diag()
            for ln in lines:
                print(ln)
    else:
        print("      --dry_run: skipping training.")
        # Still worth seeing: at init it says whether calibration put the layer in a firing regime.
        if timing_diag is not None:
            print("      Timing layer at initialiization:")
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

    # Complete the config with what only a finished run knows and write it BEFORE the ONNX
    # export, so an export failure cannot lose it.
    record_training_outcome(cfg, args=args, best=best, hist=hist,
                            final_lr=final_lr, timing_stats=timing_stats,
                            rmse=rmse)
    write_run_config(cfg, out_dir)
    # Gated, and the only route into ONNX code (export_onnx is the sole user of torch.onnx and
    # onnxruntime). Skipping it leaves the model as an export would (it restores the compiled
    # step in its finally); visualization reloads from disk anyway.
    if args.export_onnx:
        export_onnx(model, out_dir, device)
    else:
        print("    ONNX export skipped (--export_onnx 0)")

    # ── 7. Timing-layer visualization ───────────────────────────
    # Imported HERE: visualize_timing.py imports from this file, so a top-level import is
    # circular. out_dir is both the model dir and the output dir, so figures land in this run's
    # category folders, and re-reading best_model.pt and the config doubles as an end-of-run
    # check that they load.
    if args.visualize and not args.dry_run:
        print("\n[7/7] Timing-layer visualization ...")
        try:
            from visualize_timing import default_args, run_visualization
            # recon=0: phase 6 already wrote recon_*.png and transition.png with this run's
            # --recon_cycles. Use visualize_timing.py --recon 1 to re-plot a checkpoint.
            run_visualization(out_dir, out_dir,
                              default_args(gaits_dir=args.gaits_dir, recon=0))
        except Exception as e:
            # Never let plotting lose a finished run: checkpoint, config and metrics are already on disk.
            print(f"  visualization FAILED ({type(e).__name__}: {e})")
            print(f"  training output is intact; rerun with "
                  f"'python visualize_timing.py --model_dir {args.out_dir or ''}'")

    print(f"\nDone — {out_dir.resolve()}")


if __name__ == "__main__":
    main()