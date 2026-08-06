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
    spans a full gait cycle (~254 steps).  Two mechanisms provide it:
      - heterogeneous, learnable membrane time constants: each hidden
        unit's tau is initialised log-uniformly over [tau_min, tau_max]
        (default 2..500 steps) so the population tiles timescales from
        within-burst to full-cycle.
      - within-group recurrent connections on the spike outputs.
    Training is truncated BPTT over contiguous chunks with state carried
    (and detached) across chunk boundaries -- B parallel streams walking
    the spike train, exactly like deployment.

4.  One CPG neuron per leg.
    The hidden layer is split into 4 equal groups.  Group l is driven
    principally by ONE CPG neuron and reads out ONLY leg l's two joint
    angles.  Cross-group leakage is a single scalar knob (`--cross_gain`)
    so the association is architectural, not merely hoped-for.

    Which neuron drives which leg is *derived*, not guessed: the per-leg
    phase offsets baked into each gait table are measured by circular
    cross-correlation, the CPG's per-neuron burst offsets are measured
    from the spike train, and the two are matched by optimal assignment.
    Result (auto-printed at startup):

        wkF : leg->neuron [0,2,1,3]   max residual 0.026 cycle
        bk  : leg->neuron [0,2,3,1]   max residual 0.116 cycle
        wkR : leg->neuron [0,2,1,3]   max residual 0.133 cycle
        wkL : leg->neuron [0,2,1,3]   max residual 0.133 cycle

    i.e. leg 0 swings when N0 bursts, leg 1 when N2 bursts, and so on.
    The routing is a per-gait 4x4 permutation matrix applied to the input
    spikes; servo routing on the output side stays FIXED (group l always
    drives leg l's servos), which is the only physically legal choice.

    >>> CHECK THIS <<<  see LEG_COLS / SERVO_BASE below.

Leg / joint layout  (LEG_COLS)
------------------------------
The 8 gait-table columns are two joints x four legs, laid out as
    columns 0..3 = joint A of legs 0..3
    columns 4..7 = joint B of legs 0..3
so leg l == columns (l, l+4).  This was verified numerically: for every
gait, col j and col j+4 share the same circular phase offset (5/54 cycle
in wkF, 5/39 in wkL/wkR, 15..17/22 in bk) while cols 0..3 are the same
waveform shifted, and cols 4..7 are a different waveform shifted by the
same per-leg amounts.  Change LEG_COLS if your servo harness disagrees.

On the robot, `set_cmd` writes full_angles[j + SERVO_BASE] = pred[j],
so gait-table column j lands on servo index j + 8.

Usage
-----
    python cpg_lif_snn_train.py --epochs 300 --hidden 256
    python cpg_lif_snn_train.py --dry_run          # data + plots, no training
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
torch.set_float32_matmul_precision('high')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
# 0.  Robot layout  —  EDIT HERE IF YOUR HARNESS DIFFERS
# ═══════════════════════════════════════════════════════════════════

# leg l -> (gait-table column for joint A, column for joint B)
LEG_COLS   = [(0, 4), (1, 5), (2, 6), (3, 7)]
# inference writes full_angles[col + SERVO_BASE]
SERVO_BASE = 8
N_LEGS     = 4
N_JOINTS   = 8


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

CPG_W = np.asarray([
    [    0.0      , -648.52905924, -449.60304695, -413.48426163],
    [-369.91504928,     0.0      , -592.29635234, -568.0712858 ],
    [-412.08729881, -391.54918498,     0.0      , -618.03381552],
    [-498.16458351, -655.01105883, -345.38277449,     0.0      ],
], dtype=np.float64)


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

    def __init__(self, N=4, W=None, i_app=8.0,
                 vth_main=100.0, du_main=0.1, dv_main=0.3, refrac_main=1,
                 vth_fb=100.0, du_fb=1.0, dv_fb=0.0, refrac_fb=1,
                 from_fb_weight=-10000.0, to_fb_weight=10.0):
        self.N     = N
        self.W     = CPG_W.copy() if W is None else np.asarray(W, dtype=np.float64)
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


def run_cpg(N=4, tmax=120_000, warmup=2_000, i_app=8.0):
    """Warm up, then collect the spike train used for training."""
    cpg = LIFCPGStepper(N=N, i_app=i_app)
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

wkF = np.array([
    [9,49,67,38,24,20,27,14],[8,50,68,39,28,21,22,14],
    [10,51,70,41,26,22,18,14],[12,52,69,42,24,24,11,14],
    [14,52,63,44,22,26,1,15],[16,53,53,45,21,27,-2,15],
    [18,53,41,46,20,29,-1,15],[21,54,26,47,18,30,6,16],
    [22,54,23,48,18,32,9,17],[25,54,20,49,16,34,12,18],
    [26,54,17,51,16,37,17,18],[28,54,14,52,15,39,22,19],
    [30,52,11,54,14,45,27,19],[32,54,11,54,14,44,29,20],
    [33,58,13,55,15,36,27,21],[34,61,16,56,15,31,24,23],
    [36,64,18,56,14,24,23,24],[38,66,20,57,14,20,22,26],
    [39,67,22,57,14,16,21,28],[41,64,24,57,14,5,20,30],
    [42,55,26,57,14,-1,19,32],[44,44,28,57,15,-3,18,35],
    [45,30,29,57,15,1,18,38],[46,21,31,57,15,5,17,40],
    [47,19,32,56,16,9,17,43],[48,16,35,57,17,12,16,44],
    [49,12,37,62,18,17,14,35],[49,9,39,66,20,24,14,29],
    [50,8,40,68,21,28,14,23],[51,10,42,70,22,26,14,19],
    [52,12,43,70,24,24,15,17],[52,14,44,67,26,22,15,5],
    [53,16,46,59,27,21,15,-2],[53,18,47,47,29,20,16,-2],
    [54,21,48,34,30,18,16,1],[54,22,49,24,32,18,17,6],
    [54,25,50,21,34,16,18,10],[54,26,51,19,37,16,19,12],
    [54,28,52,15,39,15,20,19],[52,30,54,12,45,14,19,24],
    [54,32,55,12,44,14,20,27],[58,33,55,11,36,15,22,29],
    [61,34,56,14,31,15,24,26],[64,36,56,17,24,14,25,24],
    [66,38,57,18,20,14,27,23],[67,39,57,21,16,14,29,21],
    [64,41,57,23,5,14,31,20],[55,42,57,24,-1,14,33,20],
    [44,44,57,26,-3,15,36,19],[30,45,57,28,1,15,39,18],
    [21,46,56,30,5,15,42,17],[19,47,56,32,9,16,45,17],
    [16,48,59,33,12,17,41,17],[12,49,64,35,17,18,33,16],
], dtype=np.float32)

bk = np.array([
    [36,40,36,62,6,-3,6,1],[34,47,32,63,7,-4,7,4],
    [30,53,28,59,8,-3,9,9],[26,58,25,57,10,-2,10,10],
    [22,57,26,55,12,2,6,8],[18,51,29,52,14,8,2,7],
    [15,51,36,50,15,6,-2,6],[17,48,43,47,9,5,-3,5],
    [21,45,49,44,5,5,-4,5],[29,43,55,42,2,5,-3,5],
    [35,39,60,38,-1,6,-1,6],[42,36,63,35,-3,6,1,6],
    [49,32,62,31,-4,7,6,8],[54,28,58,28,-3,9,10,9],
    [57,26,57,24,0,10,9,11],[56,21,54,26,3,12,8,4],
    [51,17,52,31,8,15,6,1],[50,15,49,38,6,14,6,-2],
    [47,18,47,44,5,8,5,-3],[45,24,44,51,5,4,5,-4],
    [42,30,41,56,5,1,5,-3],[38,37,37,60,6,-2,6,-1],
], dtype=np.float32)

wkL = np.array([
    [47,54,58,51,-2,13,-2,2],[45,56,52,52,0,14,-3,2],
    [45,57,46,52,1,15,-4,2],[45,58,38,53,1,17,-2,2],
    [46,59,31,54,1,19,1,2],[47,59,24,54,1,22,6,3],
    [48,60,21,55,2,24,12,3],[49,58,23,56,1,30,16,3],
    [49,61,25,57,1,30,13,3],[50,67,28,57,1,23,12,4],
    [51,69,31,58,2,15,11,5],[52,68,34,58,2,8,10,5],
    [52,65,36,59,2,3,9,6],[53,60,39,61,2,-1,9,5],
    [54,55,41,62,2,-3,9,3],[54,50,43,62,2,-5,9,2],
    [54,43,47,60,3,-5,7,1],[55,35,48,58,3,-3,8,1],
    [56,28,51,57,3,1,8,-1],[57,18,52,55,3,7,9,-1],
    [57,15,54,53,4,10,10,-2],[58,13,55,50,5,16,11,-2],
    [58,16,57,49,5,16,12,-2],[59,18,58,46,6,14,14,-2],
    [59,21,60,44,6,12,14,-2],[60,25,61,43,7,10,16,0],
    [60,28,62,42,7,9,18,1],[60,31,63,42,10,8,20,3],
    [62,32,63,42,6,7,24,3],[62,35,63,44,6,6,27,1],
    [63,38,63,44,3,6,31,1],[61,40,62,45,2,7,35,1],
    [60,42,65,46,1,7,37,1],[58,45,70,47,0,7,27,1],
    [57,47,71,48,-1,7,23,2],[54,48,71,49,-1,8,14,1],
    [53,51,69,49,-2,8,8,1],[50,52,66,50,-2,9,2,1],
    [49,53,63,51,-2,12,-2,2],
], dtype=np.float32)

wkR = -(np.array([
    [-54,-47,-51,-58,-13,2,-2,2],[-56,-45,-52,-52,-14,0,-2,3],
    [-57,-45,-52,-46,-15,-1,-2,4],[-58,-45,-53,-38,-17,-1,-2,2],
    [-59,-46,-54,-31,-19,-1,-2,-1],[-59,-47,-54,-24,-22,-1,-3,-6],
    [-60,-48,-55,-21,-24,-2,-3,-12],[-58,-49,-56,-23,-30,-1,-3,-16],
    [-61,-49,-57,-25,-30,-1,-3,-13],[-67,-50,-57,-28,-23,-1,-4,-12],
    [-69,-51,-58,-31,-15,-2,-5,-11],[-68,-52,-58,-34,-8,-2,-5,-10],
    [-65,-52,-59,-36,-3,-2,-6,-9],[-60,-53,-61,-39,1,-2,-5,-9],
    [-55,-54,-62,-41,3,-2,-3,-9],[-50,-54,-62,-43,5,-2,-2,-9],
    [-43,-54,-60,-47,5,-3,-1,-7],[-35,-55,-58,-48,3,-3,-1,-8],
    [-28,-56,-57,-51,-1,-3,1,-8],[-18,-57,-55,-52,-7,-3,1,-9],
    [-15,-57,-53,-54,-10,-4,2,-10],[-13,-58,-50,-55,-16,-5,2,-11],
    [-16,-58,-49,-57,-16,-5,2,-12],[-18,-59,-46,-58,-14,-6,2,-14],
    [-21,-59,-44,-60,-12,-6,2,-14],[-25,-60,-43,-61,-10,-7,0,-16],
    [-28,-60,-42,-62,-9,-7,-1,-18],[-31,-60,-42,-63,-8,-10,-3,-20],
    [-32,-62,-42,-63,-7,-6,-3,-24],[-35,-62,-44,-63,-6,-6,-1,-27],
    [-38,-63,-44,-63,-6,-3,-1,-31],[-40,-61,-45,-62,-7,-2,-1,-35],
    [-42,-60,-46,-65,-7,-1,-1,-37],[-45,-58,-47,-70,-7,0,-1,-27],
    [-47,-57,-48,-71,-7,1,-2,-23],[-48,-54,-49,-71,-8,1,-1,-14],
    [-51,-53,-49,-69,-8,2,-1,-8],[-52,-50,-50,-66,-9,2,-1,-2],
    [-53,-49,-51,-63,-12,2,-2,2],
], dtype=np.float32))

GAIT_TABLES_ORIG = [wkF, bk, wkR, wkL]
GAIT_NAMES       = ["wkF", "bk", "wkR", "wkL"]


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
# 4.  Leg <-> CPG-neuron assignment
# ═══════════════════════════════════════════════════════════════════

def table_leg_offsets(table, leg_cols=LEG_COLS):
    """
    Per-leg phase offset baked into a gait table, measured by circular
    cross-correlation of each leg's joint-A column against leg 0's.
    Returns (n_legs,) in [0,1).
    """
    n   = table.shape[0]
    ref = table[:, leg_cols[0][0]]
    ref = (ref - ref.mean()) / (ref.std() + 1e-9)
    offs = []
    for cols in leg_cols:
        a = table[:, cols[0]]
        a = (a - a.mean()) / (a.std() + 1e-9)
        err = [np.mean((np.roll(a, -s) - ref) ** 2) for s in range(n)]
        offs.append(int(np.argmin(err)) / n)
    return np.array(offs, dtype=np.float64)


def solve_leg_routing(tables, names, neuron_offsets, leg_cols=LEG_COLS):
    """
    For each gait, assign every leg the CPG neuron whose burst onset is
    closest (circularly) to that leg's swing onset.  Optimal assignment,
    so it is always a permutation — no two legs share a neuron.

    Returns route (n_gaits, n_legs) int:  route[g, l] = CPG neuron index
    that drives leg l under gait g.
    """
    route = np.zeros((len(tables), len(leg_cols)), dtype=np.int64)
    print("\n      leg-offset / neuron-offset matching:")
    for gi, (t, nm) in enumerate(zip(tables, names)):
        psi = table_leg_offsets(t, leg_cols)
        d   = np.abs(psi[:, None] - neuron_offsets[None, :])
        d   = np.minimum(d, 1.0 - d)                     # circular distance
        rows, cols = linear_sum_assignment(d)
        route[gi, rows] = cols
        print(f"        {nm:>4s}: leg offsets={np.round(psi,3).tolist()}  "
              f"-> leg2neuron={route[gi].tolist()}  "
              f"resid={np.round(d[rows, cols],3).tolist()}")
    return route


def routing_matrices(route, n_neurons=4):
    """
    P[g] such that  x_routed = x @ P[g]  gives x_routed[l] = x[route[g,l]].
    A matmul exports to ONNX far more reliably than gather-with-batched-index.
    """
    G, L = route.shape
    P = np.zeros((G, n_neurons, L), dtype=np.float32)
    for g in range(G):
        for l in range(L):
            P[g, route[g, l], l] = 1.0
    return P


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
    """

    def __init__(self, spikes, targets, valid, t_lo, t_hi, batch,
                 switch_min=600, switch_max=3000, rng=None, n_gaits=4):
        self.spikes  = spikes
        self.targets = targets
        self.valid   = valid
        self.t_lo    = int(t_lo)
        self.t_hi    = int(t_hi)
        self.B       = int(batch)
        self.smin    = int(switch_min)
        self.smax    = int(switch_max)
        self.n_gaits = int(n_gaits)
        self.rng     = rng or np.random.default_rng(0)

        self.pos   = self.rng.integers(t_lo, t_hi, size=self.B)
        self.gait  = self.rng.integers(0, n_gaits, size=self.B)
        self.count = self.rng.integers(self.smin, self.smax, size=self.B)

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
        x   = np.zeros((L, B, self.spikes.shape[1]), dtype=np.float32)
        g   = np.zeros((L, B), dtype=np.int64)
        y   = np.zeros((L, B, N_JOINTS), dtype=np.float32)
        m   = np.zeros((L, B), dtype=np.float32)   # loss mask
        sw  = np.zeros((L, B), dtype=np.float32)   # 1 on the switch step
        reset_mask = np.zeros(B, dtype=np.float32)

        for b in range(B):
            if self.pos[b] + L > self.t_hi:
                self._rewind(b, L)
                reset_mask[b] = 1.0
            s = self.pos[b]
            x[:, b, :] = self.spikes[s:s + L]

            # gait timeline for this head over the chunk
            gs = np.full(L, self.gait[b], dtype=np.int64)
            c  = self.count[b]
            k  = 0
            while c < L:
                new_g = self.rng.integers(0, self.n_gaits)
                while new_g == gs[c] and self.n_gaits > 1:
                    new_g = self.rng.integers(0, self.n_gaits)
                gs[c:] = new_g
                sw[c, b] = 1.0
                k = c
                c += self.rng.integers(self.smin, self.smax)
            self.gait[b]  = int(gs[-1])
            self.count[b] = int(c - L)

            g[:, b] = gs
            y[:, b, :] = self.targets[gs, np.arange(s, s + L), :]
            m[:, b]    = self.valid[s:s + L].astype(np.float32)
            self.pos[b] = s + L

        return (torch.from_numpy(x), torch.from_numpy(g),
                torch.from_numpy(y), torch.from_numpy(m),
                torch.from_numpy(sw), torch.from_numpy(reset_mask))


# ═══════════════════════════════════════════════════════════════════
# 7.  Stateful leg-grouped SNN
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


class LegGroupedSNN(nn.Module):
    """
    Input  : 4 binary CPG spikes per timestep  (nothing else)
    Gait   : integer index, used for (a) input routing and (b) FiLM
    Output : 8 joint angles per timestep

    Hidden width `hidden` is split into 4 groups of `hidden//4`.  Group l:
        - is driven by CPG neuron route[g, l]   (full-strength self drive)
        - sees the other three neurons only through `cross_gain`
        - has its own recurrent weights
        - reads out ONLY leg l's two joint columns

    So "leg l is moving" and "the neuron feeding group l is bursting" are
    the same event by construction.

    State (all (B, 4, Hg)): mem1, spk1, mem2, spk2, mem_out.
    """

    def __init__(self, hidden=256, n_gaits=4, route_P=None,
                 tau_min=2.0, tau_max=500.0, cross_gain=0.25,
                 slope=25.0, thresh=1.0, n_neurons=4,
                 use_recurrence=True):
        super().__init__()
        assert hidden % N_LEGS == 0, "hidden must be divisible by 4"
        self.H       = hidden
        self.Hg      = hidden // N_LEGS
        self.G       = N_LEGS
        self.n_gaits = n_gaits
        self.slope   = slope
        self.thresh  = thresh
        # Plain Python bool, NOT a buffer or tensor: Dynamo guards on its
        # value and resolves the branch in step() at trace time, so
        # use_recurrence=False produces a graph with the recurrent einsums
        # absent rather than multiplied by zero. Changing it after
        # construction triggers a recompile.
        self.use_recurrence = bool(use_recurrence)
        Hg           = self.Hg

        if route_P is None:
            route_P = np.tile(np.eye(n_neurons, dtype=np.float32),
                              (n_gaits, 1, 1))
        self.register_buffer("route_P", torch.tensor(route_P, dtype=torch.float32))

        # ── layer 1: own-neuron drive + weak cross drive ──────────
        self.w_self     = nn.Parameter(torch.randn(self.G, Hg) * 0.8)
        self.w_cross    = nn.Parameter(torch.randn(n_neurons, self.H) * 0.2)
        self.b1         = nn.Parameter(torch.zeros(self.G, Hg))
        self.cross_gain = nn.Parameter(torch.tensor(float(cross_gain)))
        self.rec1       = nn.Parameter(torch.randn(self.G, Hg, Hg) / math.sqrt(Hg))
        self.ln1        = nn.LayerNorm(Hg)

        # ── layer 2: grouped feedforward + recurrence ─────────────
        self.w2   = nn.Parameter(torch.randn(self.G, Hg, Hg) / math.sqrt(Hg))
        self.b2   = nn.Parameter(torch.zeros(self.G, Hg))
        self.rec2 = nn.Parameter(torch.randn(self.G, Hg, Hg) / math.sqrt(Hg))
        self.ln2  = nn.LayerNorm(Hg)

        # ── readout: non-spiking leaky membrane, then 2 angles ────
        self.w_read = nn.Parameter(torch.randn(self.G, Hg, Hg) / math.sqrt(Hg))
        self.b_read = nn.Parameter(torch.zeros(self.G, Hg))
        self.w_out  = nn.Parameter(torch.randn(self.G, Hg, 2) / math.sqrt(Hg))
        self.b_out  = nn.Parameter(torch.zeros(self.G, 2))

        # ── heterogeneous, learnable time constants ───────────────
        self.beta1_logit = nn.Parameter(init_beta_logit((self.G, Hg), tau_min, tau_max))
        self.beta2_logit = nn.Parameter(init_beta_logit((self.G, Hg), tau_min, tau_max))
        self.betao_logit = nn.Parameter(init_beta_logit((self.G, Hg), 2.0, 40.0))

        # ── FiLM: per-gait scale/shift on each layer's input current ──
        self.film1 = nn.Embedding(n_gaits, 2 * self.H)
        self.film2 = nn.Embedding(n_gaits, 2 * self.H)
        for e in (self.film1, self.film2):
            nn.init.zeros_(e.weight)
            e.weight.data[:, :self.H] = 1.0     # gamma := 1, beta := 0

    # ---------------------------------------------------------------
    def init_state(self, batch, device, dtype=torch.float32):
        z = lambda: torch.zeros(batch, self.G, self.Hg, device=device, dtype=dtype)
        return (z(), z(), z(), z(), z())

    def _film(self, emb, gait):
        v = emb(gait)                                     # (B, 2H)
        gamma, beta = v[:, :self.H], v[:, self.H:]
        return (gamma.view(-1, self.G, self.Hg),
                beta.view(-1, self.G, self.Hg))

    def step(self, x, gait, state):
        """
        x     : (B, n_neurons) float — CPG spikes this timestep
        gait  : (B,) int64
        state : 5-tuple of (B, G, Hg)
        """
        mem1, spk1, mem2, spk2, memo = state
        B = x.shape[0]

        # gait-conditioned input routing: leg l <- CPG neuron route[g, l]
        P  = self.route_P[gait]                            # (B, n_neurons, G)
        xr = torch.bmm(x.unsqueeze(1), P).squeeze(1)       # (B, G)

        g1, f1 = self._film(self.film1, gait)
        g2, f2 = self._film(self.film2, gait)

        # Recurrent terms, gated by the ablation flag. Written as a
        # separate term added in the SAME position as before so that
        # use_recurrence=True is bit-identical to the pre-flag version:
        # adding a Python 0.0 is an exact IEEE-754 identity, and Inductor
        # folds it away, so the disabled path costs nothing.
        #
        # NOTE which spk is which. rec1 consumes spk1 from the PREVIOUS
        # timestep (spk1 is rebound below), and rec2 consumes spk2 from the
        # previous timestep. These two einsums are the only consumers of
        # the incoming spk1/spk2, so with recurrence off those two state
        # slots become inert -- but mem1/mem2/memo are still carried, since
        # the leaky membranes are what make this a stateful SNN at all.
        rec1_term = (torch.einsum("bgh,ghk->bgk", spk1, self.rec1)
                     if self.use_recurrence else 0.0)
        rec2_term = (torch.einsum("bgh,ghk->bgk", spk2, self.rec2)
                     if self.use_recurrence else 0.0)

        # ---- layer 1 -------------------------------------------------
        cur1 = (torch.einsum("bg,gh->bgh", xr, self.w_self)
                + self.cross_gain * (x @ self.w_cross).view(B, self.G, self.Hg)
                + rec1_term
                + self.b1)
        cur1 = self.ln1(cur1) * g1 + f1
        beta1 = torch.sigmoid(self.beta1_logit)
        mem1  = beta1 * mem1 + cur1
        spk1  = spike_fn(mem1 - self.thresh, self.slope)
        mem1  = mem1 - self.thresh * spk1

        # ---- layer 2 -------------------------------------------------
        cur2 = (torch.einsum("bgh,ghk->bgk", spk1, self.w2)
                + rec2_term
                + self.b2)
        cur2 = self.ln2(cur2) * g2 + f2
        beta2 = torch.sigmoid(self.beta2_logit)
        mem2  = beta2 * mem2 + cur2
        spk2  = spike_fn(mem2 - self.thresh, self.slope)
        mem2  = mem2 - self.thresh * spk2

        # ---- analog readout -----------------------------------------
        curo  = torch.einsum("bgh,ghk->bgk", spk2, self.w_read) + self.b_read
        betao = torch.sigmoid(self.betao_logit)
        memo  = betao * memo + curo

        out = torch.einsum("bgh,ghj->bgj", memo, self.w_out) + self.b_out  # (B,G,2)
        y   = torch.cat([out[:, :, 0], out[:, :, 1]], dim=1)               # (B,8)

        return y, (mem1, spk1, mem2, spk2, memo)

    def forward(self, x_seq, gait_seq, state=None, return_spikes=False):
        """
        x_seq    : (L, B, n_neurons)
        gait_seq : (L, B)
        """
        L, B = x_seq.shape[0], x_seq.shape[1]
        if state is None:
            state = self.init_state(B, x_seq.device, x_seq.dtype)
        ys, s1s, s2s = [], [], []
        for t in range(L):
            y, state = self.step(x_seq[t], gait_seq[t], state)
            ys.append(y)
            if return_spikes:
                s1s.append(state[1].detach())
                s2s.append(state[3].detach())
        y_seq = torch.stack(ys)                                # (L,B,8)
        if return_spikes:
            return y_seq, state, torch.stack(s1s), torch.stack(s2s)
        return y_seq, state


class SingleStepONNX(nn.Module):
    """Flat-signature wrapper so the exported graph is one timestep with
    explicit state in/out — the robot calls this once per CPG step."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spikes, gait, mem1, spk1, mem2, spk2, memo):
        y, (m1, s1, m2, s2, mo) = self.model.step(
            spikes, gait, (mem1, spk1, mem2, spk2, memo))
        return y, m1, s1, m2, s2, mo


# ═══════════════════════════════════════════════════════════════════
# 8.  Training
# ═══════════════════════════════════════════════════════════════════

def make_gait_weights(tables_orig, device):
    """Upweight gaits whose original table was coarsest (bk: 22 rows)."""
    R = max(t.shape[0] for t in tables_orig)
    w = torch.tensor([R / t.shape[0] for t in tables_orig],
                     dtype=torch.float32, device=device)
    print("      gait loss weights: " +
          "  ".join(f"{GAIT_NAMES[i]}={w[i].item():.2f}" for i in range(len(w))))
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
    """Zero the state of any stream that was rewound."""
    if reset_mask.sum() == 0:
        return state
    keep = (1.0 - reset_mask).view(-1, 1, 1)
    return tuple(s * keep for s in state)


def run_training(model, tr_sampler, va_sampler, opt, sched, device, args,
                 gait_w, out_dir):
    best = float("inf")
    best_path = out_dir / "best_model.pt"
    hist = {"train": [], "val": [], "val_sw": []}

    print(f"\n  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  "
          f"{'Val(post-sw)':>13}  {'LR':>9}")
    print("  " + "-" * 60)
    print("  (Ctrl+C stops training and proceeds to export)")

    try:
        for epoch in range(1, args.epochs + 1):
            # ---- train -------------------------------------------------
            model.train()
            state = model.init_state(args.batch, device)
            tot, nb = 0.0, 0
            for _ in range(args.chunks_per_epoch):
                x, g, y, m, sw, rst = tr_sampler.next_chunk(args.bptt)
                x, g, y, m = (x.to(device), g.to(device),
                              y.to(device), m.to(device))
                state = apply_reset(detach_state(state), rst.to(device))

                pred, state = model(x, g, state)
                loss = masked_loss(pred, y, m, g, gait_w)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                tot += loss.item(); nb += 1
            sched.step()
            tr_loss = tot / max(nb, 1)

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

            # Appended together, so an interrupt can never leave these
            # three lists at different lengths.
            hist["train"].append(tr_loss)
            hist["val"].append(va_loss)
            hist["val_sw"].append(vsw)

            flag = ""
            if va_loss < best:
                best = va_loss
                torch.save(model.state_dict(), best_path)
                flag = " *"

            if epoch % args.log_every == 0 or epoch == 1:
                print(f"  {epoch:>6}  {tr_loss:>10.6f}  {va_loss:>10.6f}  "
                      f"{vsw:>13.6f}  {opt.param_groups[0]['lr']:>9.2e}{flag}")

    except KeyboardInterrupt:
        # Return normally rather than propagating: main() then falls through
        # to plots + config + ONNX export using the best checkpoint so far,
        # so an aborted run still produces deployable artifacts.
        done = len(hist["train"])
        print()
        print("  " + "-" * 60)
        print(f"  [INTERRUPT] Ctrl+C received during epoch {done + 1}.")
        print(f"              {done} epoch(s) completed and recorded; the "
              f"partial epoch is discarded.")
        print(f"              Best val MSE so far : {best:.6f}")
        print( "              Stopping training and proceeding to export.")

    print("  " + "-" * 60)
    return best, hist


# ═══════════════════════════════════════════════════════════════════
# 9.  Plots
# ═══════════════════════════════════════════════════════════════════

def plot_cpg_raster(spikes, onsets, out_dir, n_show=1200):
    N = spikes.shape[1]
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"]
    fig, ax = plt.subplots(figsize=(15, 3.6))
    for i in range(N):
        t = np.where(spikes[:n_show, i] > 0)[0]
        ax.scatter(t, np.full_like(t, i), marker="|", s=130, lw=1.6,
                   color=colors[i % 4], label=f"N{i}")
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


def plot_leg_alignment(spikes, onsets, targets, route, out_dir,
                       t0, n_show=800, tgt_range=(0, 1)):
    """
    THE diagnostic for this rewrite: for each gait, leg l's joint-A target is
    plotted with vertical lines at the burst onsets of the neuron that drives
    leg l.  If the assignment is right, every leg's swing starts on a line.
    """
    lo, hi = tgt_range
    scale, shift = (hi - lo) / 2.0, (hi + lo) / 2.0
    G = targets.shape[0]
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"]
    sl = slice(t0, t0 + n_show)

    fig, axes = plt.subplots(G, 1, figsize=(15, 3.0 * G), sharex=True)
    for g in range(G):
        ax = axes[g]
        for l in range(N_LEGS):
            col = LEG_COLS[l][0]
            ax.plot(np.arange(t0, t0 + n_show),
                    targets[g, sl, col] * scale + shift,
                    color=colors[l], lw=1.6, label=f"leg{l} (N{route[g, l]})")
            on = onsets[route[g, l]]
            for b in on[(on >= t0) & (on < t0 + n_show)]:
                ax.axvline(b, color=colors[l], lw=1.0, alpha=0.35, ls="--")
        ax.set_title(f"{GAIT_NAMES[g]} — leg joint-A target vs driving neuron's bursts",
                     fontsize=10)
        ax.set_ylabel("angle (deg)"); ax.legend(fontsize=7, ncol=4)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("timestep")
    plt.tight_layout()
    p = out_dir / "leg_alignment.png"
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
def plot_reconstruction(model, spikes, targets, valid, route, device,
                        out_dir, tgt_range, t0, n_steps=1200, warm=600):
    """Free-run the network on held-out steps, one plot per gait."""
    lo, hi = tgt_range
    scale, shift = (hi - lo) / 2.0, (hi + lo) / 2.0
    model.eval()
    rmse = np.zeros((len(GAIT_NAMES), N_JOINTS))

    for g in range(len(GAIT_NAMES)):
        x = torch.tensor(spikes[t0 - warm:t0 + n_steps]).unsqueeze(1).to(device)
        gg = torch.full((x.shape[0], 1), g, dtype=torch.long, device=device)
        pred, _ = model(x, gg)
        pred = pred[warm:, 0].cpu().numpy() * scale + shift
        true = targets[g, t0:t0 + n_steps] * scale + shift
        v    = valid[t0:t0 + n_steps]

        fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True)
        for l in range(N_LEGS):
            for k, col in enumerate(LEG_COLS[l]):
                ax = axes[l][k]
                r = float(np.sqrt(np.mean((pred[v, col] - true[v, col]) ** 2)))
                rmse[g, col] = r
                ax.plot(true[:, col], color="#457b9d", lw=1.8, label="GT")
                ax.plot(pred[:, col], color="#e63946", lw=1.4, ls="--",
                        label="pred")
                ax.set_title(f"leg{l}  col{col}   RMSE={r:.2f}°", fontsize=9)
                ax.grid(alpha=0.25); ax.legend(fontsize=7)
        axes[-1][0].set_xlabel("timestep"); axes[-1][1].set_xlabel("timestep")
        plt.suptitle(f"{GAIT_NAMES[g]} — free-run reconstruction "
                     f"({warm}-step warm-up discarded)", fontweight="bold")
        plt.tight_layout()
        p = out_dir / f"recon_{GAIT_NAMES[g]}.png"
        plt.savefig(p, dpi=140); plt.close()
        print(f"    [saved] {p}  mean RMSE = {rmse[g].mean():.2f}°")

    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(rmse, aspect="auto", cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, label="RMSE (deg)")
    ax.set_xticks(range(N_JOINTS))
    ax.set_xticklabels([f"c{j}\nleg{j%4}" for j in range(N_JOINTS)], fontsize=8)
    ax.set_yticks(range(len(GAIT_NAMES))); ax.set_yticklabels(GAIT_NAMES)
    for g in range(len(GAIT_NAMES)):
        for j in range(N_JOINTS):
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
                    t0, g_from=0, g_to=1, warm=600, n_steps=1400,
                    switch_at=600):
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

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    for l in range(N_LEGS):
        col = LEG_COLS[l][0]
        axes[l].plot(true[:, col], color="#457b9d", lw=1.8, label="GT")
        axes[l].plot(pred[:, col], color="#e63946", lw=1.4, ls="--", label="pred")
        axes[l].axvline(switch_at, color="k", lw=1.5, ls="-.")
        axes[l].set_ylabel(f"leg{l} c{col} (deg)"); axes[l].grid(alpha=0.25)
        axes[l].legend(fontsize=7)
    axes[-1].set_xlabel("timestep")
    plt.suptitle(f"Gait switch {GAIT_NAMES[g_from]} -> {GAIT_NAMES[g_to]} "
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
        wrapper = SingleStepONNX(model).to(device).eval()

        dummy = (torch.zeros(1, 4, device=device),
                 torch.zeros(1, dtype=torch.long, device=device),
                 *[torch.zeros(1, model.G, model.Hg, device=device)
                   for _ in range(5)])

        in_names  = ["spikes", "gait",
                     "mem1_in", "spk1_in", "mem2_in", "spk2_in", "memo_in"]
        out_names = ["angles",
                     "mem1_out", "spk1_out", "mem2_out", "spk2_out", "memo_out"]

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

    # network
    ap.add_argument("--hidden",     type=int,   default=256)
    ap.add_argument("--tau_min",    type=float, default=2.0)
    ap.add_argument("--tau_max",    type=float, default=256.0,
                    help="Longest membrane time constant, in steps. Lowered "
                         "from 500: what the network actually has to hold is "
                         "'which neuron burst last, and how long ago', and "
                         "inter-burst onset spacing is only ~63 steps "
                         "(silent gap ~34). One full CPG period (~254) is "
                         "the natural ceiling; 500 was overkill. Sweep "
                         "150-256.")
    ap.add_argument("--cross_gain", type=float, default=0.25,
                    help="Initial strength of non-own-neuron drive into a leg "
                         "group. 0.0 = strictly one neuron per leg.")
    ap.add_argument("--slope",      type=float, default=25.0)
    ap.add_argument("--no_recurrence", action="store_true",
                    help="Ablate the within-group recurrent connections "
                         "(rec1/rec2, ~45%% of all parameters). The leaky "
                         "membranes and heterogeneous taus still provide "
                         "memory, so this tests whether recurrence earns its "
                         "cost. Judge on free-run reconstruction RMSE and "
                         "Val(post-sw), NOT training loss -- a smaller model "
                         "can show higher train loss and generalise the same. "
                         "Keep tau_max >= one CPG period when ablating, since "
                         "taus become the only long-timescale mechanism. "
                         "The rec1/rec2 parameters stay registered so the "
                         "state tuple, ONNX signature and checkpoints are "
                         "unchanged; they simply receive no gradient.")

    # training
    ap.add_argument("--epochs",           type=int,   default=300)
    ap.add_argument("--chunks_per_epoch", type=int,   default=40)
    ap.add_argument("--val_chunks",       type=int,   default=8)
    ap.add_argument("--bptt",             type=int,   default=256,
                    help="Gradient truncation horizon. NOT the network's "
                         "receptive field -- state is carried and detached "
                         "across chunks, so the forward pass sees unbounded "
                         "history. 256 ~= one full CPG period. Sweep "
                         "128/256/512 at fixed batch*bptt.")
    ap.add_argument("--batch",            type=int,   default=128,
                    help="Stream heads per gradient step. Raised from 32: at "
                         "these sizes the timestep loop is kernel-launch "
                         "bound, so a bigger batch is nearly free in "
                         "wall-clock. If raising further, consider an LR "
                         "rescale (sqrt rule for Adam).")
    ap.add_argument("--lr",               type=float, default=2e-3)
    ap.add_argument("--clip",             type=float, default=1.0)
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
    ap.add_argument("--log_every", type=int, default=10)
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
    print(f"Device : {device}\nOutput : {out_dir.resolve()}\n")

    # ── 1. CPG ──────────────────────────────────────────────────
    print("[1/6] Bursting-LIF CPG ...")
    spikes = run_cpg(N=4, tmax=args.tmax, warmup=args.warmup, i_app=args.i_app)

    print("\n[2/6] Burst structure & phase ...")
    onsets, period, neuron_offsets, burst_thresholds = analyse_cpg(spikes, out_dir)
    plot_cpg_raster(spikes, onsets, out_dir)

    phase = cycle_phase(len(spikes), onsets[0])

    # ── 3. Gait tables + leg routing ────────────────────────────
    print("\n[3/6] Gait tables, upsampling, leg routing ...")
    for nm, g in zip(GAIT_NAMES, GAIT_TABLES_ORIG):
        print(f"      {nm:>4s} : {g.shape[0]} rows x {g.shape[1]} joints (original)")
    gait_tables, target_rows = upsample_gait_tables(GAIT_TABLES_ORIG, GAIT_NAMES)

    route  = solve_leg_routing(GAIT_TABLES_ORIG, GAIT_NAMES, neuron_offsets)
    P      = routing_matrices(route, n_neurons=4)
    print(f"\n      leg->servo: " +
          ", ".join(f"leg{l}=servo({LEG_COLS[l][0]+SERVO_BASE},"
                    f"{LEG_COLS[l][1]+SERVO_BASE})" for l in range(N_LEGS)))

    targets, valid, tgt_range = build_targets(phase, gait_tables,
                                              phase_zero=args.phase_zero)
    print(f"      targets {targets.shape}   valid coverage "
          f"{valid.mean()*100:.2f}%   range [{tgt_range[0]:.1f}, "
          f"{tgt_range[1]:.1f}] deg")

    t_first = int(onsets[0][2])
    plot_leg_alignment(spikes, onsets, targets, route, out_dir,
                       t0=t_first, tgt_range=tgt_range)

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

    tr_sampler = StreamSampler(spikes, targets, valid, t_lo, t_split,
                               args.batch, args.switch_min, args.switch_max,
                               rng, n_gaits=len(gait_tables))
    va_sampler = StreamSampler(spikes, targets, valid, t_split, t_hi,
                               args.batch, args.switch_min, args.switch_max,
                               np.random.default_rng(args.seed + 1),
                               n_gaits=len(gait_tables))

    # ── 5. Model + training ─────────────────────────────────────
    print("\n[5/6] Model ...")
    model = LegGroupedSNN(hidden=args.hidden, n_gaits=len(gait_tables),
                          route_P=P, tau_min=args.tau_min,
                          tau_max=args.tau_max, cross_gain=args.cross_gain,
                          slope=args.slope,
                          use_recurrence=not args.no_recurrence).to(device)

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
    n_rec = sum(p.numel() for n, p in model.named_parameters()
                if n.startswith("rec"))
    n_active = n_par - (0 if model.use_recurrence else n_rec)
    print(f"      hidden={args.hidden} ({model.Hg}/leg)  params={n_par:,}")
    print(f"      recurrent params : {n_rec:,} "
          f"({100.0 * n_rec / max(n_par, 1):.0f}% of total) — "
          f"{'ACTIVE' if model.use_recurrence else 'INERT (ablated)'}")
    if not model.use_recurrence:
        print(f"      active params    : {n_active:,}")
        if args.tau_max < period:
            print(f"      WARNING: tau_max={args.tau_max:.0f} < CPG period "
                  f"{period:.0f}. With recurrence ablated, taus are the ONLY "
                  f"long-timescale memory — raise tau_max to >= one period.")
    print(f"      tau range [{args.tau_min:.0f}, {args.tau_max:.0f}] steps "
          f"vs CPG period {period:.0f}")
    gait_w = make_gait_weights(GAIT_TABLES_ORIG, device)

    # Defined for both branches so the config can always report them.
    best     = float("nan")
    hist     = {"train": [], "val": [], "val_sw": []}
    final_lr = float(args.lr)

    if not args.dry_run:
        opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=1e-5)
        best, hist = run_training(model, tr_sampler, va_sampler, opt, sched,
                                  device, args, gait_w, out_dir)
        final_lr = float(opt.param_groups[0]["lr"])
        model.load_state_dict(torch.load(out_dir / "best_model.pt",
                                         map_location=device))
        print(f"\n  best val MSE : {best:.6f}")
        plot_training_curves(hist, out_dir)
    else:
        print("      --dry_run: skipping training.")

    # ── 6. Eval + export ────────────────────────────────────────
    print("\n[6/6] Evaluation & export ...")
    t_eval = max(t_split + 800, t_lo + 800)
    rmse = plot_reconstruction(model, spikes, targets, valid, route, device,
                               out_dir, tgt_range, t0=t_eval)
    plot_transition(model, spikes, targets, device, out_dir, tgt_range,
                    t0=t_eval, g_from=0, g_to=1)

    epochs_done = len(hist["train"])
    grad_steps  = epochs_done * args.chunks_per_epoch

    cfg = {
        # ── identity ──────────────────────────────────────────────
        "model":            "cpg_lif_leggrouped_stateful",
        "config_version":   2,
        "created_utc":      datetime.now(timezone.utc).isoformat(
                                timespec="seconds"),

        # ── deployment-critical: inference.py reads these by name at
        #    the top level.  Do not move or rename them. ───────────
        "hidden":           args.hidden,
        "hidden_per_leg":   model.Hg,
        "n_gaits":          len(gait_tables),
        "n_legs":           N_LEGS,
        "n_joints":         N_JOINTS,
        "n_cpg_neurons":    4,
        "gait_names":       GAIT_NAMES,
        "leg_cols":         [list(c) for c in LEG_COLS],
        "servo_base":       SERVO_BASE,
        "route_leg2neuron": route.tolist(),
        "global_min":       float(tgt_range[0]),
        "global_max":       float(tgt_range[1]),
        "target_rows":      int(target_rows),
        "phase_zero":       float(args.phase_zero),
        "cpg_period_steps": float(period),
        "cpg": {
            "i_app": args.i_app, "vth_main": 100.0, "du_main": 0.1,
            "dv_main": 0.3, "refrac_main": 1, "vth_fb": 100.0,
            "du_fb": 1.0, "dv_fb": 0.0, "refrac_fb": 1,
            "from_fb_weight": -10000.0, "to_fb_weight": 10.0,
            "W": CPG_W.tolist(), "warmup": args.warmup,
        },
        "per_joint_rmse_deg": rmse.tolist(),

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
        "model_detail": {
            "class":            "LegGroupedSNN",
            "n_params":         int(n_par),
            "n_params_recurrent": int(n_rec),
            "n_params_active":  int(n_active),
            "use_recurrence":   bool(model.use_recurrence),
            "tau_min":          float(args.tau_min),
            "tau_max":          float(args.tau_max),
            "cross_gain_init":  float(args.cross_gain),
            "slope":            float(args.slope),
            "thresh":           1.0,
            "surrogate":        "fast-sigmoid straight-through "
                                "(plain ops, bit-exact forward)",
            "state_tensors":    ["mem1", "spk1", "mem2", "spk2", "memo"],
            "state_shape":      [1, N_LEGS, model.Hg],
            "onnx_inputs":      ["spikes", "gait", "mem1_in", "spk1_in",
                                 "mem2_in", "spk2_in", "memo_in"],
            "onnx_outputs":     ["angles", "mem1_out", "spk1_out",
                                 "mem2_out", "spk2_out", "memo_out"],
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
                                     for g in GAIT_TABLES_ORIG],
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
            "lr_schedule":        "CosineAnnealingLR",
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