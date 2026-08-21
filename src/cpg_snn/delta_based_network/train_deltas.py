"""
Bursting-LIF CPG  →  dense LIF hidden stack  →  2 spiking DELTA neurons per joint
================================================================================

What is different from train.py
-------------------------------
train.py's networks emit joint angles DIRECTLY: an analog leaky membrane
(`memo`) is projected to `n_joints` numbers every timestep, and the loss
compares those numbers to the gait table.

Here the network never emits an angle.  It emits INCREMENTS.  There are
`2 * n_scales * n_joints` spiking output neurons -- an "up" and a "down"
neuron per DELTA SCALE per joint -- and the commanded angle is a running
accumulator:

    angle[t] = pose + acc[t]
    acc[t]   = (1 - leak) * acc[t-1]
               + sum_s delta[s,j] * (up[s,j,t] - down[s,j,t])

`pose` is the standing pose (row 0 of one gait table, optionally
learnable) and `delta_j` is a learnable per-joint step size.  Two neurons
per joint rather than two per leg because joints inside one leg swing at
nearly -- but not exactly -- the same time, so they need independently
timed increments.

Consequences worth being explicit about
---------------------------------------
1.  The output is a STAIRCASE.  It cannot match a smooth gait waveform
    exactly; it tracks it to within roughly one `delta`.  That is fine --
    the robot side smooths it -- but it means there is an RMSE FLOOR set
    by `delta`, and reporting RMSE without that floor next to it is
    meaningless.  See `joint_kinematics` and the startup table.

1b. The delta LADDER (slow-twitch / fast-twitch).  A single delta has to
    satisfy two conflicting demands: big enough to reach peak velocity
    (>= vmax, see below) and small enough that its ripple is acceptable.
    When the ratio between those is large the only single-scale fix is a
    longer period -- and `bptt` and `tau_max` both track the period, so it
    gets expensive fast (on the real gait tables, a 1 degree floor wanted
    period 512).  A ladder of `--delta_scales` magnitudes, each
    `--delta_ratio` finer than the last, decouples them:

        velocity ceiling = sum_s delta[s]      (all scales can fire at once)
        resolution       = min_s delta[s]      (finest increment available)
        required period  falls by ratio^(n_scales-1)

    The scales are independently learnable, so the ratio is an
    initialisation rather than a constraint -- which also means they can
    collapse onto each other and waste half the output layer, so
    `delta_report` prints the learned ratio and warns.

2.  `delta[0]` has a hard FEASIBILITY BOUND.  An output neuron can fire at
    most once per timestep, so the fastest the accumulator can move is
    `delta_j` units/step.  If the gait needs the joint to move faster
    than that anywhere in the cycle, no amount of training can track it:

        required  vmax_j    = max over gaits, over the cycle, of
                              |d angle_j / dt|   (units per step)
        feasible  delta_j  >= vmax_j
        peak duty          =  vmax_j / delta_j   (fraction of steps the
                                                  neuron must fire at the
                                                  fastest part of the swing)

    So `delta_j` is initialised at `--delta_init_scale * vmax_j` (default
    2.0, i.e. 50% duty at peak velocity) rather than at an arbitrary
    number.  Bigger delta = more headroom, coarser steps; the two pull
    against each other and the learnable parameter has to find the
    balance.  The floor/duty trade is printed at startup for every joint.

3.  The output neurons must fire a LOT.  The spike budget follows from the
    total angular travel per cycle:

        spikes per cycle per direction  ~=  travel_j / (2 * delta_j)

    At the quadruped period (~254 steps) with delta = 2 * vmax that comes
    out around 40 spikes/cycle/direction, out of a hard maximum of one per
    step.  A randomly initialised output layer is as likely to be silent as
    saturated, so `calibrate_out_bias` bisects each output neuron's bias to
    land it on its OWN measured budget before training starts.

4.  DRIFT is a new failure mode.  With leak=0 the accumulator is a pure
    integrator, so a persistent up/down imbalance walks the joint away
    without bound and nothing local to a timestep notices.  Two things
    push back: the MSE is on the ABSOLUTE angle (not the increment), and
    StreamSampler carries `acc` across chunk boundaries -- detached, but
    not reset -- so a head walking thousands of steps is scored on the
    drift it has accumulated the whole way.  `--angle_leak` is available
    as a blunt bound (leak=1e-3 is a ~1000-step decay toward pose, i.e.
    a mild high-pass on the output waveform) and defaults to OFF.
    `drift.png` and the per-gait up/down balance in the diagnostic report
    are the things to watch.

5.  Gradients now flow THROUGH the accumulator.  d loss / d up_j[t] is
    the sum of every future residual in the chunk times delta_j, so the
    output layer's gradient scale grows with `--bptt`.  Watch the |grad|
    column against `--clip`; if it sits far above the clip, this is why.
    On the other hand it is a genuinely informative signal -- "spike here
    and everything downstream shifts up" -- which is the argument for
    plain MSE working at all here.

6.  If plain MSE will not train, the first knob is `--deriv_lambda`.
    That adds a WINDOWED-VELOCITY term, `(pred[t] - pred[t-w])` against
    `(targ[t] - targ[t-w])`, which is what the spikes control directly
    and is insensitive to accumulated DC offset.  Off by default so the
    first run measures the plain objective honestly.

Architecture (deliberately boring)
----------------------------------
    CPG spikes (n_cpg)
      -> `--n_layers` (3) dense LIF layers of `--hidden` (64) units,
         LayerNorm(affine=False) + per-gait FiLM, heterogeneous learnable
         taus over [tau_min, tau_max]
      -> 2 * n_scales * n_joints spiking LIF output neurons (an up
         and a down neuron per delta scale per joint), fully connected, NO
         LayerNorm (LN across the output dim would normalise away the very
         drive magnitude that sets firing rate, and would couple every
         joint's rate to every other joint's), subtract-reset so rate is
         graded in the drive
      -> accumulator -> angle

No timing layer and no grouping: this file is testing the OUTPUT
REPRESENTATION, and mixing in an architecture change at the same time
would make the comparison against train.py unreadable.  Swapping the
hidden stack for a TimingGroupedSNN-style timing layer is a contained
change (replace `DeltaSNN`'s hidden loop, keep the output block) and is
the obvious follow-up if the delta idea holds up.

Everything upstream of the model -- CPG, burst detection, phase, gait
tables, targets, StreamSampler, and the shared plots -- is IMPORTED from
train.py rather than copied, so a fix there lands here too.  The cost is
that a signature change in train.py breaks this file; that seems the
better failure mode than two divergent copies of the CPG.

Usage
-----
    python train_deltas.py --dry_run           # data + feasibility table
    python train_deltas.py --epochs 100
    python train_deltas.py --deriv_lambda 0.3  # if plain MSE stalls
    python train_deltas.py --hidden 128 --n_layers 3
    python train_deltas.py --angle_leak 1e-3   # if free-run drifts
    python train_deltas.py --pose fixed        # non-learnable standing pose
"""

import argparse
import json
import math
import os
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── shared pipeline, imported rather than duplicated ──────────────
# train.py does all of its work inside main(), so importing it has no
# side effects beyond module constants, the matplotlib Agg backend and
# torch.set_float32_matmul_precision('high') -- all of which we want.
# ── locate the sibling leg_grouped_timing/ folder and import train.py ──
# Resolved relative to THIS file, not cwd, so it works no matter where the
# script is invoked from. insert(0, ...) rather than append so this copy of
# train.py wins over any other train.py that might already be importable.
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_TRAIN_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "leg_grouped_timing"))
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

from train import (  # type: ignore
    # CPG + phase
    CPG_W_BY_N, CPG_FROM_FB_WEIGHT, CPG_PALETTE, LIFCPGStepper,
    cpg_weight_matrix, analyse_cpg, cycle_phase, cpg_spike_stats,
    detect_burst_threshold, burst_onsets,
    # gait tables + targets
    GAIT_FILES_BY_N, load_gait_tables, upsample_gait_tables, build_targets,
    default_leg_layout, outputs_path,
    # sampler + training utilities
    StreamSampler, masked_loss, detach_state, apply_reset,
    make_gait_weights, MetricsWriter,
    # model primitives
    spike_fn, init_beta_logit,
    # plots + misc
    plot_cpg_raster, plot_training_curves, plot_reconstruction,
    plot_transition, json_safe, git_info,
)


# ═══════════════════════════════════════════════════════════════════
# 1.  Kinematic requirements  —  what delta HAS to be able to do
# ═══════════════════════════════════════════════════════════════════

def joint_kinematics(gait_tables, period, tgt_range):
    """
    Per-joint kinematic demands of the gait set, in NORMALISED target units
    (the units the model works in: gait tables mapped to [-1, 1] by the same
    global min/max `build_targets` uses).

    Computed from the UPSAMPLED TABLES rather than from the sampled target
    sequence on purpose.  The target sequence is row-quantised -- row index
    is `floor(phase * R)`, so it holds a value for `period / R` steps and
    then jumps -- and `max |diff|` of that sequence measures the
    quantisation jump, not the gait's actual velocity.  Dividing a
    table-row difference by `steps_per_row` gives the real thing.

    The row difference is taken CIRCULARLY (last row wraps to first): the
    gait is a loop, and the wrap-around step is a real part of the motion.

    Returns a dict of (n_joints,) arrays plus the scalars used to build them:
        vmax   : peak |velocity|, normalised units per step
        travel : total variation over one cycle (sum of |diff|), per cycle
        p2p    : peak-to-peak amplitude
        scale  : degrees per normalised unit, = (hi - lo) / 2
    """
    lo, hi = tgt_range
    scale  = (hi - lo) / 2.0
    R      = gait_tables[0].shape[0]
    steps_per_row = float(period) / float(R)

    # (G, R, J) in normalised units -- same transform as build_targets.
    tabs = np.stack([((t - lo) / (hi - lo + 1e-8) * 2.0 - 1.0)
                     for t in gait_tables]).astype(np.float64)

    d = np.abs(np.diff(np.concatenate([tabs, tabs[:, :1]], axis=1), axis=1))

    return {
        "scale":         float(scale),
        "R":             int(R),
        "steps_per_row": float(steps_per_row),
        "vmax":          d.max(axis=(0, 1)) / steps_per_row,   # (J,)
        "travel":        d.sum(axis=1).max(axis=0),            # (J,)
        "p2p":           (tabs.max(axis=1) - tabs.min(axis=1)).max(axis=0),
    }


def report_kinematics(kin, delta_lad, period, n_joints):
    """
    Print the feasibility / floor / budget table and return it as a dict.

    This is the single most useful thing in the file: it says BEFORE
    training whether the chosen deltas can physically track the gait, what
    RMSE floor they imply, and how many spikes per cycle the output neurons
    therefore have to produce.  A run whose final RMSE sits at the printed
    floor is not underfitting -- it is delta-limited, and the fix is a finer
    delta or a longer period, not more epochs.

    `delta_lad` is (n_scales, n_joints), coarse first.  With a ladder the
    three quantities decouple, which is the whole reason for it:

      velocity ceiling  sum over scales of delta (every scale can fire up on
                        the same step), so peak duty is measured against the
                        SUM, not against the coarse delta alone.
      resolution        the FINEST delta -- nothing stops a joint being
                        trimmed in the smallest increment available.
      spike budget      travel / (2 * sum(delta)) per neuron, which comes out
                        EQUAL for every scale once the travel is split in
                        proportion to each scale's delta. That equal-budget
                        split is what the calibration targets, so no scale
                        starts starved relative to another.
    """
    s      = kin["scale"]
    vmax   = kin["vmax"]
    travel = kin["travel"]
    n_scales = delta_lad.shape[0]

    total  = delta_lad.sum(axis=0)                  # velocity ceiling
    finest = delta_lad[-1]                          # resolution
    duty   = vmax / np.maximum(total, 1e-12)
    # Tracking error of a first-order quantiser is ~uniform over one step,
    # so RMSE ~ delta / sqrt(12).  Approximate: it ignores the neuron's own
    # latency, so treat it as a lower bound rather than a prediction.
    floor  = finest / math.sqrt(12.0)
    budget = travel / np.maximum(2.0 * total, 1e-12)

    hdr = (f"      {'joint':>5}  {'p2p':>8}  {'vmax':>10}"
           + "".join(f"  {('delta[' + str(i) + ']'):>9}"
                     for i in range(n_scales))
           + f"  {'peak duty':>10}  {'RMSE floor':>11}  {'spk/cyc/dir':>12}")
    print(hdr)
    print("      " + "-" * (len(hdr) - 6))
    for j in range(n_joints):
        warn = "  <-- INFEASIBLE" if duty[j] > 1.0 else (
               "  <-- >70% duty" if duty[j] > 0.7 else "")
        print(f"      {j:>5}  {kin['p2p'][j]*s:>7.1f}°"
              f"  {vmax[j]*s:>7.3f}°/st"
              + "".join(f"  {delta_lad[i][j]*s:>8.3f}°"
                        for i in range(n_scales))
              + f"  {100*duty[j]:>9.1f}%  {floor[j]*s:>10.3f}°"
              f"  {budget[j]:>12.1f}{warn}")
    print("      " + "-" * (len(hdr) - 6))
    print(f"      mean RMSE floor {float(np.mean(floor))*s:.3f}°   "
          f"worst {float(np.max(floor))*s:.3f}°   "
          f"mean budget {float(np.mean(budget)):.1f} spk/cyc/dir/scale   "
          f"(max possible {period:.0f})")
    if n_scales > 1:
        print(f"      ladder: velocity ceiling is the SUM of scales "
              f"({float(np.mean(total/delta_lad[0])):.2f}x the coarse delta), "
              f"resolution is the finest ({float(np.mean(delta_lad[0]/finest)):.0f}x "
              f"finer than coarse)")
    if (duty > 1.0).any():
        bad = [int(j) for j in np.where(duty > 1.0)[0]]
        print(f"      WARNING: joint(s) {bad} need the accumulator to move "
              f"faster than every scale firing at once. Raise "
              f"--delta_init_scale, or accept that those joints will lag at "
              f"the fastest part of the swing until the learnable deltas "
              f"grow.")
    return {
        "vmax_deg_per_step": (vmax * s).tolist(),
        "p2p_deg":           (kin["p2p"] * s).tolist(),
        "travel_deg":        (travel * s).tolist(),
        "delta_deg":         (delta_lad * s).tolist(),
        "delta_total_deg":   (total * s).tolist(),
        "delta_finest_deg":  (finest * s).tolist(),
        "peak_duty":         duty.tolist(),
        "rmse_floor_deg":    (floor * s).tolist(),
        "spikes_per_cycle_per_direction": budget.tolist(),
    }


def required_period_for_floor(gait_tables, floor_target_deg, delta_scale,
                              n_scales=2, delta_ratio=4.0):
    """
    Minimum CPG period (timesteps per gait cycle) whose delta-quantisation
    RMSE floor is at or under `floor_target_deg`, for the WORST joint.

    Every term but the period comes from the gait tables, so this needs no
    CPG run -- which is what makes the sizing a clean forward pass rather
    than a fixed-point iteration:

        vmax_deg(P) = maxrowdiff_deg * R / P        (R = rows per cycle)
        delta_deg   = delta_scale * vmax_deg
        floor_deg   = delta_deg / sqrt(12)

    floor is therefore EXACTLY proportional to 1/P, and inverting gives

        P_required = delta_scale * maxrowdiff_deg * R / (target * sqrt(12))

    Row differences are taken circularly (last row wraps to first), matching
    `joint_kinematics` -- the wrap-around step is real motion.

    The max is over gaits, rows AND joints: the target is a ceiling for
    every joint, so the fastest joint in the fastest gait sets the period.
    Pass the UPSAMPLED tables, in degrees.
    """
    tabs = np.stack([np.asarray(t, dtype=np.float64) for t in gait_tables])
    R = tabs.shape[1]
    d = np.abs(np.diff(np.concatenate([tabs, tabs[:, :1]], axis=1), axis=1))
    maxrowdiff = float(d.max())                     # deg between adjacent rows
    per_joint  = d.max(axis=(0, 1))                 # (J,) worst per joint

    # The floor is set by the FINEST delta, not the coarse one. The coarse
    # delta is pinned by the velocity requirement (it has to be able to
    # reach vmax), but the resolution is whatever the smallest increment is
    # -- so with a ladder of n_scales the required period drops by
    # ratio^(n_scales-1). That factor is the whole point of multi-scale: it
    # buys resolution without buying period, and period is expensive
    # (bptt and tau_max both track it).
    fine_factor = float(delta_ratio) ** (int(n_scales) - 1)
    P_req = (delta_scale * maxrowdiff * R
             / (fine_factor * float(floor_target_deg) * math.sqrt(12.0)))
    return {
        "period_required": float(P_req),
        "maxrowdiff_deg":  maxrowdiff,
        "vmax_per_cycle_deg": maxrowdiff * R,
        "rows":            int(R),
        "per_joint_maxrowdiff_deg": per_joint.tolist(),
        "floor_target_deg": float(floor_target_deg),
        "delta_scale":      float(delta_scale),
        "n_scales":         int(n_scales),
        "delta_ratio":      float(delta_ratio),
        "fine_factor":      fine_factor,
    }


def proprio_normalisation(gait_tables, tgt_range):
    """
    Per-joint centre and half-range for the proprioceptive input, expressed
    in the SAME globally-normalised units the accumulator works in.

    The targets use one global min/max across every gait and joint (so the
    robot needs a single denormalisation), which means a narrow joint only
    ever occupies a sliver of [-1, 1).  Fed back raw, that joint's channel
    would carry almost no dynamic range.  Rescaling per joint against its
    OWN observed span fixes that:

        p_j = (y_j - centre_j) / halfrange_j      ~ [-1, 1] over its range

    Centred rather than mapped to [0, 1] so that "mid-range" is the zero
    input, matching the target convention and giving layer 0 a zero-mean
    signal.

    A joint that never moves across any gait has zero span; its half-range
    is floored (in the model) rather than dividing by zero, which makes its
    channel a constant the network can ignore.
    """
    lo_g, hi_g = tgt_range
    span = (hi_g - lo_g) + 1e-8
    tabs = np.stack([np.asarray(t, dtype=np.float64) for t in gait_tables])
    lo_j = tabs.min(axis=(0, 1))                      # (J,) degrees
    hi_j = tabs.max(axis=(0, 1))
    # degrees -> global normalised: n = (deg - lo_g)/span * 2 - 1
    centre    = ((lo_j + hi_j) / 2.0 - lo_g) / span * 2.0 - 1.0
    halfrange = (hi_j - lo_j) / span                  # *2/2 cancels
    return {
        "centre":         centre,
        "halfrange":      halfrange,
        "lo_deg":         lo_j,
        "hi_deg":         hi_j,
        "degenerate":     [int(j) for j in np.where(halfrange < 1e-3)[0]],
    }


def delta_scale_ladder(delta_coarse, n_scales, ratio):
    """
    (n_scales, n_joints) delta ladder: coarse, coarse/ratio, coarse/ratio^2...

    Slow-twitch / fast-twitch, as it were: the coarse scale is sized by what
    the joint needs to KEEP UP (it must be able to reach peak velocity), the
    fine scale by what it needs to SETTLE (resolution in the flat parts of
    the waveform). One delta cannot serve both when the ratio between peak
    velocity and acceptable ripple is large, which is exactly the bind that
    forced the period up.
    """
    d0 = np.asarray(delta_coarse, dtype=np.float64)
    return np.stack([d0 / (float(ratio) ** s) for s in range(int(n_scales))])


def _measure_period(spk):
    """Median inter-burst interval of neuron 0, same method as analyse_cpg."""
    ts = np.where(spk[:, 0] > 0)[0]
    if len(ts) < 4:
        return float("nan")
    on = burst_onsets(ts, detect_burst_threshold(ts))
    if len(on) < 3:
        return float("nan")
    return float(np.median(np.diff(on)))


def probe_period(N, i_app, vth_fb, to_fb_weight, refrac_main, fake,
                 warmup=2000, min_cycles=12):
    """
    Measure the period a given `vth_fb` actually produces, cheaply.

    Needed because the closed-form period formula holds for the FAKE CPG
    only.  fake_step_chunk lays down back-to-back bursts, so
    period = N * spikes_per_burst * (refrac+1) exactly.  The real oscillator
    inserts recovery silence between bursts -- at the shipped defaults the
    burst arithmetic gives 4*10*2 = 80 steps but the measured period is 254,
    so roughly 174 steps of every cycle is inter-burst gap that no closed
    form here predicts.  So for the real CPG we measure instead of assuming.

    Warmup is skipped for the fake path: fake_step_chunk builds its pattern
    from scratch with no dependence on oscillator state, so warming up is
    provably a no-op there.
    """
    spb = max(1, int(vth_fb // to_fb_weight))
    predicted = N * spb * (refrac_main + 1)
    # Enough steps for a stable median even if the real period runs several
    # times longer than the burst arithmetic predicts.
    n_steps = int(max(min_cycles * predicted * 4, 4000))

    cpg = LIFCPGStepper(N=N, i_app=i_app, vth_fb=float(vth_fb),
                        to_fb_weight=float(to_fb_weight),
                        refrac_main=int(refrac_main))
    if fake:
        spk = cpg.fake_step_chunk(n_steps)
    else:
        cpg.step_chunk(int(warmup))
        spk = cpg.step_chunk(n_steps)
    return _measure_period(spk), predicted


def size_cpg(period_required, N, i_app, to_fb_weight, refrac_main, fake,
             max_iter=5, verbose=True):
    """
    Smallest `vth_fb` whose MEASURED period reaches `period_required`.

    The closed form gives the first guess, which is exact for the fake CPG
    (so the loop confirms and exits after one probe).  For the real CPG the
    gap makes the closed form a lower bound, so the first guess normally
    overshoots -- the loop then walks spikes_per_burst back down, because an
    overshoot is wasted compute: every extra step of period is another
    timestep per gait cycle to simulate and backprop through.

    Returns one dict either way, so callers do not branch on fake vs real.
    """
    per_burst = N * (refrac_main + 1)          # period per spike-per-burst
    spb = max(1, int(math.ceil(period_required / per_burst)))
    best = None
    seen = {}

    for it in range(max_iter):
        if spb in seen:
            break
        vth = spb * float(to_fb_weight)
        measured, predicted = probe_period(N, i_app, vth, to_fb_weight,
                                           refrac_main, fake)
        seen[spb] = measured
        if verbose:
            print(f"      probe {it+1}: spikes/burst={spb:>3d} "
                  f"vth_fb={vth:>6.0f}  predicted={predicted:>6d}  "
                  f"measured={measured:>7.1f}")
        if not math.isfinite(measured):
            print(f"      WARNING: could not detect a period at vth_fb="
                  f"{vth:g}; keeping this value and letting analyse_cpg "
                  f"report.")
            best = (spb, measured)
            break
        if measured >= period_required:
            if best is None or spb < best[0]:
                best = (spb, measured)
            # try to come back down toward the requirement
            nxt = max(1, int(math.ceil(spb * period_required / measured)))
            if nxt >= spb:
                break
            spb = nxt
        else:
            nxt = max(spb + 1, int(math.ceil(spb * period_required
                                             / measured)))
            spb = nxt

    if best is None:                            # never reached the target
        spb_final = max(seen, key=lambda k: seen[k] if
                        math.isfinite(seen[k]) else -1)
        best = (spb_final, seen[spb_final])
        print(f"      WARNING: could not reach the required period "
              f"{period_required:.0f} within {max_iter} probes; using the "
              f"longest found ({best[1]:.1f}). The delta RMSE floor will be "
              f"above --floor_target.")

    spb_final, measured = best
    return {
        "vth_fb":           spb_final * float(to_fb_weight),
        "to_fb_weight":     float(to_fb_weight),
        "refrac_main":      int(refrac_main),
        "spikes_per_burst": int(spb_final),
        "period_predicted": float(N * spb_final * (refrac_main + 1)),
        "period_measured":  float(measured),
        "period_target":    float(period_required),
        "probes":           {int(k): float(v) for k, v in seen.items()},
    }


def run_cpg_sized(N, tmax, warmup, i_app, vth_fb, to_fb_weight, refrac_main,
                  fake_cpg):
    """
    train.py's run_cpg, but with the burst parameters actually plumbed through.

    Reimplemented locally rather than called because train.py's
    `run_cpg` builds `LIFCPGStepper(N=N, i_app=i_app)` with default
    vth_fb/to_fb_weight/refrac_main, so there is no way to set the period
    through it.  Those three ARE constructor args of LIFCPGStepper, so this
    is the same object train.py uses, with nothing bypassed -- including
    `fake_step_chunk`, which is a method on it.  Keeping this here rather
    than editing train.py means the two files stay independent; if run_cpg
    later grows these parameters this function can go away.
    """
    cpg = LIFCPGStepper(N=N, i_app=i_app, vth_fb=float(vth_fb),
                        to_fb_weight=float(to_fb_weight),
                        refrac_main=int(refrac_main))
    print(f"  N={N}  i_app={i_app}  from_fb_weight={CPG_FROM_FB_WEIGHT:g}")
    print(f"  vth_fb={vth_fb:g}  to_fb_weight={to_fb_weight:g}  "
          f"refrac_main={refrac_main}  -> "
          f"{int(vth_fb // to_fb_weight)} spikes/burst")
    if fake_cpg:
        print(f"  FAKE CPG: back-to-back bursts, no inter-burst gap "
              f"(see fake_step_chunk)")
        print(f"  Collecting {tmax} steps ...")
        spikes = cpg.fake_step_chunk(tmax)
    else:
        print(f"  Warming up CPG ({warmup} steps) ...")
        cpg.step_chunk(int(warmup))
        print(f"  Collecting {tmax} steps ...")
        spikes = cpg.step_chunk(tmax)

    counts = spikes.sum(0).astype(int)
    print(f"  Spikes per neuron : {counts.tolist()}")
    if counts.min() == 0:
        raise RuntimeError("A CPG neuron never fired — check W / i_app.")
    return spikes


# ═══════════════════════════════════════════════════════════════════
# 2.  Model
# ═══════════════════════════════════════════════════════════════════

class DeltaSNN(nn.Module):
    """
    CPG spikes -> dense LIF stack -> 2 spiking neurons per joint -> accumulator.

    State, in `state_names_in` order:
        mem1 .. memN   (B, hidden)     one per hidden layer
        mem_out        (B, 2*n_joints) output-neuron membranes
        acc            (B, n_joints)   accumulated OFFSET FROM POSE

    `acc` holds the offset, not the angle.  That matters: `apply_reset`
    zeroes state tensors for streams that were rewound, and zero is the
    correct reset value for an offset but would be a badly wrong joint
    angle.  The pose is added back in `step`, so a reset lands the robot in
    the standing pose rather than at 0 degrees, with no special-case code
    in the reset path.

    Output layer notes
    ------------------
    * No LayerNorm.  LN over the output dimension would (a) normalise away
      the drive magnitude, which is exactly what encodes firing rate here,
      and (b) make every joint's rate depend on every other joint's, since
      LN subtracts the mean across the 2*n_joints channels.
    * `subtract` reset by default.  The residual `mem - thresh` carrying
      over is what makes rate graded in the drive, and rate coding is the
      whole mechanism here.  `--out_reset zero` is kept for A/B.
    * A per-gait output bias (zero-init `Embedding`) lets a gait set tonic
      drive per neuron.  Zero-init keeps it inert at init so it cannot
      disturb `calibrate_out_bias`.

    `delta` is stored as a log so it stays strictly positive under
    unconstrained gradient descent, and so updates are multiplicative --
    delta spans a factor of a few across joints, and an additive step big
    enough to matter for the fastest joint would be huge for the slowest.
    """

    arch = "delta"

    def __init__(self, hidden=64, n_layers=3, n_gaits=4, max_gaits=16,
                 n_neurons=4, n_joints=8,
                 tau_min=2.0, tau_max=256.0,
                 tau_out_min=2.0, tau_out_max=20.0,
                 slope=25.0, out_slope=5.0, thresh=1.0,
                 delta_init=None, pose_init=None, pose_learnable=True,
                 out_reset="subtract", out_gait_bias=True, angle_leak=0.0,
                 out_mem_clip=0.0, n_scales=2,
                 proprio=True, prop_center=None, prop_halfrange=None):
        super().__init__()
        if n_gaits > max_gaits:
            raise ValueError(
                f"n_gaits ({n_gaits}) > max_gaits ({max_gaits}); raise "
                f"--max_gaits. Changing max_gaits changes the FiLM parameter "
                f"shape and so invalidates old checkpoints.")
        if n_layers < 1:
            raise ValueError(f"--n_layers must be >= 1, got {n_layers}.")
        if n_scales < 1:
            raise ValueError(f"--delta_scales must be >= 1, got {n_scales}.")

        self.H          = int(hidden)
        self.n_layers   = int(n_layers)
        self.J          = int(n_joints)
        self.n_joints   = int(n_joints)          # export_onnx reads this name
        self.n_scales   = int(n_scales)
        # 2 directions x n_scales x n_joints
        self.n_out      = 2 * self.n_scales * self.J
        self.n_neurons  = int(n_neurons)
        self.n_gaits    = int(n_gaits)
        self.max_gaits  = int(max_gaits)
        self.slope      = float(slope)
        self.out_slope  = float(out_slope)
        self.thresh     = float(thresh)
        self.out_reset  = str(out_reset)
        self.out_mem_clip = float(out_mem_clip)
        self.use_gait_bias = bool(out_gait_bias)
        # A plain float, not a buffer or a Parameter.
        #   Not learnable: it would trade waveform fidelity for drift
        #   suppression and win on the short training chunks while losing on
        #   the long free run.
        #   Not a buffer either, so it stays OUT of the state dict -- a
        #   buffer would mean a reloaded checkpoint silently overrode
        #   whatever --angle_leak the caller asked for. It belongs with
        #   slope/thresh: part of the configured model, recorded in the
        #   config JSON, not in the weights.
        self.leak = float(angle_leak)

        H = self.H
        # ── hidden stack ──────────────────────────────────────────
        # Layer 0 is driven by the CPG (n_neurons wide, and exactly one CPG
        # neuron fires per timestep, so its current is a single row of w);
        # 0.8 carries over from train.py's StatefulSNN so the two are
        # comparable at init.
        # Proprioception widens layer 0's input: the CPG spikes plus one
        # continuous channel per joint carrying that joint's own commanded
        # angle. That closes the loop -- without it the network emits
        # increments with no knowledge of where the joint ended up, so any
        # up/down imbalance accumulates invisibly and nothing local to a
        # timestep can correct it.
        self.proprio = bool(proprio)
        self.n_in = self.n_neurons + (self.J if self.proprio else 0)

        widths = [self.n_in] + [H] * (self.n_layers - 1)
        self.w_h = nn.ParameterList([
            nn.Parameter(torch.randn(w_in, H) *
                         (0.8 if i == 0 else 1.0 / math.sqrt(w_in)))
            for i, w_in in enumerate(widths)])
        if self.proprio:
            # Rescale ONLY the proprioceptive rows of layer 0. The 0.8 above
            # is tuned for a ONE-HOT input: exactly one CPG neuron fires per
            # timestep, so a hidden unit receives a single weight, std 0.8.
            # The proprioceptive block is J continuous channels at once, so
            # at the same scale it would contribute std 0.8*0.577*sqrt(J) --
            # about 1.6x the CPG's own drive at J=8, i.e. proprioception
            # would dominate the rhythm before training starts. Matching the
            # one-hot contribution gives 0.8/(0.577*sqrt(J)).
            with torch.no_grad():
                self.w_h[0][self.n_neurons:].mul_(
                    1.0 / (0.577 * math.sqrt(self.J)))
            # Per-joint affine into [-1, 1] over that joint's OWN observed
            # range, so a 46-degree joint gets the same input dynamic range
            # as a 92-degree one. Deliberately NOT clamped: when the
            # accumulator drifts past the observed range the normalised
            # value exceeds 1, and that excess magnitude is exactly the
            # signal needed to correct it -- clamping would erase the
            # difference between slightly out and far out.
            c0 = (torch.zeros(self.J) if prop_center is None else
                  torch.as_tensor(np.asarray(prop_center, dtype=np.float32)))
            h0 = (torch.ones(self.J) if prop_halfrange is None else
                  torch.as_tensor(np.asarray(prop_halfrange,
                                             dtype=np.float32)))
            self.register_buffer("prop_center", c0.clone())
            self.register_buffer("prop_halfrange", h0.clamp(min=1e-3).clone())
        self.b_h = nn.ParameterList([
            nn.Parameter(torch.zeros(H)) for _ in range(self.n_layers)])
        # affine=False: FiLM supplies the affine, so LN's own would be
        # redundant (and its gamma would fight FiLM's for the same job).
        self.ln = nn.ModuleList([
            nn.LayerNorm(H, elementwise_affine=False)
            for _ in range(self.n_layers)])
        self.beta_logit = nn.ParameterList([
            nn.Parameter(init_beta_logit((H,), tau_min, tau_max))
            for _ in range(self.n_layers)])

        # ── per-gait FiLM, over-allocated to max_gaits ─────────────
        # Applied AFTER LayerNorm, for the reason spelled out in
        # StatefulSNN's docstring: an input-concatenated gait flag is
        # partly erased by the very LN that keeps the LIF unsaturated.
        self.film = nn.ModuleList([
            nn.Embedding(max_gaits, 2 * H) for _ in range(self.n_layers)])
        for e in self.film:
            nn.init.zeros_(e.weight)
            e.weight.data[:, :H] = 1.0              # gamma := 1, beta := 0

        # ── spiking output layer ──────────────────────────────────
        # Flat index = (scale * 2 + direction) * J + joint, so the layout is
        #   [c_up(J), c_dn(J), f_up(J), f_dn(J), ...]
        # coarse-first. At n_scales=1 that degenerates to the original
        # [up(J), dn(J)], which keeps the single-scale architecture
        # reachable as a strict special case rather than a separate branch.
        self.w_out = nn.Parameter(torch.randn(H, self.n_out) / math.sqrt(H))
        self.b_out = nn.Parameter(torch.zeros(self.n_out))
        self.beta_out_logit = nn.Parameter(
            init_beta_logit((self.n_out,), tau_out_min, tau_out_max))
        if self.use_gait_bias:
            self.gb_out = nn.Embedding(max_gaits, self.n_out)
            nn.init.zeros_(self.gb_out.weight)

        # ── delta ladder (learnable, per scale per joint) ──────────
        # Shape (n_scales, J). The scales are INDEPENDENTLY learnable rather
        # than a coarse delta times a fixed ratio: pinning the ratio would
        # make the fine scale's resolution hostage to the coarse scale's
        # velocity requirement, which is the coupling the ladder exists to
        # break. The cost is that nothing forces them to stay separated --
        # they could collapse onto each other and waste half the output
        # layer -- so delta_report prints the learned ratio to catch that.
        if delta_init is None:
            delta_init = np.full((self.n_scales, self.J), 0.02,
                                 dtype=np.float64)
        d0 = torch.as_tensor(np.asarray(delta_init, dtype=np.float64),
                             dtype=torch.float32).clamp(min=1e-9)
        if d0.shape != (self.n_scales, self.J):
            raise ValueError(
                f"delta_init has shape {tuple(d0.shape)}, expected "
                f"(n_scales, n_joints) = ({self.n_scales}, {self.J}).")
        self.log_delta = nn.Parameter(torch.log(d0))

        # ── standing pose ─────────────────────────────────────────
        # Registered under ONE name either way so a checkpoint moves
        # between --pose fixed and --pose learnable without key surgery.
        p0 = torch.zeros(self.J) if pose_init is None else \
            torch.as_tensor(np.asarray(pose_init, dtype=np.float32))
        if p0.numel() != self.J:
            raise ValueError(f"pose_init has {p0.numel()} entries, "
                             f"expected n_joints={self.J}.")
        self.pose_learnable = bool(pose_learnable)
        if self.pose_learnable:
            self.pose = nn.Parameter(p0.clone())
        else:
            self.register_buffer("pose", p0.clone())

        # Built here rather than as class constants because the count
        # depends on --n_layers; export_onnx and the config read them off
        # the instance so neither can drift from the real state layout.
        self.state_names_in = tuple(
            [f"mem{i+1}_in" for i in range(self.n_layers)]
            + ["mem_out_in", "acc_in"])
        self.state_names_out = tuple(
            [f"mem{i+1}_out" for i in range(self.n_layers)]
            + ["mem_out_out", "acc_out"])

    # ---------------------------------------------------------------
    def delta(self):
        return torch.exp(self.log_delta)                    # (n_scales, J)

    def init_state(self, batch, device, dtype=torch.float32):
        mems = [torch.zeros(batch, self.H, device=device, dtype=dtype)
                for _ in range(self.n_layers)]
        mem_out = torch.zeros(batch, self.n_out, device=device, dtype=dtype)
        acc     = torch.zeros(batch, self.J, device=device, dtype=dtype)
        return tuple(mems) + (mem_out, acc)

    def step(self, x, gait, state):
        """
        x     : (B, n_neurons) CPG spikes this timestep
        gait  : (B,) int64
        state : see class docstring

        Returns (angle, new_state, (up, down)) -- the 3-tuple contract
        train.py's models use, so the shared training loop and plots do
        not branch on model type.
        """
        mems    = list(state[:self.n_layers])
        mem_out = state[self.n_layers]
        acc     = state[self.n_layers + 1]

        if self.proprio:
            # The angle the network is looking at is the one produced by the
            # PREVIOUS step -- `acc` in the incoming state is acc[t-1] -- so
            # this is causal, the same one-step-stale reading a real encoder
            # would give.
            #
            # DETACHED on purpose. Without it there is a second gradient
            # path (spike -> acc -> next input -> next spike) stacked on top
            # of the existing spike -> acc -> every later loss, and that
            # second-order "if I spike now my future input changes" term is
            # noise-dominated early while roughly doubling the graph. The
            # network still learns to USE the signal: d loss/d w_h[0] is
            # nonzero through the spikes it drives.
            y_prev = (self.pose + acc).detach()
            p_in = (y_prev - self.prop_center) / self.prop_halfrange
            h = torch.cat([x, p_in], dim=1)
        else:
            h = x
        for i in range(self.n_layers):
            cur = torch.addmm(self.b_h[i], h, self.w_h[i])
            cur = self.ln[i](cur)
            v   = self.film[i](gait)                        # (B, 2H)
            cur = cur * v[:, :self.H] + v[:, self.H:]
            beta = torch.sigmoid(self.beta_logit[i])
            m    = beta * mems[i] + cur
            s    = spike_fn(m - self.thresh, self.slope)
            mems[i] = m - self.thresh * s                   # subtract reset
            h = s

        # ── output neurons ────────────────────────────────────────
        cur_o = torch.addmm(self.b_out, h, self.w_out)       # (B, n_out)
        if self.use_gait_bias:
            cur_o = cur_o + self.gb_out(gait)
        beta_o  = torch.sigmoid(self.beta_out_logit)
        mem_out = beta_o * mem_out + cur_o
        spk     = spike_fn(mem_out - self.thresh, self.out_slope)
        if self.out_reset == "subtract":
            mem_out = mem_out - self.thresh * spk
        else:
            mem_out = mem_out * (1.0 - spk)
        # Optional residual bound, OFF by default (--out_mem_clip 0).
        # Subtract-reset lets an over-driven membrane integrate without
        # limit: measured at tau=20, a drive of 10x threshold settles at
        # mem~185, and when the drive stops the neuron keeps firing for 46
        # more steps. Whether that ever happens here is an empirical
        # question -- after calibration these neurons sit near 0.16
        # spikes/step, far from saturation -- so this is available rather
        # than applied. Note the clamp has zero gradient where active.
        if self.out_mem_clip > 0:
            c = self.out_mem_clip * self.thresh
            mem_out = mem_out.clamp(-c, c)

        # Split the flat output into (B, n_scales, 2, J) -- the layout is
        # (scale, direction, joint), coarse scale first.
        s4 = spk.reshape(-1, self.n_scales, 2, self.J)
        up, dn = s4[:, :, 0, :], s4[:, :, 1, :]      # (B, n_scales, J)

        # ── accumulate ────────────────────────────────────────────
        # Every scale contributes its own increment on the same timestep, so
        # the velocity ceiling is sum(delta) and the resolution is min(delta)
        # -- the decoupling the ladder is for. Net increment per scale, so an
        # up and a down in the same timestep cancel to nothing; that is
        # wasteful rather than wrong (--cofire_lambda can charge for it, and
        # the diagnostic report always measures it).
        acc = ((1.0 - self.leak) * acc
               + (self.delta().unsqueeze(0) * (up - dn)).sum(dim=1))
        y   = self.pose + acc                                # (B, J)

        return y, tuple(mems) + (mem_out, acc), (up, dn)

    def forward(self, x_seq, gait_seq, state=None, return_aux=False):
        """
        x_seq    : (L, B, n_neurons)
        gait_seq : (L, B)

        Returns (y_seq, state), or (y_seq, state, (up_seq, dn_seq)).
        Signature matches train.py's models so plot_reconstruction /
        plot_transition work unchanged.
        """
        L, B = x_seq.shape[0], x_seq.shape[1]
        if state is None:
            state = self.init_state(B, x_seq.device, x_seq.dtype)
        ys, ups, dns = [], [], []
        for t in range(L):
            y, state, aux = self.step(x_seq[t], gait_seq[t], state)
            ys.append(y)
            if return_aux:
                ups.append(aux[0])
                dns.append(aux[1])
        if return_aux:
            return (torch.stack(ys), state,
                    (torch.stack(ups), torch.stack(dns)))
        return torch.stack(ys), state

    def param_breakdown(self):
        n = lambda ps: int(sum(p.numel() for p in ps))
        return {
            "hidden": n(self.w_h) + n(self.b_h) + n(self.beta_logit),
            "film":   n(e.weight for e in self.film),
            "out":    int(self.w_out.numel() + self.b_out.numel()
                          + self.beta_out_logit.numel()
                          + (self.gb_out.weight.numel()
                             if self.use_gait_bias else 0)),
            "delta":  int(self.log_delta.numel()),
            # 0 when --pose fixed: it is a buffer then, not a parameter.
            "pose":   int(self.pose.numel()) if self.pose_learnable else 0,
        }


class SingleStepONNXDelta(nn.Module):
    """One timestep, flat tensors in and out, for deployment export.

    `*states` rather than named arguments because the state length depends
    on --n_layers; torch.onnx.export passes the dummy tuple positionally,
    so varargs trace fine and the names come from `input_names`.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spikes, gait, *states):
        y, new_state, aux = self.model.step(spikes, gait.long(), tuple(states))
        # up/down are exported too, as (batch, n_scales, n_joints): the
        # robot side may want the raw increments (e.g. to drive a servo in
        # steps) rather than the accumulator's opinion of the absolute
        # angle. Pair them with cfg["delta"]["degrees"], which is the
        # matching (n_scales, n_joints) table.
        return (y, aux[0], aux[1]) + tuple(new_state)


# ═══════════════════════════════════════════════════════════════════
# 3.  Output-rate measurement + bias calibration
# ═══════════════════════════════════════════════════════════════════

@contextmanager
def eager_step(model):
    """
    Temporarily restore the uncompiled `step`.

    Every distinct batch size traced by Dynamo is a separate graph, and
    past 8 recompiles it silently falls back to eager for everything.  The
    training loop runs at `--batch`, but calibration and the diagnostic
    report run at `n_gaits` and the plots run at 1 -- three shapes, doubled
    by train/eval mode, which is uncomfortably close to the limit for zero
    benefit (these paths are forward-only and run a handful of times).
    Calibration in particular would pay the compile cost on its first
    iteration and then mutate `b_out` 22 times underneath the graph.

    So: compile is for the training loop only, and everything else asks for
    eager explicitly.  A no-op when step was never compiled (CPU runs).

    Gated on an explicit `_step_is_compiled` flag rather than on
    `model.step is not model._step_eager`.  That identity test cannot work:
    `step` is a plain method, so each attribute access builds a FRESH bound
    method object and the comparison is False even when nothing was
    compiled.  (train.py's export_onnx has the same test, which is why it
    prints "using eager step() for export" on CPU runs where there was
    never a compiled step -- harmless there, since the swap is a no-op, but
    not something to copy.)
    """
    compiled = None
    if getattr(model, "_step_is_compiled", False):
        compiled = model.step
        model.step = model._step_eager
    try:
        yield model
    finally:
        if compiled is not None:
            model.step = compiled


@torch.no_grad()
def measure_out(model, spikes, n_gaits, device, period, t0=0, n_cycles=6):
    """
    Replay the CPG spike train and measure what the output layer does.

    All gaits are run in ONE forward pass by putting them in the batch
    dimension (same spike input, different gait index per column).  The
    calibration below calls this ~24 times, so a per-gait Python loop over
    thousands of timesteps would dominate startup.

    The first `period` steps are discarded: the membranes start at zero and
    the rate over that window is not the steady-state rate.

    Returns dict of (n_gaits, n_joints) arrays plus the free-run angles.
    """
    n_steps = int(min(n_cycles * period, len(spikes) - t0))
    x = torch.as_tensor(spikes[t0:t0 + n_steps], dtype=torch.float32,
                        device=device)
    x = x.unsqueeze(1).expand(n_steps, n_gaits, x.shape[-1]).contiguous()
    gg = torch.arange(n_gaits, device=device, dtype=torch.long) \
              .view(1, -1).expand(n_steps, n_gaits).contiguous()

    with eager_step(model):
        y, _, (up, dn) = model(x, gg, return_aux=True)

    warm = int(min(period, n_steps // 2))
    ncyc = max((n_steps - warm) / float(period), 1e-9)
    return {
        "y":      y,                                        # (L, G, J)
        "t0":     int(t0),
        "warm":   warm,
        # (G, n_scales, J) -- the scale axis came in with the delta ladder
        "rate_up": (up[warm:].sum(0) / ncyc).cpu().numpy(),
        "rate_dn": (dn[warm:].sum(0) / ncyc).cpu().numpy(),
        "cofire":  (up[warm:] * dn[warm:]).mean(0).cpu().numpy(),
    }


@torch.no_grad()
def calibrate_out_bias(model, spikes, n_gaits, device, period, target,
                       tol=0.5, iters=22, b_lo=-10.0, b_hi=10.0,
                       verbose=True):
    """
    Bisect each output neuron's bias so it starts near its spike BUDGET.

    Why this exists.  The budget (`travel / 2*delta`, see
    `report_kinematics`) is around 40 spikes/cycle/direction at the
    quadruped period, while a randomly initialised layer is as likely to be
    silent as saturated.  A silent output neuron is close to unrecoverable:
    the accumulator never moves, so the only gradient reaching it is the
    surrogate's tail, which at slope=5 and a membrane a few units below
    threshold is order 1e-2 of full scale and at slope=25 is order 1e-3.
    Starting in the right neighbourhood is cheaper than hoping.

    Why bias and not weight scale.  An output neuron is supposed to have a
    tonic component: a joint sweeping steadily needs increments at a steady
    rate, so a baseline drive is part of the job rather than a contaminant.
    Bias is the honest handle for that, and it leaves `w_out`'s random phase
    preference untouched.

    Monotonicity: rate is non-decreasing in bias (more current can only add
    threshold crossings), so bisection is valid.  Rate is a STEP function
    of bias, which is why the target is a band (+/- `tol`) and not a value.
    The rate targeted is the MINIMUM over gaits -- the failure that matters
    is silent-for-one-gait, not silent-on-average.
    """
    tgt = torch.as_tensor(np.asarray(target, dtype=np.float32),
                          device=device)
    lo  = torch.full_like(tgt, float(b_lo))
    hi  = torch.full_like(tgt, float(b_hi))

    def rates_at(b):
        model.b_out.data.copy_(b)
        st = measure_out(model, spikes, n_gaits, device, period, n_cycles=5)
        # Interleave back into the model's flat output order,
        # (scale*2 + direction)*J + joint, so index k of `r` is the same
        # neuron as index k of b_out. Stacking on a new axis 2 and
        # reshaping does exactly that; concatenating would not.
        r = np.stack([st["rate_up"], st["rate_dn"]], axis=2) \
              .reshape(st["rate_up"].shape[0], -1)          # (G, n_out)
        return torch.as_tensor(r, dtype=torch.float32,
                               device=device).min(dim=0).values

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r   = rates_at(mid)
        quiet = r < tgt * (1.0 - tol)
        loud  = r > tgt * (1.0 + tol)
        lo = torch.where(quiet, mid, lo)
        hi = torch.where(loud,  mid, hi)
        if not (quiet | loud).any():
            break

    final = 0.5 * (lo + hi)
    rates = rates_at(final)
    ok    = (rates >= tgt * (1.0 - tol) * 0.5) & (rates <= tgt * (1.0 + tol) * 2.0)

    rep = {
        "bias":      [float(v) for v in final.cpu()],
        "min_rate":  [float(v) for v in rates.cpu()],
        "target":    [float(v) for v in tgt.cpu()],
        "in_band":   [bool(v) for v in ok.cpu()],
    }
    if verbose:
        J, S = model.J, model.n_scales
        fmt = lambda v: " ".join(f"{x:6.1f}" for x in v)
        blk = lambda v, k, d: v[(k * 2 + d) * J:(k * 2 + d + 1) * J]
        for name, key in (("target spk/cyc", "target"),
                          ("achieved (min)", "min_rate"),
                          ("bias", "bias")):
            for k in range(S):
                tag = "coarse" if k == 0 else ("fine" if k == S - 1
                                               else f"mid{k}")
                for d, dn in ((0, "up"), (1, "dn")):
                    lbl = f"{name:>15}" if (k == 0 and d == 0) else " " * 15
                    print(f"      {lbl} {tag:>6} {dn} "
                          f"[{fmt(blk(rep[key], k, d))}]")
        bad = [i for i, v in enumerate(ok.cpu()) if not v]
        if bad:
            decode = [f"{i}(scale{(i//J)//2},"
                      f"{'up' if (i//J) % 2 == 0 else 'dn'},j{i % J})"
                      for i in bad]
            print(f"      WARNING: output neuron(s) {decode} "
                  f"could not be brought near their budget by bias alone. A "
                  f"neuron pinned at 1 spike/step is saturated -- its delta is "
                  f"too small for the required velocity; check the peak-duty "
                  f"column above.")
    return rep


# ═══════════════════════════════════════════════════════════════════
# 4.  Diagnostics
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def delta_report(model, spikes, targets, valid, period, n_gaits, device,
                 tgt_range, gait_names, t0=0, n_cycles=8, indent="      "):
    """
    Per-gait output-layer report.  Returns (lines, stats) so it can be
    handed to the training loop as a `diag` callable, matching train.py's
    `timing_diag` contract.

    The three numbers that matter, per gait:
      rate up/dn   -- against the budget printed at startup.  Both far
                      below it means the joint cannot reach the extremes.
      balance      -- (up - dn) / (up + dn) summed over the cycle.  A
                      periodic waveform needs equal up and down travel, so
                      this should sit near 0; a persistent bias is DRIFT,
                      and it will grow without bound over a long free run.
      cofire       -- fraction of steps where up and down both fire and
                      cancel. Pure waste; if it is large, --cofire_lambda.
    """
    lo, hi = tgt_range
    scale, shift = (hi - lo) / 2.0, (hi + lo) / 2.0
    model.eval()

    st = measure_out(model, spikes, n_gaits, device, period,
                     t0=t0, n_cycles=n_cycles)
    warm = st["warm"]
    # Drop the warm-up window from the error figures as well as from the
    # rates: the membranes and the accumulator start at zero, so the model
    # is at `pose` for a phase it has no way of knowing yet, and including
    # that makes the RMSE a report on the reset transient.
    y  = st["y"][warm:]                              # (L, G, J)
    L  = y.shape[0]
    a  = t0 + warm

    tg = torch.as_tensor(targets[:, a:a + L], dtype=torch.float32,
                         device=device).permute(1, 0, 2)      # (L, G, J)
    vm = torch.as_tensor(valid[a:a + L].astype(np.float32),
                         device=device).view(L, 1, 1)
    err = (y - tg) * vm
    denom = vm.sum().clamp(min=1.0) * y.shape[2]
    rmse_g = torch.sqrt((err ** 2).sum(dim=(0, 2)) / denom) * scale
    bias_g = (err.sum(dim=(0, 2)) / denom) * scale   # signed drift, degrees

    d_deg = (model.delta().detach().cpu().numpy() * scale)   # (n_scales, J)
    n_sc  = d_deg.shape[0]

    lines = [f"{indent}output layer (free run, {n_cycles} cycles, "
             f"{warm}-step warm-up discarded):"]
    lines.append(f"{indent}  {'gait':>10} {'scale':>6}  {'spk/cyc up':>11}  "
                 f"{'dn':>7}  {'balance':>8}  {'cofire':>8}  "
                 f"{'RMSE':>8}  {'mean err':>9}")
    stats = []
    for g in range(n_gaits):
        for k in range(n_sc):
            ru, rd = st["rate_up"][g][k], st["rate_dn"][g][k]
            tot    = ru + rd
            bal    = float(np.mean((ru - rd) / np.maximum(tot, 1e-9)))
            cf     = float(np.mean(st["cofire"][g][k]))
            tag    = "coarse" if k == 0 else ("fine" if k == n_sc - 1
                                              else f"mid{k}")
            # RMSE and mean error are per-gait, not per-scale (the
            # accumulator is shared), so only print them on the first row.
            rc = (f"{float(rmse_g[g]):>7.2f}°  {float(bias_g[g]):>+8.2f}°"
                  if k == 0 else " " * 19)
            nm = gait_names[g] if k == 0 else ""
            lines.append(f"{indent}  {nm:>10} {tag:>6}  {ru.mean():>11.1f}  "
                         f"{rd.mean():>7.1f}  {bal:>+8.3f}  {cf:>8.4f}  {rc}")
            stats.append({
                "gait":     gait_names[g],
                "scale":    int(k),
                "rate_up":  ru.tolist(),
                "rate_dn":  rd.tolist(),
                "balance":  bal,
                "cofire":   cf,
                "rmse_deg": float(rmse_g[g]),
                "mean_err_deg": float(bias_g[g]),
            })
    for k in range(n_sc):
        lines.append(f"{indent}  delta[{k}] (deg/spike): " +
                     " ".join(f"{v:.4f}" for v in d_deg[k]))
    if n_sc > 1:
        # The scales are independently learnable, so nothing stops them
        # converging onto each other -- which would silently waste half the
        # output layer. A learned ratio near 1 means the ladder collapsed.
        ratio = d_deg[0] / np.maximum(d_deg[-1], 1e-12)
        lines.append(f"{indent}  learned coarse/fine ratio: " +
                     " ".join(f"{v:.2f}" for v in ratio) +
                     f"   (mean {float(np.mean(ratio)):.2f})")
        if float(np.mean(ratio)) < 1.5:
            lines.append(f"{indent}  WARNING: the delta ladder has collapsed "
                         f"(coarse ~= fine). Half the output layer is doing "
                         f"the other half's job; the resolution gain is gone.")
    return lines, stats


def grad_blocks_delta(model):
    """
    Coarse parameter blocks for per-block gradient / update reporting.

    Same idea as train.py's `grad_blocks`, but the parameter names differ
    so the rules do too.  `delta` and `pose` get their own blocks despite
    being tiny: they are the two parameters that can silently absorb a
    systematic error (delta soaking up a velocity mismatch, pose soaking up
    a DC offset), and seeing them move is how you tell that is happening.

    Prefix collisions checked: "beta_out_logit" does not start with
    "beta_logit" or "b_out", so the output block and the hidden block
    cannot claim each other's tensors regardless of rule order.
    """
    rules = (
        ("out",       ("w_out", "b_out", "beta_out_logit", "gb_out")),
        ("delta",     ("log_delta",)),
        ("pose",      ("pose",)),
        ("hidden",    ("w_h", "b_h", "beta_logit")),
        ("film",      ("film",)),
        ("layernorm", ("ln",)),
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


def masked_mean(v, mask):
    """
    Mean of v over the (L,B) mask, averaging every trailing dim.

    Rank-agnostic because the output aux tensors gained a scale axis: they
    are (L, B, n_scales, n_joints) now, not (L, B, n_joints).
    """
    flat = v.reshape(v.shape[0], v.shape[1], -1).mean(dim=2)
    return ((flat * mask).sum() / mask.sum().clamp(min=1.0))


def windowed_deriv_loss(pred, targ, mask, w):
    """
    MSE on the w-step displacement:  (pred[t]-pred[t-w]) vs (targ[t]-targ[t-w]).

    Why windowed rather than a single-step difference.  The single-step
    target difference is not the gait's velocity -- the target sequence is
    row-quantised (see joint_kinematics), so `targ[t]-targ[t-1]` is zero
    for `period/R` steps and then a jump.  Regressing the increment onto
    that would ask the output neurons to reproduce a sampling artifact.
    Over a window of ~`period/R` steps or more the quantisation averages
    out and what is left is real velocity.

    Cheap in the useful way: `pred[t]-pred[t-w]` is exactly `delta` times
    the NET SPIKE COUNT in the window, so this term speaks directly to the
    quantity the output neurons control, and it is blind to accumulated DC
    offset -- which the plain MSE already handles.
    """
    if w <= 0 or pred.shape[0] <= w:
        return pred.new_zeros(())
    dp = pred[w:] - pred[:-w]
    dt = targ[w:] - targ[:-w]
    m  = mask[w:] * mask[:-w]
    err = ((dp - dt) ** 2).mean(dim=2)
    return (err * m).sum() / m.sum().clamp(min=1.0)


# ═══════════════════════════════════════════════════════════════════
# 5.  Training loop
# ═══════════════════════════════════════════════════════════════════

def run_training(model, tr_sampler, va_sampler, opt, sched, device, args,
                 gait_w, out_dir, diag=None, n_gaits=4, period=254.0):
    """
    Adapted from train.py's `run_training`.  Same metric columns and same
    Ctrl+C-exports-anyway behaviour, so runs are comparable; the
    differences are the extra loss terms (windowed derivative, co-fire,
    spike L1) and that the diagnostic hook reports the output layer.
    """
    best = float("inf")
    best_path = out_dir / "best_model.pt"
    hist = {"train": [], "val": [], "val_sw": [], "gnorm": [], "sec": [],
            "mse": [], "deriv": [], "cofire": [], "rate": [], "upd": []}
    last_stats = []

    blocks = grad_blocks_delta(model)
    for b in blocks:
        hist[f"g_{b}"] = []
        hist[f"u_{b}"] = []
    n_par = sum(p.numel() for p in model.parameters())
    exp_upd = args.lr * math.sqrt(n_par) * math.sqrt(args.chunks_per_epoch)
    metrics = MetricsWriter(out_dir / "metrics.csv")
    print(f"\n  Per-epoch metrics -> {out_dir / 'metrics.csv'}")
    print(f"  Gradient blocks: {', '.join(sorted(blocks))}")
    print(f"  |upd| = per-epoch ||delta theta||; order-of-magnitude "
          f"expectation at lr={args.lr:g} is ~{exp_upd:.2f}")
    print(f"  NOTE |grad| is expected to run HIGHER than in train.py: the "
          f"accumulator")
    print(f"  makes one output spike affect every later step in the chunk, so "
          f"the")
    print(f"  output block's gradient scales with --bptt={args.bptt}. If it "
          f"sits far")
    print(f"  above --clip={args.clip:g}, most updates are being truncated and "
          f"the LR is too hot.")

    use_aux = (args.cofire_lambda > 0 or args.spike_lambda > 0
               or args.log_every_aux)

    print(f"\n  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  "
          f"{'Val(post-sw)':>13}  {'LR':>9}  {'|grad|':>8}  {'|upd|':>8}"
          f"  {'sec':>6}")
    print("  " + "-" * 88)
    print(f"  (Ctrl+C stops training and exports.)")

    try:
        for epoch in range(1, args.epochs + 1):
            t_epoch = time.perf_counter()
            theta0 = {b: [p.detach().clone() for p in plist]
                      for b, plist in blocks.items()}
            bacc = {b: torch.zeros((), device=device) for b in blocks}

            # ---- train ------------------------------------------------
            model.train()
            state = model.init_state(args.batch, device)
            tot = gtot = 0.0
            mse_t = der_t = cof_t = rate_t = 0.0
            nb = 0
            for _ in range(args.chunks_per_epoch):
                (x, g, y, m, sw, rst,
                 ph, warm) = tr_sampler.next_chunk(args.bptt)
                x, g, y, m = (x.to(device), g.to(device),
                              y.to(device), m.to(device))
                warm = warm.to(device)
                state = apply_reset(detach_state(state), rst.to(device))

                # `warm` zeroes steps whose head had its state wiped less
                # than one cycle ago. Load-bearing here in a way it is not
                # in train.py: a freshly reset head sits at `pose` with an
                # empty accumulator, and pose is not the correct angle for
                # any particular phase, so those steps carry a large error
                # that has nothing to do with the model being wrong.
                m_eff = m * warm

                if use_aux:
                    pred, state, (up, dn) = model(x, g, state,
                                                  return_aux=True)
                else:
                    pred, state = model(x, g, state)
                    up = dn = None

                l_mse = masked_loss(pred, y, m_eff, g, gait_w)
                loss  = l_mse
                mse_t += float(l_mse.detach())

                if args.deriv_lambda > 0:
                    l_der = windowed_deriv_loss(pred, y, m_eff,
                                                args.deriv_window)
                    loss = loss + args.deriv_lambda * l_der
                    der_t += float(l_der.detach())

                if up is not None:
                    l_cof  = masked_mean(up * dn, m_eff)
                    l_rate = masked_mean(up + dn, m_eff)
                    cof_t  += float(l_cof.detach())
                    rate_t += float(l_rate.detach())
                    if args.cofire_lambda > 0:
                        loss = loss + args.cofire_lambda * l_cof
                    if args.spike_lambda > 0:
                        loss = loss + args.spike_lambda * l_rate

                opt.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for b, plist in blocks.items():
                        gs = [p.grad.detach() for p in plist
                              if p.grad is not None]
                        if gs:
                            bacc[b] += torch.linalg.vector_norm(
                                torch.stack([torch.linalg.vector_norm(gr)
                                             for gr in gs]))
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                sched.step()          # per GRADIENT STEP, matching T_max
                tot += loss.item(); gtot += float(gnorm); nb += 1

            tr_loss  = tot / max(nb, 1)
            tr_gnorm = gtot / max(nb, 1)

            # ---- validate ---------------------------------------------
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
                    m = m * warm
                    vtot += masked_loss(pred, y, m, g).item(); vn += 1

                    post = torch.zeros_like(sw)
                    idx = sw.nonzero(as_tuple=False)
                    for t_i, b_i in idx:
                        post[t_i:min(t_i + args.settle, sw.shape[0]),
                             b_i] = 1.0
                    if post.sum() > 0:
                        vsw_tot += masked_loss(pred, y, m * post, g).item()
                        vsw_n   += 1

            va_loss = vtot / max(vn, 1)
            vsw     = vsw_tot / max(vsw_n, 1) if vsw_n else float("nan")
            epoch_s = time.perf_counter() - t_epoch

            hist["train"].append(tr_loss)
            hist["val"].append(va_loss)
            hist["val_sw"].append(vsw)
            hist["gnorm"].append(tr_gnorm)
            hist["sec"].append(epoch_s)
            hist["mse"].append(mse_t / max(nb, 1))
            hist["deriv"].append(der_t / max(nb, 1))
            hist["cofire"].append(cof_t / max(nb, 1))
            hist["rate"].append(rate_t / max(nb, 1))

            with torch.no_grad():
                ublk = {}
                for b, plist in blocks.items():
                    ublk[b] = float(torch.linalg.vector_norm(torch.stack([
                        torch.linalg.vector_norm(p.detach() - p0)
                        for p, p0 in zip(plist, theta0[b])])))
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
                "mse": hist["mse"][-1], "deriv": hist["deriv"][-1],
                "cofire": hist["cofire"][-1], "spk_per_step": hist["rate"][-1],
                "sec": epoch_s, "best": int(flag.strip() == "*"),
                **{f"grad_{b}": gblk[b] for b in sorted(blocks)},
                **{f"upd_{b}": ublk[b] for b in sorted(blocks)},
            })

            if epoch % args.diag_every == 0 or epoch == 1:
                gsum = sum(gblk.values()) or 1.0
                usum = sum(ublk.values()) or 1.0
                print(f"      {'block':<10}{'|grad|':>11}{'g%':>7}"
                      f"{'|upd|':>10}{'u%':>7}{'u/g':>8}{'upd/param':>11}")
                for b in sorted(blocks, key=lambda k: -ublk[k]):
                    gs, us = 100*gblk[b]/gsum, 100*ublk[b]/usum
                    npar = sum(p.numel() for p in blocks[b])
                    print(f"      {b:<10}{gblk[b]:>11.3g}{gs:>6.1f}%"
                          f"{ublk[b]:>10.4g}{us:>6.1f}%"
                          f"{(us/gs if gs > 1e-9 else float('inf')):>8.1f}"
                          f"{ublk[b]/math.sqrt(max(npar,1)):>11.2e}")
                if diag is not None:
                    lines, last_stats = diag()
                    for ln in lines:
                        print(ln)

    except KeyboardInterrupt:
        done = len(hist["train"])
        print()
        print("  " + "-" * 88)
        print(f"  [INTERRUPT] Ctrl+C received during epoch {done + 1}.")
        print(f"              {done} epoch(s) completed and recorded.")
        print(f"              Best val MSE so far : {best:.6f}")
        print( "              Stopping training and proceeding to export.")

    metrics.close()
    print("  " + "-" * 88)
    if hist["sec"]:
        tsum = sum(hist["sec"])
        print(f"  {len(hist['sec'])} epoch(s) in {tsum:.1f}s  "
              f"(mean {tsum / len(hist['sec']):.2f}s/epoch, train+val only)")
    return best, hist, last_stats


# ═══════════════════════════════════════════════════════════════════
# 6.  Delta-specific plots
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def plot_delta_detail(model, spikes, targets, valid, device, out_dir,
                      tgt_range, t0, gait_names, n_joints, period,
                      leg_cols=None, n_cycles=2.5, warm=600):
    """
    Every joint's staircase-vs-waveform, laid out one LEG PER ROW and one
    JOINT TYPE PER COLUMN, with the up/down spike raster under each panel.

    Same grid convention as visualize_timing.py's alignment charts: reading
    DOWN a column compares the same joint type across legs, which is where
    the interesting structure lives -- the legs of a gait are meant to be
    phase-shifted copies, so a column that is not a clean set of shifted
    copies means the network is treating one leg differently.  The previous
    version stacked an arbitrary first-four joints vertically, which made
    that comparison impossible.

    Why this exists at all alongside recon_*.png: at five cycles a
    sub-degree step is a pixel, and every failure mode of the delta
    representation is local -- steps too coarse for the flat sections, a
    neuron pinned at 1 spike/step on the fast part of the swing, up and down
    cancelling, or one delta scale sitting silent.
    """
    lo, hi = tgt_range
    scale, shift = (hi - lo) / 2.0, (hi + lo) / 2.0
    model.eval()
    if leg_cols is None:                      # fall back to a single column
        leg_cols = [[j] for j in range(n_joints)]
    n_legs  = len(leg_cols)
    C       = max(len(g) for g in leg_cols)
    n_steps = int(n_cycles * period)

    x = torch.as_tensor(spikes[t0 - warm:t0 + n_steps], dtype=torch.float32,
                        device=device).unsqueeze(1)
    for g in range(len(gait_names)):
        gg = torch.full((x.shape[0], 1), g, dtype=torch.long, device=device)
        pred, _, (up, dn) = model(x, gg, return_aux=True)
        pred = pred[warm:, 0].cpu().numpy() * scale + shift
        up_a = up[warm:, 0].cpu().numpy()      # (steps, n_scales, n_joints)
        dn_a = dn[warm:, 0].cpu().numpy()
        true = targets[g, t0:t0 + n_steps] * scale + shift
        n_sc = up_a.shape[1]

        hr = 1.0 + 0.4 * n_sc
        fig, axes = plt.subplots(
            2 * n_legs, C, squeeze=False, sharex=True,
            figsize=(1.0 + 5.0 * C, (1.9 + 0.42 * n_sc) * n_legs),
            gridspec_kw={"height_ratios": [3.0, hr] * n_legs})

        ups = ["#2a9d8f", "#1d3557", "#6a4c93"]
        dns = ["#f4a261", "#e63946", "#b5179e"]
        for L in range(n_legs):
            for k in range(C):
                a, r = axes[2 * L][k], axes[2 * L + 1][k]
                if k >= len(leg_cols[L]):
                    a.axis("off"); r.axis("off"); continue
                j = leg_cols[L][k]

                a.plot(true[:, j], color="#457b9d", lw=2.0, label="GT")
                a.step(np.arange(len(pred)), pred[:, j], where="post",
                       color="#e63946", lw=1.3, label="pred")
                a.grid(alpha=0.25); a.tick_params(labelsize=7)
                dtxt = "/".join(f"{float(model.delta()[m][j]) * scale:.3f}"
                                for m in range(n_sc))
                a.set_title(f"col{j}   delta {dtxt}°", fontsize=8)
                if k == 0:
                    a.set_ylabel(f"leg {L}\n(°)", fontsize=8)
                if L == 0 and k == C - 1:
                    a.legend(fontsize=6, loc="upper right")

                # Rasters grouped by DIRECTION then scale, so all the ups sit
                # together above all the downs: the eye is comparing up
                # against down (does the net increment have the right sign
                # right now?) far more often than coarse against fine.
                ylab = []
                for m in range(n_sc):
                    tag = ("c" if m == 0 else
                           ("f" if m == n_sc - 1 else str(m)))
                    for d, (arr, cols) in enumerate(((up_a, ups),
                                                     (dn_a, dns))):
                        row = (1 - d) * n_sc + (n_sc - 1 - m)
                        ts_ = np.where(arr[:, m, j] > 0)[0]
                        r.scatter(ts_, np.full_like(ts_, row, dtype=float),
                                  marker="|", s=60, lw=1.0,
                                  color=cols[m % len(cols)])
                        ylab.append((row, f"{tag}{'up' if d == 0 else 'dn'}"))
                ylab.sort()
                r.set_yticks([v for v, _ in ylab])
                r.set_yticklabels([t for _, t in ylab], fontsize=6)
                r.set_ylim(-0.6, 2 * n_sc - 0.4)
                r.grid(axis="x", alpha=0.2); r.tick_params(labelsize=7)

        # Common y-limits DOWN each column. The point of the grid is
        # comparing one joint type across legs, and that comparison is
        # misleading if each panel is autoscaled to its own range.
        for k in range(C):
            js = [leg_cols[L][k] for L in range(n_legs)
                  if k < len(leg_cols[L])]
            if not js:
                continue
            v = np.concatenate([true[:, js].ravel(), pred[:, js].ravel()])
            pad = 0.06 * (v.max() - v.min() + 1e-6)
            for L in range(n_legs):
                if k < len(leg_cols[L]):
                    axes[2 * L][k].set_ylim(v.min() - pad, v.max() + pad)

        for k in range(C):
            axes[-1][k].set_xlabel("timestep", fontsize=8)
        plt.suptitle(f"{gait_names[g]} — delta output detail "
                     f"({n_cycles:g} cycles, {warm}-step warm-up discarded)",
                     fontweight="bold")
        plt.tight_layout()
        p = out_dir / f"delta_detail_{gait_names[g]}.png"
        plt.savefig(p, dpi=140); plt.close()
        print(f"    [saved] {p}")


@torch.no_grad()
def plot_drift(model, spikes, targets, valid, device, out_dir, tgt_range,
               t0, gait_names, n_joints, period, n_cycles=40, warm=600):
    """
    Long free run, per-cycle mean error per joint.

    The accumulator is (by default) a pure integrator, so an up/down
    imbalance of even a fraction of a spike per cycle walks the joint away
    without bound.  Nothing in a 256-step training chunk sees that clearly,
    and `recon_*.png` at 1200 steps covers under five cycles.  This is the
    plot that catches it: a flat line is fine, a ramp is drift, and the
    slope is degrees per cycle.
    """
    lo, hi = tgt_range
    scale = (hi - lo) / 2.0
    model.eval()
    n_steps = int(min(n_cycles * period, len(spikes) - t0 - 1,
                      targets.shape[1] - t0 - 1))
    if n_steps < 4 * period:
        print("    [skip] drift plot: not enough held-out steps.")
        return {}

    x = torch.as_tensor(spikes[t0 - warm:t0 + n_steps], dtype=torch.float32,
                        device=device).unsqueeze(1)
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    out = {}
    per_cycle = int(round(period))
    ncyc = n_steps // per_cycle
    v = valid[t0:t0 + ncyc * per_cycle].astype(np.float32)

    for g in range(len(gait_names)):
        gg = torch.full((x.shape[0], 1), g, dtype=torch.long, device=device)
        pred, _ = model(x, gg)
        pred = pred[warm:warm + ncyc * per_cycle, 0].cpu().numpy()
        true = targets[g, t0:t0 + ncyc * per_cycle]
        err  = (pred - true) * scale * v[:, None]          # (steps, J)

        e = err.reshape(ncyc, per_cycle, n_joints)
        w = v.reshape(ncyc, per_cycle)
        denom = np.maximum(w.sum(axis=1), 1e-6)[:, None]
        mean_err = e.sum(axis=1) / denom                    # (ncyc, J)
        rms_err  = np.sqrt((e ** 2).sum(axis=1) / denom)

        c = CPG_PALETTE[g % len(CPG_PALETTE)]
        ax[0].plot(mean_err.mean(axis=1), lw=1.8, color=c,
                   label=gait_names[g])
        ax[1].plot(rms_err.mean(axis=1), lw=1.8, color=c,
                   label=gait_names[g])
        # Least-squares slope over the last 3/4 of the run: the first
        # cycles still carry the warm-up transient.
        k0 = ncyc // 4
        xs = np.arange(ncyc - k0, dtype=np.float64)
        slope = float(np.polyfit(xs, mean_err[k0:].mean(axis=1), 1)[0])
        out[gait_names[g]] = {
            "mean_err_deg_final": float(mean_err[-1].mean()),
            "rms_err_deg_final":  float(rms_err[-1].mean()),
            "drift_deg_per_cycle": slope,
        }

    ax[0].axhline(0, color="k", lw=0.8, alpha=0.5)
    ax[0].set_ylabel("per-cycle MEAN error (deg)\n= drift")
    ax[1].set_ylabel("per-cycle RMS error (deg)")
    ax[1].set_xlabel(f"cycle ({per_cycle} steps each)")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8, ncol=len(gait_names))
    plt.suptitle(f"Accumulator drift over {ncyc} cycles of free run "
                 f"(leak={model.leak:g})", fontweight="bold")
    plt.tight_layout()
    p = out_dir / "drift.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"    [saved] {p}")
    for k, v2 in out.items():
        print(f"      {k:>10}: drift {v2['drift_deg_per_cycle']:+.4f} °/cycle,"
              f" final mean err {v2['mean_err_deg_final']:+.2f}°,"
              f" RMS {v2['rms_err_deg_final']:.2f}°")
    return out


# ═══════════════════════════════════════════════════════════════════
# 7.  ONNX export
# ═══════════════════════════════════════════════════════════════════

def export_onnx(model, out_dir, device, cfg):
    """Single-timestep export.  Wrapped in try/except -- this file is
    exploratory and a tracing failure should not throw away the run's
    plots and metrics."""
    model.eval()
    compiled_step = None
    if getattr(model, "_step_is_compiled", False):
        # torch.onnx.export does not trace reliably through a compiled
        # callable, and the wrapper calls model.step().
        compiled_step = model.step
        model.step = model._step_eager
        print("    [onnx] using eager step() for export")
    try:
        wrapper = SingleStepONNXDelta(model).to(device).eval()
        dummy = (torch.zeros(1, model.n_neurons, device=device),
                 torch.zeros(1, dtype=torch.long, device=device),
                 *model.init_state(1, device))
        in_names  = ["spikes", "gait"] + list(model.state_names_in)
        out_names = ["angles", "up", "down"] + list(model.state_names_out)
        path = out_dir / "cpg_delta_snn_step.onnx"
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
    except Exception as e:                                   # noqa: BLE001
        print(f"    [onnx] export FAILED ({type(e).__name__}: {e})")
        print(f"    [onnx] continuing — weights are in best_model.pt")
    finally:
        if compiled_step is not None:
            model.step = compiled_step

    cfg_path = out_dir / "cpg_delta_snn_config.json"
    with open(cfg_path, "w") as f:
        json.dump(json_safe(cfg), f, indent=2, default=str)
    print(f"    [saved] config -> {cfg_path}")


# ═══════════════════════════════════════════════════════════════════
# 8.  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Bursting-LIF CPG -> LIF stack -> 2 delta neurons/joint")

    # ── CPG (identical to train.py) ────────────────────────────────
    ap.add_argument("--tmax",   type=int,   default=50_000)
    ap.add_argument("--warmup", type=int,   default=2_000)
    ap.add_argument("--i_app",  type=float, default=8.0)
    ap.add_argument("--n_cpg_neurons", type=int, default=4,
                    choices=sorted(CPG_W_BY_N))
    ap.add_argument("--floor_target", type=float, default=1.0,
                    help="Target delta-quantisation RMSE floor, in DEGREES, "
                         "for the worst joint. The CPG period is sized so "
                         "that delta_init_scale*vmax/sqrt(12) meets this. "
                         "Computed from the gait tables alone, so a gait that "
                         "already meets the target at a short period gets a "
                         "short period. Overridden by an explicit --vth_fb.")
    ap.add_argument("--vth_fb", type=float, default=None,
                    help="Feedback-neuron threshold. Default None = derived "
                         "from --floor_target. Set explicitly to pin the CPG "
                         "period and skip auto-sizing (e.g. 100 reproduces "
                         "the shipped 80-step period at N=4).")
    ap.add_argument("--to_fb_weight", type=float, default=10.0,
                    help="Charge the feedback neuron accumulates per main "
                         "spike. Spikes per burst is vth_fb/to_fb_weight, so "
                         "this is the quantum the period is rounded to.")
    ap.add_argument("--refrac_main", type=int, default=1,
                    help="Main-neuron refractory period, so the within-burst "
                         "ISI is refrac_main+1. Raising this lengthens the "
                         "period by THINNING the spikes rather than adding "
                         "them -- same period, sparser drive, hidden "
                         "membranes decay further between inputs. Prefer "
                         "letting --floor_target raise vth_fb instead.")
    ap.add_argument("--fake_cpg", type=int, default=1, choices=[0, 1],
                    help="1 (default) = idealised CPG with no pauses between "
                         "bursts. Requires the updated run_cpg in train.py "
                         "that accepts this parameter.")

    # ── gait tables (identical to train.py) ────────────────────────
    ap.add_argument("--gaits_dir", type=str, default="../gaits")
    ap.add_argument("--gaits", type=str, nargs="*", default=None)
    ap.add_argument("--leg_cols", type=str, default=None,
                    help="JSON list of equal-size column groups, one per "
                         "leg. Presentation only in this file (the plots "
                         "group by it); no architectural role, since the "
                         "hidden stack is dense.")

    # ── delta output representation ────────────────────────────────
    ap.add_argument("--delta_init_scale", type=float, default=2.0,
                    help="delta_j is initialised at this multiple of the "
                         "joint's PEAK REQUIRED VELOCITY (units/step, "
                         "measured from the gait tables). 1.0 is the hard "
                         "feasibility floor -- the neuron would have to fire "
                         "every single step at the fastest part of the swing "
                         "-- so the default 2.0 means 50%% peak duty. Raising "
                         "it buys headroom and costs resolution: the RMSE "
                         "floor is ~delta/sqrt(12). Since delta is learnable "
                         "this is only a starting point, but a bad one is "
                         "expensive: too small saturates the neuron (gradient "
                         "goes flat) and too large makes every step visible.")
    ap.add_argument("--proprio", type=int, default=1, choices=[0, 1],
                    help="Feed each joint's own commanded angle back as an "
                         "input channel (closed loop). 1 = on. The value fed "
                         "back is the network's COMMANDED angle from its own "
                         "accumulator, not a measured encoder reading, so "
                         "there is nothing to wire on the robot and no "
                         "train/deploy mismatch. Detached, so no gradient "
                         "flows around the loop. --proprio 0 gives the "
                         "original open-loop integrator for A/B.")
    ap.add_argument("--delta_scales", type=int, default=2,
                    help="Number of delta magnitudes per joint (slow-twitch / "
                         "fast-twitch). Each scale gets its OWN up and down "
                         "neuron, so the output layer is 2*delta_scales*"
                         "n_joints wide. 2 (default) gives a coarse pair "
                         "sized by peak velocity and a fine pair sized by "
                         "resolution. Why this exists: one delta has to be "
                         ">= vmax to keep up AND small enough not to ripple, "
                         "and when those conflict the only single-scale fix "
                         "is a longer period -- which bptt and tau_max both "
                         "track, so it gets expensive fast. A ladder buys "
                         "resolution^(scales-1) without buying period. "
                         "--delta_scales 1 reproduces the original 2-neuron "
                         "architecture exactly (and old 1-scale checkpoints "
                         "only load with it).")
    ap.add_argument("--delta_ratio", type=float, default=4.0,
                    help="Each successive scale is 1/ratio of the previous, "
                         "so at the default the fine delta is a quarter of "
                         "the coarse one. This sets the INITIALISATION only: "
                         "the scales are independently learnable, so the "
                         "ratio can drift (delta_report prints the learned "
                         "value and warns if the ladder collapses). The "
                         "required period falls by ratio^(scales-1), so this "
                         "is the main lever on --floor_target vs period.")
    ap.add_argument("--freeze_delta", action="store_true",
                    help="Hold delta at its initialisation. Useful for "
                         "attributing an RMSE change to the network rather "
                         "than to delta having grown.")
    ap.add_argument("--pose", type=str, default="learnable",
                    choices=["fixed", "learnable"],
                    help="Standing pose, i.e. what the accumulator counts "
                         "from. Both modes INITIALISE from row 0 of gait "
                         "--pose_gait. 'learnable' additionally makes it a "
                         "trained per-joint offset: it enters the output at "
                         "every timestep (angle = pose + acc), so unlike the "
                         "membrane states it receives gradient on every step "
                         "and can absorb a systematic DC error that the "
                         "spikes would otherwise have to correct. 'fixed' is "
                         "the more debuggable choice -- any DC error is then "
                         "unambiguously the accumulator's.")
    ap.add_argument("--pose_gait", type=int, default=0,
                    help="Which gait's row 0 to use as the standing pose. "
                         "Note that with a SHARED pose and gaits whose row 0 "
                         "differ, the network must spike its way from this "
                         "pose to the other gaits' phase-0 angles. A per-gait "
                         "pose (Embedding(max_gaits, n_joints)) is the "
                         "obvious extension if that turns out to matter.")
    ap.add_argument("--angle_leak", type=float, default=0.0,
                    help="Per-step decay of the accumulator toward pose: "
                         "acc <- (1-leak)*acc + delta*(up-dn). 0 = pure "
                         "integrator (the honest version of the idea). A "
                         "small value bounds drift at the cost of a "
                         "low-frequency high-pass on the output: leak=1e-3 is "
                         "a ~1000-step (≈4 cycle) decay. Reach for this only "
                         "if drift.png shows a ramp that training will not "
                         "flatten. Not learnable on purpose -- it would trade "
                         "waveform fidelity for drift suppression and win on "
                         "short training chunks while losing on the long "
                         "free run.")
    ap.add_argument("--out_reset", type=str, default="subtract",
                    choices=["subtract", "zero"],
                    help="Output-neuron membrane reset. 'subtract' keeps the "
                         "residual, which is what makes firing RATE graded in "
                         "the drive -- and rate is the code here.")
    ap.add_argument("--no_out_gait_bias", action="store_true",
                    help="Drop the per-gait output bias (zero-init "
                         "Embedding(max_gaits, 2*n_joints)). It is cheap and "
                         "lets a gait set per-neuron tonic drive, i.e. a "
                         "baseline increment rate; zero-init keeps it inert "
                         "at init so it cannot disturb calibration.")
    ap.add_argument("--tau_out_min", type=float, default=2.0)
    ap.add_argument("--tau_out_max", type=float, default=20.0,
                    help="Tau init ceiling for the OUTPUT membranes. Short on "
                         "purpose: an output neuron's job is to convert the "
                         "hidden layer's instantaneous drive into a rate, and "
                         "a long tau would average that drive over a large "
                         "fraction of the cycle -- smearing exactly the "
                         "timing distinction between joints inside one leg "
                         "that these neurons exist to represent.")
    ap.add_argument("--out_mem_clip", type=float, default=0.0,
                    help="Clip the output membrane to +/- this many "
                         "thresholds after reset. <=0 (the default) disables "
                         "it, giving textbook subtract-reset. Turn it on only "
                         "if delta_detail_*.png shows a neuron stuck firing "
                         "after its drive stops: with subtract reset an "
                         "over-driven membrane integrates without bound, and "
                         "measured at tau=20 a 10x-threshold drive settles at "
                         "mem~185 and then keeps firing for 46 steps after the "
                         "drive stops. Clipping at 2.0 cuts that to 1 step, "
                         "but it is not free -- clamp has zero gradient where "
                         "it is active, so a saturated neuron goes invisible "
                         "to backprop at exactly the moments it misbehaves.")
    ap.add_argument("--out_slope", type=float, default=5.0,
                    help="Surrogate slope for the OUTPUT layer only "
                         "(--slope still applies to the hidden stack). The "
                         "surrogate derivative is 1/(slope*|x|+1)^2, so at 25 "
                         "a neuron a few units from threshold is nearly "
                         "invisible to gradients. This is the layer the whole "
                         "task gradient has to pass through, so it gets the "
                         "gentler surrogate.")

    # ── loss terms ────────────────────────────────────────────────
    ap.add_argument("--deriv_lambda", type=float, default=0.0,
                    help="Weight on the windowed-velocity term: "
                         "(pred[t]-pred[t-w]) vs (targ[t]-targ[t-w]). This is "
                         "the FIRST thing to try if plain MSE stalls -- it "
                         "speaks directly to what the output spikes control "
                         "(the net spike count in a window) and is blind to "
                         "accumulated DC offset, which the plain MSE already "
                         "covers. Off by default so run 1 measures the plain "
                         "objective honestly. Try 0.3-1.0.")
    ap.add_argument("--deriv_window", type=int, default=None,
                    help="Window w for --deriv_lambda, in steps. Default "
                         "None = max(4, round(period/16)). Must be at least "
                         "the target's row dwell time (period/target_rows) or "
                         "the term regresses onto the gait table's row "
                         "quantisation instead of onto real velocity.")
    ap.add_argument("--cofire_lambda", type=float, default=0.0,
                    help="Penalty on up and down firing in the same timestep "
                         "(they cancel to zero net increment). Wasteful "
                         "rather than wrong, so off by default; the "
                         "'cofire' column in the diagnostic report says "
                         "whether it is worth turning on.")
    ap.add_argument("--spike_lambda", type=float, default=0.0,
                    help="L1 on total output spikes per step. Energy matters "
                         "on neuromorphic hardware, but note the budget: "
                         "these neurons NEED roughly travel/(2*delta) spikes "
                         "per cycle to reach the joint's extremes, which is "
                         "several times the CPG's rate. Applying this before "
                         "the network can track the waveform will just starve "
                         "it. Tune after a working baseline, not before.")

    # ── architecture ──────────────────────────────────────────────
    ap.add_argument("--hidden",   type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=3,
                    help="Number of dense LIF hidden layers between the CPG "
                         "and the output neurons.")
    ap.add_argument("--max_gaits", type=int, default=16)
    ap.add_argument("--tau_min",  type=float, default=2.0)
    ap.add_argument("--tau_max",  type=float, default=None,
                    help="Longest HIDDEN membrane tau. Default None = the "
                         "measured CPG period rounded up to a multiple of 64. "
                         "The hidden stack is the only long-timescale memory "
                         "in the model, so this must exceed one period or the "
                         "network cannot know where in the cycle it is.")
    ap.add_argument("--slope",    type=float, default=25.0)

    # ── training ──────────────────────────────────────────────────
    ap.add_argument("--epochs",           type=int,   default=100)
    ap.add_argument("--chunks_per_epoch", type=int,   default=40)
    ap.add_argument("--val_chunks",       type=int,   default=2)
    ap.add_argument("--bptt",             type=int,   default=None,
                    help="Gradient truncation horizon; default None = period "
                         "rounded up to a multiple of 64. Note this has an "
                         "extra effect here that it does not have in train.py: "
                         "the accumulator makes one output spike affect every "
                         "later step in the chunk, so the output block's "
                         "gradient magnitude scales with bptt.")
    ap.add_argument("--batch",            type=int,   default=128)
    ap.add_argument("--lr",               type=float, default=2e-3,
                    help="Lower than train.py's 4e-3 default: the gradient "
                         "reaching the output layer is summed over the chunk "
                         "by the accumulator, so the same LR is effectively "
                         "hotter here. Sweep it against the |grad| column.")
    ap.add_argument("--clip",             type=float, default=1.0)
    ap.add_argument("--switch_min",       type=int,   default=600)
    ap.add_argument("--switch_max",       type=int,   default=3000)
    ap.add_argument("--settle",           type=int,   default=100)
    ap.add_argument("--val_frac",         type=float, default=0.15)
    ap.add_argument("--phase_zero",       type=float, default=0.0)

    # ── calibration ───────────────────────────────────────────────
    ap.add_argument("--no_calibrate", action="store_true",
                    help="Skip output-bias calibration. Expect silent or "
                         "saturated output neurons at init; see "
                         "calibrate_out_bias's docstring for why that is "
                         "hard to recover from.")
    ap.add_argument("--rate_target_scale", type=float, default=1.0,
                    help="Multiplier on the measured per-neuron spike budget "
                         "used as the calibration target.")

    # ── misc ──────────────────────────────────────────────────────
    ap.add_argument("--freeze_blocks", type=str, default="",
                    help="Comma-separated grad_blocks_delta names to freeze, "
                         "e.g. 'hidden,film'.")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--log_every",  type=int, default=1)
    ap.add_argument("--diag_every", type=int, default=10,
                    help="Epoch cadence for the per-block gradient table and "
                         "the output-layer report.")
    ap.add_argument("--log_every_aux", action="store_true",
                    help="Always compute up/down spikes during training so "
                         "the cofire and spike-rate columns are populated "
                         "even when their lambdas are 0. Small extra memory "
                         "(two (L,B,J) tensors retained per chunk).")
    ap.add_argument("--dry_run",    action="store_true",
                    help="Build data, print the feasibility table, run "
                         "calibration and the init-time diagnostics, skip "
                         "training.")
    ap.add_argument("--out_dir",    type=str, default="deltas",
                    help="Resolved as outputs/<out_dir>. Defaults to "
                         "'deltas' rather than '' so a run here cannot "
                         "overwrite train.py's artifacts.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = outputs_path(this_file_dir, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device : {device}\nOutput : {out_dir.resolve()}")
    print(f"Arch   : delta ({args.n_layers} x {args.hidden} LIF -> "
          f"2 spiking neurons per joint -> accumulator)\n")

    # ── 0. Gait tables + leg layout ───────────────────────────────
    print("[0/9] Gait tables ...")
    gaits_dir = Path(this_file_dir + "/" + args.gaits_dir)
    if args.gaits is not None:
        gait_files = list(args.gaits)
        print(f"      --gaits override: {gait_files}")
    else:
        gait_files = GAIT_FILES_BY_N.get(args.n_cpg_neurons)
        if gait_files is None:
            raise ValueError(
                f"No default gait file list for n_cpg_neurons="
                f"{args.n_cpg_neurons} (have {sorted(GAIT_FILES_BY_N)}). "
                f"Pass --gaits explicitly.")
        species = {4: "quadruped", 6: "hexapod"}.get(args.n_cpg_neurons, "?")
        print(f"      n_cpg_neurons={args.n_cpg_neurons} -> {species} "
              f"gait set: {gait_files}")
    gait_tables_orig, gait_names = load_gait_tables(gait_files, gaits_dir)
    n_joints = gait_tables_orig[0].shape[1]
    for nm, gt in zip(gait_names, gait_tables_orig):
        print(f"      {nm:>18s} : {gt.shape[0]} rows x {gt.shape[1]} joints")

    if args.leg_cols is not None:
        leg_cols = [list(c) for c in json.loads(args.leg_cols)]
        flat = sorted(c for grp in leg_cols for c in grp)
        if flat != list(range(n_joints)):
            raise ValueError(f"--leg_cols {leg_cols} is not a partition of "
                             f"0..{n_joints - 1}.")
        n_legs, layout_src = len(leg_cols), "user"
    else:
        n_legs, leg_cols = default_leg_layout(args.n_cpg_neurons, n_joints)
        layout_src = "default"
    print(f"      gait layout: n_legs={n_legs}  n_joints={n_joints}  "
          f"source={layout_src}")
    print(f"      output neurons: {2 * args.delta_scales * n_joints} "
          f"({args.delta_scales} scale(s) x 2 directions x {n_joints} joints)")
    if not (0 <= args.pose_gait < len(gait_names)):
        raise ValueError(f"--pose_gait {args.pose_gait} out of range for "
                         f"{len(gait_names)} gait(s).")

    # ── 1. Gait-table upsampling (moved BEFORE the CPG) ───────────
    # target_rows and the max row-to-row angle change depend only on the
    # gait tables, so the period the delta representation needs is knowable
    # before the oscillator exists. Doing this first is what lets the CPG be
    # sized to the gait rather than the gait be quantised to the CPG.
    print("\n[1/9] Upsampling gait tables ...")
    gait_tables, target_rows = upsample_gait_tables(gait_tables_orig,
                                                   gait_names)

    # ── 2. Size the CPG to the required RMSE floor ────────────────
    print("\n[2/9] Sizing the CPG period to the delta RMSE floor ...")
    req = required_period_for_floor(gait_tables, args.floor_target,
                                    args.delta_init_scale,
                                    n_scales=args.delta_scales,
                                    delta_ratio=args.delta_ratio)
    print(f"      {req['rows']} rows/cycle, worst adjacent-row change "
          f"{req['maxrowdiff_deg']:.3f}°  -> peak velocity "
          f"{req['vmax_per_cycle_deg']:.1f}°/cycle")
    print(f"      {args.delta_scales} delta scale(s), ratio "
          f"{args.delta_ratio:g}  -> finest delta is 1/"
          f"{req['fine_factor']:g} of the coarse one")
    print(f"      floor = coarse_delta / ({req['fine_factor']:g} x sqrt(12)) "
          f"<= {args.floor_target:g}°  requires period >= "
          f"{req['period_required']:.0f} steps")
    if args.delta_scales > 1:
        solo = req["period_required"] * req["fine_factor"]
        print(f"      (a single delta scale would need period "
              f"{solo:.0f} -- hence the ladder)")

    if args.vth_fb is not None:
        spb = max(1, int(args.vth_fb // args.to_fb_weight))
        meas, pred = probe_period(args.n_cpg_neurons, args.i_app,
                                  args.vth_fb, args.to_fb_weight,
                                  args.refrac_main, bool(args.fake_cpg))
        cpgp = {"vth_fb": float(args.vth_fb),
                "to_fb_weight": float(args.to_fb_weight),
                "refrac_main": int(args.refrac_main),
                "spikes_per_burst": spb,
                "period_predicted": float(pred),
                "period_measured": float(meas),
                "period_target": float(req["period_required"]),
                "probes": {spb: float(meas)}}
        print(f"      --vth_fb {args.vth_fb:g} given explicitly; auto-sizing "
              f"skipped (measured period {meas:.1f})")
    else:
        cpgp = size_cpg(req["period_required"], args.n_cpg_neurons,
                        args.i_app, args.to_fb_weight, args.refrac_main,
                        bool(args.fake_cpg))

    print(f"      chosen: vth_fb={cpgp['vth_fb']:g}  "
          f"spikes/burst={cpgp['spikes_per_burst']}  "
          f"refrac_main={cpgp['refrac_main']}  "
          f"-> period {cpgp['period_measured']:.1f} steps")
    implied = (args.delta_init_scale * req["vmax_per_cycle_deg"]
               / (req["fine_factor"] * cpgp["period_measured"]
                  * math.sqrt(12.0)))
    print(f"      implied RMSE floor at that period: {implied:.3f}° "
          f"(target {args.floor_target:g}°)")
    if args.tmax < 40 * cpgp["period_measured"]:
        print(f"      NOTE --tmax {args.tmax} is only "
              f"{args.tmax / cpgp['period_measured']:.0f} gait cycles at this "
              f"period. Raising the period costs training data unless tmax "
              f"rises with it; consider --tmax "
              f"{int(200 * cpgp['period_measured']):d}.")

    # ── 3. CPG ────────────────────────────────────────────────────
    print("\n[3/9] Bursting-LIF CPG ...")
    spikes = run_cpg_sized(N=args.n_cpg_neurons, tmax=args.tmax,
                           warmup=args.warmup, i_app=args.i_app,
                           vth_fb=cpgp["vth_fb"],
                           to_fb_weight=cpgp["to_fb_weight"],
                           refrac_main=cpgp["refrac_main"],
                           fake_cpg=bool(args.fake_cpg))

    print("\n[4/9] Burst structure & phase ...")
    onsets, period, neuron_offsets, burst_thresholds = analyse_cpg(spikes,
                                                                   out_dir)
    plot_cpg_raster(spikes, onsets, out_dir)
    phase = cycle_phase(len(spikes), onsets[0])

    # The sizing probe and analyse_cpg use the same burst-detection code on
    # the same generator, so a disagreement here means the period is not
    # stable over the full tmax -- and a mis-detected period silently
    # corrupts delta, tau_max, bptt and the target phase all at once.
    if (math.isfinite(cpgp["period_measured"]) and
            abs(period - cpgp["period_measured"])
            > 0.05 * cpgp["period_measured"]):
        print(f"      WARNING: period over the full run ({period:.1f}) "
              f"differs from the sizing probe ({cpgp['period_measured']:.1f}) "
              f"by more than 5%. Check spk/burst above: if it is near 1, or "
              f"the per-neuron ISI is not bimodal, burst detection has "
              f"failed and every downstream number is wrong.")

    round64 = lambda v: int(64 * math.ceil(v / 64.0))
    if args.tau_max is None:
        args.tau_max = float(round64(period))
        print(f"      tau_max      : {args.tau_max:6.1f}  (hidden stack; "
              f"period {period:.0f} rounded up to a multiple of 64)")
    if args.bptt is None:
        args.bptt = round64(period)
        print(f"      bptt         : {args.bptt:6d}  (~one cycle)")
    cpg_rate, cpg_R = cpg_spike_stats(spikes, phase, period)
    print(f"      CPG stats    : {cpg_rate:.2f} spikes/cycle per neuron, "
          f"phase concentration R={cpg_R:.3f}")

    # ── 5. Targets ────────────────────────────────────────────────
    print("\n[5/9] Building targets ...")
    targets, valid, tgt_range = build_targets(phase, gait_tables,
                                              phase_zero=args.phase_zero)
    print(f"      targets {targets.shape}   valid coverage "
          f"{valid.mean()*100:.2f}%   range [{tgt_range[0]:.1f}, "
          f"{tgt_range[1]:.1f}] deg")

    if args.deriv_window is None:
        args.deriv_window = max(4, int(round(period / 16.0)))
    dwell = period / float(target_rows)
    print(f"      deriv_window : {args.deriv_window} steps "
          f"(target row dwell time is {dwell:.1f} steps)")
    if dwell < 2.0:
        print(f"      WARNING: each gait-table row lasts only {dwell:.2f} "
              f"timesteps, so the target is being played at ~one row per "
              f"step. The waveform has no temporal headroom for the "
              f"accumulator to track it; raise the period.")
    if args.deriv_lambda > 0 and args.deriv_window < dwell:
        print(f"      WARNING: --deriv_window {args.deriv_window} is shorter "
              f"than the target's row dwell time ({dwell:.1f} steps), so the "
              f"velocity term is partly regressing onto the gait table's row "
              f"quantisation rather than onto real motion.")

    # ── 6. Delta sizing + feasibility ─────────────────────────────
    print("\n[6/9] Delta sizing & feasibility ...")
    kin = joint_kinematics(gait_tables, period, tgt_range)
    # Coarse delta is sized by the VELOCITY requirement (smallest that can
    # still reach vmax at the chosen duty); the finer scales are that
    # divided down by --delta_ratio, and they are what set the RMSE floor.
    delta_coarse = args.delta_init_scale * kin["vmax"]
    delta_init   = delta_scale_ladder(delta_coarse, args.delta_scales,
                                      args.delta_ratio)
    print(f"      target rows {kin['R']}  ->  {kin['steps_per_row']:.2f} "
          f"timesteps per gait-table row")
    print(f"      delta[0] = {args.delta_init_scale:g} x peak velocity, "
          f"each further scale / {args.delta_ratio:g}")
    kin_report = report_kinematics(kin, delta_init, period, n_joints)

    pose_init = ((gait_tables[args.pose_gait][0] - tgt_range[0])
                 / (tgt_range[1] - tgt_range[0] + 1e-8) * 2.0 - 1.0)
    print(f"      pose ({args.pose}) from {gait_names[args.pose_gait]} row 0: "
          f"[" + " ".join(f"{v * kin['scale'] + (tgt_range[1]+tgt_range[0])/2:.1f}"
                          for v in pose_init) + "] deg")

    prop = proprio_normalisation(gait_tables, tgt_range)
    if args.proprio:
        print(f"      proprioception ON: {n_joints} extra input channel(s), "
              f"each joint's commanded angle scaled to +/-1 over its own "
              f"range")
        print(f"        per-joint span (deg): " +
              " ".join(f"{v:.1f}" for v in (prop['hi_deg'] - prop['lo_deg'])))
        if prop["degenerate"]:
            print(f"        WARNING: joint(s) {prop['degenerate']} never move "
                  f"across any gait; their channels are constant.")
    else:
        print(f"      proprioception OFF (--proprio 0): open-loop integrator, "
              f"nothing observes the accumulator")

    # ── 5. Samplers ───────────────────────────────────────────────
    print("\n[7/9] Stream samplers ...")
    T       = len(spikes)
    t_lo    = int(onsets[0][2])
    t_split = int(T * (1.0 - args.val_frac))
    t_hi    = int(onsets[0][-2])
    print(f"      train steps [{t_lo}, {t_split})   "
          f"val steps [{t_split}, {t_hi})")
    if t_split - t_lo < 4 * args.bptt or t_hi - t_split < 2 * args.bptt:
        raise ValueError("Not enough timesteps — raise --tmax or lower --bptt.")

    warm_steps = int(round(period))
    print(f"      post-reset warm-up: {warm_steps} steps excluded from the "
          f"loss (a reset head sits at pose with an empty accumulator)")
    tr_sampler = StreamSampler(spikes, targets, valid, t_lo, t_split,
                               args.batch, args.switch_min, args.switch_max,
                               rng, n_gaits=len(gait_tables), device=device,
                               phase=phase, warm_steps=warm_steps)
    va_sampler = StreamSampler(spikes, targets, valid, t_split, t_hi,
                               args.batch, args.switch_min, args.switch_max,
                               np.random.default_rng(args.seed + 1),
                               n_gaits=len(gait_tables), device=device,
                               phase=phase, warm_steps=warm_steps)

    # ── 6. Model ──────────────────────────────────────────────────
    print("\n[8/9] Model ...")
    model = DeltaSNN(
        hidden=args.hidden, n_layers=args.n_layers,
        n_gaits=len(gait_tables), max_gaits=args.max_gaits,
        n_neurons=args.n_cpg_neurons, n_joints=n_joints,
        n_scales=args.delta_scales,
        proprio=bool(args.proprio),
        prop_center=prop["centre"], prop_halfrange=prop["halfrange"],
        tau_min=args.tau_min, tau_max=args.tau_max,
        tau_out_min=args.tau_out_min, tau_out_max=args.tau_out_max,
        slope=args.slope, out_slope=args.out_slope,
        delta_init=delta_init, pose_init=pose_init,
        pose_learnable=(args.pose == "learnable"),
        out_reset=args.out_reset,
        out_gait_bias=not args.no_out_gait_bias,
        angle_leak=args.angle_leak,
        out_mem_clip=args.out_mem_clip).to(device)

    if args.freeze_delta:
        model.log_delta.requires_grad_(False)
        print(f"      delta FROZEN at init")
    if args.freeze_blocks:
        want = {b.strip() for b in args.freeze_blocks.split(",") if b.strip()}
        all_blocks = grad_blocks_delta(model)
        unknown = want - set(all_blocks)
        if unknown:
            raise ValueError(f"--freeze_blocks: unknown {sorted(unknown)}; "
                             f"available {sorted(all_blocks)}")
        n_frozen = 0
        for b in want:
            for p in all_blocks[b]:
                p.requires_grad_(False)
                n_frozen += p.numel()
        print(f"      FROZEN {sorted(want)}: {n_frozen:,} params")

    n_par = sum(p.numel() for p in model.parameters())
    print(f"      params={n_par:,}   hidden={args.hidden} x {args.n_layers}"
          f"   out_reset={args.out_reset}   leak={args.angle_leak:g}")
    for k, v in model.param_breakdown().items():
        print(f"        {k:<8s}: {v:>9,}  ({100.0 * v / n_par:4.1f}%)")
    if args.tau_max < period:
        print(f"      WARNING: tau_max={args.tau_max:.0f} < period "
              f"{period:.0f}. The hidden membranes are the only "
              f"long-timescale memory — the network cannot hold phase.")

    # Compile the single timestep, not forward(): forward loops over
    # --bptt steps in Python and compiling that would trace-unroll the
    # whole loop into one enormous graph.  Every other path in this file
    # (calibration, the diagnostic report, the plots) goes through
    # `eager_step`, so only the training batch shape is ever traced.
    model._step_eager = model.step
    model._step_is_compiled = False
    if device.type == "cuda":
        model.step = torch.compile(model.step, dynamic=False)
        model._step_is_compiled = True
        print("      torch.compile: step() compiled (dynamic=False)")
    else:
        print(f"      torch.compile: SKIPPED (device={device.type})")

    gait_w = make_gait_weights(gait_tables_orig, gait_names, device)

    # ── Output-bias calibration ───────────────────────────────────
    # Before the optimiser is built: it writes b_out in place.
    calib = {}
    if not args.no_calibrate:
        print("\n      Calibrating output-neuron biases to the spike budget ...")
        # budget is travel/(2*sum(delta)) per joint, which is the SAME for
        # every scale once travel is split in proportion to each scale's
        # delta -- so every output neuron gets the same per-joint target and
        # no scale starts starved relative to another. Tiled in the model's
        # flat order: (scale*2 + direction)*J + joint.
        budget = np.asarray(kin_report["spikes_per_cycle_per_direction"],
                            dtype=np.float64) * args.rate_target_scale
        target = np.tile(budget, 2 * args.delta_scales)
        calib = calibrate_out_bias(model, spikes, len(gait_tables), device,
                                   period, target)
    else:
        print("\n      --no_calibrate: output biases left at zero.")

    # ── Diagnostic hook ───────────────────────────────────────────
    t_diag = max(t_split, t_lo) + 200
    diag = lambda: delta_report(
        model, spikes, targets, valid, period, len(gait_tables), device,
        tgt_range, gait_names, t0=t_diag, n_cycles=8, indent="      ")

    best     = float("nan")
    hist     = {"train": [], "val": [], "val_sw": [], "gnorm": [], "sec": [],
                "mse": [], "deriv": [], "cofire": [], "rate": [], "upd": []}
    final_lr = float(args.lr)
    stats    = []

    if not args.dry_run:
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr)
        total_steps = args.epochs * args.chunks_per_epoch
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=1e-5)
        best, hist, stats = run_training(
            model, tr_sampler, va_sampler, opt, sched, device, args,
            gait_w, out_dir, diag=diag, n_gaits=len(gait_tables),
            period=period)
        final_lr = float(opt.param_groups[0]["lr"])
        model.load_state_dict(torch.load(out_dir / "best_model.pt",
                                         map_location=device))
        print(f"\n  best val MSE : {best:.6f}")
        plot_training_curves(hist, out_dir)
        print("\n  Output layer at best checkpoint:")
        lines, stats = diag()
        for ln in lines:
            print(ln)
    else:
        print("      --dry_run: skipping training.")
        print("      Output layer at initialisation:")
        lines, stats = diag()
        for ln in lines:
            print(ln)

    # ── 7. Eval + export ──────────────────────────────────────────
    print("\n[9/9] Evaluation & export ...")
    t_eval = max(t_split + 800, t_lo + 800)
    # All of these free-run at batch 1, a shape the training loop never
    # uses; eager keeps them from each costing a Dynamo graph.
    with eager_step(model):
        # 5 cycles, not train.py's fixed 1200 steps: the period here is
        # sized to the gait rather than inherited, so a step count that
        # framed ~5 cycles at period 254 frames 15+ at period 80 and the
        # waveform becomes unreadable.
        recon_steps = int(round(5 * period))
        rmse = plot_reconstruction(model, spikes, targets, valid, device,
                                   out_dir, tgt_range, t0=t_eval,
                                   gait_names=gait_names, leg_cols=leg_cols,
                                   n_joints=n_joints, n_steps=recon_steps)
        # One switch, with ~2 cycles either side of it.
        plot_transition(model, spikes, targets, device, out_dir, tgt_range,
                        t0=t_eval, gait_names=gait_names, leg_cols=leg_cols,
                        g_from=0, g_to=1,
                        n_steps=int(round(4 * period)),
                        switch_at=int(round(2 * period)))
        plot_delta_detail(model, spikes, targets, valid, device, out_dir,
                          tgt_range, t_eval, gait_names, n_joints, period,
                          leg_cols=leg_cols)
        drift = plot_drift(model, spikes, targets, valid, device, out_dir,
                           tgt_range, t_eval, gait_names, n_joints, period)

    # RMSE against the floor delta implies -- the comparison that makes
    # the number interpretable.
    floor = np.asarray(kin_report["rmse_floor_deg"])
    print(f"\n      mean RMSE {rmse.mean():.2f}°  vs delta-implied floor "
          f"{floor.mean():.2f}°  (ratio {rmse.mean()/max(floor.mean(),1e-9):.1f}x)")
    print(f"      A ratio near 1 means the run is DELTA-LIMITED: lower "
          f"--delta_init_scale rather than training longer.")

    epochs_done = len(hist["train"])
    d_final = model.delta().detach().cpu().numpy()

    cfg = {
        "model":          "cpg_lif_delta_accumulator",
        "arch":           "delta",
        "config_version": 1,
        "created_utc":    datetime.now(timezone.utc).isoformat(
                              timespec="seconds"),

        # ── deployment-critical, top level ────────────────────────
        "hidden":         int(args.hidden),
        "n_layers":       int(args.n_layers),
        "max_gaits":      int(args.max_gaits),
        "n_gaits":        len(gait_tables),
        "n_legs":         int(n_legs),
        "n_joints":       int(n_joints),
        "n_out_neurons":  int(2 * args.delta_scales * n_joints),
        "n_delta_scales": int(args.delta_scales),
        "n_cpg_neurons":  int(args.n_cpg_neurons),
        # Proprioception needs NO new ONNX input: the angle is recomputed
        # inside step() from the acc state the graph already carries, so the
        # robot runs the identical single-step graph and feeds nothing extra.
        "proprioception": {
            "enabled":   bool(args.proprio),
            "source":    "commanded angle from the network's own accumulator "
                         "(pose + acc), one step stale, detached",
            "n_channels": int(n_joints) if args.proprio else 0,
            "formula":   "p_j = (pose_j + acc_j - centre_j) / halfrange_j, "
                         "never clamped",
            "centre_normalised":    prop["centre"].tolist(),
            "halfrange_normalised": prop["halfrange"].tolist(),
            "per_joint_lo_deg":     prop["lo_deg"].tolist(),
            "per_joint_hi_deg":     prop["hi_deg"].tolist(),
        },
        "gait_names":     gait_names,
        "gait_files":     gait_files,
        "gaits_dir":      str(gaits_dir.resolve()),
        "leg_cols":       [list(c) for c in leg_cols],
        "leg_layout_source": layout_src,
        "global_min":     float(tgt_range[0]),
        "global_max":     float(tgt_range[1]),
        "target_rows":    int(target_rows),
        "phase_zero":     float(args.phase_zero),
        "cpg_period_steps": float(period),

        # ── the delta representation: everything the robot needs to
        #    reconstruct an angle from a spike train ────────────────
        # Both normalised and degree forms are written: the ONNX graph
        # works in normalised units, a servo command is in degrees, and
        # having to re-derive the conversion on the Pi is how sign and
        # scale errors happen.
        "delta": {
            "normalised":   d_final.tolist(),
            "degrees":      (d_final * kin["scale"]).tolist(),
            "learnable":    not args.freeze_delta,
            "init_scale":   float(args.delta_init_scale),
            "init_normalised": delta_init.tolist(),
            "n_scales":     int(args.delta_scales),
            "ratio_init":   float(args.delta_ratio),
            "shape":        "(n_scales, n_joints), coarse scale first",
            "total_normalised":  d_final.sum(axis=0).tolist(),
            "finest_normalised": d_final[-1].tolist(),
        },
        "pose": {
            "normalised": model.pose.detach().cpu().numpy().tolist(),
            "degrees":    (model.pose.detach().cpu().numpy() * kin["scale"]
                           + (tgt_range[1] + tgt_range[0]) / 2.0).tolist(),
            "mode":       args.pose,
            "source_gait": gait_names[args.pose_gait],
        },
        "accumulator": {
            "leak":     float(args.angle_leak),
            "formula":  "acc <- (1-leak)*acc + sum_over_scales("
                        "delta[s]*(up[s]-down[s])); "
                        "angle_normalised = pose + acc; "
                        "angle_deg = angle_normalised*(max-min)/2 + (max+min)/2",
            # flat output index = (scale*2 + direction)*n_joints + joint
            "flat_index_formula": "(scale*2 + direction)*n_joints + joint, "
                                  "direction 0=up 1=down, scale 0=coarsest",
            "up_index":   [[(k * 2 + 0) * n_joints + j for j in range(n_joints)]
                           for k in range(args.delta_scales)],
            "down_index": [[(k * 2 + 1) * n_joints + j for j in range(n_joints)]
                           for k in range(args.delta_scales)],
            "reset_value": "acc := 0 (i.e. angle := pose)",
        },
        "cpg": {
            "i_app": args.i_app, "vth_main": 100.0, "du_main": 0.1,
            "dv_main": 0.3, "refrac_main": 1, "vth_fb": 100.0,
            "du_fb": 1.0, "dv_fb": 0.0, "refrac_fb": 1,
            "from_fb_weight": CPG_FROM_FB_WEIGHT,
            "to_fb_weight": 10.0,
            "N": int(args.n_cpg_neurons),
            "W": cpg_weight_matrix(args.n_cpg_neurons).tolist(),
            "warmup": args.warmup,
            "fake_cpg": bool(args.fake_cpg),
            # Deployment must construct LIFCPGStepper with these, or it will
            # run at a different period than the network was trained for.
            "vth_fb":       cpgp["vth_fb"],
            "to_fb_weight": cpgp["to_fb_weight"],
            "refrac_main":  cpgp["refrac_main"],
            "spikes_per_burst": cpgp["spikes_per_burst"],
            "period_formula": "fake_cpg only: N * (vth_fb//to_fb_weight) * "
                              "(refrac_main+1); the real oscillator adds "
                              "inter-burst gap on top, so its period is "
                              "measured rather than predicted",
            "period_predicted": cpgp["period_predicted"],
            "period_sizing_probe": cpgp["period_measured"],
            "sizing_probes": cpgp["probes"],
            "period_measured_full_run": float(period),
            "sizing": req,
        },

        "per_joint_rmse_deg": rmse.tolist(),
        "kinematics":         kin_report,
        "out_calibration":    calib,
        "output_layer_stats": stats,
        "drift":              drift,
        "cpg_spike_stats":    {"spikes_per_cycle": cpg_rate,
                               "concentration_R": cpg_R},

        "args": vars(args),

        "run": {
            "git":      git_info(),
            "argv":     sys.argv,
            "cwd":      os.getcwd(),
            "script":   os.path.abspath(__file__),
            "out_dir":  str(out_dir.resolve()),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python":   sys.version.split()[0],
            "torch":    torch.__version__,
            "numpy":    np.__version__,
            "device":   str(device),
            "cuda_available": torch.cuda.is_available(),
            "seed":     args.seed,
        },

        "model_detail": {
            "class":        type(model).__name__,
            "n_params":     int(n_par),
            "param_breakdown": model.param_breakdown(),
            "output_kind":  f"2 x {args.delta_scales} spiking LIF neurons "
                            f"per joint (up/down per delta scale) driving a "
                            f"per-joint accumulator",
            "out_reset":    args.out_reset,
            # Deployment MUST reproduce this: an unclipped membrane on the
            # robot would give a different spike train from the trained one.
            "out_mem_clip": float(args.out_mem_clip),
            "out_gait_bias": not args.no_out_gait_bias,
            "tau_min":      float(args.tau_min),
            "tau_max":      float(args.tau_max),
            "tau_out_min":  float(args.tau_out_min),
            "tau_out_max":  float(args.tau_out_max),
            "slope":        float(args.slope),
            "out_slope":    float(args.out_slope),
            "thresh":       1.0,
            "layernorm":    "hidden layers only (affine=False); never on the "
                            "output layer",
            "film":         "per-gait Embedding(max_gaits, 2*hidden) after "
                            "LayerNorm, one per hidden layer",
            "surrogate":    "fast-sigmoid straight-through",
            "state_tensors": [n.replace("_in", "")
                              for n in model.state_names_in],
            "state_shapes": [list(s.shape)
                             for s in model.init_state(1, "cpu")],
            "onnx_inputs":  ["spikes", "gait"] + list(model.state_names_in),
            "onnx_outputs": ["angles", "up", "down"]
                            + list(model.state_names_out),
            "weights_file": "best_model.pt",
            "warm_steps":   warm_steps,
        },

        "data": {
            "tmax": int(args.tmax), "warmup": int(args.warmup),
            "t_lo": int(t_lo), "t_split": int(t_split), "t_hi": int(t_hi),
            "target_rows": int(target_rows),
            "target_range_deg": [float(tgt_range[0]), float(tgt_range[1])],
        },

        "training": {
            "dry_run":          bool(args.dry_run),
            "epochs_requested": int(args.epochs),
            "epochs_completed": int(epochs_done),
            "gradient_steps":   int(epochs_done * args.chunks_per_epoch),
            "batch":            int(args.batch),
            "bptt":             int(args.bptt),
            "lr_initial":       float(args.lr),
            "lr_final":         final_lr,
            "optimizer":        "Adam",
            "grad_clip":        float(args.clip),
            "loss_terms": {
                "mse":           1.0,
                "deriv_lambda":  float(args.deriv_lambda),
                "deriv_window":  int(args.deriv_window),
                "cofire_lambda": float(args.cofire_lambda),
                "spike_lambda":  float(args.spike_lambda),
            },
            "best_val_mse": best,
            "history":      hist,
        },
    }
    export_onnx(model, out_dir, device, cfg)
    print(f"\nDone — {out_dir.resolve()}")


if __name__ == "__main__":
    main()