"""
Visualise a trained CPG -> timing-layer -> grouped-SNN run.
===========================================================

Loads a checkpoint (.pt) and its config (.json) written by train.py, replays
the CPG bit-identically from the config's own parameters, and plots what the
timing layer is doing against the CPG and against the gait tables.

Usage
-----
    python visualize_timing.py                       # reads outputs/, writes into it
    python visualize_timing.py --model_dir outputs/run7
    python visualize_timing.py --ckpt epoch100.pt    # another snapshot -> viz_epoch100/
    python visualize_timing.py --scores_only         # alignment scores only, no figures
    python visualize_timing.py --drive_metric spikes # cpg_lif: exact spike counts
    python visualize_timing.py --gaits wkF bk         # subset
    python visualize_timing.py --n_cycles 4 --no_pred # tighter / faster


What is and is not recoverable
------------------------------
The TIMING layer is exact.  `TimingGroupedSNN.timing_only` re-runs the real
`_timing` method, so the rasters here are the same spikes the sub-networks
saw -- not a reconstruction.

The SUB-NETWORK hidden spikes are NOT recoverable from a checkpoint plus a
state trace, and this script does not pretend otherwise.  With the
subtractive reset (--hidden_reset subtract):

    mem_post = mem_pre - thresh * spk        spk = (mem_pre >= thresh)

a spiking unit lands at `mem_pre - thresh >= 0` and a silent unit lands at
`mem_pre < thresh`, and those ranges overlap -- given only `mem_post` the two
are indistinguishable.  With --hidden_reset zero (the default) a spiking unit
lands at exactly 0, which makes this far less ambiguous, though a unit that
was simply never driven also sits at 0.  Recovering them exactly needs either
the pre-reset value or a `record=` path added to `TimingGroupedSNN.forward`
(which is safe to add: the ONNX wrappers call `step`, not `forward`).  Until
then this script plots sub-network MEMBRANES, which are exact and carry most
of the same information, rather than inventing a raster.

`--arch dense` checkpoints have no timing layer; the script says so and emits
the CPG / gait-table / tau figures only.

Outputs (into --out_dir)
------------------------
    timing_alignment_<gait>.png   CPG raster + timing raster + per-leg GT/pred,
                                  shared time axis.  The main figure.  Under
                                  cpg_lif the second raster has one lane per
                                  sub-network (its drive LIF's spikes).
    phase_fold_<gait>.png         Same thing with time folded onto cycle phase:
                                  per-leg GT vs phase with that leg's timing
                                  neuron's phase histogram behind it.
    alignment_summary.png         |residual| heatmap, timing neuron vs leg, per
                                  gait, plus rate and concentration R.
    routing_matrices.png          Measured per-gait CPG->timing routing.
                                  --drive_source timing only.
    drive_gates.png               Per-gait CPG-channel selection, the counterpart
                                  figure for --drive_source cpg / cpg_lif. The
                                  drive is |s_gate|*||u|| under cpg and exact
                                  spike counts under cpg_lif (see --drive_metric).
                                  Exactly one of these two is produced.
    timing_rate_history.png       Per-timing-unit spk/cycle over training,
                                  one gait per line. --drive_source timing.
    drive_fanin_history.png       Channels carrying 90% of each
                                  sub-network's drive, over training.
                                  --drive_source cpg / cpg_lif counterpart.
    timing_alignment_by_joint.png Timing rasters regrouped by gait (rows) and joint
                                  (columns), one lane per leg, to compare a
                                  joint's timing across gaits.
    alignment_history.png         Velocity and corner alignment scores over
                                  training, per gait (timing and cpg_lif).
    tau_distributions.png         Learned time constants vs their init range
                                  and vs the CPG period.
    membranes_<gait>.png          Sub-network membrane traces (sampled units).
    timing_summary.json           Per-row phase / rate / alignment numbers,
                                  machine-readable.
    alignment_scores.csv          The velocity and corner alignment scores alone,
                                  one row per (gait, row): easy to open or stack.

    A checkpoint other than best_model.pt (--ckpt epoch100.pt) writes into
    <run>/viz_<ckpt>/ and suffixes timing_summary / alignment_scores with its
    name, so nothing from best_model.pt is overwritten.  --scores_only skips every
    figure and every prediction and writes just the two data files.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from train import (
    TimingGroupedSNN, drive_report, drive_spike_report, default_leg_layout,
    alignment_pair, alignment_section, _ALIGN_VEL_TAU, _ALIGN_COR_TAU,
    LIFCPGStepper, cpg_weight_matrix,
    detect_burst_threshold, burst_onsets,
    upsample_gait_tables, build_group_cols,
    build_targets, plot_reconstruction, plot_transition,
    load_gait_tables, GAIT_FILES_BY_N, outputs_path, out_path,
    joint_type_names,
    cfg_get, build_model_from_cfg, load_run,
    N_LEGS, N_JOINTS, CPG_PALETTE, CPG_FROM_FB_WEIGHT,
)

TIMING_PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261",
                  "#6a0572", "#8ecae6", "#ffb703", "#023047"]
GT_COLOR   = "#457b9d"
PRED_COLOR = "#e63946"


# ═══════════════════════════════════════════════════════════════════
# 1. Config / checkpoint loading -- see train.py
# ═══════════════════════════════════════════════════════════════════
# cfg_get / build_model_from_cfg / load_run live in train.py so this file and inference.py share
# one copy of the backward-compatibility rules.


# ═══════════════════════════════════════════════════════════════════
# 2.  Replay the CPG from the config
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


# ═══════════════════════════════════════════════════════════════════
# 3.  Model probes
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def timing_raster(model, spikes, gait_idx, device):
    """(T, n_timing) exact timing-layer spikes for a constant gait."""
    x  = torch.as_tensor(spikes, dtype=torch.float32, device=device).unsqueeze(1)
    gg = torch.full((x.shape[0], 1), int(gait_idx),
                    dtype=torch.long, device=device)
    return model.timing_only(x, gg)[:, 0].cpu().numpy()


def _read_metrics_csv(path):
    """Minimal metrics.csv reader: the header fixes column names; other rows parse as float
    where possible, and "" (a value not measured that epoch) becomes NaN. Plain `csv`,
    not pandas.
    """
    import csv
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v == "" or v is None:
                    parsed[k] = float("nan")
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v          # non-numeric column, keep as-is
            rows.append(parsed)
    return rows


def plot_timing_rate_history(metrics_csv, timing_cols, leg_cols, gait_names,
                             out_dir, dpi, prefix="spk", unit="t",
                             ylabel="spk/cycle", fname=None, what="T",
                             title=None, label="timing rate history"):
    """How each timing unit's firing rate evolved over training, one gait per line, on the
    SAME leg grid as plot_alignment (n_legs rows x C columns), so a panel's position means
    the same thing in both figures.

    The panel at (leg, k) shows the timing unit RESPONSIBLE for that column
    (`leg_cols[leg][k]` looked up through `timing_cols`), not "unit 3*leg+k": those
    coincide at --timing_shape per_leg but not at per_joint.

    Reads metrics.csv ("spk_{gait}_t{u}", blank on epochs that were not timing_log_every
    multiples, which become gaps in the line). Older runs without those columns get a
    printed note and an early return.
    """
    fname = fname or "timing_rate_history.png"
    rows = _read_metrics_csv(metrics_csv)
    if not rows:
        print(f"  [{label}] {metrics_csv} is empty, skipping")
        return
    epochs = np.array([r["epoch"] for r in rows])

    cols = set(rows[0].keys())
    if not any(c.startswith(f"{prefix}_") for c in cols):
        print(f"  [{label}] no {prefix}_<gait>_{unit}<unit> columns in "
              f"{metrics_csv} -- this run predates that logging, or was "
              f"trained in the other --drive_source mode. Skipping.")
        return

    n_legs = len(leg_cols)
    C      = len(leg_cols[0])
    owner  = {c: t for t, tc in enumerate(timing_cols) for c in tc}
    tnames = joint_type_names(C)

    fig, axes = plt.subplots(n_legs, C, figsize=(4.2 * C, 2.6 * n_legs),
                             squeeze=False, sharex=True)

    missing_cols = set()
    for leg in range(n_legs):
        for k in range(C):
            ax  = axes[leg, k]
            col = leg_cols[leg][k]
            u   = owner.get(col)
            if u is None:
                ax.set_visible(False)
                continue
            for gi, gname in enumerate(gait_names):
                key = f"{prefix}_{gname}_{unit}{u}"
                if key not in cols:
                    missing_cols.add(key)
                    continue
                y = np.array([r.get(key, float("nan")) for r in rows])
                ok = np.isfinite(y)
                if not ok.any():
                    continue
                ax.plot(epochs[ok], y[ok], color=TIMING_PALETTE[gi % len(TIMING_PALETTE)],
                        marker=".", ms=3, lw=1.1, label=gname)
            ax.set_title(f"leg {leg} · {tnames[k]} (col {col}) ← {what}{u}",
                        fontsize=8)
            ax.grid(alpha=0.2)
            if leg == n_legs - 1:
                ax.set_xlabel("epoch", fontsize=7)
            if k == 0:
                ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)

    # One legend for the whole figure (the gait set and colors are the same in every panel).
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        # Panel (0,0) may have been skipped as not-yet-firing; take the legend from any panel with lines.
        for ax in axes.ravel():
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc="upper center",
                  ncol=min(len(labels), 8), fontsize=8,
                  bbox_to_anchor=(0.5, 1.02))

    if missing_cols:
        print(f"  [{label}] {len(missing_cols)} expected "
              f"column(s) not found in metrics.csv (e.g. "
              f"{sorted(missing_cols)[:3]}) -- gait list may have changed "
              f"mid-run; those lines are simply absent.")

    fig.suptitle(title or ("Timing-layer spk/cycle over training, by gait  "
                           "(rows = legs, cols = joints within a leg)"),
                 fontsize=10, fontweight="bold", y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, out_dir, fname, dpi)


def plot_joint_gait_grid(by_joint, leg_cols, period, out_dir, dpi):
    """Timing rasters regrouped by joint and compared across gaits: one row per gait, one
    column per joint type (coxa, femur, ...), and in each cell one lane per leg, so a leg keeps
    its lane and color everywhere. A lane is whatever drives that joint: its timing unit
    (--drive_source timing) or its sub-network's collapsed drive train (cpg_lif); under plain cpg
    the ticks are shaded by drive share. A unit that drives a whole leg (a per_leg shape) cannot
    be split by joint, so those go in a single "whole leg" column.

    by_joint: gait -> (trains (L, n_row) over the plotted window, cols_of[row])
    """
    from matplotlib.colors import to_rgba
    gaits = list(by_joint)
    if not gaits:
        return
    leg_of = {c: l for l, cols in enumerate(leg_cols) for c in cols}
    k_of = {c: k for cols in leg_cols for k, c in enumerate(cols)}
    n_legs, n_k = len(leg_cols), max(len(c) for c in leg_cols)

    def slot(cols):
        """(leg, column key): a joint index for a single-joint unit, "leg" for a whole leg."""
        return leg_of.get(cols[0]), (k_of.get(cols[0]) if len(cols) == 1 else "leg")
    keys = []
    for g in gaits:
        for cols in by_joint[g][1]:
            kk = slot(cols)[1]
            if kk is not None and kk not in keys:
                keys.append(kk)
    keys.sort(key=lambda kk: (kk == "leg", kk if kk != "leg" else 0))
    jnames = joint_type_names(n_k)
    titles = [jnames[kk] if kk != "leg" else "whole leg (one unit per leg)" for kk in keys]
    lnames = (["LF", "LM", "LR", "RF", "RM", "RR"] if n_legs == 6
              else [f"leg {l}" for l in range(n_legs)])
    fig, axes = plt.subplots(len(gaits), len(keys), sharex=True, sharey=True, squeeze=False,
                             figsize=(1.4 + 4.6 * len(keys), 0.7 + (0.28 * n_legs + 0.45) * len(gaits)))
    n_cyc = max(by_joint[g][0].shape[0] for g in gaits) / period
    for r, g in enumerate(gaits):
        trains, cols_of = by_joint[g]
        top = float(trains.max()) if trains.size else 0.0
        for j, cols in enumerate(cols_of):
            l, kk = slot(cols)
            if l is None or kk not in keys or j >= trains.shape[1]:
                continue
            idx = np.flatnonzero(trains[:, j] > 0)
            if idx.size:
                shade = np.clip(trains[idx, j] / top, 0.15, 1.0) if top > 0 else np.ones(idx.size)
                base = TIMING_PALETTE[l % len(TIMING_PALETTE)]
                # vlines in DATA units, so a tick fits its lane at any lane count
                axes[r, keys.index(kk)].vlines(idx / period, l - 0.4, l + 0.4, lw=1.2,
                                               colors=[to_rgba(base, float(x)) for x in shade])
    for r, g in enumerate(gaits):
        for c in range(len(keys)):
            ax = axes[r, c]
            for cyc in range(1, int(np.ceil(n_cyc))):
                ax.axvline(cyc, color="0.75", lw=0.6, zorder=0)
            ax.set_xlim(0, n_cyc); ax.set_ylim(n_legs - 0.5, -0.5)       # first leg on top
            ax.set_yticks(range(n_legs)); ax.set_yticklabels(lnames, fontsize=7)
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(g, fontsize=9)
            if r == len(gaits) - 1:
                ax.set_xlabel("CPG cycles", fontsize=8)
    fig.suptitle("Timing rasters by joint, across gaits   (lane = leg; a leg keeps its lane and color)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    _savefig(fig, out_dir, "timing_alignment_by_joint.png", dpi)


def plot_alignment_history(metrics_csv, gait_names, out_dir, dpi, ramp=None, lines=True):
    """Alignment scores over training, from metrics.csv ("avel_/acor_<gait>_g<N>", blank on
    epochs that were not timing_log_every multiples). One row per gait, velocity score left
    and corner score right: thin lines are the sub-networks, the bold line their mean, the
    band their min-max, and the dashed line 0 = random placement. `ramp` = (hold, warmup)
    epochs shades the sparsity-penalty ramp. Runs without those columns (plain cpg, dense,
    or older than the logging) get a printed note and an early return.
    """
    import re
    import warnings
    label = "alignment history"
    rows = _read_metrics_csv(metrics_csv)
    if not rows:
        print(f"  [{label}] {metrics_csv} is empty, skipping")
        return
    epochs = np.array([r["epoch"] for r in rows])
    pat = re.compile(r"^(avel|acor)_(.+)_g(\d+)$")
    series = {}                                  # (kind, gait) -> {sub-network: (E,)}
    for col in rows[0]:
        m = pat.match(col)
        if m:
            y = np.array([r.get(col, float("nan")) for r in rows])
            series.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = y
    gaits = [g for g in gait_names if any((k, g) in series for k in ("avel", "acor"))]
    if not gaits:
        print(f"  [{label}] no avel_/acor_ columns in {metrics_csv} -- logged only for "
              f"--drive_source timing or cpg_lif, in runs made after that logging was "
              f"added. Skipping.")
        return

    kinds = (("avel", "Velocity alignment", "#0b5fa5"), ("acor", "Corner alignment", "#c1272d"))
    fig, axes = plt.subplots(len(gaits), 2, sharex=True, squeeze=False,
                             figsize=(11.5, 1.2 + 2.3 * len(gaits)))
    for r, gait in enumerate(gaits):
        for c, (kind, title, color) in enumerate(kinds):
            ax = axes[r, c]
            if ramp:
                ax.axvspan(ramp[0], ramp[0] + ramp[1], color="0.92", zorder=0,
                           label="penalty ramp" if (r, c) == (0, 0) else None)
            ax.axhline(0, color="0.4", ls="--", lw=0.8, zorder=1)
            subs = series.get((kind, gait), {})
            if subs:
                Y = np.vstack([subs[g] for g in sorted(subs)])
                seen = np.isfinite(Y).any(axis=0)
                if lines:
                    for y in Y:
                        ok = np.isfinite(y)
                        ax.plot(epochs[ok], y[ok], color=color, lw=0.7, alpha=0.22)
                with warnings.catch_warnings():          # all-NaN epochs are expected
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean, lo, hi = np.nanmean(Y, 0), np.nanmin(Y, 0), np.nanmax(Y, 0)
                ax.fill_between(epochs[seen], lo[seen], hi[seen], color=color, alpha=0.10, lw=0)
                ax.plot(epochs[seen], mean[seen], color=color, lw=2.2, marker="o", ms=3.2,
                        label="mean over sub-networks" if (r, c) == (0, 0) else None)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(gait)
            ax.grid(alpha=0.25)
    for c in range(2):                           # one y range per column, so rows compare
        lo = min(ax.get_ylim()[0] for ax in axes[:, c])
        hi = max(ax.get_ylim()[1] for ax in axes[:, c])
        for ax in axes[:, c]:
            ax.set_ylim(min(lo, -0.05), max(hi, 0.05))
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="best", fontsize=8, framealpha=0.85)
    fig.suptitle("Alignment score over training   (0 = random placement, 1 = ideal)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    _savefig(fig, out_dir, "alignment_history.png", dpi)


def plot_drive_fanin_history(metrics_csv, group_cols, leg_cols, gait_names,
                             out_dir, dpi):
    """The --drive_source cpg / cpg_lif counterpart of plot_timing_rate_history: how many
    CPG channels carry 90% of each sub-network's drive, over training, one gait per line.
    Same grid and reader; the differences are the metrics.csv prefix (fanin_ rather than
    spk_) and that the panel at (leg, k) is the SUB-NETWORK emitting that column (looked up
    through `group_cols`, since there are no timing units).

    This is the curve --drive_lambda is supposed to bend: falling = selection concentrating
    onto fewer phases; flat near n_cpg = still listening to everything (--drive_lambda too
    small, or the task needs broad coverage). Under cpg_lif it is the fanin90 of |s_gate|;
    the exact active count is nact_<gait>_g<N> in metrics.csv.
    """
    plot_timing_rate_history(
        metrics_csv, group_cols, leg_cols, gait_names, out_dir, dpi,
        prefix="fanin", unit="g", ylabel="channels for 90% of drive",
        fname="drive_fanin_history.png", what="g",
        title=("CPG channels carrying 90% of each sub-network's drive, over "
               "training, by gait  (rows = legs, cols = joints within a leg)"),
        label="drive fanin history")


@torch.no_grad()
def predict_and_membranes(model, spikes, gait_idx, device, warm, tgt_range,
                          keep_membranes=True, max_mem_steps=1500):
    """Free-run the full model; returns predictions in DEGREES for steps [warm:] plus
    per-timestep membrane snapshots if asked. Steps model.step manually because forward()
    discards the intermediate state the membranes need.
    """
    lo, hi = tgt_range
    scale, shift = (hi - lo) / 2.0, (hi + lo) / 2.0

    x = torch.as_tensor(spikes, dtype=torch.float32, device=device)
    T = x.shape[0]
    # Constant across the run, so built once rather than inside the T-step loop.
    g = torch.full((1,), int(gait_idx), dtype=torch.long, device=device)
    state = model.init_state(1, device)
    preds, mems = [], []
    for t in range(T):
        y, state, _ = model.step(x[t:t + 1], g, state)
        preds.append(y[0].cpu().numpy())
        # Capped: a state snapshot is ~12 KB, so a long window would cost hundreds of MB for traces the
        # membrane plot never reads.
        if keep_membranes and warm <= t < warm + max_mem_steps:
            # state: 9 tensors for timing_grouped (see TimingGroupedSNN), (mem1, mem2, memo) for dense.
            mems.append([s[0].cpu().numpy().copy() for s in state])

    pred = np.stack(preds)[warm:] * scale + shift
    return pred, mems


def gt_degrees(gait_tables, phase, gait_idx, phase_zero):
    """Gait-table angles at each timestep's phase, in degrees.  Mirrors
    build_targets' row indexing exactly (same modulo, same rounding)."""
    tbl = gait_tables[gait_idx]
    R   = tbl.shape[0]
    ph  = np.where(np.isnan(phase), 0.0, phase)
    row = (((ph + phase_zero) % 1.0) * R).astype(np.int64) % R
    return tbl[row]


# ═══════════════════════════════════════════════════════════════════
# 4.  Phase statistics
# ═══════════════════════════════════════════════════════════════════

def drive_channels(share, frac=0.90, alpha_min=0.38):
    # frac=None selects every STRICTLY POSITIVE channel: right once the drive is a spike COUNT (cpg_lif),
    # where "zero spikes" is exact and a 90% rule would hide channels that really fire (it keeps 15 of
    # 18 at a near-uniform row). The 90% rule stays correct for the continuous drive under cpg, where
    # nothing is exactly zero and some cutoff is unavoidable.
    """Which channels to draw for one sub-network, and how strongly. `share` is that
    sub-network's normaliized drive over the C CPG channels; returns (idx, alpha), idx
    sorted by descending share.

    Each obvious approach fails alone: all channels with alpha proportional to share is
    unreadable at 18 channels (a flat row is 18 faint rugs in a gray smear), and a top-k
    loses the gradation and needs an arbitrary cutoff. So keep the channels carrying `frac`
    of the drive (the same 90% rule as the fan-in metric, so plot and number agree) and set
    alpha by share RELATIVE TO THE LARGEST: the dominant channel is always opaque, so a
    concentrated sub-network reads as one solid rug and a flat one as many mid-alpha rugs.
    """
    share = np.asarray(share, dtype=np.float64)
    tot = share.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.empty(0, int), np.empty(0, float)
    p = share / tot
    order = np.argsort(-p)
    if frac is None:
        idx = order[p[order] > 0.0]
        if idx.size == 0:
            return np.empty(0, int), np.empty(0, float)
    else:
        keep = int((np.cumsum(p[order]) < frac).sum()) + 1
        idx = order[:keep]
    # Raw share-relative alpha, mapped into [alpha_min, 1]: without the floor the weakest member of the
    # 90% set gets ~0.03 (drawn but invisible), so rugs could not be counted against the fan-in. The
    # floor says "faint", not "absent".
    rel = p[idx] / p[idx].max()
    alpha = alpha_min + (1.0 - alpha_min) * rel
    return idx, alpha


def circular_stats(phases, weights=None):
    """Circular mean in [0,1) and concentration R in [0,1] of a set of
    cycle phases.  R is the thing to read: a unit can fire at a healthy rate
    and still be useless if R is low, because then it is firing all over the
    cycle and its mean phase means nothing.

    `weights` lets a spike count for less than one. Used by --drive_source
    cpg / cpg_lif, where a sub-network's effective input phase is the
    DRIVE-WEIGHTED mean over the channels feeding it: a channel supplying 40%
    of the drive should move that mean twice as far as one supplying 20%."""
    p = np.asarray(phases, dtype=np.float64)
    w = (np.ones_like(p) if weights is None
         else np.asarray(weights, dtype=np.float64))
    ok = np.isfinite(p) & np.isfinite(w)
    p, w = p[ok], w[ok]
    if p.size == 0 or w.sum() <= 0:
        return float("nan"), 0.0
    z = np.sum(w * np.exp(1j * 2.0 * np.pi * p)) / w.sum()
    return float((np.angle(z) / (2.0 * np.pi)) % 1.0), float(abs(z))


def fundamental_phase(cycle_traj):
    """Phase of the first Fourier component of one cycle of a waveform, in [0,1): the
    reference for "where in its cycle is this leg". Chosen over "swing onset", which needs an
    arbitrary threshold or derivative test that behaves differently across gaits; the
    fundamental is parameter-free, robust to noise and comparable across gaits. It is a
    consistent reference, not a biomechanical event (for a strongly non-sinusoidal
    trajectory it differs from the visible footfall instant).
    """
    y = np.asarray(cycle_traj, dtype=np.float64)
    y = y - y.mean()
    if not np.any(np.abs(y) > 1e-12):
        return float("nan")
    F = np.fft.rfft(y)
    if len(F) < 2:
        return float("nan")
    return float((np.angle(F[1]) / (2.0 * np.pi)) % 1.0)


def circ_diff(a, b):
    """Signed circular difference a - b, wrapped to [-0.5, 0.5) cycles.
    The exact antipode lands at -0.5 rather than +0.5; the magnitude is what
    the residual is read for, so the sign convention there is immaterial."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float((a - b + 0.5) % 1.0 - 0.5)


# ═══════════════════════════════════════════════════════════════════
# 5.  Figures
# ═══════════════════════════════════════════════════════════════════

def _savefig(fig, out_dir, name, dpi):
    # out_path routes by filename prefix into recons/ timing_alignments/ membrane_waveforms/
    # phase_folds/ (see OUT_ROUTES in train.py).
    p = out_path(out_dir, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {p}")


def unit_train(tspk, g, ch):
    """One unit's binary train for either raster layout: (L, lanes) where a lane IS the
    unit (timing layer, raw CPG channels), or (L, G, C) under cpg_lif, where the unit is
    (sub-network, channel) because the drive LIF thins and delays its channel's burst, so
    plotting the raw channel would show input the sub-network never received.
    """
    return tspk[:, g, ch] if tspk.ndim == 3 else tspk[:, ch]


def plot_alignment(spikes, tspk, gt, pred, phase, onsets, period, burst_thr,
                   timing_cols, leg_cols, gait_name, out_dir, dpi, t_lo, t_hi,
                   drive_share=None, drive_frac=0.90):
    """The main figure: two full-width rasters on top, then a grid of ONE ROW PER LEG and
    ONE COLUMN PER JOINT WITHIN THAT LEG (from `leg_cols`, the robot's anatomy, not the
    network's grouping: 6 x 3 for the hexapod, 4 x 2 for the quadruped, whatever the
    structure). Joints are split rather than overlaid because their amplitudes differ too
    much for one y-scale.

    The second raster is the timing layer's spikes under drive_source=timing. Under cpg /
    cpg_lif timing_only returns the CPG train unchanged (a copy of the first raster), so it
    becomes one lane per SUB-NETWORK, drawn with the same channel selection and alphas as
    the rug beneath that sub-network's panel.

    `drive_share` switches to cpg / cpg_lif mode: pass that gait's (G, C) normaliized drive
    and `timing_cols=group_cols`, and each panel overlays the CPG channels feeding its
    sub-network, alpha-weighted by share (see `drive_channels`). Otherwise each panel's rug
    is the timing unit responsible for that column (looked up through `timing_cols`, not
    `group_cols`, which has a different length when --timing_shape and --decoder_shape
    differ); with per_leg timing a leg's three panels share one rug. Every axis shares x
    with the rasters.
    """
    n_cpg = spikes.shape[1]
    # tspk is (L, lanes) or (L, G, C); axis 1 is the lane count on the timing-layer path (the cpg
    # path overrides it with n_lane).
    G     = tspk.shape[1]                   # timing neurons (raster lanes)
    n_legs = len(leg_cols)
    C     = len(leg_cols[0])
    tnames = joint_type_names(C)
    # column -> the timing unit responsible for it, built from timing_cols so the index is always a
    # valid raster lane.
    owner = {c: t for t, cols in enumerate(timing_cols) for c in cols}
    sl    = slice(t_lo, t_hi)
    t     = np.arange(t_lo, t_hi)

    # Both rasters are sized PER LANE, not by a fixed ratio, so lane spacing stays constant whatever
    # the lane counts (a fixed height crammed 18 timing lanes into a 6-lane slot). RASTER_PER_LANE
    # gives the old 1.1 at 6 lanes; MIN_RASTER_H keeps a 3-neuron CPG from getting a sliver.
    RASTER_PER_LANE, MIN_RASTER_H, LEG_H = 0.183, 1.1, 1.5
    # Lanes in the SECOND raster: one per SUB-NETWORK under cpg / cpg_lif (6 sub-networks vs 18
    # channels at per_leg), so not tspk.shape[1].
    n_lane = drive_share.shape[0] if drive_share is not None else G
    CPG_H = max(MIN_RASTER_H, RASTER_PER_LANE * n_cpg)
    tim_h = max(MIN_RASTER_H, RASTER_PER_LANE * n_lane)
    units = CPG_H + tim_h + LEG_H * n_legs
    fig = plt.figure(figsize=(max(14.0, 5.8 * C), 1.05 * units + 1.0))
    # top/bottom pinned so the suptitle sits close to the first raster (bbox_inches="tight" crops
    # the outer margin, not internal gaps).
    gs  = gridspec.GridSpec(2 + n_legs, C, hspace=0.62, wspace=0.20,
                            top=0.935, bottom=0.045, left=0.055, right=0.99,
                            height_ratios=[CPG_H, tim_h] + [LEG_H] * n_legs)

    # The two rasters span every column; the leg rows do not.
    ax_cpg = fig.add_subplot(gs[0, :])
    ax_tim = fig.add_subplot(gs[1, :], sharex=ax_cpg)
    ax_leg = [[fig.add_subplot(gs[2 + j, k], sharex=ax_cpg)
               for k in range(C)] for j in range(n_legs)]
    all_axes = [ax_cpg, ax_tim] + [a for row in ax_leg for a in row]

    # sharex= does not auto-hide tick labels like plt.subplots(sharex=True), so hide them here;
    # only the bottom leg row needs them.
    for a in all_axes:
        if a not in ax_leg[-1]:
            a.tick_params(labelbottom=False)

    on_win = onsets[(onsets >= t_lo) & (onsets < t_hi)]

    def cycle_lines(ax):
        for b in on_win:
            ax.axvline(b, color="k", lw=0.9, alpha=0.35, ls="--", zorder=0)

    # ── CPG raster ────────────────────────────────────────────────
    for i in range(n_cpg):
        idx = np.where(spikes[sl, i] > 0)[0] + t_lo
        ax_cpg.scatter(idx, np.full(len(idx), i), marker="|", s=110, lw=1.5,
                       color=CPG_PALETTE[i % len(CPG_PALETTE)])
    ax_cpg.set_yticks(range(n_cpg))
    ax_cpg.set_yticklabels([f"CPG {i}" for i in range(n_cpg)], fontsize=8)
    ax_cpg.set_ylim(-0.6, n_cpg - 0.4)
    cycle_lines(ax_cpg)
    ax_cpg.set_title(f"CPG raster   (dashed = neuron-0 burst onset, "
                     f"period ≈ {period:.0f} steps)", fontsize=9)
    ax_cpg.grid(axis="x", alpha=0.15)

    # ── second raster: what each SUB-NETWORK is actually driven by ──
    # Under cpg / cpg_lif: one lane per sub-network, filled by the SAME rule as the rug under that
    # sub-network's waveform panel (the channels carrying 90% of its drive, or every channel with
    # spikes under the spikes metric; alpha by share relative to the largest; colored by channel),
    # so the raster and the panels line up by construction.
    if drive_share is not None:
        for j in range(n_lane):
            for ch, a in zip(*drive_channels(drive_share[j],
                                             frac=drive_frac)):
                idx = np.where(unit_train(tspk, j, int(ch))[sl] > 0)[0] + t_lo
                ax_tim.scatter(idx, np.full(len(idx), j), marker="|", s=110,
                               lw=1.5, alpha=float(a),
                               color=TIMING_PALETTE[int(ch)
                                                    % len(TIMING_PALETTE)])
        lane_lab = [f"g{j}({len(drive_channels(drive_share[j], frac=drive_frac)[0])})"
                    for j in range(n_lane)]
        tim_title = ("Sub-network drive raster  (lane g = the CPG channels "
                     "driving sub-network g, alpha by drive share — same "
                     "selection and colors as the rugs below; (n) = that "
                     "sub-network's fanin90)")
    else:
        for j in range(n_lane):
            idx = np.where(tspk[sl, j] > 0)[0] + t_lo
            ax_tim.scatter(idx, np.full(len(idx), j), marker="|", s=110,
                           lw=1.5,
                           color=TIMING_PALETTE[j % len(TIMING_PALETTE)])
        lane_lab = [f"T{j}" for j in range(n_lane)]
        tim_title = ("Timing layer raster  (exact — same spikes the "
                     "sub-networks saw)")
    ax_tim.set_yticks(range(n_lane))
    ax_tim.set_yticklabels(lane_lab, fontsize=8 if n_lane <= 12 else 7)
    ax_tim.set_ylim(-0.6, n_lane - 0.4)
    cycle_lines(ax_tim)
    ax_tim.set_title(tim_title, fontsize=9)
    ax_tim.grid(axis="x", alpha=0.15)

    # ── leg rows x joint columns ──────────────────────────────────
    for j in range(n_legs):
        for k, c in enumerate(leg_cols[j]):
            ax = ax_leg[j][k]
            # Whichever timing neuron drives this column, not this leg: they
            # differ once n_timing > n_legs.
            g_own = owner.get(c)
            if drive_share is not None and g_own is not None:
                # --drive_source cpg / cpg_lif: the panel's owner is a
                # SUB-NETWORK, and its input is a weighted mix of CPG channels
                # rather than one timing unit. Draw the channels carrying 90%
                # of its drive (every channel with a nonzero spike count under
                # the spikes metric), each at an alpha set by its share
                # relative to the largest, and color by CHANNEL so a rug can
                # be cross-referenced against drive_gates.png's x axis. The
                # dominant channel's rug is fully opaque, so one solid rug =
                # concentrated and many mid-alpha rugs = unselected.
                ch_idx, ch_a = drive_channels(drive_share[g_own],
                                              frac=drive_frac)
                rugs = [(int(ch), float(a),
                         TIMING_PALETTE[int(ch) % len(TIMING_PALETTE)])
                        for ch, a in zip(ch_idx, ch_a)]
            else:
                rugs = ([(g_own, 1.0,
                          TIMING_PALETTE[g_own % len(TIMING_PALETTE)])]
                        if g_own is not None else [])
            # Burst onsets come from the STRONGEST contributor only. Onsets
            # from 17 weakly-weighted channels would be a picket fence.
            spk_idx = (np.where(unit_train(tspk, g_own if g_own is not None
                                            else 0, rugs[0][0])[sl] > 0)[0]
                       + t_lo if rugs else np.empty(0, int))
            bursts = spike_bursts(spk_idx, burst_thr)
            col_ = rugs[0][2] if rugs else "#c8cdd4"

            ax.plot(t, gt[sl, c], color=GT_COLOR, lw=1.7, label="GT", zorder=3)
            if drive_share is not None and g_own is not None:
                # len(rugs) IS fanin90 for this sub-network -- same 90% rule
                # -- so the number is printed and the rugs can be counted
                # against it instead of guessed at.
                lab = (f" ← g{g_own} ch{rugs[0][0]} "
                       + (f"(active {len(rugs)})" if drive_frac is None
                          else f"(fanin90 {len(rugs)})"))
            else:
                lab = f" ← T{g_own}" if g_own is not None else ""
            title = f"leg {j} · {tnames[k]} (col {c}){lab}"
            if pred is not None:
                ax.plot(t, pred[sl, c], color=PRED_COLOR, lw=1.2, ls="--",
                        alpha=0.9, label="pred", zorder=4)
                r = float(np.sqrt(np.nanmean(
                    (pred[sl, c] - gt[sl, c]) ** 2)))
                title += f"   RMSE={r:.2f}°"

            # Timing spikes for THIS leg, on every one of its joint panels:
            # one timing neuron drives the whole group, so the same spike
            # train is the relevant overlay for each of its joints.
            #
            # Drawn as a RUG along the bottom of the panel rather than as
            # full-height lines.  Full-height was readable at ~10 spikes per
            # cycle but the min_count objective drives the rate to 40-60, at
            # which point full-height bars cover the trace they are meant to be
            # compared against.  Burst ONSETS stay full-height, since those are
            # the few landmarks worth reading against the waveform directly.
            lo, hi = ax.get_ylim()
            rug = lo + 0.10 * (hi - lo)
            for ch, a, cc in rugs:
                idx = np.where(unit_train(tspk, g_own if g_own is not None
                                          else 0, ch)[sl] > 0)[0] + t_lo
                ax.vlines(idx, lo, rug, color=cc, alpha=0.75 * a, lw=1.0,
                          zorder=5)
            for bst in bursts:
                ax.axvline(bst[0], color=col_, lw=1.3, alpha=0.45, zorder=1)
            ax.set_ylim(lo, hi)

            cycle_lines(ax)
            ax.set_title(title, fontsize=8)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=7)
            if k == 0:
                ax.set_ylabel(f"leg {j}\n(°)", fontsize=8)
            if j == 0 and k == C - 1:
                ax.legend(fontsize=6, loc="upper right")

    for k in range(C):
        ax_leg[-1][k].set_xlabel("CPG timestep", fontsize=8)
    ax_cpg.set_xlim(t_lo, t_hi)
    rug_note = ("2nd raster lane g = rug of sub-network g, same rule: the "
                "CPG channels driving it, alpha by drive share (ch# in each "
                "title = strongest); full-height line = the strongest "
                "channel's burst onset"
                if drive_share is not None else
                "rug = the timing neuron driving THAT joint (T# in each "
                "title); full-height line = its burst onset")
    fig.suptitle(f"{gait_name} — timing alignment   "
                 f"rows = legs, columns = joints within a leg   |   "
                 f"{rug_note}",
                 fontsize=11, fontweight="bold")
    _savefig(fig, out_dir, f"timing_alignment_{gait_name}.png", dpi)


def plot_phase_fold(tspk, phase, gait_tables, gait_idx, timing_cols,
                    gait_name, phase_zero, out_dir, dpi, n_bins=72,
                    drive_share=None, drive_frac=0.90):
    """Time folded onto cycle phase, so alignment is one picture per TIMING UNIT: the
    trajectory of the column(s) it is responsible for over one cycle, with its spike-phase
    histogram behind it. One row per timing unit, indexed through `timing_cols` (not per
    sub-network through group_cols, a different length when --timing_shape and
    --decoder_shape differ). With per_joint timing a row is one joint and the reference
    fundamental is that joint's own, the sharper measure; with per_leg timing it shows the
    whole leg.

    `drive_share` switches to cpg / cpg_lif mode: pass that gait's (G, C) normalized drive
    and `timing_cols=group_cols`. A row is then a SUB-NETWORK, whose histogram pools the CPG
    channels feeding it with each spike weighted by its channel's share, so the circular mean
    is its effective input phase (a channel supplying 40% of the drive moves it twice as far
    as one supplying 20%).
    """
    # Rows are sub-networks when drive_share is given (tspk may then be (L, G, C), whose axis 1 is
    # already G), else timing units.
    G   = (drive_share.shape[0] if drive_share is not None
           else tspk.shape[1])
    tbl = gait_tables[gait_idx]
    R   = tbl.shape[0]
    ok  = np.isfinite(phase)

    # Shorter rows with many units (per_joint timing gives 18; 2.5in each is an unreadable figure).
    row_h = 2.5 if G <= 8 else 1.6
    fig, axes = plt.subplots(G, 1, figsize=(11, row_h * G), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    x_tbl = ((np.arange(R) / R) - phase_zero) % 1.0
    order = np.argsort(x_tbl)

    for j in range(G):
        ax   = axes[j]
        col_ = TIMING_PALETTE[j % len(TIMING_PALETTE)]
        cols = timing_cols[j]

        for k, c in enumerate(cols):
            ax.plot(x_tbl[order], tbl[order, c], color=GT_COLOR, lw=1.9,
                    ls="-" if k == 0 else "-.", label=f"GT c{c}", zorder=3)
        un = "g" if drive_share is not None else "T"
        ax.set_ylabel(f"{un}{j} cols {cols} (°)", fontsize=8)
        ax.grid(alpha=0.2)

        if drive_share is not None:
            # Pool the channels carrying 90% of this sub-network's drive,
            # each spike weighted by its channel's share.
            ch_idx, _ = drive_channels(drive_share[j], frac=drive_frac)
            ph_l, w_l = [], []
            tot = float(drive_share[j].sum()) or 1.0
            for ch in ch_idx:
                sel = (unit_train(tspk, j, int(ch)) > 0) & ok
                if not sel.any():
                    continue
                ph_l.append(phase[sel])
                w_l.append(np.full(int(sel.sum()),
                                   float(drive_share[j][int(ch)]) / tot))
            ph_j = np.concatenate(ph_l) if ph_l else np.empty(0)
            w_j  = np.concatenate(w_l) if w_l else np.empty(0)
        else:
            m    = (tspk[:, j] > 0) & ok
            ph_j = phase[m]
            w_j  = None

        ax2 = ax.twinx()
        if ph_j.size > 0:
            ax2.hist(ph_j, bins=n_bins, range=(0.0, 1.0), weights=w_j,
                     color=col_, alpha=0.28, zorder=1)
            mu, R_ = circular_stats(ph_j, w_j)
            ax2.axvline(mu, color=col_, lw=2.0, ls="-", zorder=2)
            f_ph = fundamental_phase(tbl[order, cols[0]])
            res  = circ_diff(mu, f_ph)
            ax.set_title(
                f"{un}{j}: mean phase {mu:.3f}  R={R_:.2f}  |  "
                f"col {cols[0]} fundamental {f_ph:.3f}  |  "
                f"residual {res:+.3f} cyc", fontsize=8)
        else:
            ax2.set_title(
                (f"g{j}: NO DRIVE — no channel feeding it fires"
                 if drive_share is not None else
                 f"T{j}: NO SPIKES — every sub-network it feeds "
                 f"gets no input from it"),
                fontsize=8, color="#e63946")
        ax2.set_ylabel("T spikes", fontsize=7)
        ax2.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper left")

    axes[-1].set_xlabel("cycle phase  (0 = neuron-0 burst onset)")
    axes[-1].set_xlim(0.0, 1.0)
    fig.suptitle(f"{gait_name} — phase-folded timing alignment",
                 fontsize=11, fontweight="bold")
    _savefig(fig, out_dir, f"phase_fold_{gait_name}.png", dpi)


def plot_alignment_summary(summary, gait_names, G, out_dir, dpi):
    """Three heatmaps, gait x timing-neuron: absolute residual, firing rate and concentration
    R. The residual panel says whether the learned per-gait routing found an alignment the
    old fixed permutation could not.
    """
    ng   = len(gait_names)
    pre  = "g" if summary[gait_names[0]][0].get("row_kind") == "sub_network" else "T"
    res  = np.full((ng, G), np.nan)
    rate = np.full((ng, G), np.nan)
    conc = np.full((ng, G), np.nan)
    for gi, gname in enumerate(gait_names):
        for j in range(G):
            d = summary[gname][j]
            res[gi, j]  = abs(d["residual_cycles"]) if d["residual_cycles"] is not None else np.nan
            rate[gi, j] = d["rate_per_cycle"]
            conc[gi, j] = d["R"]

    fig, axes = plt.subplots(1, 3, figsize=(4.6 * 3, 1.0 * ng + 2.4))
    panels = [
        (res,  "|residual| (cycles)", "YlOrRd", None,  "{:.3f}"),
        (rate, "timing rate (spk/cycle)", "Blues", None, "{:.1f}"),
        (conc, "concentration R", "Greens", (0.0, 1.0), "{:.2f}"),
    ]
    for ax, (M, title, cmap, lim, fmt) in zip(axes, panels):
        vmin, vmax = (lim if lim else (0.0, np.nanmax(M) if np.isfinite(M).any() else 1.0))
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(G))
        ax.set_xticklabels([f"{pre}{j}" for j in range(G)], fontsize=8)
        ax.set_yticks(range(ng))
        ax.set_yticklabels(gait_names, fontsize=8)
        ax.set_title(title, fontsize=9)
        for a in range(ng):
            for b in range(G):
                v = M[a, b]
                if np.isfinite(v):
                    hot = v > vmin + 0.6 * (vmax - vmin)
                    ax.text(b, a, fmt.format(v), ha="center", va="center",
                            fontsize=7, color="white" if hot else "black")
    fig.suptitle("Timing-layer alignment summary  "
                 "(residual = timing mean phase − leg fundamental phase)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir, "alignment_summary.png", dpi)


@torch.no_grad()
def plot_drive_gates(model, gait_names, leg_cols, out_dir, dpi,
                     gait_idx=None, drive_all=None, metric="product",
                     report_ctx=None):
    """Per-gait CPG-channel selection, for --drive_source cpg / cpg_lif: the counterpart of
    plot_routing and the central figure for those modes. One panel per gait, rows =
    sub-network, columns = CPG channel (channel index is essentially cycle phase, ~c/n_cpg,
    so phase runs along x). Values are normalized across each ROW ("what fraction of this
    joint's drive comes from each phase"); absolute magnitude is discarded (under cpg `u`
    can absorb overall scale). The title carries the mean fanin90 and participation ratio
    (the mean exact active count under the spikes metric), since a flat-but-dim panel and a
    concentrated one look alike.

    The quantity follows --drive_metric. Under cpg the default is `drive_strength`,
    |s_gate| * ||u[g,c,:]||_1 (the two are multiplicatively redundant, so |s_gate| alone
    carries only part of the scale), read from the parameters directly: sound because
    s_gate and u sit before the sum over channels. Under cpg_lif there is a threshold in
    between, so the default replays the drive LIF and counts spikes; --drive_metric gain
    plots |s_gate|, what the penalty acts on.

    Reading it:
      - a column concentrated on few rows: that joint samples at a few phases and is silent
        elsewhere (what --drive_lambda drives toward; a temporal claim, since channels are
        phase-localiized).
      - the annotated argmax per column is the joint's dominant phase; how well it matches
        the joint's motion is scored by the alignment scores in timing_summary.json.
      - columns differing across gaits: per-gait selection is being used; identical
        everywhere: gaits are not differentiating.
      - several joints sharing a dominant channel is EXPECTED (that is a tripod).
    """
    if getattr(model, "uses_timing", True):
        return None
    # Under cpg the drive is |s_gate|*||u[g,c,:]||_1, not |s_gate| alone (reading |s_gate| made every
    # sub-network look dominated by channel 0: all-ones init, argmax ties go to index 0); under cpg_lif
    # it is |s_gate|. Shared with train.py's drive_report so figure and printout agree.
    G, C = model.G, model.n_neurons
    ng = len(gait_names)
    # drive_strength is indexed by the model's REAL gait index; `gait_names` is the plotted list
    # (--gaits can subset or reorder it), so select the real rows rather than S[:ng].
    gidx = list(range(ng)) if gait_idx is None else list(gait_idx)
    if len(gidx) != ng:
        raise ValueError(f"gait_idx has {len(gidx)} entries for "
                         f"{ng} gait names")
    Dall = (np.asarray(drive_all) if drive_all is not None
            else model.drive_strength().detach().cpu().numpy())
    bad = [i for i in gidx if not 0 <= i < Dall.shape[0]]
    if bad:
        raise ValueError(
            f"gait index/indices {bad} are outside drive_strength's "
            f"{Dall.shape[0]} rows (sized by the MODEL's n_gaits): a --gaits "
            f"selection naming a gait the model was not trained on cannot be "
            f"plotted here.")
    S = Dall[gidx]                                           # (ng, G, C)

    # Normalize every panel first, then share one data-driven color scale: a fixed vmax=1.0 would wash
    # the figure out (an even two-channel split tops out at 0.5).
    Mn = []
    for gi in range(ng):
        # (G, C): rows = sub-network, cols = CPG channel; channel index is essentially cycle PHASE, which
        # belongs on x.
        M = S[gi]
        row = M.sum(axis=1, keepdims=True)
        Mn.append(M / np.where(row > 0, row, 1.0))     # share per sub-network
    vmax = max(float(m.max()) for m in Mn) or 1.0

    # Two different concentration metrics, both shown and named: fanin90 (channels needed to reach 90%
    # of the drive: what drive_report prints and metrics.csv logs, so it goes first) and PR (the
    # participation ratio drive_penalty('pr') minimizes). They are not interchangeable: PR is
    # systematically LOWER on a skewed row (one dominant channel plus a thin tail gives PR ~3 where
    # fanin90 is ~15).
    pr = [float((m.sum(axis=1) ** 2 / np.maximum((m ** 2).sum(axis=1), 1e-12))
                .mean()) for m in Mn]
    f90m, actm = [], []
    for m in Mn:
        srt = -np.sort(-m, axis=1)
        f90m.append(float(((np.cumsum(srt, axis=1) < 0.90).sum(axis=1) + 1)
                          .mean()))
        actm.append(float((m > 0).sum(axis=1).mean()))

    # Near-square grid rather than 1 x ng: each panel is G x C (18 x 18), so one row makes them tall
    # and thin. ncols from sqrt: 4 gaits -> 2x2, 6 -> 2x3, 8 -> 3x3 with one cell hidden.
    ncols = int(math.ceil(math.sqrt(ng)))
    nrows = int(math.ceil(ng / ncols))
    fig, axes2 = plt.subplots(nrows, ncols,
                              figsize=(3.4 * ncols, (1.2 + 0.20 * G) * nrows),
                              squeeze=False)
    axes = axes2.ravel()
    for extra in range(ng, nrows * ncols):
        axes[extra].set_visible(False)
    argmax_tbl, im = {}, None
    for gi, gname in enumerate(gait_names):
        M  = Mn[gi]
        ax = axes[gi]
        im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(C))
        ax.set_xticklabels([f"{i}" for i in range(C)], fontsize=6)
        ax.set_yticks(range(G))
        ax.set_yticklabels([f"g{j}" for j in range(G)], fontsize=6)
        # Under the spikes metric "active" is exact, so lead with it;
        # fanin90's cutoff is only needed when nothing is ever zero.
        ax.set_title((f"{gname}  (mean active {actm[gi]:.1f}, "
                      f"mean PR {pr[gi]:.1f}, of {C})"
                      if metric == "spikes" else
                      f"{gname}  (mean fanin90 {f90m[gi]:.1f}, "
                      f"mean PR {pr[gi]:.1f}, of {C})"), fontsize=8)
        if gi // ncols == nrows - 1 or gi + ncols >= ng:
            ax.set_xlabel("CPG channel ~ phase", fontsize=7)
        if M.sum() > 0:
            dom = M.argmax(axis=1)
            argmax_tbl[gname] = [int(d) for d in dom]
            for j, d in enumerate(dom):
                # White, not red: the dominant cell is the BRIGHTEST, so a dark marker on it is invisible.
                ax.text(d, j, "x", ha="center", va="center",
                        color="w", fontsize=5, fontweight="bold")
        if gi % ncols == 0:
            ax.set_ylabel("sub-network", fontsize=7)
    if im is not None:
        fig.colorbar(im, ax=[a for a in axes if a.get_visible()],
                     fraction=0.02,
                     label="share of this sub-network's drive")
    fig.suptitle("Per-gait CPG channel selection "
                 "(|s_gate|*||u||, normalized per sub-network)   "
                 "x = dominant channel\n"
                 "fanin90 = channels carrying 90% of the drive (the metric "
                 "drive_report prints and metrics.csv logs);   "
                 "PR = (sum d)^2 / sum d^2, what --drive_penalty pr "
                 "minimizes.  PR reads LOWER on a skewed row.",
                 fontsize=9, fontweight="bold")
    _savefig(fig, out_dir, "drive_gates.png", dpi)

    # Effective fan-in per (gait, sub-network), the number --drive_lambda is meant to reduce.
    # train.py's drive_report is called verbatim, not reimplemented: two copies of one statistic
    # cannot be compared safely (this file used to print min/median/max while train.py printed
    # the mean), and one function makes printout and figure agree by construction, so any remaining
    # difference is the MODEL (a checkpoint other than the one that produced the log). Only when the
    # plotted set is the model's own gait order: drive_report labels row gi with names[gi], so a
    # --gaits subset or reordering would attach numbers to the wrong gait.
    if gidx == list(range(ng)):
        # Whichever reporter the training log used, called verbatim so figure and log cannot disagree;
        # drive_spike_report needs the CPG train, so it is only callable when one was passed in.
        rep = ("drive_spike_report" if metric == "spikes" else "drive_report")
        print(f"    train.py {rep}, verbatim (same function the training "
              f"log uses):")
        try:
            if metric == "spikes":
                if report_ctx is None:
                    raise ValueError("drive_spike_report needs the CPG train; "
                                     "pass report_ctx=(spikes, device, period)")
                sp_, dev_, per_ = report_ctx
                lines, _ = drive_spike_report(
                    model, sp_, ng, dev_, per_, t0=0,
                    n_steps=min(int(6*per_), len(sp_)-1),
                    gait_names=list(gait_names), indent="      ")
            else:
                lines, _ = drive_report(model, ng,
                                        gait_names=list(gait_names),
                                        indent="      ")
            for ln in lines:
                print(ln)
        except Exception as exc:             # never let a print kill a figure
            print(f"      [{rep} unavailable: {exc}]")
    else:
        print(f"    (skipping train.py's drive_report: --gaits selected "
              f"{gidx}, not the model's own order, and it labels rows by "
              f"position)")

    # Per-sub-network values too, so the comparison can be elementwise.
    eff = []
    for gi in range(ng):
        M = S[gi]                                       # (G, C)
        tot = M.sum(axis=1, keepdims=True)
        frac = M / np.where(tot > 0, tot, 1.0)
        srt = -np.sort(-frac, axis=1)
        k90 = (np.cumsum(srt, axis=1) < 0.90).sum(axis=1) + 1
        eff.append(k90)
        lbl = "active (exact)" if metric == "spikes" else "fanin90"
        vals = (S[gi] > 0).sum(axis=1) if metric == "spikes" else k90
        print(f"    {gait_names[gi]:>8}: {lbl} per sub-network "
              f"{np.array2string(vals, max_line_width=200)}")
        print(f"    {'':>8}  mean {vals.mean():.2f}  median "
              f"{int(np.median(vals))}  min {int(vals.min())}  "
              f"max {int(vals.max())}  (of {C})")
        if metric == "spikes":
            print(f"    {'':>8}  fanin90 for comparison: mean "
                  f"{k90.mean():.2f} -- LOWER than the active count because "
                  f"a 90% cutoff drops real channels")
    return {"dominant_channel": argmax_tbl,
            "channels_for_90pct_drive": [[int(v) for v in e] for e in eff]}


def plot_routing(model, gait_names, device, period, out_dir, dpi,
                 n_probe_cycles=4):
    """Effective CPG -> timing routing, MEASURED rather than read off `w_in_gait`:
    thresholds, membrane decay and the reset sit between a weight and a spike, so the matrix
    alone does not say what the layer does. For each gait and CPG neuron, feed a synthetic
    burst on that neuron alone and count which timing units respond. The annotated argmax
    per column is the strongest CPG driver of each timing unit. Reading it:
      - orderings that differ across gaits: the per-gait weights are being used
      - identical everywhere: gaits are not differentiating (the timing layer collapsed to
        a gait-independent clock, or the gait set does not need distinct routings)
      - several timing units sharing one driver is EXPECTED (a tripod: 3 legs at one phase,
        3 at the opposite)
    """
    # Under cpg / cpg_lif there is no router to probe: timing_only returns the CPG train unchanged,
    # so this would plot an identity diagonal that looks like a result and is not one.
    # plot_drive_gates is the figure for those modes.
    if not getattr(model, "uses_timing", True):
        print("  [routing] drive_source=cpg/cpg_lif: no CPG->timing router to probe, "
              "skipping (see drive_gates.png instead)")
        return None

    n_cpg, n_t = model.n_neurons, model.n_timing
    ng = len(gait_names)
    P  = int(period)

    # Synthetic burst probe: 10 spikes 3 steps apart (the original CPG's burst shape; fixed, not read
    # from this run's config), repeated for a few cycles so membranes settle.
    M = np.zeros((ng, n_cpg, n_t))
    for i in range(n_cpg):
        probe = np.zeros((P * n_probe_cycles, n_cpg), dtype=np.float32)
        for c in range(n_probe_cycles):
            for k in range(10):
                t = c * P + k * 3
                if t < probe.shape[0]:
                    probe[t, i] = 1.0
        x = torch.as_tensor(probe, device=device).unsqueeze(1)
        for g in range(ng):
            gg = torch.full((x.shape[0], 1), g, dtype=torch.long, device=device)
            M[g, i] = model.timing_only(x, gg)[:, 0].sum(0).cpu().numpy()

    v = float(M.max()) or 1.0
    fig, axes = plt.subplots(1, ng, figsize=(2.9 * ng, 2.6 + 0.28 * n_cpg),
                             squeeze=False)
    axes = axes[0]
    argmax_tbl, im = {}, None
    for gi, gname in enumerate(gait_names):
        ax = axes[gi]
        im = ax.imshow(M[gi], cmap="viridis", vmin=0, vmax=v, aspect="auto")
        ax.set_xticks(range(n_t))
        ax.set_xticklabels([f"T{j}" for j in range(n_t)], fontsize=7)
        ax.set_yticks(range(n_cpg))
        ax.set_yticklabels([f"N{i}" for i in range(n_cpg)], fontsize=7)
        ax.set_title(gname, fontsize=8)
        if M[gi].sum() > 0:
            dom = M[gi].argmax(axis=0)
            argmax_tbl[gname] = [int(d) for d in dom]
            for j in range(n_t):
                ax.add_patch(plt.Rectangle((j - 0.5, dom[j] - 0.5), 1, 1,
                                           fill=False, edgecolor="w", lw=1.8))
        for i in range(n_cpg):
            for j in range(n_t):
                ax.text(j, i, f"{M[gi, i, j]:.0f}", ha="center", va="center",
                        fontsize=6,
                        color="black" if M[gi, i, j] > 0.6 * v else "white")
    if im is not None:
        plt.colorbar(im, ax=axes.tolist(), fraction=0.02,
                     label="timing spikes per probe")
    fig.suptitle("Measured CPG → timing routing  "
                 "(probe one CPG neuron, count timing responses; "
                 "boxed = strongest driver)",
                 fontsize=9, fontweight="bold")
    _savefig(fig, out_dir, "routing_matrices.png", dpi)

    print("\n  Strongest CPG driver per timing unit (measured):")
    for gname in gait_names:
        dom = argmax_tbl.get(gname)
        if dom is None:
            print(f"    {gname:>22s} : NO timing response to any probe")
            continue
        print(f"    {gname:>22s} : " +
              "  ".join(f"T{j}<-N{d}" for j, d in enumerate(dom)))
    uniq = {tuple(d) for d in argmax_tbl.values()}
    if len(uniq) <= 1:
        print("    All gaits share one routing: the per-gait weights are not "
              "differentiating them (the timing layer may have collapsed to a "
              "gait-independent clock, or this gait set does not need distinct routings).")
    else:
        print(f"    {len(uniq)} distinct routings across {len(argmax_tbl)} "
              f"gaits — the per-gait gate is being used.")
    shared = [d for d in uniq if len(set(d)) < n_t]
    if shared:
        print(f"    {len(shared)}/{len(uniq)} routings have timing units "
              f"sharing a CPG driver (expected: that is what a tripod is).")
    return argmax_tbl


def plot_taus(model, period, cfg, out_dir, dpi):
    """Learned time constants against their init ranges and the CPG period: is
    --tau_timing_max binding, and did the sub-network taus spread across the cycle?
    """
    tau = lambda logit: (
        -1.0 / np.log(np.clip(1.0 / (1.0 + np.exp(-logit.detach().cpu()
                                                  .numpy().ravel())),
                              1e-9, 1 - 1e-12)))

    series = []
    if hasattr(model, "beta_t_logit"):
        series.append(("timing", tau(model.beta_t_logit),
                       float(cfg_get(cfg, "tau_timing_min", 2.0)),
                       float(cfg_get(cfg, "tau_timing_max",
                                     period / max(int(cfg_get(
                                         cfg, "n_cpg_neurons", 4)), 1)))))
    tmin = float(cfg_get(cfg, "tau_min", 2.0))
    tmax = float(cfg_get(cfg, "tau_max", 256.0))
    series += [("hidden 1", tau(model.beta1_logit), tmin, tmax),
               ("hidden 2", tau(model.beta2_logit), tmin, tmax),
               # 40.0 (not period-derived) matches build_model_from_cfg's fallback, the literal hardcoded value
               # before this became a derived option, so it is the right init-range marker for older configs.
               ("readout",  tau(model.betao_logit), 2.0,
                float(cfg_get(cfg, "tau_readout_max", 40.0)))]

    fig, axes = plt.subplots(1, len(series), figsize=(3.5 * len(series), 3.4))
    if len(series) == 1:
        axes = [axes]
    for ax, (name, vals, lo, hi) in zip(axes, series):
        bins = np.logspace(np.log10(max(1.0, min(vals.min(), lo) * 0.7)),
                           np.log10(max(vals.max(), hi, period) * 1.4), 34)
        ax.hist(vals, bins=bins, color="#457b9d", alpha=0.75,
                edgecolor="white", lw=0.4)
        ax.set_xscale("log")
        ax.axvline(lo, color="#2a9d8f", ls=":", lw=1.6, label=f"init lo {lo:g}")
        ax.axvline(hi, color="#e63946", ls=":", lw=1.6, label=f"init hi {hi:g}")
        ax.axvline(period, color="k", ls="--", lw=1.4,
                   label=f"CPG period {period:.0f}")
        ax.set_title(f"{name}  (n={vals.size})", fontsize=9)
        ax.set_xlabel("tau (steps)", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.25)
        frac = float(np.mean(vals > 0.95 * hi))
        if frac > 0.15:
            ax.text(0.02, 0.95, f"{frac*100:.0f}% pinned at init hi",
                    transform=ax.transAxes, fontsize=7, color="#e63946",
                    va="top")
    fig.suptitle("Learned membrane time constants", fontsize=10,
                 fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir, "tau_distributions.png", dpi)


def plot_membranes(mems, state_names, gait_name, out_dir, dpi,
                   n_units=6, n_show=1200, seed=0):
    """Sub-network membrane traces for a few sampled units per group. Membranes, not spikes,
    on purpose: exact hidden rasters are not recoverable from a checkpoint (see the module
    docstring).
    """
    if not mems:
        return
    stacks = [np.stack([m[k] for m in mems[:n_show]]) for k in range(len(mems[0]))]
    rng = np.random.default_rng(seed)

    # Skip the timing membranes: tiny, and already covered exactly by the rasters. This plot is about
    # the sub-networks, whose spikes are not recoverable (see the module docstring).
    rows = [(k, nm) for k, nm in enumerate(state_names)
            if nm not in ("mem_timing", "since_upd")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 2.4 * len(rows)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (k, nm) in zip(axes, rows):
        A = stacks[k]                          # (T, G, Hg) or (T, H)
        if A.ndim == 2:
            A = A[:, None, :]
        G, Hg = A.shape[1], A.shape[2]
        for g in range(G):
            pick = rng.choice(Hg, size=min(n_units, Hg), replace=False)
            for u in pick:
                ax.plot(A[:, g, u], lw=0.7, alpha=0.55,
                        color=TIMING_PALETTE[g % len(TIMING_PALETTE)])
        ax.set_ylabel(nm, fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_title(f"{nm}: {min(n_units, Hg)} sampled units per group "
                     f"(color = group)", fontsize=8)
    axes[-1].set_xlabel("timestep (after warm-up)")
    fig.suptitle(f"{gait_name} — sub-network membranes "
                 f"(exact; hidden spikes are not recoverable from a checkpoint)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir, f"membranes_{gait_name}.png", dpi)


# ═══════════════════════════════════════════════════════════════════
# 6.  Main
# ═══════════════════════════════════════════════════════════════════

def _transition_pairs(spec, names):
    """
    Parse --transitions into a list of (from_index, to_index).

    "0>1,0>4"        explicit index pairs
    "tripod>ripple"  names, resolved against the config's gait list
    "chain"          0>1, 1>2, ... -- every consecutive pair
    "all"            every ordered pair (n*(n-1) plots; use with few gaits)
    """
    n = len(names)
    if spec == "chain":
        return [(i, i + 1) for i in range(n - 1)]
    if spec == "all":
        return [(a, b) for a in range(n) for b in range(n) if a != b]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ">" not in part:
            raise SystemExit(f"--transitions entry {part!r} is not 'from>to'")
        a, b = (x.strip() for x in part.split(">", 1))
        idx = []
        for tok in (a, b):
            if tok.isdigit():
                i = int(tok)
                if not 0 <= i < n:
                    raise SystemExit(f"--transitions index {i} out of range "
                                     f"0..{n - 1}")
                idx.append(i)
            elif tok in names:
                idx.append(names.index(tok))
            else:
                raise SystemExit(f"--transitions: unknown gait {tok!r}; "
                                 f"config has {names}")
        out.append(tuple(idx))
    return out


def _build_parser():
    ap = argparse.ArgumentParser(
        description="Visualise the timing layer of a trained CPG-SNN run.")
    ap.add_argument("--model_dir", type=str, default="",
                    help="Directory holding the checkpoint and config, "
                         "resolved as outputs/<model_dir> — e.g. --model_dir "
                         "test1 reads from outputs/test1. Default '' means "
                         "outputs/ itself.")
    ap.add_argument("--ckpt",      type=str, default="best_model.pt",
                    help="Checkpoint in the run's model/ folder: best_model.pt "
                         "(default, lowest val loss) or a snapshot such as epoch100.pt "
                         "(--save_model_every); a wrong name lists the ones that "
                         "exist.")
    ap.add_argument("--cfg",       type=str, default="cpg_lif_snn_config.json")
    ap.add_argument("--out_dir",   type=str, default=None,
                    help="Where plots and timing_summary.json are written, resolved as "
                         "outputs/<out_dir>; default = the model dir itself, so "
                         "figures join that run's category folders (a non-best --ckpt "
                         "defaults to <model_dir>/viz_<ckpt>/).")
    ap.add_argument("--gaits_dir", type=str, default="../gaits",
                    help="Folder of {name}.csv gait tables, resolved as "
                         "this_file_dir/<gaits_dir> — same default as "
                         "train.py's --gaits_dir.")
    ap.add_argument("--drive_metric", type=str, default="auto",
                    choices=["auto", "product", "gain", "spikes"],
                    help="[--drive_source cpg / cpg_lif] Which drive representation "
                         "figures and printouts use. 'auto' (default) follows the "
                         "checkpoint: 'product' (|s_gate|*||u||, with fanin90) under "
                         "cpg, 'spikes' (exact per-unit spike counts) under cpg_lif. "
                         "'gain' plots |s_gate|, the penalized quantity under cpg_lif "
                         "but MEANINGLESS under cpg, where u can rescale it freely.")
    ap.add_argument("--gaits",     type=str, nargs="*", default=None,
                    help="Gait names to plot (default: all in the config).")
    ap.add_argument("--recon", type=int, default=1,
                    help="1 = also regenerate recon_<gait>.png and transition_*.png "
                         "(how an already-trained checkpoint gets re-plotted with a "
                         "different --recon_cycles); train.py passes 0 since it just "
                         "made them.")
    ap.add_argument("--recon_cycles", type=float, default=4.0,
                    help="CPG cycles per reconstruction plot. In cycles rather "
                         "than timesteps so the plot stays readable whether "
                         "the period is 352 (real CPG) or 120 (--fake_cpg).")
    ap.add_argument("--transitions", type=str, default="0>1",
                    help="Which gait transitions to plot: 'from>to' pairs by "
                         "index or name, comma-separated ('0>1,tripod>ripple'); "
                         "'chain' for every consecutive pair; 'all' for every "
                         "ordered pair.")
    ap.add_argument("--n_cycles",   type=float, default=3.0,
                    help="Gait cycles shown on the time-axis figures.")
    ap.add_argument("--warm_cycles", type=float, default=4.0,
                    help="Cycles of free-run discarded before recording, so "
                         "the plots show settled behavior rather than the "
                         "zero-state transient.")
    ap.add_argument("--fold_cycles", type=float, default=40.0,
                    help="Cycles accumulated for the phase-fold histograms "
                         "and the alignment statistics. More is better here — "
                         "it only costs CPU and it tightens R.")
    ap.add_argument("--scores_only", action="store_true",
                    help="Compute ONLY the alignment scores and the per-row timing "
                         "summary: replays the CPG and the spike trains the scores "
                         "need, skipping predictions, membranes, figures and the "
                         "reconstruction. Writes timing_summary*.json and "
                         "alignment_scores*.csv.")
    ap.add_argument("--align_vel_tau", type=float, default=_ALIGN_VEL_TAU,
                    help="Credit horizon (steps) of the VELOCITY alignment "
                         "score. Default matches train.py's.")
    ap.add_argument("--align_cor_tau", type=float, default=_ALIGN_COR_TAU,
                    help="Credit horizon (steps) of the CORNER alignment "
                         "score. Default matches train.py's.")
    ap.add_argument("--no_pred",    action="store_true",
                    help="Skip the model forward pass (CPG + GT + timing only).")
    ap.add_argument("--no_membranes", action="store_true")
    ap.add_argument("--dpi",  type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main():
    args = _build_parser().parse_args()
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = outputs_path(this_file_dir, args.model_dir)
    out_dir = (None if args.out_dir is None
               else outputs_path(this_file_dir, args.out_dir))
    run_visualization(model_dir, out_dir, args)


def default_args(**overrides):
    """The CLI defaults as a plain namespace, so train.py can call run_visualization
    without argparse.
    """
    a = _build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def run_visualization(model_dir, out_dir=None, args=None):
    """Everything the CLI does, callable directly. `out_dir=None` means the run directory
    itself (what train.py passes too): figures land in that run's category folders (see
    OUT_ROUTES in train.py). The checkpoint and config are re-read from disk rather than
    taking a live model, which doubles as an end-of-run check that the saved artifacts load
    and run.
    """
    args = args if args is not None else default_args()
    model_dir = Path(model_dir)
    # Default is the run directory ITSELF (out_path routes each figure into its category folder, so a
    # subfolder would only bury them), so standalone and train.py-called runs write to the same place.
    # A checkpoint other than best_model.pt must never overwrite what train.py wrote for it, so unless
    # --out_dir is given it goes to viz_<checkpoint>/ and the two data files carry the checkpoint name.
    ckpt_stem = Path(args.ckpt).stem
    is_best = (Path(args.ckpt).name == "best_model.pt")
    data_sfx = "" if is_best else f"_{ckpt_stem}"
    if out_dir is None:
        out_dir = (Path(model_dir) if is_best
                   else Path(model_dir) / f"viz_{ckpt_stem}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.scores_only:
        # Everything the scores do not need, off: predictions are the one expensive step (a network
        # forward per gait) and nothing in the scores reads them.
        args.no_pred = True
        args.no_membranes = True
        args.recon = 0
        print("  --scores_only: skipping predictions, membranes, figures and "
              "the reconstruction.")
    if not is_best:
        print(f"  checkpoint {args.ckpt}: writing to {out_dir} "
              f"(data files suffixed '{data_sfx}')")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\nModel  : {model_dir}\nOutput : "
          f"{out_dir.resolve()}\n")

    # ── 1. load ─────────────────────────────────────────────────
    print("[1/5] Loading checkpoint + config ...")
    cfg, model, arch = load_run(model_dir, args.ckpt, args.cfg, device)

    gait_files = cfg_get(cfg, "gait_files")
    if gait_files is None:
        # Older config without "gait_files": assume the standard set for its species, as train.py's own
        # fallback does when --gaits isn't given.
        n_cpg = int(cfg_get(cfg, "n_cpg_neurons", 4))
        gait_files = GAIT_FILES_BY_N.get(n_cpg)
        if gait_files is None:
            raise ValueError(
                f"Config has no 'gait_files' and n_cpg_neurons={n_cpg} has "
                f"no standard gait set (know {sorted(GAIT_FILES_BY_N)}).")
        print(f"  NOTE: config has no 'gait_files' key — assuming the "
              f"standard n_cpg_neurons={n_cpg} set: {gait_files}")
    # Relative to THIS file, not model_dir, so it resolves the same from the CLI or from train.py.
    gaits_dir = Path(os.path.dirname(os.path.abspath(__file__)),
                     args.gaits_dir)
    # Anatomical leg layout for the timing-alignment grid; falls back to the network's own grouping for
    # configs predating leg_cols (exact whenever n_timing == n_legs, the only case then).
    cfg_leg_cols = cfg_get(cfg, "leg_cols")

    gait_tables_orig, all_names = load_gait_tables(gait_files, gaits_dir)
    print(f"  Loaded {len(all_names)} gait CSV(s) from {gaits_dir.resolve()}: "
          f"{all_names}")
    # Fallbacks for configs without leg_cols: the network's grouping when it has one (DenseSNN has no
    # group_cols), otherwise the layout rule training uses.
    _mgroups = getattr(model, "group_cols", None)
    if cfg_leg_cols:
        leg_cols = [list(c) for c in cfg_leg_cols]
    elif _mgroups is not None:
        leg_cols = [list(c) for c in _mgroups]
    else:
        leg_cols = [list(c) for c in default_leg_layout(
            int(cfg_get(cfg, "n_cpg_neurons", 0) or 0),
            int(gait_tables_orig[0].shape[1]))[1]]
    print(f"  Alignment grid: {len(leg_cols)} legs x {len(leg_cols[0])} "
          f"joints/leg"
          + ("" if cfg_leg_cols else "  (leg_cols absent from config — "
                                     "using the network grouping)"))

    # ── timing units vs sub-networks ─────────────────────────────
    # Two different counts (--timing_shape vs --decoder_shape); conflating them breaks per-timing-unit
    # loops (at per_joint timing + per_leg decoders that is 18 lookups into a 6-entry list).
    #   group_cols[g]  : output columns sub-network g EMITS
    #   timing_cols[t] : output columns timing unit t is RESPONSIBLE FOR (what its phase should align to)
    #   timing_map[g]  : which timing units feed sub-network g
    # timing_cols falls back to group_cols for configs predating --timing_shape (one unit per
    # sub-network). group_cols is empty under --arch dense, which has no sub-networks and never reads these.
    group_cols  = [list(g) for g in (_mgroups or [])]
    cfg_tcols   = cfg_get(cfg, "timing_cols")
    timing_cols = ([list(c) for c in cfg_tcols] if cfg_tcols
                   else [list(g) for g in group_cols])
    timing_map  = cfg_get(cfg, "timing_map")
    if arch == "timing_grouped" and not getattr(model, "uses_timing", True):
        # cpg / cpg_lif: no timing layer. n_timing is the CPG channel count there, and the
        # config's timing_cols (from --timing_shape) is recorded but unused, so comparing the two
        # is meaningless; with --n_cpg_neurons 18 and per_joint they only matched by coincidence.
        print(f"  Structure: no timing layer (--drive_source "
              f"{cfg_get(cfg, 'drive_source', 'cpg')}): {model.n_neurons} CPG channel(s) -> "
              f"{len(group_cols)} sub-network(s) ({cfg_get(cfg, 'decoder_shape', 'per_leg')})")
    elif arch == "timing_grouped":
        n_t = int(getattr(model, "n_timing", len(timing_cols)))
        if len(timing_cols) != n_t:
            raise ValueError(
                f"timing_cols has {len(timing_cols)} entries but the model "
                f"has {n_t} timing units. The config's timing_cols and "
                f"checkpoint disagree — treat every figure as suspect.")
        print(f"  Structure: {n_t} timing unit(s) "
              f"({cfg_get(cfg, 'timing_shape', 'per_leg')}) -> "
              f"{len(group_cols)} sub-network(s) "
              f"({cfg_get(cfg, 'decoder_shape', 'per_leg')}), fan-in K="
              f"{len(timing_map[0]) if timing_map else 1}"
              + ("" if cfg_tcols else "  (timing_cols absent from config — "
                                      "assuming one unit per sub-network)"))

    tgt_range  = (float(cfg_get(cfg, "global_min", -124.0)),
                  float(cfg_get(cfg, "global_max", 124.0)))
    phase_zero = float(cfg_get(cfg, "phase_zero", 0.0))
    target_rows = int(cfg_get(cfg, "target_rows",
                              max(t.shape[0] for t in gait_tables_orig)))
    gait_tables, _ = upsample_gait_tables(gait_tables_orig, all_names,
                                          target_rows, verbose=False)

    if args.gaits:
        unknown = [g for g in args.gaits if g not in all_names]
        if unknown:
            raise SystemExit(f"Unknown gait(s) {unknown}; config has {all_names}")
        sel = [(all_names.index(g), g) for g in args.gaits]
    else:
        sel = list(enumerate(all_names))

    if arch != "timing_grouped":
        print(f"\n  arch={arch} has NO timing layer: emitting the CPG, gait-table and "
              f"tau figures only (timing_summary / alignment_scores are written with "
              f"no rows).")

    # ── 2. replay CPG ───────────────────────────────────────────
    print("\n[2/5] Replaying the CPG from config ...")
    period_hint = float(cfg_get(cfg, "cpg_period_steps", 254.0))
    n_steps = int(max(args.fold_cycles, args.n_cycles + args.warm_cycles)
                  * period_hint) + 400
    spikes  = replay_cpg(cfg, n_steps)
    onsets, period, phase, burst_thr = cpg_phase(spikes)
    print(f"  measured period = {period:.1f} steps  "
          f"(config said {period_hint:.1f})  burst ISI thr = {burst_thr:.1f}")
    if abs(period - period_hint) > 0.05 * period_hint:
        print("  WARNING: replayed period differs from the config by >5%. "
              "The CPG parameters in the config may not match the run.")

    warm = int(args.warm_cycles * period)
    # Window for the time-axis figures: start after warm-up, on a burst onset
    # so cycle boundaries land cleanly.
    cand = onsets[onsets >= warm]
    t_lo = int(cand[0]) if len(cand) else warm
    t_hi = int(min(len(spikes), t_lo + args.n_cycles * period))

    # ── 3. per-gait probes + figures ────────────────────────────
    print("\n[3/5] Per-gait figures ...")
    # Computed once before the loop: drive_strength returns all gaits at once and is a pure parameter
    # read. None under --drive_source timing.
    drive_all, drive_spk_all, drive_metric = None, None, "product"
    if arch == "timing_grouped" and not getattr(model, "uses_timing", True):
        is_lif = bool(getattr(model, "drive_lif", False))
        drive_metric = (("spikes" if is_lif else "product")
                        if args.drive_metric == "auto" else args.drive_metric)
        if drive_metric == "spikes" and not is_lif:
            raise SystemExit(
                "--drive_metric spikes needs --drive_source cpg_lif; this "
                "checkpoint has no drive LIF layer to count spikes from.")
        print(f"  drive metric: {drive_metric}"
              + ("  (|s_gate|*||u||, fanin90 by a 90% cutoff)"
                 if drive_metric == "product" else
                 "  (exact per-unit spike counts, no cutoff)"
                 if drive_metric == "spikes" else
                 "  (|s_gate| alone -- the penaliized quantity under cpg_lif)"))

        if drive_metric == "spikes":
            # Shares come from SPIKE COUNTS (cpg_lif): "channel c delivers nothing" is the exact statement
            # "unit (g,c) emitted zero spikes". The WHOLE replay, not a fixed prefix: the window must cover
            # [t_lo, t_hi) (a spikes[:6*period] prefix left the last plotted period blank at the defaults), and
            # replaying everything gives the LIF its real membrane state at t_lo.
            L = spikes.shape[0]
            xs = torch.as_tensor(spikes[:L], dtype=torch.float32,
                                 device=device).unsqueeze(1)
            cnts, trains = [], []
            for gi in range(len(all_names)):
                gt = torch.full((L, 1), gi, dtype=torch.long, device=device)
                sp = model.drive_spikes_only(xs, gt)         # (L,1,G,C)
                tr = sp[:, 0].cpu().numpy()                  # (L,G,C)
                trains.append(tr)
                # Counted over the plotted window only, so the shares that pick and shade the rugs describe the
                # steps drawn.
                cnts.append(tr[t_lo:t_hi].sum(axis=0))
            D = np.stack(cnts).astype(np.float64)            # (n_gaits,G,C)
            drive_spk_all = trains
        elif drive_metric == "gain":
            W = model.s_gate.weight.detach()[:len(all_names)]
            D = W.abs().reshape(len(all_names), model.G,
                                model.n_neurons).cpu().numpy()
        else:
            D = model.drive_strength().detach().cpu().numpy()
        drive_all = D / np.maximum(D.sum(axis=2, keepdims=True), 1e-12)

    summary, plotted = {}, []
    align_rows = []
    by_joint = {}             # gait -> (row trains over the plotted window, cols per row)
    for gi, gname in sel:
        print(f"  {gname}:")
        gt = gt_degrees(gait_tables, phase, gi, phase_zero)

        pred, mems = None, []
        if not args.no_pred:
            # Only free-run as far as the plotted window: the phase-fold statistics need timing spikes
            # (cheap), not predictions.
            p, mems = predict_and_membranes(
                model, spikes[:t_hi], gi, device, warm, tgt_range,
                keep_membranes=not args.no_membranes)
            pred = np.full(gt.shape, np.nan, dtype=np.float64)
            pred[warm:warm + len(p)] = p
            rmse = np.sqrt(np.nanmean(
                (pred[t_lo:t_hi] - gt[t_lo:t_hi]) ** 2, axis=0))
            print(f"    per-column RMSE (°): " +
                  "  ".join(f"c{c}={rmse[c]:.2f}" for c in range(len(rmse))))

        # Membranes are not timing-specific (every arch has them), so this runs before the timing-only bail.
        if mems and not args.no_membranes:
            plot_membranes(mems, [n.replace("_in", "")
                                  for n in model.state_names_in],
                           gname, out_dir, args.dpi, seed=args.seed)

        if arch != "timing_grouped":
            continue

        # G here is the TIMING-unit count (raster lanes); everything below indexes timing_cols, which is
        # this long by construction.
        tspk = timing_raster(model, spikes, gi, device)
        G    = tspk.shape[1]

        # cpg / cpg_lif: a panel belongs to a SUB-NETWORK fed by a weighted mix of CPG channels, so the row
        # map becomes group_cols and the per-panel overlay becomes that sub-network's drive shares
        # (otherwise channel N would be paired with sub-network N, which only type-checks because n_cpg
        # happens to equal G).
        if drive_all is not None:
            rows_map = group_cols
            d_share  = drive_all[gi] if gi < len(drive_all) else None
        else:
            rows_map, d_share = timing_cols, None

        # Under cpg_lif the rug must show the DRIVE LIF's spikes, not the raw CPG channel's (the unit thins
        # and delays its burst). tspk_g is (L, G, C) there, (L, lanes) otherwise.
        tspk_g = tspk
        if drive_spk_all is not None and gi < len(drive_spk_all):
            tr = drive_spk_all[gi]                            # (Ld, G, C)
            Ld = min(tr.shape[0], tspk.shape[0])
            tspk_g = np.zeros((tspk.shape[0],) + tr.shape[1:], dtype=tr.dtype)
            tspk_g[:Ld] = tr[:Ld]

        # frac=None under the spikes metric: draw every channel that fires, not a 90% subset.
        d_frac = None if drive_metric == "spikes" else 0.90
        if not args.scores_only:
            plot_alignment(spikes, tspk_g, gt, pred, phase, onsets, period,
                           burst_thr, rows_map, leg_cols, gname, out_dir,
                           args.dpi, t_lo, t_hi, drive_share=d_share,
                           drive_frac=d_frac)
            plot_phase_fold(tspk_g, phase, gait_tables, gi, rows_map, gname,
                            phase_zero, out_dir, args.dpi,
                            drive_share=d_share, drive_frac=d_frac)

        # statistics over the full fold window
        ok    = np.isfinite(phase)
        ncyc  = max(np.isfinite(phase).sum() / period, 1e-9)
        tbl   = gait_tables[gi]
        x_tbl = ((np.arange(tbl.shape[0]) / tbl.shape[0]) - phase_zero) % 1.0
        order = np.argsort(x_tbl)
        rows  = []
        # Under cpg_lif the per-row train is the COLLAPSED one, the OR over the sub-network's C drive units:
        # valid because the units are mutually exclusive (one channel each, zero reset, CPG channels never
        # coincide), and it is exactly the input-event train the sub-network receives (comparable to the
        # old timing layer's spk/cyc).
        coll = None
        if tspk_g.ndim == 3:
            coll = (tspk_g.sum(axis=2) > 0).astype(np.float32)   # (L, G)
            if float(tspk_g.sum(axis=2).max()) > 1.0 + 1e-6:
                print("    WARNING drive units of one sub-network fired on "
                      "the same timestep; the collapsed train is not valid "
                      "(needs zero reset).")
        # One input train per row, by mode: cpg_lif = the collapsed drive-LIF train; cpg = the CPG channels
        # weighted by that sub-network's drive share (not raw channel j); timing = the unit's own spikes.
        # Target columns come from rows_map (group_cols under the cpg modes), not timing_cols: they coincide
        # only at per_joint.
        is_drive = drive_all is not None
        if coll is not None:
            row_trains = coll                                    # (L, G)
        elif is_drive and d_share is not None:
            row_trains = tspk @ d_share.T                        # (L, G)
        else:
            row_trains = tspk                                    # (L, T)
        n_row = row_trains.shape[1]
        cols_of = rows_map if is_drive else timing_cols
        un = "g" if is_drive else "T"
        # Alignment is scored over the plotted window [t_lo, t_hi): whole cycles of one gait (the score's
        # circular wrap needs that), past warm-up.
        aw = slice(t_lo, t_hi)
        ph_w = np.asarray(phase[aw], dtype=np.float64)
        vel_l, cor_l = [], []
        for j in range(n_row):
            tr_j = row_trains[:, j]
            m = (tr_j > 0) & ok
            mu, R_ = circular_stats(phase[m], tr_j[m])
            f_ph   = fundamental_phase(tbl[order, cols_of[j][0]])
            res    = circ_diff(mu, f_ph)
            feeds  = ([g for g, mm in enumerate(timing_map) if j in mm]
                      if (timing_map and not is_drive) else [j])
            # Both scores take derivatives from the SMOOTH table at the exact phase (joint_demand), never from
            # `gt`, a staircase lookup that would make the corner score noise.
            a_vel, a_cor = alignment_pair(
                tr_j[aw], gait_tables[gi], cols_of[j], ph_w, phase_zero,
                ok[aw], args.align_vel_tau, args.align_cor_tau)
            vel_l.append(a_vel); cor_l.append(a_cor)
            rows.append({
                "timing_neuron":   j,
                "row_kind":        "sub_network" if is_drive else "timing_unit",
                "cols":            list(cols_of[j]),
                "feeds_subnets":   feeds,
                "rate_per_cycle":  float(tr_j.sum() / ncyc),
                "mean_phase":      None if not np.isfinite(mu) else float(mu),
                "R":               float(R_),
                "leg_fundamental_phase": None if not np.isfinite(f_ph) else float(f_ph),
                "residual_cycles": None if not np.isfinite(res) else float(res),
                "residual_steps":  None if not np.isfinite(res) else float(res * period),
                "alignment_velocity": None if not np.isfinite(a_vel) else float(a_vel),
                "alignment_corner":   None if not np.isfinite(a_cor) else float(a_cor),
                "dead":            bool(tr_j.sum() == 0),
            })
            tag = "  <-- DEAD" if rows[-1]["dead"] else ""
            print(f"    {un}{j} (cols {list(cols_of[j])}): "
                  f"rate={rows[-1]['rate_per_cycle']:6.2f}/cyc  "
                  f"phase={mu:.3f}  R={R_:.2f}  "
                  f"residual={res:+.3f} cyc ({res * period:+.0f} steps){tag}")
        # Collected, printed once in an ALIGNMENT section after every gait.
        align_rows.append((gname, vel_l, cor_l))
        by_joint[gname] = (np.asarray(row_trains[aw], dtype=float), [list(c) for c in cols_of])
        summary[gname] = rows
        plotted.append(gname)

    # ── alignment section: once, after every gait, grouped by statistic ──
    if align_rows:
        print()
        for ln in alignment_section(align_rows, indent="  "):
            print(ln)

    # ── 4. cross-gait figures ───────────────────────────────────
    routing = drive_gates = None
    if not args.scores_only:
        print("\n[4/5] Cross-gait figures ...")
    if arch == "timing_grouped" and summary and not args.scores_only:
        plot_joint_gait_grid(by_joint, leg_cols, period, out_dir, args.dpi)
        # Column count from the summary's own rows: timing units, or sub-networks under the cpg
        # modes (model.n_timing is the CPG channel count there, which dropped sub-networks).
        plot_alignment_summary(summary, plotted, len(summary[plotted[0]]),
                               out_dir, args.dpi)
        routing = plot_routing(model, all_names, device, period,
                               out_dir, args.dpi)
        # Mutually exclusive with plot_routing: each returns None in the
        # other's mode, so exactly one of the two figures is produced.
        drive_gates = plot_drive_gates(
            model, plotted, leg_cols, out_dir, args.dpi,
            gait_idx=[i for i, _ in sel], drive_all=drive_all,
            metric=drive_metric, report_ctx=(spikes, device, period))
        metrics_csv = model_dir / "metrics.csv"
        if metrics_csv.exists():
            # Which per-unit history exists depends on the drive mode; each plotter no-ops with a printed note
            # if its columns are absent, so calling both is safe.
            if getattr(model, "uses_timing", True):
                plot_timing_rate_history(metrics_csv, timing_cols, leg_cols,
                                         plotted, out_dir, args.dpi)
            else:
                plot_drive_fanin_history(metrics_csv, group_cols, leg_cols,
                                         plotted, out_dir, args.dpi)
            # Penalty-ramp strip: hold defaults to 0 for configs predating --lambda_hold.
            warm_ep = cfg_get(cfg, "lambda_warmup")
            ramp = (float(cfg_get(cfg, "lambda_hold", 0)), float(warm_ep)) if warm_ep else None
            plot_alignment_history(metrics_csv, plotted, out_dir, args.dpi, ramp)
        else:
            print(f"  [unit history] {metrics_csv} not found, skipping")
    if not args.scores_only:
        plot_taus(model, period, cfg, out_dir, args.dpi)

    # ── 4b. reconstruction + transitions ────────────────────────
    # The figures train.py makes at the end of a run, regenerated so an already-trained model can be
    # re-plotted (the only way to apply a new --recon_cycles to an old checkpoint). targets/valid come
    # from train.py's own build_targets, so they are built exactly as in training.
    if args.recon:
        print(f"\n[4b/5] Reconstruction & transitions "
              f"({args.recon_cycles:g} cycles per plot) ...")
        targets, valid, _ = build_targets(phase, gait_tables, phase_zero)
        n_joints = int(cfg_get(cfg, "n_joints", gait_tables[0].shape[1]))
        T_all = spikes.shape[0]
        # Mirrors main(): evaluate clear of the train/val split and of the
        # first burst onset.
        t_split = int(T_all * (1.0 - float(cfg_get(cfg, "val_frac", 0.15))))
        t_eval  = max(t_split + 800, int(onsets[0]) + 800)
        if t_eval + int(2.5 * args.recon_cycles * period) > T_all:
            t_eval = max(int(onsets[0]) + 800,
                         T_all - int(2.5 * args.recon_cycles * period))
        need = int(2.5 * args.recon_cycles * period)
        print(f"  t_eval={t_eval} of {T_all} steps "
              f"(t_split={t_split}, first onset={int(onsets[0])}, "
              f"window needs {need})")
        if t_eval + need > T_all:
            # Better a loud warning than plots that quietly run off the end of the replay (--n_steps lengthens
            # it; --recon_cycles shortens the window).
            print(f"  WARNING: the replayed CPG is too short for "
                  f"{args.recon_cycles:g} cycles at period {period:.0f} — "
                  f"need {t_eval + need} steps, have {T_all}. Plots will be "
                  f"truncated. Raise --n_steps or lower --recon_cycles.")

        plot_reconstruction(model, spikes, targets, valid, device,
                            out_dir, tgt_range, t0=t_eval,
                            gait_names=all_names, leg_cols=leg_cols,
                            n_joints=n_joints, period=period,
                            n_cycles=args.recon_cycles)

        pairs = _transition_pairs(args.transitions, all_names)
        print(f"  transitions: " + ", ".join(
            f"{all_names[a]}->{all_names[b]}" for a, b in pairs))
        for a, b in pairs:
            plot_transition(model, spikes, targets, device, out_dir,
                            tgt_range, t0=t_eval, gait_names=all_names,
                            leg_cols=leg_cols, g_from=a, g_to=b,
                            warm=int(round(2.0 * period)),
                            n_steps=int(round(2.0 * args.recon_cycles * period)),
                            switch_at=int(round(args.recon_cycles * period)))

    # ── 5. dump ─────────────────────────────────────────────────
    print(f"\n[5/5] Writing timing_summary{data_sfx}.json and "
          f"alignment_scores{data_sfx}.csv ...")
    blob = {
        "source": {"model_dir": str(model_dir), "ckpt": args.ckpt,
                   "cfg": args.cfg, "arch": arch},
        "cpg": {"period_measured": float(period),
                "period_from_config": period_hint,
                "burst_isi_threshold": float(burst_thr),
                "n_steps": int(n_steps), "warm_steps": int(warm)},
        "window": {"t_lo": int(t_lo), "t_hi": int(t_hi)},
        "alignment": summary,
        "learned_routing_argmax": routing,
        # Populated only under cpg / cpg_lif; None otherwise.
        "drive_gates": drive_gates,
        "alignment_score_note": (
            "alignment_velocity / alignment_corner: how well the row's input "
            "spikes coincide with the joint's speed |dtheta/dt| / curvature "
            "|dtheta^2/dt^2|, both taken from the SMOOTH gait table at each "
            "step's exact phase (not the staircase per-step target). Per "
            "step, the demand at that step, or decayed credit for higher "
            "demand shortly AHEAD (horizons in alignment_horizons), never "
            "for demand behind it; averaged over the spikes and "
            "chance-anchored: 0 = random placement, 1 = every spike at the "
            "peak, <0 = spikes where demand is low (not bounded at -1; only "
            "compare positive values across joints). Scored over the "
            "plotted window [t_lo, t_hi)."),
        "residual_note": ("residual = timing-neuron circular mean phase minus "
                          "the phase of the first Fourier component of that "
                          "leg's first gait-table column; a consistent "
                          "reference, not a biomechanical footfall event"),
    }
    blob["checkpoint"] = str(args.ckpt)
    blob["alignment_horizons"] = {"velocity": float(args.align_vel_tau),
                                  "corner": float(args.align_cor_tau)}
    p = out_path(out_dir, f"timing_summary{data_sfx}.json")
    p.write_text(json.dumps(blob, indent=2))
    print(f"    [saved] {p}")

    # The alignment numbers alone as a flat table (one row per gait and row): easy to open or stack
    # across runs and checkpoints.
    import csv as _csv
    pc = out_path(out_dir, f"alignment_scores{data_sfx}.csv")
    with open(pc, "w", newline="") as fh:
        wr = _csv.writer(fh)
        wr.writerow(["checkpoint", "gait", "row_kind", "row", "cols",
                     "align_velocity", "align_corner", "rate_per_cycle",
                     "dead"])
        for gname in plotted:
            for r in summary.get(gname, []):
                fmt = lambda v: "" if v is None else f"{v:.4f}"
                wr.writerow([args.ckpt, gname, r.get("row_kind", ""),
                             r["timing_neuron"],
                             " ".join(str(c) for c in r["cols"]),
                             fmt(r.get("alignment_velocity")),
                             fmt(r.get("alignment_corner")),
                             f"{r['rate_per_cycle']:.3f}", int(r["dead"])])
    print(f"    [saved] {pc}")
    print(f"\nDone — {out_dir.resolve()}")


if __name__ == "__main__":
    main()