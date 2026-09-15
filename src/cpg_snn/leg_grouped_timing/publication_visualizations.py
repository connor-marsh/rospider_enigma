"""
Publication figures for the CPG-SNN controller
==============================================
Vector PDF, larger type, and three figures whose LAYOUT differs from the
diagnostic ones in visualize_timing.py -- not merely their styling:

  fig_reconstruction   6 joints (rows) x 4 gaits (columns), 3 CPG cycles each
  fig_transitions      2 joints over one continuous 12-cycle axis with a gait
                       change every 2 cycles (5 transitions, 6 gaits)
  fig_timing_alignment 6 legs (rows), each leg's 3 joints overlaid and scaled
                       to their own range, under a shared CPG and timing
                       raster with burst onsets aligned across every row
  fig_training_curves  loss against epoch on a log axis with the learning-rate
                       schedule beneath, read from the run's metrics.csv

Kept in its own module rather than behind a flag in visualize_timing.py
because none of these share layout code with their diagnostic counterparts,
so a flag would have meant `if publication: <entirely different body>`. It
also keeps figure churn during paper revisions out of the file train.py runs
automatically after every training run.

Runnable directly, or via `visualize_timing.py --publication 1`:

    python publication_visualizations.py --model_dir test1
    python publication_visualizations.py --model_dir test1 --pub_cycles 4
"""

import argparse
import contextlib
import csv
import os
from pathlib import Path

import numpy as np
import torch

from train import (
    cfg_get, load_run, outputs_path, GAIT_FILES_BY_N, load_gait_tables,
    upsample_gait_tables, default_leg_layout, joint_type_names,
)

# Figure content is fixed by the paper, not discovered from the run, so the
# selections live here as named defaults and are overridable from the CLI.
PUB_RECON_GAITS = ["tripod", "tripod_right", "ripple_backwards", "ripple_left"]
PUB_TRANSITION_SEQ = ["tripod", "tripod_left", "tripod_backwards",
                      "ripple", "ripple_right", "ripple_backwards"]
# (leg, joint-within-leg) pairs, in the row order the figure wants.
PUB_RECON_JOINTS = [(0, 0), (3, 0), (0, 1), (3, 1), (0, 2), (3, 2)]
PUB_TRANSITION_JOINTS = [(0, 0), (3, 0)]

# Direction suffixes already present in a gait name. A name carrying none of
# them is the forward variant, which is only implicit in the filename, so it
# gets spelled out rather than left ambiguous next to its siblings.
_DIRECTIONS = ("right", "left", "backwards", "forwards")


def gait_label(name):
    """'tripod' -> 'Tripod forwards';  'ripple_backwards' -> 'Ripple backwards'."""
    parts = name.split("_")
    if parts[-1] not in _DIRECTIONS:
        parts.append("forwards")
    txt = " ".join(parts)
    return txt[:1].upper() + txt[1:]


def joint_label(leg, k, n_per_leg):
    """'Leg 0 - Coxa'. Hyphenated so the leg and the joint read as separate."""
    nm = joint_type_names(n_per_leg)[k]
    return f"Leg {leg} - {nm[:1].upper() + nm[1:]}"


# Joint traces get their own strong palette, kept clear of the CPG raster's
# turbo and the timing raster's TIMING_COLORS so the three legends cannot be
# confused.
JOINT_COLORS = ["#0b5fa5", "#e08214", "#1a7a3c"]
# Timing-neuron colours. Hand-picked rather than sampled from tab20, half of
# whose entries are pale tints that vanish as 1 px rug ticks. Consecutive
# entries are far apart in hue because the neurons sharing a leg row are
# consecutive indices, so that is where contrast actually has to hold.
TIMING_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080",
    "#f032e6", "#9a6324", "#800000", "#808000", "#000075", "#b8860b",
    "#1b9e77", "#d95f02", "#7570b3", "#66a61e", "#a6761d", "#525252",
]
GT_C   = "#1f3a5f"
PRED_C = "#c1272d"
SPK_C  = "#2a9d8f"
GRID_C = "#b0b7c3"


@contextlib.contextmanager
def publication_style(scale=1.0):
    """
    Larger type and cleaner axes, restored on exit.

    Set through a context manager rather than a global style file so that
    importing this module cannot change how visualize_timing's diagnostic
    figures come out. Sizes are absolute points at the figure sizes used
    below, so a figure scaled down to one journal column stays legible.
    """
    import matplotlib as mpl
    prev = dict(mpl.rcParams)
    s = float(scale)
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size":          13 * s,
        "axes.titlesize":     15 * s,
        "axes.titleweight":   "bold",
        "axes.labelsize":     15 * s,
        # Axis labels stay regular weight: bold is reserved for titles and
        # captions, so that emphasis still means something.
        "axes.labelweight":   "normal",
        "xtick.labelsize":    12 * s,
        "ytick.labelsize":    12 * s,
        "legend.fontsize":    12 * s,
        "figure.titlesize":   20 * s,
        "figure.titleweight": "bold",
        "axes.linewidth":     1.1,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.color":         GRID_C,
        "grid.alpha":         0.35,
        "grid.linewidth":     0.7,
        "lines.linewidth":    1.9,
        "legend.frameon":     False,
        # Vector output: real curves, no resampling, and text stays text
        # (TrueType rather than Type 3, which many publishers require).
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    })
    try:
        yield
    finally:
        mpl.rcParams.update(prev)


def _save(fig, out_dir, name, also_png=True, dpi=200):
    """PDF is the deliverable; the PNG is only so you can eyeball it."""
    import matplotlib.pyplot as plt
    d = Path(out_dir, "publication")
    d.mkdir(parents=True, exist_ok=True)
    paths = [d / f"{name}.pdf"] + ([d / f"{name}.png"] if also_png else [])
    for p in paths:
        fig.savefig(p, dpi=dpi)
        print(f"    [saved] {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Gait / joint resolution
# ═══════════════════════════════════════════════════════════════════

def resolve_gaits(wanted, available, label):
    """Names -> indices, failing loudly with the actual gait list."""
    out = []
    for nm in wanted:
        if nm not in available:
            raise SystemExit(
                f"{label}: this run has no gait {nm!r}.\n"
                f"  available: {available}\n"
                f"  override with the matching --pub_* flag.")
        out.append(available.index(nm))
    return out


def resolve_joints(pairs, leg_cols, label):
    """(leg, joint-in-leg) -> (column index, display label)."""
    out = []
    for leg, k in pairs:
        if not 0 <= leg < len(leg_cols):
            raise SystemExit(f"{label}: leg {leg} out of range "
                             f"0..{len(leg_cols) - 1}")
        if not 0 <= k < len(leg_cols[leg]):
            raise SystemExit(f"{label}: joint {k} out of range for leg {leg}")
        out.append((leg_cols[leg][k],
                    joint_label(leg, k, len(leg_cols[leg]))))
    return out


# ═══════════════════════════════════════════════════════════════════
# Figure 1: reconstruction grid
# ═══════════════════════════════════════════════════════════════════

def fig_reconstruction(model, spikes, gait_tables, device, tgt_range,
                       gait_names, leg_cols, period, out_dir,
                       gaits=None, joints=None, n_cycles=3.0, warm_cycles=2.0):
    """
    Rows are joints, columns are gaits -- the transpose of the diagnostic
    recon, which puts one gait per file with every joint stacked.

    y is shared along each ROW so the same joint is directly comparable
    across gaits, which is the comparison the figure exists to make; sharing
    it across rows instead would flatten the small-range tibia traces.
    """
    import matplotlib.pyplot as plt
    from visualize_timing import predict_and_membranes

    gi = resolve_gaits(gaits or PUB_RECON_GAITS, gait_names, "--pub_recon_gaits")
    jc = resolve_joints(joints or PUB_RECON_JOINTS, leg_cols,
                        "--pub_recon_joints")
    warm = int(round(warm_cycles * period))
    n = int(round(n_cycles * period))

    # One free-run per gait, reused down its column.
    preds, gts = {}, {}
    for g in gi:
        pred, _ = predict_and_membranes(model, spikes[:warm + n], g, device,
                                        warm, tgt_range, keep_membranes=False)
        preds[g] = pred[-n:]
        tb = gait_tables[g]
        rows = (np.arange(n) % tb.shape[0]).astype(int)
        gts[g] = tb[rows]

    R, C = len(jc), len(gi)
    fig, axes = plt.subplots(R, C, figsize=(3.4 * C, 2.16 * R),
                             sharex=True, squeeze=False)
    t = np.arange(n) / period

    for r, (col, jname) in enumerate(jc):
        lo = min(min(gts[g][:, col].min(), preds[g][:, col].min()) for g in gi)
        hi = max(max(gts[g][:, col].max(), preds[g][:, col].max()) for g in gi)
        pad = 0.08 * max(hi - lo, 1e-6)
        for c, g in enumerate(gi):
            ax = axes[r][c]
            ax.plot(t, gts[g][:, col], color=GT_C, label="Target")
            ax.plot(t, preds[g][:, col], color=PRED_C, ls="--",
                    label="Prediction")
            ax.set_ylim(lo - pad, hi + pad)
            if c:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(gait_label(gait_names[g]), pad=8)
            if r == R - 1:
                ax.set_xlabel("CPG cycles")
        # One line, degree sign inline: the two-line form was the main
        # source of vertical crowding between rows.
        axes[r][0].set_ylabel(f"{jname}  (\u00b0)")

    axes[0][C - 1].legend(loc="upper right", ncol=1)
    fig.align_ylabels([axes[r][0] for r in range(R)])
    # The column titles occupy ~0.32 in ABOVE the axes top, which is what the
    # suptitle has to clear -- placing it relative to the axes alone put it
    # straight through them.
    fig.tight_layout()
    fig.subplots_adjust(top=0.921)
    fig.suptitle("Joint angle reconstruction across gaits", y=0.972)
    _save(fig, out_dir, "fig_reconstruction")


# ═══════════════════════════════════════════════════════════════════
# Figure 2: transition sequence
# ═══════════════════════════════════════════════════════════════════

def fig_transitions(model, spikes, gait_tables, device, tgt_range,
                    gait_names, leg_cols, period, out_dir,
                    sequence=None, joints=None, cycles_per_gait=2.0,
                    warm_cycles=2.0):
    """
    One continuous free run through a sequence of gaits, switching every
    `cycles_per_gait` cycles. Unlike the diagnostic transition plot this
    never resets state between segments -- the point is that a switch costs
    one index change mid-stride, so the trace has to be continuous to show it.
    """
    import matplotlib.pyplot as plt

    seq = resolve_gaits(sequence or PUB_TRANSITION_SEQ, gait_names,
                        "--pub_transition_seq")
    jc = resolve_joints(joints or PUB_TRANSITION_JOINTS, leg_cols,
                        "--pub_transition_joints")
    seg = int(round(cycles_per_gait * period))
    warm = int(round(warm_cycles * period))
    total = seg * len(seq)
    if warm + total > spikes.shape[0]:
        raise SystemExit(
            f"the replayed CPG is {spikes.shape[0]} steps but this figure "
            f"needs {warm + total} ({len(seq)} gaits x {cycles_per_gait:g} "
            f"cycles + {warm_cycles:g} warm). Raise --n_steps.")

    lo, hi = tgt_range
    scale, shift = (hi - lo) / 2.0, (hi + lo) / 2.0
    n_joints = gait_tables[0].shape[1]

    # Free run: warm up on the first gait, then walk the sequence.
    model.eval()
    with torch.no_grad():
        state = model.init_state(1, device)
        gt_buf = torch.zeros(1, dtype=torch.long, device=device)
        out = np.zeros((total, n_joints), np.float32)
        gait_of_step = np.zeros(total, np.int32)
        for k in range(warm + total):
            i = k - warm
            g = seq[0] if i < 0 else seq[min(i // seg, len(seq) - 1)]
            gt_buf.fill_(g)
            x = torch.as_tensor(spikes[k], dtype=torch.float32,
                                device=device).unsqueeze(0)
            y, state, _ = model.step(x, gt_buf, state)
            if i >= 0:
                out[i] = y[0].cpu().numpy() * scale + shift
                gait_of_step[i] = g

    R = len(jc)
    fig, axes = plt.subplots(R, 1, figsize=(13.5, 2.75 * R), sharex=True,
                             squeeze=False)
    axes = [a[0] for a in axes]
    t = np.arange(total) / period
    sw = [i * seg / period for i in range(1, len(seq))]

    for r, (col, jname) in enumerate(jc):
        ax = axes[r]
        # Target for the gait active at each step, so the reference is
        # piecewise and the discontinuity at a switch is visible.
        tgt = np.empty(total, np.float32)
        for i in range(total):
            tb = gait_tables[gait_of_step[i]]
            tgt[i] = tb[i % tb.shape[0], col]
        ax.plot(t, tgt, color=GT_C, lw=1.6, label="Target")
        ax.plot(t, out[:, col], color=PRED_C, lw=1.7, ls="--",
                label="Prediction")
        for s in sw:
            ax.axvline(s, color="k", lw=1.1, ls=":", alpha=0.55)
        # One point smaller than the global axes.labelsize: these labels are
        # the longest in any of the figures and only two rows tall, so at full
        # size the rotated text from adjacent rows collided.
        ax.set_ylabel(f"{jname}  (\u00b0)",
                      fontsize=plt.rcParams["axes.labelsize"] - 1)
        ax.set_xlim(0, total / period)

    # Gait labels along the top, centred on their segment.
    top = axes[0]
    for i, g in enumerate(seq):
        top.annotate(gait_label(gait_names[g]),
                     xy=((i + 0.5) * seg / period, 1.03),
                     xycoords=("data", "axes fraction"),
                     ha="center", va="bottom", fontsize=12,
                     fontweight="bold", rotation=0)
    axes[-1].set_xlabel("CPG cycles")
    axes[0].legend(loc="lower right", ncol=2)
    fig.align_ylabels(axes)
    # Same reasoning as the reconstruction figure, except what sits above the
    # top axes here is the per-segment gait annotation at axes fraction 1.03.
    fig.tight_layout(h_pad=2.0)
    fig.subplots_adjust(top=0.855)
    fig.suptitle("Gait transitions during continuous operation", y=0.952)
    _save(fig, out_dir, "fig_transitions")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: timing alignment
# ═══════════════════════════════════════════════════════════════════

def fig_timing_alignment(model, spikes, gait_tables, device, tgt_range,
                         gait_names, leg_cols, period, onsets, out_dir,
                         gait, n_cycles=3.0, warm_cycles=2.0):
    """
    One row per leg, that leg's joints overlaid and each scaled to its OWN
    range so a 5-degree tibia adjustment is as visible as a 60-degree coxa
    swing. The CPG raster, the timing raster and every leg row share one x
    axis with burst onsets marked, so a timing spike can be traced upward to
    the CPG burst that caused it and downward to the joint motion it drives.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from visualize_timing import predict_and_membranes, timing_raster

    if not getattr(model, "use_timing", True) or \
            not hasattr(model, "timing_map_list"):
        print("    (skipping timing alignment: this model has no timing layer)")
        return

    warm = int(round(warm_cycles * period))
    n = int(round(n_cycles * period))
    pred, _ = predict_and_membranes(model, spikes[:warm + n], gait, device,
                                    warm, tgt_range, keep_membranes=False)
    pred = pred[-n:]
    tspk_all = timing_raster(model, spikes[:warm + n], gait, device)[-n:]
    cpg = spikes[warm:warm + n]
    n_cpg = cpg.shape[1]
    jnames = joint_type_names(len(leg_cols[0]))
    owner = {c: g for g, cols in enumerate(model.group_cols) for c in cols}

    L = len(leg_cols)
    fig = plt.figure(figsize=(13.0, 2.4 + 1.72 * L))
    gs = gridspec.GridSpec(2 + L, 1, height_ratios=[0.9, 0.9] + [1.6] * L,
                           hspace=0.18, left=0.075, right=0.985,
                           top=0.944, bottom=0.085)
    ax_cpg = fig.add_subplot(gs[0])
    ax_tim = fig.add_subplot(gs[1], sharex=ax_cpg)
    ax_leg = [fig.add_subplot(gs[2 + i], sharex=ax_cpg) for i in range(L)]
    # sharex= does NOT hide inner tick labels the way plt.subplots(sharex=True)
    # does, so without this every row prints its own x labels across the panel
    # above it.
    for a in [ax_cpg, ax_tim] + ax_leg[:-1]:
        a.tick_params(labelbottom=False)
    t = np.arange(n) / period

    # Burst onsets inside the window, in cycles -- the alignment guides.
    on = [(o - warm) / period for o in onsets
          if warm <= o < warm + n]

    def guides(ax):
        for o in on:
            ax.axvline(o, color="0.45", lw=1.0, ls="--", alpha=0.7, zorder=0)

    cmap = plt.get_cmap("turbo")
    # One colour per TIMING neuron, used for both its raster lane and its rug
    # ticks in the leg rows below, so a tick can be matched to the lane that
    # produced it.
    T_all = tspk_all.shape[1]
    tcol = [TIMING_COLORS[i % len(TIMING_COLORS)] for i in range(T_all)]
    for i in range(n_cpg):
        idx = np.where(cpg[:, i] > 0)[0]
        ax_cpg.scatter(idx / period, np.full(len(idx), i), marker="|",
                       s=130, lw=1.6,
                       color=cmap(0.12 + 0.76 * i / max(n_cpg - 1, 1)))
    # A tick per neuron but text only at the ends: with a lane every ~6 px
    # the full set of labels overlapped, while dropping ticks entirely loses
    # the lane positions.
    ax_cpg.set_yticks(range(n_cpg))
    ax_cpg.set_yticklabels([f"N{i}" if i in (0, n_cpg - 1) else ""
                            for i in range(n_cpg)])
    ax_cpg.set_ylim(-0.7, n_cpg - 0.3)
    ax_cpg.set_ylabel("CPG")
    ax_cpg.grid(False)
    guides(ax_cpg)

    T = T_all
    for j in range(T):
        idx = np.where(tspk_all[:, j] > 0)[0]
        ax_tim.scatter(idx / period, np.full(len(idx), j), marker="|",
                       s=110, lw=1.6, color=tcol[j])
    ax_tim.set_ylim(-0.7, T - 0.3)
    ax_tim.set_yticks(range(T))
    ax_tim.set_yticklabels([f"T{i}" if i in (0, T - 1) else ""
                            for i in range(T)])
    ax_tim.set_ylabel("Timing")
    ax_tim.grid(False)
    guides(ax_tim)

    tb = gait_tables[gait]
    for li, cols in enumerate(leg_cols):
        ax = ax_leg[li]
        for k, c in enumerate(cols):
            lo_c, hi_c = float(tb[:, c].min()), float(tb[:, c].max())
            rng = max(hi_c - lo_c, 1e-6)
            nm = jnames[k][:1].upper() + jnames[k][1:]
            ax.plot(t, 100.0 * (pred[:, c] - lo_c) / rng,
                    color=JOINT_COLORS[k % len(JOINT_COLORS)],
                    lw=1.9, label=nm if li == 0 else None)
        # This leg's timing spikes as a rug at the panel base, so they sit
        # directly under the motion they cause and line up with the rasters.
        for k, c in enumerate(cols):
            g_own = owner.get(c)
            if g_own is None or g_own >= T:
                continue
            idx = np.where(tspk_all[:, g_own] > 0)[0]
            ax.vlines(idx / period, -9 - 5 * k, -3 - 5 * k,
                      color=tcol[g_own], lw=1.3, alpha=0.95)
        ax.set_ylim(-9 - 5 * len(cols), 108)
        ax.set_yticks([0, 50, 100])
        # Short label: "(% range)" on two lines was tall enough to collide
        # with the neighbouring row's label at this row height.
        ax.set_ylabel(f"Leg {li} (%)")
        guides(ax)

    ax_leg[-1].set_xlabel("CPG cycles        "
                          "(each joint scaled to its own min-max range)")
    ax_cpg.set_xlim(0, n / period)
    fig.suptitle(f"Timing-layer alignment - {gait_label(gait_names[gait])}",
                 y=0.986)
    fig.align_ylabels([ax_cpg, ax_tim] + ax_leg)
    # Figure-level legend below the axes: on ax_leg[0] it sat on top of the
    # traces it was labelling.
    h, lb = ax_leg[0].get_legend_handles_labels()
    if h:
        fig.legend(h, lb, loc="lower center", ncol=len(h),
                   bbox_to_anchor=(0.5, 0.0))
    _save(fig, out_dir, f"fig_timing_alignment_{gait_names[gait]}")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: training curves
# ═══════════════════════════════════════════════════════════════════

def read_metrics(model_dir):
    """
    metrics.csv as {column: [float, ...]}, blanks becoming NaN.

    Read from the CSV rather than from the config's `history`, because the CSV
    is appended and flushed every epoch: a run stopped with Ctrl+C still has a
    complete one, whereas `history` only lands on a clean exit.
    """
    p = Path(model_dir, "metrics.csv")
    if not p.is_file():
        return None, p
    cols = {}
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if k is None:
                    continue
                try:
                    cols.setdefault(k, []).append(
                        float(v) if v not in ("", None) else float("nan"))
                except ValueError:
                    pass
    return ({k: np.asarray(v) for k, v in cols.items()} if cols else None), p


def fig_training_curves(model_dir, out_dir):
    """
    Loss against epoch on a log axis, with the learning-rate schedule below.

    Two stacked panels rather than a twin y axis: the schedule is context for
    reading the loss curve, and overlaying it on the same axes invites the
    reader to compare two quantities that share no units.
    """
    import matplotlib.pyplot as plt

    m, p = read_metrics(model_dir)
    if m is None:
        print(f"    (skipping training curves: no readable {p})")
        return
    e = m.get("epoch")
    if e is None or len(e) < 2:
        print("    (skipping training curves: fewer than two epochs logged)")
        return

    series = [("train", "Train", GT_C, "-"),
              ("val", "Validation", "#1a7a3c", "--"),
              ("val_post_switch", "Validation (post-switch)", PRED_C, ":")]
    have_lr = "lr" in m and np.isfinite(m["lr"]).any()
    if have_lr:
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2.6, 1.0]})
        ax, ax_lr = axes
    else:
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        ax_lr = None

    for key, label, colour, ls in series:
        if key not in m or not np.isfinite(m[key]).any():
            continue
        ax.plot(e, m[key], color=colour, ls=ls, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("Masked MSE")
    ax.legend(loc="upper right")

    if "best" in m:
        bi = np.where(m["best"] > 0.5)[0]
        if len(bi):
            k = bi[-1]
            ax.plot([e[k]], [m["train"][k]], marker="o", ms=7,
                    mfc="none", mec="k", mew=1.4, zorder=5)
            ax.annotate("Best", xy=(e[k], m["train"][k]),
                        xytext=(6, 10), textcoords="offset points",
                        fontsize=11)

    if ax_lr is not None:
        ax_lr.plot(e, m["lr"], color="#6a6a6a")
        ax_lr.set_ylabel("Learning rate")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_xlim(e[0], e[-1])
    else:
        ax.set_xlabel("Epoch")
        ax.set_xlim(e[0], e[-1])

    fig.align_ylabels(axes if ax_lr is not None else [ax])
    fig.tight_layout()
    fig.subplots_adjust(top=0.915 if ax_lr is not None else 0.885)
    fig.suptitle("Training convergence", y=0.972)
    _save(fig, out_dir, "fig_training_curves")


# ═══════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════

def run_publication(model_dir, out_dir=None, args=None):
    """Everything the CLI does, callable from visualize_timing.py."""
    from visualize_timing import replay_cpg, cpg_phase

    args = args if args is not None else default_args()
    model_dir = Path(model_dir)
    out_dir = Path(out_dir) if out_dir is not None else model_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\nModel  : {model_dir}\n"
          f"Output : {(out_dir / 'publication').resolve()}\n")

    print("[1/3] Loading run ...")
    cfg, model, arch = load_run(model_dir, args.ckpt, args.cfg, device)
    model.eval()

    gait_files = cfg_get(cfg, "gait_files") or GAIT_FILES_BY_N.get(
        int(cfg_get(cfg, "n_cpg_neurons", 4)))
    gaits_dir = Path(os.path.dirname(os.path.abspath(__file__)),
                     args.gaits_dir)
    tables_orig, all_names = load_gait_tables(gait_files, gaits_dir)
    n_joints = int(cfg_get(cfg, "n_joints", tables_orig[0].shape[1]))
    tgt_range = (float(cfg_get(cfg, "global_min", -124.0)),
                 float(cfg_get(cfg, "global_max", 124.0)))
    cfg_legs = cfg_get(cfg, "leg_cols")
    leg_cols = ([list(c) for c in cfg_legs] if cfg_legs else
                default_leg_layout(int(cfg_get(cfg, "n_cpg_neurons", 4)),
                                   n_joints)[1])
    print(f"  gaits: {all_names}")
    print(f"  legs : {leg_cols}")

    print("\n[2/3] Replaying the CPG ...")
    spikes = replay_cpg(cfg, args.n_steps)
    onsets, period, _, _ = cpg_phase(spikes)
    tables, _ = upsample_gait_tables(tables_orig, all_names,
                                     int(round(period)), verbose=False)
    print(f"  period {period:.1f} steps, {spikes.shape[0]} steps replayed")

    print("\n[3/3] Figures ...")
    with publication_style(args.font_scale):
        fig_reconstruction(model, spikes, tables, device, tgt_range,
                           all_names, leg_cols, period, out_dir,
                           gaits=_split(args.pub_recon_gaits),
                           n_cycles=args.pub_cycles)
        fig_transitions(model, spikes, tables, device, tgt_range,
                        all_names, leg_cols, period, out_dir,
                        sequence=_split(args.pub_transition_seq),
                        cycles_per_gait=args.pub_cycles_per_gait)
        fig_training_curves(model_dir, out_dir)
        for nm in _split(args.pub_align_gaits) or [all_names[0]]:
            fig_timing_alignment(
                model, spikes, tables, device, tgt_range, all_names,
                leg_cols, period, onsets, out_dir,
                gait=resolve_gaits([nm], all_names, "--pub_align_gaits")[0],
                n_cycles=args.pub_cycles)
    print("\nDone.")


def _split(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _build_parser():
    ap = argparse.ArgumentParser(
        description="Publication figures for a trained CPG-SNN run: vector "
                    "PDF, larger type, and layouts chosen for the paper "
                    "rather than for diagnosis.")
    ap.add_argument("--model_dir", type=str, default="",
                    help="Resolved as outputs/<model_dir>.")
    ap.add_argument("--out_dir",   type=str, default=None,
                    help="Default: the model dir. Figures go in "
                         "<out_dir>/publication/.")
    ap.add_argument("--ckpt", type=str, default="best_model.pt")
    ap.add_argument("--cfg",  type=str, default="cpg_lif_snn_config.json")
    ap.add_argument("--gaits_dir", type=str, default="../gaits")
    ap.add_argument("--n_steps", type=int, default=6000,
                    help="CPG steps to replay. Must cover the longest figure: "
                         "the transition sequence needs "
                         "(gaits x cycles_per_gait + warm) cycles.")
    ap.add_argument("--font_scale", type=float, default=1.0,
                    help="Multiplies every font size. Raise for a figure that "
                         "will be printed small.")
    ap.add_argument("--pub_cycles", type=float, default=3.0,
                    help="CPG cycles per panel in the reconstruction and "
                         "timing-alignment figures.")
    ap.add_argument("--pub_cycles_per_gait", type=float, default=2.0,
                    help="Cycles held per gait in the transition figure; with "
                         "6 gaits this gives a 12-cycle axis and 5 switches.")
    ap.add_argument("--pub_recon_gaits", type=str,
                    default=",".join(PUB_RECON_GAITS),
                    help="Columns of the reconstruction figure, in order.")
    ap.add_argument("--pub_transition_seq", type=str,
                    default=",".join(PUB_TRANSITION_SEQ),
                    help="Gait order for the transition figure.")
    ap.add_argument("--pub_align_gaits", type=str, default="tripod",
                    help="One timing-alignment figure per gait listed.")
    return ap


def default_args(**overrides):
    a = _build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def main():
    args = _build_parser().parse_args()
    this_dir = os.path.dirname(os.path.abspath(__file__))
    run_publication(outputs_path(this_dir, args.model_dir),
                    outputs_path(this_dir, args.out_dir)
                    if args.out_dir is not None else None,
                    args)


if __name__ == "__main__":
    main()