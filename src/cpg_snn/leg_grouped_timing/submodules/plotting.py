"""
Every figure: end-of-training plots (CPG raster, training curves, reconstruction,
transitions), the diagnostic timing and drive figures, and the publication figures.

Publication figures (visualize.py --publication; vector PDF, larger type, paper layouts):

  fig_reconstruction   6 joints (rows) x 4 gaits (columns), 3 CPG cycles each
  fig_transitions      2 joints over one continuous 12-cycle axis with a gait
                       change every 2 cycles (5 transitions, 6 gaits)
  fig_timing_alignment 6 legs (rows), each leg's 3 joints overlaid and scaled
                       to their own range, under a shared CPG and timing
                       raster with burst onsets aligned across every row
                       (also written as dense_timing_alignment_<gait>: raster
                       colored per leg, carpets per joint, spike runs merged
                       into blocks)
  fig_training_curves  loss against epoch on a log axis with the learning-rate
                       schedule beneath, read from the run's metrics.csv
"""

import contextlib
import csv
import math
import matplotlib.gridspec as gridspec
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .analysis import (
    circ_diff, circular_stats, drive_channels, drive_report,
    drive_spike_report, fundamental_phase, unit_train,
)
from .common import cfg_get, joint_type_names, out_path
from .cpg import spike_bursts


# Long enough for a 6-neuron CPG; indexed modulo its own length everywhere.
CPG_PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261",
               "#6a0572", "#8ecae6"]


# ═══════════════════════════════════════════════════════════════════
# Plots
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

TIMING_PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261",
                  "#6a0572", "#8ecae6", "#ffb703", "#023047"]
GT_COLOR   = "#457b9d"
PRED_COLOR = "#e63946"


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


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

def _savefig(fig, out_dir, name, dpi):
    # out_path routes by filename prefix into recons/ timing_alignments/ membrane_waveforms/
    # phase_folds/ (see OUT_ROUTES in submodules/common.py).
    p = out_path(out_dir, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {p}")


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
    # it is |s_gate|. Shared with analysis.drive_report so figure and printout agree.
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
    # analysis.drive_report is called verbatim, not reimplemented: two copies of one statistic
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
        print(f"    (skipping analysis.drive_report: --gaits selected "
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
# Timing-neuron colors. Hand-picked rather than sampled from tab20, half of
# whose entries are pale tints that vanish as 1 px rug ticks. Consecutive
# entries are far apart in hue because the neurons sharing a leg row are
# consecutive indices, so that is where contrast actually has to hold.
TIMING_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080",
    "#f032e6", "#9a6324", "#800000", "#808000", "#000075", "#b8860b",
    "#1b9e77", "#d95f02", "#7570b3", "#66a61e", "#a6761d", "#525252",
]
# One color per LEG for the dense figure's timing raster. Drawn from
# TIMING_COLORS but skipping the blue/orange/green hues of JOINT_COLORS, which
# color the carpets there, so a leg is never mistaken for a joint.
LEG_COLORS = ["#e6194b", "#911eb4", "#008080", "#9a6324", "#000075", "#808000"]
GT_C   = "#1f3a5f"
PRED_C = "#c1272d"
SPK_C  = "#2a9d8f"
GRID_C = "#b0b7c3"


@contextlib.contextmanager
def publication_style(scale=1.0):
    """
    Larger type and cleaner axes, restored on exit.

    Set through a context manager rather than a global style file so that
    importing this module cannot change how the diagnostic
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
    from .analysis import predict_and_membranes

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
    _save(fig, out_dir, "reconstruction")


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

    # Gait labels along the top, centered on their segment.
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
    _save(fig, out_dir, "transitions")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: timing alignment
# ═══════════════════════════════════════════════════════════════════

def _spike_blocks(train, period, max_gap=2):
    """
    (start, width) bars in CPG cycles for broken_barh. Spikes at most `max_gap`
    steps apart (one empty step between them, the CPG's in-burst spacing) merge
    into one block, so back-to-back bursts become one continuous block; an
    isolated spike is a one-step block.
    """
    idx = np.flatnonzero(np.asarray(train))
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(s / period, (e - s + 1) / period) for s, e in zip(starts, ends)]


def fig_timing_alignment(model, spikes, gait_tables, device, tgt_range,
                         gait_names, leg_cols, period, onsets, out_dir,
                         gait, n_cycles=3.0, warm_cycles=2.0, dense=False):
    """
    One row per leg, that leg's joints overlaid and each scaled to its OWN
    range so a 5-degree tibia adjustment is as visible as a 60-degree coxa
    swing. The CPG raster, the timing raster and every leg row share one x
    axis with burst onsets marked, so a timing spike can be traced upward to
    the CPG burst that caused it and downward to the joint motion it drives.

    Under --drive_source cpg_lif the timing raster has one lane per
    SUB-NETWORK: its C drive-LIF units collapsed into one effective timing
    neuron, every active channel drawn, alpha by spike share and color by
    channel -- the same rule, via the same functions, as the diagnostic
    timing-alignment plot in visualize.py. Each joint's rug shows its
    sub-network's lane.

    dense=True draws the same figure as dense_timing_alignment_<gait>: raster
    lanes colored per LEG, carpets in their joint's trace color, and spikes merged
    into solid blocks
    (see _spike_blocks), so spike density reads as block width against the
    joint's velocity.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    # drive_channels and unit_train are imported rather than re-implemented
    # so this figure selects and shades channels by EXACTLY the rule the
    # diagnostic timing-alignment plot uses.
    from .analysis import (predict_and_membranes, timing_raster,
                           drive_channels, unit_train)

    # Skip only architectures with no timing structure at all. The old guard
    # tested `use_timing`, which no model defines (the attribute is
    # `uses_timing`), so it never fired -- harmless for the timing layer but
    # it would have silently admitted anything.
    if not hasattr(model, "timing_map_list"):
        print("    (skipping timing alignment: this model has no timing layer)")
        return
    uses_timing = bool(getattr(model, "uses_timing", True))
    is_lif = bool(getattr(model, "drive_lif", False))

    warm = int(round(warm_cycles * period))
    n = int(round(n_cycles * period))
    pred, _ = predict_and_membranes(model, spikes[:warm + n], gait, device,
                                    warm, tgt_range, keep_membranes=False)
    pred = pred[-n:]
    cpg = spikes[warm:warm + n]

    # What the second raster and the rugs show depends on the drive mode.
    #
    #   timing    the timing layer's own spikes, one lane per timing unit.
    #
    #   cpg_lif   the DRIVE LIF's spikes, (n, G, C): one unit per
    #             (sub-network, channel). Using timing_raster here would be
    #             wrong -- under the cpg modes timing_only returns the CPG
    #             train unchanged, so the second raster was a verbatim copy
    #             of the first and each rug indexed CPG channel g_own, i.e.
    #             one channel per joint chosen by index coincidence. The
    #             C units of a sub-network are mutually exclusive (one
    #             channel each, zero reset, CPG channels never coincide), so
    #             a lane collapses them into ONE effective timing neuron per
    #             sub-network: every active channel drawn, alpha by spike
    #             share, color by channel. Shares are counted over the
    #             plotted window only, matching the diagnostic figures.
    #
    #   cpg       no spiking drive layer; the raw CPG channels are the
    #             drive, selected by the 90% rule on |s_gate|*||u|| because
    #             a continuous drive is never exactly zero.
    sel = None
    if uses_timing:
        tspk_all = timing_raster(model, spikes[:warm + n], gait, device)[-n:]
    else:
        G_sub = int(model.G)
        if is_lif:
            xs = torch.as_tensor(spikes[:warm + n], dtype=torch.float32,
                                 device=device).unsqueeze(1)
            gs_ = torch.full((warm + n, 1), int(gait), dtype=torch.long,
                             device=device)
            with torch.no_grad():
                tr = model.drive_spikes_only(xs, gs_)        # (L,1,G,C)
            tspk_all = tr[:, 0].cpu().numpy()[-n:]           # (n,G,C)
            weight = tspk_all.sum(axis=0)                    # spike counts
            frac = None                                      # every active ch
        else:
            tspk_all = cpg                                   # (n,C)
            weight = (model.drive_strength().detach().cpu().numpy()[gait])
            frac = 0.90
        share = weight / np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
        sel = [drive_channels(share[g], frac=frac) for g in range(G_sub)]
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
    # One color per TIMING neuron, used for both its raster lane and its rug
    # ticks in the leg rows below, so a tick can be matched to the lane that
    # produced it.
    T_all = tspk_all.shape[1]
    tcol = [TIMING_COLORS[i % len(TIMING_COLORS)] for i in range(T_all)]
    for i in range(n_cpg):
        idx = np.where(cpg[:, i] > 0)[0]
        ax_cpg.vlines(idx / period, i - 0.4, i + 0.4, lw=1.6,
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

    def lane_train(j):
        """Binary (n,) train of lane j: a timing unit, or (cpg / cpg_lif) the
        OR of the channels selected for sub-network j."""
        if sel is None:
            return tspk_all[:, j] > 0
        out = np.zeros(n, dtype=bool)
        for ch in sel[j][0]:
            out |= unit_train(tspk_all, j, int(ch)) > 0
        return out

    # Ticks are vlines in DATA units (80% of a lane). Scatter markers are sized
    # in points, so at 18 lanes a tick covered ~3.5 lanes and each sub-network's
    # spikes bled across its neighbors.
    T = T_all if sel is None else len(sel)
    lane_name = "T" if sel is None else "g"
    lane_col = [TIMING_COLORS[j % len(TIMING_COLORS)] for j in range(T)]
    if dense:
        # One color per LEG: a lane takes the leg of the columns it drives. A
        # lane is a sub-network under cpg / cpg_lif, a timing unit otherwise
        # (mapped to the sub-network it feeds), so this holds for any shape.
        col_leg = {c: li for li, cols in enumerate(leg_cols) for c in cols}
        unit_grp = {u: g for g, us in enumerate(model.timing_map_list)
                    for u in us}
        def lane_leg(j):
            g = j if sel is not None else unit_grp.get(j)
            cols = model.group_cols[g] if g is not None and g < len(model.group_cols) else []
            return col_leg.get(cols[0]) if cols else None
        lane_col = [LEG_COLORS[lane_leg(j) % len(LEG_COLORS)]
                    if lane_leg(j) is not None else "0.4" for j in range(T)]
    for j in range(T):
        if dense:
            ax_tim.broken_barh(_spike_blocks(lane_train(j), period),
                               (j - 0.4, 0.8), facecolor=lane_col[j],
                               edgecolor="none")
        elif sel is None:
            idx = np.where(tspk_all[:, j] > 0)[0]
            ax_tim.vlines(idx / period, j - 0.4, j + 0.4, lw=1.6,
                          color=tcol[j])
        else:
            # Same rule as the rugs below, so lane g reads as exactly the
            # rug beneath its joints.
            for ch, a in zip(*sel[j]):
                idx = np.where(unit_train(tspk_all, j, int(ch)) > 0)[0]
                ax_tim.vlines(idx / period, j - 0.4, j + 0.4, lw=1.6,
                              alpha=float(a),
                              color=TIMING_COLORS[int(ch)
                                                  % len(TIMING_COLORS)])
    ax_tim.set_ylim(-0.7, T - 0.3)
    ax_tim.set_yticks(range(T))
    ax_tim.set_yticklabels([f"{lane_name}{i}" if i in (0, T - 1) else ""
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
            if dense:
                # Carpet in its joint's trace color, so each one reads as
                # belonging to the waveform drawn above it.
                ax.broken_barh(_spike_blocks(lane_train(g_own), period),
                               (-9 - 5 * k, 6),
                               facecolor=JOINT_COLORS[k % len(JOINT_COLORS)],
                               edgecolor="none")
            elif sel is None:
                idx = np.where(tspk_all[:, g_own] > 0)[0]
                ax.vlines(idx / period, -9 - 5 * k, -3 - 5 * k,
                          color=tcol[g_own], lw=1.3, alpha=0.95)
            else:
                # Every channel feeding this joint's sub-network, alpha by
                # share, color by channel -- identical to its raster lane.
                for ch, a in zip(*sel[g_own]):
                    idx = np.where(unit_train(tspk_all, g_own, int(ch))
                                   > 0)[0]
                    ax.vlines(idx / period, -9 - 5 * k, -3 - 5 * k,
                              color=TIMING_COLORS[int(ch)
                                                  % len(TIMING_COLORS)],
                              lw=1.3, alpha=0.95 * float(a))
        ax.set_ylim(-9 - 5 * len(cols), 108)
        ax.set_yticks([0, 50, 100])
        # Short label: "(% range)" on two lines was tall enough to collide
        # with the neighboring row's label at this row height.
        ax.set_ylabel(f"Leg {li} (%)")
        guides(ax)

    ax_leg[-1].set_xlabel("CPG cycles        "
                          "(each joint scaled to its own min-max range)")
    ax_cpg.set_xlim(0, n / period)
    title = "Timing-layer spike density" if dense else "Timing-layer alignment"
    fig.suptitle(f"{title} - {gait_label(gait_names[gait])}", y=0.986)
    fig.align_ylabels([ax_cpg, ax_tim] + ax_leg)
    # Figure-level legend below the axes: on ax_leg[0] it sat on top of the
    # traces it was labelling.
    h, lb = ax_leg[0].get_legend_handles_labels()
    if h:
        fig.legend(h, lb, loc="lower center", ncol=len(h),
                   bbox_to_anchor=(0.5, 0.0))
    _save(fig, out_dir,
          f"{'dense_' if dense else ''}timing_alignment_{gait_names[gait]}")


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
                # Column names and values are STRIPPED. A metrics.csv written
                # with padded columns for readability yields keys like
                # " train     ", which silently missed every lookup and
                # produced an empty figure.
                k = k.strip()
                v = v.strip() if isinstance(v, str) else v
                # NaN on a bad parse rather than skipping the append, so a
                # single malformed cell cannot desynchronise column lengths.
                try:
                    val = float(v) if v not in ("", None) else float("nan")
                except (TypeError, ValueError):
                    val = float("nan")
                cols.setdefault(k, []).append(val)
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
        print(f"    (skipping training curves: fewer than two epochs in {p}; "
              f"columns found: {sorted(m)})")
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

    n_plotted = 0
    for key, label, colour, ls in series:
        if key not in m or not np.isfinite(m[key]).any():
            continue
        ax.plot(e, m[key], color=colour, ls=ls, label=label)
        n_plotted += 1
    if not n_plotted:
        # Fail loudly: matplotlib only warns ("No artists with labels"), which
        # leaves an empty figure on disk looking like a rendering fault rather
        # than a column-name mismatch.
        plt.close(fig)
        print(f"    (skipping training curves: none of "
              f"{[k for k, *_ in series]} present in {p}; "
              f"columns found: {sorted(m)})")
        return
    ax.set_yscale("log")
    ax.set_ylabel("Masked MSE")
    ax.legend(loc="upper right")

    if ax_lr is not None:
        from matplotlib.ticker import FuncFormatter
        lr = m["lr"]
        ax_lr.plot(e, lr, color="#6a6a6a")
        ax_lr.set_ylabel("Learning rate")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_xlim(e[0], e[-1])
        # The schedule anneals to eta_min, not to zero, but on a linear axis
        # spanning 0 to ~4e-3 a final value of 1e-5 is a pixel off the floor
        # and the default bottom tick reads "0.000" -- which states something
        # false. Keeping the axis linear preserves the cosine shape, so
        # instead the floor is lifted above zero (removing the 0.000 tick
        # entirely) and the true minimum gets its own labeled tick.
        lo = float(np.nanmin(lr))
        hi = float(np.nanmax(lr))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo > 0:
            from matplotlib.ticker import MaxNLocator
            ax_lr.set_ylim(lo * 0.45, hi * 1.06)
            # Round tick values: deriving them from hi/4 gave 0.00399902 and
            # friends, since hi is the measured maximum rather than the
            # nominal one.
            nice = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(
                0.0, hi)
            ticks = [lo] + [float(t) for t in nice
                            if lo * 3.0 < t <= hi * 1.001]
            ax_lr.set_yticks(ticks)
            ax_lr.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _: (f"{v:.0e}".replace("e-0", "e-")
                              if v < 0.2 * hi else f"{v:g}")))
    else:
        ax.set_xlabel("Epoch")
        ax.set_xlim(e[0], e[-1])

    fig.align_ylabels(axes if ax_lr is not None else [ax])
    fig.tight_layout()
    fig.subplots_adjust(top=0.915 if ax_lr is not None else 0.885)
    fig.suptitle("Training convergence", y=0.972)
    _save(fig, out_dir, "training_curves")