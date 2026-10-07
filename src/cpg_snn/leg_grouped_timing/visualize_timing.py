"""
Visualise a trained CPG -> timing-layer -> grouped-SNN run.
===========================================================

Loads a checkpoint (.pt) and its config (.json) written by train.py, replays
the CPG bit-identically from the config's own parameters, and plots what the
timing layer is doing against the CPG and against the gait tables.

Usage
-----
    python visualize_timing.py                       # outputs/  ->  outputs/visualize
    python visualize_timing.py --model_dir outputs/run7
    python visualize_timing.py --gaits wkF bk         # subset
    python visualize_timing.py --n_cycles 4 --no_pred # tighter / faster


What is and is not recoverable
------------------------------
The TIMING layer is exact.  `TimingGroupedSNN.timing_only` re-runs the real
`_timing` method, so the rasters here are the same spikes the sub-networks
saw -- not a reconstruction.

The SUB-NETWORK hidden spikes are NOT recoverable from a checkpoint plus a
state trace, and this script does not pretend otherwise.  The reset is
subtractive:

    mem_post = mem_pre - thresh * spk        spk = (mem_pre >= thresh)

so a spiking unit lands at `mem_pre - thresh >= 0` and a silent unit lands at
`mem_pre < thresh`, and those ranges overlap -- given only `mem_post` the two
are indistinguishable.  Recovering them exactly needs either the pre-reset
value or a `record=` path added to `TimingGroupedSNN.forward` (which is safe
to add: the ONNX wrappers call `step`, not `forward`).  Until then this script
plots sub-network MEMBRANES, which are exact and carry most of the same
information, rather than inventing a raster.

`--arch dense` checkpoints have no timing layer; the script says so and emits
the CPG / gait-table / tau figures only.

Outputs (into --out_dir)
------------------------
    timing_alignment_<gait>.png   CPG raster + timing raster + per-leg GT/pred,
                                  shared time axis.  The main figure.
    phase_fold_<gait>.png         Same thing with time folded onto cycle phase:
                                  per-leg GT vs phase with that leg's timing
                                  neuron's phase histogram behind it.
    alignment_summary.png         |residual| heatmap, timing neuron vs leg, per
                                  gait, plus rate and concentration R.
    routing_matrices.png          Measured per-gait CPG->timing routing.
                                  --drive_source timing only.
    drive_gates.png               Per-gait CPG-channel selection (drive =
                                  |s_gate|*||u||), the counterpart figure
                                  for --drive_source cpg. Exactly one of
                                  these two is produced.
    timing_rate_history.png       Per-timing-unit spk/cycle over training,
                                  one gait per line. --drive_source timing.
    drive_fanin_history.png       Channels carrying 90% of each
                                  sub-network's drive, over training.
                                  --drive_source cpg counterpart.
    tau_distributions.png         Learned time constants vs their init range
                                  and vs the CPG period.
    membranes_<gait>.png          Sub-network membrane traces (sampled units).
    timing_summary.json           Every number in the figures, machine-readable.
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
    TimingGroupedSNN, drive_report, drive_spike_report,
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
# 1.  Config / checkpoint loading  --  see train.py
# ═══════════════════════════════════════════════════════════════════
# cfg_get / build_model_from_cfg / load_run live in train.py so this file and
# run_inference.py share one copy of the backward-compatibility rules rather
# than each carrying their own (which has already caused drift once).


# ═══════════════════════════════════════════════════════════════════
# 2.  Replay the CPG from the config
# ═══════════════════════════════════════════════════════════════════

def replay_cpg(cfg, n_steps):
    """
    Re-run the exact CPG the model was trained against.

    Every parameter comes out of cfg["cpg"], including the weight matrix, so
    this is reproducible even if train.py's defaults change later.  The warm-up
    is replayed too: burst phase depends on it.

    cfg["fake_cpg"] selects fake_step_chunk's back-to-back no-gap bursts, so a
    model trained on the synthetic pattern is visualised against that same
    pattern rather than the real oscillator.  Defaults to False for configs
    written before that option existed.
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
    """
    Minimal metrics.csv reader: header row fixes column names, every other
    row is parsed as float where possible, "" (MetricsWriter's blank for a
    value not measured this epoch) becomes NaN.

    Plain `csv`, not pandas -- nothing else in this file uses pandas and a
    second tabular library for one reader is not worth the import.
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
    """
    How each timing unit's firing rate evolved over training, one gait per
    line, laid out on the SAME leg grid as plot_alignment: n_legs rows x
    C columns (3x6 for the hexapod), so a panel's position means the same
    thing in both figures -- row = leg, column = joint-within-leg.

    The panel at (leg, k) shows the timing unit RESPONSIBLE for that column
    (`leg_cols[leg][k]`, looked up through `timing_cols` exactly as
    plot_alignment's `owner` does), not literally "unit number 3*leg+k" --
    those coincide at --timing_shape per_leg but not at per_joint, and this
    plot is only informative once it is right about which unit is which.

    Reads metrics.csv rather than taking an array, because the history is
    only known epoch-by-epoch where the eval landed: MetricsWriter leaves
    "spk_{gait}_t{u}" blank on every epoch that wasn't a timing_log_every
    multiple, and this function is what turns those blanks into gaps in the
    line rather than a crash.  Requires the run to have been trained with the
    metrics.csv change that adds these columns -- older runs simply won't
    have them, and this prints a note and returns rather than erroring.
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

    # One legend for the whole figure rather than one per panel -- the gait
    # set and its colours are the same in every panel, so n_legs*C copies of
    # it would be pure repetition.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        # Panel (0,0) may have been the one skipped as not-yet-firing; pull
        # the legend from whichever panel actually has lines.
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


def plot_drive_fanin_history(metrics_csv, group_cols, leg_cols, gait_names,
                             out_dir, dpi):
    """
    The --drive_source cpg counterpart of plot_timing_rate_history: how many
    CPG channels carry 90% of each sub-network's drive, over training, one
    gait per line.

    Same grid and the same reader, parameterised -- the only real differences
    are the metrics.csv prefix (fanin_ rather than spk_) and the owner map.
    The panel at (leg, k) is the SUB-NETWORK that emits that column, so the
    lookup goes through `group_cols`, not `timing_cols`: under cpg drive
    there are no timing units, and it is the sub-network's own channel
    selection being plotted.

    This is the curve --drive_lambda is supposed to bend. Falling = selection
    concentrating onto fewer phases. Flat near n_cpg = the sub-network is
    still listening to everything, so either the L1 weight is too small or
    the task genuinely needs broad phase coverage there.
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
    """
    Free-run the full model.  Returns predictions in DEGREES for steps
    [warm:], plus per-timestep membrane snapshots if asked.

    Stepped manually rather than via forward() only because forward() throws
    the intermediate state away and the membranes are the point here; the
    arithmetic is still model.step, so nothing is duplicated.
    """
    lo, hi = tgt_range
    scale, shift = (hi - lo) / 2.0, (hi + lo) / 2.0

    x = torch.as_tensor(spikes, dtype=torch.float32, device=device)
    T = x.shape[0]
    # Constant across the run, so built once: it is a (1,) index, but
    # allocating it inside a T-step Python loop is pure overhead.
    g = torch.full((1,), int(gait_idx), dtype=torch.long, device=device)
    state = model.init_state(1, device)
    preds, mems = [], []
    for t in range(T):
        y, state, _ = model.step(x[t:t + 1], g, state)
        preds.append(y[0].cpu().numpy())
        # Capped: at hidden=256 x 4 groups a full state snapshot is ~12 KB,
        # so recording every step of a long window would run to hundreds of
        # MB for traces the membrane plot never reads.
        if keep_membranes and warm <= t < warm + max_mem_steps:
            # state is (mem_timing, mem1, mem2, memo) for timing_grouped,
            # (mem1, mem2, memo) for dense.
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
    # frac=None selects every STRICTLY POSITIVE channel instead of a
    # cumulative-mass subset. That is the right selector once the drive is a
    # spike COUNT (drive_source=cpg_lif): "this channel delivers nothing" is
    # then the exact integer statement "zero spikes", so there is no cutoff
    # to choose, and a 90% rule would hide genuinely active channels -- at a
    # near-uniform 18-channel row it keeps only 15 and drops 3 that really
    # do spike. The 90% rule remains correct for the continuous drive of
    # drive_source=cpg, where nothing is ever exactly zero and some cutoff
    # is unavoidable.
    """
    Which channels to draw for one sub-network, and how strongly.

    `share` is that sub-network's normalised drive over the C CPG channels.
    Returns (idx, alpha) with idx sorted by descending share.

    Combines the two obvious approaches because each alone fails:

      - Drawing ALL channels with alpha proportional to share is honest but
        unreadable at 18 channels. In the unselected case every share is
        ~1/18, so 18 near-identical faint rugs overlap into a grey smear
        and you cannot see that nothing was selected.

      - Drawing only the top channels loses the gradation, and any cutoff
        is arbitrary.

    So: keep the channels carrying `frac` of the drive -- the same 90% rule
    the fan-in metric uses, so the plot and the number agree on what counts
    as significant -- and within that set set alpha by share RELATIVE TO THE
    LARGEST, not to the sum. Relative-to-largest matters: the dominant
    channel is then always fully opaque, so a concentrated sub-network reads
    as one solid rug and a flat one as many equally-mid-alpha rugs. Scaling
    alpha by the raw share would make everything faint whenever selection is
    spread, confounding "weakly driven" with "many channels".
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
    # Raw share-relative alpha, then mapped into [alpha_min, 1]. Without the
    # floor the weakest member of the 90% set gets alpha = its share divided
    # by the largest share, which for a dominant-plus-tail row is ~0.03 --
    # drawn, but invisible, so the rugs could not be counted against the
    # reported fan-in. Every channel in the set is by definition part of the
    # 90%, so each should be legible; the floor says "faint" rather than
    # "absent" while the ordering still shows which dominates.
    rel = p[idx] / p[idx].max()
    alpha = alpha_min + (1.0 - alpha_min) * rel
    return idx, alpha


def circular_stats(phases, weights=None):
    """Circular mean in [0,1) and concentration R in [0,1] of a set of
    cycle phases.  R is the thing to read: a unit can fire at a healthy rate
    and still be useless if R is low, because then it is firing all over the
    cycle and its mean phase means nothing.

    `weights` lets a spike count for less than one. Used by --drive_source
    cpg, where a sub-network's effective input phase is the DRIVE-WEIGHTED
    mean over the channels feeding it: a channel supplying 40% of the drive
    should move that mean twice as far as one supplying 20%."""
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
    """
    Phase of the first Fourier component of one cycle of a waveform, in [0,1).

    Used as the reference point for "where in its cycle is this leg".  Chosen
    over something like 'swing onset' deliberately: onset needs a threshold or
    a derivative test, both of which are arbitrary and behave differently
    across gaits with different amplitudes.  The fundamental is
    parameter-free, robust to noise, and comparable across gaits.

    Caveat worth remembering when reading the residuals: for a strongly
    non-sinusoidal trajectory the fundamental is not the same as the visually
    obvious footfall instant.  It is a consistent reference, not a
    biomechanical event.
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
    # out_path routes by filename prefix into recons/ timing_alignments/
    # membrane_waveforms/ phase_folds/, leaving anything else loose in the run
    # directory. See OUT_ROUTES in train.py.
    p = out_path(out_dir, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {p}")


def unit_train(tspk, g, ch):
    """
    One unit's binary train, for either raster layout.

    tspk is (L, lanes) when a lane IS the unit -- the timing layer, or the
    raw CPG channels. Under drive_source=cpg_lif it is (L, G, C) instead,
    because the unit that matters is (sub-network, channel): the drive LIF
    thins and delays its channel's burst, so sub-network g's copy of channel
    c is NOT the same train as channel c itself, and plotting the raw
    channel would show input the sub-network never received.
    """
    return tspk[:, g, ch] if tspk.ndim == 3 else tspk[:, ch]


def plot_alignment(spikes, tspk, gt, pred, phase, onsets, period, burst_thr,
                   timing_cols, leg_cols, gait_name, out_dir, dpi, t_lo, t_hi,
                   drive_share=None, drive_frac=0.90):
    """
    The main figure.  Two full-width rasters on top, then a grid of ONE ROW
    PER LEG and ONE COLUMN PER JOINT WITHIN THAT LEG.

    The second raster is the CPG's own train under drive_source=timing, but
    under drive_source=cpg that would be a verbatim copy of the first (
    timing_only returns the input unchanged there), so it becomes one lane
    per SUB-NETWORK, drawn with the same channel selection and alphas as the
    rug beneath that sub-network's waveform panel.  Lane g is then literally
    the rug below it, which makes the raster and the panels correspond by
    construction.

    The grid comes from `leg_cols` (the robot's anatomy), NOT from the
    network's grouping. Those coincide at --decoder_shape per_leg, but at
    per_joint every sub-network is a single column, which used to give 18
    single-panel rows -- far too many to read. Anatomy is the right axis for
    this plot either way: 6 rows x 3 columns for the hexapod, 4 x 2 for the
    quadruped, regardless of the network structure.

    Splitting per joint rather than overlaying them matters because the joints
    within a leg have different amplitudes and shapes; overlaid on one axis the
    small-amplitude ones were unreadable and a shared y-scale flattened them.

    `drive_share` switches this to --drive_source cpg mode: pass that gait's
    (G, C) normalised drive and `timing_cols=group_cols`, and each panel then
    overlays the CPG channels feeding its SUB-NETWORK, alpha-weighted by
    share (see `drive_channels`), instead of one timing unit's rug. Without
    it nothing changes.

    Each panel's spike rug is the timing unit responsible for THAT column,
    looked up through `timing_cols` -- NOT through group_cols, which is
    indexed by sub-network and is a different length once --timing_shape and
    --decoder_shape differ. With per_leg timing all three of a leg's panels
    share one rug; with per_joint timing each panel shows its own. Every axis
    shares x with the rasters; vertical alignment across the whole figure is
    the point.
    """
    n_cpg = spikes.shape[1]
    # tspk is (L, lanes) or (L, G, C); either way axis 1 is the lane count
    # for the timing-layer path, and the cpg path overrides it with n_lane.
    G     = tspk.shape[1]                   # timing neurons (raster lanes)
    n_legs = len(leg_cols)
    C     = len(leg_cols[0])
    tnames = joint_type_names(C)
    # column -> the timing unit responsible for it. Built from timing_cols so
    # the index is always a valid raster lane.
    owner = {c: t for t, cols in enumerate(timing_cols) for c in cols}
    sl    = slice(t_lo, t_hi)
    t     = np.arange(t_lo, t_hi)

    # Both rasters are sized PER LANE, not by a fixed ratio. A fixed height
    # crams the lanes together as soon as either count grows: --n_timing 18
    # put 18 lanes in a 6-lane slot, and --n_cpg_neurons 18 then did the same
    # to the CPG raster. Scaling each by its own lane count keeps the lane
    # spacing constant no matter what the two counts are, and the figure grows
    # to match so the extra space is ADDED rather than taken from the trace
    # rows below.
    #
    # RASTER_PER_LANE is set so that the 6-lane case comes out at the 1.1 it
    # used to be, i.e. existing 6-neuron figures are unchanged. MIN_RASTER_H
    # stops a 3-neuron CPG getting a sliver too thin to label.
    RASTER_PER_LANE, MIN_RASTER_H, LEG_H = 0.183, 1.1, 1.5
    # Lanes in the SECOND raster. Under drive_source=cpg it has one lane per
    # SUB-NETWORK, not one per raster channel -- and those counts differ as
    # soon as --decoder_shape is not per_joint (6 sub-networks against 18
    # channels at per_leg), so this cannot be tspk.shape[1].
    n_lane = drive_share.shape[0] if drive_share is not None else G
    CPG_H = max(MIN_RASTER_H, RASTER_PER_LANE * n_cpg)
    tim_h = max(MIN_RASTER_H, RASTER_PER_LANE * n_lane)
    units = CPG_H + tim_h + LEG_H * n_legs
    fig = plt.figure(figsize=(max(14.0, 5.8 * C), 1.05 * units + 1.0))
    # top/bottom pinned so the suptitle sits close to the first raster instead
    # of leaving a band of dead space (bbox_inches="tight" crops the OUTER
    # margin, not internal gaps).
    gs  = gridspec.GridSpec(2 + n_legs, C, hspace=0.62, wspace=0.20,
                            top=0.935, bottom=0.045, left=0.055, right=0.99,
                            height_ratios=[CPG_H, tim_h] + [LEG_H] * n_legs)

    # The two rasters span every column; the leg rows do not.
    ax_cpg = fig.add_subplot(gs[0, :])
    ax_tim = fig.add_subplot(gs[1, :], sharex=ax_cpg)
    ax_leg = [[fig.add_subplot(gs[2 + j, k], sharex=ax_cpg)
               for k in range(C)] for j in range(n_legs)]
    all_axes = [ax_cpg, ax_tim] + [a for row in ax_leg for a in row]

    # sharex= does not auto-hide tick labels the way plt.subplots(sharex=True)
    # does, so without this every row prints its own time labels directly above
    # the next row's title.  Only the bottom leg row needs them.
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
    # Under drive_source=cpg this used to be a verbatim copy of the CPG
    # raster, because timing_only returns the CPG train unchanged -- two
    # identical panels and no information. Instead give it one lane per
    # sub-network and fill each lane using the SAME rule as the rug under
    # that sub-network's waveform panel: the channels carrying 90% of its
    # drive, alpha by share relative to the largest, coloured by channel.
    # Lane g then reads as exactly the rug you see below it, so the raster
    # and the panels line up by construction instead of by coincidence.
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
                     "selection and colours as the rugs below; (n) = that "
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
                # --drive_source cpg: the panel's owner is a SUB-NETWORK, and
                # its input is a weighted mix of CPG channels rather than one
                # timing unit. Draw the channels carrying 90% of its drive,
                # each at an alpha set by its share relative to the largest,
                # and colour by CHANNEL so a rug can be cross-referenced
                # against drive_gates.png's x axis. The dominant channel's
                # rug is fully opaque, so one solid rug = concentrated and
                # many mid-alpha rugs = unselected.
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
    """
    Time folded onto cycle phase.  Removes the 'which cycle' axis so
    alignment is a single picture per TIMING UNIT: the trajectory of the
    column(s) that unit is responsible for over one cycle, with its own
    spike-phase histogram behind it.

    One row per timing unit, indexed through `timing_cols` -- not per
    sub-network through group_cols, which is a different length once
    --timing_shape and --decoder_shape differ. With per_joint timing each row
    is a single joint and the reference fundamental is that joint's own, which
    is the sharper alignment measure; with per_leg timing a row shows the
    whole leg, as before.

    `drive_share` switches to --drive_source cpg mode: pass that gait's
    (G, C) normalised drive and `timing_cols=group_cols`. A row is then a
    SUB-NETWORK, and its histogram pools the CPG channels feeding it with
    each spike WEIGHTED by that channel's share -- so the circular mean
    becomes the sub-network's effective input phase, the quantity that
    should line up with its joint's peak velocity. A channel supplying 40%
    of the drive moves that mean twice as far as one supplying 20%, which a
    plain unweighted pool over the same channels would get wrong.
    """
    # Rows are sub-networks when drive_share is given (tspk may then be
    # (L, G, C), whose axis 1 already is G) and timing units otherwise.
    G   = (drive_share.shape[0] if drive_share is not None
           else tspk.shape[1])
    tbl = gait_tables[gait_idx]
    R   = tbl.shape[0]
    ok  = np.isfinite(phase)

    # Shorter rows once there are many units: per_joint timing gives 18 of
    # them, and 2.5in each is a 45-inch figure no one can read.
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
    """
    Three heatmaps, gait x timing-neuron: absolute residual, firing rate, and
    concentration R.  The residual panel is the successor to the old routing
    measurement -- it is the number that says whether the learned per-gait
    routing found an alignment the deleted fixed permutation could not.
    """
    ng   = len(gait_names)
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
        ax.set_xticklabels([f"T{j}" for j in range(G)], fontsize=8)
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
    """
    Per-gait CPG-channel selection, for --drive_source cpg.  The counterpart
    of plot_routing, and the central figure for that mode.

    Reads the parameters directly, which is legitimate here in a way the old
    routing weight heatmap was not: s_gate and u sit BEFORE the sum over
    channels, so their product IS that channel's contribution to the drive --
    there is no threshold or membrane between them and their effect to make
    the number misleading.  (plot_routing has to probe precisely because a
    timing unit sits in the way.)

    The quantity plotted is `drive_strength`, |s_gate| * ||u[g,c,:]||_1, not
    |s_gate| alone: the two are multiplicatively redundant, so s_gate by
    itself carries only part of the scale.

    One panel per gait, rows = sub-network, columns = CPG channel.  Channel
    index is essentially cycle phase (channel c bursts at ~c/n_cpg), so phase
    runs along x.  Values are normalised across each ROW, so a row reads as
    "what fraction of this joint's drive comes from each phase".  The panel
    title carries the mean effective fan-in, because a flat-but-dim panel and
    a concentrated one are otherwise hard to distinguish by eye -- the colour
    scale is data-driven, so both fill it.  Absolute
    magnitude is deliberately discarded: `u` can absorb overall scale, so
    only the relative split is meaningful.

    Reading it, given that CPG channel c bursts at cycle phase ~c/n_cpg:
      - a column concentrated on few rows => that joint samples at a few
        phases and is silent elsewhere. This is what --drive_lambda drives
        toward, and because CPG channels are phase-localised it is a
        TEMPORAL claim, not merely a structural one.
      - the annotated argmax per column is the dominant phase for that
        joint; comparing it to the joint's peak-velocity phase is the
        alignment measurement, and unlike the timing-layer version the
        reference phases here are fixed and known rather than themselves
        measured.
      - columns DIFFERING across gaits => the per-gait selection is being
        used. Identical everywhere => gaits are not differentiating.
      - several joints sharing a dominant channel is EXPECTED: that is what
        a tripod is.
    """
    if getattr(model, "uses_timing", True):
        return None
    # drive_strength (= |s_gate| * ||u[g,c,:]||_1), not |s_gate| alone: the
    # two enter the forward pass only as a product and u's row norm varies
    # per (g,c). Reading |s_gate| made every sub-network look dominated by
    # channel 0, because s_gate inits to all-ones and argmax returns index 0
    # on an exact tie. Shared with train.py's drive_report so the figure and
    # the printed numbers are the same quantity.
    G, C = model.G, model.n_neurons
    ng = len(gait_names)
    # drive_strength is indexed by the model's REAL gait index. `gait_names`
    # is the plotted list, which --gaits can subset or reorder, so taking
    # S[:ng] and indexing by loop position silently paired each panel with
    # the wrong gait's drive. Select the real rows instead.
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
            f"{Dall.shape[0]} rows. drive_strength is sized by the MODEL's "
            f"n_gaits, so a --gaits selection naming a gait the model was "
            f"not trained on cannot be plotted here.")
    S = Dall[gidx]                                           # (ng, G, C)

    # Normalise every panel first, then share one colour scale across them
    # set from the data. A fixed vmax=1.0 would wash the whole figure out:
    # a sub-network split evenly between two channels tops out at 0.5, so
    # nothing ever reaches 1 and the interesting structure sits in the
    # bottom half of the colormap.
    Mn = []
    for gi in range(ng):
        # (G, C): rows = sub-network, cols = CPG channel. Channel index is
        # essentially cycle PHASE (channel c bursts at ~c/n_cpg), and phase
        # belongs on the x axis.
        M = S[gi]
        row = M.sum(axis=1, keepdims=True)
        Mn.append(M / np.where(row > 0, row, 1.0))     # share per sub-network
    vmax = max(float(m.max()) for m in Mn) or 1.0

    # Two DIFFERENT concentration metrics, both shown and each named:
    #
    #   fanin90 -- channels needed to reach 90% of the drive. The metric
    #              drive_report prints, metrics.csv logs, and check_subnet
    #              reports, so it is the one to put first.
    #   PR      -- participation ratio, (sum d)^2 / sum d^2. What
    #              drive_penalty("pr") actually minimises.
    #
    # They are not interchangeable and PR is systematically LOWER on a
    # skewed row: one dominant channel plus a thin tail gives PR ~3 where
    # fanin90 is ~15. This title used to print PR and call it "fan-in",
    # which read as a contradiction of every other fan-in number in the
    # pipeline. Same numbers as before, honestly labelled, plus the one
    # that matches the rest of the tooling.
    pr = [float((m.sum(axis=1) ** 2 / np.maximum((m ** 2).sum(axis=1), 1e-12))
                .mean()) for m in Mn]
    f90m, actm = [], []
    for m in Mn:
        srt = -np.sort(-m, axis=1)
        f90m.append(float(((np.cumsum(srt, axis=1) < 0.90).sum(axis=1) + 1)
                          .mean()))
        actm.append(float((m > 0).sum(axis=1).mean()))

    # Near-square grid rather than 1 x ng. Each panel is G x C (18 x 18 on
    # the hexapod), so a single row of them makes every panel tall and thin
    # and the two tick axes end up on wildly different scales. ncols from
    # sqrt keeps panels close to square: 4 gaits -> 2x2, 6 -> 2x3, 8 -> 3x3
    # with one cell hidden.
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
                # White, not red: the dominant cell is the BRIGHTEST one, so
                # a dark-palette marker on it is invisible.
                ax.text(d, j, "x", ha="center", va="center",
                        color="w", fontsize=5, fontweight="bold")
        if gi % ncols == 0:
            ax.set_ylabel("sub-network", fontsize=7)
    if im is not None:
        fig.colorbar(im, ax=[a for a in axes if a.get_visible()],
                     fraction=0.02,
                     label="share of this sub-network's drive")
    fig.suptitle("Per-gait CPG channel selection "
                 "(|s_gate|*||u||, normalised per sub-network)   "
                 "x = dominant channel\n"
                 "fanin90 = channels carrying 90% of the drive (the metric "
                 "drive_report prints and metrics.csv logs);   "
                 "PR = (sum d)^2 / sum d^2, what --drive_penalty pr "
                 "minimises.  PR reads LOWER on a skewed row.",
                 fontsize=9, fontweight="bold")
    _savefig(fig, out_dir, "drive_gates.png", dpi)

    # Effective fan-in per (gait, sub-network): how many channels carry most
    # of the drive. The number --drive_lambda is meant to reduce, so it is
    # worth printing rather than leaving to be eyeballed off the heatmap.
    #
    # train.py's own drive_report is called verbatim rather than
    # reimplemented here. Two copies of the same statistic in two files
    # cannot be compared safely: this file used to print min/MEDIAN/max
    # while train.py printed MEAN, and on a skewed fan-in distribution
    # (most sub-networks at 1, a few much higher) those differ by a lot with
    # nothing wrong in either. Calling the one function makes the training
    # printout and this figure agree by construction, so any remaining
    # difference is the MODEL -- a checkpoint that is not the one that
    # produced the training log -- and not the measurement.
    # Only when the plotted set is the model's own gait order: drive_report
    # labels its row gi with names[gi], so a --gaits subset or reordering
    # would attach the right numbers to the wrong gait name.
    if gidx == list(range(ng)):
        # Whichever reporter the training log used, called verbatim so the
        # figure and the log cannot disagree. drive_spike_report needs the
        # CPG train, so it is only callable when one was passed in.
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

    # Per-sub-network values too, so the comparison can be elementwise
    # instead of going through any summary statistic at all.
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
    """
    Effective CPG -> timing routing, MEASURED rather than read off a weight.

    There is no per-gait routing matrix to inspect any more: the path is
    a shared LIF router with a per-gait FiLM gate, so the routing is a
    property of the whole layer's dynamics, not of one parameter. It is
    recovered by probing -- for each gait and each CPG neuron, feed a
    synthetic burst on that neuron alone and count which timing units
    respond.

    This is strictly more honest than the old weight heatmap even when a
    weight existed: it reports what the layer DOES (after thresholds,
    membrane decay and the gate) rather than what one matrix contains.

    The annotated argmax per column is the strongest CPG driver for each
    timing unit. Reading it:
      - orderings that DIFFER across gaits => the per-gait gate is being
        used, which is the whole reason the router exists
      - orderings identical everywhere => gaits are not differentiating,
        and either the gate has collapsed or the gait set genuinely does
        not need distinct routings
      - several timing units sharing one driver is EXPECTED, not a bug:
        that is what a tripod is (3 legs at one phase, 3 at the opposite).
    """
    # Under drive_source="cpg" there is no router to probe: timing_only
    # returns the CPG train unchanged, so this would plot an identity
    # diagonal, identical in every gait, which looks like a result and is
    # not one. plot_drive_gates is the figure for that mode.
    if not getattr(model, "uses_timing", True):
        print("  [routing] drive_source=cpg: no CPG->timing router to probe, "
              "skipping (see drive_gates.png instead)")
        return None

    n_cpg, n_t = model.n_neurons, model.n_timing
    ng = len(gait_names)
    P  = int(period)

    # Synthetic burst probe: 10 spikes ~3 steps apart, matching the CPG's
    # measured burst shape, repeated for a few cycles so membranes settle.
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
        print("    All gaits share one routing — the per-gait FiLM gate is "
              "not differentiating them. Either it has collapsed (check the "
              "gate values) or this gait set doesn't need distinct routings.")
    else:
        print(f"    {len(uniq)} distinct routings across {len(argmax_tbl)} "
              f"gaits — the per-gait gate is being used.")
    shared = [d for d in uniq if len(set(d)) < n_t]
    if shared:
        print(f"    {len(shared)}/{len(uniq)} routings have timing units "
              f"sharing a CPG driver (expected: that is what a tripod is).")
    return argmax_tbl


def plot_taus(model, period, cfg, out_dir, dpi):
    """Learned time constants against their init ranges and the CPG period.
    The question this answers: is --tau_timing_max binding, and did the
    sub-network taus actually spread across the cycle."""
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
               # 40.0 (not period-derived) matches build_model_from_cfg's
               # fallback: that was the literal hardcoded value before this
               # became a derived option, so it's the right thing to draw as
               # the init-range marker for a config that predates it.
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
    """
    Sub-network membrane traces for a few sampled units per group.

    Membranes, not spikes, on purpose: see the module docstring -- exact
    hidden rasters are not recoverable from a checkpoint, and a guessed
    raster would be worse than an honest membrane trace.
    """
    if not mems:
        return
    stacks = [np.stack([m[k] for m in mems[:n_show]]) for k in range(len(mems[0]))]
    rng = np.random.default_rng(seed)

    # Skip the router/timing membranes: they are tiny and already fully
    # covered by the exact rasters elsewhere. This plot is about the
    # sub-networks, whose spikes are NOT recoverable (see module docstring).
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
                     f"(colour = group)", fontsize=8)
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
    ap.add_argument("--ckpt",      type=str, default="best_model.pt")
    ap.add_argument("--cfg",       type=str, default="cpg_lif_snn_config.json")
    ap.add_argument("--out_dir",   type=str, default=None,
                    help="Where plots and timing_summary.json are written, "
                         "resolved as outputs/<out_dir> (same rule as "
                         "--model_dir). Default None = the model dir itself, "
                         "so figures join that run's existing category "
                         "folders and nothing needs changing between runs "
                         "besides --model_dir.")
    ap.add_argument("--gaits_dir", type=str, default="../gaits",
                    help="Folder of {name}.csv gait tables, resolved as "
                         "this_file_dir/<gaits_dir> — same default as "
                         "train.py's --gaits_dir.")
    ap.add_argument("--drive_metric", type=str, default="auto",
                    choices=["auto", "product", "gain", "spikes"],
                    help="[--drive_source cpg / cpg_lif] Which drive "
                         "representation the figures and printouts use. "
                         "'auto' (default) follows the checkpoint: 'product' "
                         "(|s_gate|*||u||, with fanin90) under cpg, 'spikes' "
                         "(exact per-unit spike counts, no cutoff) under "
                         "cpg_lif. Override to compare representations on one "
                         "checkpoint: 'gain' plots |s_gate| alone, which is "
                         "the penalised quantity under cpg_lif but is "
                         "MEANINGLESS under cpg, where u can rescale it "
                         "freely.")
    ap.add_argument("--gaits",     type=str, nargs="*", default=None,
                    help="Gait names to plot (default: all in the config).")
    ap.add_argument("--recon", type=int, default=1,
                    help="1 = also regenerate recon_<gait>.png and "
                         "transition_*.png. This is how an ALREADY-TRAINED "
                         "checkpoint gets re-plotted with a different "
                         "--recon_cycles; train.py passes 0 because it has "
                         "just made them itself.")
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
                         "the plots show settled behaviour rather than the "
                         "zero-state transient.")
    ap.add_argument("--fold_cycles", type=float, default=40.0,
                    help="Cycles accumulated for the phase-fold histograms "
                         "and the alignment statistics. More is better here — "
                         "it only costs CPU and it tightens R.")
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
    """
    The CLI defaults as a plain namespace, so train.py can call
    run_visualization without going through argparse.
    """
    a = _build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def run_visualization(model_dir, out_dir=None, args=None):
    """
    Everything the CLI does, callable directly.

    `out_dir=None` means the run directory itself, which is what train.py
    passes explicitly too -- the figures land in that run's own category
    folders (see OUT_ROUTES in train.py) alongside its other output.

    The checkpoint and config are re-read from disk rather than taking a live
    model, which also serves as an end-of-run check that the saved artifacts
    actually load and run.
    """
    args = args if args is not None else default_args()
    model_dir = Path(model_dir)
    # Default is the run directory ITSELF, not a nested visualize/ folder:
    # out_path routes every figure into recons/ timing_alignments/
    # membrane_waveforms/ phase_folds/ misc_info/ inside it, so a subfolder
    # would just bury those one level deeper. Standalone and
    # called-from-train.py therefore write to the same place.
    out_dir = Path(out_dir) if out_dir is not None else Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        # Older config, predates recording "gait_files" — assume it used
        # the standard set for its species, exactly as train.py's own
        # fallback does when --gaits isn't given.
        n_cpg = int(cfg_get(cfg, "n_cpg_neurons", 4))
        gait_files = GAIT_FILES_BY_N.get(n_cpg)
        if gait_files is None:
            raise ValueError(
                f"Config has no 'gait_files' and n_cpg_neurons={n_cpg} has "
                f"no standard gait set (know {sorted(GAIT_FILES_BY_N)}).")
        print(f"  NOTE: config has no 'gait_files' key — assuming the "
              f"standard n_cpg_neurons={n_cpg} set: {gait_files}")
    # Relative to THIS file, not to model_dir, so it resolves the same
    # whether called from the CLI or from train.py.
    gaits_dir = Path(os.path.dirname(os.path.abspath(__file__)),
                     args.gaits_dir)
    # Anatomical leg layout, used for the timing-alignment grid. Falls back
    # to the network's own grouping for configs written before leg_cols was
    # recorded -- which is exact whenever n_timing == n_legs, the only case
    # that existed then.
    cfg_leg_cols = cfg_get(cfg, "leg_cols")

    gait_tables_orig, all_names = load_gait_tables(gait_files, gaits_dir)
    print(f"  Loaded {len(all_names)} gait CSV(s) from {gaits_dir.resolve()}: "
          f"{all_names}")
    leg_cols = ([list(c) for c in cfg_leg_cols] if cfg_leg_cols
                else [list(c) for c in model.group_cols])
    print(f"  Alignment grid: {len(leg_cols)} legs x {len(leg_cols[0])} "
          f"joints/leg"
          + ("" if cfg_leg_cols else "  (leg_cols absent from config — "
                                     "using the network grouping)"))

    # ── timing units vs sub-networks ─────────────────────────────
    # These are two different counts now (--timing_shape vs --decoder_shape),
    # and conflating them is what breaks this file: every per-timing-neuron
    # loop was indexing `group_cols`, which is as long as the SUB-NETWORK
    # count. At timing per_joint + decoder per_leg that is 18 lookups into a
    # 6-entry list.
    #
    #   group_cols[g]  : output columns sub-network g EMITS
    #   timing_cols[t] : output columns timing unit t is RESPONSIBLE FOR,
    #                    i.e. what its phase should align to
    #   timing_map[g]  : which timing units feed sub-network g
    #
    # timing_cols falls back to group_cols, which is exact for every config
    # written before --timing_shape existed: there was one timing unit per
    # sub-network, so a unit's columns WERE its group's columns.
    group_cols  = [list(g) for g in model.group_cols]
    cfg_tcols   = cfg_get(cfg, "timing_cols")
    timing_cols = ([list(c) for c in cfg_tcols] if cfg_tcols
                   else [list(g) for g in group_cols])
    timing_map  = cfg_get(cfg, "timing_map")
    if arch == "timing_grouped":
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
        print(f"\n  arch={arch} has NO timing layer — emitting the CPG, "
              f"gait-table and tau figures only.")

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
    # Computed once before the loop rather than per gait: drive_strength
    # returns all gaits at once, and it is a pure parameter read. None under
    # --drive_source timing, which leaves every plot call unchanged.
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
                 "  (|s_gate| alone -- the penalised quantity under cpg_lif)"))

        if drive_metric == "spikes":
            # Shares come from SPIKE COUNTS, which is the point of cpg_lif:
            # "channel c delivers nothing" is the exact integer statement
            # "unit (g,c) emitted zero spikes", with no threshold to pick.
            # The WHOLE replay, not a fixed prefix. This used to take
            # spikes[:6*period] -- a window anchored at t=0 -- while the
            # time-axis figures plot [t_lo, t_hi), which starts after
            # --warm_cycles and runs --n_cycles. With the defaults (3 and 4)
            # that window is [3P, 7P), so everything past 6P came out as
            # zeros and the last plotted period was blank. Replaying all of
            # it also means the LIF's membrane state at t_lo is the real one
            # rather than a fresh zero, and the shares below are measured
            # over the same steps the figures show.
            L = spikes.shape[0]
            xs = torch.as_tensor(spikes[:L], dtype=torch.float32,
                                 device=device).unsqueeze(1)
            cnts, trains = [], []
            for gi in range(len(all_names)):
                gt = torch.full((L, 1), gi, dtype=torch.long, device=device)
                sp = model.drive_spikes_only(xs, gt)         # (L,1,G,C)
                tr = sp[:, 0].cpu().numpy()                  # (L,G,C)
                trains.append(tr)
                # Counted over the plotted window only, so the shares that
                # pick and shade the rugs describe exactly the steps drawn.
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
    for gi, gname in sel:
        print(f"  {gname}:")
        gt = gt_degrees(gait_tables, phase, gi, phase_zero)

        pred, mems = None, []
        if not args.no_pred:
            # Only free-run as far as the plotted window: the phase-fold
            # statistics need timing spikes (cheap), not predictions, so
            # stepping the whole fold window through the sub-networks would
            # be ~4x the work for nothing.
            p, mems = predict_and_membranes(
                model, spikes[:t_hi], gi, device, warm, tgt_range,
                keep_membranes=not args.no_membranes)
            pred = np.full(gt.shape, np.nan, dtype=np.float64)
            pred[warm:warm + len(p)] = p
            rmse = np.sqrt(np.nanmean(
                (pred[t_lo:t_hi] - gt[t_lo:t_hi]) ** 2, axis=0))
            print(f"    per-column RMSE (°): " +
                  "  ".join(f"c{c}={rmse[c]:.2f}" for c in range(len(rmse))))

        # Membranes are NOT timing-specific -- every arch has mem1/mem2/memo
        # -- so this runs before the timing-only bail below.
        if mems and not args.no_membranes:
            plot_membranes(mems, [n.replace("_in", "")
                                  for n in model.state_names_in],
                           gname, out_dir, args.dpi, seed=args.seed)

        if arch != "timing_grouped":
            continue

        # G here is the TIMING-unit count (raster lanes), not the
        # sub-network count. Everything below indexes timing_cols, which is
        # this long by construction.
        tspk = timing_raster(model, spikes, gi, device)
        G    = tspk.shape[1]

        # --drive_source cpg: the raster lanes are CPG channels, and a panel
        # belongs to a SUB-NETWORK fed by a weighted mix of them -- so the
        # row map becomes group_cols and the per-panel overlay becomes that
        # sub-network's drive shares. Without this the plots would pair
        # channel N with sub-network N, which is meaningless and only
        # type-checks because n_cpg happens to equal G.
        if drive_all is not None:
            rows_map = group_cols
            d_share  = drive_all[gi] if gi < len(drive_all) else None
        else:
            rows_map, d_share = timing_cols, None

        # Under cpg_lif the rug must show the DRIVE LIF's spikes, not the raw
        # CPG channel's: the unit thins and delays its channel's burst, so
        # the two differ, and only the former is what the sub-network
        # actually received. tspk_g is (L, G, C) there, (L, lanes) otherwise.
        tspk_g = tspk
        if drive_spk_all is not None and gi < len(drive_spk_all):
            tr = drive_spk_all[gi]                            # (Ld, G, C)
            Ld = min(tr.shape[0], tspk.shape[0])
            tspk_g = np.zeros((tspk.shape[0],) + tr.shape[1:], dtype=tr.dtype)
            tspk_g[:Ld] = tr[:Ld]

        # frac=None under the spikes metric: "active" is then exact, so
        # every channel that fires should be drawn rather than a 90% subset.
        d_frac = None if drive_metric == "spikes" else 0.90
        plot_alignment(spikes, tspk_g, gt, pred, phase, onsets, period,
                       burst_thr, rows_map, leg_cols, gname, out_dir,
                       args.dpi, t_lo, t_hi, drive_share=d_share,
                       drive_frac=d_frac)
        plot_phase_fold(tspk_g, phase, gait_tables, gi, rows_map, gname,
                        phase_zero, out_dir, args.dpi, drive_share=d_share,
                        drive_frac=d_frac)

        # statistics over the full fold window
        ok    = np.isfinite(phase)
        ncyc  = max(np.isfinite(phase).sum() / period, 1e-9)
        tbl   = gait_tables[gi]
        x_tbl = ((np.arange(tbl.shape[0]) / tbl.shape[0]) - phase_zero) % 1.0
        order = np.argsort(x_tbl)
        rows  = []
        # Under cpg_lif the per-row train is the COLLAPSED one: the OR over
        # that sub-network's C drive units. Valid as a single train because
        # the units are mutually exclusive (one channel each, zero reset,
        # CPG channels never coincide), and it is exactly the input-event
        # train the sub-network receives -- directly comparable to the old
        # timing layer's spk/cyc. Without this the loop would report the raw
        # CPG channel's phase, which the drive LIF has thinned and delayed.
        coll = None
        if tspk_g.ndim == 3:
            coll = (tspk_g.sum(axis=2) > 0).astype(np.float32)   # (L, G)
            if float(tspk_g.sum(axis=2).max()) > 1.0 + 1e-6:
                print("    WARNING drive units of one sub-network fired on "
                      "the same timestep; the collapsed train is not valid "
                      "(needs zero reset).")
        n_row = coll.shape[1] if coll is not None else G
        for j in range(n_row):
            m = ((coll[:, j] > 0) if coll is not None
                 else (tspk[:, j] > 0)) & ok
            mu, R_ = circular_stats(phase[m])
            f_ph   = fundamental_phase(tbl[order, timing_cols[j][0]])
            res    = circ_diff(mu, f_ph)
            rows.append({
                "timing_neuron":   j,
                "cols":            list(timing_cols[j]),
                "feeds_subnets":   ([g for g, m in enumerate(timing_map)
                                     if j in m] if timing_map else [j]),
                "rate_per_cycle":  float((coll[:, j] if coll is not None
                                          else tspk[:, j]).sum() / ncyc),
                "mean_phase":      None if not np.isfinite(mu) else float(mu),
                "R":               float(R_),
                "leg_fundamental_phase": None if not np.isfinite(f_ph) else float(f_ph),
                "residual_cycles": None if not np.isfinite(res) else float(res),
                "residual_steps":  None if not np.isfinite(res) else float(res * period),
                "dead":            bool((coll[:, j] if coll is not None
                                         else tspk[:, j]).sum() == 0),
            })
            tag = "  <-- DEAD" if rows[-1]["dead"] else ""
            print(f"    T{j} (cols {timing_cols[j]}): "
                  f"rate={rows[-1]['rate_per_cycle']:6.2f}/cyc  "
                  f"phase={mu:.3f}  R={R_:.2f}  "
                  f"leg_fund={f_ph:.3f}  "
                  f"residual={res:+.3f} cyc ({res * period:+.0f} steps){tag}")
        summary[gname] = rows
        plotted.append(gname)

    # ── 4. cross-gait figures ───────────────────────────────────
    print("\n[4/5] Cross-gait figures ...")
    routing = drive_gates = None
    if arch == "timing_grouped" and summary:
        plot_alignment_summary(summary, plotted, model.n_timing,
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
            # Which per-unit history exists depends on the drive mode, and
            # each plotter no-ops with a printed note if its columns are
            # absent -- so calling both is safe and covers a config that
            # does not record drive_source.
            if getattr(model, "uses_timing", True):
                plot_timing_rate_history(metrics_csv, timing_cols, leg_cols,
                                         plotted, out_dir, args.dpi)
            else:
                plot_drive_fanin_history(metrics_csv, group_cols, leg_cols,
                                         plotted, out_dir, args.dpi)
        else:
            print(f"  [unit history] {metrics_csv} not found, skipping")
    plot_taus(model, period, cfg, out_dir, args.dpi)

    # ── 4b. reconstruction + transitions ────────────────────────
    # These are the figures train.py makes at the end of a run, regenerated
    # here so an ALREADY-TRAINED model can be re-plotted -- which is the only
    # way to apply a changed --recon_cycles to an old checkpoint.
    #
    # targets/valid come from train.py's own build_targets, on the same phase
    # array used above, so the targets are constructed identically to training
    # rather than re-derived here and free to drift.
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
            # Better a loud warning than plots that quietly run off the end of
            # the replay. --n_steps lengthens the replay; --recon_cycles
            # shortens the window.
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
    print("\n[5/5] Writing timing_summary.json ...")
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
        # Populated only under drive_source="cpg"; None otherwise, exactly as
        # learned_routing_argmax is None in that mode.
        "drive_gates": drive_gates,
        "residual_note": ("residual = timing-neuron circular mean phase minus "
                          "the phase of the first Fourier component of that "
                          "leg's first gait-table column; a consistent "
                          "reference, not a biomechanical footfall event"),
    }
    p = out_path(out_dir, "timing_summary.json")
    p.write_text(json.dumps(blob, indent=2))
    print(f"    [saved] {p}")
    print(f"\nDone — {out_dir.resolve()}")


if __name__ == "__main__":
    main()