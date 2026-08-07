"""
Visualise a trained CPG -> timing-layer -> grouped-SNN run.
===========================================================

Loads a checkpoint (.pt) and its config (.json) written by train.py, replays
the CPG bit-identically from the config's own parameters, and plots what the
timing layer is doing against the CPG and against the gait tables.

Usage
-----
    python visualize.py                              # outputs/  ->  outputs/visualize
    python visualize.py --model_dir outputs/run7
    python visualize.py --gaits wkF bk               # subset
    python visualize.py --n_cycles 4 --no_pred       # tighter / faster


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
    routing_matrices.png          Learned per-gait CPG->timing weights, i.e.
                                  what replaced the old fixed permutation.
    tau_distributions.png         Learned time constants vs their init range
                                  and vs the CPG period.
    membranes_<gait>.png          Sub-network membrane traces (sampled units).
    timing_summary.json           Every number in the figures, machine-readable.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from train import (
    StatefulSNN, TimingGroupedSNN,
    LIFCPGStepper, cpg_weight_matrix,
    detect_burst_threshold, burst_onsets,
    upsample_gait_tables, build_group_cols,
    load_gait_tables, GAIT_FILES_BY_N,
    N_LEGS, N_JOINTS, CPG_PALETTE, CPG_FROM_FB_WEIGHT,
)

TIMING_PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261",
                  "#6a0572", "#8ecae6", "#ffb703", "#023047"]
GT_COLOR   = "#457b9d"
PRED_COLOR = "#e63946"


# ═══════════════════════════════════════════════════════════════════
# 1.  Config / checkpoint loading
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
        # router_hidden default 16 matches train.py; a config from before the
        # router existed has no such key and cannot be loaded into this class
        # at all (its checkpoint has w_in_gait, which no longer exists), so
        # load_run's strict=False report is what will flag that.
        model = TimingGroupedSNN(
            hidden_per_group = int(cfg_get(cfg, "hidden", 256)),
            n_timing         = n_timing,
            group_cols       = group_cols,
            tau_timing_min   = float(cfg_get(cfg, "tau_timing_min", 2.0)),
            tau_timing_max   = float(cfg_get(cfg, "tau_timing_max", 64.0)),
            router_hidden    = int(cfg_get(cfg, "router_hidden", 16)),
            readout_hidden   = int(cfg_get(cfg, "readout_hidden", 32)),
            tau_router_min   = float(cfg_get(cfg, "tau_router_min", 2.0)),
            tau_router_max   = float(cfg_get(cfg, "tau_router_max", 64.0)),
            timing_slope     = float(cfg_get(cfg, "timing_slope", 5.0)),
            timing_w_scale   = float(cfg_get(cfg, "timing_w_scale", 0.5)),
            # sub_ln changes FORWARD BEHAVIOUR, not shapes, so a wrong value
            # loads cleanly and then quietly computes something else.
            sub_ln           = str(cfg_get(cfg, "sub_ln", "l2")),
            **common)
    else:
        model = StatefulSNN(hidden=int(cfg_get(cfg, "hidden", 256)), **common)

    return model.to(device), arch


def load_run(model_dir, ckpt_name, cfg_name, device):
    model_dir = Path(model_dir)
    cfg_path  = model_dir / cfg_name
    ckpt_path = model_dir / ckpt_name
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
# 2.  Replay the CPG from the config
# ═══════════════════════════════════════════════════════════════════

def replay_cpg(cfg, n_steps):
    """
    Re-run the exact CPG the model was trained against.

    Every parameter comes out of cfg["cpg"], including the weight matrix, so
    this is reproducible even if train.py's defaults change later.  The warm-up
    is replayed too: burst phase depends on it.
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

    warmup = int(c.get("warmup", 2000))
    cpg.step_chunk(warmup)
    spikes = cpg.step_chunk(n_steps)

    counts = spikes.sum(0).astype(int)
    print(f"  CPG replayed: N={N}  warmup={warmup}  steps={n_steps}  "
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

def circular_stats(phases):
    """Circular mean in [0,1) and concentration R in [0,1] of a set of
    cycle phases.  R is the thing to read: a unit can fire at a healthy rate
    and still be useless if R is low, because then it is firing all over the
    cycle and its mean phase means nothing."""
    p = np.asarray(phases, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan"), 0.0
    z = np.mean(np.exp(1j * 2.0 * np.pi * p))
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
    p = Path(out_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {p}")


def plot_alignment(spikes, tspk, gt, pred, phase, onsets, period, burst_thr,
                   group_cols, gait_name, out_dir, dpi, t_lo, t_hi):
    """
    The main figure: CPG raster, timing raster, and one row per group showing
    that group's gait-table columns with its OWN timing neuron's spikes drawn
    over them.  Shared x axis, so vertical alignment is the whole point --
    if timing neuron g fires at a consistent place in leg g's cycle, it is
    visible directly.
    """
    n_cpg = spikes.shape[1]
    G     = tspk.shape[1]
    sl    = slice(t_lo, t_hi)
    t     = np.arange(t_lo, t_hi)

    fig = plt.figure(figsize=(17, 3.4 + 1.9 * G))
    gs  = gridspec.GridSpec(2 + G, 1, hspace=0.55,
                            height_ratios=[1.1, 1.1] + [1.5] * G)
    axes = [fig.add_subplot(gs[0])]
    for i in range(1, 2 + G):
        axes.append(fig.add_subplot(gs[i], sharex=axes[0]))

    # Shared x axis via sharex= does not auto-hide tick labels on the upper
    # subplots (that only happens with plt.subplots(sharex=True)), so every
    # row was showing its own numeric time labels sitting right above the
    # next row's title. Only the bottom row needs them.
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    on_win = onsets[(onsets >= t_lo) & (onsets < t_hi)]

    def cycle_lines(ax):
        for b in on_win:
            ax.axvline(b, color="k", lw=0.9, alpha=0.35, ls="--", zorder=0)

    # ── CPG raster ────────────────────────────────────────────────
    ax = axes[0]
    for i in range(n_cpg):
        idx = np.where(spikes[sl, i] > 0)[0] + t_lo
        ax.scatter(idx, np.full(len(idx), i), marker="|", s=110, lw=1.5,
                   color=CPG_PALETTE[i % len(CPG_PALETTE)])
    ax.set_yticks(range(n_cpg))
    ax.set_yticklabels([f"CPG {i}" for i in range(n_cpg)], fontsize=8)
    ax.set_ylim(-0.6, n_cpg - 0.4)
    cycle_lines(ax)
    ax.set_title(f"{gait_name} — CPG raster   "
                 f"(dashed = neuron-0 burst onset, period ≈ {period:.0f} steps)",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.15)

    # ── timing raster ─────────────────────────────────────────────
    ax = axes[1]
    for j in range(G):
        idx = np.where(tspk[sl, j] > 0)[0] + t_lo
        ax.scatter(idx, np.full(len(idx), j), marker="|", s=110, lw=1.5,
                   color=TIMING_PALETTE[j % len(TIMING_PALETTE)])
    ax.set_yticks(range(G))
    ax.set_yticklabels([f"T{j}" for j in range(G)], fontsize=8)
    ax.set_ylim(-0.6, G - 0.4)
    cycle_lines(ax)
    ax.set_title("Timing layer raster  (exact — same spikes the sub-networks saw)",
                 fontsize=9)
    ax.grid(axis="x", alpha=0.15)

    # ── one row per group ─────────────────────────────────────────
    for j in range(G):
        ax   = axes[2 + j]
        col_ = TIMING_PALETTE[j % len(TIMING_PALETTE)]
        cols = group_cols[j]

        for k, c in enumerate(cols):
            ls = "-" if k == 0 else "-."
            ax.plot(t, gt[sl, c], color=GT_COLOR, lw=1.7, ls=ls,
                    label=f"GT c{c}", zorder=3)
            if pred is not None:
                ax.plot(t, pred[sl, c], color=PRED_COLOR, lw=1.2, ls="--",
                        alpha=0.85, label=f"pred c{c}", zorder=4)

        # every timing spike as a faint tick, burst onsets as solid lines
        idx = np.where(tspk[sl, j] > 0)[0] + t_lo
        lo, hi = ax.get_ylim()
        ax.vlines(idx, lo, hi, color=col_, alpha=0.16, lw=1.2, zorder=1)
        for bst in spike_bursts(idx, burst_thr):
            ax.axvline(bst[0], color=col_, lw=1.5, alpha=0.85, zorder=2)
        ax.set_ylim(lo, hi)

        cycle_lines(ax)
        ax.set_ylabel(f"leg {j}\ncols {cols} (°)", fontsize=8)
        ax.legend(fontsize=6, ncol=4, loc="upper right")
        ax.grid(alpha=0.2)
        ax.set_title(f"leg {j} trajectory vs timing neuron T{j}  "
                     f"(solid vertical = T{j} burst onset)", fontsize=8)

    axes[-1].set_xlabel("CPG timestep")
    axes[0].set_xlim(t_lo, t_hi)
    _savefig(fig, out_dir, f"timing_alignment_{gait_name}.png", dpi)


def plot_phase_fold(tspk, phase, gait_tables, gait_idx, group_cols,
                    gait_name, phase_zero, out_dir, dpi, n_bins=72):
    """
    Time folded onto cycle phase.  Removes the 'which cycle' axis so
    alignment is a single picture per leg: the leg's trajectory over one
    cycle, with its timing neuron's spike-phase histogram behind it.
    """
    G   = tspk.shape[1]
    tbl = gait_tables[gait_idx]
    R   = tbl.shape[0]
    ok  = np.isfinite(phase)

    fig, axes = plt.subplots(G, 1, figsize=(11, 2.5 * G), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    x_tbl = ((np.arange(R) / R) - phase_zero) % 1.0
    order = np.argsort(x_tbl)

    for j in range(G):
        ax   = axes[j]
        col_ = TIMING_PALETTE[j % len(TIMING_PALETTE)]

        for k, c in enumerate(group_cols[j]):
            ax.plot(x_tbl[order], tbl[order, c], color=GT_COLOR, lw=1.9,
                    ls="-" if k == 0 else "-.", label=f"GT c{c}", zorder=3)
        ax.set_ylabel(f"leg {j} (°)", fontsize=9)
        ax.grid(alpha=0.2)

        m  = (tspk[:, j] > 0) & ok
        ax2 = ax.twinx()
        if m.sum() > 0:
            ax2.hist(phase[m], bins=n_bins, range=(0.0, 1.0),
                     color=col_, alpha=0.28, zorder=1)
            mu, R_ = circular_stats(phase[m])
            ax2.axvline(mu, color=col_, lw=2.0, ls="-", zorder=2)
            f_ph = fundamental_phase(tbl[order, group_cols[j][0]])
            res  = circ_diff(mu, f_ph)
            ax.set_title(
                f"T{j}: mean phase {mu:.3f}  R={R_:.2f}  |  "
                f"leg {j} fundamental {f_ph:.3f}  |  "
                f"residual {res:+.3f} cyc", fontsize=8)
        else:
            ax2.set_title(f"T{j}: NO SPIKES — sub-network {j} gets no input",
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
                       float(cfg_get(cfg, "tau_timing_max", 64.0))))
    tmin = float(cfg_get(cfg, "tau_min", 2.0))
    tmax = float(cfg_get(cfg, "tau_max", 256.0))
    series += [("hidden 1", tau(model.beta1_logit), tmin, tmax),
               ("hidden 2", tau(model.beta2_logit), tmin, tmax),
               ("readout",  tau(model.betao_logit), 2.0, 40.0)]

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
            if nm not in ("mem_timing", "mem_router")]
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

def main():
    ap = argparse.ArgumentParser(
        description="Visualise the timing layer of a trained CPG-SNN run.")
    ap.add_argument("--model_dir", type=str, default="outputs",
                    help="Directory holding the checkpoint and config.")
    ap.add_argument("--ckpt",      type=str, default="best_model.pt")
    ap.add_argument("--cfg",       type=str, default="cpg_lif_snn_config.json")
    ap.add_argument("--out_dir",   type=str, default="outputs/visualize")
    ap.add_argument("--gaits_dir", type=str, default="../gaits",
                    help="Folder of {name}.csv gait tables, resolved as "
                         "this_file_dir/<gaits_dir> — same default as "
                         "train.py's --gaits_dir.")
    ap.add_argument("--gaits",     type=str, nargs="*", default=None,
                    help="Gait names to plot (default: all in the config).")
    ap.add_argument("--n_cycles",   type=float, default=6.0,
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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = Path(this_file_dir + "/" + args.model_dir)
    out_dir   = Path(this_file_dir + "/" + args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    gaits_dir = Path(this_file_dir + "/" + args.gaits_dir)
    gait_tables_orig, all_names = load_gait_tables(gait_files, gaits_dir)
    print(f"  Loaded {len(all_names)} gait CSV(s) from {gaits_dir.resolve()}: "
          f"{all_names}")

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

        if arch != "timing_grouped":
            continue

        tspk = timing_raster(model, spikes, gi, device)
        G    = tspk.shape[1]
        group_cols = model.group_cols

        plot_alignment(spikes, tspk, gt, pred, phase, onsets, period,
                       burst_thr, group_cols, gname, out_dir, args.dpi,
                       t_lo, t_hi)
        plot_phase_fold(tspk, phase, gait_tables, gi, group_cols, gname,
                        phase_zero, out_dir, args.dpi)
        if mems and not args.no_membranes:
            plot_membranes(mems, [n.replace("_in", "")
                                  for n in model.state_names_in],
                           gname, out_dir, args.dpi, seed=args.seed)

        # statistics over the full fold window
        ok    = np.isfinite(phase)
        ncyc  = max(np.isfinite(phase).sum() / period, 1e-9)
        tbl   = gait_tables[gi]
        x_tbl = ((np.arange(tbl.shape[0]) / tbl.shape[0]) - phase_zero) % 1.0
        order = np.argsort(x_tbl)
        rows  = []
        for j in range(G):
            m = (tspk[:, j] > 0) & ok
            mu, R_ = circular_stats(phase[m])
            f_ph   = fundamental_phase(tbl[order, group_cols[j][0]])
            res    = circ_diff(mu, f_ph)
            rows.append({
                "timing_neuron":   j,
                "cols":            list(group_cols[j]),
                "rate_per_cycle":  float(tspk[:, j].sum() / ncyc),
                "mean_phase":      None if not np.isfinite(mu) else float(mu),
                "R":               float(R_),
                "leg_fundamental_phase": None if not np.isfinite(f_ph) else float(f_ph),
                "residual_cycles": None if not np.isfinite(res) else float(res),
                "residual_steps":  None if not np.isfinite(res) else float(res * period),
                "dead":            bool(tspk[:, j].sum() == 0),
            })
            tag = "  <-- DEAD" if rows[-1]["dead"] else ""
            print(f"    T{j} (cols {group_cols[j]}): "
                  f"rate={rows[-1]['rate_per_cycle']:6.2f}/cyc  "
                  f"phase={mu:.3f}  R={R_:.2f}  "
                  f"leg_fund={f_ph:.3f}  "
                  f"residual={res:+.3f} cyc ({res * period:+.0f} steps){tag}")
        summary[gname] = rows
        plotted.append(gname)

    # ── 4. cross-gait figures ───────────────────────────────────
    print("\n[4/5] Cross-gait figures ...")
    routing = None
    if arch == "timing_grouped" and summary:
        plot_alignment_summary(summary, plotted, model.n_timing,
                               out_dir, args.dpi)
        routing = plot_routing(model, all_names, device, period,
                               out_dir, args.dpi)
    plot_taus(model, period, cfg, out_dir, args.dpi)

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
        "residual_note": ("residual = timing-neuron circular mean phase minus "
                          "the phase of the first Fourier component of that "
                          "leg's first gait-table column; a consistent "
                          "reference, not a biomechanical footfall event"),
    }
    p = out_dir / "timing_summary.json"
    p.write_text(json.dumps(blob, indent=2))
    print(f"    [saved] {p}")
    print(f"\nDone — {out_dir.resolve()}")


if __name__ == "__main__":
    main()