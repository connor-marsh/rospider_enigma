"""
Visualization flows and their shared command line: run_visualization (diagnostic
figures and alignment scores) and run_publication (paper figures). visualize.py
picks one; train.py calls run_visualization at the end of a run.

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
from pathlib import Path

import numpy as np
import torch

from .analysis import (
    _ALIGN_COR_TAU, _ALIGN_VEL_TAU, alignment_pair, alignment_section,
    circ_diff, circular_stats, fundamental_phase, gt_degrees,
    predict_and_membranes, timing_raster,
)
from .common import PROJECT_DIR, cfg_get, out_path
from .cpg import cpg_phase, replay_cpg
from .data import (
    GAIT_FILES_BY_N, build_targets, default_leg_layout, load_gait_tables,
    upsample_gait_tables,
)
from .plotting import (
    PUB_RECON_GAITS, PUB_TRANSITION_SEQ, fig_reconstruction, fig_timing_alignment,
    fig_training_curves, fig_transitions, plot_alignment, plot_alignment_history,
    plot_alignment_summary, plot_drive_fanin_history, plot_drive_gates, plot_joint_gait_grid,
    plot_membranes, plot_phase_fold, plot_reconstruction, plot_routing,
    plot_taus, plot_timing_rate_history, plot_transition, publication_style,
    resolve_gaits,
)
from .run_io import load_run


# ═══════════════════════════════════════════════════════════════════
# Visualization flows
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
        description="Visualize a trained CPG-SNN run: diagnostic figures, or "
                    "publication figures with --publication.")
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
    # ── publication mode ──
    ap.add_argument("--publication", action="store_true",
                    help="Make the publication figures instead of the diagnostic ones: vector "
                         "PDF, larger type and paper layouts, written to <out_dir>/publication/. "
                         "Uses the --pub_* options, --n_steps and --font_scale below.")
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
    OUT_ROUTES in submodules/common.py). The checkpoint and config are re-read from disk rather than
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
    gaits_dir = Path(PROJECT_DIR,
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
    # from data.build_targets, so they are built exactly as in training.
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


# ═══════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════

def run_publication(model_dir, out_dir=None, args=None):
    """Everything the CLI does, callable from visualize.py."""
    from .cpg import replay_cpg, cpg_phase

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
    gaits_dir = Path(PROJECT_DIR,
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
            gi = resolve_gaits([nm], all_names, "--pub_align_gaits")[0]
            for dense in (False, True):
                fig_timing_alignment(
                    model, spikes, tables, device, tgt_range, all_names,
                    leg_cols, period, onsets, out_dir, gait=gi,
                    n_cycles=args.pub_cycles, dense=dense)
    print("\nDone.")


def _split(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


build_parser = _build_parser          # public name for visualize.py