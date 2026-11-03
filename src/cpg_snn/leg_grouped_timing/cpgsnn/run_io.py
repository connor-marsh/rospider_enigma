"""
Saving and loading a run: the run config written before and after training, rebuilding a
model from a config (with the fallbacks for older configs), loading a checkpoint, and
ONNX export.
"""

import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .common import cfg_get, git_info, in_path, json_safe, out_path
from .cpg import CPG_FROM_FB_WEIGHT, cpg_weight_matrix
from .data import N_JOINTS, N_LEGS, build_group_cols
from .model import DenseSNN, SingleStepONNX, SingleStepONNXTiming, TimingGroupedSNN


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
# ONNX export
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
        # change what visualize.py loads.
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
            "script":         os.path.abspath(sys.argv[0]),
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
