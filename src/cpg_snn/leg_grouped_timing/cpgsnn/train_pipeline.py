"""
The steps of a training run, in the order train.py calls them.

Each step reads what it needs from `run`, a namespace that carries the run's state
between steps, and stores what later steps use. The step bodies are main()'s old
phases, unchanged.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from .analysis import drive_report, drive_spike_report, timing_report
from .common import PROJECT_DIR, in_path, outputs_path
from .cpg import CPG_PARAMS, analyse_cpg, cpg_spike_stats, cycle_phase, run_cpg
from .data import (
    GAIT_FILES_BY_N, build_partition, build_targets, build_timing_map,
    default_leg_layout, load_gait_tables, upsample_gait_tables,
)
from .model import DenseSNN, TimingGroupedSNN, _TAU_DRIVE_BURSTS, grad_blocks
from .plotting import plot_cpg_raster, plot_reconstruction, plot_training_curves, plot_transition
from .run_io import build_run_config, export_onnx, record_training_outcome, write_run_config
from .training import (
    StreamSampler, _BAND_FRAC_MIN_COUNT, _BAND_FRAC_PLAIN, _SAT_FRAC,
    calibrate_gains, make_gait_weights, make_spike_objective, run_training,
)


def setup(args, run):
    """Seeds, device and the run's output folder."""

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    this_file_dir = PROJECT_DIR
    out_dir = outputs_path(this_file_dir, args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device : {device}\nOutput : {out_dir.resolve()}")
    print(f"Arch   : {args.arch}\n")
    run.device = device
    run.out_dir = out_dir
    run.rng = rng
    run.this_file_dir = this_file_dir



def load_gaits(args, run):
    """[0/6] Gait tables, the leg layout, and the timing/decoder structure."""
    this_file_dir = run.this_file_dir

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
    run.gait_files = gait_files
    run.gait_names = gait_names
    run.gait_tables_orig = gait_tables_orig
    run.gaits_dir = gaits_dir
    run.group_cols = group_cols
    run.layout_src = layout_src
    run.leg_cols = leg_cols
    run.n_joints = n_joints
    run.n_legs = n_legs
    run.timing_cols = timing_cols
    run.timing_map = timing_map



def run_cpg_phase(args, run):
    """[1/6] Run the bursting-LIF CPG."""

    # ── 1. CPG ──────────────────────────────────────────────────
    print("\n[1/6] Bursting-LIF CPG ...")
    # Resolved once from CPG_PARAMS' keys and used for BOTH the oscillator and the config dump, so
    # they cannot describe different CPGs (cpg.replay_cpg rebuilds the oscillator
    # from the dump).
    cpg_params = {name: getattr(args, f"cpg_{name}") for name in CPG_PARAMS}
    spikes = run_cpg(N=args.n_cpg_neurons, tmax=args.tmax, warmup=args.warmup,
                     i_app=args.i_app, fake_cpg=bool(args.fake_cpg),
                     cpg_params=cpg_params)
    run.cpg_params = cpg_params
    run.spikes = spikes



def analyze_cpg(args, run):
    """[2/6] Burst structure and phase; period-derived defaults; CPG statistics."""
    out_dir, spikes = run.out_dir, run.spikes

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
    run.burst_thresholds = burst_thresholds
    run.cpg_R = cpg_R
    run.cpg_rate = cpg_rate
    run.neuron_offsets = neuron_offsets
    run.onsets = onsets
    run.period = period
    run.phase = phase



def build_targets_phase(args, run):
    """[3/6] Upsample the gait tables and build the per-step targets."""
    gait_names, gait_tables_orig, phase = run.gait_names, run.gait_tables_orig, run.phase

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
    run.align_ctx = align_ctx
    run.gait_tables = gait_tables
    run.target_rows = target_rows
    run.targets = targets
    run.tgt_range = tgt_range
    run.valid = valid



def make_samplers(args, run):
    """[4/6] Train/val stream samplers (truncated BPTT, state carried)."""
    device, gait_tables, onsets, period = run.device, run.gait_tables, run.onsets, run.period
    phase, rng, spikes, targets = run.phase, run.rng, run.spikes, run.targets
    valid = run.valid

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
    run.t_hi = t_hi
    run.t_lo = t_lo
    run.t_split = t_split
    run.tr_sampler = tr_sampler
    run.va_sampler = va_sampler
    run.warm_steps = warm_steps



def build_model(args, run):
    """[5/6] Build, freeze and compile the model, and print its structure."""
    device, gait_tables, group_cols, n_joints = run.device, run.gait_tables, run.group_cols, run.n_joints
    period, timing_map = run.period, run.timing_map

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
    run.model = model
    run.n_par = n_par



def prepare_training(args, run):
    """Gait weights and the periodic timing/drive diagnostic."""
    align_ctx, cpg_rate, device, gait_names = run.align_ctx, run.cpg_rate, run.device, run.gait_names
    gait_tables, gait_tables_orig, model, period = run.gait_tables, run.gait_tables_orig, run.model, run.period
    phase, spikes, t_lo, t_split = run.phase, run.spikes, run.t_lo, run.t_split
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
    run.gait_w = gait_w
    run.sat_rate = sat_rate
    run.timing_diag = timing_diag



def calibrate(args, run):
    """Timing-layer gain calibration (--drive_source timing only)."""
    cpg_rate, device, gait_tables, model = run.cpg_rate, run.device, run.gait_tables, run.model
    period, spikes = run.period, run.spikes

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
    run.calib = calib



def make_objective(args, run):
    """The spike-statistics objective."""
    cpg_R, cpg_rate, gait_tables, period = run.cpg_R, run.cpg_rate, run.gait_tables, run.period

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
    run.spike_obj = spike_obj



def write_config(args, run):
    """Write the run config BEFORE training, so snapshots are loadable mid-run."""
    burst_thresholds, calib, cpg_R, cpg_params = run.burst_thresholds, run.calib, run.cpg_R, run.cpg_params
    cpg_rate, device, gait_files, gait_names = run.cpg_rate, run.device, run.gait_files, run.gait_names
    gait_tables, gait_tables_orig, gaits_dir, group_cols = run.gait_tables, run.gait_tables_orig, run.gaits_dir, run.group_cols
    layout_src, leg_cols, model, n_joints = run.layout_src, run.leg_cols, run.model, run.n_joints
    n_legs, n_par, neuron_offsets, onsets = run.n_legs, run.n_par, run.neuron_offsets, run.onsets
    out_dir, period, spike_obj, spikes = run.out_dir, run.period, run.spike_obj, run.spikes
    t_hi, t_lo, t_split, target_rows = run.t_hi, run.t_lo, run.t_split, run.target_rows
    tgt_range, timing_cols, timing_map, valid = run.tgt_range, run.timing_cols, run.timing_map, run.valid
    warm_steps = run.warm_steps

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
    run.best = best
    run.cfg = cfg
    run.final_lr = final_lr
    run.hist = hist
    run.timing_stats = timing_stats



def train(args, run):
    """Train (or skip under --dry_run), then reload the best checkpoint."""
    best, device, gait_names, gait_tables = run.best, run.device, run.gait_names, run.gait_tables
    gait_w, hist, model, out_dir = run.gait_w, run.hist, run.model, run.out_dir
    period, sat_rate, spike_obj, tgt_range = run.period, run.sat_rate, run.spike_obj, run.tgt_range
    timing_diag, tr_sampler, va_sampler = run.timing_diag, run.tr_sampler, run.va_sampler

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
            print("      Timing layer at initialization:")
            lines, timing_stats = timing_diag()
            for ln in lines:
                print(ln)
    run.best = best
    run.final_lr = final_lr
    run.hist = hist
    run.timing_stats = timing_stats



def evaluate(args, run):
    """[6/6] Reconstruction and transition figures; record the outcome."""
    best, cfg, device, final_lr = run.best, run.cfg, run.device, run.final_lr
    gait_names, hist, leg_cols, model = run.gait_names, run.hist, run.leg_cols, run.model
    n_joints, out_dir, period, spikes = run.n_joints, run.out_dir, run.period, run.spikes
    t_lo, t_split, targets, tgt_range = run.t_lo, run.t_split, run.targets, run.tgt_range
    timing_stats, valid = run.timing_stats, run.valid

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



def export(args, run):
    """ONNX export (--export_onnx 1)."""
    device, model, out_dir = run.device, run.model, run.out_dir
    # Gated, and the only route into ONNX code (export_onnx is the sole user of torch.onnx and
    # onnxruntime). Skipping it leaves the model as an export would (it restores the compiled
    # step in its finally); visualization reloads from disk anyway.
    if args.export_onnx:
        export_onnx(model, out_dir, device)
    else:
        print("    ONNX export skipped (--export_onnx 0)")



def visualize(args, run):
    """[7/7] End-of-run visualization, then the final message."""
    out_dir = run.out_dir

    # ── 7. Timing-layer visualization ───────────────────────────
    # Imported here so a run loads the visualization code only when --visualize asks for it.
    # out_dir is both the model dir and the output dir, so figures land in this run's
    # category folders, and re-reading best_model.pt and the config doubles as an end-of-run
    # check that they load.
    if args.visualize and not args.dry_run:
        print("\n[7/7] Timing-layer visualization ...")
        try:
            from .viz_pipeline import default_args, run_visualization
            # recon=0: phase 6 already wrote recon_*.png and transition.png with this run's
            # --recon_cycles. Use visualize.py --recon 1 to re-plot a checkpoint.
            run_visualization(out_dir, out_dir,
                              default_args(gaits_dir=args.gaits_dir, recon=0))
        except Exception as e:
            # Never let plotting lose a finished run: checkpoint, config and metrics are already on disk.
            print(f"  visualization FAILED ({type(e).__name__}: {e})")
            print(f"  training output is intact; rerun with "
                  f"'python visualize.py --model_dir {args.out_dir or ''}'")

    print(f"\nDone — {out_dir.resolve()}")
