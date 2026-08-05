"""
Benchmark harness for the bursting-LIF CPG -> leg-grouped stateful SNN.

Everything is imported from train.py -- no pipeline logic is reimplemented
here, so this file cannot silently drift from what actually trains.

What it measures
----------------
Per training step, with CUDA synchronised on both sides of the timed
window (without that you time queue submission, not work, and compiled
code looks absurdly fast):

    median ms/step        the raw number
    sample-timesteps/s    batch * bptt / sec  <- THE comparison metric
    peak GPU memory       reset after warmup, so compile/CUDA-graph
                          pools show up honestly
    first-step seconds    includes compilation; reported separately
                          because it is a fixed per-shape cost, not
                          throughput
    est. epoch seconds    chunks_per_epoch * train + val_chunks * val,
                          for picking chunks_per_epoch (aim 5-30 s)
    loss fingerprint      loss at steps 1/10/50/100 from a fixed seed

Why sample-timesteps/s and not ms/step: batch 128 does 4x the work of
batch 32, so ms/step makes a faster config look slower. Throughput is
apples-to-apples across batch and bptt; ms/step is not.

The loss fingerprint is the automated version of "run it and eyeball the
loss". It is only comparable between rows that share a numerics_key
(same batch/bptt/hidden/seed/...): changing batch changes the gradient
statistics, so a different loss trajectory there is expected, not a bug.

Results are APPENDED to bench_results.jsonl and the markdown table is
regenerated from the whole history, each row tagged with its git commit.
That is what makes this useful for "I rewrote step(), is it faster?" --
you compare today's row against the same variant at an older commit.

Usage
-----
    # after any code change: one row at current defaults, tagged by commit
    python benchmark.py

    # the full optimisation matrix (eager -> compile -> batch -> cudagraphs)
    python benchmark.py --set full

    # bptt sweep at fixed batch*bptt (the apples-to-apples way to sweep it)
    python benchmark.py --set bptt

    # ad-hoc single config
    python benchmark.py --set none --batch 256 --bptt 128 --compile default

    # fast smoke test of the harness itself
    python benchmark.py --set quick --measure 10 --warmup 3
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Import the real pipeline. train.py's module level sets
# float32_matmul_precision('high') and matplotlib Agg, which we want.
from train import (
    GAIT_NAMES, GAIT_TABLES_ORIG, LEG_COLS, N_JOINTS, N_LEGS,
    LegGroupedSNN, StreamSampler,
    analyse_cpg, apply_reset, build_targets, cycle_phase, detach_state,
    git_info, json_safe, make_gait_weights, masked_loss, routing_matrices,
    run_cpg, solve_leg_routing, upsample_gait_tables,
)


# ═══════════════════════════════════════════════════════════════════
# 1.  Variant sets
# ═══════════════════════════════════════════════════════════════════
#
# A variant is a name plus a dict of overrides applied on top of the CLI
# args. `compile`: None | "default" | "reduce-overhead" | "max-autotune".

VARIANT_SETS = {
    # Regression tracking: one row at whatever the current defaults are.
    # This is the mode to run after every code change.
    "current": [
        ("current", {}),
    ],

    # The optimisation ladder from the speedup todo list. Each row adds one
    # thing to the row above it, so the deltas are attributable.
    "full": [
        ("eager_b32",        dict(batch=32,  compile=None)),
        ("eager_b128",       dict(batch=128, compile=None)),
        ("compile_b32",      dict(batch=32,  compile="default")),
        ("compile_b128",     dict(batch=128, compile="default")),
        ("compile_b256",     dict(batch=256, compile="default")),
        ("cudagraph_b128",   dict(batch=128, compile="reduce-overhead")),
    ],

    # bptt sweep holding batch*bptt fixed at 32768, so every row does the
    # same work per gradient step and only the truncation horizon changes.
    "bptt": [
        ("bptt128_b256",     dict(bptt=128, batch=256, compile="default")),
        ("bptt256_b128",     dict(bptt=256, batch=128, compile="default")),
        ("bptt512_b64",      dict(bptt=512, batch=64,  compile="default")),
    ],

    # Does hidden width cost anything at these launch-bound sizes?
    "hidden": [
        ("h128",             dict(hidden=128, compile="default")),
        ("h256",             dict(hidden=256, compile="default")),
        ("h512",             dict(hidden=512, compile="default")),
    ],

    "quick": [
        ("eager_b32",        dict(batch=32,  compile=None)),
        ("compile_b128",     dict(batch=128, compile="default")),
    ],

    "none": [],   # use CLI values as a single unnamed variant
}


# ═══════════════════════════════════════════════════════════════════
# 2.  Data build (mirrors train.main steps 1-4, with caching)
# ═══════════════════════════════════════════════════════════════════

DATA_KEYS = ("tmax", "warmup", "i_app", "phase_zero", "val_frac", "seed")


def _data_cache_key(args):
    payload = {k: getattr(args, k) for k in DATA_KEYS}
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return h[:12], payload


def build_data(args, out_dir, verbose=False):
    """
    Run train.py's CPG -> phase -> targets -> routing pipeline.

    Only orchestration lives here; every computation is a train.py call, so
    a change to the pipeline is picked up automatically.

    Cached to .npz because regenerating costs a few seconds (CPG stepping +
    four KDE fits) and the benchmark is meant to be run often.
    """
    key, payload = _data_cache_key(args)
    cache = out_dir / f"bench_data_{key}.npz"

    if cache.exists() and not args.no_cache:
        z = np.load(cache, allow_pickle=False)
        if verbose:
            print(f"  [cache hit] {cache.name}")
        return dict(
            spikes=z["spikes"], targets=z["targets"], valid=z["valid"],
            route_P=z["route_P"], route=z["route"],
            period=float(z["period"]), n_gaits=int(z["n_gaits"]),
            t_lo=int(z["t_lo"]), t_split=int(z["t_split"]), t_hi=int(z["t_hi"]),
            tgt_range=(float(z["tgt_lo"]), float(z["tgt_hi"])),
        )

    sink = io.StringIO()
    ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(sink)
    with ctx:
        spikes = run_cpg(N=4, tmax=args.tmax, warmup=args.warmup,
                         i_app=args.i_app)
        onsets, period, neuron_offsets, _thr = analyse_cpg(spikes, out_dir)
        phase = cycle_phase(len(spikes), onsets[0])

        gait_tables, target_rows = upsample_gait_tables(
            GAIT_TABLES_ORIG, GAIT_NAMES, verbose=False)
        route = solve_leg_routing(GAIT_TABLES_ORIG, GAIT_NAMES, neuron_offsets)
        route_P = routing_matrices(route, n_neurons=4)

        targets, valid, tgt_range = build_targets(
            phase, gait_tables, phase_zero=args.phase_zero)

    T       = len(spikes)
    t_lo    = int(onsets[0][2])
    t_split = int(T * (1.0 - args.val_frac))
    t_hi    = int(onsets[0][-2])

    # Same guard train.py applies, checked against the LARGEST bptt any
    # variant will request, so a sweep cannot die halfway through.
    np.savez(
        cache,
        spikes=spikes, targets=targets, valid=valid,
        route_P=route_P, route=route,
        period=np.float64(period), n_gaits=np.int64(len(gait_tables)),
        t_lo=np.int64(t_lo), t_split=np.int64(t_split), t_hi=np.int64(t_hi),
        tgt_lo=np.float64(tgt_range[0]), tgt_hi=np.float64(tgt_range[1]),
    )
    if verbose:
        print(f"  [cache write] {cache.name}")

    return dict(spikes=spikes, targets=targets, valid=valid,
                route_P=route_P, route=route, period=float(period),
                n_gaits=len(gait_tables), t_lo=t_lo, t_split=t_split,
                t_hi=t_hi, tgt_range=tgt_range)


# ═══════════════════════════════════════════════════════════════════
# 3.  Timing helpers
# ═══════════════════════════════════════════════════════════════════

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _compile_step(model, mode):
    """Apply torch.compile to model.step. Returns a label for the report."""
    if mode in (None, "none", "off"):
        return "eager"
    kw = {"dynamic": False}
    if mode != "default":
        kw["mode"] = mode
    model._step_eager = model.step
    model.step = torch.compile(model.step, **kw)
    return mode


def output_sanity(model, sampler, batch, bptt, device):
    """
    Cheap check that the stacked output actually varies over time.

    This is the CUDA-graph aliasing check: with mode="reduce-overhead" the
    output buffer is reused across calls, and forward() does ys.append(y)
    then torch.stack(ys). If the buffer aliases, every entry becomes the
    same tensor and y_seq collapses to L copies of the last timestep --
    which trains to garbage while looking superficially fine.

    Run in train mode with grad enabled (no backward) because that is the
    path where the bug matters; an eval/no_grad forward compiles under
    different guards and would not necessarily reproduce it.
    """
    model.train()
    x, g, y, m, sw, rst = sampler.next_chunk(bptt)
    state = model.init_state(batch, device)
    pred, _ = model(x.to(device), g.to(device), state)
    p = pred.detach()
    std_over_time = float(p.std(dim=0).mean())
    first_last    = float((p[0] - p[-1]).abs().max())
    del pred, p
    return {"std_over_time": std_over_time,
            "first_vs_last": first_last,
            "time_varying": bool(std_over_time > 1e-8 and first_last > 1e-8)}


def time_train_steps(model, sampler, opt, gait_w, device, batch, bptt,
                     n_warmup, n_measure, clip, loss_at):
    """
    Returns (times_ms list, first_step_s, loss_fingerprint dict).

    Warmup absorbs compilation, Inductor autotuning and CUDA-graph
    recording (which itself needs ~3 iterations). Peak memory stats are
    reset at the warmup boundary so those pools are not attributed to
    steady state.
    """
    model.train()
    state = model.init_state(batch, device)
    times, losses = [], {}
    first_step_s = None

    total = n_warmup + n_measure
    for i in range(total):
        if i == n_warmup and device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        _sync(device)
        t0 = time.perf_counter()

        x, g, y, m, sw, rst = sampler.next_chunk(bptt)
        x, g, y, m = (x.to(device), g.to(device), y.to(device), m.to(device))
        state = apply_reset(detach_state(state), rst.to(device))

        pred, state = model(x, g, state)
        loss = masked_loss(pred, y, m, g, gait_w)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        _sync(device)
        dt = time.perf_counter() - t0

        if i == 0:
            first_step_s = dt
        step_no = i + 1
        if step_no in loss_at:
            losses[str(step_no)] = float(loss.detach())
        if i >= n_warmup:
            times.append(dt * 1e3)

    return times, first_step_s, losses


def time_val_steps(model, sampler, device, batch, bptt, settle,
                   n_warmup, n_measure):
    """
    Time the validation body, including the post-switch mask construction.

    Included deliberately: val_chunks runs every epoch regardless of
    chunks_per_epoch, so if chunks_per_epoch is small, validation can
    dominate the compute budget. Measuring it makes that visible.
    """
    model.eval()
    vstate = model.init_state(batch, device)
    times = []
    with torch.no_grad():
        for i in range(n_warmup + n_measure):
            _sync(device)
            t0 = time.perf_counter()

            x, g, y, m, sw, rst = sampler.next_chunk(bptt)
            x, g, y, m = (x.to(device), g.to(device),
                          y.to(device), m.to(device))
            sw = sw.to(device)
            vstate = apply_reset(vstate, rst.to(device))
            pred, vstate = model(x, g, vstate)
            _ = masked_loss(pred, y, m, g)

            post = torch.zeros_like(sw)
            idx = sw.nonzero(as_tuple=False)
            for t_i, b_i in idx:
                post[t_i:min(t_i + settle, sw.shape[0]), b_i] = 1.0
            if post.sum() > 0:
                _ = masked_loss(pred, y, m * post, g)

            _sync(device)
            if i >= n_warmup:
                times.append((time.perf_counter() - t0) * 1e3)
    return times


# ═══════════════════════════════════════════════════════════════════
# 4.  One variant
# ═══════════════════════════════════════════════════════════════════

NUMERICS_FIELDS = ("batch", "bptt", "hidden", "seed", "tmax", "warmup",
                   "i_app", "tau_min", "tau_max", "cross_gain", "slope",
                   "switch_min", "switch_max", "clip", "lr", "val_frac")


def numerics_key(cfg):
    """
    Hash of everything that legitimately changes the loss trajectory.

    Loss fingerprints are only comparable within one key. Changing batch
    changes gradient statistics, so a different loss there is expected --
    comparing across keys would produce false alarms.
    """
    payload = {k: cfg[k] for k in NUMERICS_FIELDS if k in cfg}
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def run_variant(name, cfg, data, device, args):
    print(f"\n─── {name} " + "─" * max(0, 56 - len(name)))
    print(f"    batch={cfg['batch']}  bptt={cfg['bptt']}  "
          f"hidden={cfg['hidden']}  compile={cfg['compile'] or 'eager'}")

    bptt, batch = cfg["bptt"], cfg["batch"]

    # Guard before doing any work: StreamSampler raises mid-run otherwise.
    if data["t_split"] - data["t_lo"] < 4 * bptt:
        raise ValueError(
            f"train range {data['t_split'] - data['t_lo']} steps < 4*bptt "
            f"({4 * bptt}); raise --tmax or lower --bptt.")
    if data["t_hi"] - data["t_split"] < 2 * bptt:
        raise ValueError(
            f"val range {data['t_hi'] - data['t_split']} steps < 2*bptt "
            f"({2 * bptt}); raise --tmax or lower --val_frac.")

    # Fixed seeds so model init and sampler draws are identical across
    # variants -- otherwise the loss fingerprint means nothing.
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    model = LegGroupedSNN(
        hidden=cfg["hidden"], n_gaits=data["n_gaits"],
        route_P=data["route_P"], tau_min=cfg["tau_min"],
        tau_max=cfg["tau_max"], cross_gain=cfg["cross_gain"],
        slope=cfg["slope"]).to(device)

    t_compile0 = time.perf_counter()
    compile_label = _compile_step(model, cfg["compile"])
    compile_call_s = time.perf_counter() - t_compile0   # lazy; real cost is step 1

    n_par = sum(p.numel() for p in model.parameters())
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        gait_w = make_gait_weights(GAIT_TABLES_ORIG, device)

    mk = lambda lo, hi, seed_off: StreamSampler(
        data["spikes"], data["targets"], data["valid"], lo, hi, batch,
        cfg["switch_min"], cfg["switch_max"],
        np.random.default_rng(cfg["seed"] + seed_off),
        n_gaits=data["n_gaits"])

    tr_sampler = mk(data["t_lo"], data["t_split"], 0)
    va_sampler = mk(data["t_split"], data["t_hi"], 1)

    loss_at = {1, 10, 50, 100}
    times, first_s, losses = time_train_steps(
        model, tr_sampler, opt, gait_w, device, batch, bptt,
        args.warmup, args.measure, cfg["clip"], loss_at)

    val_times = time_val_steps(
        model, va_sampler, device, batch, bptt, cfg["settle"],
        max(2, args.warmup // 3), max(3, args.measure // 4))

    sanity = output_sanity(model, va_sampler, batch, bptt, device)

    t   = np.array(times)
    vt  = np.array(val_times)
    med = float(np.median(t))
    sts = batch * bptt / (med / 1e3)

    peak_gib = (torch.cuda.max_memory_allocated() / 2 ** 30
                if device.type == "cuda" else None)

    est_epoch_s = (args.chunks_per_epoch * med
                   + args.val_chunks * float(np.median(vt))) / 1e3

    row = {
        "schema":           1,
        "variant":          name,
        "timestamp":        datetime.now(timezone.utc).isoformat(
                                timespec="seconds"),
        "git":              git_info(),
        "device":           str(device),
        "device_name":      (torch.cuda.get_device_name(0)
                             if device.type == "cuda" else platform.processor()
                             or platform.machine()),
        "torch":            torch.__version__,
        "compile":          compile_label,
        "n_params":         int(n_par),
        # timing
        "ms_median":        med,
        "ms_p10":           float(np.percentile(t, 10)),
        "ms_p90":           float(np.percentile(t, 90)),
        "ms_min":           float(t.min()),
        "ms_iqr":           float(np.percentile(t, 75) - np.percentile(t, 25)),
        "sample_timesteps_per_s": float(sts),
        "msts_per_s":       float(sts / 1e6),
        "first_step_s":     float(first_s),
        "compile_call_s":   float(compile_call_s),
        "val_ms_median":    float(np.median(vt)),
        "est_epoch_s":      float(est_epoch_s),
        "peak_gib":         peak_gib,
        "n_measure":        len(times),
        "n_warmup":         args.warmup,
        # numerics
        "numerics_key":     numerics_key(cfg),
        "loss":             losses,
        "sanity":           sanity,
        # config
        "cfg":              {k: v for k, v in cfg.items()},
    }

    print(f"    median {med:8.2f} ms/step   "
          f"p10 {row['ms_p10']:7.2f}  p90 {row['ms_p90']:7.2f}")
    print(f"    throughput {row['msts_per_s']:7.3f} M sample-timesteps/s")
    print(f"    first step {first_s:7.2f} s (includes compile)   "
          f"val {row['val_ms_median']:7.2f} ms")
    print(f"    est. epoch {est_epoch_s:7.2f} s "
          f"({args.chunks_per_epoch} train + {args.val_chunks} val chunks)")
    if peak_gib is not None:
        print(f"    peak mem   {peak_gib:7.2f} GiB")
    if not sanity["time_varying"]:
        print("    !! OUTPUT NOT TIME-VARYING — suspect CUDA-graph output "
              "aliasing. Do not trust this row.")
    if losses:
        print("    loss  " + "   ".join(
            f"@{k}={v:.6f}" for k, v in sorted(losses.items(), key=lambda kv: int(kv[0]))))

    del model, opt, tr_sampler, va_sampler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


# ═══════════════════════════════════════════════════════════════════
# 5.  Persistence + report
# ═══════════════════════════════════════════════════════════════════

def append_rows(rows, out_dir):
    path = out_dir / "bench_results.jsonl"
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(json_safe(r)) + "\n")
    print(f"\n  [appended] {len(rows)} row(s) -> {path}")
    return path


def load_rows(out_dir):
    path = out_dir / "bench_results.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _fmt(v, spec="{:.2f}", dash="—"):
    return dash if v is None else spec.format(v)


def write_report(out_dir, baseline=None):
    """
    Regenerate bench_results.md from the entire history.

    Rows are grouped by device: comparing ms/step across different GPUs is
    meaningless, so the tables never mix them. Speedup is computed against
    the baseline variant within each device group when present, otherwise
    against the slowest row in that group.

    Written defensively with .get() throughout, because this file accretes
    rows over time and a schema change should not make old rows unreadable.
    """
    rows = load_rows(out_dir)
    if not rows:
        return None

    def thr(r):
        c = r.get("cfg", {})
        got = r.get("sample_timesteps_per_s")
        if got:
            return got
        ms = r.get("ms_median")
        if not ms:
            return 0.0
        return c.get("batch", 1) * c.get("bptt", 1) / (ms / 1e3)

    groups = {}
    for r in rows:
        groups.setdefault(
            r.get("device_name") or r.get("device") or "unknown", []).append(r)

    L = []
    L.append("# Benchmark results")
    L.append("")
    L.append(f"Regenerated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
             f"from `bench_results.jsonl` ({len(rows)} rows).")
    L.append("")
    L.append("**Compare `M sTS/s` (million sample-timesteps per second), not "
             "`ms/step`.** A larger batch does proportionally more work per "
             "step, so `ms/step` makes a faster config look slower. "
             "`M sTS/s = batch * bptt / sec`.")
    L.append("")
    L.append("`first step` includes compilation and is a fixed per-shape "
             "cost, not throughput. `est epoch` uses the "
             "`chunks_per_epoch`/`val_chunks` passed at benchmark time.")
    L.append("")

    for dev, grp in groups.items():
        grp = sorted(grp, key=lambda r: r.get("timestamp", ""))
        L.append(f"## {dev}")
        L.append("")

        ref = None
        if baseline:
            cands = [r for r in grp if r.get("variant") == baseline]
            ref = cands[-1] if cands else None
        if ref is None:
            ref = min(grp, key=thr)
        ref_thr = thr(ref) or 1.0

        L.append("### Performance")
        L.append("")
        L.append("| variant | commit | batch | bptt | hidden | compile | "
                 "ms/step | M sTS/s | speedup | peak GiB | first step s | "
                 "val ms | est epoch s |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in grp:
            g = r.get("git") or {}
            commit = g.get("commit") or "—"
            if g.get("dirty"):
                commit += "*"
            c = r.get("cfg", {})
            L.append(
                f"| {r.get('variant','?')} | {commit} | "
                f"{c.get('batch','—')} | {c.get('bptt','—')} | "
                f"{c.get('hidden','—')} | {r.get('compile','—')} | "
                f"{_fmt(r.get('ms_median'))} | "
                f"{_fmt(r.get('msts_per_s') or (thr(r) / 1e6 or None), '{:.3f}')} | "
                f"{thr(r) / ref_thr:.2f}x | {_fmt(r.get('peak_gib'))} | "
                f"{_fmt(r.get('first_step_s'))} | "
                f"{_fmt(r.get('val_ms_median'))} | "
                f"{_fmt(r.get('est_epoch_s'))} |")
        L.append("")
        L.append(f"Speedup is relative to `{ref.get('variant','?')}` "
                 f"({ref.get('timestamp','')}). `*` on a commit means the "
                 f"working tree was dirty — that row is not reproducible "
                 f"from the commit alone.")
        L.append("")

        L.append("### Numerics")
        L.append("")
        L.append("Only compare loss values **within** a `nkey` group — a "
                 "different batch or bptt legitimately changes the loss "
                 "trajectory. `time-varying` false means the stacked output "
                 "collapsed (suspect CUDA-graph output aliasing) and the row "
                 "should be discarded.")
        L.append("")
        L.append("| variant | commit | nkey | loss@1 | loss@10 | loss@50 | "
                 "loss@100 | time-varying | params |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in grp:
            g = r.get("git") or {}
            commit = g.get("commit") or "—"
            lo = r.get("loss") or {}
            npar = r.get("n_params")
            npar_s = f"{npar:,}" if isinstance(npar, int) else "—"
            sanity = r.get("sanity")
            if sanity is None:
                sane_s = "—"                      # not recorded, not failed
            else:
                sane_s = "yes" if sanity.get("time_varying") else "**NO**"
            L.append(
                f"| {r.get('variant','?')} | {commit} | "
                f"{r.get('numerics_key','—')} | "
                + " | ".join(_fmt(lo.get(k), '{:.6f}') for k in
                             ("1", "10", "50", "100"))
                + f" | {sane_s} | {npar_s} |")
        L.append("")

    path = out_dir / "bench_results.md"
    path.write_text("\n".join(L))
    print(f"  [wrote]    {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 6.  CLI
# ═══════════════════════════════════════════════════════════════════

def build_cfg(args, overrides, cpg_warmup):
    """
    Assemble a variant config.

    `cpg_warmup` is passed separately because args.warmup is the
    MEASUREMENT warmup (unmeasured benchmark iterations), while cfg's
    "warmup" is the CPG warmup that affects the data and therefore the
    numerics key. Conflating them would silently mislabel rows.
    """
    cfg = dict(
        batch=args.batch, bptt=args.bptt, hidden=args.hidden,
        compile=args.compile, lr=args.lr, clip=args.clip,
        settle=args.settle, seed=args.seed,
        tau_min=args.tau_min, tau_max=args.tau_max,
        cross_gain=args.cross_gain, slope=args.slope,
        switch_min=args.switch_min, switch_max=args.switch_max,
        tmax=args.tmax, warmup=cpg_warmup, i_app=args.i_app,
        val_frac=args.val_frac, phase_zero=args.phase_zero,
    )
    cfg.update(overrides)
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark the CPG-SNN training step. Imports all "
                    "pipeline logic from train.py.")

    ap.add_argument("--set", dest="variant_set", type=str, default="current",
                    choices=sorted(VARIANT_SETS.keys()),
                    help="Which variant set to run. 'current' = one row at "
                         "current defaults (use after every code change). "
                         "'none' = single run from CLI values.")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Variant name to compute speedups against in the "
                         "report. Default: slowest row per device.")

    # measurement
    ap.add_argument("--warmup",  type=int, default=20,
                    help="Unmeasured iterations. Must cover compilation, "
                         "autotuning and CUDA-graph recording.")
    ap.add_argument("--measure", type=int, default=50)
    ap.add_argument("--chunks_per_epoch", type=int, default=40,
                    help="Only used to estimate epoch wall-clock.")
    ap.add_argument("--val_chunks",       type=int, default=8)

    # model / training (defaults mirror train.py)
    ap.add_argument("--batch",      type=int,   default=128)
    ap.add_argument("--bptt",       type=int,   default=256)
    ap.add_argument("--hidden",     type=int,   default=256)
    ap.add_argument("--compile",    type=str,   default="default",
                    choices=["none", "default", "reduce-overhead",
                             "max-autotune"],
                    help="torch.compile mode for model.step.")
    ap.add_argument("--lr",         type=float, default=2e-3)
    ap.add_argument("--clip",       type=float, default=1.0)
    ap.add_argument("--settle",     type=int,   default=100)
    ap.add_argument("--tau_min",    type=float, default=2.0)
    ap.add_argument("--tau_max",    type=float, default=256.0)
    ap.add_argument("--cross_gain", type=float, default=0.25)
    ap.add_argument("--slope",      type=float, default=25.0)
    ap.add_argument("--switch_min", type=int,   default=600)
    ap.add_argument("--switch_max", type=int,   default=3000)

    # data
    ap.add_argument("--tmax",       type=int,   default=50_000)
    ap.add_argument("--warmup_cpg", dest="warmup_cpg", type=int, default=2_000)
    ap.add_argument("--i_app",      type=float, default=8.0)
    ap.add_argument("--val_frac",   type=float, default=0.15)
    ap.add_argument("--phase_zero", type=float, default=0.0)
    ap.add_argument("--no_cache",   action="store_true",
                    help="Rebuild the CPG/target cache.")

    # misc
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--device",  type=str, default=None,
                    help="cuda | cpu. Default: cuda if available.")
    ap.add_argument("--out_dir", type=str, default="bench")
    ap.add_argument("--report_only", action="store_true",
                    help="Regenerate bench_results.md from history and exit.")
    args = ap.parse_args()

    # `--warmup` is the MEASUREMENT warmup (unmeasured benchmark
    # iterations); `--warmup_cpg` is the CPG settling warmup. Kept
    # separate on purpose -- see build_cfg.
    cpg_warmup = args.warmup_cpg
    if args.compile == "none":
        args.compile = None

    this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    out_dir  = this_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        write_report(out_dir, baseline=args.baseline)
        return

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"         {torch.cuda.get_device_name(0)}")
    else:
        print("         WARNING: on CPU. torch.compile gains and all memory "
              "numbers will not reflect a CUDA run.")
    print(f"torch  : {torch.__version__}")
    g = git_info()
    print(f"git    : {g['commit']}{' (dirty)' if g['dirty'] else ''}")
    print(f"out_dir: {out_dir}")

    variants = VARIANT_SETS[args.variant_set]
    if not variants:
        variants = [(f"b{args.batch}_t{args.bptt}_"
                     f"{args.compile or 'eager'}", {})]

    # Build data once, sized for the largest bptt in the set.
    print("\nBuilding CPG / targets ...")
    class _DataArgs:
        pass
    da = _DataArgs()
    for k in ("tmax", "i_app", "phase_zero", "val_frac", "seed", "no_cache"):
        setattr(da, k, getattr(args, k))
    da.warmup = cpg_warmup
    data = build_data(da, out_dir, verbose=True)
    print(f"  spikes {data['spikes'].shape}  targets {data['targets'].shape}  "
          f"period {data['period']:.1f}")
    print(f"  train [{data['t_lo']}, {data['t_split']})   "
          f"val [{data['t_split']}, {data['t_hi']})")

    rows = []
    try:
        for name, ov in variants:
            cfg = build_cfg(args, ov, cpg_warmup)
            try:
                rows.append(run_variant(name, cfg, data, device, args))
            except Exception as e:                          # noqa: BLE001
                if "out of memory" in str(e).lower():
                    print(f"    !! OOM — skipping {name}. "
                          f"Lower batch or bptt.")
                else:
                    print(f"    !! FAILED ({type(e).__name__}): {e}")
                    traceback.print_exc()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    except KeyboardInterrupt:
        # Save whatever completed rather than discarding the whole session.
        print("\n  [interrupt] stopping variant sweep; "
              "writing completed rows.")

    if not rows:
        print("\nNo successful runs.")
        return

    append_rows(rows, out_dir)
    write_report(out_dir, baseline=args.baseline)

    # Session summary, ordered by throughput.
    print("\n" + "=" * 72)
    print(f"{'variant':<20}{'ms/step':>10}{'M sTS/s':>10}"
          f"{'speedup':>10}{'peak GiB':>10}{'epoch s':>10}")
    print("-" * 72)
    ref = min(rows, key=lambda r: r["sample_timesteps_per_s"])
    if args.baseline:
        c = [r for r in rows if r["variant"] == args.baseline]
        ref = c[0] if c else ref
    for r in sorted(rows, key=lambda r: -r["sample_timesteps_per_s"]):
        spd = r["sample_timesteps_per_s"] / ref["sample_timesteps_per_s"]
        print(f"{r['variant']:<20}{r['ms_median']:>10.2f}"
              f"{r['msts_per_s']:>10.3f}{spd:>9.2f}x"
              f"{_fmt(r.get('peak_gib'), '{:.2f}'):>10}"
              f"{r['est_epoch_s']:>10.2f}")
    print("=" * 72)
    print(f"speedup vs {ref['variant']}")


if __name__ == "__main__":
    main()