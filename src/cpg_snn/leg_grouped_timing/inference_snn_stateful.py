"""
inference_snn_stateful.py
=========================
Stateful inference for the CPG -> SNN gait decoder, loading `best_model.pt`
directly with PyTorch (no ONNX, no robot).

Per spike event:
  * stateful  -> CPG_SNN_Stateful.step(), membranes carried between events
  * stateless -> the original cpg_utils.CPG_SNN on a deque of the last
                 seq_len events, exactly as inference_snn.py does today

Both run off the same CPG stream with the same weights, so the printed
`RMSE sl-GT` column should reproduce what inference_snn.py already gets.
If it does not, the harness is wrong and the stateful column means nothing.

Usage:
    python inference_snn_stateful.py --out_dir outputs --t_max 50000
    python inference_snn_stateful.py --no_parity        # stateful only
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cpg_utils import BLIF_CPG, CPG_SNN
from stateful_snn import CPG_SNN_Stateful, load_checkpoint
from plotting_utils import savefig


# ═══════════════════════════════════════════════════════════════════
# 1.  Config + gait tables (must match training)
# ═══════════════════════════════════════════════════════════════════

def load_config(out_dir):
    cfg_path = Path(out_dir) / "cpg_snn_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}\n"
                                "Run train_snn.py first.")
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"  config: {cfg_path}")
    print(f"    gait_period    = {cfg['gait_period']:.1f}")
    print(f"    global min/max = {cfg['global_min']:.1f} / {cfg['global_max']:.1f}")
    print(f"    seq_len        = {int(cfg['seq_len'])}")
    print(f"    cpg_start_time = {int(cfg.get('cpg_start_time', 90))}")
    return cfg


def load_gait_tables(this_file_dir, target_rows):
    """Load and cubic-upsample the gait tables, matching training."""
    from scipy.interpolate import interp1d
    gait_names = ["bittle_wkF", "bittle_bk", "bittle_wkL", "bittle_wkR"]
    tables = []
    for name in gait_names:
        gt = np.loadtxt(f"{this_file_dir}/gaits/{name}.csv",
                        delimiter=",", dtype=np.float32)
        if gt.shape[0] != target_rows:
            f = interp1d(np.linspace(0, 1, gt.shape[0]), gt, axis=0,
                         kind="cubic", fill_value="extrapolate")
            gt = f(np.linspace(0, 1, target_rows)).astype(np.float32)
        tables.append(gt)
    return gait_names, tables


# ═══════════════════════════════════════════════════════════════════
# 2.  Inference loop
# ═══════════════════════════════════════════════════════════════════

def run_inference(cfg, ckpt, out_dir, args, schedule):
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    gait_period = float(cfg["gait_period"])
    global_min  = float(cfg["global_min"])
    global_max  = float(cfg["global_max"])
    seq_len     = int(cfg["seq_len"])
    n_neurons   = int(cfg["n_neurons"])
    n_gaits     = int(cfg["n_gaits"])
    n_joints    = int(cfg["n_joints"])
    cpg_start   = int(cfg.get("cpg_start_time", 90))
    target_rows = int(cfg.get("target_rows", 54))

    scale = (global_max - global_min) / 2.0
    shift = (global_max + global_min) / 2.0

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    gait_names, gait_tables = load_gait_tables(this_file_dir, target_rows)

    # ── Models: same weights, two calling conventions ─────────────
    sd, arch = load_checkpoint(ckpt)
    beta = args.beta if args.beta is not None else (arch["beta"] or 0.9)
    n_in = arch["n_in"]
    use_phase = (n_in - n_neurons - n_gaits) == 2
    print(f"  checkpoint: n_in={n_in} hidden={arch['hidden']} "
          f"n_out={arch['n_out']} beta={beta:.3f}  use_phase={use_phase}")

    model_sf = CPG_SNN_Stateful(n_in=n_in, hidden=arch["hidden"],
                               n_out=arch["n_out"], n_gaits=n_gaits,
                               beta=beta).to(device)
    model_sf.load_state_dict(sd, strict=True)   # loud on any key mismatch
    model_sf.eval()
    state = model_sf.init_state(batch=1, device=device)

    model_sl = None
    if args.parity:
        model_sl = CPG_SNN(n_in=n_in, hidden=arch["hidden"],
                           n_out=arch["n_out"], n_gaits=n_gaits,
                           beta=beta).to(device)
        model_sl.load_state_dict(sd, strict=True)
        model_sl.eval()

    # ── CPG warm-up: identical to training ────────────────────────
    cpg = BLIF_CPG(N=n_neurons, t_max=args.t_max)
    for _ in range(cpg_start):
        cpg.step()
    print(f"  CPG warmed up ({cpg_start} steps)\n")

    hist    = deque(maxlen=seq_len)      # stateless sliding window
    x_buf   = torch.zeros(1, n_in, device=device)
    win_buf = torch.zeros(seq_len, 1, n_in, device=device)

    rec = {k: [] for k in ("t", "neuron", "phase_deg", "gait",
                           "pred_sf", "pred_sl", "true", "chunk_end")}
    lat_sf, lat_sl = [], []
    active_gait, sched_ptr = schedule[0][1], 0
    n_events, n_multi, steps_done = 0, 0, 0

    print(f"  Running {args.t_max - cpg_start} CPG steps "
          f"(burn_in={args.burn_in} events) ...")

    for _ in range(cpg_start, args.t_max):
        while sched_ptr < len(schedule) and steps_done >= schedule[sched_ptr][0]:
            active_gait = schedule[sched_ptr][1]
            print(f"    step {steps_done:>6d}: gait -> {gait_names[active_gait]}")
            sched_ptr += 1

        spikes, _, t_now = cpg.step()
        steps_done += 1
        spikes = np.asarray(spikes) > 0

        if not spikes.any():
            continue                     # model timestep == spike event
        if spikes.sum() > 1:
            n_multi += 1

        phase = float(2.0 * np.pi * (t_now % gait_period) / gait_period)

        # ── feature vector: same layout as encode_spike_events +
        #    the per-event gait flag from build_dataset ────────────
        feat = np.zeros(n_in, dtype=np.float32)
        feat[:n_neurons] = spikes.astype(np.float32)
        if use_phase:
            feat[n_neurons]     = np.sin(phase)
            feat[n_neurons + 1] = np.cos(phase)
            feat[n_neurons + 2 + active_gait] = 1.0
        else:
            feat[n_neurons + active_gait] = 1.0
        n_events += 1

        # ── periodic reset: n_events % reset_every == 1 means this event
        #    starts a fresh chunk, so the event at ...% == 0 is the last of
        #    a chunk and matches the stateless window exactly ───────
        chunk_end = False
        if args.reset_every > 0:
            if (n_events - 1) % args.reset_every == 0:
                state = model_sf.init_state(batch=1, device=device)
            chunk_end = (n_events % args.reset_every == 0)

        # ── stateful: one step, memory carried ───────────────────
        x_buf[0] = torch.from_numpy(feat).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_sf, state = model_sf.step(x_buf, state)
        lat_sf.append((time.perf_counter() - t0) * 1000.0)
        pred_sf = out_sf[0].cpu().numpy() * scale + shift

        # ── stateless: seq_len window, memory zeroed each call ───
        hist.append(feat)
        pred_sl = None
        if model_sl is not None and len(hist) == seq_len:
            win_buf[:, 0, :] = torch.from_numpy(np.stack(hist)).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out_sl = model_sl(win_buf)
            lat_sl.append((time.perf_counter() - t0) * 1000.0)
            pred_sl = out_sl[0].cpu().numpy() * scale + shift

        if n_events <= args.burn_in:
            continue
        if model_sl is not None and pred_sl is None:
            continue

        gt = gait_tables[active_gait]
        row = int(phase / (2.0 * np.pi) * gt.shape[0]) % gt.shape[0]

        rec["t"].append(t_now)
        rec["neuron"].append(int(np.argmax(spikes)))
        rec["phase_deg"].append(np.degrees(phase))
        rec["gait"].append(active_gait)
        rec["pred_sf"].append(pred_sf)
        rec["pred_sl"].append(pred_sl if pred_sl is not None
                              else np.full(n_joints, np.nan, np.float32))
        rec["true"].append(gt[row].astype(np.float32))
        rec["chunk_end"].append(1.0 if chunk_end else 0.0)

    print(f"\n  events={n_events}  recorded={len(rec['t'])}")
    if n_multi:
        print(f"  [warn] {n_multi} events had >1 neuron spiking at once "
              f"(multi-hot input never seen in training)")
    if lat_sf:
        print(f"  stateful  latency: mean {np.mean(lat_sf):.3f} ms")
    if lat_sl:
        print(f"  stateless latency: mean {np.mean(lat_sl):.3f} ms "
              f"({seq_len} model steps per prediction)")

    if not rec["t"]:
        return None
    res = {k: np.asarray(v, dtype=np.float32) for k, v in rec.items()}
    res["neuron"]     = np.asarray(rec["neuron"], dtype=np.int32)
    res["gait"]       = np.asarray(rec["gait"],   dtype=np.int32)
    res["gait_names"] = gait_names
    return res


# ═══════════════════════════════════════════════════════════════════
# 3.  Parity report + plot
# ═══════════════════════════════════════════════════════════════════

def report(res, parity):
    sf, gt = res["pred_sf"], res["true"]
    J = sf.shape[1]
    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2, axis=0))

    if not parity:
        r = rmse(sf, gt)
        print("\n  RMSE stateful vs GT (deg):  "
              + "  ".join(f"J{j+1}={r[j]:.2f}" for j in range(J)))
        print(f"  mean = {r.mean():.3f}")
        return

    ok = ~np.isnan(res["pred_sl"]).any(axis=1)
    sf, sl, gt = sf[ok], res["pred_sl"][ok], gt[ok]
    r_sf, r_sl, r_d = rmse(sf, gt), rmse(sl, gt), rmse(sf, sl)

    print(f"\n  ── {ok.sum()} events ──")
    print(f"  {'joint':>6}  {'sf-GT':>9}  {'sl-GT':>9}  {'sf-sl':>9}  {'corr':>7}")
    for j in range(J):
        c = (np.corrcoef(sf[:, j], sl[:, j])[0, 1]
             if sl[:, j].std() > 1e-9 else np.nan)
        print(f"  {j+1:>6}  {r_sf[j]:>9.3f}  {r_sl[j]:>9.3f}"
              f"  {r_d[j]:>9.3f}  {c:>7.3f}")
    print(f"  {'mean':>6}  {r_sf.mean():>9.3f}  {r_sl.mean():>9.3f}"
          f"  {r_d.mean():>9.3f}")
    print("\n  sl-GT is the baseline: it should match inference_snn.py.")

    ce = res["chunk_end"][ok] > 0
    if ce.any():
        print(f"\n  chunk-end events only ({ce.sum()}): "
              f"sf-GT = {np.sqrt(np.mean((sf[ce]-gt[ce])**2)):.3f}   "
              f"sl-GT = {np.sqrt(np.mean((sl[ce]-gt[ce])**2)):.3f}   "
              f"sf-sl = {np.sqrt(np.mean((sf[ce]-sl[ce])**2)):.3e}")
        print("  With --reset_every=seq_len, sf-sl here should be ~0 "
              "(float noise). If it is not, step() is wrong.")

    print(f"\n  {'gait':>18}  {'n':>6}  {'sf-GT':>9}  {'sl-GT':>9}")
    for g, name in enumerate(res["gait_names"]):
        m = res["gait"][ok] == g
        if m.any():
            print(f"  {name:>18}  {m.sum():>6}"
                  f"  {np.sqrt(np.mean((sf[m]-gt[m])**2)):>9.3f}"
                  f"  {np.sqrt(np.mean((sl[m]-gt[m])**2)):>9.3f}")


def dump_chunk_ends(res, reset_every, n_dump=20, n_full=5):
    """
    Print the stateful vs stateless outputs at every chunk-end event — the
    events where the stateful memory has seen exactly `reset_every` events
    since a reset, so with reset_every == seq_len the two computations are
    identical and the difference should be float noise.
    """
    ok = ~np.isnan(res["pred_sl"]).any(axis=1)
    ce = np.where((res["chunk_end"] > 0) & ok)[0]
    if len(ce) == 0:
        print("\n  No chunk-end events recorded "
              "(need --reset_every > 0).")
        return

    sf, sl, gt = res["pred_sf"], res["pred_sl"], res["true"]
    J = sf.shape[1]

    print(f"\n  ── chunk-end events (memory age = {reset_every} events, "
          f"{len(ce)} total, showing first {min(n_dump, len(ce))}) ──")
    print(f"  {'#':>4}  {'cpg_t':>8}  {'gait':>4}  "
          f"{'max|sf-sl|':>11}  {'mean|sf-sl|':>12}  {'max|sf-GT|':>11}")
    for k, i in enumerate(ce[:n_dump]):
        d = np.abs(sf[i] - sl[i])
        print(f"  {k:>4}  {int(res['t'][i]):>8}  {int(res['gait'][i]):>4}  "
              f"{d.max():>11.3e}  {d.mean():>12.3e}  "
              f"{np.abs(sf[i] - gt[i]).max():>11.3f}")

    print(f"\n  ── per-joint values at the first {min(n_full, len(ce))} "
          f"chunk-end events (degrees) ──")
    for k, i in enumerate(ce[:n_full]):
        print(f"\n  event #{k}  cpg_t={int(res['t'][i])}  "
              f"gait={res['gait_names'][int(res['gait'][i])]}")
        print("    " + "".join(f"{'J'+str(j+1):>10}" for j in range(J)))
        for name, arr in (("stateful ", sf[i]), ("stateless", sl[i]),
                          ("GT       ", gt[i])):
            print(f"    {name}" + "".join(f"{v:>10.3f}" for v in arr))
        print("    diff     " + "".join(f"{v:>10.2e}"
                                        for v in (sf[i] - sl[i])))

    d_all = np.abs(sf[ce] - sl[ce])
    print(f"\n  over all {len(ce)} chunk-end events:  "
          f"max|sf-sl| = {d_all.max():.3e}   mean|sf-sl| = {d_all.mean():.3e}")
    print(f"  RMSE sf-GT = {np.sqrt(np.mean((sf[ce]-gt[ce])**2)):.3f}   "
          f"RMSE sl-GT = {np.sqrt(np.mean((sl[ce]-gt[ce])**2)):.3f}")
    print("  If reset_every == seq_len, max|sf-sl| should be ~1e-5 or less.")


def plot_parity(res, out_dir, n_show=600, n_joints_show=4):
    sf, sl, gt = res["pred_sf"], res["pred_sl"], res["true"]
    ok = ~np.isnan(sl).any(axis=1)
    sf, sl, gt = sf[ok], sl[ok], gt[ok]
    ce = res["chunk_end"][ok] > 0
    n = min(n_show, len(sf))
    J = min(n_joints_show, sf.shape[1])
    ev = np.arange(n)
    ce_idx = np.where(ce[:n])[0]

    fig, axes = plt.subplots(J, 1, figsize=(15, 2.6 * J), sharex=True,
                             squeeze=False)
    for j in range(J):
        ax = axes[j][0]
        for e in ce_idx:
            ax.axvline(e, color="k", lw=0.7, alpha=0.25)
        ax.plot(ev, gt[:n, j], color="#457b9d", lw=1.8, label="GT")
        ax.plot(ev, sl[:n, j], color="#2a9d8f", lw=1.4, ls="--",
                label="stateless")
        ax.plot(ev, sf[:n, j], color="#e63946", lw=1.4, ls=":",
                label="stateful")
        if len(ce_idx):
            ax.scatter(ce_idx, sf[ce_idx, j], s=22, color="#e63946",
                       zorder=5, label="chunk end" if j == 0 else None)
        ax.set_ylabel(f"J{j+1} (deg)", fontsize=9)
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=8, ncol=4, loc="upper right")
    axes[-1][0].set_xlabel("spike-event index")
    plt.suptitle("Stateful vs stateless inference, identical weights",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir, "parity.png")


# ═══════════════════════════════════════════════════════════════════
# 4.  Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--ckpt",    type=str, default=None)
    p.add_argument("--t_max",   type=int, default=50_000)
    p.add_argument("--burn_in", type=int, default=200,
                   help="Events discarded while the stateful memory settles")
    p.add_argument("--no_parity", dest="parity", action="store_false")
    p.add_argument("--reset_every", type=int, default=0,
                   help="Zero the stateful memory every N events (0 = never). "
                        "N=seq_len makes every Nth event exactly reproduce "
                        "the stateless computation.")
    p.add_argument("--dump", type=int, default=20,
                   help="Print stateful vs stateless at the first N "
                        "chunk-end events (needs --reset_every)")
    p.add_argument("--beta",   type=float, default=None)
    p.add_argument("--device", type=str,   default=None)
    args = p.parse_args()

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = Path(this_file_dir) / args.out_dir
    ckpt = Path(args.ckpt) if args.ckpt else out_dir / "best_model.pt"

    cfg = load_config(out_dir)

    n_gaits = int(cfg["n_gaits"])
    times = [i * args.t_max // (n_gaits + 1) for i in range(n_gaits + 1)]
    schedule = [(times[i], i % n_gaits) for i in range(n_gaits + 1)]

    res = run_inference(cfg, ckpt, out_dir, args, schedule)
    if res is None:
        print("No events recorded.")
        return

    report(res, args.parity)
    if args.parity and args.reset_every > 0 and args.dump > 0:
        dump_chunk_ends(res, args.reset_every, n_dump=args.dump)
    if args.parity:
        plot_parity(res, out_dir / "stateful")
        print(f"\nDone — {(out_dir / 'stateful').resolve()}")


if __name__ == "__main__":
    main()