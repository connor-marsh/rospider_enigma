"""
CPG Spike Train → SNN → Robust Multi-Gait Joint Angle Prediction
=================================================================

Key design decisions
--------------------
1.  Bursting CPG period estimation.
    The CPG neurons fire in bursts (not single spikes per cycle).
    One gait cycle = one complete rotation through all four neuron
    bursts (neuron 0 burst → 1 → 2 → 3 → neuron 0 again).
    The gait period is therefore the inter-BURST interval of neuron 0:
    time from the first spike of one neuron-0 burst to the first spike
    of the next neuron-0 burst.

    The within-burst vs between-burst ISI threshold is found
    automatically from the antimode of the log-ISI kernel density
    estimate of neuron 0 — no manual tuning required.  A diagnostic
    plot of the ISI distribution and detected threshold is saved so
    the split can be visually verified.

    A second diagnostic plot overlays gait-table joint angles against
    detected burst boundaries so the phase→row correspondence can be
    verified before committing to training.

2.  Per-event gait flag.
    Every spike event carries its own 4-dim one-hot gait flag.
    During a gait transition the sliding window naturally contains a
    mix of old- and new-flag events as the buffer fills — matching
    inference exactly.

3.  Two window types (mixed at build time):
    a) Pure-gait windows   — all seq_len events share the same flag.
       Stride-1, then randomly subsampled to hit transition_frac.
    b) Transition windows  — single A→B switch at a point sampled
       from the last quarter of the window.  All 12 ordered pairs.
    Target = new-gait row at phase of last spike in both cases.

4.  No temporal-position cue.
    Pure windows subsampled in random order.  Transition switch points
    randomised per window.  Global shuffle of final dataset.

5.  Separate val tracking: val_pure MSE and val_trans MSE logged
    independently every epoch.

6.  Optuna sweep over seq_len, hidden, beta, transition_frac, lr.

Input feature vector per spike event  (length = N + 2 + n_gaits = 10):
    [one_hot_neuron(4),  sin(phase)(1),  cos(phase)(1),  gait_flag(4)]

Phase
-----
φ = 2π · (t mod gait_period) / gait_period
where gait_period = median inter-burst interval of neuron 0 (post burn-in).

CPG Integration
---------------
Both training data generation and deployment use the same chunk-based
BDF integrator (CPGChunkStepper).  Spike events are detected inline
during integration — identical to what the Raspberry Pi will run —
eliminating any train/deploy mismatch from batch vs streaming integration.

The warm-up phase (cpg_start_time steps) is run in chunks before
data collection begins, matching the deployment boot sequence exactly.
"""

import argparse
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate
from pathlib import Path

from training_utils import GaitDataset, build_dataset, eval_epoch, export_to_onnx, make_gait_weighted_criterion, make_loader, train_epoch, train_val_test_split, upsample_gait_tables
from cpg_utils import CPG_SNN, encode_spike_events, estimate_gait_period, run_blif_cpg, sigmoid, neuron_eqs, make_network
from plotting_utils import plot_blif_cpg, plot_burst_gait_overlay, plot_cpg_vm, plot_gait_reconstruction, plot_inference, plot_spike_events, plot_training_curves


def run_training(model, train_loader, val_pure_loader, val_trans_loader,
                 optimizer, scheduler, criterion, device,
                 epochs, out_dir, weighted=False):
    best_val  = float("inf")
    best_path = out_dir / "best_model.pt"
    history   = {"train": [], "val_pure": [], "val_trans": []}

    def run_epoch(epoch, best_val):
        tl = train_epoch(model, train_loader, optimizer, criterion,
                         device, weighted=weighted)
        vp = eval_epoch(model, val_pure_loader,  criterion, device,
                        weighted=weighted)
        vt = eval_epoch(model, val_trans_loader, criterion, device,
                        weighted=weighted)
        scheduler.step()

        history["train"].append(tl)
        history["val_pure"].append(vp)
        history["val_trans"].append(vt)

        valid_vals   = [v for v in (vp, vt) if not np.isnan(v)]
        val_combined = float(np.mean(valid_vals)) if valid_vals else float("inf")

        flag = ""
        if val_combined < best_val:
            best_val = val_combined
            torch.save(model.state_dict(), best_path)
            flag = " ✓"

        if epoch % 10 == 0 or epoch == 1:
            vp_s = f"{vp:.6f}" if not np.isnan(vp) else "       nan"
            vt_s = f"{vt:.6f}" if not np.isnan(vt) else "       nan"
            print(f"  {epoch:>6}  {tl:>10.6f}  {vp_s:>10}"
                  f"  {vt_s:>10}"
                  f"  {optimizer.param_groups[0]['lr']:>8.2e}{flag}")
            
        return best_val

    print(f"\n  {'Epoch':>6}  {'Train':>10}  {'Val-Pure':>10}"
          f"  {'Val-Trans':>10}  {'LR':>8}")
    print("  " + "-" * 58)

    try:
        for epoch in range(1, epochs + 1):
            best_val = run_epoch(epoch, best_val)

        #### Wrap above for loop in below profiler code if you want
        # import torch.profiler

        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     record_shapes=True
        # ) as prof:
            
        #     # RUN JUST 2 BATCHES HERE
        #     # for i, batch in enumerate(train_loader):
        #     #     if i >= 2: break
        #     #     # ... your train loop body ...

        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    except KeyboardInterrupt:
        print("\n  [interrupt] Ctrl+C received; stopping.")

    print("  " + "-" * 58)
    return best_val, history


def main():
    parser = argparse.ArgumentParser(
        description="CPG-SNN robust multi-gait controller — chunk-based CPG")

    # ── CPG ─────────────────────────────────────────────────────
    parser.add_argument("--tmax",            type=int,   default=10000)
    parser.add_argument("--cpg_start_time",  type=int,   default=90)
    parser.add_argument("--chunk_size",      type=int,   default=1,
                        help="Steps per solve_ivp call in CPGChunkStepper. "
                             "Must match the value used at deployment on RPi.")
    parser.add_argument("--spike_thresh",    type=float, default=-2.0,
                        help="Upward vm crossing threshold for spike detection")

    # ── Network ──────────────────────────────────────────────────
    parser.add_argument("--seq_len",         type=int,   default=20,
                        help="Spike events per input window. "
                             "Needs to span ~1 full gait cycle (~32 spikes) "
                             "for reliable phase tracking.")
    parser.add_argument("--hidden",          type=int,   default=128)
    parser.add_argument("--beta",            type=float, default=0.9)
    parser.add_argument("--use_phase",       type=bool, default=False)

    # ── Training ─────────────────────────────────────────────────
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch",           type=int,   default=256)
    parser.add_argument("--val",             type=float, default=0.15)
    parser.add_argument("--test",            type=float, default=0.10)
    parser.add_argument("--transition_frac", type=float, default=0.30)

    # ── Period estimation ────────────────────────────────────────
    parser.add_argument("--burnin_bursts",   type=int,   default=5)
    parser.add_argument("--kde_bw",          type=float, default=0.3)

    # ── Misc ─────────────────────────────────────────────────────
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--out_dir",         type=str,   default="outputs")

    N = 4

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = Path(this_file_dir + "/" + args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device : {device}\n")

    # ── 1. CPG — chunk-based (same integrator as deployment) ────
    print("[1/6] Running CPG via CPGChunkStepper ...")
    print(f"      chunk_size={args.chunk_size}  spike_thresh={args.spike_thresh}")


    ### Code for running MTF network
    # spike_times, spike_neurons, vm_record = run_cpg_chunked(
    #     N=N,
    #     tmax=args.tmax,
    #     cpg_start_time=args.cpg_start_time,
    #     chunk_size=args.chunk_size,
    #     spike_thresh=args.spike_thresh,
    # )

    spike_times, spike_neurons, spike_array, vms = run_blif_cpg(N=N, t_max = args.tmax, cpg_start_time=args.cpg_start_time)
    plot_blif_cpg(spike_array, vms, out_dir, n_show=400)
    
    print(f"      Collected {len(spike_times)} spike events "
          f"over t=[{spike_times[0]:.0f}, {spike_times[-1]:.0f}]")
    # plot_cpg_vm(vm_record, out_dir)

    # ── 2. Burst-based gait period ───────────────────────────────
    print("\n[2/6] Estimating gait period from burst structure ...")
    gait_period, burst_thresh = estimate_gait_period(
        spike_times, spike_neurons, out_dir,
        N=N, burnin_bursts=args.burnin_bursts, kde_bw=args.kde_bw)

    base_feats, event_phases = encode_spike_events(
        spike_times, spike_neurons, gait_period, N=N, use_phase=args.use_phase)
    plot_spike_events(spike_times, spike_neurons, gait_period, out_dir, N=N)

    # ── 3. Gait tables ──────────────────────────────────────────
    print("\n[3/6] Loading and upsampling gait tables ...")

    gait_names = [
        "tripod", "tripod_huge", "tripod_right", "tripod_huge_right",
        "ripple", "ripple_tiny", "ripple_right", "ripple_tiny_right",
        "tripod_backwards", "tripod_huge_backwards", "tripod_left", "tripod_huge_left",
        "ripple_backwards", "ripple_tiny_backwards", "ripple_left", "ripple_tiny_left",
    ]
    gait_names = gait_names[0:8:2]
    
    gait_names=["bittle_wkF", "bittle_bk", "bittle_wkL", "bittle_wkR"]

    gait_tables_orig = []
    for name in gait_names:
        gait_table = np.loadtxt(f"{this_file_dir}/gaits/{name}.csv",
                                delimiter=",", dtype=np.float32)
        gait_tables_orig.append(gait_table)

    

    for name, g in zip(gait_names, gait_tables_orig):
        print(f"      {name:>4s} : {g.shape[0]} rows × {g.shape[1]} joints (original)")

    # # Upsample to equal row count — equalises phase target resolution
    gait_tables, target_rows = upsample_gait_tables(
        gait_tables_orig, gait_names)
    n_joints = gait_tables[0].shape[1]

    print("\n      Generating burst/gait overlay diagnostic ...")
    plot_burst_gait_overlay(
        spike_times, spike_neurons, gait_period, burst_thresh,
        gait_tables_orig, gait_names, out_dir, n_cycles=6, N=N)
    
    # ── 4. Dataset ──────────────────────────────────────────────
    print("\n[4/6] Building dataset ...")
    print(f"      seq_len={args.seq_len}  "
          f"transition_frac={args.transition_frac:.2f}  "
          f"hidden={args.hidden}")
    rng = np.random.default_rng(args.seed)
    X, y, tgt_range, pure_mask, labels = build_dataset(
        base_feats, event_phases, gait_tables,
        seq_len=args.seq_len,
        transition_frac=args.transition_frac,
        rng=rng)
    n_in = X.shape[2]
    print(f"\n      X : {X.shape}   y : {y.shape}")
    print(f"      tgt_range : [{tgt_range[0]:.1f}, {tgt_range[1]:.1f}]")

    (train_ds, val_pure_ds, val_trans_ds,
     test_pure_ds, test_trans_ds) = train_val_test_split(
        X, y, pure_mask, labels,
        val_frac=args.val, test_frac=args.test, seed=args.seed)

    train_loader = make_loader(train_ds,      args.batch, shuffle=True)
    vp_loader    = make_loader(val_pure_ds,   args.batch, False)
    vt_loader    = make_loader(val_trans_ds,  args.batch, False)
    tp_loader    = make_loader(test_pure_ds,  args.batch, False)
    tt_loader    = make_loader(test_trans_ds, args.batch, False)

    # ── 5. Train ────────────────────────────────────────────────
    print("\n[5/6] Training SNN ...")
    model = CPG_SNN(n_in=n_in, hidden=args.hidden,
                    n_out=n_joints, n_gaits=len(gait_tables),
                    beta=args.beta).to(device)
    if device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters : {n_params:,}")
    print(f"      n_in={n_in}  hidden={args.hidden}  "
          f"beta={args.beta:.3f}  lr={args.lr:.2e}")

    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    # Weighted criterion upweights gaits with fewer original rows (bk)
    criterion  = make_gait_weighted_criterion(gait_tables_orig, device)

    best_val, history = run_training(
        model, train_loader, vp_loader, vt_loader,
        optimizer, scheduler, criterion, device,
        epochs=args.epochs, out_dir=out_dir, weighted=True)

    model.load_state_dict(torch.load(out_dir / "best_model.pt",
                                     map_location=device))

    # Use plain MSE for test reporting so numbers are in comparable units
    plain_mse  = nn.MSELoss()
    # Wrap plain MSE to accept 3-tuple interface
    plain_crit = lambda p, t, _: plain_mse(p, t)
    tp_mse = eval_epoch(model, tp_loader, plain_crit, device, weighted=True)
    tt_mse = eval_epoch(model, tt_loader, plain_crit, device, weighted=True)
    print(f"\n  Test MSE (plain)  pure       : {tp_mse:.6f}")
    print(f"  Test MSE (plain)  transition : {tt_mse:.6f}")

    # ── 6. Plots + export ────────────────────────────────────────
    print("\n[6/6] Generating plots and exporting ...")
    plot_training_curves(history, out_dir)
    full_ds = GaitDataset(X, y, labels)
    plot_inference(model, full_ds, device, out_dir,
                   n_joints=n_joints, n_gaits=len(gait_tables), N=N, use_phase=args.use_phase)
    plot_gait_reconstruction(
        model, X, y, pure_mask, labels, device, out_dir,
        n_joints=n_joints, tgt_range=tgt_range,
        gait_names=gait_names, n_samples=300)

    inference_config = {
        "gait_period":     gait_period,
        "burst_threshold": burst_thresh,
        "global_min":      tgt_range[0],
        "global_max":      tgt_range[1],
        "seq_len":         args.seq_len,
        "n_neurons":       N,
        "n_gaits":         len(gait_tables),
        "n_joints":        n_joints,
        "gait_names":      gait_names,
        # Stepper parameters — deployment must use these exact values
        "chunk_size":      args.chunk_size,
        "spike_thresh":    args.spike_thresh,
        "cpg_start_time":  args.cpg_start_time,
        # Upsampling — inference must upsample gait tables identically
        "target_rows":     target_rows,
        # Feature dim — used to verify ONNX input shape
        "n_in":            n_in,
    }
    export_to_onnx(model, seq_len=args.seq_len, n_in=n_in,
                   out_dir=out_dir, device=device,
                   inference_config=inference_config)

    print(f"\nDone — outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()