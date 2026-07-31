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

from cpg_utils import CPG_SNN, encode_spike_events, estimate_gait_period, run_blif_cpg, sigmoid, neuron_eqs, make_network
from plotting_utils import (
    plot_burst_gait_overlay,
    plot_cpg_vm,
    plot_gait_reconstruction,
    plot_inference,
    plot_spike_events,
    plot_training_curves,
)


# Shared CPG definitions now live in cpg_utils.py.

# ═══════════════════════════════════════════════════════════════════
# 1.  CPG Dynamics (shared by both integrators)
# ═══════════════════════════════════════════════════════════════════

# Shared CPG helper functions now live in cpg_utils.py.

# ═══════════════════════════════════════════════════════════════════
# 3.  Burst detection + gait period estimation
# ═══════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════
# 4b.  Gait table upsampling
# ═══════════════════════════════════════════════════════════════════

def upsample_gait_tables(gait_tables, gait_names, target_rows=None):
    """
    Upsample all gait tables to the same number of rows via cubic
    interpolation along the phase axis.

    With different row counts (wkF=54, bk=22, wkL=39, wkR=39), the
    phase → target mapping has different angular resolution per gait.
    Shorter tables produce coarser targets (larger quantisation error)
    that inflate the apparent loss for those gaits and make it harder
    for the SNN to learn smooth joint trajectories.

    Upsampling to a common row count (default: max across all tables)
    equalises target resolution without changing the gait shape —
    cubic interpolation preserves the continuous joint trajectory.

    The original gait tables are stored in the config so that the
    inference script can also upsample identically for GT comparison.

    Parameters
    ----------
    gait_tables  : list of (rows_i, J) float32 arrays
    gait_names   : list of str
    target_rows  : int or None  (None → use max row count)

    Returns
    -------
    upsampled    : list of (target_rows, J) float32 arrays
    target_rows  : int  (stored in config for inference)
    """
    from scipy.interpolate import interp1d

    if target_rows is None:
        target_rows = max(g.shape[0] for g in gait_tables)

    upsampled = []
    for gt, name in zip(gait_tables, gait_names):
        n_orig = gt.shape[0]
        if n_orig == target_rows:
            upsampled.append(gt.copy())
            print(f"      {name:>4s} : {n_orig} rows (unchanged)")
        else:
            x_orig = np.linspace(0.0, 1.0, n_orig)
            x_new  = np.linspace(0.0, 1.0, target_rows)
            interp = interp1d(x_orig, gt, axis=0, kind='cubic',
                              fill_value='extrapolate')
            gt_up  = interp(x_new).astype(np.float32)
            upsampled.append(gt_up)
            print(f"      {name:>4s} : {n_orig} → {target_rows} rows "
                  f"(cubic upsampled)")

    return upsampled, target_rows


# ═══════════════════════════════════════════════════════════════════
# 5.  Diagnostic: burst boundaries vs gait table
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# 6.  Helper: phase → gait-table row
# ═══════════════════════════════════════════════════════════════════

def phase_to_row(phase_rad, n_rows):
    return (phase_rad / (2.0 * np.pi) * n_rows).astype(int) % n_rows


# ═══════════════════════════════════════════════════════════════════
# 7.  Dataset builder
# ═══════════════════════════════════════════════════════════════════

def build_dataset(base_feats, event_phases, gait_tables,
                  seq_len=32, transition_frac=0.30, rng=None):
    """
    Build the full training dataset with pure-gait and transition windows.

    FiLM architecture change
    ------------------------
    The gait flag is NO LONGER concatenated into the per-event feature
    vector.  Instead the gait index is stored in `labels` and passed
    separately to the FiLM conditioning layer at forward time.

    Why: LIF neurons cannot discriminate a static, constant-per-window
    gait flag.  Any non-zero fc1 weight from the flag drives the LIF
    membrane to a saturated firing rate that encodes *magnitude* (always
    the same for a given gait) rather than gait *identity*.  The hidden
    spike pattern becomes identical for all gaits, making multi-gait
    decoding impossible regardless of depth or hidden size.

    With FiLM, the gait index conditions the analog readout membrane via
    learned per-gait scale (gamma) and shift (beta), completely bypassing
    the spike-discretisation bottleneck.

    Input feature per event: [one_hot_neuron(N), sin_abs, cos_abs,
                               sin_rel, cos_rel]  — N+4 dims, NO gait flag.

    Window label = target gait index (int).  For transition windows this
    is the *new* gait (the one whose table provides the target angles),
    matching inference where gait_idx switches atomically on gesture input.

    Parameters
    ----------
    base_feats      : (E, N+4)  float32  from encode_spike_events
    event_phases    : (E,)      float32  absolute phase rad per event
    gait_tables     : list of (target_rows, J) float32  (already upsampled)
    seq_len         : int
    transition_frac : float  fraction of windows that are transition type
    rng             : np.random.Generator

    Returns
    -------
    X         : (N_total, seq_len, N+4)   float32  — no gait flag
    y         : (N_total, J)              float32  normalised targets
    tgt_range : (min, max)
    pure_mask : (N_total,)  bool
    labels    : (N_total,)  int32         gait index for FiLM conditioning
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_gaits = len(gait_tables)
    E       = len(base_feats)
    J       = gait_tables[0].shape[1]
    N_win   = E - seq_len

    if N_win <= 0:
        raise ValueError(
            f"seq_len ({seq_len}) >= spike events ({E}). "
            "Increase tmax or reduce seq_len.")

    all_vals         = np.concatenate([g.flatten() for g in gait_tables])
    tgt_min, tgt_max = float(all_vals.min()), float(all_vals.max())

    def normalise(arr):
        return ((arr - tgt_min) / (tgt_max - tgt_min + 1e-8) * 2 - 1
                ).astype(np.float32)

    gait_norms  = [normalise(g) for g in gait_tables]
    last_phases = event_phases[seq_len - 1: seq_len - 1 + N_win]

    # ── Pure-gait windows ────────────────────────────────────────
    pure_X_parts, pure_y_parts, pure_lbl = [], [], []
    gait_onehot = np.eye(n_gaits, dtype=np.float32)
    for g in range(n_gaits):
        flag     = gait_onehot[g]
        n_rows_g = gait_norms[g].shape[0]
        flag_col = np.tile(flag, (seq_len, 1))          # (seq_len, n_gaits)
        windows  = np.stack(
            [np.concatenate([base_feats[s: s + seq_len], flag_col], axis=1)
             for s in range(N_win)])                    # (N_win, seq_len, N+4+n_gaits)
        row_idx  = phase_to_row(last_phases, n_rows_g)
        targets  = gait_norms[g][row_idx]
        pure_X_parts.append(windows)
        pure_y_parts.append(targets)
        pure_lbl.append(np.full(N_win, g, dtype=np.int32))

    pure_X   = np.concatenate(pure_X_parts, axis=0)
    pure_y   = np.concatenate(pure_y_parts, axis=0)
    pure_lbl = np.concatenate(pure_lbl,     axis=0)

    # ── Transition windows ────────────────────────────────────────
    # Per-event gait flag switches from flag_a to flag_b at a random
    # point in the last quarter of the window — matches inference where
    # the gesture sensor fires mid-stride.
    sw_low  = (3 * seq_len) // 4
    sw_high = seq_len - 1
    pairs = [(a, b) for a, b in itertools.product(range(n_gaits), repeat=2)
             if a != b]

    trans_X_parts, trans_y_parts, trans_lbl = [], [], []
    for (ga, gb) in pairs:
        flag_a   = gait_onehot[ga]
        flag_b   = gait_onehot[gb]
        n_rows_b = gait_norms[gb].shape[0]
        switch_pts = rng.integers(sw_low, sw_high + 1, size=N_win)
        windows = []
        for k in range(N_win):
            p        = int(switch_pts[k])
            flag_col = np.empty((seq_len, n_gaits), dtype=np.float32)
            flag_col[:p]  = flag_a
            flag_col[p:]  = flag_b
            windows.append(
                np.concatenate([base_feats[k: k + seq_len], flag_col], axis=1))
        windows = np.stack(windows)
        row_idx = phase_to_row(last_phases, n_rows_b)
        targets = gait_norms[gb][row_idx]
        trans_X_parts.append(windows)
        trans_y_parts.append(targets)
        trans_lbl.append(np.full(N_win, gb, dtype=np.int32))

    trans_X   = np.concatenate(trans_X_parts, axis=0)
    trans_y   = np.concatenate(trans_y_parts, axis=0)
    trans_lbl = np.concatenate(trans_lbl,     axis=0)

    # ── Subsample pure to hit transition_frac ────────────────────
    n_trans       = len(trans_X)
    n_pure_target = max(1, int(round(n_trans * (1.0 - transition_frac)
                                     / transition_frac)))
    n_pure_target = min(n_pure_target, len(pure_X))
    idx      = rng.permutation(len(pure_X))[:n_pure_target]
    pure_X   = pure_X[idx]
    pure_y   = pure_y[idx]
    pure_lbl = pure_lbl[idx]

    # ── Merge + shuffle ───────────────────────────────────────────
    X         = np.concatenate([pure_X,   trans_X],   axis=0).astype(np.float32)
    y         = np.concatenate([pure_y,   trans_y],   axis=0).astype(np.float32)
    labels    = np.concatenate([pure_lbl, trans_lbl], axis=0)
    pure_mask = np.concatenate(
        [np.ones(len(pure_X),  dtype=bool),
         np.zeros(len(trans_X), dtype=bool)], axis=0)
    shuf = rng.permutation(len(X))
    X, y, labels, pure_mask = (X[shuf], y[shuf], labels[shuf], pure_mask[shuf])

    actual_frac = (~pure_mask).sum() / len(pure_mask)
    print(f"  Pure windows       : {pure_mask.sum():>8,}")
    print(f"  Transition windows : {(~pure_mask).sum():>8,}"
          f"  (actual frac = {actual_frac:.2f})")
    print(f"  Total              : {len(X):>8,}")
    print(f"  Feature dim        : {X.shape[2]}  (one-hot + 2 phase + {n_gaits} gait flag)")

    return X, y, (tgt_min, tgt_max), pure_mask, labels


# ═══════════════════════════════════════════════════════════════════
# 8.  Dataset / DataLoader
# ═══════════════════════════════════════════════════════════════════

class GaitDataset(Dataset):
    def __init__(self, X, y, labels=None):
        self.X      = torch.tensor(X, dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.float32)
        # labels: gait index per window, used for weighted loss
        self.labels = (torch.tensor(labels, dtype=torch.long)
                       if labels is not None
                       else torch.zeros(len(X), dtype=torch.long))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.labels[idx]


def snn_collate(batch):
    X, y, lbl = zip(*batch)
    return torch.stack(X).permute(1, 0, 2), torch.stack(y), torch.stack(lbl)


def make_loader(ds, batch_size, shuffle, num_workers=4, pin_memory=True):
    if len(ds) == 0:
        return []
    
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=snn_collate, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0) # Keeps workers alive between epochs
    )


def train_val_test_split(X, y, pure_mask, labels,
                          val_frac=0.15, test_frac=0.10, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n   = len(X)
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val - n_test
    tr_idx   = idx[:n_train]
    val_idx  = idx[n_train: n_train + n_val]
    test_idx = idx[n_train + n_val:]

    def split_mask(indices):
        pm = pure_mask[indices]
        return indices[pm], indices[~pm]

    vp_idx, vt_idx = split_mask(val_idx)
    tp_idx, tt_idx = split_mask(test_idx)

    train_ds      = GaitDataset(X[tr_idx],  y[tr_idx],  labels[tr_idx])
    val_pure_ds   = GaitDataset(X[vp_idx],  y[vp_idx],  labels[vp_idx])
    val_trans_ds  = GaitDataset(X[vt_idx],  y[vt_idx],  labels[vt_idx])
    test_pure_ds  = GaitDataset(X[tp_idx],  y[tp_idx],  labels[tp_idx])
    test_trans_ds = GaitDataset(X[tt_idx],  y[tt_idx],  labels[tt_idx])

    print(f"  Train           : {len(train_ds):>8,}")
    print(f"  Val   pure      : {len(val_pure_ds):>8,}")
    print(f"  Val   trans     : {len(val_trans_ds):>8,}")
    print(f"  Test  pure      : {len(test_pure_ds):>8,}")
    print(f"  Test  trans     : {len(test_trans_ds):>8,}")

    return train_ds, val_pure_ds, val_trans_ds, test_pure_ds, test_trans_ds


# ═══════════════════════════════════════════════════════════════════
# 9.  SNN Model
# ═══════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════
# 10.  Training
# ═══════════════════════════════════════════════════════════════════

def make_gait_weighted_criterion(gait_tables_orig, device):
    """
    Per-gait MSE weight inversely proportional to angular range.

    Gaits with a larger joint-angle range (wkF, wkL, wkR) produce
    larger absolute errors for the same fractional error, but the SNN
    is trained on NORMALISED targets [−1, 1] so MSE is already scale-
    equalised.  The residual imbalance comes from the fact that gaits
    with FEWER original table rows have coarser phase targets, making
    them harder to fit precisely.

    Weight = target_rows / row_count_i
    This upweights gaits that were upsampled more aggressively (bk: 22→54
    gets weight 2.45×) and keeps wkF at 1.0, so the loss gradient is
    proportional to the difficulty rather than the raw row count.

    Returns a callable loss(pred, target, gait_labels) that computes
    per-sample MSE weighted by the label's gait weight.
    """
    max_rows = max(g.shape[0] for g in gait_tables_orig)
    weights  = torch.tensor(
        [max_rows / g.shape[0] for g in gait_tables_orig],
        dtype=torch.float32, device=device)
    print(f"  Gait loss weights: "
          + "  ".join(f"g{i}={weights[i].item():.2f}"
                      for i in range(len(gait_tables_orig))))

    def weighted_criterion(pred, target, gait_labels):
        # pred, target : (B, J)
        # gait_labels  : (B,) int  — new gait index for each window
        w   = weights[gait_labels]           # (B,)
        mse = ((pred - target) ** 2).mean(dim=1)  # (B,)
        return (w * mse).mean()

    return weighted_criterion


# def train_epoch(model, loader, optimizer, criterion, device,
#                 weighted=False):
#     model.train()
#     total = 0.0
#     for batch in loader:
#         X, y, glbl = batch
#         X, y, glbl = X.to(device), y.to(device), glbl.to(device)
#         optimizer.zero_grad()
#         pred = model(X, glbl)                  # FiLM: pass gait index
#         loss = criterion(pred, y, glbl) if weighted else criterion(pred, y)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         total += loss.item()
#     return total / len(loader)


# @torch.no_grad()
# def eval_epoch(model, loader, criterion, device, weighted=False):
#     model.eval()
#     if not loader:
#         return float("nan")
#     total = 0.0
#     for batch in loader:
#         X, y, glbl = batch
#         X, y, glbl = X.to(device), y.to(device), glbl.to(device)
#         pred  = model(X, glbl)                 # FiLM: pass gait index
#         loss  = criterion(pred, y, glbl) if weighted else criterion(pred, y)
#         total += loss.item()
#     return total / len(loader)
def train_epoch(model, loader, optimizer, criterion, device, weighted=False):
    model.train()
    total_loss = 0.0  # Or keep as 0.0 scalar float

    for batch in loader:
        X, y, glbl = batch
        X, y, glbl = X.to(device, non_blocking=True), y.to(device, non_blocking=True), glbl.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        pred = model(X, glbl)
        loss = criterion(pred, y, glbl) if weighted else criterion(pred, y)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Detach loss so it doesn't hold memory, but stay on GPU/scalar float sum
        total_loss += loss.detach()

    # Convert to standard Python float ONCE per epoch
    return (total_loss / len(loader)).item()
@torch.no_grad()
def eval_epoch(model, loader, criterion, device, weighted=False):
    model.eval()
    if not loader:
        return float("nan")
    
    total_loss = 0.0
    for batch in loader:
        X, y, glbl = batch
        X, y, glbl = X.to(device, non_blocking=True), y.to(device, non_blocking=True), glbl.to(device, non_blocking=True)
        
        pred = model(X, glbl)
        loss = criterion(pred, y, glbl) if weighted else criterion(pred, y)
        
        total_loss += loss

    return (total_loss / len(loader)).item()


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


# ═══════════════════════════════════════════════════════════════════
# 11.  ONNX export
# ═══════════════════════════════════════════════════════════════════

def export_to_onnx(model, seq_len, n_in, out_dir, device,
                    inference_config=None):
    import json

    model.eval()
    dummy_x   = torch.zeros(seq_len, 1, n_in, device=device)
    onnx_path = out_dir / "cpg_snn.onnx"
    torch.onnx.export(
        model, dummy_x, str(onnx_path),
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=["spike_window"],
        output_names=["joint_angles"],
        dynamic_axes={"spike_window": {1: "batch_size"},
                      "joint_angles": {0: "batch_size"}})
    print(f"  [saved] ONNX → {onnx_path}")

    try:
        import onnxruntime as ort
        sess    = ort.InferenceSession(str(onnx_path),
                                       providers=["CPUExecutionProvider"])
        pt_out  = model(dummy_x).squeeze(0).detach().cpu().numpy()
        ort_out = sess.run(["joint_angles"],
                           {"spike_window": dummy_x.cpu().numpy()})[0].squeeze(0)
        diff    = float(np.abs(pt_out - ort_out).max())
        print(f"  PyTorch vs ONNX max diff : {diff:.2e}"
              f"  ({'OK' if diff < 1e-4 else 'WARNING'})")
    except ImportError:
        print("  onnxruntime not installed — skipping sanity check.")

    if inference_config is not None:
        cfg_path = out_dir / "cpg_snn_config.json"
        clean = {}
        for k, v in inference_config.items():
            if isinstance(v, list):
                clean[k] = v
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = float(v) if isinstance(v, (int, float)) else v
        # Also save chunk_size so the deployment stepper uses the same value
        clean["chunk_size"] = int(inference_config.get("chunk_size", 50))
        with open(cfg_path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"  [saved] config → {cfg_path}")
        print(f"          gait_period     = {clean.get('gait_period', 0):.1f}")
        print(f"          burst_threshold = {clean.get('burst_threshold', 0):.1f}")
        print(f"          chunk_size      = {clean.get('chunk_size', 0)}")
        print(f"          global_min/max  = "
              f"{clean.get('global_min', 0):.1f} / "
              f"{clean.get('global_max', 0):.1f}")

    return onnx_path


def encode_spike_events_phaseless(spike_times, spike_neurons, gait_period, N=4):
    # ── Absolute phase ────────────────────────────────────────────
    abs_phase = (2.0 * np.pi
                 * (spike_times % gait_period) / gait_period
                 ).astype(np.float32)

    # ── Relative phase (ISI-based) ────────────────────────────────
    isis = np.empty(len(spike_times), dtype=np.float32)
    isis[0]  = gait_period          # neutral: first event has no predecessor
    isis[1:] = np.diff(spike_times)
    rel_phase = (2.0 * np.pi * isis / gait_period).astype(np.float32)

    one_hot = np.zeros((len(spike_times), N), dtype=np.float32)
    one_hot[np.arange(len(spike_times)), spike_neurons] = 1.0

    base_feats = one_hot

    print(f"  Base feature matrix : {base_feats.shape}"
          f"  ({N} one-hot + sin/cos abs-phase + sin/cos rel-phase)"
          f"  [gait flag added per-event in build_dataset]")
    return base_feats, abs_phase

# ═══════════════════════════════════════════════════════════════════
# 15.  Main
# ═══════════════════════════════════════════════════════════════════

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
    parser.add_argument("--seq_len",         type=int,   default=3,
                        help="Spike events per input window. "
                             "Needs to span ~1 full gait cycle (~32 spikes) "
                             "for reliable phase tracking.")
    parser.add_argument("--hidden",          type=int,   default=128)
    parser.add_argument("--beta",            type=float, default=0.9)

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

    N = 6

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
    
    # spike_times, spike_neurons, vm_record = run_cpg_chunked(
    #     N=N,
    #     tmax=args.tmax,
    #     cpg_start_time=args.cpg_start_time,
    #     chunk_size=args.chunk_size,
    #     spike_thresh=args.spike_thresh,
    # )
    print(os.getcwd())
    spike_times, spike_neurons = run_blif_cpg(N=N, t_max = args.tmax, cpg_start_time=args.cpg_start_time)
    
    print(f"      Collected {len(spike_times)} spike events "
          f"over t=[{spike_times[0]:.0f}, {spike_times[-1]:.0f}]")
    # plot_cpg_vm(vm_record, out_dir)

    # ── 2. Burst-based gait period ───────────────────────────────
    print("\n[2/6] Estimating gait period from burst structure ...")
    gait_period, burst_thresh = estimate_gait_period(
        spike_times, spike_neurons, out_dir,
        N=N, burnin_bursts=args.burnin_bursts, kde_bw=args.kde_bw)

    base_feats, event_phases = encode_spike_events_phaseless(
        spike_times, spike_neurons, gait_period, N=N)
    plot_spike_events(spike_times, spike_neurons, gait_period, out_dir, N=N)

    # ── 3. Gait tables ──────────────────────────────────────────
    print("\n[3/6] Loading and upsampling gait tables ...")

    base_gait_names = [
        "tripod", "tripod_huge", "tripod_right", "tripod_huge_right",
        "ripple", "ripple_tiny", "ripple_right", "ripple_tiny_right",
    ]
    mirrored_gait_names = [
        "tripod_backwards", "tripod_huge_backwards", "tripod_left", "tripod_huge_left",
        "ripple_backwards", "ripple_tiny_backwards", "ripple_left", "ripple_tiny_left",
    ]
    gait_names = base_gait_names + mirrored_gait_names

    gait_tables_orig = []
    for name in base_gait_names:
        gait_table = np.loadtxt(f"{this_file_dir}/gaits/{name}.csv",
                                delimiter=",", dtype=np.float32)
        gait_tables_orig.append(gait_table)

    for name in mirrored_gait_names:
        base_name = name.replace("_backwards", "").replace("_left", "_right")
        gait_table = np.loadtxt(f"{this_file_dir}/gaits/{base_name}.csv",
                                delimiter=",", dtype=np.float32)
        gait_tables_orig.append(np.flip(gait_table, axis=0).copy())

    # gait_tables_orig = [wkF, bk, wkL, wkR]
    # gait_names       = ["wkF", "bk", "wkL", "wkR"]

    gait_names = gait_names[0:8:2]
    gait_tables_orig = gait_tables_orig[0:8:2]

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
                   n_joints=n_joints, n_gaits=len(gait_tables))
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