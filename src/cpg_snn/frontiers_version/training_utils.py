
import itertools
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.set_float32_matmul_precision('high')

def export_to_onnx(model, seq_len, n_in, out_dir, device,
                    inference_config=None):
    import json

    model.eval()
    dummy_x   = torch.zeros(seq_len, 1, n_in, device=device)
    onnx_path = out_dir / "cpg_snn.onnx"
    model_to_export = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.onnx.export(
        model_to_export, dummy_x, str(onnx_path),
        export_params=True, opset_version=18,
        do_constant_folding=True,
        input_names=["spike_window"],
        output_names=["joint_angles"],
        dynamic_axes={"spike_window": {1: "batch_size"},
                      "joint_angles": {0: "batch_size"}},
        dynamo=False)
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
    print(f"  Feature dim        : {X.shape[2]}  (one-hot + 2 or 0 phases + {n_gaits} gait flag)")

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
        persistent_workers=(num_workers > 0), # Keeps workers alive between epochs
        drop_last=True
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


#### These are the old versions which arent CUDA optimized and dont detach the loss, so they hold memory across epochs.
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