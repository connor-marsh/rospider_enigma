"""Reusable plotting helpers for the CPG-SNN training and inference scripts."""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


CPG_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"]
PRED_COLORS = ["#e63946", "#f4a261", "#2a9d8f", "#6a0572"]
TRUE_COLOR = "#457b9d"


def savefig(fig, out_dir, name, dpi=150, bbox_inches=None):
    path = Path(out_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if bbox_inches is None:
        fig.savefig(path, dpi=dpi)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    print(f"  [saved] {path}")
    return path

def plot_blif_cpg(spikes, vms, out_dir, n_show=400):

    fig, axes = plt.subplots(1, 1)
    for i in range(spikes.shape[0]):
        axes.plot(spikes[i, :n_show], label=f"Neuron {i+1} spikes")
    axes.legend()


    plt.suptitle("BLIF CPG Spikes", fontsize=12)
    plt.tight_layout()
    savefig(fig, out_dir, "blif_cpg.png")

def plot_cpg_vm(vm_record, out_dir, n_show=30_000):
    t_axis = vm_record["t"] if isinstance(vm_record, dict) else vm_record.t
    y_mat = vm_record["y"] if isinstance(vm_record, dict) else vm_record.y
    N = y_mat.shape[0] // 4

    n_pts = min(n_show, t_axis.shape[0])
    t_plot = t_axis[:n_pts]
    y_plot = y_mat[:, :n_pts]

    fig, axes = plt.subplots(N, 1, figsize=(14, 8), sharex=True)
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a0572", "#8ecae6"]
    if N == 1:
        axes = [axes]
    for i in range(N):
        axes[i].plot(t_plot, y_plot[i * 4, :], color=colors[i], lw=0.9)
        axes[i].set_ylabel(f"CPG {i}\n$v_m$", fontsize=9)
        axes[i].axhline(-2.0, color="k", ls="--", lw=0.7, alpha=0.5, label="threshold")
        axes[i].grid(True, alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel(f"Time (first {n_pts} steps)")
    plt.suptitle("CPG Membrane Potentials — chunk-based integrator", fontsize=12)
    plt.tight_layout()
    savefig(fig, out_dir, "cpg_vm.png")


def plot_spike_events(spike_times, spike_neurons, gait_period, out_dir, n_show=3_000, N=4):
    if len(spike_times) == 0:
        return
    mask = spike_times <= spike_times[0] + n_show
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a0572", "#8ecae6"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, height_ratios=[3, 1])
    for i in range(N):
        idx = np.where((spike_neurons == i) & mask)[0]
        ax1.scatter(spike_times[idx], np.full(len(idx), i), marker="|", s=150, lw=1.8, color=colors[i], label=f"Neuron {i}")
    ax1.set_yticks(range(N))
    ax1.set_yticklabels([f"CPG {i}" for i in range(N)])
    title = f"Spike Events  (gait_period ≈ {gait_period:.0f} steps)" if np.isfinite(gait_period) else "Spike Events"
    ax1.set_title(title)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, axis="x", alpha=0.2)
    t_show = spike_times[mask]
    if np.isfinite(gait_period):
        phase = np.degrees(2.0 * np.pi * (t_show % gait_period) / gait_period)
        ax2.plot(t_show, phase, color="#6a0572", lw=1.2)
        ax2.set_ylabel("Phase (°)"); ax2.set_xlabel("Time")
        ax2.set_title("Gait phase at each spike event")
    else:
        ax2.axis("off")
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    savefig(fig, out_dir, "spike_events.png")


def plot_spike_event_overview(spike_times, spike_neurons, rec_phase_deg, rec_pred, rec_true, rec_gait_idx, n_joints, gait_names, out_dir):
    E = len(spike_times)
    if E == 0:
        return
    ev_idx = np.arange(E)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 1, hspace=0.5, height_ratios=[1, 1, 3])

    ax0 = fig.add_subplot(gs[0])
    for i in range(4):
        mask = spike_neurons == i
        ax0.scatter(ev_idx[mask], np.full(mask.sum(), i), marker="|", s=120, lw=1.6, color=CPG_COLORS[i], label=f"CPG {i}")
    ax0.set_yticks(range(4))
    ax0.set_yticklabels([f"CPG {i}" for i in range(4)])
    ax0.set_title("Neuron identity at each spike event")
    ax0.legend(fontsize=7, loc="upper right", ncol=4)
    ax0.grid(True, axis="x", alpha=0.2)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    sc = ax1.scatter(ev_idx, rec_phase_deg, c=rec_phase_deg, cmap="hsv", s=8, vmin=0, vmax=360)
    plt.colorbar(sc, ax=ax1, label="Phase (°)", pad=0.01)
    ax1.set_ylabel("Phase (°)")
    ax1.set_title("Gait phase at each spike event")
    ax1.set_ylim(-5, 370)
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    jcols = plt.cm.tab10(np.linspace(0, 1, n_joints))
    for j in range(n_joints):
        ax2.plot(ev_idx, rec_true[:, j], color=jcols[j], lw=1.4, alpha=0.9, label=f"J{j+1}")
        ax2.plot(ev_idx, rec_pred[:, j], color=jcols[j], lw=1.2, alpha=0.75, ls="--")
    ax2.set_xlabel("Spike event index")
    ax2.set_ylabel("Angle (°)")
    ax2.set_title("All joints: GT (solid) vs Predicted (dashed)")
    ax2.legend(fontsize=7, ncol=4, loc="upper right", title="solid=GT  dashed=pred")
    ax2.grid(True, alpha=0.2)

    gait_palette = ["#e6f0ff", "#fff3e6", "#e6fff3", "#ffe6f0"]
    prev_g, prev_e = int(rec_gait_idx[0]), 0
    for e in range(1, E):
        g = int(rec_gait_idx[e])
        if g != prev_g or e == E - 1:
            for ax in [ax0, ax1, ax2]:
                ax.axvspan(prev_e, e, alpha=0.15, color=gait_palette[prev_g % len(gait_palette)])
            prev_g, prev_e = g, e

    plt.suptitle(f"Spike-event Inference Overview  ({E} total events)", fontsize=13, fontweight="bold")
    savefig(fig, out_dir, "spike_event_overview.png")


def plot_latency(latencies, out_dir):
    if len(latencies) == 0:
        return
    lat = np.array(latencies)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(lat, bins=50, color="#457b9d", edgecolor="white", lw=0.5)
    ax1.axvline(np.mean(lat), color="#e63946", lw=1.5, ls="--", label=f"mean={np.mean(lat):.2f} ms")
    ax1.axvline(np.median(lat), color="#f4a261", lw=1.5, ls="--", label=f"median={np.median(lat):.2f} ms")
    ax1.set_xlabel("Latency (ms)"); ax1.set_ylabel("Count")
    ax1.set_title("Inference Latency Distribution")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)
    sl = np.sort(lat)
    cdf = np.arange(1, len(lat) + 1) / len(lat)
    ax2.plot(sl, cdf * 100, color="#457b9d", lw=1.8)
    ax2.axvline(np.percentile(lat, 95), color="#e63946", lw=1.5, ls="--", label=f"p95={np.percentile(lat, 95):.2f} ms")
    ax2.axvline(np.percentile(lat, 99), color="#f4a261", lw=1.5, ls="--", label=f"p99={np.percentile(lat, 99):.2f} ms")
    ax2.set_xlabel("Latency (ms)"); ax2.set_ylabel("Cumulative %")
    ax2.set_title("Latency CDF")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)
    plt.suptitle("ONNX Inference Latency  (per spike event)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir, "inference_latency.png")


def plot_training_curves(history, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], label="Train", lw=2, color="#457b9d")
    ax.plot(epochs, history["val_pure"], label="Val (pure)", lw=2, color="#2a9d8f", ls="--")
    ax.plot(epochs, history["val_trans"], label="Val (trans)", lw=2, color="#e63946", ls=":")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training — pure vs transition validation loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig, out_dir, "training_curves.png")


def plot_burst_gait_overlay(spike_times, spike_neurons, gait_period, threshold, gait_tables, gait_names, out_dir, n_cycles=6, N=4):
    t0 = spike_times[spike_neurons == 0]
    isis_n0 = np.diff(t0)
    burst_starts = [t0[0]]
    for i in range(1, len(t0)):
        if isis_n0[i - 1] > threshold:
            burst_starts.append(t0[i])
    burst_starts = np.array(burst_starts)

    if len(burst_starts) < n_cycles + 1:
        n_cycles = len(burst_starts) - 1
    bs = burst_starts[:n_cycles + 1]

    colors_n = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a0572", "#8ecae6"]
    colors_gt = ["#e63946", "#f4a261", "#2a9d8f", "#6a0572", "#8ecae6", "#ffb703"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    t_end = bs[-1] + gait_period * 0.1
    ax0 = axes[0]
    for i in range(N):
        mask = (spike_times >= bs[0]) & (spike_times <= t_end) & (spike_neurons == i)
        ax0.scatter(spike_times[mask], np.full(mask.sum(), i), marker="|", s=120, lw=1.6, color=colors_n[i], label=f"Neuron {i}")
    for b in bs:
        ax0.axvline(b, color="k", lw=1.0, alpha=0.5, ls="--")
    ax0.set_yticks(range(N))
    ax0.set_yticklabels([f"N{i}" for i in range(N)])
    ax0.set_title(f"Spike raster — first {n_cycles} gait cycles  (dashed = neuron-0 burst start)")
    ax0.legend(loc="upper right", fontsize=8, ncol=4)
    ax0.grid(True, axis="x", alpha=0.2)

    ax1 = axes[1]
    t_fine = np.linspace(bs[0], bs[-1], 500)
    phase_f = (2.0 * np.pi * (t_fine % gait_period) / gait_period)

    for g_idx, (gt, name) in enumerate(zip(gait_tables, gait_names)):
        n_rows = gt.shape[0]
        row_idx = (phase_f / (2.0 * np.pi) * n_rows).astype(int) % n_rows
        ax1.plot(t_fine, gt[row_idx, 0], color=colors_gt[g_idx % len(colors_gt)], lw=1.5, label=f"{name} J1")
    for b in bs:
        ax1.axvline(b, color="k", lw=1.0, alpha=0.5, ls="--")
    ax1.set_xlabel("Simulation time (steps)")
    ax1.set_ylabel("Joint 1 angle (°)")
    ax1.set_title("Gait-table joint 1 angle phase-indexed over gait cycles")
    ax1.legend(fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.25)

    plt.tight_layout()
    savefig(fig, out_dir, "burst_gait_overlay.png")


def plot_inference(model, dataset, device, out_dir, n_joints, n_gaits, sample_idx=0, N=4, use_phase=True):
    model.eval()
    X_np, y_np, lbl = dataset[sample_idx]
    gait_idx_t = lbl.unsqueeze(0).to(device)
    X_in = X_np.unsqueeze(1).to(device)
    with torch.no_grad():
        pred, spk1, spk2 = model(X_in, gait_idx_t, return_recordings=True)
    pred_np = pred.squeeze(0).cpu().numpy()
    y_np = y_np.numpy()
    spk1_np = spk1[:, 0, :].cpu().numpy()
    spk2_np = spk2[:, 0, :].cpu().numpy()
    onehot = X_np[:, :N].numpy()
    sin_ph = X_np[:, N].numpy()
    cos_ph = X_np[:, N + 1].numpy()
    gait_name = f"Gait {lbl.item()}"

    T, hidden = spk1_np.shape
    n_show = min(24, hidden)
    show_idx = np.random.choice(hidden, n_show, replace=False)
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a0572", "#8ecae6"]

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    for i in range(N):
        t_ev = np.where(onehot[:, i] > 0.5)[0]
        ax0.scatter(t_ev, np.full_like(t_ev, i), marker="|", s=120, color=colors[i % len(colors)])
    ax0.set_yticks(range(N))
    ax0.set_yticklabels([f"CPG {i}" for i in range(N)])
    ax0.set_title(f"Input spike-event window (seq_len={T})  [{gait_name}]")
    ax0.grid(True, axis="x", alpha=0.2)
    ax0b = ax0.twinx()
    ax0b.plot(sin_ph, color="purple", lw=1.2, alpha=0.6, ls="--", label="sin(φ_abs)")
    ax0b.plot(cos_ph, color="gray", lw=1.2, alpha=0.6, ls=":", label="cos(φ_abs)")
    ax0b.set_ylabel("Phase channels", fontsize=8)
    ax0b.legend(fontsize=7, loc="upper right")

    ax_gait = fig.add_subplot(gs[0, 1])
    N_feat = N + 2 if use_phase else N
    gait_fl = X_np[:, N_feat:].numpy()
    for g in range(n_gaits):
        ax_gait.plot(gait_fl[:, g], label=f"Gait {g}", lw=1.5, color=colors[g % len(colors)])
    ax_gait.set_title(f"Per-event gait flag — {gait_name}")
    ax_gait.set_xlabel("Event index"); ax_gait.set_ylabel("Flag value")
    ax_gait.set_ylim(-0.1, 1.1); ax_gait.legend(fontsize=7)
    ax_gait.grid(True, alpha=0.2)

    ax1 = fig.add_subplot(gs[1, 0])
    for row, nid in enumerate(show_idx):
        t_spk = np.where(spk1_np[:, nid] > 0.5)[0]
        ax1.scatter(t_spk, np.full_like(t_spk, row), marker="|", s=60, color="#457b9d")
    ax1.set_title(f"Hidden Layer 1 ({n_show}/{hidden} neurons)")
    ax1.set_xlabel("Event index"); ax1.set_ylabel("Neuron")
    ax1.grid(True, axis="x", alpha=0.2)

    ax2 = fig.add_subplot(gs[1, 1])
    for row, nid in enumerate(show_idx):
        t_spk = np.where(spk2_np[:, nid] > 0.5)[0]
        ax2.scatter(t_spk, np.full_like(t_spk, row), marker="|", s=60, color="#f4a261")
    ax2.set_title(f"Hidden Layer 2 ({n_show}/{hidden} neurons)")
    ax2.set_xlabel("Event index"); ax2.set_ylabel("Neuron")
    ax2.grid(True, axis="x", alpha=0.2)

    ax3 = fig.add_subplot(gs[2, :])
    x = np.arange(n_joints); w = 0.35
    ax3.bar(x - w / 2, y_np, w, label="True", color="#457b9d", alpha=0.85)
    ax3.bar(x + w / 2, pred_np, w, label="Predicted", color="#f4a261", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"J{i+1}" for i in range(n_joints)], rotation=45, ha="right")
    ax3.set_ylabel("Angle (normalised)")
    ax3.set_title("True vs Predicted Joint Angles")
    ax3.legend(); ax3.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"SNN Inference — Sample #{sample_idx}  ({gait_name})", fontsize=13)
    savefig(fig, out_dir, "snn_inference.png")


def plot_inference_summary(rec_pred, rec_true, rec_gait_idx, n_joints, gait_names, out_dir, n_samples_per_gait=500):
    out_dir = out_dir / "recons_inference"
    n_gaits = len(gait_names)
    all_true_denorm, all_pred_denorm = [], []

    for g in range(n_gaits):
        mask = rec_gait_idx == g
        if not mask.any():
            all_true_denorm.append(np.zeros((0, n_joints)))
            all_pred_denorm.append(np.zeros((0, n_joints)))
            print(f"  {gait_names[g]}: no events — skipping.")
            continue

        n_plot = min(n_samples_per_gait, mask.sum())
        true_plot = rec_true[mask][:n_plot]
        pred_plot = rec_pred[mask][:n_plot]
        all_true_denorm.append(true_plot)
        all_pred_denorm.append(pred_plot)

        cols = min(4, n_joints)
        rows = int(np.ceil(n_joints / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
        ev = np.arange(n_plot)
        for j in range(n_joints):
            ax = axes[j // cols][j % cols]
            rmse = np.sqrt(np.mean((pred_plot[:, j] - true_plot[:, j]) ** 2))
            ax.plot(ev, true_plot[:, j], label="GT", color=TRUE_COLOR, lw=1.8, zorder=3)
            ax.plot(ev, pred_plot[:, j], label="Predicted", color=PRED_COLORS[g % len(PRED_COLORS)], lw=1.5, ls="--", alpha=0.9, zorder=2)
            err = np.abs(pred_plot[:, j] - true_plot[:, j])
            ax.fill_between(ev, pred_plot[:, j] - err, pred_plot[:, j] + err, color=PRED_COLORS[g % len(PRED_COLORS)], alpha=0.12, zorder=1)
            ax.set_title(f"Joint {j+1}  (RMSE={rmse:.2f}°)", fontsize=9)
            ax.set_xlabel("Spike-event window index", fontsize=8)
            ax.set_ylabel("Angle (°)", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.25)
        for j in range(n_joints, rows * cols):
            axes[j // cols][j % cols].set_visible(False)
        plt.suptitle(f"GT vs Predicted — {gait_names[g]}  (first {n_plot} windows)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        savefig(fig, out_dir, f"gait_reconstruction_{gait_names[g]}.png")

    active = [g for g in range(n_gaits) if len(all_true_denorm[g]) > 0]
    if active:
        fig, axes = plt.subplots(len(active), 1, figsize=(14, 3 * len(active)), sharex=False)
        if len(active) == 1:
            axes = [axes]
        for row, g in enumerate(active):
            ax = axes[row]
            true_plot = all_true_denorm[g]
            pred_plot = all_pred_denorm[g]
            ev = np.arange(len(true_plot))
            rmse = np.sqrt(np.mean((pred_plot[:, 0] - true_plot[:, 0]) ** 2))
            ax.plot(ev, true_plot[:, 0], label="GT", color=TRUE_COLOR, lw=1.8)
            ax.plot(ev, pred_plot[:, 0], label="Predicted", color=PRED_COLORS[g % len(PRED_COLORS)], lw=1.5, ls="--", alpha=0.9)
            ax.fill_between(ev, true_plot[:, 0], pred_plot[:, 0], alpha=0.18, color=PRED_COLORS[g % len(PRED_COLORS)])
            ax.set_title(f"{gait_names[g]} — Joint 1  (RMSE={rmse:.2f}°)", fontsize=10)
            ax.set_ylabel("Angle (°)", fontsize=8)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Spike-event window index", fontsize=9)
        plt.suptitle("All Gaits — Joint 1 GT vs Predicted  (summary)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        savefig(fig, out_dir, "gait_reconstruction_summary.png")

    rmse_mat = np.full((n_gaits, n_joints), np.nan)
    for g in range(n_gaits):
        if len(all_true_denorm[g]) == 0:
            continue
        for j in range(n_joints):
            rmse_mat[g, j] = np.sqrt(np.mean((all_pred_denorm[g][:, j] - all_true_denorm[g][:, j]) ** 2))
    fig, ax = plt.subplots(figsize=(max(6, n_joints * 0.9), n_gaits * 0.9 + 1.5))
    im = ax.imshow(rmse_mat, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="RMSE (°)")
    ax.set_xticks(range(n_joints))
    ax.set_xticklabels([f"J{j+1}" for j in range(n_joints)], fontsize=9)
    ax.set_yticks(range(n_gaits))
    ax.set_yticklabels(gait_names, fontsize=9)
    ax.set_title("Per-Joint RMSE Heatmap across Gaits", fontsize=11)
    vmax = np.nanmax(rmse_mat)
    for g in range(n_gaits):
        for j in range(n_joints):
            if not np.isnan(rmse_mat[g, j]):
                ax.text(j, g, f"{rmse_mat[g, j]:.1f}", ha="center", va="center", fontsize=8, color="white" if rmse_mat[g, j] > vmax * 0.6 else "black")
    plt.tight_layout()
    savefig(fig, out_dir, "rmse_heatmap.png")


def plot_gait_reconstruction(model, X, y, pure_mask, labels, device, out_dir, n_joints, tgt_range, gait_names, n_samples=300):
    out_dir = out_dir / "recons_training"
    model.eval()
    tgt_min, tgt_max = tgt_range
    scale = (tgt_max - tgt_min) / 2.0
    shift = (tgt_max + tgt_min) / 2.0
    n_gaits = len(gait_names)
    colors = ["#e63946", "#f4a261", "#2a9d8f", "#6a0572", "#8ecae6", "#ffb703"]
    TRUE_C = "#457b9d"

    def predict_batch(indices):
        X_t = torch.tensor(X[indices]).permute(1, 0, 2).to(device)
        lbl_t = torch.tensor(labels[indices], dtype=torch.long).to(device)
        with torch.no_grad():
            pred = model(X_t, lbl_t).cpu().numpy()
        return y[indices] * scale + shift, pred * scale + shift

    rmse_pure = np.full((n_gaits, n_joints), np.nan)
    rmse_trans = np.full((n_gaits, n_joints), np.nan)

    for g, name in enumerate(gait_names):
        for wtype, mask_cond, suffix, color, rmse_arr in [
            ("pure", pure_mask, "pure", TRUE_C, rmse_pure),
            ("trans", ~pure_mask, "trans", colors[g % len(colors)], rmse_trans),
        ]:
            idx = np.where((labels == g) & mask_cond)[0]
            if len(idx) == 0:
                continue
            idx = idx[:n_samples]
            true_arr, pred_arr = predict_batch(idx)

            cols = min(4, n_joints)
            rows = int(np.ceil(n_joints / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
            for j in range(n_joints):
                ax = axes[j // cols][j % cols]
                rmse = np.sqrt(np.mean((pred_arr[:, j] - true_arr[:, j]) ** 2))
                rmse_arr[g, j] = rmse
                ax.plot(true_arr[:, j], label="GT", color=TRUE_C, lw=1.8)
                ax.plot(pred_arr[:, j], label="Pred", color=color, lw=1.5, ls="--", alpha=0.9)
                err = np.abs(pred_arr[:, j] - true_arr[:, j])
                ax.fill_between(range(len(idx)), pred_arr[:, j] - err, pred_arr[:, j] + err, color=color, alpha=0.12)
                ax.set_title(f"J{j+1}  RMSE={rmse:.2f}°", fontsize=9)
                ax.set_xlabel("Window", fontsize=8)
                ax.set_ylabel("Angle (°)", fontsize=8)
                ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
            for j in range(n_joints, rows * cols):
                axes[j // cols][j % cols].set_visible(False)
            plt.suptitle(f"{name} — {wtype} ({len(idx)} samples)", fontsize=11, fontweight="bold")
            plt.tight_layout()
            savefig(fig, out_dir, f"recon_{name}_{suffix}.png")

    vmax = np.nanmax(np.stack([rmse_pure, rmse_trans]))
    fig, axes = plt.subplots(1, 2, figsize=(max(8, n_joints * 1.2), n_gaits + 2.0))
    for ax, rmse_mat, title in zip(axes, [rmse_pure, rmse_trans], ["RMSE — Pure windows (°)", "RMSE — Transition windows (°)"]):
        im = ax.imshow(rmse_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="RMSE (°)")
        ax.set_xticks(range(n_joints))
        ax.set_xticklabels([f"J{j+1}" for j in range(n_joints)], fontsize=9)
        ax.set_yticks(range(n_gaits))
        ax.set_yticklabels(gait_names, fontsize=9)
        ax.set_title(title, fontsize=10)
        for g in range(n_gaits):
            for j in range(n_joints):
                v = rmse_mat[g, j]
                if not np.isnan(v):
                    ax.text(j, g, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white" if v > vmax * 0.6 else "black")
    plt.suptitle("Per-Joint RMSE: Pure vs Transition Windows", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir, "rmse_heatmap.png")
