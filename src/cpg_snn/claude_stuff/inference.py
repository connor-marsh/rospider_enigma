"""
Deployment side for the bursting-LIF CPG + leg-grouped stateful SNN.

Drop-in replacement for the old CPGChunkStepper / ONNXGaitPredictor /
run_inference trio.  Three things changed and they all simplify the robot:

1.  No ODE solver.  `LIFCPGStepper.step()` is one integer timestep of pure
    numpy — a 4x4 matmul and two LIF updates.  There is no chunk-vs-batch
    integration mismatch left to worry about, so `chunk_size` is gone from
    the config; chunking here is purely a serial-write batching knob.

2.  No SpikeEventBuffer, no window, no phase computation on the critical
    path.  The SNN is stateful: it is called once per CPG timestep with the
    4 spike bits and its own previous state.  `gait_period` is never used
    by the controller — it appears in the config for diagnostics only.

3.  Inference runs every timestep, not every spike event.  With the
    supplied CPG that is 254 calls per gait cycle instead of ~40, so the
    per-call latency budget is tighter; servo writes are decimated
    separately via --servo_every.

Requires from the training run:
    cpg_lif_snn_step.onnx
    cpg_lif_snn_config.json
"""

import argparse
import json
import os
import threading as _threading
import time
from pathlib import Path

import numpy as np

# Reuse the exact CPG classes from the trainer so there is one definition.
from train import (
    LIFGeneralArray, BurstingLIF, LIFCPGStepper, CPG_W,
    GAIT_TABLES_ORIG, GAIT_NAMES, upsample_gait_tables,
    detect_burst_threshold,
)


# ═══════════════════════════════════════════════════════════════════
# 1.  Stateful ONNX predictor
# ═══════════════════════════════════════════════════════════════════

class StatefulSNNPredictor:
    """
    Wraps the single-timestep ONNX graph and owns the recurrent state.

    Call `step(spikes, gait_idx)` once per CPG timestep.  State is kept in
    numpy between calls, so a dropped or duplicated call desynchronises the
    controller from the CPG — keep the loop 1:1.
    """

    def __init__(self, onnx_path, n_legs, hidden_per_leg, intra_threads=2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so,
                                         providers=["CPUExecutionProvider"])
        self.state_names_in = ["mem1_in", "spk1_in", "mem2_in",
                               "spk2_in", "memo_in"]
        self.n_legs = n_legs
        self.hg     = hidden_per_leg
        self.reset()

    def reset(self):
        z = lambda: np.zeros((1, self.n_legs, self.hg), dtype=np.float32)
        self.state = [z() for _ in range(5)]

    def step(self, spikes, gait_idx):
        feed = {"spikes": spikes.reshape(1, -1).astype(np.float32),
                "gait":   np.array([gait_idx], dtype=np.int64)}
        for name, val in zip(self.state_names_in, self.state):
            feed[name] = val
        out = self.sess.run(None, feed)
        self.state = list(out[1:])
        return out[0][0]                      # (8,) normalised angles


# ═══════════════════════════════════════════════════════════════════
# 2.  Online phase tracker  (diagnostics / GT overlay ONLY)
# ═══════════════════════════════════════════════════════════════════

class OnlinePhase:
    """
    Streaming burst-onset detector on CPG neuron 0.

    Used only to index the gait lookup table for the ground-truth overlay
    in the recorded plots.  The controller does not consume it.
    """

    def __init__(self, burst_thresh, init_period=254.0):
        self.thr        = float(burst_thresh)
        self.period     = float(init_period)
        self.last_spike = None
        self.last_onset = None

    def update(self, t, spiked):
        if spiked:
            if self.last_spike is None or (t - self.last_spike) > self.thr:
                if self.last_onset is not None:
                    p = t - self.last_onset
                    if 0.5 * self.period < p < 2.0 * self.period:
                        self.period = 0.9 * self.period + 0.1 * p
                self.last_onset = t
            self.last_spike = t
        if self.last_onset is None:
            return 0.0
        return float(np.clip((t - self.last_onset) / self.period, 0.0, 0.999))


# ═══════════════════════════════════════════════════════════════════
# 3.  Shared state / serial worker  (unchanged in spirit)
# ═══════════════════════════════════════════════════════════════════

class SharedState:
    def __init__(self):
        self._lock = _threading.Lock()
        self._gait = 0
        self._cmd  = None

    def set_gait(self, g):
        with self._lock:
            self._gait = int(g)

    def get_gait(self):
        with self._lock:
            return self._gait

    def set_cmd(self, cmd):
        with self._lock:
            self._cmd = cmd

    def pop_cmd(self):
        with self._lock:
            c, self._cmd = self._cmd, None
        return c


def serial_worker(shared, n_joints, stop_event, port=None, baud=115200):
    """Replace the body with your own link; kept minimal on purpose."""
    ser = None
    if port is not None:
        import serial
        ser = serial.Serial(port, baud, timeout=0.05)
    while not stop_event.is_set():
        cmd = shared.pop_cmd()
        if cmd is None:
            time.sleep(0.001)
            continue
        if ser is not None:
            tag, angles, delay = cmd
            ser.write((tag + " " + " ".join(str(a) for a in angles) + "\n").encode())
    if ser is not None:
        ser.close()


# ═══════════════════════════════════════════════════════════════════
# 4.  Inference loop
# ═══════════════════════════════════════════════════════════════════

def run_inference(cfg, onnx_path, out_dir,
                  t_max=50_000,
                  gait_schedule=None,
                  record=True,
                  robot=True,
                  serial_port=None,
                  servo_every=5,
                  onnx_threads=2):
    """
    Parameters
    ----------
    cfg           : dict from cpg_lif_snn_config.json
    gait_schedule : list of (step, gait_idx)
    servo_every   : write to the servo bus every k timesteps (the SNN still
                    runs every timestep; this only throttles serial traffic)
    """
    global_min  = float(cfg["global_min"])
    global_max  = float(cfg["global_max"])
    n_gaits     = int(cfg["n_gaits"])
    n_legs      = int(cfg["n_legs"])
    n_joints    = int(cfg["n_joints"])
    hg          = int(cfg["hidden_per_leg"])
    gait_names  = cfg["gait_names"]
    leg_cols    = [tuple(c) for c in cfg["leg_cols"]]
    servo_base  = int(cfg["servo_base"])
    target_rows = int(cfg["target_rows"])
    phase_zero  = float(cfg.get("phase_zero", 0.0))
    period_hint = float(cfg.get("cpg_period_steps", 254.0))
    cpgp        = cfg["cpg"]

    scale = (global_max - global_min) / 2.0
    shift = (global_max + global_min) / 2.0

    print(f"  legs->servos: " + ", ".join(
        f"leg{l}=({leg_cols[l][0]+servo_base},{leg_cols[l][1]+servo_base})"
        for l in range(n_legs)))
    print(f"  leg->neuron routing per gait: {cfg['route_leg2neuron']}")

    gait_tables, _ = upsample_gait_tables(GAIT_TABLES_ORIG, gait_names,
                                          target_rows, verbose=False)

    shared     = SharedState()
    stop_event = _threading.Event()
    if robot:
        ser_thread = _threading.Thread(
            target=serial_worker,
            args=(shared, n_joints, stop_event),
            kwargs={"port": serial_port}, daemon=True, name="serial-worker")
        ser_thread.start()
        print("  Serial thread started.")
    else:
        ser_thread = None
        print("  --no_robot: serial thread suppressed.")

    predictor = StatefulSNNPredictor(onnx_path, n_legs, hg,
                                     intra_threads=onnx_threads)
    cpg = LIFCPGStepper(
        N=int(cfg.get("n_cpg_neurons", 4)),
        W=np.asarray(cpgp["W"]), i_app=cpgp["i_app"],
        vth_main=cpgp["vth_main"], du_main=cpgp["du_main"],
        dv_main=cpgp["dv_main"], refrac_main=cpgp["refrac_main"],
        vth_fb=cpgp["vth_fb"], du_fb=cpgp["du_fb"], dv_fb=cpgp["dv_fb"],
        refrac_fb=cpgp["refrac_fb"],
        from_fb_weight=cpgp["from_fb_weight"],
        to_fb_weight=cpgp["to_fb_weight"])

    # ── Boot: identical to training ──────────────────────────────
    warmup = int(cpgp.get("warmup", 2000))
    print(f"  Warming up CPG ({warmup} steps) ...")
    warm_spk = cpg.step_chunk(warmup)
    print("  CPG settled.")

    # burst threshold from the warm-up trace (diagnostics only)
    ts0 = np.where(warm_spk[:, 0] > 0)[0]
    burst_thr = detect_burst_threshold(ts0) if len(ts0) > 8 else 30.0
    phaser = OnlinePhase(burst_thr, init_period=period_hint)
    print(f"  Burst ISI threshold (diagnostics) = {burst_thr:.1f}")

    # The SNN state starts at zero; give it a few cycles of CPG input with
    # the initial gait before trusting the output.
    settle = int(3 * period_hint)
    g0 = gait_schedule[0][1] if gait_schedule else 0
    for _ in range(settle):
        predictor.step(cpg.step(), g0)
    shared.set_gait(g0)
    print(f"  SNN state settled over {settle} steps.\n")

    schedule = sorted(gait_schedule, key=lambda x: x[0]) if gait_schedule else []
    sched_ptr = 0

    rec = {k: [] for k in ("t", "spikes", "gait", "pred", "true", "phase")}
    latencies = []

    print(f"Running inference: {t_max} steps ...")
    try:
        for step in range(t_max):
            while sched_ptr < len(schedule) and step >= schedule[sched_ptr][0]:
                shared.set_gait(schedule[sched_ptr][1])
                print(f"  step {step:>7d}: gait -> "
                      f"{gait_names[schedule[sched_ptr][1]]}")
                sched_ptr += 1

            active_gait = shared.get_gait()

            t0 = time.perf_counter()
            spk = cpg.step()
            pred_norm = predictor.step(spk, active_gait)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            pred_deg = pred_norm * scale + shift

            if robot and (step % servo_every == 0):
                full_angles = [0] * 16
                for j in range(n_joints):
                    full_angles[j + servo_base] = int(np.clip(pred_deg[j], -124, 124))
                shared.set_cmd(["L", full_angles, 0.0])

            if record:
                ph = phaser.update(cpg.t, spk[0] > 0)
                row = int(((ph + phase_zero) % 1.0) * target_rows) % target_rows
                rec["t"].append(cpg.t)
                rec["spikes"].append(spk.copy())
                rec["gait"].append(active_gait)
                rec["phase"].append(ph)
                rec["pred"].append(pred_deg.copy())
                rec["true"].append(gait_tables[active_gait][row].astype(np.float32))
    finally:
        stop_event.set()
        if ser_thread is not None:
            ser_thread.join(timeout=2.0)
            print("  Serial thread stopped.")

    lat = np.array(latencies)
    print(f"\nCPG step + SNN step latency ({len(lat)} steps):")
    for name, v in (("mean", np.mean(lat)), ("median", np.median(lat)),
                    ("p95", np.percentile(lat, 95)),
                    ("p99", np.percentile(lat, 99)), ("max", np.max(lat))):
        print(f"  {name:<7}: {v:.3f} ms")
    print(f"  sustainable gait rate: "
          f"{1000.0 / (np.mean(lat) * period_hint):.2f} cycles/s")

    if not record:
        return None
    return {k: np.array(v) for k, v in rec.items()} | {
        "latencies": lat, "gait_names": gait_names,
        "leg_cols": leg_cols, "n_joints": n_joints}


# ═══════════════════════════════════════════════════════════════════
# 5.  Plots
# ═══════════════════════════════════════════════════════════════════

def plot_run(res, out_dir, n_show=3000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sl = slice(0, min(n_show, len(res["t"])))
    t  = res["t"][sl]
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"]

    fig, axes = plt.subplots(6, 1, figsize=(15, 14), sharex=True,
                             height_ratios=[1.2, 0.6, 1, 1, 1, 1])
    ax = axes[0]
    for i in range(res["spikes"].shape[1]):
        idx = np.where(res["spikes"][sl, i] > 0)[0]
        ax.scatter(t[idx], np.full_like(idx, i), marker="|", s=90,
                   color=colors[i % 4])
    ax.set_yticks(range(4)); ax.set_yticklabels([f"CPG {i}" for i in range(4)])
    ax.set_title("CPG raster"); ax.grid(axis="x", alpha=0.2)

    axes[1].step(t, res["gait"][sl], where="post", color="#6a0572", lw=1.6)
    axes[1].set_yticks(range(len(res["gait_names"])))
    axes[1].set_yticklabels(res["gait_names"])
    axes[1].set_ylabel("gait"); axes[1].grid(alpha=0.25)

    for l, (a, b) in enumerate(res["leg_cols"]):
        ax = axes[2 + l]
        ax.plot(t, res["true"][sl, a], color="#457b9d", lw=1.6, label=f"GT c{a}")
        ax.plot(t, res["pred"][sl, a], color="#e63946", lw=1.2, ls="--",
                label=f"pred c{a}")
        on = np.where(res["spikes"][sl, l] > 0)[0]
        ax.set_ylabel(f"leg{l} (deg)"); ax.legend(fontsize=7); ax.grid(alpha=0.25)
    axes[-1].set_xlabel("CPG timestep")
    plt.suptitle("Deployment trace — stateful SNN driven by bursting-LIF CPG",
                 fontweight="bold")
    plt.tight_layout()
    p = out_dir / "deploy_trace.png"
    plt.savefig(p, dpi=140); plt.close()
    print(f"  [saved] {p}")

    v = np.ones(len(res["t"]), dtype=bool)
    rmse = np.sqrt(np.mean((res["pred"][v] - res["true"][v]) ** 2, axis=0))
    print("  per-column RMSE (deg): " +
          "  ".join(f"c{j}={rmse[j]:.2f}" for j in range(len(rmse))))


# ═══════════════════════════════════════════════════════════════════
# 6.  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str,
                    default="output_cpg_lif_leggrouped")
    ap.add_argument("--out_dir",   type=str, default="inference_out")
    ap.add_argument("--t_max",     type=int, default=20_000)
    ap.add_argument("--no_robot",  action="store_true")
    ap.add_argument("--serial_port", type=str, default=None)
    ap.add_argument("--servo_every", type=int, default=5)
    ap.add_argument("--onnx_threads", type=int, default=2)
    args = ap.parse_args()

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = Path(this_file_dir + "/" + args.out_dir)

    md  = Path(this_file_dir + "/" + args.model_dir)
    cfg = json.loads((md / "cpg_lif_snn_config.json").read_text())
    onnx_path = md / "cpg_lif_snn_step.onnx"

    # one gait per quarter of the run
    q = args.t_max // 4
    schedule = [(0, 0), (q, 1), (2 * q, 2), (3 * q, 3)]

    res = run_inference(cfg, onnx_path, Path(out_dir),
                        t_max=args.t_max, gait_schedule=schedule,
                        record=True, robot=not args.no_robot,
                        serial_port=args.serial_port,
                        servo_every=args.servo_every,
                        onnx_threads=args.onnx_threads)
    if res is not None:
        plot_run(res, Path(out_dir))


if __name__ == "__main__":
    main()