"""
Multi-gait CPG → SNN inference  (ONNX Runtime, Raspberry Pi)
=============================================================
Loads cpg_snn.onnx and cpg_snn_config.json produced by the training
script.  No PyTorch required at runtime.

Dependencies:
    pip install onnxruntime numpy scipy matplotlib

Usage:
    python cpg_snn_inference.py --out_dir outputs --t_max 50000

    # Offline testing (no robot connected):
    python cpg_snn_inference.py --out_dir outputs --t_max 50000 --no_robot

Gait switching:
    Call shared.set_gait(idx) at any time during the inference loop.
    idx: 0=wkF  1=bk  2=wkL  3=wkR
    The per-event gait flag in the sliding window will transition
    naturally as new events push old ones out of the buffer.

Changes vs original
-------------------
1. OnlineBurstPeriodEstimator removed entirely.  Training and inference
   use the same BLIF_CPG stepper with the same initial conditions, producing
   bit-identical spike times.  The gait_period from cpg_snn_config.json
   is used directly for all phase computations — no estimation, no EMA lag.
   The estimator caused sin/cos phase errors up to ±1.07 due to modulo
   wraparound at a slightly wrong period boundary.

2. BLIF_CPG is run one step at a time with the same warm-up schedule
   from config, so training and inference always use the same stepper
   behaviour.

3. --no_robot flag suppresses autoConnect() and the serial thread so
   the script can be run offline for testing / analysis.
"""

import argparse
import json
import os
import time
from collections import deque
import numpy as np
import onnxruntime as ort
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# from servo_controller_msgs.msg import ServoPosition, ServosPosition # type: ignore
import rclpy
from rclpy.node import Node

from cpg_utils import BLIF_CPG
from plotting_utils import (
    plot_cpg_vm,
    plot_inference_summary as plot_gait_reconstruction,
    plot_latency,
    plot_spike_event_overview,
    plot_spike_events,
)


# ═══════════════════════════════════════════════════════════════════
# 1.  Config loader
# ═══════════════════════════════════════════════════════════════════

def load_config(out_dir):
    """
    Load cpg_snn_config.json saved by the training script.

    Returns a dict with keys:
        gait_period, burst_threshold, global_min, global_max,
        seq_len, n_gaits, n_joints, gait_names, cpg_start_time
    """
    cfg_path = Path(out_dir) / "cpg_snn_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg_path}\n"
            "Run the training script first to generate cpg_snn_config.json.")
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"  Loaded config from {cfg_path}")
    print(f"    gait_period     = {cfg['gait_period']:.1f} steps")
    print(f"    burst_threshold = {cfg['burst_threshold']:.1f} steps")
    print(f"    global_min/max  = {cfg['global_min']:.1f} / {cfg['global_max']:.1f}")
    print(f"    seq_len         = {cfg['seq_len']}")
    print(f"    cpg_start_time  = {cfg.get('cpg_start_time', 5000)}")
    print(f"    target_rows     = {cfg.get('target_rows', 'not set (no upsampling)')}")
    print(f"    n_in            = {cfg.get('n_in', 'not set')}")
    return cfg


# ═══════════════════════════════════════════════════════════════════
# 2.  Gait tables  (must match training exactly)
# ═══════════════════════════════════════════════════════════════════

# ── Runtime upsampling ───────────────────────────────────────────
# Applied after config is loaded; target_rows comes from config.
# Defined here so it can be called in run_inference.

def upsample_gait_tables(gait_tables, gait_names, target_rows):
    """Cubic interpolation to equalise row counts — must match training."""
    from scipy.interpolate import interp1d
    upsampled = []
    for gt, name in zip(gait_tables, gait_names):
        n_orig = gt.shape[0]
        if n_orig == target_rows:
            upsampled.append(gt.copy())
        else:
            x_orig = np.linspace(0.0, 1.0, n_orig)
            x_new  = np.linspace(0.0, 1.0, target_rows)
            interp = interp1d(x_orig, gt, axis=0, kind='cubic',
                              fill_value='extrapolate')
            upsampled.append(interp(x_new).astype(np.float32))
            print(f"  {name}: {n_orig} → {target_rows} rows (cubic upsampled)")
    return upsampled

CPG_COLORS  = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"]
PRED_COLORS = ["#e63946", "#f4a261", "#2a9d8f", "#6a0572"]
TRUE_COLOR  = "#457b9d"


# Shared BLIF CPG helpers now live in cpg_utils.py.

# ═══════════════════════════════════════════════════════════════════
# 4.  ONNX predictor
# ═══════════════════════════════════════════════════════════════════

class ONNXGaitPredictor:
    """
    Wraps the ONNX session.
    Single input: spike_window (seq_len, 1, N+4+n_gaits) — gait flag
    is already baked into the per-event feature vector.
    The gait_idx argument is kept for API compatibility but ignored.
    """

    def __init__(self, onnx_path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        opts.intra_op_num_threads = 4
        self.sess        = ort.InferenceSession(
            str(onnx_path), sess_options=opts,
            providers=["CPUExecutionProvider"])
        inp              = self.sess.get_inputs()[0]
        self.input_name  = inp.name
        self.output_name = self.sess.get_outputs()[0].name
        self.seq_len     = inp.shape[0]
        self.n_in        = inp.shape[2]
        self._x = np.zeros((self.seq_len, 1, self.n_in), dtype=np.float32)
        print(f"  ONNX loaded  |  seq_len={self.seq_len}  n_in={self.n_in}")

    def predict(self, window, gait_idx=None):
        """
        window   : (seq_len, n_in)  float32  — includes gait one-hot flag
        gait_idx : ignored (flag already in window)
        Returns  : (n_joints,)      float32  normalised angles
        """
        self._x[:, 0, :] = window
        return self.sess.run(
            [self.output_name],
            {self.input_name: self._x})[0].squeeze(0)



# ═══════════════════════════════════════════════════════════════════
# 8.  Main inference loop
# ═══════════════════════════════════════════════════════════════════


def run_inference(cfg, onnx_path, out_dir, args,
                  gait_schedule=None,
                  record=True):
    """
    Two-thread inference loop.

    Parameters
    ----------
    cfg           : dict from cpg_snn_config.json
    onnx_path     : Path to cpg_snn.onnx
    out_dir       : Path for output plots
    args          : argparse.Namespace with t_max, robot_mode, run_on_spikes_only
    gait_schedule : list of (step, gait_idx) for scripted switching
    record        : store data for plots if True
    """
    # ── Read all stepper params from config ──────────────────────
    gait_period     = float(cfg["gait_period"])
    burst_threshold = float(cfg["burst_threshold"])
    global_min      = float(cfg["global_min"])
    global_max      = float(cfg["global_max"])
    seq_len         = int(cfg["seq_len"])
    n_neurons         = int(cfg["n_neurons"])
    n_gaits         = int(cfg["n_gaits"])
    n_joints        = int(cfg["n_joints"])
    n_in            = int(cfg["n_in"])
    gait_names      = cfg["gait_names"]
    cpg_start_time  = int(cfg.get("cpg_start_time", 5000))
    target_rows     = int(cfg.get("target_rows",    54))
    scale           = (global_max - global_min) / 2.0
    shift           = (global_max + global_min) / 2.0

    # Read args
    t_max      = int(args.t_max)
    robot_mode = str(args.robot_mode)
    run_on_spikes_only = bool(args.run_on_spikes_only)

    print(f"  Stepper params from config: cpg_start_time={cpg_start_time}")

    this_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Upsample gait tables to match training target resolution
    gait_names = [
        "tripod", "tripod_huge", "tripod_right", "tripod_huge_right",
        "ripple", "ripple_tiny", "ripple_right", "ripple_tiny_right",
        "tripod_backwards", "tripod_huge_backwards", "tripod_left", "tripod_huge_left",
        "ripple_backwards", "ripple_tiny_backwards", "ripple_left", "ripple_tiny_left",
    ]
    gait_names = gait_names[0:8:2]
    
    gait_names=["bittle_wkF", "bittle_bk", "bittle_wkL", "bittle_wkR"]

    GAIT_TABLES_ORIG = []
    for name in gait_names:
        gait_table = np.loadtxt(f"{this_file_dir}/gaits/{name}.csv",
                                delimiter=",", dtype=np.float32)
        GAIT_TABLES_ORIG.append(gait_table)


    GAIT_TABLES = upsample_gait_tables(
        GAIT_TABLES_ORIG, gait_names, target_rows)


    # ── Inference components ─────────────────────────────────────
    predictor = ONNXGaitPredictor(onnx_path)
    cpg = BLIF_CPG(N=n_neurons, t_max=t_max)

    if gait_schedule:
        schedule = sorted(gait_schedule, key=lambda x: x[0])
    else:
        schedule = []

    # ── Boot sequence: identical to training ─────────────────────
    # Training does:
    #   1. Warm up BLIF_CPG for cpg_start_time steps
    #   2. Collect spikes for tmax - cpg_start_time steps
    #   3. Run estimate_gait_period on those spikes -> gait_period
    #   4. Use that fixed gait_period for ALL phase computations
    #
    # Inference does EXACTLY the same thing.  The config stores the
    # gait_period computed during training, so we can simply read it
    # without re-estimating.  BLIF_CPG produces bit-identical
    # spike times (same ICs, same integrator), so
    # t % gait_period gives identical phase values to training.
    #
    # There is NO online period estimator. There never was a reason for
    # one once both training and inference used the same integrator.

    print(f"Warming up CPG ({cpg_start_time} steps) ...")
    for _ in range(cpg_start_time):
        cpg.step()
    print(f"  CPG settled.  Using fixed gait_period = {gait_period:.1f} steps\n")

    feature_history = deque(maxlen=seq_len)

    # ── Recording buffers ─────────────────────────────────────────
    rec_t, rec_neuron  = [], []
    rec_phase_deg      = []
    rec_gait_idx       = []
    rec_pred, rec_true = [], []
    all_vm             = []
    latencies          = []

    # ── Main inference loop ───────────────────────────────────────
    print(f"Running inference: {t_max} steps")
    sched_ptr  = 0
    steps_done = 0
    active_gait = 0
    print(schedule)

    for _ in range(cpg_start_time, t_max):
        
        while (sched_ptr < len(schedule)
                and steps_done >= schedule[sched_ptr][0]):
            active_gait = schedule[sched_ptr][1]
            print(f"  step {steps_done:>6d}: gait → "
                    f"{gait_names[active_gait]}")
            sched_ptr += 1

        spikes, _, t_now = cpg.step()
        steps_done += 1


        abs_phase_rad = float(
            2.0 * np.pi * (t_now % gait_period) / gait_period)

        # If there are no spikes and we are running on spikes only, skip this timestep.
        if run_on_spikes_only and not np.any(spikes):
            continue

        # Create network input vector for this timestep.
        feat = np.zeros(n_in, dtype=np.float32)

        for neuron_id in range(n_neurons):
            if spikes[neuron_id]:
                feat[neuron_id] = 1.0

        # This if statement allows for both phase and non-phase included versions of the model
        if n_in - n_neurons - n_gaits == 2:
            print("phase")
            feat[n_neurons] = float(np.sin(abs_phase_rad))
            feat[n_neurons+1] = float(np.cos(abs_phase_rad))
            feat[n_neurons+2 + active_gait] = 1.0
        else:
            feat[n_neurons + active_gait] = 1.0

        feature_history.append(feat)

        if len(feature_history) < seq_len:
            continue

        window = np.stack(list(feature_history), axis=0)
        t0 = time.perf_counter()
        pred_norm = predictor.predict(window, active_gait)
        pred = pred_norm * scale + shift
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        if robot_mode != "no_robot":
            servo_id = [5, 3, 1, 11, 9, 7, 17, 15, 13, 18, 16, 14, 12, 10, 8, 6, 4, 2]
            msg = ServosPosition()
            msg.duration = 0.02
            position_msgs = []
            for i in range(len(pred)):
                position = ServoPosition()
                position.id = servo_id[i]
                position.position = float(pred[i])
                position_msgs.append(position)
            msg.position = position_msgs
            msg.position_unit = "pulse"
            publishers.publish(msg)

        if record:
            gait_table = GAIT_TABLES[active_gait]
            row_idx = (int(abs_phase_rad / (2.0 * np.pi) * gait_table.shape[0]) % gait_table.shape[0])
            rec_t.append(t_now)
            rec_neuron.append(neuron_id)
            rec_phase_deg.append(float(np.degrees(abs_phase_rad)))
            rec_gait_idx.append(active_gait)
            rec_pred.append(pred.copy())
            rec_true.append(gait_table[row_idx].astype(np.float32))

        time.sleep(0.03)

    if latencies:
        lat = np.array(latencies)
        print(f"\nONNX inference latency ({len(lat)} events):")
        print(f"  mean   : {np.mean(lat):.3f} ms")
        print(f"  median : {np.median(lat):.3f} ms")
        print(f"  p95    : {np.percentile(lat, 95):.3f} ms")
        print(f"  max    : {np.max(lat):.3f} ms")
        print(f"\n  Fixed gait_period : {gait_period:.1f} steps"
              f"  (from training config)")

    if not record:
        return None

    return {
        "rec_t":         np.array(rec_t,        dtype=np.float32),
        "rec_neuron":    np.array(rec_neuron,    dtype=np.int32),
        "rec_phase_deg": np.array(rec_phase_deg, dtype=np.float32),
        "rec_gait_idx":  np.array(rec_gait_idx,  dtype=np.int32),
        "rec_pred":      np.array(rec_pred,       dtype=np.float32),
        "rec_true":      np.array(rec_true,       dtype=np.float32),
        "all_vm":        np.array(all_vm,         dtype=np.float32),
        "latencies":     np.array(latencies,      dtype=np.float32),
        "period_final":  gait_period,
        "n_joints":      n_joints,
        "gait_names":    gait_names,
    }


# ═══════════════════════════════════════════════════════════════════
# 9.  Entry point
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="CPG-SNN multi-gait inference (ONNX Runtime)")
    parser.add_argument("--out_dir",   type=str,  default="outputs",
                        help="Directory containing cpg_snn.onnx and "
                             "cpg_snn_config.json")
    parser.add_argument("--t_max",    type=int,  default=5000,
                        help="Inference steps after warm-up")
    parser.add_argument("--robot_mode", type=str, default="no_robot", help="Options: no_robot, bittle, bittle_sim, unitree_sim, rospider")
    parser.add_argument("--run_on_spikes_only", type=bool, default=True, help="If True, only predict outputs on a timestep with a spike")
    args = parser.parse_args()

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = Path(this_file_dir + "/" + args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "cpg_snn.onnx"

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}\n"
            "Run train_snn.py (training) first.")

    print("Loading config ...")
    cfg = load_config(out_dir)

    # ── Robot connection (skipped in --no_robot mode) ────────────
    global command_topics, JOINT_DIRECTIONS, JOINT_OFFSETS, publishers, PUBLISH_RATE, rate
    if args.robot_mode=="bittle_sim" or args.robot_mode=="unitree_sim":
        from std_msgs.msg import Float64
        
        PUBLISH_RATE = 10.0 # Hz (Control frequency)
        rclpy.init_node('gait_decoder_commander_node', anonymous=True)
        rate = rclpy.Rate(PUBLISH_RATE)
        if args.robot_mode=="bittle_sim":
            command_topics = [
                'left_front_shoulder_joint_position_controller/command', # 0: Front Right Shoulder
                'right_front_shoulder_joint_position_controller/command', # 1: Front Right Knee
                'right_back_shoulder_joint_position_controller/command', # 2: Front Left Shoulder
                'left_back_shoulder_joint_position_controller/command', # 3: Front Left Knee
                'left_front_knee_joint_position_controller/command', # 4: Back Right Shoulder
                'right_front_knee_joint_position_controller/command', # 5: Back Right Knee
                'right_back_knee_joint_position_controller/command', # 6: Back Left Shoulder
                'left_back_knee_joint_position_controller/command', # 7: Back Left Knee
            ]
            JOINT_OFFSETS = [
                0,0,0,0,0,0,0,0
            ]
            JOINT_DIRECTIONS = [
                1, -1, -1, 1, 1, -1, -1, 1
            ]
            
        elif args.robot_mode=="unitree_sim":
            command_topics = [
                    "/a1_gazebo/FL_thigh_joint/command",
                    "/a1_gazebo/FR_thigh_joint/command",
                    "/a1_gazebo/RR_thigh_joint/command",
                    "/a1_gazebo/RL_thigh_joint/command",
                    "/a1_gazebo/FL_calf_joint/command",
                    "/a1_gazebo/FR_calf_joint/command",
                    "/a1_gazebo/RR_calf_joint/command",
                    "/a1_gazebo/RL_calf_joint/command",
                    "/a1_gazebo/FL_hip_joint/command",
                    "/a1_gazebo/FR_hip_joint/command",
                    "/a1_gazebo/RR_hip_joint/command",
                    "/a1_gazebo/RL_hip_joint/command"
                    ]
            JOINT_OFFSETS = [
                0,0,0,0,-np.pi*0.5,-np.pi*0.5,-np.pi*0.5,-np.pi*0.5
            ]
            JOINT_DIRECTIONS = [
                1, 1, 1, 1, 1, 1, 1, 1
            ]
        
        publishers = {}
        for topic_name in command_topics:
            publishers[topic_name] = rclpy.Publisher(topic_name, Float64, queue_size=1)

    elif args.robot_mode == "rospider":
        PUBLISH_RATE = 10.0 # Hz (Control frequency)
        rclpy.init()
        node = Node("run_inference")
        rate = node.create_rate(PUBLISH_RATE)
        publishers = node.create_publisher(ServosPosition, 'servo_controller', 1)
        

        

    else:
        print("Skipping Robot Connection")

    # ── Scripted gait schedule ───────────────────────────────────
    # Uncomment and edit to test scripted gait transitions:
    t = args.t_max
    n_gaits = min(int(cfg["n_gaits"]), 8)
    gait_times = [i*t//(n_gaits+1) for i in range(n_gaits+1)]
    gait_schedule = [(gait_times[i], i%(n_gaits)) for i in range(n_gaits+1)]

    data = run_inference(
        cfg, onnx_path, out_dir, args,
        gait_schedule=(None if args.robot_mode=="bittle" else gait_schedule),
        record=True)

    if data is None or len(data["rec_t"]) == 0:
        print("No spike events recorded — check CPG parameters.")
        return

    E = len(data["rec_t"])
    print(f"\nTotal spike events recorded : {E}")
    for g, name in enumerate(data["gait_names"]):
        n = (data["rec_gait_idx"] == g).sum()
        print(f"  {name}: {n} events")

    print("\nGenerating plots ...")
    # plot_cpg_vm({"t": np.arange(len(data["all_vm"])), "y": data["all_vm"].T}, out_dir, n_show=30_000)
    plot_spike_events(np.asarray(data["rec_t"]), np.asarray(data["rec_neuron"]),
                      data["period_final"], out_dir)
    plot_gait_reconstruction(
        np.asarray(data["rec_pred"]), np.asarray(data["rec_true"]), np.asarray(data["rec_gait_idx"]),
        n_joints=data["n_joints"],
        gait_names=data["gait_names"],
        out_dir=out_dir,
        n_samples_per_gait=500)
    plot_spike_event_overview(
        np.asarray(data["rec_t"]), np.asarray(data["rec_neuron"]), np.asarray(data["rec_phase_deg"]),
        np.asarray(data["rec_pred"]), np.asarray(data["rec_true"]), np.asarray(data["rec_gait_idx"]),
        n_joints=data["n_joints"],
        gait_names=data["gait_names"],
        out_dir=out_dir)
    plot_latency(data["latencies"], out_dir)

    print(f"\nDone — outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
