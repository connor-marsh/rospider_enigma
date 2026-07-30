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
   use the same CPGChunkStepper with the same initial conditions, producing
   bit-identical spike times.  The gait_period from cpg_snn_config.json
   is used directly for all phase computations — no estimation, no EMA lag.
   The estimator caused sin/cos phase errors up to ±1.07 due to modulo
   wraparound at a slightly wrong period boundary.

2. CPGChunkStepper is instantiated from config values
   (chunk_size, spike_thresh, cpg_start_time) instead of hardcoded
   constants, so training and inference always use identical stepper
   parameters.

3. --no_robot flag suppresses autoConnect() and the serial thread so
   the script can be run offline for testing / analysis.
"""

import argparse
import json
import os
import time
import numpy as np
import onnxruntime as ort
from scipy.integrate import solve_ivp
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from servo_controller_msgs.msg import ServoPosition, ServosPosition # type: ignore
import rclpy
from rclpy.node import Node

from cpg_utils import BLIF_CPG, make_network
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
        seq_len, n_gaits, n_joints, gait_names,
        chunk_size, spike_thresh, cpg_start_time
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
    print(f"    chunk_size      = {cfg.get('chunk_size', 50)}")
    print(f"    spike_thresh    = {cfg.get('spike_thresh', -2.0)}")
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

class CPGChunkStepper:
    """
    Chunk-based BDF integrator — identical to the one used in training.
    Parameters are loaded from cpg_snn_config.json so training and
    inference always use the same chunk_size and spike_thresh.
    """

    def __init__(self, N=4, dt=1.0, chunk_size=50, spike_thresh=-2.0):
        self.N            = N
        self.dt           = dt
        self.chunk_size   = chunk_size
        self.spike_thresh = spike_thresh
        self.t            = 0.0
        alpha             = [-2.0,  2.0, -1.5,  1.5]
        delta             = [ 0.0,  0.0, -1.5, -1.5]
        g_inh, Iapp       = -0.3, -1.6
        self.network      = make_network(N, alpha, delta, g_inh, Iapp)
        self.S            = np.zeros(N * 4)
        self.S[::4]       = -1.0
        self.prev_vm      = self.S[::4].copy()

    def step_chunk(self):
        t_start = self.t
        t_end   = self.t + self.chunk_size * self.dt
        t_eval  = np.arange(t_start + self.dt,
                             t_end   + self.dt * 0.5, self.dt)
        sol = solve_ivp(self.network,
                        (t_start, t_end), self.S,
                        method="BDF", t_eval=t_eval,
                        dense_output=False)
        vm_all = sol.y[::4, :]   # (N, chunk_size)

        spike_events = []
        prev = self.prev_vm.copy()
        for k in range(vm_all.shape[1]):
            curr    = vm_all[:, k]
            crossed = (curr > self.spike_thresh) & (prev <= self.spike_thresh)
            for neuron_id in np.where(crossed)[0]:
                spike_events.append((float(sol.t[k]), int(neuron_id)))
            prev = curr

        self.S       = sol.y[:, -1]
        self.t       = float(sol.t[-1])
        self.prev_vm = vm_all[:, -1].copy()
        return spike_events, vm_all[:, -1].astype(np.float32), vm_all.T


# ═══════════════════════════════════════════════════════════════════
# 4.  Spike-event sliding window buffer
# ═══════════════════════════════════════════════════════════════════

class SpikeWindowBuffer:
    """Simple rolling buffer of spike-event feature vectors."""

    def __init__(self, seq_len, N=4, n_gaits=4):
        self.seq_len = seq_len
        self.N = N
        self.n_gaits = n_gaits
        self.n_in = N + 2 + n_gaits
        self.buf = np.zeros((seq_len, self.n_in), dtype=np.float32)
        self.count = 0

    def push(self, neuron_id, abs_phase_rad, gait_idx):
        feat = np.zeros(self.n_in, dtype=np.float32)
        feat[neuron_id] = 1.0
        feat[self.N] = float(np.sin(abs_phase_rad))
        feat[self.N + 1] = float(np.cos(abs_phase_rad))
        feat[self.N + 2 + gait_idx] = 1.0

        if self.count < self.seq_len:
            self.buf[self.count] = feat
            self.count += 1
        else:
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = feat

    def get(self):
        return self.buf.copy()

    @property
    def is_primed(self):
        return self.count >= self.seq_len


# ═══════════════════════════════════════════════════════════════════
# 5.  ONNX predictor
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
# 6.  Shared state + serial thread
# ═══════════════════════════════════════════════════════════════════

# The old threading-based serial bridge is no longer needed for the
# simplified inference loop. Gait updates are handled directly in the
# main loop below.

        


# ═══════════════════════════════════════════════════════════════════
# 7.  Visualisation
# ═══════════════════════════════════════════════════════════════════

# Shared plotting helpers now live in plotting_utils.py.

# ═══════════════════════════════════════════════════════════════════
# 8.  Main inference loop
# ═══════════════════════════════════════════════════════════════════

test_gait = np.array([[424.0, 316.0, 682.0, 580.0, 320.0, 649.0, 492.0, 329.0, 616.0, 393.0, 682.0, 306.0, 553.0, 680.0, 341.0, 476.0, 661.0, 409.0],
[421.0, 316.0, 684.0, 578.0, 320.0, 650.0, 491.0, 330.0, 613.0, 395.0, 682.0, 307.0, 556.0, 680.0, 342.0, 478.0, 661.0, 407.0],
[418.0, 316.0, 685.0, 576.0, 320.0, 651.0, 489.0, 331.0, 611.0, 398.0, 682.0, 308.0, 558.0, 680.0, 343.0, 479.0, 662.0, 405.0],
[416.0, 316.0, 686.0, 573.0, 320.0, 652.0, 488.0, 332.0, 609.0, 402.0, 683.0, 309.0, 560.0, 680.0, 343.0, 480.0, 663.0, 402.0],
[414.0, 316.0, 686.0, 570.0, 319.0, 653.0, 487.0, 332.0, 608.0, 405.0, 683.0, 310.0, 562.0, 680.0, 344.0, 482.0, 664.0, 399.0],
[413.0, 316.0, 686.0, 567.0, 319.0, 654.0, 486.0, 332.0, 607.0, 409.0, 683.0, 311.0, 563.0, 680.0, 344.0, 484.0, 665.0, 396.0],
[413.0, 316.0, 687.0, 563.0, 319.0, 655.0, 486.0, 332.0, 607.0, 413.0, 683.0, 312.0, 563.0, 680.0, 344.0, 486.0, 667.0, 392.0],
[414.0, 305.0, 693.0, 560.0, 319.0, 656.0, 487.0, 324.0, 612.0, 417.0, 683.0, 314.0, 562.0, 690.0, 338.0, 488.0, 668.0, 389.0],
[416.0, 293.0, 698.0, 556.0, 319.0, 657.0, 488.0, 314.0, 619.0, 420.0, 683.0, 315.0, 561.0, 700.0, 332.0, 490.0, 669.0, 386.0],
[419.0, 282.0, 703.0, 553.0, 319.0, 658.0, 489.0, 305.0, 626.0, 424.0, 683.0, 317.0, 558.0, 710.0, 326.0, 492.0, 670.0, 383.0],
[423.0, 273.0, 706.0, 550.0, 319.0, 658.0, 492.0, 296.0, 633.0, 427.0, 683.0, 318.0, 554.0, 719.0, 321.0, 494.0, 670.0, 381.0],
[428.0, 264.0, 708.0, 547.0, 318.0, 659.0, 495.0, 288.0, 641.0, 429.0, 683.0, 319.0, 548.0, 727.0, 316.0, 496.0, 671.0, 379.0],
[434.0, 258.0, 708.0, 545.0, 318.0, 659.0, 499.0, 280.0, 649.0, 431.0, 683.0, 320.0, 542.0, 734.0, 312.0, 497.0, 672.0, 377.0],
[440.0, 254.0, 707.0, 544.0, 318.0, 659.0, 503.0, 274.0, 657.0, 433.0, 683.0, 321.0, 536.0, 740.0, 308.0, 498.0, 672.0, 376.0],
[447.0, 252.0, 704.0, 543.0, 318.0, 660.0, 508.0, 268.0, 664.0, 434.0, 683.0, 321.0, 529.0, 743.0, 306.0, 499.0, 672.0, 375.0],
[453.0, 253.0, 700.0, 542.0, 318.0, 660.0, 513.0, 265.0, 670.0, 434.0, 683.0, 321.0, 521.0, 744.0, 304.0, 499.0, 672.0, 375.0],
[460.0, 256.0, 695.0, 542.0, 318.0, 660.0, 518.0, 263.0, 675.0, 434.0, 683.0, 321.0, 514.0, 744.0, 304.0, 499.0, 672.0, 374.0],
[465.0, 260.0, 689.0, 541.0, 318.0, 660.0, 523.0, 264.0, 679.0, 435.0, 683.0, 322.0, 506.0, 741.0, 305.0, 500.0, 673.0, 374.0],
[471.0, 267.0, 682.0, 540.0, 318.0, 660.0, 528.0, 267.0, 682.0, 437.0, 683.0, 322.0, 500.0, 736.0, 307.0, 501.0, 673.0, 372.0], 
[475.0, 275.0, 675.0, 538.0, 318.0, 661.0, 533.0, 272.0, 683.0, 439.0, 683.0, 323.0, 493.0, 729.0, 310.0, 502.0, 673.0, 371.0],
[479.0, 283.0, 667.0, 535.0, 318.0, 661.0, 537.0, 278.0, 683.0, 441.0, 682.0, 325.0, 488.0, 721.0, 315.0, 504.0, 674.0, 369.0],
[482.0, 293.0, 660.0, 532.0, 318.0, 661.0, 541.0, 287.0, 681.0, 444.0, 682.0, 326.0, 484.0, 712.0, 319.0, 506.0, 675.0, 366.0],
[484.0, 303.0, 653.0, 528.0, 318.0, 662.0, 543.0, 296.0, 677.0, 447.0, 682.0, 328.0, 480.0, 702.0, 325.0, 508.0, 675.0, 364.0],
[486.0, 312.0, 646.0, 525.0, 318.0, 662.0, 545.0, 307.0, 673.0, 450.0, 682.0, 330.0, 479.0, 692.0, 331.0, 510.0, 676.0, 361.0],
[486.0, 322.0, 640.0, 521.0, 318.0, 663.0, 546.0, 317.0, 667.0, 453.0, 682.0, 332.0, 478.0, 681.0, 336.0, 513.0, 677.0, 359.0],
[486.0, 322.0, 641.0, 517.0, 318.0, 663.0, 545.0, 317.0, 667.0, 456.0, 682.0, 334.0, 478.0, 681.0, 336.0, 515.0, 677.0, 356.0], 
[485.0, 322.0, 641.0, 514.0, 318.0, 663.0, 545.0, 317.0, 666.0, 459.0, 681.0, 336.0, 479.0, 681.0, 336.0, 518.0, 678.0, 354.0],
[484.0, 322.0, 642.0, 510.0, 318.0, 663.0, 543.0, 317.0, 666.0, 462.0, 681.0, 338.0, 481.0, 681.0, 336.0, 520.0, 678.0, 351.0], 
[483.0, 321.0, 644.0, 507.0, 318.0, 663.0, 541.0, 318.0, 664.0, 465.0, 681.0, 340.0, 483.0, 681.0, 336.0, 523.0, 679.0, 349.0],
[481.0, 321.0, 646.0, 505.0, 318.0, 663.0, 539.0, 318.0, 663.0, 467.0, 680.0, 341.0, 486.0, 681.0, 336.0, 525.0, 679.0, 348.0], 
[479.0, 321.0, 648.0, 502.0, 318.0, 663.0, 537.0, 318.0, 661.0, 468.0, 680.0, 343.0, 489.0, 681.0, 336.0, 526.0, 680.0, 346.0], 
[476.0, 320.0, 650.0, 501.0, 318.0, 664.0, 534.0, 318.0, 659.0, 470.0, 680.0, 344.0, 492.0, 681.0, 336.0, 527.0, 680.0, 345.0], 
[474.0, 320.0, 652.0, 500.0, 318.0, 664.0, 531.0, 319.0, 657.0, 470.0, 680.0, 344.0, 496.0, 681.0, 336.0, 528.0, 680.0, 345.0], 
[471.0, 319.0, 655.0, 500.0, 318.0, 664.0, 528.0, 319.0, 655.0, 471.0, 680.0, 344.0, 500.0, 681.0, 335.0, 528.0, 680.0, 344.0]])

def run_inference(cfg, onnx_path, out_dir,
                  t_max=50_000,
                  gait_schedule=None,
                  record=True,
                  robot_mode="no_robot"):
    """
    Two-thread inference loop.

    Parameters
    ----------
    cfg           : dict from cpg_snn_config.json
    onnx_path     : Path to cpg_snn.onnx
    out_dir       : Path for output plots
    t_max         : CPG integration steps after warm-up
    gait_schedule : list of (step, gait_idx) for scripted switching
    record        : store data for plots if True
    robot         : if False, skip serial thread and robot commands
    """
    # ── Read all stepper params from config ──────────────────────
    gait_period     = float(cfg["gait_period"])
    burst_threshold = float(cfg["burst_threshold"])
    global_min      = float(cfg["global_min"])
    global_max      = float(cfg["global_max"])
    seq_len         = int(cfg["seq_len"])
    n_gaits         = int(cfg["n_gaits"])
    n_joints        = int(cfg["n_joints"])
    gait_names      = cfg["gait_names"]
    chunk_size      = int(cfg.get("chunk_size",     50))
    spike_thresh    = float(cfg.get("spike_thresh", -2.0))
    cpg_start_time  = int(cfg.get("cpg_start_time", 5000))
    target_rows     = int(cfg.get("target_rows",    54))
    scale           = (global_max - global_min) / 2.0
    shift           = (global_max + global_min) / 2.0

    print(f"  Stepper params from config: "
          f"chunk_size={chunk_size}  spike_thresh={spike_thresh}  "
          f"cpg_start_time={cpg_start_time}")

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Upsample gait tables to match training target resolution
    base_gait_names = [
            "tripod", "tripod_huge", "tripod_right", "tripod_huge_right",
            "ripple", "ripple_tiny", "ripple_right", "ripple_tiny_right",
        ]
    mirrored_gait_names = [
        "tripod_backwards", "tripod_huge_backwards", "tripod_left", "tripod_huge_left",
        "ripple_backwards", "ripple_tiny_backwards", "ripple_left", "ripple_tiny_left",
    ]
    gait_names = base_gait_names + mirrored_gait_names

    GAIT_TABLES_ORIG = []
    for name in base_gait_names:
        gait_table = np.loadtxt(f"{this_file_dir}/gaits/{name}.csv",
                                delimiter=",", dtype=np.float32)
        GAIT_TABLES_ORIG.append(gait_table)

    # for name in mirrored_gait_names:
    #     base_name = name.replace("_backwards", "").replace("_left", "_right")
    #     gait_table = np.loadtxt(f"{this_file_dir}/gaits/{base_name}.csv",
    #                             delimiter=",", dtype=np.float32)
    #     GAIT_TABLES_ORIG.append(np.flip(gait_table, axis=0).copy())

    GAIT_TABLES = upsample_gait_tables(
        GAIT_TABLES_ORIG, gait_names, target_rows)

    # # ── Shared state ─────────────────────────────────────────────
    # shared     = SharedState()
    # stop_event = _threading.Event()

    # # ── Serial thread (only when robot is connected) ─────────────
    # if robot_mode != "no_robot":
    #     ser_thread = _threading.Thread(
    #         target=serial_worker,
    #         args=(shared, n_joints, stop_event),
    #         daemon=True,
    #         name="serial-worker")
    #     ser_thread.start()
    #     print("  Serial thread started.")
    # else:
    #     ser_thread = None
    #     print("  No robot: serial thread suppressed.")

    # ── Inference components ─────────────────────────────────────
    predictor = ONNXGaitPredictor(onnx_path)
    cpg = BLIF_CPG(N=6, t_max=t_max)

    if gait_schedule:
        schedule = sorted(gait_schedule, key=lambda x: x[0])
    else:
        schedule = []

    # ── Boot sequence: identical to training ─────────────────────
    # Training does:
    #   1. Warm up CPGChunkStepper for cpg_start_time steps
    #   2. Collect spikes for tmax - cpg_start_time steps
    #   3. Run estimate_gait_period on those spikes -> gait_period
    #   4. Use that fixed gait_period for ALL phase computations
    #
    # Inference does EXACTLY the same thing.  The config stores the
    # gait_period computed during training, so we can simply read it
    # without re-estimating.  The CPGChunkStepper produces bit-identical
    # spike times (same ICs, same chunk_size, same integrator), so
    # t % gait_period gives identical phase values to training.
    #
    # There is NO online period estimator. There never was a reason for
    # one once both training and inference used the same integrator.

    print(f"Warming up CPG ({cpg_start_time} steps) ...")
    for _ in range(cpg_start_time):
        cpg.step()
    print(f"  CPG settled.  Using fixed gait_period = {gait_period:.1f} steps\n")

    event_buf = SpikeWindowBuffer(seq_len, N=6, n_gaits=n_gaits)

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
    for current_time in range(cpg_start_time, t_max):
        
        while (sched_ptr < len(schedule)
                and steps_done >= schedule[sched_ptr][0]):
            active_gait = schedule[sched_ptr][1]
            print(f"  step {steps_done:>6d}: gait → "
                    f"{gait_names[schedule[sched_ptr][1]]}")
            sched_ptr += 1

        spikes, _, t_now = cpg.step()
        spike_events = []
        for neuron_id in range(len(spikes)):
            if spikes[neuron_id]:
                spike_events.append((t_now, neuron_id))

        steps_done += 1

        for (t_now, neuron_id) in spike_events:
            abs_phase_rad = float(
                2.0 * np.pi * (t_now % gait_period) / gait_period)

            event_buf.push(neuron_id, abs_phase_rad, active_gait)

            if not event_buf.is_primed:
                continue

            t0 = time.perf_counter()
            window = event_buf.get()
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
    parser.add_argument("--t_max",    type=int,  default=10_000,
                        help="Inference steps after warm-up")
    parser.add_argument("--robot_mode", type=str, default="rospider", help="Options: no_robot, bittle, bittle_sim, unitree_sim")
    args = parser.parse_args()

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = Path(this_file_dir + "/" + args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "cpg_snn.onnx"

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}\n"
            "Run cpg_snn_chunked.py (training) first.")

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
    num_gaits = 9
    gait_times = [i*t//num_gaits for i in range(num_gaits)]
    gait_schedule = [(gait_times[i], i%(num_gaits-1)) for i in range(num_gaits)]

    data = run_inference(
        cfg, onnx_path, out_dir,
        t_max=args.t_max,
        gait_schedule=(None if args.robot_mode=="bittle" else gait_schedule),
        record=True,
        robot_mode=args.robot_mode)

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
