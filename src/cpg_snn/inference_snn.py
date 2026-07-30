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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from servo_controller_msgs.msg import ServoPosition, ServosPosition
import rclpy
from rclpy.node import Node


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

wkF = np.array([
    [9,49,67,38,24,20,27,14],[8,50,68,39,28,21,22,14],
    [10,51,70,41,26,22,18,14],[12,52,69,42,24,24,11,14],
    [14,52,63,44,22,26,1,15],[16,53,53,45,21,27,-2,15],
    [18,53,41,46,20,29,-1,15],[21,54,26,47,18,30,6,16],
    [22,54,23,48,18,32,9,17],[25,54,20,49,16,34,12,18],
    [26,54,17,51,16,37,17,18],[28,54,14,52,15,39,22,19],
    [30,52,11,54,14,45,27,19],[32,54,11,54,14,44,29,20],
    [33,58,13,55,15,36,27,21],[34,61,16,56,15,31,24,23],
    [36,64,18,56,14,24,23,24],[38,66,20,57,14,20,22,26],
    [39,67,22,57,14,16,21,28],[41,64,24,57,14,5,20,30],
    [42,55,26,57,14,-1,19,32],[44,44,28,57,15,-3,18,35],
    [45,30,29,57,15,1,18,38],[46,21,31,57,15,5,17,40],
    [47,19,32,56,16,9,17,43],[48,16,35,57,17,12,16,44],
    [49,12,37,62,18,17,14,35],[49,9,39,66,20,24,14,29],
    [50,8,40,68,21,28,14,23],[51,10,42,70,22,26,14,19],
    [52,12,43,70,24,24,15,17],[52,14,44,67,26,22,15,5],
    [53,16,46,59,27,21,15,-2],[53,18,47,47,29,20,16,-2],
    [54,21,48,34,30,18,16,1],[54,22,49,24,32,18,17,6],
    [54,25,50,21,34,16,18,10],[54,26,51,19,37,16,19,12],
    [54,28,52,15,39,15,20,19],[52,30,54,12,45,14,19,24],
    [54,32,55,12,44,14,20,27],[58,33,55,11,36,15,22,29],
    [61,34,56,14,31,15,24,26],[64,36,56,17,24,14,25,24],
    [66,38,57,18,20,14,27,23],[67,39,57,21,16,14,29,21],
    [64,41,57,23,5,14,31,20],[55,42,57,24,-1,14,33,20],
    [44,44,57,26,-3,15,36,19],[30,45,57,28,1,15,39,18],
    [21,46,56,30,5,15,42,17],[19,47,56,32,9,16,45,17],
    [16,48,59,33,12,17,41,17],[12,49,64,35,17,18,33,16],
], dtype=np.float32)

bk = np.array([
    [36,40,36,62,6,-3,6,1],[34,47,32,63,7,-4,7,4],
    [30,53,28,59,8,-3,9,9],[26,58,25,57,10,-2,10,10],
    [22,57,26,55,12,2,6,8],[18,51,29,52,14,8,2,7],
    [15,51,36,50,15,6,-2,6],[17,48,43,47,9,5,-3,5],
    [21,45,49,44,5,5,-4,5],[29,43,55,42,2,5,-3,5],
    [35,39,60,38,-1,6,-1,6],[42,36,63,35,-3,6,1,6],
    [49,32,62,31,-4,7,6,8],[54,28,58,28,-3,9,10,9],
    [57,26,57,24,0,10,9,11],[56,21,54,26,3,12,8,4],
    [51,17,52,31,8,15,6,1],[50,15,49,38,6,14,6,-2],
    [47,18,47,44,5,8,5,-3],[45,24,44,51,5,4,5,-4],
    [42,30,41,56,5,1,5,-3],[38,37,37,60,6,-2,6,-1],
], dtype=np.float32)

wkL = np.array([
    [47,54,58,51,-2,13,-2,2],[45,56,52,52,0,14,-3,2],
    [45,57,46,52,1,15,-4,2],[45,58,38,53,1,17,-2,2],
    [46,59,31,54,1,19,1,2],[47,59,24,54,1,22,6,3],
    [48,60,21,55,2,24,12,3],[49,58,23,56,1,30,16,3],
    [49,61,25,57,1,30,13,3],[50,67,28,57,1,23,12,4],
    [51,69,31,58,2,15,11,5],[52,68,34,58,2,8,10,5],
    [52,65,36,59,2,3,9,6],[53,60,39,61,2,-1,9,5],
    [54,55,41,62,2,-3,9,3],[54,50,43,62,2,-5,9,2],
    [54,43,47,60,3,-5,7,1],[55,35,48,58,3,-3,8,1],
    [56,28,51,57,3,1,8,-1],[57,18,52,55,3,7,9,-1],
    [57,15,54,53,4,10,10,-2],[58,13,55,50,5,16,11,-2],
    [58,16,57,49,5,16,12,-2],[59,18,58,46,6,14,14,-2],
    [59,21,60,44,6,12,14,-2],[60,25,61,43,7,10,16,0],
    [60,28,62,42,7,9,18,1],[60,31,63,42,10,8,20,3],
    [62,32,63,42,6,7,24,3],[62,35,63,44,6,6,27,1],
    [63,38,63,44,3,6,31,1],[61,40,62,45,2,7,35,1],
    [60,42,65,46,1,7,37,1],[58,45,70,47,0,7,27,1],
    [57,47,71,48,-1,7,23,2],[54,48,71,49,-1,8,14,1],
    [53,51,69,49,-2,8,8,1],[50,52,66,50,-2,9,2,1],
    [49,53,63,51,-2,12,-2,2],
], dtype=np.float32)

wkR = -(np.array([
    [-54,-47,-51,-58,-13,2,-2,2],[-56,-45,-52,-52,-14,0,-2,3],
    [-57,-45,-52,-46,-15,-1,-2,4],[-58,-45,-53,-38,-17,-1,-2,2],
    [-59,-46,-54,-31,-19,-1,-2,-1],[-59,-47,-54,-24,-22,-1,-3,-6],
    [-60,-48,-55,-21,-24,-2,-3,-12],[-58,-49,-56,-23,-30,-1,-3,-16],
    [-61,-49,-57,-25,-30,-1,-3,-13],[-67,-50,-57,-28,-23,-1,-4,-12],
    [-69,-51,-58,-31,-15,-2,-5,-11],[-68,-52,-58,-34,-8,-2,-5,-10],
    [-65,-52,-59,-36,-3,-2,-6,-9],[-60,-53,-61,-39,1,-2,-5,-9],
    [-55,-54,-62,-41,3,-2,-3,-9],[-50,-54,-62,-43,5,-2,-2,-9],
    [-43,-54,-60,-47,5,-3,-1,-7],[-35,-55,-58,-48,3,-3,-1,-8],
    [-28,-56,-57,-51,-1,-3,1,-8],[-18,-57,-55,-52,-7,-3,1,-9],
    [-15,-57,-53,-54,-10,-4,2,-10],[-13,-58,-50,-55,-16,-5,2,-11],
    [-16,-58,-49,-57,-16,-5,2,-12],[-18,-59,-46,-58,-14,-6,2,-14],
    [-21,-59,-44,-60,-12,-6,2,-14],[-25,-60,-43,-61,-10,-7,0,-16],
    [-28,-60,-42,-62,-9,-7,-1,-18],[-31,-60,-42,-63,-8,-10,-3,-20],
    [-32,-62,-42,-63,-7,-6,-3,-24],[-35,-62,-44,-63,-6,-6,-1,-27],
    [-38,-63,-44,-63,-6,-3,-1,-31],[-40,-61,-45,-62,-7,-2,-1,-35],
    [-42,-60,-46,-65,-7,-1,-1,-37],[-45,-58,-47,-70,-7,0,-1,-27],
    [-47,-57,-48,-71,-7,1,-2,-23],[-48,-54,-49,-71,-8,1,-1,-14],
    [-51,-53,-49,-69,-8,2,-1,-8],[-52,-50,-50,-66,-9,2,-1,-2],
    [-53,-49,-51,-63,-12,2,-2,2],
], dtype=np.float32))

GAIT_TABLES_ORIG = [wkF, bk, wkL, wkR]   # original row counts

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


# ═══════════════════════════════════════════════════════════════════
# 3.  CPG dynamics + chunk stepper
# ═══════════════════════════════════════════════════════════════════

class LIFGeneralArray:
	def __init__(self,N,vth,du,dv,bias,u=0,v=0,ufloor=0,vfloor=0,refractory_period=0):
		# To get the SW CFG Model Comment out the left shift and multiplications
		self.vth = vth
		self.du = du
		self.dv = dv
		self.bias = bias
		self.u = np.ones(N)*u
		self.v = np.ones(N)*v
		self.ufloor = ufloor
		self.vfloor = vfloor
		self.refractory_period = refractory_period
		self.time_since_last_spike = np.zeros(N)
		self.N = N

	def next_step(self, current):
		self.u = self.u * (1 - self.du) + current
		self.v = self.v * (1 - self.dv) + self.u + self.bias
		
		refractory_mask = self.time_since_last_spike > 0
		self.v[refractory_mask] = 0
		self.time_since_last_spike = np.clip(self.time_since_last_spike - 1, 0, None)

		# self.u = np.clip(self.u,self.ufloor,None)
		# self.v = np.clip(self.v,self.vfloor,None)

		spike = self.v >= self.vth
		self.v[spike] = 0
		self.time_since_last_spike[spike] = self.refractory_period
		return spike.astype(np.float32)

	def reset(self, u=0, v=0):
		self.u.fill(u)
		self.v.fill(v)

class BurstingLIF:
	def __init__(self,N,vth_main,du_main,dv_main,refrac_main,
			  vth_fb,du_fb,dv_fb,refrac_fb,from_fb_weight,to_fb_weight):
		self.n_main = LIFGeneralArray(N,vth_main,du_main,dv_main,bias=0,u=0,v=0,ufloor=-vth_main*100,vfloor=-vth_main*100,refractory_period=refrac_main)
		self.n_fb = LIFGeneralArray(N,vth_fb,du_fb,dv_fb,bias=0,u=0,v=0,ufloor=-vth_fb*100,vfloor=-vth_fb*100,refractory_period=refrac_fb)
		
		self.input_2_feedback_neuron_weight = to_fb_weight
		self.feedback_2_input_neuron_weight = from_fb_weight

		self.fb_current = np.zeros(N)

	def forward(self,current):
		input_neuron_current = current + self.fb_current
		input_neuron_spike = self.n_main.next_step(input_neuron_current)

		to_fb_current = input_neuron_spike*self.input_2_feedback_neuron_weight

		fb_neuron_spike = self.n_fb.next_step(to_fb_current)

		self.fb_current = fb_neuron_spike*self.feedback_2_input_neuron_weight

		return input_neuron_spike, fb_neuron_spike
	
	def reset(self):
		self.n_main.reset()
		self.n_fb.reset()

class BLIF_CPG:
    def __init__(self, N=4, t_max=2000):
        # Input Neuron Params
        vth_main = 100
        du_main = 0.1
        dv_main = 0.3
        refrac_main = 1

        # Feedback Neuron Params
        vth_fb = 100
        du_fb = 1.0
        dv_fb = 0.
        refrac_fb = 1

        # Weight Params
        from_fb_weight = -1000000
        to_fb_weight = 10

        # Number of CPG Neurons

        self.burstingNeuron1 = BurstingLIF(N,vth_main,du_main,dv_main,refrac_main,vth_fb,du_fb,dv_fb,refrac_fb,from_fb_weight,to_fb_weight)
        self.weight_matrix = []
        # Weight initialization

        ############# This configuration works best for 3 neuron CPG
        if N == 3:
            self.weight_matrix = np.asarray([[   0.         ,-523.65135942 ,-593.28982051],
                                        [-696.81822016  ,  0.         ,-632.34680962],
                                        [-687.56816569 ,-577.5693762   ,  0.        ]])

        ############# This configuration works best for 4 neuron CPG
        elif N==4:
            self.weight_matrix = np.asarray([[   0.         ,-648.52905924 ,-449.60304695 ,-413.48426163],
                                        [-369.91504928 ,   0.         ,-592.29635234 ,-568.0712858 ],
                                        [-412.08729881 ,-391.54918498 ,   0.         ,-618.03381552],
                                        [-498.16458351 ,-655.01105883 ,-345.38277449 ,   0.        ]])

        ############# This configuration works best for 6 neuron CPG
        elif N==6:
            self.weight_matrix = np.asarray([[   0.,         -375.86210512, -518.18703523, -371.82375498, -399.74231244,
                                        -487.45119873],
                                        [-531.99480471,    0.,         -489.1139223,  -128.33470562, -404.33117771,
                                        -628.03347932],
                                        [-529.89653583, -418.34662835,    0.,         -543.37143674, -336.83773596,
                                        -679.12224243],
                                        [-674.09562904, -130.56007131, -297.35360394,    0.,         -363.1208234,
                                        -425.10847629],
                                        [-486.03391005, -386.7920052,  -412.91478912, -437.7646991,     0.,
                                        -288.47748806],
                                        [-112.97808475, -510.59115452, -367.63412082, -374.83106147, -393.86103887,
                                            0.        ]])

        # Run the system

        i_scale = 8.

        self.i_app = np.ones((N,t_max))*i_scale

        self.bn_spikes = np.zeros((N,t_max))

        self.inter_neuron_current = np.zeros(N)

        self.currents = np.zeros((N,t_max))

        self.t = 0

    def step(self):
        c_in = self.inter_neuron_current + self.i_app[:,self.t]

        self.currents[:,self.t] = c_in

        n_main,_ = self.burstingNeuron1.forward(c_in)

        self.bn_spikes[:,self.t] = n_main

        self.inter_neuron_current = self.weight_matrix @ n_main

        self.t += 1

        return n_main, self.burstingNeuron1.n_main.v, self.t
    
def run_blif_cpg(t_max=2000, N=4, cpg_start_time=100):
    network = BLIF_CPG(N=N, t_max=t_max)

    for t in range(cpg_start_time):
        _, _, _ = network.step()

    bn_spikes = []
    v_ms = []
    for t in range(cpg_start_time, t_max):
        spikes, v_m, _ = network.step()
        bn_spikes.append(spikes)
        v_ms.append(v_m)

    
    bn_spikes = np.array(bn_spikes).T
    print(bn_spikes.shape)

    fig, axes = plt.subplots(1, 1)
    for i in range(N):
        axes.plot(bn_spikes[i], label=f"Neuron {i+1} spikes")
    axes.legend()
    plt.tight_layout()

    fig2, axes2 = plt.subplots(N, 2)
    for i in range(N):
        axes2[i,0].plot(v_ms[i])
        axes2[i,0].set_title(f"Input Current to Neuron {i+1}")

        axes2[i,1].plot(bn_spikes[i])
        axes2[i,1].set_title(f"Neuron {i+1} spikes")

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    plt.close(fig2)

    spike_times = []
    spike_neurons = []
    for t in range(cpg_start_time, t_max):
        for i in range(N):
            if bn_spikes[i][t-cpg_start_time]:
                spike_times.append(t)
                spike_neurons.append(i)
    return np.array(spike_times), np.array(spike_neurons)

def sigmoid(x, b=5.0, dsyn=-1.0):
    return 1.0 / (1.0 + np.exp(-b * (x - dsyn)))


def neuron_eqs(S, I, alpha, delta, Tf, Ts, Tus):
    vm, vf, vs, vus = S
    dvm = (-vm
           - alpha[0] * np.tanh(vf  - delta[0])
           - alpha[1] * np.tanh(vs  - delta[1])
           - alpha[2] * np.tanh(vs  - delta[2])
           - alpha[3] * np.tanh(vus - delta[3])
           + I)
    dvf  = (vm - vf)  / Tf
    dvs  = (vm - vs)  / Ts
    dvus = (vm - vus) / Tus
    return [dvm, dvf, dvs, dvus]


def make_network(N, alpha, delta, g_inh, Iapp):
    asyn = g_inh * np.ones((N, N))
    np.fill_diagonal(asyn, 0.0)

    def network(t, S):
        dS   = []
        Vs   = np.array([S[i * 4 + 2] for i in range(N)])
        Isyn = asyn @ sigmoid(Vs)
        for i in range(N):
            dS.extend(neuron_eqs(
                S[i * 4:(i + 1) * 4], Iapp + Isyn[i],
                alpha, delta, 1.0, 50.0, 2500.0))
        return dS

    return network


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

class SpikeEventBuffer:
    """
    Fixed-length FIFO of spike-event feature vectors.

    Each event: [one_hot_neuron(N), sin(φ_abs), cos(φ_abs),
                                    sin(φ_rel), cos(φ_rel),
                                    one_hot_gait(n_gaits)]  — N+4+n_gaits dims.

    The gait flag is concatenated per event, matching training exactly.
    LayerNorm before each LIF layer prevents the constant flag from
    saturating the membrane — it normalises across the hidden dimension
    so the four flags produce four discriminable pre-threshold values.
    """

    def __init__(self, seq_len, N=4, n_gaits=4):
        self.seq_len   = seq_len
        self.N         = N
        self.n_gaits   = n_gaits
        self.n_in      = N + 2 + n_gaits
        self.buf       = np.zeros((seq_len, self.n_in), dtype=np.float32)
        self.n_pushed  = 0
        self._last_t   = None

    def push(self, neuron_id, t_now, abs_phase_rad, gait_period, gait_idx):
        if self._last_t is None:
            isi = gait_period
        else:
            isi = t_now - self._last_t
        self._last_t  = t_now
        #rel_phase_rad = 2.0 * np.pi * isi / gait_period

        feat = np.zeros(self.n_in, dtype=np.float32)
        feat[neuron_id]                  = 1.0
        feat[self.N]                     = float(np.sin(abs_phase_rad))
        feat[self.N + 1]                 = float(np.cos(abs_phase_rad))
        #feat[self.N + 2]                 = float(np.sin(rel_phase_rad))
        #feat[self.N + 3]                 = float(np.cos(rel_phase_rad))
        feat[self.N + 2 + gait_idx]      = 1.0
        self.buf[:-1] = self.buf[1:]
        self.buf[-1]  = feat
        self.n_pushed += 1

    def get(self):
        return self.buf.copy()

    @property
    def is_primed(self):
        return self.n_pushed >= self.seq_len


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

import threading as _threading

class SharedState:
    """
    Lock-protected shared state between inference and serial threads.
    """

    def __init__(self):
        self._lock      = _threading.Lock()
        self._gait_idx  = 0
        self._cmd       = None
        self._cmd_ready = False

    def get_gait(self):
        with self._lock:
            return self._gait_idx

    def set_cmd(self, cmd):
        with self._lock:
            self._cmd       = cmd
            self._cmd_ready = True

    def get_cmd(self):
        with self._lock:
            return self._cmd, self._cmd_ready

    def set_gait(self, idx):
        with self._lock:
            self._gait_idx = int(idx)


def serial_worker(shared, n_joints, stop_event):
    """
    Daemon thread owning all serial I/O.
    """

    cur_gesture = 0
    while not stop_event.is_set():

        cmd, ready = shared.get_cmd()
        if ready and cmd is not None:

            if cmd[0] == "I": # REAL HARDWARE:
                from PetoiRobot import send, goodPorts, readGestureVal
                cmd, ready = shared.get_cmd()
                if ready and cmd is not None:
                    try:
                        send(goodPorts, cmd)
                    except Exception as e:
                        print(f"  [serial] send error: {e}")

                try:
                    gesture = readGestureVal()
                    if gesture is not None and gesture != -1:
                        if gesture != cur_gesture:
                            print(f"  [serial] gait {cur_gesture} → {gesture}")
                            cur_gesture = gesture
                        shared.set_gait(cur_gesture)
                except Exception as e:
                    print(f"  [serial] gesture read error: {e}")

            else: # SIM
                from std_msgs.msg import Float64
                for i, joint_name in enumerate(command_topics[:8]):
                    msg = Float64(data=JOINT_DIRECTIONS[i]*(cmd[i]+JOINT_OFFSETS[i]))
                    publishers[joint_name].publish(msg)
                
                if len(command_topics) > 8:
                    publishers["/a1_gazebo/FL_hip_joint/command"].publish(0)
                    publishers["/a1_gazebo/FR_hip_joint/command"].publish(0)
                    publishers["/a1_gazebo/RR_hip_joint/command"].publish(-0.1)
                    publishers["/a1_gazebo/RL_hip_joint/command"].publish(0.1)
                rate.sleep()

        


# ═══════════════════════════════════════════════════════════════════
# 7.  Visualisation
# ═══════════════════════════════════════════════════════════════════

def plot_cpg_vm(all_vm, out_dir, n_show=30_000):
    T      = len(all_vm)
    n_show = min(n_show, T)
    t      = np.arange(n_show)
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    for i in range(4):
        axes[i].plot(t, all_vm[:n_show, i],
                     color=CPG_COLORS[i], linewidth=0.9)
        axes[i].axhline(-2.0, color="k", ls="--", lw=0.7, alpha=0.5,
                         label="threshold" if i == 0 else "")
        axes[i].set_ylabel(f"CPG {i}\n$v_m$", fontsize=9)
        axes[i].grid(True, alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel(f"Inference timestep (first {n_show})")
    plt.suptitle("CPG Membrane Potentials (vm) — Inference", fontsize=12)
    plt.tight_layout()
    p = out_dir / "cpg_vm.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  [saved] {p}")


def plot_spike_events(rec_t, rec_neuron, gait_period, out_dir,
                      n_show=3_000):
    if len(rec_t) == 0:
        return
    mask   = rec_t <= rec_t[0] + n_show
    t_show = rec_t[mask]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                    sharex=True, height_ratios=[3, 1])
    for i in range(4):
        idx = np.where((rec_neuron == i) & mask)[0]
        ax1.scatter(rec_t[idx], np.full(len(idx), i),
                    marker="|", s=150, lw=1.8,
                    color=CPG_COLORS[i], label=f"Neuron {i}")
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([f"CPG {i}" for i in range(4)])
    ax1.set_title(f"Spike Events  (period ≈ {gait_period:.0f} steps)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, axis="x", alpha=0.2)
    phase = np.degrees(2.0 * np.pi * (t_show % gait_period) / gait_period)
    ax2.plot(t_show, phase, color="#6a0572", lw=1.2)
    ax2.set_ylabel("Phase (°)"); ax2.set_xlabel("Simulation time")
    ax2.set_title("Gait phase at each spike event")
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    p = out_dir / "spike_events.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  [saved] {p}")


def plot_gait_reconstruction(rec_pred, rec_true, rec_gait_idx,
                              n_joints, gait_names, out_dir,
                              n_samples_per_gait=500):
    n_gaits = len(gait_names)
    all_true_denorm, all_pred_denorm = [], []

    for g in range(n_gaits):
        mask = rec_gait_idx == g
        if not mask.any():
            all_true_denorm.append(np.zeros((0, n_joints)))
            all_pred_denorm.append(np.zeros((0, n_joints)))
            print(f"  {gait_names[g]}: no events — skipping.")
            continue

        n_plot    = min(n_samples_per_gait, mask.sum())
        true_plot = rec_true[mask][:n_plot]
        pred_plot = rec_pred[mask][:n_plot]
        all_true_denorm.append(true_plot)
        all_pred_denorm.append(pred_plot)

        cols = min(4, n_joints)
        rows = int(np.ceil(n_joints / cols))
        fig, axes = plt.subplots(rows, cols,
                                  figsize=(5 * cols, 3.2 * rows),
                                  squeeze=False)
        ev = np.arange(n_plot)
        for j in range(n_joints):
            ax   = axes[j // cols][j % cols]
            rmse = np.sqrt(np.mean((pred_plot[:, j] - true_plot[:, j]) ** 2))
            ax.plot(ev, true_plot[:, j], label="GT",
                    color=TRUE_COLOR, lw=1.8, zorder=3)
            ax.plot(ev, pred_plot[:, j], label="Predicted",
                    color=PRED_COLORS[g % len(PRED_COLORS)],
                    lw=1.5, ls="--", alpha=0.9, zorder=2)
            err = np.abs(pred_plot[:, j] - true_plot[:, j])
            ax.fill_between(ev, pred_plot[:, j] - err,
                            pred_plot[:, j] + err,
                            color=PRED_COLORS[g % len(PRED_COLORS)],
                            alpha=0.12, zorder=1)
            ax.set_title(f"Joint {j+1}  (RMSE={rmse:.2f}°)", fontsize=9)
            ax.set_xlabel("Spike-event window index", fontsize=8)
            ax.set_ylabel("Angle (°)", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.25)
        for j in range(n_joints, rows * cols):
            axes[j // cols][j % cols].set_visible(False)
        plt.suptitle(f"GT vs Predicted — {gait_names[g]}  "
                     f"(first {n_plot} windows)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = out_dir / f"gait_reconstruction_{gait_names[g]}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [saved] {p}")

    active = [g for g in range(n_gaits) if len(all_true_denorm[g]) > 0]
    if active:
        fig, axes = plt.subplots(len(active), 1,
                                  figsize=(14, 3 * len(active)),
                                  sharex=False)
        if len(active) == 1:
            axes = [axes]
        for row, g in enumerate(active):
            ax        = axes[row]
            true_plot = all_true_denorm[g]
            pred_plot = all_pred_denorm[g]
            ev        = np.arange(len(true_plot))
            rmse      = np.sqrt(np.mean(
                (pred_plot[:, 0] - true_plot[:, 0]) ** 2))
            ax.plot(ev, true_plot[:, 0], label="GT",
                    color=TRUE_COLOR, lw=1.8)
            ax.plot(ev, pred_plot[:, 0], label="Predicted",
                    color=PRED_COLORS[g % len(PRED_COLORS)],
                    lw=1.5, ls="--", alpha=0.9)
            ax.fill_between(ev, true_plot[:, 0], pred_plot[:, 0],
                            alpha=0.18,
                            color=PRED_COLORS[g % len(PRED_COLORS)])
            ax.set_title(f"{gait_names[g]} — Joint 1  (RMSE={rmse:.2f}°)",
                         fontsize=10)
            ax.set_ylabel("Angle (°)", fontsize=8)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Spike-event window index", fontsize=9)
        plt.suptitle("All Gaits — Joint 1 GT vs Predicted  (summary)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = out_dir / "gait_reconstruction_summary.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [saved] {p}")

    rmse_mat = np.full((n_gaits, n_joints), np.nan)
    for g in range(n_gaits):
        if len(all_true_denorm[g]) == 0:
            continue
        for j in range(n_joints):
            rmse_mat[g, j] = np.sqrt(np.mean(
                (all_pred_denorm[g][:, j]
                 - all_true_denorm[g][:, j]) ** 2))
    fig, ax = plt.subplots(
        figsize=(max(6, n_joints * 0.9), n_gaits * 0.9 + 1.5))
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
                ax.text(j, g, f"{rmse_mat[g, j]:.1f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if rmse_mat[g, j] > vmax * 0.6
                        else "black")
    plt.tight_layout()
    p = out_dir / "rmse_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {p}")


def plot_spike_event_overview(rec_t, rec_neuron, rec_phase_deg,
                               rec_pred, rec_true, rec_gait_idx,
                               n_joints, gait_names, out_dir):
    E = len(rec_t)
    if E == 0:
        return
    ev_idx = np.arange(E)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 1, hspace=0.5, height_ratios=[1, 1, 3])

    ax0 = fig.add_subplot(gs[0])
    for i in range(4):
        mask = rec_neuron == i
        ax0.scatter(ev_idx[mask], np.full(mask.sum(), i),
                    marker="|", s=120, lw=1.6,
                    color=CPG_COLORS[i], label=f"CPG {i}")
    ax0.set_yticks(range(4))
    ax0.set_yticklabels([f"CPG {i}" for i in range(4)])
    ax0.set_title("Neuron identity at each spike event")
    ax0.legend(fontsize=7, loc="upper right", ncol=4)
    ax0.grid(True, axis="x", alpha=0.2)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    sc  = ax1.scatter(ev_idx, rec_phase_deg,
                      c=rec_phase_deg, cmap="hsv", s=8,
                      vmin=0, vmax=360)
    plt.colorbar(sc, ax=ax1, label="Phase (°)", pad=0.01)
    ax1.set_ylabel("Phase (°)")
    ax1.set_title("Gait phase at each spike event")
    ax1.set_ylim(-5, 370)
    ax1.grid(True, alpha=0.2)

    ax2   = fig.add_subplot(gs[2], sharex=ax0)
    jcols = plt.cm.tab10(np.linspace(0, 1, n_joints))
    for j in range(n_joints):
        ax2.plot(ev_idx, rec_true[:, j],
                 color=jcols[j], lw=1.4, alpha=0.9, label=f"J{j+1}")
        ax2.plot(ev_idx, rec_pred[:, j],
                 color=jcols[j], lw=1.2, alpha=0.75, ls="--")
    ax2.set_xlabel("Spike event index")
    ax2.set_ylabel("Angle (°)")
    ax2.set_title("All joints: GT (solid) vs Predicted (dashed)")
    ax2.legend(fontsize=7, ncol=4, loc="upper right",
               title="solid=GT  dashed=pred")
    ax2.grid(True, alpha=0.2)

    gait_palette = ["#e6f0ff", "#fff3e6", "#e6fff3", "#ffe6f0"]
    prev_g, prev_e = int(rec_gait_idx[0]), 0
    for e in range(1, E):
        g = int(rec_gait_idx[e])
        if g != prev_g or e == E - 1:
            for ax in [ax0, ax1, ax2]:
                ax.axvspan(prev_e, e, alpha=0.15,
                           color=gait_palette[prev_g % len(gait_palette)])
            prev_g, prev_e = g, e

    plt.suptitle(f"Spike-event Inference Overview  ({E} total events)",
                 fontsize=13, fontweight="bold")
    p = out_dir / "spike_event_overview.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {p}")


def plot_latency(latencies, out_dir):
    if len(latencies) == 0:
        return
    lat = np.array(latencies)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(lat, bins=50, color="#457b9d", edgecolor="white", lw=0.5)
    ax1.axvline(np.mean(lat),   color="#e63946", lw=1.5, ls="--",
                label=f"mean={np.mean(lat):.2f} ms")
    ax1.axvline(np.median(lat), color="#f4a261", lw=1.5, ls="--",
                label=f"median={np.median(lat):.2f} ms")
    ax1.set_xlabel("Latency (ms)"); ax1.set_ylabel("Count")
    ax1.set_title("Inference Latency Distribution")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)
    sl  = np.sort(lat)
    cdf = np.arange(1, len(lat) + 1) / len(lat)
    ax2.plot(sl, cdf * 100, color="#457b9d", lw=1.8)
    ax2.axvline(np.percentile(lat, 95), color="#e63946", lw=1.5, ls="--",
                label=f"p95={np.percentile(lat, 95):.2f} ms")
    ax2.axvline(np.percentile(lat, 99), color="#f4a261", lw=1.5, ls="--",
                label=f"p99={np.percentile(lat, 99):.2f} ms")
    ax2.set_xlabel("Latency (ms)"); ax2.set_ylabel("Cumulative %")
    ax2.set_title("Latency CDF")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)
    plt.suptitle("ONNX Inference Latency  (per spike event)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = out_dir / "inference_latency.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  [saved] {p}")


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

    # Upsample gait tables to match training target resolution
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
    # cpg       = CPGChunkStepper(spike_thresh=spike_thresh, chunk_size=chunk_size)
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

    print(f"Warming up CPG ({cpg_start_time} steps in "
          f"chunks of {chunk_size}) ...")
    warmup_chunks = int(np.ceil(cpg_start_time / chunk_size))
    for _ in range(cpg_start_time):
        cpg.step()
    print(f"  CPG settled.  Using fixed gait_period = {gait_period:.1f} steps\n")

    event_buf = SpikeEventBuffer(seq_len, N=6, n_gaits=n_gaits)

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
    test_count = 0
    for current_time in range(t_max):
        #print(current_time)
        active_gait = 0
        while (sched_ptr < len(schedule)
                and steps_done >= schedule[sched_ptr][0]):
            active_gait = schedule[sched_ptr][1]
            print(f"  step {steps_done:>6d}: gait → "
                    f"{gait_names[schedule[sched_ptr][1]]}")
            sched_ptr += 1

        spikes, final_vm, t = cpg.step()
        spike_events = []
        for n in range(len(spikes)):
            if spikes[n]:
                spike_events.append((t, n))

        # if record:
        #     for row in chunk_vm:
        #         all_vm.append(row.copy())

        steps_done += chunk_size

        if not spike_events:
            continue

        for (t_now, neuron_id) in spike_events:
            

            # Use the FIXED gait_period from config for ALL phase
            # computations.  Do NOT use an online estimator here —
            # a 2% period error causes modulo wraparound at the wrong
            # time, giving sin/cos errors up to ±1.07.
            abs_phase_rad = float(
                2.0 * np.pi * (t_now % gait_period) / gait_period)

            event_buf.push(neuron_id, t_now, abs_phase_rad,
                            gait_period, active_gait)

            if not event_buf.is_primed:
                continue
            # ── Predict ──────────────────────────────────────
            t0        = time.perf_counter()
            window    = event_buf.get()
            pred_norm = predictor.predict(window, active_gait)  # FiLM
            pred  = pred_norm * scale + shift
            lat_ms    = (time.perf_counter() - t0) * 1000
            latencies.append(lat_ms)

            # PUBLISH TO ROS TOPIC HERE
            servo_id = [5, 3, 1, 11, 9, 7, 17, 15, 13, 18, 16, 14, 12, 10, 8, 6, 4, 2]
            #print(pred)
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
            test_count+=1

            # from std_msgs.msg import Float64
            # for i, joint_name in enumerate(command_topics[:8]):
            #     msg = Float64(data=JOINT_DIRECTIONS[i]*(cmd[i]+JOINT_OFFSETS[i]))
            #     publishers[joint_name].publish(msg)

            # # Hand off command (no-op if no serial thread)
            
            # if robot_mode=="bittle":
            #     cmd_flat = sum(
            #         [[j + 8, int(np.clip(pred[j], -124, 124))]
            #             for j in range(n_joints)], [])
            #     shared.set_cmd(['I', cmd_flat, 0.0])
            # elif robot_mode=="bittle_sim":
            #     cmd = [int(np.clip(pred[j], -124, 124)) for j in range(n_joints)]
            #     cmd = np.radians(np.array(cmd))
            #     shared.set_cmd(cmd)
            # elif robot_mode=="unitree_sim":
            #     cmd = [int(pred[j]) for j in range(n_joints)]
            #     cmd = np.radians(np.array(cmd))
            #     shared.set_cmd(cmd)

            if record:
                gait_table = GAIT_TABLES[active_gait]
                row_idx    = (int(abs_phase_rad / (2.0 * np.pi)
                                * gait_table.shape[0])
                                % gait_table.shape[0])
                rec_t.append(t_now)
                rec_neuron.append(neuron_id)
                rec_phase_deg.append(float(np.degrees(abs_phase_rad)))
                rec_gait_idx.append(active_gait)
                rec_pred.append(pred.copy())
                rec_true.append(
                    gait_table[row_idx].astype(np.float32))
        
        time.sleep(0.07)

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
    parser.add_argument("--t_max",    type=int,  default=50_000,
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
    if args.robot_mode=="bittle":
        from PetoiRobot import autoConnect, send, goodPorts
        autoConnect()
        send(goodPorts, ['XAd', 0])
        send(goodPorts, ['XGp', 0])
        print("  Robot connected.")
    elif args.robot_mode=="bittle_sim" or args.robot_mode=="unitree_sim":
        import rospy
        from std_msgs.msg import Float64
        
        PUBLISH_RATE = 10.0 # Hz (Control frequency)
        rospy.init_node('gait_decoder_commander_node', anonymous=True)
        rate = rospy.Rate(PUBLISH_RATE)
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
            publishers[topic_name] = rospy.Publisher(topic_name, Float64, queue_size=1)

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
    num_gaits = 5
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
    # plot_cpg_vm(data["all_vm"], out_dir, n_show=30_000)
    plot_spike_events(data["rec_t"], data["rec_neuron"],
                      data["period_final"], out_dir)
    plot_gait_reconstruction(
        data["rec_pred"], data["rec_true"], data["rec_gait_idx"],
        n_joints=data["n_joints"],
        gait_names=data["gait_names"],
        out_dir=out_dir,
        n_samples_per_gait=500)
    plot_spike_event_overview(
        data["rec_t"], data["rec_neuron"], data["rec_phase_deg"],
        data["rec_pred"], data["rec_true"], data["rec_gait_idx"],
        n_joints=data["n_joints"],
        gait_names=data["gait_names"],
        out_dir=out_dir)
    plot_latency(data["latencies"], out_dir)

    print(f"\nDone — outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
