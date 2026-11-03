"""
Bursting-LIF CPG -> stateful SNN -> multi-gait joint angles
===========================================================

A discrete bursting-LIF CPG (LIFGeneralArray + BurstingLIF + an inhibitory
coupling matrix; one step() == one timestep) emits raw spikes. They are the
network's ONLY input -- no phase channel, no gait period -- so phase has to be
held in state: hidden units get learnable membrane taus, initialized
log-uniformly over [tau_min, tau_max], forming a temporal basis over "time since
last burst". Training is truncated BPTT over contiguous chunks with state carried
(detached) across chunk boundaries, like deployment.

--arch dense           one fully connected network; the ablation baseline.
--arch timing_grouped  CPG -> (timing layer | direct CPG drive) -> G disconnected
                       sub-networks, each two spiking layers plus a block-diagonal
                       analog readout; group g writes only its own gait-table
                       columns. --drive_source picks the drive: timing (learned
                       timing LIF layer, per-gait CPG->timing weights), cpg
                       (per-gait gain on each CPG channel) or cpg_lif (the same
                       gains through a fixed-threshold LIF). See the
                       TimingGroupedSNN docstring and --help.

CPG size is an argument: --n_cpg_neurons selects a coupling matrix from
CPG_W_BY_N and sizes the SNN input; nothing downstream assumes a particular N.

Leg / joint layout: LEG_COLS (quadruped, 8 joints: columns j and j+4 are one
leg's two joints) and HEXAPOD_LEG_COLS (18 joints, 3 per leg, LF/LM/LR/RF/RM/RR)
only group columns for the diagnostic plots. A run resolves them from
--n_cpg_neurons via default_leg_layout(), or from --leg_cols. Servo routing is
inference.py's concern; the gait-table column order is fixed.

Usage
    python train.py --epochs 300 --hidden 256
    python train.py --dry_run                         # data + plots, no training
    python train.py --arch timing_grouped --hidden 128
    python train.py --n_cpg_neurons 18 --drive_source cpg_lif --drive_lambda 1e-3

The steps live in submodules/train_pipeline.py; see FILES.md for where everything else lives.
"""
from types import SimpleNamespace

from submodules import train_pipeline as tp
from submodules.train_args import build_parser


def main():
    args = build_parser().parse_args()
    run = SimpleNamespace()        # the run's state, passed from step to step
    tp.setup(args, run)                       # seeds, device, output folder
    tp.load_gaits(args, run)                  # [0/6] gait tables, leg layout, structure
    tp.run_cpg_phase(args, run)               # [1/6] run the CPG
    tp.analyze_cpg(args, run)                 # [2/6] bursts, phase, period-based defaults
    tp.build_targets_phase(args, run)         # [3/6] upsampled tables -> targets
    tp.make_samplers(args, run)               # [4/6] train/val stream samplers
    tp.build_model(args, run)                 # [5/6] build, freeze, compile the model
    tp.prepare_training(args, run)            # gait weights, periodic diagnostic
    tp.calibrate(args, run)                   # timing-layer gain calibration
    tp.make_objective(args, run)              # spike objective
    tp.write_config(args, run)                # config, written before training
    tp.train(args, run)                       # train, then reload the best checkpoint
    tp.evaluate(args, run)                    # [6/6] reconstruction, transitions, outcome
    tp.export(args, run)                      # ONNX export (--export_onnx 1)
    tp.visualize(args, run)                   # [7/7] end-of-run visualization


if __name__ == "__main__":
    main()