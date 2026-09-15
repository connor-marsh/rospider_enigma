#!/bin/bash
#python3 src/cpg_snn/leg_grouped_timing/train.py --n_cpg_neurons 6 --calibrate_gains 1 --out_dir hexapod_tiny_8_4 --hidden 8 --epochs 200 --readout_hidden 4
#python3 src/cpg_snn/leg_grouped_timing/train.py --n_cpg_neurons 6 --calibrate_gains 1 --out_dir hexapod_tiny_8_8 --hidden 8 --epochs 200 --readout_hidden 8
#python3 src/cpg_snn/leg_grouped_timing/train.py --n_cpg_neurons 6 --calibrate_gains 1 --out_dir hexapod_tiny_4_2 --hidden 4 --epochs 200 --readout_hidden 2
#python3 src/cpg_snn/leg_grouped_timing/train.py --n_cpg_neurons 6 --calibrate_gains 1 --out_dir hexapod_tiny_4_4 --hidden 4 --epochs 200 --readout_hidden 4
python3 src/cpg_snn/leg_grouped_timing/train.py --n_cpg_neurons 6 --calibrate_gains 1 --out_dir hexapod_dense_8 --hidden 8 --epochs 200 --arch dense
python3 src/cpg_snn/leg_grouped_timing/train.py --n_cpg_neurons 6 --calibrate_gains 1 --out_dir hexapod_dense_16 --hidden 16 --epochs 200 --arch dense
python3 src/cpg_snn/leg_grouped_timing/train.py --n_cpg_neurons 6 --calibrate_gains 1 --out_dir hexapod_dense_32 --hidden 32 --epochs 200 --arch dense
python3 src/cpg_snn/leg_grouped_timing/train.py --n_cpg_neurons 6 --calibrate_gains 1 --out_dir hexapod_dense_64 --hidden 64 --epochs 200 --arch dense
