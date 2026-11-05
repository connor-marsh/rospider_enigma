# FILES

Where everything lives. Scripts (run these) sit in this folder; library modules (imported, never
run directly) are in `submodules/`. A module only imports the ones listed beside it, so a change in
one file only affects the files that import it.

## Scripts

| File | What it does | Imports from submodules |
|---|---|---|
| `train.py` | train a model (the steps are in train_pipeline) | train_args, train_pipeline |
| `visualize.py` | diagnostic figures, or publication figures with `--publication` | common, viz_pipeline |
| `sweep.py` | parameter sweeps over train.py, packed onto one GPU (no torch import) | (reads submodules/train_args.py and cpg.py as text) |
| `inference.py` | run a trained model on the robot | common, cpg, run_io |
| `live_visualization.py` | live window for inference, plus a visualizer server for remote use | common, data |

## Modules (`submodules/`), lowest layer first

| Module | Holds | Imports |
|---|---|---|
| `common.py` | paths, output routing, config lookups, small helpers | - |
| `cpg.py` | the CPG: parameters, oscillators, running/replaying, burst and phase analysis | common |
| `data.py` | robot layout, joint partitions, gait tables, targets | - |
| `model.py` | the networks (DenseSNN, TimingGroupedSNN) and the spike function | data |
| `run_io.py` | run config write/read, rebuild a model from a config, load a checkpoint, ONNX | common, cpg, data, model |
| `analysis.py` | measurements without plotting: reports, alignment scores, replay helpers | data |
| `training.py` | StreamSampler, spike objectives, calibration, losses, the training loop | common, model |
| `plotting.py` | every figure: end-of-training, diagnostic, and publication | analysis, common, cpg |
| `train_args.py` | the training command line (build_parser) | analysis, cpg, data, training |
| `train_pipeline.py` | the steps of a training run, called in order by train.py | analysis, common, cpg, data, model, plotting, run_io, training |
| `viz_pipeline.py` | run_visualization, run_publication, and the visualization command line | analysis, common, cpg, data, plotting, run_io |

## Main names per module

- **common**: `json_safe`, `git_info`, `route_subdir`, `out_path`, `in_path`, `joint_type_names`, `outputs_path`, `cfg_get`
- **cpg**: `cpg_weight_matrix`, `LIFGeneralArray`, `BurstingLIF`, `LIFCPGStepper`, `run_cpg`, `detect_burst_threshold`, `burst_onsets`, `analyse_cpg`, `cycle_phase`, `cpg_spike_stats`, `replay_cpg`, `cpg_phase`, `spike_bursts`
- **data**: `build_group_cols`, `build_partition`, `build_timing_map`, `load_gait_tables`, `default_leg_layout`, `upsample_gait_tables`, `build_targets`, `joint_demand`
- **model**: `spike_fn`, `init_beta_logit`, `DenseSNN`, `TimingGroupedSNN`, `SingleStepONNX`, `SingleStepONNXTiming`, `grad_blocks`
- **run_io**: `build_model_from_cfg`, `load_run`, `build_run_config`, `record_training_outcome`, `write_run_config`, `export_onnx`
- **analysis**: `drive_report`, `drive_spike_report`, `timing_report`, `alignment_score`, `alignment_pair`, `alignment_section`, `timing_raster`, `predict_and_membranes`, `gt_degrees`, `drive_channels`, `circular_stats`, `fundamental_phase`, `circ_diff`, `unit_train`
- **training**: `StreamSampler`, `measure_rates`, `calibrate_gains`, `SpikeObjective`, `NoSpikeObjective`, `CPGMatchSpikeObjective`, `MinCountSpikeObjective`, `make_spike_objective`, `reinit_timing_units`, `make_gait_weights`, `masked_loss`, `detach_state`, `apply_reset`, `MetricsWriter`, `lambda_schedule`, `lambda_schedule_text`, `run_training`
- **plotting**: `plot_cpg_raster`, `plot_training_curves`, `plot_reconstruction`, `plot_transition`, `plot_timing_rate_history`, `plot_joint_gait_grid`, `plot_alignment_history`, `plot_drive_fanin_history`, `plot_alignment`, `plot_phase_fold`, `plot_alignment_summary`, `plot_drive_gates`, `plot_routing`, `plot_taus`, `plot_membranes`, `gait_label`, `joint_label`, `publication_style`, `resolve_gaits`, `resolve_joints`, `fig_reconstruction`, `fig_transitions`, `fig_timing_alignment`, `read_metrics`, `fig_training_curves`
- **train_args**: `build_parser`
- **train_pipeline**: `setup`, `load_gaits`, `run_cpg_phase`, `analyze_cpg`, `build_targets_phase`, `make_samplers`, `build_model`, `prepare_training`, `calibrate`, `make_objective`, `write_config`, `train`, `evaluate`, `export`, `visualize`
- **viz_pipeline**: `default_args`, `run_visualization`, `run_publication`

## Other folders and files

- `docs/`: `notes.md` (the curated summary: current state, decisions and their reasons,
  open questions and ideas; rewritten as understanding changes), `history.md` (one dated
  entry per chat, newest first; only ever added to), and the `*_todo.md` action lists.
- `.githooks/`: on every commit, checkout, merge and rebase, writes `project_manifest.txt`
  here (one hash per tracked .py/.md file in this folder). Enabled once with
  `git config core.hooksPath src/cpg_snn/leg_grouped_timing/.githooks`.
- `project_manifest.txt`: gitignored; upload it to the project with the code files.
- `outputs/`: training runs and `zsweeps/` (gitignored).
- the gait CSV folder, resolved from `--gaits_dir` (default `../gaits`).

## Which files a task needs

- training loop, penalties, sampler: `submodules/training.py` (plus `train_pipeline.py` if the run's steps change)
- architecture: `submodules/model.py`, plus `run_io.py` if the config must record something new
- a new option: `submodules/train_args.py` plus the module that uses it
- a figure: `submodules/plotting.py` (plus `viz_pipeline.py` to wire it into a run)
- the CPG: `submodules/cpg.py`;  gait data or layout: `submodules/data.py`