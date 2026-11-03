"""
The training command line: every flag train.py accepts, with its help text.
"""

import argparse

from .analysis import _ALIGN_COR_TAU, _ALIGN_VEL_TAU
from .cpg import CPG_PARAMS, CPG_W_BY_N
from .data import STRUCTURE_SHAPES
from .training import SPIKE_OBJECTIVES


def build_parser():
    """The training CLI: every flag train.py accepts."""
    ap = argparse.ArgumentParser(
        description="Bursting-LIF CPG -> leg-grouped stateful SNN")

    # CPG
    ap.add_argument("--tmax",   type=int,   default=50_000,
                    help="Steps of CPG spike train to collect (the CPG is periodic "
                         "after warmup, so more only duplicates the cycle). Phase and "
                         "switch timing are randomiized independently by StreamSampler.")
    ap.add_argument("--warmup", type=int,   default=2_000)
    ap.add_argument("--i_app",  type=float, default=8.0)
    ap.add_argument("--fake_cpg", type=int, default=1,
                    help="1 = use fake_step_chunk's back-to-back, no-gap bursts "
                         "instead of the real oscillator (temporary, to test a "
                         "continuous-activity CPG). Required for --n_cpg_neurons 18.")
    ap.add_argument("--n_cpg_neurons", type=int, default=6,
                    choices=sorted(CPG_W_BY_N),
                    help="CPG size: selects the coupling matrix from CPG_W_BY_N and "
                         "sets the input width. 6 = one neuron per leg; 18 = one per "
                         "joint (needs --fake_cpg 1, its matrix is a placeholder), "
                         "which narrows the window a timing unit is confined to from "
                         "1/6 to 1/18 of a cycle and lengthens the fake-CPG period, so "
                         "every period-derived default moves.")

    # Tunable CPG parameters, generated from CPG_PARAMS (see its comment); the two to reach for
    # are --cpg_vth_fb and --cpg_refrac_main.
    for _name, (_default, _help) in CPG_PARAMS.items():
        ap.add_argument(f"--cpg_{_name}", type=type(_default),
                        default=_default,
                        help=f"[CPG] {_help} Default {_default!r}.")

    # gait tables
    ap.add_argument("--gaits_dir", type=str, default="../gaits",
                    help="Folder of {name}.csv gait tables, resolved as "
                         "this_file_dir/<gaits_dir> (a sibling of this script's folder "
                         "by default).")
    ap.add_argument("--gaits", type=str, nargs="*", default=None,
                    help="CSV stems (no extension) to load, overriding the "
                         "--n_cpg_neurons default. (visualize.py's --gaits instead "
                         "filters which loaded gaits to plot.)")
    ap.add_argument("--leg_cols", type=str, default=None,
                    help="JSON list of equal-size column groups, one per leg; "
                         "overrides the default for --n_cpg_neurons (quadruped "
                         "LEG_COLS / HEXAPOD_LEG_COLS).")

    # architecture
    ap.add_argument("--arch", type=str, default="timing_grouped",
                    choices=["dense", "timing_grouped"],
                    help="dense: one fully connected network (the ablation baseline). "
                         "timing_grouped: CPG -> timing layer or direct CPG drive -> "
                         "disconnected sub-networks.")
    ap.add_argument("--timing_shape", type=str, default="per_joint",
                    choices=list(STRUCTURE_SHAPES),
                    help="[timing_grouped] Timing layer structure: per_leg = one LIF "
                         "per leg, per_joint = one per output column. Independent of "
                         "--n_cpg_neurons. Ignored under cpg / cpg_lif.")
    ap.add_argument("--decoder_shape", type=str, default=None,
                    choices=list(STRUCTURE_SHAPES),
                    help="[timing_grouped] Sub-network structure: per_leg (one per "
                         "leg, emitting its columns) or per_joint (one per column); "
                         "None = match --timing_shape. Together they fix the fan-in K: "
                         "a sub-network is fed by every timing unit whose columns "
                         "overlap its own (per_joint timing + per_leg decoders gives "
                         "K=3).")
    ap.add_argument("--readout_hidden", type=int, default=32,
                    help="[timing_grouped] Width of the analog readout membrane per "
                         "group. Not the output width: each unit has its own tau, so "
                         "the width is the temporal basis available to synthesiize the "
                         "waveform (a full Hg-wide w_read was 40%% of the model). "
                         "Sweep 3/8/32/128 to find where it binds.")
    ap.add_argument("--tau_readout_max", type=float, default=None,
                    help="[timing_grouped] Tau init ceiling for the readout membrane; "
                         "None = period/(2*pi), the corner frequency at the gait "
                         "fundamental. Much shorter than --tau_max on purpose: memo "
                         "RENDERS the current angle (local) rather than remembering "
                         "the phase (global), and a tau near the period passes only "
                         "~14%% of the fundamental, i.e. measures the cycle mean.")
    ap.add_argument("--timing_reset", type=str, default="zero",
                    choices=["zero", "subtract"],
                    help="[timing_grouped] Timing-layer reset. 'zero' matches the CPG "
                         "(v[spike]=0); 'subtract' leaves mem-thresh, which re-fires "
                         "on silent steps (ISIs 2,1,1,2,..., ~50%% more spikes per "
                         "burst than the CPG). A/B only. Ignored under cpg / cpg_lif.")
    ap.add_argument("--recon_cycles", type=float, default=4.0,
                    help="CPG cycles shown in each recon_<gait>.png (in cycles, so it "
                         "reads the same at any period).")
    ap.add_argument("--hidden_reset", type=str, default="zero",
                        choices=["zero", "subtract"],
                        help="[timing_grouped] Hidden-layer reset. 'zero' (default) "
                             "matches the CPG and caps each unit at one spike per timing "
                             "spike; with --bias_mode voltage and --film_mode gamma_only "
                             "it makes the sub-networks FULLY naturally gated (see "
                             "TimingGroupedSNN). 'subtract' turns injection magnitude into "
                             "a spike count and measured slightly better RMSE, but leaves "
                             "the loss partly insensitive to when timing spikes land, "
                             "which makes alignment seed-dependent.")
    ap.add_argument("--timing_slope", type=float, default=5.0,
                    help="[timing_grouped] Surrogate slope for the timing layer only "
                         "(--slope applies to the sub-networks). Gentler by default "
                         "because a dead timing unit cuts off a whole sub-network. "
                         "Ignored under cpg / cpg_lif.")
    ap.add_argument("--tau_timing_min", type=float, default=2.0)
    ap.add_argument("--tau_timing_max", type=float, default=None,
                    help="[timing_grouped] Tau init ceiling for the timing layer; None "
                         "= period/n_cpg_neurons, one inter-burst gap. A gap, not a "
                         "cycle: the CPG already bursts at every k/n_cpg phase, so a "
                         "unit only needs memory to bridge adjacent phases, and a long "
                         "tau smooths the burst structure away (modulation depth fell "
                         "17x from tau 10 to 352) and stops decaying between cycles. "
                         "Init range only (learnable). Ignored under cpg / cpg_lif.")
    ap.add_argument("--calibrate_gains", type=int, default=1,
                    help="[timing_grouped, --drive_source timing] 1 = before training, "
                         "bisect each timing unit's CPG->timing weight column until "
                         "its rate lands in a band set as a fraction of the CPG period "
                         "(1/8-1/6, or 1/6-1/3 with min_count), capped below the spike "
                         "ceiling. Forward passes only. 0 = skip.")
    ap.add_argument("--calibrate_per_gait", type=int, default=1,
                    help="[timing_grouped, needs --calibrate_gains 1] 1 = one gain per "
                         "(gait, unit) pair. 0 = the old one gain per unit, bisected "
                         "against the MINIMUM rate across gaits, which is a saturation "
                         "generator (one weak gait dragged the others 3-48x over band) "
                         "that training cannot undo; kept for A/B. Ignored under cpg / "
                         "cpg_lif.")
    ap.add_argument("--spike_objective", type=str, default="min_count",
                    choices=sorted(SPIKE_OBJECTIVES),
                    help="[timing_grouped] Shapes the timing layer's spike train. "
                         "'min_count' (default): a flat L1 cost per spike, with the "
                         "task loss deciding where spikes are worth spending (an "
                         "adaptive sampling clock); needs the output to depend on the "
                         "spike train alone (see NATURAL GATING). 'cpg_match': match "
                         "the CPG's rate and burst tightness; conflicts with "
                         "min_count's premise. 'none'. New strategies: subclass "
                         "SpikeObjective. Forced to 'none' under cpg / cpg_lif.")
    ap.add_argument("--spike_stats_lambda", type=float, default=0.0,
                    help="[timing_grouped] Weight on the spike objective. For "
                         "min_count it is the loss cost of a unit spiking every step "
                         "(at ~17%% duty, ~60 spk/cyc, the penalty is 8.5e-4, "
                         "comparable to a converged task loss). Raise if rates stay "
                         "high, lower if units collapse to silence. 0 disables. "
                         "Ignored under cpg / cpg_lif.")
    ap.add_argument("--min_count_floor", type=float, default=2.0,
                    help="[--spike_objective min_count] Spikes per cycle per (gait, "
                         "unit) below which the penalty reverses (a two-sided hinge), "
                         "so descent pushes the rate back UP instead of pruning to "
                         "silence, which the task loss cannot undo. Set it to the "
                         "fewest spikes a sub-network can use: the floor is an "
                         "attractor. 0 = original un-floored L1. Ignored under cpg / "
                         "cpg_lif.")
    ap.add_argument("--min_count_floor_weight", type=float, default=1.0,
                    help="[--spike_objective min_count] Weight of the below-floor half "
                         "of the hinge, relative to lam. 1.0 = symmetric; 0.0 = "
                         "one-sided (pruning stops at the floor but nothing pushes "
                         "back). Raise if units still die. Ignored under cpg / "
                         "cpg_lif.")
    ap.add_argument("--align_vel_tau", type=float, default=_ALIGN_VEL_TAU,
                    help="Credit horizon (steps) of the VELOCITY alignment score: a "
                         "spike gets the joint speed at its own step plus decaying "
                         "credit for faster motion up to ~this far ahead. Fixed, not "
                         "derived from learned taus, so runs are scored on one "
                         "yardstick.")
    ap.add_argument("--align_cor_tau", type=float, default=_ALIGN_COR_TAU,
                    help="Credit horizon (steps) of the CORNER (|theta''|) alignment "
                         "score: ~1 = credit for being at a corner, not anticipating "
                         "one.")
    ap.add_argument("--save_model_every", type=int, default=50,
                    help="Save epoch<N>.pt every N epochs regardless of val, alongside "
                         "best_model.pt (0 = off). Load with --ckpt epoch<N>.pt "
                         "downstream. The final epoch is saved only if a multiple of "
                         "N.")
    ap.add_argument("--lambda_hold", type=int, default=50,
                    help="[timing_grouped] Epochs at the start with every sparsity "
                         "penalty at EXACTLY zero, before --lambda_warmup's ramp: with "
                         "the s_gate init spread the PR penalty has a gradient from "
                         "step 1, which would amplify init noise while the task is "
                         "still finding which channels matter. 0 = ramp from epoch 1.")
    ap.add_argument("--lambda_warmup", type=int, default=20,
                    help="[timing_grouped] Epochs of linear ramp from 0 to full "
                         "strength, after --lambda_hold, for EVERY sparsity penalty "
                         "(--spike_stats_lambda and --drive_lambda). Both can lock in "
                         "structure before the task has said what it needs: PR is "
                         "rich-get-richer (the init leader wins), and an L1 spike cost "
                         "can prune a gated unit into silence it cannot recover from. "
                         "0 = no ramp.")
    ap.add_argument("--reinit_dead_after", type=int, default=2,
                    help="[timing_grouped] Rescue a failed (gait, timing unit) pair "
                         "after this many consecutive timing reports in the same "
                         "state. Per gait, matching w_in_gait: dead pairs get the "
                         "column re-rolled and the bias zeroed; saturated pairs get it "
                         "scaled by --sat_scale (a magnitude failure, so the direction "
                         "is kept); a unit dead in EVERY gait also gets its w1 slots "
                         "re-rolled. Neither failure self-corrects. 0 disables. "
                         "Ignored under cpg / cpg_lif.")
    ap.add_argument("--sat_scale", type=float, default=0.5,
                    help="[timing_grouped] Factor applied to a saturated (gait, unit) "
                         "column when --reinit_dead_after triggers; a badly saturated "
                         "unit may take several triggers. 1.0 leaves saturated units "
                         "alone. Ignored under cpg / cpg_lif.")
    ap.add_argument("--bias_mode", type=str, default="voltage",
                    choices=["current", "voltage", "none"],
                    help="[timing_grouped] Neuron excitability: 'current' = additive "
                         "input current; 'voltage' (default) = per-unit offset to the "
                         "spiking threshold, with no bias current in the spiking path; "
                         "'none'. Equivalent for an ungated layer with subtractive "
                         "reset; otherwise voltage is preferable since a threshold "
                         "never touches the membrane (so a silent sub-network provably "
                         "cannot spike) and stays out of the per-spike injection. See "
                         "BIAS MODES in TimingGroupedSNN.")
    ap.add_argument("--gate_mode", type=str, default="none",
                    choices=["none", "decay"],
                    help="[timing_grouped] 'none' (default): no explicit gate; with "
                         "--bias_mode voltage the sub-networks only get current on a "
                         "timing spike anyway (see NATURAL GATING in "
                         "TimingGroupedSNN). 'decay': input and sub-net spikes "
                         "explicitly gated (membranes still leak), for A/B. The "
                         "forward pass equals 'none' under the natural-gating "
                         "defaults, but the gradient differs: the gate is spk_t, so it "
                         "amplifies the gradient on spiking steps (up to 6x) and "
                         "zeroes it on silent ones, removing the 'fire HERE' signal; "
                         "it also caps each unit at one spike per timing spike.")
    ap.add_argument("--sub_film", type=str, default="both",
                    choices=["none", "l1", "l2", "both"],
                    help="[timing_grouped] Which sub-network layers get per-gait FiLM. "
                         "'both' (default). 'none' forces the sub-networks to infer "
                         "the gait only from their timing spikes (a clean test of "
                         "whether the timing layer encodes gait; the tables stay "
                         "allocated, ~19%% of params, but get no gradient). Run it "
                         "with --spike_stats_lambda 0 first: that penalty pins rate "
                         "and burst concentration to the CPG's, leaving phase the only "
                         "axis able to carry gait.")
    ap.add_argument("--film_mode", type=str, default="gamma_only",
                    choices=["gamma_only", "gamma_beta"],
                    help="Which FiLM halves apply. 'gamma_only' (default): "
                         "multiplicative only; gamma vanishes where the injection is "
                         "zero, so per-gait conditioning survives WITHOUT tonic drive "
                         "(what makes the sub-networks fully naturally gated; see "
                         "NATURAL GATING) and beats --sub_film none, since w1/w2 have "
                         "no gait axis. 'gamma_beta': also the additive beta, which is "
                         "tonic drive: it smooths the output but makes the loss "
                         "insensitive to timing-spike phase. The beta half is "
                         "allocated either way (checkpoint shapes unchanged).")
    ap.add_argument("--readout_gait_bias", type=int, default=1,
                    help="[timing_grouped] 1 = a learned per-gait constant CURRENT "
                         "into the readout membrane. Nothing else in the sub-networks "
                         "has a gait-dependent DC under --film_mode gamma_only, so "
                         "gaits differing in MEAN angle would otherwise buy the offset "
                         "with spike rate (costly, shrinks downward range, leaves "
                         "ripple). A current, not added to y, so it fades in over a "
                         "gait switch; memo never spikes, so it adds no tonic drive to "
                         "a spiking layer. 0 = off (what older configs reconstruct "
                         "as).")
    ap.add_argument("--drive_source", type=str, default="timing",
                    choices=["timing", "cpg", "cpg_lif"],
                    help="[timing_grouped] What drives the sub-networks. 'timing' "
                         "(default): the learned timing layer. 'cpg': skip it and "
                         "drive each sub-network straight off the CPG's own channels, "
                         "a fixed, evenly tiled phase basis (a timing unit cannot move "
                         "its phase freely: swapping channels needs a two-step weight "
                         "move that fires at both phases, which gradient descent will "
                         "not traverse). Per-gait selection is s_gate, sparsified by "
                         "--drive_lambda; needs --gate_mode none and forces "
                         "--spike_objective none. 'cpg_lif': as cpg plus a "
                         "fixed-threshold LIF per (sub-network, channel) between the "
                         "gain and the expansion, so u receives binary SPIKES: halving "
                         "s then drops spikes, which u can only partly make up for "
                         "(see --drive_penalty), so --drive_lambda acts on |s_gate| "
                         "alone, reporters print exact spike counts instead of "
                         "fanin90, and a low gain restores sub-burst placement.")
    ap.add_argument("--drive_lambda", type=float, default=0.0,
                    help="[--drive_source cpg / cpg_lif] Weight on the drive sparsity "
                         "term (--drive_penalty), on the --lambda_hold / "
                         "--lambda_warmup schedule. Because CPG channels are "
                         "phase-localiized, concentrating a sub-network's drive is a "
                         "temporal statement, not just a structural one. 'pr' is "
                         "rich-get-richer: a weight that dominates the task gradient "
                         "lets init pick the winner, so start small. 0 = off. (Was "
                         "--drive_l1.)")
    ap.add_argument("--drive_penalty", type=str, default="pr",
                    choices=["pr", "l1"],
                    help="[--drive_source cpg / cpg_lif] Sparsity term. 'pr' "
                         "(default): participation ratio, the effective number of "
                         "channels per sub-network (what drive_report prints as "
                         "fanin); scale-invariant. 'l1': mean of the drive. Under cpg "
                         "(drive = |s_gate|*||u||_1) it FAILED: the network shrank "
                         "every drive together and compensated via u and the hidden "
                         "thresholds (max channel share 0.08 at fanin ~15.8). Under "
                         "cpg_lif (drive = |s_gate|) it is UNTESTED and the escape is "
                         "bounded: s_gate can fall from 1.1 to ~0.25 (one late-burst "
                         "spike, penalty ~23%%) while u, FiLM gamma and the hidden "
                         "thresholds make up the charge, though the dropped spikes "
                         "lose within-burst timing and count; only below ~0.24 does a "
                         "channel go silent.")
    ap.add_argument("--synaptic", type=str, default="none",
                    choices=["none", "hidden", "all"],
                    help="Second-order synaptic filter (snnTorch Synaptic / Loihi "
                         "CUBA): syn = alpha*syn + injection, mem = beta*mem + syn. "
                         "'none' (default): injection goes straight into the membrane. "
                         "'hidden': sub-net layers 1 and 2. 'all': also the analog "
                         "readout. Spreading one injection over many steps shrinks the "
                         "output step per spike and keeps hidden membranes near "
                         "threshold for a window, so rates stay learnable upward, and "
                         "it decays to zero in true silence, so spike placement still "
                         "matters. One extra state tensor per layer.")
    ap.add_argument("--tau_syn_min", type=float, default=2.0,
                    help="[--synaptic] Tau init floor for the synaptic "
                         "current.")
    ap.add_argument("--tau_syn_max", type=float, default=None,
                    help="[--synaptic] Tau init ceiling for the synaptic current; None "
                         "= period/16 (a short window: the filter smooths the "
                         "injection, it is not the memory). A guess worth sweeping; "
                         "taus stay learnable.")
    ap.add_argument("--sub_ln", type=str, default="l2",
                    choices=["none", "l1", "l2", "both"],
                    help="[timing_grouped] Which sub-network layers get LayerNorm (no "
                         "affine; FiLM supplies it); never the timing layer. Default "
                         "l2, not both: layer 1's only input is one binary channel, so "
                         "its pre-activation has two possible values and LN would "
                         "normaliize the amplitude away, leaving FiLM gamma to restore "
                         "it.")

    ap.add_argument("--hidden",     type=int,   default=128,
                    help="Hidden width: the TOTAL for --arch dense (H x H layers), PER "
                         "GROUP for timing_grouped (w2/w_read are (G, H, H)); at G=4, "
                         "hidden=128 is near dense hidden=256's parameter count.")
    ap.add_argument("--max_gaits",  type=int,   default=8,
                    help="Rows allocated in the FiLM embedding tables (only the first "
                         "n_gaits are used), so parameter shapes do not depend on the "
                         "gait count and checkpoints transfer between runs with "
                         "different numbers of gaits. Changing it does NOT; unused "
                         "rows are near-identity, so an added gait starts from a "
                         "sensible routing.")
    ap.add_argument("--tau_min",    type=float, default=2.0)
    ap.add_argument("--tau_max",    type=float, default=None,
                    help="Longest SUB-NETWORK membrane tau, in steps; None = the "
                         "measured CPG period rounded up to a multiple of 64 (256 at "
                         "N=4, 384 at N=6). The sub-networks must hold position across "
                         "a cycle, so this tracks the period; the timing layer's does "
                         "not.")
    ap.add_argument("--slope",      type=float, default=25.0)

    # training
    ap.add_argument("--epochs",           type=int,   default=100)
    ap.add_argument("--chunks_per_epoch", type=int,   default=40)
    ap.add_argument("--val_chunks",       type=int,   default=8)
    ap.add_argument("--bptt",             type=int,   default=None,
                    help="Gradient truncation horizon (not the receptive field: state "
                         "is carried and detached across chunks). None = the period "
                         "rounded up to a multiple of 64, ~one cycle. Sweep "
                         "128/256/512 at fixed batch*bptt; batch*bptt is the compute "
                         "budget, so a longer bptt costs proportionally more unless "
                         "batch drops.")
    ap.add_argument("--batch",            type=int,   default=128,
                    help="Stream heads per gradient step. Activation memory is ~12 * "
                         "batch * bptt * n_timing * hidden * 4 bytes (TBPTT keeps ~12 "
                         "(B,G,Hg) tensors per step): ~1.8 GiB at 128 and ~3.6 GiB at "
                         "256 for hexapod n_timing=6, bptt=384, hidden=128. batch*bptt "
                         "is the compute budget. The timestep loop is kernel-launch "
                         "bound, so a bigger batch is nearly free in wall-clock; "
                         "consider an LR rescale (sqrt rule) if raising it.")
    ap.add_argument("--lr",               type=float, default=4e-3,
                    help="Kept at 2e-3 after raising batch 32->128 so the "
                         "first benchmark was a clean comparison. Adam's "
                         "sqrt-scaling rule suggests ~4e-3 at batch 128 — "
                         "worth sweeping, judged on the |grad| column and "
                         "free-run RMSE rather than train loss.")
    ap.add_argument("--clip",             type=float, default=1.0,
                    help="Gradient-norm clip. Watch the |grad| column: if the "
                         "pre-clip norm sits far above this, clipping is "
                         "truncating most updates and stability is NOT "
                         "evidence the LR is well chosen.")
    ap.add_argument("--switch_min",       type=int,   default=600)
    ap.add_argument("--switch_max",       type=int,   default=3000)
    ap.add_argument("--settle",           type=int,   default=100,
                    help="Steps after a gait switch counted as 'post-switch'.")
    ap.add_argument("--val_frac",         type=float, default=0.15)
    ap.add_argument("--phase_zero",       type=float, default=0.0,
                    help="Global rotation of gait-table row 0 relative to "
                         "neuron-0 burst onset, in cycles.")

    # misc
    ap.add_argument("--freeze_blocks", type=str, default="",
                    help="Comma-separated grad_blocks names to freeze "
                         "(requires_grad=False), e.g. "
                         "'sub_l1,sub_l2,sub_film'. Empty = train everything.")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--timing_log_every", type=int, default=10,
                    help="[timing_grouped] Epoch cadence for the timing-layer "
                         "firing report. Slower than --log_every because it "
                         "prints one line per gait; the compute is "
                         "negligible (timing units only, batch 1).")
    ap.add_argument("--export_onnx", type=int, default=0,
                    help="1 = export a single-step ONNX graph (cpg_lif_snn_step.onnx) "
                         "and check it against onnxruntime if installed. Default 0: no "
                         "ONNX code runs or is imported, so environments without onnx "
                         "/ onnxscript / onnxruntime work.")
    ap.add_argument("--visualize", type=int, default=1,
                    help="1 = run visualize.py's timing-layer analysis at the "
                         "end of training, writing into this run's own output "
                         "folders. 0 = skip (run visualize.py by hand later).")
    ap.add_argument("--dry_run",   action="store_true",
                    help="Build data + diagnostics, skip training.")
    ap.add_argument("--out_dir",   type=str, default="",
                    help="Resolved as outputs/<out_dir> — e.g. --out_dir "
                         "test1 writes to outputs/test1. Default '' means "
                         "outputs/ itself.")
    return ap
