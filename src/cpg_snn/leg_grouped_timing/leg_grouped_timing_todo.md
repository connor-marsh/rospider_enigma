# Leg-grouped timing architecture — open work and context

Successor to `architecture_change_todo.md`, which is closed out. That file
tracked *building* the leg-grouped timing architecture; this one tracks making
it **work better than the ungated baseline**, which it currently does not.

Everything below is written to be read cold. Section 1 is the state of the
code, section 2 is the central unsolved problem, section 3 is ranked open work,
and **section 4 is negative results — read it before proposing anything, it
records four plausible ideas that were tried and measured as worse.**

---

## 0. One-paragraph summary

CPG spikes drive a small **timing layer** (one LIF neuron per leg or per
joint), which drives *N* fully disconnected **sub-networks** ("Angle Decoder"
in the diagrams), one per timing neuron, each emitting that group's joint
angles. The point is that the timing layer learns the *rhythm* in a few hundred
parameters and the sub-networks learn *angles* given a clean phase reference.
It trains, it separates gaits, and dead neurons are solved. The unsolved problem
is that **event-gating the sub-networks — the thing that makes spike timing
matter at all — costs roughly 2.5–3× in free-run RMSE** (8–10 gated vs 2–4
ungated, in pulse-width units), and every attempt so far to recover that has
failed.

---

## 1. Where the code is

### Files

| file | role |
|---|---|
| `train.py` | CPG, model, training, and the shared helpers everything else imports (`cfg_get`, `build_model_from_cfg`, `load_run`, `out_path`/`in_path`, `OUT_ROUTES`, gait loading) |
| `visualize_timing.py` | post-run timing analysis. Importable (`run_visualization`) and standalone. `train.py` runs it automatically at the end unless `--visualize 0` |
| `run_inference.py` | robot inference in PyTorch (no ONNX), ROS2 `ServosPosition`. Gait switching via `--gait_switch none/schedule/keyboard` (gesture/joystick raise `NotImplementedError`) |
| `live_visualization.py` | real-time matplotlib view during inference (`--viz`): CPG spike raster, network schematic with spikes propagating, per-joint-type scrolling traces, gait-switch banner |


### Output layout

`--out_dir test_run` → `THIS_FILE_DIR/outputs/test_run/`, routed by
`OUT_ROUTES` in `train.py` (the single source of truth):

```
outputs/test_run/
    metrics.csv  training_curves.png  transition.png  rmse_heatmap.png
    model/              best_model.pt, cpg_lif_snn_config.json,
                        cpg_lif_snn_step.onnx(.data)
    misc_info/          alignment_summary, burst_threshold, cpg_raster,
                        routing_matrices, tau_distributions, timing_summary.json
    recons/  timing_alignments/  membrane_waveforms/  phase_folds/
```

`in_path()` falls back to the loose location, so pre-migration runs still load.

### The model (`TimingGroupedSNN`)

```
CPG spikes ──per-gait W──> timing LIF ──gate──> N × [ LIF → LIF → analog readout ] ──> joints
```

- **CPG→timing** is `w_in_gait`, an `Embedding(max_gaits, n_cpg*n_timing)` — a
  free per-gait routing matrix, nothing shared. Plus a per-gait bias/threshold.
  There is **no FiLM on the timing layer**: its gamma was provably absorbable
  into a free per-gait matrix (verified to 1e-16), and its beta survives as the
  per-gait bias.
- **Sub-networks** are block-diagonal `(G, Hg, Hg)`, no cross-talk. Readout is
  a narrower analog membrane bank, `readout_hidden` wide (32), whose per-unit
  taus make it a bank of low-pass filters summed by `w_out`.
- **FiLM** (`film1`, `film2`) conditions the sub-networks per gait. This is
  ~28% of the parameters and is a prime suspect (see §2).

### Flags that matter, and current defaults

| flag | default | note |
|---|---|---|
| `--arch` | `timing_grouped` | `dense` still works for A/B |
| `--n_cpg_neurons` | 4 | 6 for hexapod; picks the gait CSV set automatically |
| `--n_timing` | `n_legs` | set `18` for per-joint groups on the hexapod |
| `--hidden` | 128 | **per group** |
| `--readout_hidden` | 32 | see §4 for why not 3 |
| `--gate_mode` | `freeze` | `none` / `decay` / `freeze` — **the central variable, see §2** |
| `--bias_mode` | `voltage` | `current` / `voltage` / `none` |
| `--timing_reset` | `zero` | matches the CPG; `subtract` runs away |
| `--sub_ln` | `l2` | LN on layer 1 is expressively vestigial under gating |
| `--spike_objective` | `min_count` | `cpg_match` / `min_count` / `none`, strategy pattern |
| `--spike_stats_lambda` | 0.005 | **set 0 when testing gating changes** |
| `--fake_cpg` | 1 | back-to-back bursts, no inter-burst gaps |
| `--tau_max`, `--bptt`, `--tau_timing_max`, `--tau_readout_max` | auto | derived from the *measured* CPG period |
| `--freeze_blocks` | "" | e.g. `sub_l1,sub_l2,sub_film` |
| `--visualize` | 1 | run `visualize_timing.py` at the end |

---

## 2. The central problem

### The invariance that started it

Without gating, **the task loss is exactly invariant to the timing layer's
burst phase.** A sub-network whose taus span a cycle has a complete
"time-since-burst" basis, so shifting the burst by any amount is absorbed by
relearning the waveform offset. Alignment therefore *cannot* emerge from
end-to-end training — this is a gauge freedom, not weak pressure that better
hyperparameters would sharpen.

Gating breaks the invariance: if a sub-network can only change state when its
timing neuron spikes, spike placement becomes load-bearing in the task loss.
That is also exactly the event-driven property the neuromorphic goal wants, so
**alignment and efficiency are the same requirement.**

### What gating actually cost

| gate_mode | behaviour | free-run RMSE |
|---|---|---|
| `none` | biases flow every step, membranes always advance | **2–4** |
| `decay` | input+spikes gated, membranes still leak | 8–10 |
| `freeze` | nothing moves without a spike; `beta**Δt` at the next update | worse still |

`freeze` is exact: state at each update is identical to per-step decay
(verified to 2.2e-16), so it is a pure zero-order hold on the *output*, and it
is the form that permits genuinely sparse updates at deployment.

### Why it is worse — the jerk

Under gating the output jerks hard at each spike and drifts smoothly between.
Diagnosis, in order of confidence:

1. **The injection must be large.** To hold an output level `M` with decay `τ`
   and inter-spike gap `D`, the injection must supply `M·(1−exp(−D/τ))`, and
   that arrives in a **single timestep**. At 12 spikes/cycle that is ~40% of
   the signal level, instantaneously. A first-order leaky integrator's impulse
   response jumps to full amplitude at t=0 *by definition*, so this is
   arithmetic, not a learning failure. **No amount of training or
   regularisation fixes it — the filter order has to change.**
2. **The injection is near-stereotyped.** `cur1 = gate·(w1 + b1)` has no
   dependence on `t`, membrane, or input, so every spike delivers nearly the
   same kick; all variation comes from membrane state. Measured: ~4 of 38
   active layer-1 units differ between any two spikes. This is a *partial*
   cost, not a ceiling — see §4.
3. **The readout can coast too well.** With `Ho=32` and `tau≤56`, a
   sum-of-exponentials readout holds <1° error for **183 steps** — half a
   cycle — needing only ~2 spikes/cycle. So the L1 spike penalty finds a
   legitimately cheap low-spike solution, and the network then jerks between
   the sparse injections it chose.

### The recommended fix, not yet built

**Second-order synaptic filter (CUBA / `Synaptic` in snnTorch).** The spike
lands on a synaptic current that itself decays, driving the membrane:

```
I    = exp(-Δt/τ_syn)·I  + injection      # gated; spike lands here
memo = exp(-Δt/τ_mem)·memo + I
```

The impulse response becomes a difference of exponentials — **exactly zero at
t=0**, smooth rise, then decay. Measured output ripple relative to its own
mean, `τ_mem=19`:

| spikes/cycle | 1st order (current) | 2nd, τ_s=5 | 2nd, τ_s=10 |
|---|---|---|---|
| 20 | 26.4% | 4.8% | **2.7%** |
| 12 | 47.3% | 12.4% | **6.4%** |
| 6 | 99.6% | 42.3% | **24.2%** |

Why it fits the constraints: keeps exponential coasting, keeps event-based
operation (`β^Δt` works on both variables), keeps alignment pressure (output
still cannot move without a spike), and costs one state tensor `(B, G, Ho)` if
applied to the readout only. Make `τ_syn` **learnable per-unit** like the other
taus — that gives a bank of second-order filters with varied `(τ_syn, τ_mem)`
pairs, strictly richer than the current first-order bank. Caveat: it introduces
a lag of ~`τ_syn` before a spike's peak effect, so start small (3–8).

### The other principled fix (user's parallel work)

**Delta / increment encoding on the output**: each output spike moves the angle
by a fixed increment, so the output is smooth by construction and spike rate
becomes naturally proportional to `|dθ/dt|`. Same underlying idea as
second-order filtering — make a spike's effect on the output *incremental*
rather than *impulsive*. Being pursued in a separate chat; not mutually
exclusive with the above.

### A hard ceiling to know about

With `--fake_cpg`, period is **120 steps with 60 CPG spikes** (a spike on every
even step, 50% duty). A timing unit needs input to fire and, with
`timing_reset="zero"`, fires at most once per input — so **60 spikes/cycle is
the ceiling.** Zero-order-hold sampling needs ~39/cycle for 1.5° and ~59 for
1.0°, i.e. 66–99% of that ceiling. **High accuracy and leg-specific timing are
therefore in direct tension**: at those rates every timing neuron is firing at
nearly every opportunity, leaving no room to be *differently* timed.
`refrac_main=0` in `fake_step_chunk` would give 10 consecutive spikes per
burst, period 60, 100% duty, removing the ceiling — not currently a CLI flag.
Separately, the coworker is retuning the real CPG toward continuous activity
with smaller gaps, which addresses the same thing.

---

## 3. Open work, ranked

**A. Second-order synaptic readout.** §2. The highest-value untried idea, with
a measured 4–10× ripple reduction at the same spike rate. Readout only to
start; one state tensor.

**B. Multiple timing neurons per sub-network** (`n_timing = k·n_legs`). Makes
`cur1` depend on *which* channel fired rather than only *that* one did, which
is the direct attack on the stereotyped-injection cost. Needs
`build_group_cols` generalised to k-per-group. **Note `--n_timing 18` already
gives per-joint groups with no code change** and is the cheap version of this —
worth running first. Memory: 3× activations, so pair with `--batch 48`.

**C. `--sub_film none`** (already implemented, untested). Forces the
sub-networks to infer gait solely from their timing neuron's spike train — the
clean test of whether the timing layer is really encoding gait, and it would
free ~28% of parameters. **Run with `--spike_stats_lambda 0`**: the penalty
pins rate and burst concentration to the CPG's values for every gait, leaving
phase as the only axis able to carry gait, and if a leg's phase is legitimately
the same in two gaits while its waveform differs, the two constraints are
jointly unsatisfiable.

**D. Gait conditioning without a per-gait weight table** (was item 11).
`w_in_gait` gives each gait a free matrix: `n_gaits` appears in a parameter
shape, nothing transfers between gaits, and unused `max_gaits` rows are random
noise rather than inert identities. It *works* — phases separate — but it is
not the end state. Options, cheapest first: shared base + zero-init per-gait
delta (`w_in = w_shared + w_delta[gait]`, weight-decayed); per-gait soft
permutation over CPG neurons (expresses "same drive strengths, different
footfall order", which is the thing that actually varies); larger gate init
noise; and the hybrid discrete/continuous scheme (old item 2), which is
probably the right destination.

**E. IK-derived footfall phases** (was item 13). Joint-angle fundamental phase
is **not** footfall phase — the CSVs are angles, and some columns rise and fall
continuously with no distinguishable start. Forward kinematics on the joint
angles gives each foot's trajectory, hence real touchdown phases. Two distinct
uses: *diagnostic* (honest alignment residual, successor to the
0.116–0.133-cycle measurement that killed the old fixed routing) and
*supervision* (label-free, since targets are computed from tables that already
exist; would likely fix D outright). Prefer an auxiliary loss that stays on
over a hard two-stage freeze, to keep it end-to-end. **Blocked for supervision
by the phase-quantisation ceiling**: with `timing_reset="zero"` a timing unit's
membrane peaks at its burst end then only decays, so achievable phases are
pinned to the CPG's burst phases (measured at N=6: {0.000, 0.148, 0.321,
0.497, 0.670, 0.821}, i.e. the k/6 grid) ±~0.05 cycle. Diagnostic use is
unblocked.

**F. Two-burst-per-cycle in the concentration term** (was item 10).
`spike_stats_penalty` scores burst tightness with the magnitude of the *first*
Fourier component, so two bursts at opposite phases cancel to R≈0 and are
punished as hard as uniform smear. Correct for every gait currently in use
(each leg swings once per cycle) but baked in, not configurable. Fix: also
score `k=2` and target `max(R_1, R_2)` — one extra einsum pair.

**G. Cross-talk variant** (was item 3b). Every sub-network sees all `n_timing`
spikes: replace layer 1's outer product with
`einsum("bt,tgh->bgh", spk_t, w1_cross)`. Deliberately left unbuilt so the
comparison is between two committed variants.

**H. Named legs in the visualisations** (was item 9). `HEXAPOD_LEG_NAMES =
["LF","LM","LR","RF","RM","RR"]` exists and nothing reads it. Needs a
quadruped equivalent, threading through config as `leg_names`, a `--leg_names`
companion to `--leg_cols`, and swapping the numeric labels in
`visualize_timing.py` and `live_visualization.py`.

**I. `max_gaits` waste** (was item 1). At `max_gaits=16` with 2 gaits in use,
14 of 16 sub-network FiLM rows get no gradient — 27.7% of the model idle. Left
at 16 because changing it invalidates checkpoints and 16 is the hexapod gait
count. Resolves itself if D or C lands.

**J. Dynamo recompile limit.** `train.py` still has no
`torch._dynamo.config.recompile_limit = 64`. Currently ~3 guard sets against a
default limit of 8; past the limit Dynamo silently reverts to eager, which is
~3× slower by the project's own benchmarks. Cheap insurance.

**K. Real holdout.** The `val_frac` time split is a phase-shifted duplicate of
train, so plain val is not a decision criterion. Judge on **free-run RMSE** and
`Val(post-sw)`. A genuine holdout would hold out
`(gait_from → gait_to, phase_at_switch)` combinations.

**Left behind in `architecture_change_todo.md`** (both Low, both still valid,
neither blocking anything): a parallel/associative scan over timesteps to
replace the Python loop, and moving the timing layer out of the torch class so
it can run as plain numpy on hardware. Read them there if either becomes
relevant.

---

## 4. Negative results — do not redo these

| tried | result | why it matters |
|---|---|---|
| **Shared LIF router** with per-gait FiLM gate replacing `w_in_gait` | Timing phases came out near-identical across gaits; routing matrices near-copies | Expressiveness was verified *first* (400 random gates → 248 distinct routings, incl. many-to-one/tripod), so this was not a capacity failure. Likely causes: `film1`+`film2` carry ~3,072 params/gait vs the gate's ~44 and sit 1–2 spiking layers from the loss instead of 3–4 (each layer attenuates by the surrogate derivative, ~1e-3 at slope 25); and shared weights average conflicting per-gait gradients. Reverted to the per-gait table. |
| **`--freeze_blocks sub_l1,sub_l2,sub_film`** | **~2× worse** train loss (0.00461 vs 0.00234 @ep20) | Those 151k params (85% of the model) do real work. Kills the "linear readout on a random reservoir" hypothesis. |
| **`--slope 5`** (widen the sub-network surrogate) | **~22% worse** train, 26% worse val, and *widening* gaps | Did raise `sub_l1` gradient 10.5× and `sub_l2` 3.2× as predicted, so gradient starvation was **not** the bottleneck. Under Adam magnitude is normalised away; what matters is direction *quality*, and a wider surrogate is a worse approximation to the true derivative. Keep `--slope 25`. |
| **LR sweep 2e-3 → 8e-3 (4×)** | Near-identical curves | Adam's update is scale-invariant in the gradient, so `\|grad\|` was never an LR diagnostic. `\|upd\|` (per-epoch ‖Δθ‖) is now logged instead. Do not re-sweep. |
| **Two-sided R** in the concentration term (to stop over-tight bursts) | Contributes ~1.6e-07 to the loss | R is far too insensitive at high concentration: spacing-1 gives R=0.99869 vs spacing-2's 0.99475. The real cause was **subtractive reset** on the timing layer leaving residual above threshold, producing ISIs of 2,1,1,2,1,1… and 15 spikes/burst instead of 10. `timing_reset="zero"` fixed it structurally: clean spacing-2, 10 spikes, R=0.99475 — *exactly* the CPG's own value. |
| **`readout_hidden = 3`** (one unit per output column) | Rejected on analysis, not run | `memo` is where temporal filtering happens and each unit has its own tau, so its width is the size of the temporal *basis*, not the output width. Filter-then-combine ≠ combine-then-filter when taus differ, so projecting to 3 before the membrane leaves only 3 distinct (projection, tau) pairs for the whole group. 32 is the working default; sweep 8/32/128. |
| **`tau_readout_max` ≈ period** (intuition: "it's the output, it should span the cycle") | Backwards | A leaky integrator with tau near the period passes only ~14% of the gait fundamental relative to DC — it measures the cycle *mean*. `memo`'s job is to *render* the current value (local); `mem1`/`mem2` do the *remembering* (global). Hence `period/(2π)`, which reproduces the value hardcoded in the original `StatefulSNN` (254/2π = 40.4 vs 40.0) — that number was right for the quadruped and simply did not generalise. **Note this inverts under gating**: with sparse injections, longer readout tau reduces the required refresh and hence the jerk. |
| **LayerNorm on sub-net layer 1** | Expressively vestigial under gating | `cur1` never varies (one constant vector), so LN maps one constant to another and a following per-unit affine (FiLM) can reproduce any target exactly (verified error 0). LN on layer 2 *does* real work — its input `spk1` genuinely varies. Hence `--sub_ln l2`. |
| **Learnable threshold with the reset subtracting it** | Not equivalent to a bias current (60 vs 75 spikes on the same input), and invites runaway | A unit that learns a low threshold would fire easily *and* reset by almost nothing. `bias_mode="voltage"` compares against `thresh − v` but resets by the **fixed** `thresh`, which is the variant that *is* exactly equivalent to a bias current for an ungated layer with subtractive reset. |

### Training-curve shape, so it is not re-diagnosed

Loss falls fast for ~20 epochs then improves slowly but **does not plateau**:
train loss fits `epoch^-0.43` (R²=0.985) with a best-fit asymptote of *zero*.
But 20× the compute buys only ~3.5× lower train loss, and
**`val_post_switch` has exponent −0.11 with R²=0.43** — i.e. no real trend. So
the metric still improving is the one that matters least. Read that as an
architecture ceiling, not undertraining. For architecture A/B, ~30 epochs is
enough; save long runs for after the architecture stops changing.

Also: `T_max` for the cosine schedule is tied to `--epochs`, so "just run
longer" changes the LR trajectory too. Set `--epochs` to the real target or
switch to a constant LR for that one experiment.

---

## 5. Diagnostics already available

- **`metrics.csv`** per run, flushed every epoch (survives Ctrl+C): epoch,
  train, val, val_post_switch, lr, grad_norm, **update_norm**, spike_penalty,
  sec, best, plus `grad_<block>` and `upd_<block>` per parameter block.
- **`|upd|` not `|grad|`.** Adam normalises per parameter, so a small gradient
  says nothing about whether the model is moving. `u/g` per block is the
  diagnostic: a block with 1% of the signal and 50% of the movement is
  diffusing. Measured at epoch 20 — `readout` g=86% u=23%, `sub_l2` g=4.6%
  u=50.5% (u/g=11), `sub_film` g=1.1% u=19.3% (u/g=17). Movement per parameter
  is nearly uniform (2.1e-3…6.4e-3) while gradient per parameter spans 100×.
  **That is Adam working as designed, and given the freeze result above it is
  useful work, not noise — do not read high u/g as pathology.**
- **`timing_report`** each `--timing_log_every` epochs: per gait, per timing
  neuron, spikes/cycle, circular mean phase, **R** (concentration), plus a
  **phase-separation** number (max deviation on the unit circle; below ~0.1
  means gaits are not being distinguished). Rate and R being identical across
  gaits is *expected* — the spike objective pins both — so **phase is the only
  statistic free to differentiate.**
- **`calibrate_gains`** pre-training: bisects each timing unit's `w_in_gait`
  column into a CPG-derived spikes-per-cycle band. Reports units it cannot fix,
  which means net-negative input current (gain cannot fix a sign).
- **`visualize_timing.py`**: `timing_alignment_*` (time domain, spikes as a rug
  under each joint trace), `phase_fold_*` (folded over ~30 cycles — answers "is
  the alignment *consistent*", which a 3-cycle window cannot),
  `alignment_summary`, `routing_matrices` (measured by probing one CPG neuron
  at a time, not read off a weight), `tau_distributions`.
- **`live_visualization.py`** during inference.

## 6. Measured constants worth reusing

| quantity | value |
|---|---|
| real CPG period, N=6 / N=4 | 352 / 254 steps |
| `fake_cpg` period, N=6 | 120 steps, 60 spikes/cycle (50% duty) |
| CPG burst | 10 spikes at spacing 2 (`refrac_main=1`), R = 0.99475 |
| N=6 burst phases | {0.000, 0.148, 0.321, 0.497, 0.670, 0.821} = the k/6 grid |
| tripod / ripple required phases | {0, 0.5} and {0, ⅓, ⅔} — **both land exactly on the CPG grid**, so the timing layer's task is *selection*, not temporal transformation |
| `from_fb_weight` | fixed at −1e6, works for N=3/4/6 |
| readout coast (Ho=32, tau≤56) | 183 steps under 1° |
| activation memory | ≈ `12·batch·bptt·n_timing·hidden·4` bytes (1.81 GB at 128/384/6/128) |
| param count, hexapod h=128 Ho=32 mg=16 | ~177k: `sub_l2` 56%, `sub_film` 28%, `readout` 14%, `timing` 0.4% |

