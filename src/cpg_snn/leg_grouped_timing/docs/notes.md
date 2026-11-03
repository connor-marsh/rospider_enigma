# NOTES: history, measurements and design rationale

This holds what was cut from the comments in `train.py`, `visualize_timing.py` and
`inference.py` when they were trimmed: why things are the way they are, what was tried
and reverted, and the numbers behind decisions. The code comments keep only the
non-obvious *why*; the story and the evidence live here.

**Provenance tags** (read the numbers accordingly):
- **[code]** was a comment in the source before the cut; measured in an earlier session.
- **[session]** derived in the conversation that produced this file, mostly *toy
  simulations*, not trained networks. Treat as direction, not measurement.
- **[your runs]** observations you reported from real training runs.

---

## 1. How the architecture got here

- **CPG.** A 4-neuron BDF/`solve_ivp` conductance CPG was replaced by the discrete
  bursting-LIF CPG (no ODE solver, no chunk/batch mismatch, no ARM/x86 FPU divergence;
  one `step()` is one timestep). Measured on the original 4-neuron matrix at `i_app=8`:
  period ~254 steps, 10 spikes/burst, burst ~29 steps (duty ~11%), phase offsets
  0.000 / 0.248 / 0.500 / 0.748. [code]
- **No phase input.** The sin/cos phase channel was removed; the network sees only raw
  CPG spikes, so phase must be held in state by learnable log-uniform membrane taus. [code]
- **Recurrence removed.** Within-group recurrent connections (`rec1`/`rec2`, 32,768
  params = 45% of the model) were ablated: free-run RMSE and Val(post-switch) unchanged,
  slightly better step time and memory. [code]
- **Leg-grouped dense design removed.** Four groups driven by a per-gait permutation of
  CPG neurons. The permutation's phase residuals were 0.116-0.133 cycle for 3 of 4 gaits
  (~34 steps at period 254, longer than a ~29-step burst), so alignment only held for
  `wkF`; and since `w_cross` already spanned all neurons, routing was an init prior, not a
  capability. `--arch dense` stays as the fully connected ablation. [code]
- **`timing_grouped`**: CPG -> timing layer -> G disconnected sub-networks. The timing
  layer learns the rhythm in a few hundred parameters; sub-networks learn angles from a
  clean phase reference.
- **Gait-table layout.** For every gait, column j and j+4 share a circular phase offset
  (5/54 cycle in `wkF`, 5/39 in `wkL`/`wkR`, 15..17/22 in `bk`), so quadruped leg l is
  columns (l, l+4). Hexapod is leg-major, 3 per leg, LF/LM/LR/RF/RM/RR. [code]

### The shared router that was reverted [code]
One pair of weight matrices with a per-gait FiLM gate on a 16-unit LIF hidden layer
(`W_eff(g) = W2 diag(gamma_g) W1`). It was expressive enough (randomized gates reach
hundreds of distinct routings, including the many-to-one patterns a tripod needs), yet
the learned timing phases came out essentially identical across gaits and the measured
routing matrices were near-copies. Two likely reasons:
1. *Competing capacity.* Sub-network FiLM tables carry ~3,072 params per gait against the
   router gate's ~44, and sit 1-2 spiking layers from the loss instead of 3-4. Each
   spiking layer attenuates gradient by the surrogate derivative `1/(slope*|x|+1)^2`
   (order 1e-3 at slope 25), so gradient descent put the gait knowledge where it was
   cheapest and the timing layer collapsed to a gait-independent clock.
2. *Gradient averaging.* With W1/W2 shared, gradients from gaits wanting different
   routings partially cancel.

The per-gait `Embedding` (`w_in_gait`) demonstrably separates gaits. Its costs:
`n_gaits` appears in a parameter shape, nothing transfers between gaits, per-gait
capacity grows linearly. A conditioning scheme that separates gaits without a per-gait
table is tracked in `architecture_change_todo.md`. Routing is deliberately not a
permutation: a tripod needs 3 legs at one phase and 3 at the opposite.

No FiLM on the timing layer: with a free per-gait `W_g`, `gamma_g*(xW_g+b)+beta_g ==
x(W_g gamma_g)+(gamma_g b+beta_g)` (verified numerically to 1e-16), so gamma is
absorbable; the per-gait bias is not, so the embedding supplies it. [code]

---

## 2. Bias modes, natural gating, and what the event gate does to gradients

- **Voltage vs current bias** are exactly equivalent for an ungated layer with
  subtractive reset (substitute `m' = mem - b/(1-beta)`; identical spike trains,
  verified numerically). Voltage is preferred because a bias current is tonic drive
  (flows every step) while a threshold offset never touches the membrane. The reset still
  subtracts the fixed `thresh`: resetting by the offset runs away, and is not the variant
  equivalent to a bias current (60 vs 75 spikes on the same input). [code]
- **No LayerNorm on the timing layer.** Over 6-16 units the mean *is* the signal ("some
  CPG neuron fired" is nearly all common mode), and LN(bias) becomes a fixed nonzero
  vector once the bias trains away from uniform, giving every unit tonic drive. A
  256-wide layer tolerates this; a 6-wide one does not. [code]
- **Natural gating** needs three leaks closed: bias current (`bias_mode voltage`), FiLM
  beta (`film_mode gamma_only`), subtractive reset (`hidden_reset zero`). The reset is the
  expensive one: subtraction turns injection *magnitude* into a spike *count* and measured
  slightly better RMSE. The trade taken is RMSE against a loss surface where alignment is
  determined rather than lucky: with a leak open, the loss is partly insensitive to *when*
  timing spikes land, so alignment is seed-dependent even when RMSE is good. [code]
- **Why explicit gating (`gate_mode decay`) trains differently** though its forward pass
  is identical under natural gating. The gate *is* `spk_t` and carries its surrogate
  gradient, so it multiplies `spk_t` in five extra places (cur1, spk1, cur2, spk2, curo).
  A path with k gate factors goes as `s^(1+k)`, derivative `(1+k)s^k`: up to 6x on a
  spiking step, and *exactly zero* on a silent step (both factors vanish), whereas the
  ungated `d(cur1)/d(spk_t) = w1` at every step. The silent-step term is the only route by
  which "this unit should have fired here" reaches the timing layer, so gating biases
  spike counts down twice over and makes dead timing units more likely. [code]
- Known cost of natural gating: a weaker "put the spike here" gradient. The suggested
  fix, a second-order synaptic filter, became `--synaptic`. [code]

---

## 3. Timing-layer health: calibration, floors, taus

- **Per-gait calibration.** One gain per unit bisected against the *minimum* rate across
  gaits is a saturation generator. Over 24 draws of the real init (w ~ N(`_W_IN_INIT`,
  `_W_IN_INIT`), 6 CPG inputs, 4 gaits) it left max-over-gaits rates of 5-48 spk/cyc
  against a [1, 5] band: typically 3-10x over, 20-48x when one gait's column summed near
  zero. Training cannot undo it: a unit at the ceiling sits where the surrogate is down
  1e2-1e3x (gradient ~1e-6), and Adam's scale-invariance means a 10,000x larger
  `--spike_stats_lambda` moved the gain 1.5% in a 4000-step simulation (rate stayed 39
  spk/cyc; the gain needed to move ~40x). [code]
- **Saturation threshold.** With the plain band (25-33% of the ceiling), min-over-gaits
  calibration left a (gait, unit) pair at 51.5 of a 60 ceiling (86%); a 0.9 threshold
  would have said nothing, hence `_SAT_FRAC`. [code]
- **Band as a fraction of the period, not of the ceiling.** `--cpg_refrac_main`
  lengthens the period without changing `n_cpg * cpg_rate`, so a ceiling fraction would
  smear the same spike count over more steps without the band noticing. At the default
  (refrac 1, ISI 2) period == 2 * ceiling, giving 15-20 spk/cyc (plain) and 20-40
  (min_count) at n_cpg=6. The saturation cap stays on the ceiling. [code]
- **min_count floor.** Without a floor the penalty prunes to zero and a silent unit is
  unrecoverable through the task loss (zero gradient to the `w1` slots it feeds). One run
  pruned several sub-networks to 2-5 well-placed spikes/cycle and others to 0. The floor
  term still reaches a silent unit through the surrogate (~0.03 at `timing_slope=5`). A
  linear `lam*(duty - floor)` has the same gradient as `lam*duty`, so only a hinge works. [code]
- **`--spike_stats_lambda` sizing.** At ~17% duty (~60 spk/cyc at period 352) the
  penalty is 8.5e-4, comparable to a converged task loss of ~1e-3. [code]
- **`tau_timing_max` = one inter-burst gap** (59 steps at N=6/P=352, 63.5 at N=4/P=254).
  Within-cycle modulation depth falls from 1.18 at tau=10 to 0.068 at tau=352 (17x), and
  near the period the membrane stops decaying between cycles. [code]
- **`tau_readout_max` = period/(2*pi).** A tau near the period passes ~14% of the
  fundamental (measures the cycle mean); at the corner passage is 0.71. It reproduces the
  value hardcoded in the original `DenseSNN` (254/2*pi = 40.4 vs 40.0), which did not
  generalize to 352 (where it should be ~56). [code]
- **`timing_reset subtract`** re-fires on silent steps (ISIs 2,1,1,2,..., ~50% more
  spikes per burst) and caused the observed over-firing and too-tight bursts. [code]
- **Readout width `Ho`.** `Ho == Hg` was ~40% of the model at hidden=128. `Ho` is the size
  of the temporal basis (each unit has its own tau), not the output width. [code]
- **`--sub_film none`** leaves the FiLM tables allocated (~19% of params) with no gradient. [code]
- **Readout gait bias** is a current into `memo` (not added to `y`) so it fades in over
  the readout tau. `memo` leaks: a level held at ~15 spk/cyc with tau_o 60 sags ~12%
  between spikes. [code]
- **Synaptic filter** (snnTorch `Synaptic` / Loihi CUBA): the step on the spike is smaller
  by the filter's peak-to-initial ratio, ~3.9x at alpha = beta = 0.9. `--tau_syn_max` =
  period/16 is a guess. [code]
- **CPG facts.** `from_fb_weight = -1e6` confirmed for N=3/4/6. R=1.0 is unsatisfiable
  (10 spikes at 2-step spacing in a 352-step cycle top out near 0.995). N=18 is a
  zeros-matrix placeholder usable only with `--fake_cpg`; it narrows the window a timing
  unit is confined to from 1/6 to 1/18 of a cycle. `--tmax 150k` was ~590 copies of the
  254-step cycle. [code]

---

## 4. The `cpg` / `cpg_lif` drive

- **Factorization.** `s_gate` (per gait) times `u` (gait-shared) replaces `w_in_gait` +
  `w1`: rank 1 over gait per (g, c), ~7.5x fewer parameters than a full per-gait `w1` at
  hexapod sizes. `u` *is* the layer-1 weight matrix (`w1` under another name). [code]
- **The `C = n_neurons` shadowing bug.** The `cpg` branch rebound `C` (output columns)
  to the CPG width, so `w_out` became `(G, Ho, n_cpg)` and `y_grp.flatten(1)` indexed with
  `out_perm` selected only sub-network 0's columns: every joint was driven by sub-network 0
  and the other 17 never reached the loss. Any checkpoint whose parameter count matches
  `(G, Ho, n_cpg)` was trained that way and is void. Finite-difference checks confirmed g0
  was the only connected sub-network. The `assert C == len(group_cols[0])` guards it. [code/session]
- **`u` init.** Dense at `_W1_INIT` with no `1/sqrt(C)` (it would under-drive layer 1 by
  4.2x, since channels never co-fire). Sparse is worse: at fan-in 2 with random signs 25%
  of units have no positive weight (measured 32% of 128); dense over 18 makes it 2^-18. [code]
- **`cpg_lif` design.**
  - *Fixed threshold:* a learnable one reopens the escape completely (at v=0.5 a halved s
    recovers its spike rate exactly).
  - *Zero reset:* makes a sub-network's units mutually exclusive, so they collapse to one
    effective timing neuron for reporting.
  - *Fixed tau = 2 burst widths:* residual after a cycle is `exp(-n_cpg/2)` (1.2e-4 at
    n_cpg 18, independent of burst width); retention across one ISI ~0.9. The gain needed
    to fire on spike 1..5 of a 5-spike burst is 1.00 / 0.52 / 0.37 / 0.29 / 0.24.
  - Spike count is a staircase in s (flat over ~94% of its range), so the penalty acts on
    the smooth `|s_gate|`. [code]
- **`s_gate` init.** Exactly 1.0 makes the drive exactly uniform, PR's maximum: `dPR/ds`
  is 0.0 there and 3.4e-2 at a 1% spread. Mean 1.1, not 1.0: with the strict `>`, s=1.0
  fires on spikes 2 and 4 only (2/5) while s=1.01 fires on all 5. Small spread because PR
  is rich-get-richer: in a toy, a wide N(0.8, 0.4) init let the init's strongest channel
  win 90% of the time against 52% for a tight one. Under `cpg`, `u`'s ~9% row-norm spread
  broke the tie by accident. [code]
- **L1 under `cpg` failed**: the network shrank every drive together and compensated via
  `u` and the hidden thresholds, giving a max channel share of 0.08 at fanin ~15.8 (a
  uniform 18-way split at tiny magnitude); its per-channel gradient `||u||_1` varies only
  ~7% across 128 entries. [code]
- **L1 under `cpg_lif` is untested, and the escape is bounded, not closed** [session,
  toy; `tau_drive` = 20, 5-spike bursts, ISI 2]:

  | gain s | spikes/burst | fires on spike # | `u` gain to match charge | `|s|` penalty |
  |---|---|---|---|---|
  | 1.10 | 5 | 1-5 | x1.0 | 100% |
  | 0.80 / 0.60 | 2 | 2, 4 | x2.5 | 73% / 55% |
  | 0.45 | 1 | 3 | x5.0 | 41% |
  | 0.25 | 1 | 5 | x5.0 | 23% |
  | 0.24 | 0 | - | impossible | 22% |

  Amplitude at burst end *can* be matched per hidden unit (weight x1.57 for tau=2 up to
  x4.90 for tau=192, with 0% gap afterwards), but not timing: a hidden unit that would
  fire on drive spike k fires on spike 5 instead (up to 8 steps later), and a unit that
  would fire on every drive spike can fire at most once. So L1 thins and delays a
  surviving channel before it prunes it.

---

## 5. Training and evaluation mechanics

- **Warm-up mask** [session]. `since_reset` started at 0, so only epoch 1 was protected;
  every later per-epoch `init_state` was unnoticed and the cold-start transient was
  scored (1/8 of an 8-chunk val, 1/40 of train). Under `--synaptic` it made val loss ~10x
  higher at unchanged RMSE and meant `best_model.pt` was chosen on that artifact.
  `mark_all_reset()` fixes it; val is now weighted by valid steps. Old and new val numbers
  are not comparable.
- **Lambda schedule.** [your runs] *Ramp from epoch 1, warmup 30 (no synaptic):* fanin90
  ~16 at init, 12-14 at epochs 10-20, 3-5 at epoch 30, then back to ~8 and flat; `nact_`
  dipped to 3-10 then recovered to 8-12; val loss bumped slightly at the dip, train loss
  looked normal. *Hold 50 + warmup 20, synaptic on:* fanin90 stayed ~16 until epoch 50,
  fell to ~2 by epoch 60 (penalty at half strength) and stayed; `nact_` 2-4. Reading
  [session]: the dip is overshoot (PR is rich-get-richer and Adam revives channels the task
  still needs), and `nact_` falling with fanin90 shows it was real, not a gain-inflation
  artifact. With a hold, the task has already spread the gains, so PR starts with a large
  gradient (toy: 0.034 at 1% spread, 0.19 at 5%, 0.38 at 10%).
- **Synaptic needs fewer channels** [session toy, linear fit of a gait-like waveform]:
  RMSE (normalized) 0.141 with 6 synaptic channels against 0.290 without; 0.035 vs 0.151
  at 12. [your runs]: synaptic on gave fanin ~2.5, `nact_` 3-6, same RMSE.
- **PR vs spike count.** PR is scale-invariant, so it counts channels, not spikes: 8
  channels at gain 0.30 and at 1.10 have the same PR but 8 vs 40 spikes/cycle.
- **cpg_lif at readout tau 5, first order, fully natural gating** [your runs]: lowest
  RMSE and val loss on the project, good alignment, no jagged response. The same settings
  with `--drive_source timing` gave spiky output, dying/saturating units, 1-2 vs ~70 spk/cyc
  across units and no clear alignment.

### Why the timing drive struggles without leaks [session, toy]
Fitting a gait-like waveform through a fast first-order readout (tau 5, no hidden
dynamics), one shared input line (timing) against separate per-channel lines (`cpg_lif`):

| spikes/cycle | shared line: error / jaggedness | separate lines: error / jaggedness |
|---|---|---|
| 5 | 0.211 / 5.16 | 0.296 / 2.23 |
| 20 | 0.078 / 3.02 | 0.102 / 2.72 |
| 30 | 0.110 / 3.50 | 0.075 / 2.71 |
| 60 | 0.210 / 7.16 | 0.073 / 2.67 |

A single line can fit as well but only in a narrow spike-count window (every spike adds
the same shape, so extra spikes cannot be made harmless), and is 1.5-3x more jagged. This
fits the dead/saturated attractors and the spiky output. With leaks open the hidden layer
shapes each spike's effect, which widens the window. **Clumping** (timing units firing on
consecutive CPG channels): with a shared line, a single localized feature favors adjacent
channels (3-channel fits: 25% of the best 20 sets contiguous for a femur-like lift vs ~2%
for all sets; separate lines 5%); your counterexamples (2-3 consecutive channels with
several slope changes) show the real network is state-dependent, so this is a tendency,
not a requirement.

---

## 6. Alignment scores

- **Derivatives come from the smooth table at each step's exact phase**, not the
  per-step target: `build_targets` looks up the lower table row, a staircase. Measured
  on a 512-row table at 180 steps/cycle: 8.4% (1st) and 143% (2nd derivative) error from
  the staircase vs 0.1% / 2.1% from the table. [session]
- **Chance-anchored**: `S = (spike-weighted mean of w - mean of w) / (max w - mean w)`.
  With max-only normalization random placement scores ~0.64 on a sinusoid but ~0.25 on a
  sharp waveform. Per-step credit is a decayed *maximum* (horizons: 3 steps velocity, 1
  corner); a decayed *sum* put the best spike before fast motion under `--synaptic`. [session]
- **Test bump** (flat, elliptical swing; velocity / corner): corners only 0.70 / 0.36;
  corners + sporadic flat-region spikes 0.45 / 0.23; dense inside the swing 0.07 / -0.01;
  flat region only -0.06 / -0.01; random ~0. An ellipse is steepest at its *ends*, so
  spikes at swing start and end are where the joint moves fastest. [session]
- **Observed** [your runs]: scores around +-0.25, mean ~0; sub-networks with well-placed
  sparse spikes next to ones with no alignment; alignment does not change RMSE, so
  nothing in the objective rewards it. Whether alignment is seed- or task-determined is
  still untested (the seed test).
- **Not adopted:** an inverse-filter "demand" score (the readout's order sets which
  derivative matters: speed for first order, acceleration for second) [session; you
  distrusted the math]; a swing/stance score from IK footfalls (pending effort).

---

## 7. Tooling notes

- Config is written before training and completed at the end; `epoch0.pt` and
  `epoch<N>.pt` snapshots are loadable mid-run. `--ckpt <name>` loads another snapshot and
  writes to `viz_<ckpt>/` with suffixed data files. `--scores_only` computes only the
  alignment tables. ONNX export is opt-in (`--export_onnx 1`).
- Bugs worth remembering: `hidden_reset` was read by `build_model_from_cfg` but never
  written, so zero-reset checkpoints reconstructed as subtractive; dense runs failed at
  `run_training` (unbound `sat_rate`) and in the visualizer (`model.group_cols`);
  `python train.py --help` crashed on a lone `%` in a help string.
- Visualizer history: under `cpg` modes the second raster was a verbatim copy of the CPG
  raster, and the drive-spike replay used a `spikes[:6*period]` prefix that left the last
  plotted period blank.

## 8. Ideas floated, not done
- Weight the task loss by joint speed (justified by the task, gives spikes a reason to
  land during fast motion). Alignment as a reward term was rejected: it should be
  emergent; it may be useful later as a selection criterion for a genetic search.
- RigL/SET-style regrowth, or mask-then-fine-tune after pruning (silent channels are
  exactly zero, so masking changes nothing at the moment it is applied).
- A circular-window concentration penalty that treats adjacent channels as equivalent
  (rejected for now: one window per sub-network is restrictive).
- Shorter `--tau_max` or readout tau, to test whether internal memory makes alignment
  unnecessary; a seed-dependence test of which sub-networks align.