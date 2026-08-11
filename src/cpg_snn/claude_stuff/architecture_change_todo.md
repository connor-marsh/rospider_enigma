# Architecture change — todo list

Everything below was left open, accepted-but-unresolved, or explicitly deferred in
`snn_training_speedup_todo.md` (speedup work — now essentially done). Consolidated
into one flat, numbered list rather than separate bites/follow-ups/ideas sections.

---

## Pertaining too architecture change

**1.** [ ] **`max_gaits=16` fixes the FiLM table shape** — *Medium*
   > 18.7% of params at h=128, so a checkpoint trained on `n_gaits=4` loads
   into a run using up to 16. Changing `max_gaits` itself, unlike `n_gaits`,
   invalidates old checkpoints — that's the tradeoff being made, just
   flagging it so it doesn't surprise anyone later.
   >
   > **Update (timing_grouped).** `max_gaits` now also sizes `w_in_gait`, the
   per-gait CPG→timing matrix. Unlike the FiLM tables, whose unused rows are
   inert identities (gamma=1, beta=0, no gradient), the unused `w_in_gait`
   rows hold **random** values. So loading a 4-gait checkpoint into an 8-gait
   run gives the 4 new gaits a random routing rather than a trained one. That
   is arguably correct — a genuinely new gait needs a new routing — but it
   means "checkpoint transfers across `n_gaits`" is weaker than it was: FiLM
   transfers as identity, routing transfers as noise.

**2.** [ ] **Hybrid gait conditioning (discrete pattern + continuous parameters)** — *Medium*
   > Gait is currently one integer index into an over-allocated FiLM embedding,
   which handles *growing the gait count* but not *structure*: some
   variation is genuinely categorical (tripod vs ripple vs wave are
   different footfall sequences — interpolating them is meaningless) while
   some is continuous (`tripod` vs `tripod_huge` is one pattern at two
   amplitudes; leg height and step length likewise). A small discrete
   embedding for pattern class concatenated with continuous inputs for the
   metric parameters would give interpolation between amplitudes for free
   and remove `n_gaits` from the architecture entirely. Implementation
   sketch: replace `Embedding(max_gaits, 2H)` with
   `Embedding(n_patterns, d) ⊕ continuous_dims → Linear → ReLU → Linear(·, 2H)`.
   The nonlinearity matters — a bare `Linear` on a code is additive in the
   code, which for a binary gait index would force
   `FiLM(0011) = FiLM(0001) + FiLM(0010) − b`, imposing arbitrary structure
   over a meaningless labelling.
   >
   > Note this now applies to **two** lookups, not one: FiLM *and* `w_in_gait`.
   The continuous/categorical split is arguably a better fit for routing than
   for FiLM — amplitude variants of one pattern should share a footfall
   order, so `tripod` and `tripod_huge` want the *same* routing and different
   FiLM. A hybrid scheme could express that; two independent
   `Embedding(max_gaits, ·)` tables cannot.

**3.** [~] **Reintroduce leg grouping, properly this time** — *High*
   > Removed along with input routing because the old block-diagonal layers 2+
   made the network four independent sub-networks with no principled way to
   align each group to a CPG neuron (the routing permutation's phase
   alignment only held for 1 of 4 gaits). A better alignment mechanism is
   needed before grouping is worth bringing back — not just re-adding the
   block-diagonal structure blind.
   >
   > **The alignment mechanism is now a learned per-gait matrix.** `TimingGroupedSNN`
   inserts a **timing layer** of `n_timing` LIF neurons between the CPG and
   the sub-networks: CPG spikes → timing layer (dense, FiLM, **no LayerNorm**)
   → `n_timing` fully disconnected sub-networks, one per timing neuron. CPG→timing
   is `w_in_gait`, an `Embedding(max_gaits, n_cpg_neurons * n_timing)` reshaped
   per gait, which is the continuous learned successor to the deleted
   `solve_leg_routing` permutation.
   >
   > Why it has to be per-gait and not FiLM: only one CPG neuron fires per
   timestep, so timing neuron *j*'s input current takes one of
   `n_cpg_neurons + 1` discrete values fixed by column *j* of the matrix.
   FiLM contributes one scale and one shift per unit, so the **ordering** of
   those values across CPG neurons is identical for every gait — timing neuron
   *j* is always driven hardest by the same CPG neuron. Gaits whose footfall
   **order** differs (not just amplitude or phase offset) are unreachable
   under a shared `w_in`, and the sub-networks would have to re-derive phase
   internally, which is the exact thing the timing layer exists to remove.
   The four current gait tables may not disagree on order, but other tables
   in the library do, so assume they can.
   >
   > Division of labour: the timing layer learns the **rhythm** in a few
   hundred parameters; the sub-networks learn **angles** given a clean phase
   reference, rather than four copies each re-deriving phase from raw spikes.

**3a.** [x] **Do it with no cross talk** — *High*
   > Implemented as `--arch timing_grouped`. Sub-network *g*'s only spike
   input is timing neuron *g*; layers 2+ and the analog readout are block
   diagonal `(G, Hg, Hg)`; group *g* writes only its own gait-table columns
   (`build_group_cols`: `n_timing == n_legs` → `LEG_COLS[l]`,
   `n_timing == n_joints` → `[j]`, anything else rejected). There is no path
   between groups anywhere after the timing layer. **Untested** — no torch in
   the authoring environment, so it is shape-reasoned, not run. First thing
   to do is `--arch timing_grouped --dry_run`, which exercises construction,
   `init_state`, `timing_only`, both plot paths and ONNX export without
   spending an epoch.

**3b.** [ ] **Do it with cross talk** — *Medium*
   > Compare results. One-line change to sub-net layer 1: replace the outer
   product `spk_t.unsqueeze(-1) * w1 + b1` with
   `einsum("bt,tgh->bgh", spk_t, w1_cross)` and `w1` becomes
   `(n_timing, G, Hg)`. Deliberately not wired behind a flag, so the
   comparison is between two committed variants rather than an untested
   branch. Judge on free-run RMSE and `Val(post-sw)` at matched gradient
   steps, not train loss.

**3c.** [ ] **Sweep the timing layer's shape and firing regime** — *High*
   > The knobs that actually decide whether 3a works, in rough order of
   expected impact:
   > - `--n_timing`: `n_legs` (4, default) vs `n_joints` (8). Columns *j* and
   >   *j+4* share a circular phase offset in every gait, so 8 timing neurons
   >   would learn 4 distinct phases and duplicate each — `n_legs` is the
   >   principled default, but `n_joints` gives each column a private decoder
   >   and is worth one run.
   > - `--timing_w_scale` (default 0.5). With no LayerNorm on the timing
   >   layer this sets the firing regime directly against `thresh=1.0`. Raise
   >   if units are dead, lower if saturated; read it off the `spk/cyc` column.
   > - `--sub_ln` (default `both`). **Ablate `l1` first.** That layer's only
   >   input is one binary channel, so LayerNorm normalises away the amplitude
   >   of the sole drive and leaves FiLM gamma to restore it. Might be fine,
   >   might be a handicap. `l2` is the conventional case and probably wants
   >   to stay on.
   > - `--tau_timing_max` (default 64). Separate from `--tau_max` because the
   >   sample is tiny: with `n_timing=4`, a log-uniform draw over [2, 256] puts
   >   ~1 unit above tau=64, so coverage is decided by the seed rather than by
   >   tiling. Taus stay learnable, so this is an init range, not a cap.
   > - `--hidden`: **now means PER GROUP** for this arch (total for `dense`).
   >   At G=4, `--hidden 128` lands near dense `--hidden 256`'s parameter
   >   count and is the honest matched-parameter baseline; `--hidden 256` is
   >   ~4x. Run both before concluding anything about the architecture rather
   >   than about capacity.

**3d.** [ ] **Reconsider having entirely separate CPG→timing weights per gait** — *Medium*
   > Requested explicitly. `w_in_gait` currently gives each gait its own
   `(n_cpg_neurons, n_timing)` matrix with nothing shared, which is the
   maximally expressive option and also the one with the least inductive
   bias: nothing says two gaits that differ only in amplitude should share a
   routing, and nothing transfers when a gait is added (see item 1).
   Alternatives, cheapest first:
   > - **Shared base + per-gait delta**: `w_in = w_shared + w_delta[gait]`,
   >   with `w_delta` weight-decayed or initialised at zero, so gaits start
   >   tied and only pay for the differences they need.
   > - **Per-gait soft permutation**: keep one shared `(n_cpg, n_timing)`
   >   magnitude matrix and give each gait a learned soft permutation
   >   (Sinkhorn / softmax over CPG neurons) applied to it. Expresses "same
   >   drive strengths, different footfall order", which is the actual thing
   >   that varies, in `n_cpg * n_timing` params per gait instead of a free
   >   matrix.
   > - **Fold into item 2's hybrid scheme**: routing conditioned on pattern
   >   class only, FiLM conditioned on pattern + continuous parameters. This
   >   is probably the right end state.
   >
   > Also fix while here: **FiLM gamma on the timing layer is now redundant**
   with `w_in_gait` — both are per-gait and multiplicative, so gamma can only
   rescale what the matrix already scaled. `beta` is not redundant (additive
   tonic drive shifts threshold-crossing time). Kept as-is because the layer
   is specified to have FiLM and because gamma becomes load-bearing again the
   moment `w_in_gait` is made gait-shared by any of the options above. If
   per-gait routing is kept permanently, consider dropping timing gamma.

**3e.** [ ] **Graph diagnostics for the timing layer** — *Medium*
   > Print-based diagnostics exist: `timing_report` prints, per gait and per
   timing neuron, **spk/cyc** (0.00 = DEAD, with an explicit warning), the
   **circular mean phase** of firing, and **R**, the circular concentration.
   R is the one to watch — a unit can have a healthy rate and still be
   useless if its spikes are smeared around the cycle, in which case `phase`
   is meaningless. Runs every `--timing_log_every` epochs (default 10), again
   on the restored best checkpoint, and once at init under `--dry_run`; the
   final stats land in the config as `timing_layer_stats`.
   >
   > Still wanted as plots:
   > - **Timing raster overlaid on the CPG raster**, with neuron-0 burst
   >   onsets marked, so re-timing is visible directly.
   > - **Timing onset phase vs that leg's swing onset in the gait table**, per
   >   gait. This is the direct successor to the 0.116–0.133-cycle residual
   >   measurement that killed the old routing, and it is the plot that says
   >   whether `w_in_gait` learned an alignment the fixed permutation could
   >   not.
   > - **Phase histogram per timing neuron per gait**, which is the visual
   >   version of the R statistic.
   > - **Learned tau distribution**, timing layer vs sub-networks, to check
   >   whether `--tau_timing_max` is binding.

**3f.** [ ] **Dead-timing-neuron protection** — *Medium*
   > Sharper failure mode than anything in the dense arch: each sub-network's
   entire input is one binary channel, so a silent timing neuron freezes its
   leg at whatever the decaying membranes settle to, and the surrogate
   gradient through a never-crossing membrane is weak enough that it tends
   not to recover on its own. `timing_report` detects it; nothing currently
   prevents it. Options if it shows up in practice: a firing-rate
   regulariser on the timing layer, a lower threshold or higher
   `--timing_w_scale`, or re-init of dead units mid-run. Don't add any of
   these pre-emptively — wait and see whether the report ever fires.

**4.** [ ] **`inference.py` is broken** — *High*
   > `StatefulSNNPredictor` still hardcodes 5 state tensors (`state_names_in`,
   `[z() for _ in range(5)]`, `out[1:]`) from before the recurrence removal,
   and the state shape is now `(B, H)` rather than `(B, 4, H/4)` after the
   leg-grouping removal — so it needs updating on both counts, not just the
   tensor count. Needs a matching update once a model is actually ready to
   deploy.
   >
   > **Two more breakages since.** (a) Its import line does
   `from train import (..., CPG_W, ...)`; `CPG_W` no longer exists, replaced
   by `CPG_W_BY_N` + `cpg_weight_matrix(N)`, so the file now fails at
   *import* rather than at runtime. (b) `--arch timing_grouped` has a
   **4-tensor mixed-rank** state — `mem_timing (B, n_timing)` plus
   `mem1/mem2/memo (B, n_timing, hidden)` — so the predictor has to branch on
   `cfg["arch"]`, not just count tensors. The config already carries
   everything needed: `arch`, `n_timing`, `hidden`, `hidden_is_per_group`,
   `group_cols`, and `model_detail.onnx_inputs` / `state_shapes` read off the
   live model at export time. `LIFCPGStepper(N=...)` also no longer defaults
   to 4 — N is required — and `cfg["cpg"]["N"]` / `["W"]` now carry the actual
   values used.

## Miscellaneous items

**1.** [ ] **Find the largest usable LR** — *Low*
   > Still at 2e-3. `|grad|` and loss both trend smoothly down with no
   oscillation, which rules out the LR being *too high* but says nothing
   about *too low* — smooth slow descent looks the same as smooth fast
   descent. Plan: **LR range test from a warm checkpoint** — load
   `best_model.pt` after ~50 epochs so the loss surface is near-stationary,
   ramp `lr` exponentially over ~300 steps logging loss per step, take
   roughly ⅓ to ¹⁄₁₀ of the divergence point. Don't run it from scratch — the
   65→3 initialisation transient happens at any LR and would swamp the
   signal. Confirm the bracket with two full runs judged on free-run RMSE and
   `Val(post-sw)` at matched gradient steps, not train loss. Subsumes the
   batch-size LR rescale question below — no reason to tune LR twice.
   >
   > Redo per arch. `timing_grouped` has a very different parameter geometry
   (embedding lookups + block-diagonal einsums, ~4x the params at equal
   `--hidden`), so the dense LR bracket does not transfer.

**2.** [ ] **Batch size may still want an LR rescale** — *Low*
   > Batch went 32→128 (4×) but `lr` stayed at 2e-3, deferred deliberately so
   the first benchmark was a clean comparison. Adam sqrt-rule suggests
   ~4e-3. Fold this into item 1's LR range test rather than treating it as a
   separate sweep.

**3.** [ ] **`train.py` has no Dynamo recompile-limit protection** — *Medium*
   > `benchmark.py` raises the limit to 64, resets between variants, and
   verifies compilation three ways; `train.py` does neither. Currently ~3
   guard sets per run (train+grad, val+no_grad, plot at batch=1), safely
   under the default limit of 8, but no margin if a 4th guard set appears —
   and this exact bug silently corrupted two benchmark sessions before
   detection existed. Cheap insurance: call
   `torch._dynamo.config.recompile_limit = 64` (or whatever the installed
   torch version calls it) once at startup.
   >
   > **Margin is thinner now.** `timing_grouped` adds no guard set by design —
   `timing_report` deliberately calls the eager `_timing` rather than the
   compiled `step`, precisely so the batch-1 diagnostic doesn't add one — but
   the arch has more shapes in play and the headroom under 8 is no longer
   comfortable. Promote this to do-it-now if a fourth guard set appears.

**4.** [ ] **`val_frac` time split is not a real holdout** — *Low*
   > Unfixed by design. The val region is a phase-shifted duplicate of train,
   since the CPG is exactly periodic. Continue judging on free-run
   reconstruction RMSE and `Val(post-sw)` rather than plain val loss — that's
   already how the recurrence-removal decision was made, just not written
   down in code. A genuine holdout would hold out
   `(gait_from → gait_to, phase_at_switch)` combinations, not a time slice.
   Worth doing once the architecture stabilises rather than before every
   experiment.

**5.** [x] **`val_chunks` overhead vs `chunks_per_epoch`** — *Low*
   > ~~`val_chunks=8` runs every epoch regardless of `chunks_per_epoch`, ~20%
   overhead at defaults (8 val / 40 train chunks).~~ Changed default to
   `val_chunks=2`, since free-run RMSE (not val loss) is the metric that
   actually matters — val loss is only a per-epoch progress check now, not a
   decision criterion. Still worth remembering: don't shrink
   `chunks_per_epoch` far without checking `val_chunks` proportionally, or
   validation can end up dominating the compute budget again at more extreme
   ratios.

**6.** [ ] **Parallel scan over timesteps** — *Low*
   > The one remaining *structural* lever on speed, as opposed to the
   launch-count reductions already done. `mem_t = β·mem_{t-1} + cur_t` is a
   linear recurrence and is parallelisable by associative scan in O(log T)
   depth, but the spike-and-reset nonlinearity breaks that directly. Would
   need the reset approximated or restructured — this is a research
   question, not a speed optimisation, and shouldn't be attempted casually.

**7.** [x] **Verify the ported N=3 and N=6 CPG matrices before using them** — *High*
   > `--n_cpg_neurons {3,4,6}` selects from `CPG_W_BY_N`, with the N=3 and N=6
   matrices ported from the old `cpg_utils.py::BLIF_CPG`. The original concern
   was a parameter-regime mismatch: that file ran `from_fb_weight = -1e6`
   while this one defaulted to `-1e4`, and since that value is the
   burst-terminating kick landing in the slow `u` filter (`du=0.1`), recovery
   time scales like `log(|from_fb| / i_app) / du` — roughly 111 steps at -1e6
   vs 68 at -1e4 — so a mismatched regime would produce a different
   oscillator than the matrices were tuned for.
   >
   > **Resolved.** Confirmed that `-1e6` works correctly for both the
   original N=4 case and the ported N=3/N=6 matrices, so there is no regime
   to mismatch. `from_fb_weight` is no longer an argument at all —
   `--from_fb_weight` is removed and `CPG_FROM_FB_WEIGHT = -1_000_000.0` is
   hardcoded in `train.py` next to `CPG_W_BY_N`, used unconditionally by
   `LIFCPGStepper` regardless of `N`. `analyse_cpg`'s burst-phase-offset
   warning stays (still a useful general sanity check on the coupling matrix
   vs `i_app`), just no longer framed as a regime check.

**8.** [ ] **Move the timing layer out of the torch class** — *Low*
   > Longer-term intent: run the timing layer as a separate structure the way
   the CPG is run, rather than as LIF neurons inside
   `TimingGroupedSNN.step`. That would make it a fixed (or separately
   trained) rhythm generator that the sub-networks consume, and would shrink
   the exported ONNX graph's state to the sub-network membranes only.
   Deliberately deferred: keeping it inside the class means it is trained
   end-to-end by the same surrogate gradient as everything else, which is
   the version worth measuring first. `_timing` is already factored out as
   its own method, so the extraction is mechanical when the time comes.

**9.** [ ] **Named legs for quadruped and hexapod, for use in visualizations** — *Low*
   > Requested directly. Every leg-indexed label right now is a bare index —
   `leg{l}` / `T{l}` in visualize.py's plots — not an anatomical name. `HEXAPOD_LEG_NAMES =
   ["LF","LM","LR","RF","RM","RR"]` already exists next to `HEXAPOD_LEG_COLS`
   but nothing reads it yet. Needs:
   > - `QUADRUPED_LEG_NAMES`, the equivalent constant for the 4-leg case
   >   (convention TBD — e.g. `["FL","FR","BL","BR"]` matching LEG_COLS'
   >   ordering, needs to be checked against which physical leg column 0
   >   actually corresponds to).
   > - Thread a name list through the same path `leg_cols` already takes:
   >   `default_leg_layout` returns it (or a parallel lookup keyed the same
   >   way), main() passes it down, and it lands in cfg (e.g. `"leg_names"`)
   >   so visualize.py reads it back instead of re-deriving it from
   >   `n_cpg_neurons`.
   > - A user-supplied `--leg_cols` has no matching names by construction; add
   >   `--leg_names` alongside it, and fall back to numeric labels when only
   >   one of the two is given rather than guessing a pairing.
   > - Swap the numeric labels for real names in visualize.py's
   >   `plot_alignment`, `plot_phase_fold`, `plot_alignment_summary`, and
   >   `plot_routing`, and in train.py's leg->servo startup print.

**10.** [ ] **The spike-concentration penalty forbids two bursts per cycle** — *Medium*
   > `spike_stats_penalty` scores burst tightness with the circular
   concentration R of a unit's spike phases:
   >
   > `R = |sum_t spk_t * exp(i*2*pi*phase_t)| / sum_t spk_t`
   >
   > That is the magnitude of the FIRST Fourier component, so it measures
   "all spikes at one cycle phase". Two bursts at opposite phases cancel to
   R ≈ 0 and get punished as hard as spikes smeared uniformly around the
   cycle, even though a two-burst pattern is perfectly structured.
   >
   > Accepted deliberately for now: every gait currently in use swings each
   leg exactly once per cycle, so one-burst-per-cycle is the correct target
   and R is the right statistic. But it is baked into the objective, not a
   configurable choice, and it will silently fight any gait where a leg
   should swing twice per cycle (some ripple/wave variants, or any gait
   defined at double the CPG's fundamental).
   >
   > Fix when needed: score the second Fourier component as well, and take
   the max — `R_k = |sum_t spk_t * exp(i*2*pi*k*phase_t)| / sum_t spk_t` for
   k in {1, 2}, target `max(R_1, R_2)`. That rewards "clustered at one phase
   OR at two opposite phases" and leaves the one-burst case scoring exactly
   as it does today. Cheap — one extra einsum pair. Alternatively expose the
   harmonic as an arg, but auto-detecting it from the gait table's own
   spectrum would be more in keeping with how the other targets are derived
   (measured from data, not configured).
   >
   > Verification note: the current two-term penalty was checked numerically
   before use — a CPG-like burst (10 spikes, 3 steps apart) and the same 10
   spikes spread evenly over the cycle have IDENTICAL rates but penalties of
   3.2e-5 vs 9.7e-3, a ~300x separation. That separation is entirely the R
   term, which is what makes it worth keeping despite this limitation.

**11.** [ ] **Find a gait-conditioning scheme that isn't a per-gait weight table** — *High*
   > `w_in_gait` is back: an `Embedding(max_gaits, n_cpg_neurons * n_timing)`
   giving each gait its own free CPG→timing routing matrix, nothing shared.
   It is the version that demonstrably separates gaits, and it is not the
   version we want long-term.
   >
   > **What's wrong with it.** `n_gaits` appears in a parameter shape.
   Nothing transfers between gaits — an added gait starts from a random
   routing, and unused `max_gaits` rows are random noise rather than inert
   identities (item 1). Per-gait capacity grows linearly in the gait count.
   And it cannot express "these two gaits share a footfall order but differ
   in amplitude" even when that is true, because the two rows are
   independent by construction.
   >
   > **What was tried and failed.** A shared pair of weight matrices with a
   per-gait FiLM gate on a 16-unit LIF hidden layer, i.e.
   `W_eff(g) = W2 diag(gamma_g) W1`. Expressiveness was verified in advance:
   400 random gates over a shared router produced 248 distinct routings, 247
   of them many-to-one, including 30 tripod-shaped 2-driver patterns. Despite
   that, the trained result had **near-identical timing phases across gaits
   and near-identical measured routing matrices**. Two probable mechanisms,
   both pointing the same way:
   > - *Competing capacity.* `film1`+`film2` carry ~3,072 params per gait
   >   against the router gate's ~44, and sit 1–2 spiking layers from the loss
   >   rather than 3–4. Each spiking layer attenuates gradient by the
   >   surrogate derivative `1/(slope·|x|+1)²` (~1e-3 at slope 25), so the
   >   sub-network gate's gradient is orders of magnitude larger. Gradient
   >   descent put gait knowledge where it was cheapest, and the timing layer
   >   collapsed to a gait-independent clock.
   > - *Gradient averaging.* With `W1`/`W2` shared, gradients from gaits
   >   wanting different routings land in the same weights and partially
   >   cancel, resolving toward one compromise routing. A per-gait table
   >   removes the averaging: gait *g*'s weights only ever see gait *g*'s
   >   gradient.
   >
   > **Directions worth trying, cheapest first.**
   > - **Remove the competition instead of strengthening the router.**
   >   `--sub_film none` forces the sub-networks to infer gait solely from
   >   their timing neuron's spike train. Now implemented but untested. Run it
   >   with `--spike_stats_lambda 0`: the penalty pins rate and burst
   >   concentration to the CPG's values for every gait, leaving phase as the
   >   only axis able to carry gait, and if a leg's phase is legitimately the
   >   same in two gaits while its waveform differs, the two constraints are
   >   jointly unsatisfiable.
   > - **Shared base + per-gait delta**, `w_in = w_shared + w_delta[gait]`
   >   with `w_delta` zero-init and weight-decayed, so gaits start tied and
   >   pay only for differences they need. Keeps per-gait gradients unaveraged
   >   while making sharing the default rather than the constraint.
   > - **Larger gate init noise.** Gaits currently start within 5% of each
   >   other (`1.0 + 0.05*randn`). Combined with weak gradients they may never
   >   separate. One-line change, worth trying before abandoning any gated
   >   scheme.
   > - **Auxiliary loss directly on timing-phase separation** — e.g. reward
   >   pairwise distance between gaits' phase vectors. Makes separation an
   >   objective rather than hoping it emerges. Risk: it is a made-up target,
   >   and the right phases are exactly what we don't know.
   > - **Hybrid discrete/continuous conditioning** (item 2) subsumes much of
   >   this and is probably the right end state.
   >
   > **Diagnostic now in place:** `timing_report` prints a "phase separation
   across gaits" number (max deviation on the unit circle). Below ~0.1 means
   the gaits are not being distinguished. Rate and R being identical across
   gaits is expected and is not evidence of collapse — the spike-statistics
   penalty pins both.

**12.** [x] **Timing-layer bursts didn't match the CPG's** — *Medium*
   > Observed: timing units either hit R≈1.0 (burst tighter than the CPG's) or
   sat at R≈0.8–0.9 with one small burst plus scattered spikes, and none
   produced one CPG-sized burst per cycle. Suspected, correctly, that timing
   units were firing on *consecutive* timesteps while the CPG fires every
   *other* timestep.
   >
   > **Cause: subtractive reset.** The CPG (`LIFGeneralArray`) sets
   `v[spike] = 0`; the timing layer subtracted the threshold instead, leaving
   `mem - thresh` behind. When that residual is still above threshold the unit
   fires again on the next step with no input at all. Since CPG spikes arrive
   every 2nd step (`refrac_main=1`), the result was ISIs of 2,1,1,2,1,1,… —
   consecutive spikes filling the gaps — and **15 spikes per burst instead of
   10**, so one mechanism caused both the over-firing and the too-tight R.
   >
   > **Fix: `--timing_reset zero`, now the default.** Measured: reset-to-zero
   gives clean spacing-2 ISIs, 10 spikes, and R = 0.99475 — which is *exactly*
   the CPG's own R. No loss term needed for burst width.
   >
   > Also ruled out along the way: making the concentration term two-sided.
   R is far too insensitive at high concentration to discriminate burst
   spacing — spacing 1 gives R = 0.99869 vs spacing 2's 0.99475, a squared
   difference of 1.55e-05, contributing ~1.6e-07 to the loss at λ=0.01.
   Structural fix, not a loss fix.

**13.** [ ] **Derive footfall phases by inverse kinematics, and consider using them to train the timing layer** — *High*
   > The gait CSVs are joint angles, not foot positions, so the **fundamental
   phase of a joint column is not the footfall phase**. Trying to read
   footfall timing off the tables directly does not work: some columns rise
   and fall continuously through the cycle with no distinguishable "start",
   and even where a clear feature exists there is no guarantee it corresponds
   to touchdown or liftoff. `visualize.py`'s `fundamental_phase` is fine as an
   arbitrary-but-consistent reference for comparing a timing neuron against
   its own leg across runs, and it is explicitly NOT a biomechanical event —
   don't let it drift into being read as one.
   >
   > **What would actually work:** forward kinematics on the joint angles to
   get each foot's trajectory in the body frame, then read stance/swing from
   the foot's vertical position or its velocity relative to the body. That
   needs link lengths and the joint-axis convention for the specific robot,
   which is the only real cost here — the maths is small. Output would be a
   per-gait, per-leg touchdown phase, i.e. the footfall pattern, computed
   rather than asserted.
   >
   > **Two separate uses, worth keeping distinct:**
   > - *Diagnostic.* Compare each timing neuron's measured spike phase against
   >   its leg's real touchdown phase. That is the honest version of the
   >   alignment residual, and the direct successor to the 0.116–0.133-cycle
   >   measurement that killed the old fixed routing permutation.
   > - *Supervision.* Pretrain or auxiliary-train the timing layer to fire at
   >   the derived touchdown phases. This is the label-free version of the
   >   separate-training idea (train CPG→timing first, then the sub-networks)
   >   that came out of lab discussion — no hand-labelled timing data, since
   >   the labels are computed from the gait tables that already exist. It
   >   would very likely fix the gait-differentiation failure in item 11
   >   outright, because it makes phase separation an explicit objective
   >   rather than something hoped to emerge. Cost: it is no longer end-to-end,
   >   which was the thing worth preserving, so prefer an auxiliary loss that
   >   stays on alongside the task loss over a hard two-stage freeze.
   >
   > **Ceiling to be aware of before investing in this.** Achievable timing
   phases are currently quantised to the CPG's burst phases: with
   `timing_reset="zero"` and no input between bursts, a timing unit's membrane
   peaks at the end of its burst and then only decays, so it cannot cross
   threshold 30 steps later during silence — no value of `tau_timing_max`
   enables that. Measured N=6 burst phases are {0.000, 0.148, 0.321, 0.497,
   0.670, 0.821}, i.e. the k/6 grid, so any required footfall phase more than
   about ±0.05 cycle off that grid is unreachable and IK-derived targets would
   be asking for something the architecture cannot produce. Being addressed
   separately by tuning the CPG toward more continuous activity with smaller
   inter-burst gaps (coworker, later) — that work is a prerequisite for
   IK-derived supervision being fully usable, though the diagnostic use is
   valuable immediately.