# Architecture change — todo list

Everything below was left open, accepted-but-unresolved, or explicitly deferred in
`snn_training_speedup_todo.md` (speedup work — now essentially done). Consolidated
into one flat, numbered list rather than separate bites/follow-ups/ideas sections.

---

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

**6.** [ ] **`max_gaits=16` fixes the FiLM table shape** — *Medium*
   > 18.7% of params at h=128, so a checkpoint trained on `n_gaits=4` loads
   into a run using up to 16. Changing `max_gaits` itself, unlike `n_gaits`,
   invalidates old checkpoints — that's the tradeoff being made, just
   flagging it so it doesn't surprise anyone later.

**7.** [ ] **Hybrid gait conditioning (discrete pattern + continuous parameters)** — *Medium*
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

**8.** [ ] **Reintroduce leg grouping, properly this time** — *High*
   > Removed along with input routing because the old block-diagonal layers 2+
   made the network four independent sub-networks with no principled way to
   align each group to a CPG neuron (the routing permutation's phase
   alignment only held for 1 of 4 gaits). A better alignment mechanism is
   needed before grouping is worth bringing back — not just re-adding the
   block-diagonal structure blind.

**9.** [ ] **Parallel scan over timesteps** — *Low*
   > The one remaining *structural* lever on speed, as opposed to the
   launch-count reductions already done. `mem_t = β·mem_{t-1} + cur_t` is a
   linear recurrence and is parallelisable by associative scan in O(log T)
   depth, but the spike-and-reset nonlinearity breaks that directly. Would
   need the reset approximated or restructured — this is a research
   question, not a speed optimisation, and shouldn't be attempted casually.

**10.** [ ] **`inference.py` is broken** — *High*
   > `StatefulSNNPredictor` still hardcodes 5 state tensors (`state_names_in`,
   `[z() for _ in range(5)]`, `out[1:]`) from before the recurrence removal,
   and the state shape is now `(B, H)` rather than `(B, 4, H/4)` after the
   leg-grouping removal — so it needs updating on both counts, not just the
   tensor count. Needs a matching update once a model is actually ready to
   deploy.

**11.** [ ] **Old `.onnx` / `best_model.pt` artifacts are incompatible** — *Low*
   > Incompatible with the current architecture (recurrence removed, leg
   grouping removed, FiLM reshaped). Any of the architecture changes above
   will invalidate checkpoints again — expect to retrain from scratch after
   each one, not fine-tune from an old checkpoint.

   