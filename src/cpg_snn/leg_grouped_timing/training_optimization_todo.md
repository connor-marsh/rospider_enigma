# SNN training speedup — todo list

## Speed — do in this order

- [x] **1. Replace `FastSigmoidSpike` with a straight-through `spike_fn`** (plain tensor ops, no custom `autograd.Function`). Verified forward + gradient parity — both diffs exactly `0.0` after parenthesising as `hard + (surr - surr.detach())`, which makes `surr - surr.detach()` an exact IEEE-754 zero.
- [x] **2. `model.step = torch.compile(model.step, dynamic=False)`.** Compiled `step`, not `forward`. Guarded behind `device.type == "cuda"` with a print either way. `export_onnx` temporarily restores the eager step, since `torch.onnx.export` doesn't trace reliably through a compiled callable.
- [x] **3. Raise batch 32 → 128.** Done; b512 also measured and is better still.
- [x] **4. Benchmark.** `benchmark.py` built (imports everything from `train.py`), results in `outputs/bench/bench_results.{jsonl,md}`.

### Measured (RTX 3050 Laptop, commit 72c85ad, warm 20 / meas 50)

| variant | ms/step | M sTS/s | speedup | peak GiB |
|---|---|---|---|---|
| eager_b32 | 1480.48 | 0.006 | 1.00x | 0.16 |
| eager_b128 | 1498.74 | 0.021 | 3.82x | 0.59 |
| compile_b32 | 489.62 | 0.017 | 3.02x | 0.11 |
| compile_b128 | 382.37 | 0.086 | 15.49x | 0.40 |
| compile_b256 | 525.24 | 0.125 | 22.55x | 0.79 |
| compile_b512 | 536.69 | 0.244 | **44.14x** | 1.56 |

Compile alone ≈ 3.0x at fixed batch. The rest is batch scaling: per-step time is
roughly flat from b128 to b512, i.e. still hard launch-bound. **Noise floor on this
machine is ~15–27%** (identical configs across sessions), so ignore smaller deltas.

- [x] **5. Routing matrices gone entirely** — not hoisted, deleted. Per-gait input routing was an initialisation prior, not a capability (`w_cross` already spanned all neurons), and its phase alignment held for only 1 of 4 gaits (residuals 0.116–0.133 cycle ≈ 34 steps vs a ~29-step burst). FiLM lookups stay in the loop: they are now two `Embedding` gathers on a `(B,)` index, and hoisting them would mean materialising `(L,B,2H)` per chunk.
- [x] **6. Input path folded — by deletion.** Going fully connected collapses routing `bmm` + self `einsum` + cross `matmul` into one `addmm`, same ~6 launches saved as the planned per-gait `W_in` precompute, with no per-chunk rebuild. Every layer now uses `addmm` so biases are folded too.
- [ ] ~~7. Swap `einsum` → `bmm`~~ **Dropped.** The stated benefit (einsum re-parsing its spec per call) is an eager-mode cost only — under `torch.compile` the parse happens once at trace time. A naive `bmm` rewrite also needs transposes einsum avoids. Moot now anyway: flattening replaced every `einsum` in `step` with `addmm`.
- [x] **8. `spikes` / `targets` / `valid` live on the training device**; batches gathered on-device via `idx = pos[None,:] + off[:,None]`. Verified bit-identical to the old numpy path over 30 chunks including rewinds, with the RNG stream in lockstep. Low payoff as predicted (~0.1% of epoch time) — done for having no data pipeline left to reason about.
- [x] **9. Skip rewriting `StreamSampler` as a `DataLoader`.** Decision held — dataset is 21MB, nothing to parallelise across worker processes.

## Hyperparameters

- [x] **`t_max`: 150k → 50k.** CPG is exactly periodic after warmup, so 150k was ~590 duplicate copies of the same ~254-step cycle.
- [x] **`bptt`: 512 → 256** (≈ one CPG period). Still worth sweeping 128/256/512 at fixed `batch × bptt` (`--set bptt`) and judging on post-switch val + free-run RMSE.
- [x] **`tau_max`: 500 → 256.** Now the *only* long-timescale memory (recurrence removed), so do **not** sweep below one CPG period (~254). `train.py` warns if `tau_max < period`.
- [x] **`batch`: 32 → 128** default. b512 measured better; consider raising the default.
- [ ] **`lr`**: still 2e-3, and `lr` is the ONLY knob that moves Adam's step size (see bite #11 — the clip threshold does not). Evidence so far rules out "too high" but not "too low", so there may be headroom. Plan: **LR range test from a warm checkpoint** — load `best_model.pt` after ~50 epochs so the loss surface is near-stationary, then ramp `lr` exponentially over ~300 steps logging loss per step, and take roughly ⅓ to ¹⁄₁₀ of the divergence point. Running it from scratch would be swamped by the 65→3 initialisation transient, which happens at any LR. Confirm the bracket with two full runs judged on free-run RMSE and `Val(post-sw)` at matched gradient steps.
- [x] **`chunks_per_epoch`**: set so one epoch takes ~5–30 s wall-clock. 
- [x] **`hidden`: default 256 → 128** with the move to fully connected, since dense `w2`/`w_read` are 4× the block-diagonal versions — h=128 lands near the old grouped h=256 parameter count. Still untested; `--set hidden` sweeps 128/256/512.
- [ ] **`max_gaits`: 16.** Rows allocated in the FiLM tables; only the first `n_gaits` are used and the rest are inert. Costs 18.7% of params at h=128. Changing it invalidates checkpoints (changing `n_gaits` does not — that is the point).
- **Budget to hold fixed** when comparing configs: ~12–15k total gradient steps, i.e. `epochs × chunks_per_epoch`. Config records `gradient_steps` and `sample_timesteps`.

## Things that bite — audited at commit 72c85ad

Status: ✅ satisfied · ⚠️ known/accepted · ❌ open · ❓ needs confirming

| # | Status | Warning | Where it stands |
|---|---|---|---|
| 1 | ✅ | **`T_max` must equal the number of `sched.step()` calls.** Otherwise the cosine finishes early and training proceeds at `eta_min`. Fails silently. | `sched.step()` is inside the epoch loop, outside the chunk loop → called `args.epochs` times; `T_max=args.epochs`. Consistent. |
| 2 | ⚠️ | **`val_frac` time split is not a real holdout.** Val region is a phase-shifted duplicate of train (CPG is periodic). | Still true and unfixed by design. Behaviourally followed: the recurrence decision used free-run RMSE + `Val(post-sw)`, not plain val loss. Not documented in code. A genuine holdout would hold out `(gait_from → gait_to, phase_at_switch)` combinations, not a time slice. |
| 3 | ✅ | **Changing `chunks_per_epoch` silently changes the LR schedule** while `sched.step()` is per-epoch. | Fix: `T_max = epochs * chunks_per_epoch` and step per chunk. Blocks the `chunks_per_epoch` tuning item above. |
| 4 | ✅ | **`dynamic=False` + differing train/val batch size → recompile.** | Train and val both use `args.batch`. Plotting at batch=1 adds one extra graph; harmless. |
| 5 | ⚠️ | **Dynamo silently falls back to eager after 8 recompiles.** Diagnose with `TORCH_LOGS="recompiles,graph_breaks"`. | `benchmark.py` raises the limit to 64, resets between variants, and verifies compilation three ways. **`train.py` does neither** — currently ~3 guard sets (train+grad, val+no_grad, plot b=1), safely under 8, but unprotected if a 4th appears. This bug silently corrupted two benchmark sessions before detection was added. |
| 6 | ✅ | **Graph break causes inside `step`:** `.item()`, `print()`, `if tensor:`, custom `autograd.Function`. | `step` is clean. The `use_recurrence` bool branch is gone with the recurrence removal. |
| 7 | ✅ | **Confirm you're actually on CUDA.** | Device printed; compile prints `SKIPPED (device=..., not cuda)` when it doesn't fire. |
| 8 | ⚠️ | **`val_chunks=8` runs every epoch regardless of `chunks_per_epoch`.** | 8 val / 40 train chunks ≈ 20% overhead at defaults — acceptable. `benchmark.py` times val separately (with its own warmup floor) so the cost is visible. Don't shrink `chunks_per_epoch` much. |
| 9 | ✅ | **Don't put mutable sampler state (`pos`, `gait`, `count`) in a `Dataset` with `num_workers > 0`.** | N/A — `DataLoader` never adopted (speed item 9). |
| 10 | ❌ | **Changing batch size may require an LR rescale** — sqrt rule for Adam: 4× batch → ~2× LR. | **Open.** Batch 32→128 but `lr` still 2e-3; deferred deliberately so the first benchmark was a clean comparison. Try 4e-3. Interacts with #11. |
| 11 | ✅ | **`clip_grad_norm_` behaves very differently under Adam than under SGD.** Reading the pre-clip norm as an LR diagnostic is a mistake. | **Investigated; largely a non-issue.** Measured `|grad|`: ~3000 (epoch 1, initialisation transient — will be large at any LR), ~30 (epochs 2–4), ~3 (epoch 50), loss and `\|grad\|` both trending smoothly down. Three corrections to the original wording, which was wrong: **(a)** the pre-clip norm is computed by `clip_grad_norm_` *before* `opt.step()`, so it is a property of the loss surface and current weights and cannot indicate that the LR is too hot — LR does not enter until step 3. **(b)** Clipping does NOT shrink Adam's steps. Adam's update `lr·m̂/√v̂` is scale-invariant in the gradient, so step norm ≈ `lr·√N` whether the gradient arrives at norm 30 or 1; simulated clip ∈ {0.5, 1, 10} at fixed `lr` gave update norms identical to 6 significant figures. "Clipping truncates most of the update" is an SGD intuition. Clipping's real function under Adam is outlier protection — keeping one pathological batch from spiking `v`. **(c)** Therefore `clip=1, lr=20` is *not* equivalent to `clip=10, lr=2` under Adam: they differ by exactly the LR ratio (10×). Under plain SGD they would be identical. Raising `--clip` for speed would do nothing; the only argument for raising it to ~5–10 is diagnostic, so `\|grad\|` stops being censored 100% of the time. |
| 11b | ⚠️ | **Stability under clipping is asymmetric evidence about the LR.** | Smooth monotone loss with clipping active IS evidence the LR is not too *high*: an over-large LR would oscillate even on unit-norm clipped steps. It is NOT evidence the LR is not too *low* — slow smooth descent looks the same. So the current run rules out "too hot" and says nothing about headroom. Diagnose from loss/`\|grad\|` trends across epochs and from an LR sweep, never from the pre-clip norm value. |
| 12 | ✅ | **"Epoch" in `train.py` carries no meaning about data coverage** — `StreamSampler` never exhausts. Report `steps × bptt × batch`. | Config records `gradient_steps` and `sample_timesteps`. |
| 13 | ✅ | **Re-run the ONNX parity check after changing `spike_fn`.** Confirm diff < 1e-4. | Check is present in `export_onnx` and prints `PyTorch vs ONNX max diff`. 
| 14 | ✅ | Removing per-batch `.item()` removes a backpressure/sync point; peak activation memory can rise. | N/A for `train.py` — `tot += loss.item()` is still in the chunk loop. That warning was about the old `training_utils.py`. |

## Resolved / ruled out

- **`mode="reduce-overhead"` (CUDA graphs): ruled out, structurally.** `forward()` calls `step()` ~256 times before `backward()`, and autograd retains every invocation's outputs for the backward pass — but CUDA graphs provide one static buffer set, so invocation N+1 overwrites tensors N still needs. Fails with `accessing tensor output of CUDAGraphs that has been overwritten`. `cudagraph_mark_step_begin()` would assert something false (previous outputs are *not* dead), and cloning 256× gives back the launch savings. Only fix is compiling the whole 256-step `forward` as one graph — rejected on compile time. Failed loudly, not silently.
- **Full unroll (`torch.compile(model.forward)`): rejected.** ~33k nodes; minutes of compile per shape, recompiled for every `bptt` swept.
- **Recurrent connections (`rec1`/`rec2`): removed.** 32,768 params = 44.7% of the model. Ablation showed unchanged free-run reconstruction RMSE and `Val(post-sw)`, with slightly better step time and memory. State went 5 tensors → 3 `(mem1, mem2, memo)`; leaky membranes with heterogeneous learnable taus are now the sole memory mechanism. Only ~6 of ~30 post-fusion kernel launches removed, hence the modest speed gain — the model is launch-bound, not FLOP-bound. Pre-removal commit reproduces the test.
- **Leg grouping + per-gait input routing: removed.** The network is now fully connected — every CPG spike to every hidden unit, both hidden layers dense, readout from the whole hidden state to all 8 joints. State is `(B, H)` rather than `(B, 4, H/4)`. Deleted `table_leg_offsets`, `solve_leg_routing`, `routing_matrices`, `plot_leg_alignment`, the `route_P` buffer, `w_self`/`w_cross`/`cross_gain` and the `scipy.optimize` import (~100 lines). `LEG_COLS`/`SERVO_BASE` survive as presentation and servo wiring only. Class renamed `LegGroupedSNN` → `StatefulSNN`. Note going dense 4×s `w2`/`w_read`: flat h=128 ≈ 43,784 params (comparable to the grouped h=256's 40,457), flat h=256 ≈ 153,096.
- **`forward(return_spikes=True)`: deleted.** Dead code that indexed `state[1]`/`state[3]`, i.e. the removed spike state slots. Restoring hidden-layer rasters needs a separate non-exported path, since making `step` return spikes would change the ONNX signature.

## Architecture ideas not yet tried

- **Hybrid gait conditioning (discrete pattern + continuous parameters).** Today the gait is a single integer index into a FiLM embedding, over-allocated to `--max_gaits` rows so parameter shapes stay fixed as the gait count changes. That handles *growth* but not *structure*: some gait variation is genuinely categorical (tripod vs ripple vs wave are different footfall sequences and interpolating them is meaningless) while some is continuous (`tripod` vs `tripod_huge` is one pattern at two amplitudes; leg height and step length likewise). A small discrete embedding for pattern class concatenated with continuous inputs for the metric parameters would give interpolation between amplitudes for free and stop `n_gaits` appearing in the architecture at all. Implementation: replace `Embedding(max_gaits, 2H)` with `Embedding(n_patterns, d) ⊕ continuous_dims → Linear → ReLU → Linear(·, 2H)`. The nonlinearity matters — a bare `Linear` on a code is additive in the code, which for binary gait indices would force `FiLM(0011) = FiLM(0001) + FiLM(0010) − b`, imposing arbitrary structure over a meaningless labelling.
- **Reintroduce leg grouping properly.** Removed along with routing because layers 2+ being block diagonal made the network four independent sub-networks with no principled way to align each group to a CPG neuron. A better alignment mechanism is planned.
- **Parallel scan over timesteps.** The only remaining structural lever on speed. `mem_t = β·mem_{t-1} + cur_t` is a linear recurrence and is parallelisable by associative scan in O(log T) depth, but the spike-and-reset makes it nonlinear. Would need the reset approximated or restructured — research, not optimisation.

## Known follow-ups outside this list

- **`inference.py` is broken** by the recurrence removal: `StatefulSNNPredictor` hardcodes 5 state tensors (`state_names_in`, `[z() for _ in range(5)]`, `out[1:]`). Needs updating to 3.
- Old `.onnx` / `best_model.pt` artifacts are incompatible with the current architecture. Retrain.