# Benchmark results

Regenerated 2026-08-05T20:07:23+00:00 from `bench_results.jsonl` (6 rows).

**Compare `M sTS/s` (million sample-timesteps per second), not `ms/step`.** A larger batch does proportionally more work per step, so `ms/step` makes a faster config look slower. `M sTS/s = batch * bptt / sec`.

`first step` includes compilation and is a fixed per-shape cost, not throughput. `est epoch` uses the `chunks_per_epoch`/`val_chunks` passed at benchmark time.

## NVIDIA GeForce RTX 3050 Laptop GPU

### Performance

| variant | commit | batch | bptt | hidden | compile | compiled? | warm/meas | ms/step | M sTS/s | speedup | peak GiB | first step s | val ms | est epoch s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eager_b32 | 72c85ad* | 32 | 256 | 256 | eager | n/a | 20/50 | 1480.48 | 0.006 | 1.00x | 0.16 | 2.64 | 475.45 | 63.02 |
| eager_b128 | 72c85ad* | 128 | 256 | 256 | eager | n/a | 20/50 | 1551.26 | 0.021 | 3.82x | 0.59 | 1.64 | 497.44 | 66.03 |
| compile_b32 | 72c85ad* | 32 | 256 | 256 | default | yes | 20/50 | 489.62 | 0.017 | 3.02x | 0.11 | 3.63 | 128.20 | 20.61 |
| compile_b128 | 72c85ad* | 128 | 256 | 256 | default | yes | 20/50 | 382.37 | 0.086 | 15.49x | 0.40 | 1.33 | 134.49 | 16.37 |
| compile_b256 | 72c85ad* | 256 | 256 | 256 | default | yes | 20/50 | 525.24 | 0.125 | 22.55x | 0.79 | 1.59 | 136.83 | 22.10 |
| compile_b512 | 72c85ad* | 512 | 256 | 256 | default | yes | 20/50 | 536.69 | 0.244 | 44.14x | 1.56 | 2.17 | 160.53 | 22.75 |

Speedup is relative to `eager_b32` (2026-08-05T19:59:14+00:00). `*` on a commit means the working tree was dirty — that row is not reproducible from the commit alone.

`compiled?` verifies the variant actually compiled rather than silently falling back to eager after exhausting Dynamo's recompile limit. **FELL BACK** rows report eager timings under a compiled label and must not be compared. `warm/meas` is warmup/measured iteration counts — a small `meas` means the median is a small-sample statistic.

### Numerics

Only compare loss values **within** a `nkey` group — a different batch or bptt legitimately changes the loss trajectory. A dash means the step was not reached; `nan` or `inf` means the loss actually went non-finite. `varying` = no means the stacked output collapsed (suspect CUDA-graph output aliasing) and the row should be discarded.

| variant | commit | nkey | loss@1 | loss@10 | loss@50 | loss@100 | loss@200 | loss@500 | 1st nonfinite | varying | params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eager_b32 | 72c85ad | 89a0a662ed | 65.228745 | 2.687127 | 0.357140 | — | — | — | — | yes | 73,225 |
| eager_b128 | 72c85ad | 1776b9530e | 68.390335 | 3.566397 | 0.367278 | — | — | — | — | yes | 73,225 |
| compile_b32 | 72c85ad | 89a0a662ed | 65.228722 | 5.610857 | 0.257647 | — | — | — | — | yes | 73,225 |
| compile_b128 | 72c85ad | 1776b9530e | 68.390076 | 3.319583 | 0.404944 | — | — | — | — | yes | 73,225 |
| compile_b256 | 72c85ad | b1b631a076 | 65.096909 | 3.775005 | 0.318218 | — | — | — | — | yes | 73,225 |
| compile_b512 | 72c85ad | b90054a825 | 61.849525 | 4.065162 | 0.277278 | — | — | — | — | yes | 73,225 |
