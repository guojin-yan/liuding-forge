# Performance Tuning Guide

JYPPX performance depends on the input pipeline, batch shape, device, and
optimizer state. Change one setting at a time and record throughput with the
LibTorch and compiler versions.

## DataLoader optimization

Start with `num_workers = 0` as a baseline. Increase it until the device is
fed continuously; a useful starting point is logical CPU cores minus two,
capped by the number of batches that can be in flight. `prefetch_factor`
controls queued batches per worker. Values of 2 or 4 are usually sufficient;
larger values consume memory without helping CPU-bound datasets.

If the desired effective batch does not fit device memory, use accumulation:

```cpp
jyppx::training::TrainerOptions options;
options.accumulate_steps = 4;
options.gradient_clip_norm = 1.0;
```

Use a fixed `seed` for deterministic shuffled experiments. Custom datasets
should return cloned samples when a shared backing tensor could be mutated.

## Training optimization

Use Adam or AdamW as a robust baseline and tune learning rate before changing
momentum or betas. Larger batches may permit a larger learning rate, but check
the first epochs rather than assuming linear scaling. Use `drop_last` when a
model requires equal batch shapes.

Select precision explicitly through `TrainerOptions::precision`. CUDA float16
requires an enabled `GradScaler` and its checkpoint participant in
`FitRequest`; CPU/CUDA bfloat16 deliberately run without loss scaling. There
is no implicit `use_amp` option. Overflowed float16 updates do not advance the
Trainer global step or scheduler and publish `optimizer.applied=0`.

## Profiling and memory

Profile a complete epoch, separating loader wait from forward/backward time.
The DataLoader benchmark simulates preprocessing:

```text
build-framework/tests/benchmarks/Release/jyppx_bench_dataloader.exe \
  --work-ms 2 --batches 64 --batch-size 32
```

`bench_trainer_throughput` reports samples per second for batch sizes 16, 32,
64, and 128. `bench_optimizer_convergence` emits 100 loss values for SGD,
Adam, and AdamW. `bench_sgd_adapter_overhead` measures the local median cost of
the adapter preflight relative to direct LibTorch SGD. `bench_memory_usage`
reports resident bytes for worker counts 0, 1, 4, and 8. Compare PyTorch with
the same seed, model, tensors, dtype, and steps. These are engineering
measurements, not universal limits.
