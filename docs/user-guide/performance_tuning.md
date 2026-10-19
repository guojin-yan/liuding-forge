# Performance Tuning Guide

JYPPX performance depends on the input pipeline, batch shape, device, and
optimizer state. Change one setting at a time and record throughput with the
LibTorch and compiler versions.

## DataPipeline optimization

Start with `num_workers = 0` as a baseline. Increase it until the device is
fed continuously; a useful starting point is logical CPU cores minus two,
capped by the number of batches that can be in flight. `prefetch_batches`
is the total scheduled-but-uncommitted batch capacity for the pipeline. Start
with two to four batches and increase it only when measurements show consumer
starvation; larger values consume memory and enlarge bounded replay work.

If the desired effective batch does not fit device memory, use accumulation:

```cpp
jyppx::training::TrainerOptions options;
options.accumulate_steps = 4;
options.gradient_clip_norm = 1.0;
```

Use a fixed sampler seed and pass the logical epoch to `set_epoch()` for
deterministic shuffled experiments. A deterministic batch factory must derive
random work from `DataWorkerContext::worker_seed`. Custom datasets should
return cloned samples when a shared backing tensor could be mutated.

For CUDA input, return CPU tensors from the batch factory and enable
`pin_memory` with a CUDA `target_device`. Workers validate and pin batches;
the thread calling `next()` performs the device transfer on its current CUDA
stream. Install a custom stream guard around `next()` when input copies must be
ordered on a non-default training stream. `non_blocking` requests asynchronous
transfer but does not create or synchronize a hidden stream.

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

Profile a complete epoch, separating pipeline wait from forward/backward time.
The DataPipeline benchmark simulates preprocessing:

```text
build-framework/tests/benchmarks/Release/jyppx_bench_data_pipeline.exe \
  --work-ms 2 --batches 64 --batch-size 32
```

`bench_trainer_throughput` reports samples per second for batch sizes 16, 32,
64, and 128. `bench_optimizer_convergence` emits 100 loss values for SGD,
Adadelta, Adagrad, Adam, AdamW, LAMB, NAdam, and RMSprop.
`bench_sgd_adapter_overhead` and
`bench_adam_adapter_overhead` measure the local median cost of adapter
preflight relative to direct LibTorch execution. The equivalent
`bench_adamw_adapter_overhead` covers decoupled AdamW, while
`bench_rmsprop_adapter_overhead` covers centered RMSprop with momentum and
`bench_adagrad_adapter_overhead` covers decayed Adagrad with a nonzero initial
accumulator.
The supported LibTorch C++ candidates expose neither NAdam, Adadelta, nor LAMB,
so `bench_nadam_native_throughput`, `bench_adadelta_native_throughput`, and
`bench_lamb_native_throughput` report
absolute median time per step instead of invalid adapter/native ratios.
`bench_memory_usage` reports resident bytes for worker counts 0, 1, 4, and 8.
Compare PyTorch with the same seed, model, tensors, dtype, and steps. These are
engineering measurements, not universal limits.

For a real-GPU end-to-end baseline, build `jyppx_bench_cuda_training` against
CUDA LibTorch and invoke it through `tools/run_cuda_baseline.py`. The frozen
workload covers FP32 with one SGD optimizer and FP16 autocast with two disjoint
SGD optimizers plus one Trainer-owned GradScaler cycle. Each process verifies
the committed Trainer and optimizer steps before publishing throughput, mean
step time, peak allocated VRAM, and peak reserved VRAM. Timing is enclosed by
CUDA synchronization. The runner keeps raw process output, samples transient
GPU state, requires at least five stable samples per mode, and binds references
to the complete build/workload/GPU environment.

For optimizer-only diagnosis, build `jyppx_bench_cuda_optimizer` and pass
`--suite optimizer` to the same runner. This removes data loading, model
forward/backward, and Trainer orchestration from the timed region while keeping
real CUDA synchronization and allocator peaks. It runs all eight optimizers by
default. Compare each mode only with a reference created from the same mode and
environment key; absolute timings across algorithms reflect different update
semantics and state sizes and are not directly interchangeable.
