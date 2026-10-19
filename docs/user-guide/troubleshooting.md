# Troubleshooting Guide

This guide covers common configuration, training, and deployment failures.
When reporting a problem, include the CMake command, LibTorch version, device,
and first exception message.

## Build issues

### LibTorch cannot be found

If CMake reports `Could NOT find Torch`, point `CMAKE_PREFIX_PATH` at the
`libtorch` directory containing `share/cmake/Torch/TorchConfig.cmake`:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/libtorch
```

Remove the old CMake cache after changing LibTorch. The bundled layout is
detected automatically when it is installed in `third_party/install/libtorch`.

### MSVC crashes in a Torch DLL

Official Windows LibTorch packages use the release dynamic CRT. JYPPX selects
`/MD` and `_ITERATOR_DEBUG_LEVEL=0` for every configuration. Do not override
those settings or mix Debug and Release LibTorch binaries. Reconfigure from a
clean build directory after changing the toolchain.

## Runtime issues

### CUDA out of memory

Reduce `batch_size`, use `TrainerOptions::accumulate_steps`, or configure a CPU
`SingleDeviceStrategy`. A smaller pipeline `prefetch_batches` value limits
queued batches. Release references to old batches before the next iteration.

### Loss becomes NaN or Inf

Check inputs and targets with `jyppx::is_finite`, lower the learning rate, and
inspect normalization divisors. With AMP, use `GradScaler`; an overflow sets
`found_inf()` and skips the optimizer step while backing off the scale.

### DataPipeline order or worker failures

DataPipeline output order follows the sampler even when workers finish out of
order. Use `SequentialSampler` to isolate ordering bugs, or use a fixed sampler
seed and explicit `set_epoch()` calls for repeatable shuffled runs. A worker's
non-JYPPX exception is reported as `jyppx::BackendError` by `next()`, so catch
it at the job boundary and reproduce with `num_workers = 0`. Dataset access
and captured factory state must be thread-safe when workers are enabled.

### Checkpoint does not load

Prepare a fresh domain with identical Trainer options and participant IDs,
topology, model shapes, optimizer groups, scheduler tree, callbacks, data and
RNG scope. Then pass a complete `CheckpointStateSource` to
`restore_checkpoint()`. Current schemas intentionally reject old archives and
partial participant sets; use an explicit offline migration or start a new
checkpoint lineage.

## Performance diagnosis

Warm up for one epoch before measuring. Compare `num_workers = 0, 1, 2, 4`
and adjust total `prefetch_batches`; too many workers add context-switch
overhead.
The benchmark executables under `tests/benchmarks` print CSV that can be saved
with the commit and compared across machines.
