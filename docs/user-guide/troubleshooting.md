# Troubleshooting Guide

This guide covers common configuration, training, and deployment failures. When reporting a problem, include the CMake command, LibTorch version, device, and first exception message.

## Build issues

### LibTorch cannot be found

If CMake reports `Could NOT find Torch`, point `CMAKE_PREFIX_PATH` at the
`libtorch` directory containing `share/cmake/Torch/TorchConfig.cmake`:

```bash
cmake -S . -B out/build/release -DCMAKE_PREFIX_PATH=/opt/libtorch \ -DTorch_DIR=/opt/libtorch/share/cmake/Torch
```

Reuse the existing build directory and reconfigure with explicit
`CMAKE_PREFIX_PATH` and `Torch_DIR` when changing LibTorch, then rebuild the
affected targets. A compiler or generator change requires a separate build directory. The bundled layout is detected automatically when installed in
`third_party/install/libtorch`.

### MSVC crashes in a Torch DLL

Official Windows LibTorch packages use the release dynamic CRT. JYPPX selects
`/MD` and `_ITERATOR_DEBUG_LEVEL=0` for every configuration. Do not override
those settings or mix Debug and Release LibTorch binaries. Reconfigure from a clean build directory after changing the toolchain.

## Runtime issues

### Windows 训练完成后进程不退出

Windows 的 LibTorch 2.14.1 OpenMP 版本需要在主线程进入训练或配置线程数量前调用 `jyppx::liuding::initialize_runtime()`。该入口初始化辅助 Caffe2 线程池，避免 Windows 终止其他线程后，其 DLL 退出清理再次等待已终止线程持有的锁。框架 CLI 已接入；直接使用 C++ SDK 的程序应在 `main` 开始时调用：

```cpp
#include <jyppx/liuding/core/runtime.hpp>
#include <ATen/Parallel.h>

int main() {
  jyppx::liuding::initialize_runtime();
  at::set_num_threads(4);
  // 在这里创建模型并执行训练。
}
```

初始化保留原有 intra-op 和 inter-op 配置，重复调用不会覆盖后续设置；OpenMP/MKL 仍按调用方设置并行计算，辅助 pthreadpool 内的工作改为同步执行。其他运行时配置无需调整。若程序或其他库已先初始化多线程辅助池，该入口不能事后修复；应调整启动顺序，不能用强制退出或忽略超时替代正常资源释放。

### CUDA out of memory

Reduce `batch_size`, use `TrainerOptions::accumulate_steps`, or configure a CPU
`SingleDeviceStrategy`. A smaller pipeline `prefetch_batches` value limits
queued batches. Release references to old batches before the next iteration.

### Loss becomes NaN or Inf

Check inputs and targets with `jyppx::liuding::is_finite`, lower the learning rate, and
inspect normalization divisors. With AMP, use `GradScaler`; an overflow sets
`found_inf()` and skips the optimizer step while backing off the scale.

### DataPipeline order or worker failures

DataPipeline output order follows the sampler even when workers finish out of order. Use `SequentialSampler` to isolate ordering bugs, or use a fixed sampler seed and explicit `set_epoch()` calls for repeatable shuffled runs. A worker's non-JYPPX exception is reported as `jyppx::liuding::BackendError` by `next()`, so catch it at the job boundary and reproduce with `num_workers = 0`. Dataset access and captured factory state must be thread-safe when workers are enabled.

### Checkpoint does not load

Prepare a fresh domain with identical Trainer options and participant IDs, topology, model shapes, optimizer groups, scheduler tree, callbacks, data and RNG scope. Then pass a complete `CheckpointStateSource` to
`restore_checkpoint()`. Each participant accepts only its declared current
schema and any explicitly supported N-1 reader. Unknown versions, incompatible
topology and partial participant sets are rejected. Keep the original domain
and checkpoint available for rollback; use an explicit offline migration or
start a new checkpoint lineage when no supported reader exists.

## Performance diagnosis

Warm up for one epoch before measuring. Compare `num_workers = 0, 1, 2, 4`
and adjust total `prefetch_batches`; too many workers add context-switch
overhead.
The benchmark executables under `tests/benchmarks` print CSV that can be saved
with the commit and compared across machines.
