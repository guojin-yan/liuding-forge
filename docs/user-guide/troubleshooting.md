# 故障排查

报告问题时提供 CMake 配置、LibTorch 版本、设备、实际启动命令和首个异常。环境组合以[兼容矩阵](../../compatibility.json)为准。

## 构建问题

### 找不到 LibTorch

将 `CMAKE_PREFIX_PATH` 指向包含 `share/cmake/Torch/TorchConfig.cmake` 的 LibTorch 目录；必要时显式设置 `Torch_DIR`：

```sh
cmake -S . -B out/build/release -DCMAKE_PREFIX_PATH=/path/to/libtorch -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
```

保留已有兼容缓存的生成器、编译器和依赖路径。改变编译器或生成器时使用新的构建目录；修改依赖后重建受影响目标。默认依赖布局可使用 `third_party/install/libtorch`。

### MSVC 在 Torch DLL 内崩溃

Windows 官方 LibTorch 二进制使用 Release 动态 CRT。本项目选择 `/MD` 和 `_ITERATOR_DEBUG_LEVEL=0`，所有配置须保持一致；避免混用 Debug/Release 的 LibTorch 和不匹配的标准库布局。

## 运行问题

### Windows 训练完成后进程不退出

在主线程首次配置或使用 LibTorch、启动 worker 之前调用 `jyppx::liuding::initialize_runtime()`。该入口针对 Windows LibTorch 2.14.1 的适用 OpenMP 配置初始化辅助线程池，规避 DLL 退出期间的锁等待。CLI 已调用；直接消费 SDK 的应用应在 `main` 开始时调用：

```cpp
#include <jyppx/liuding/core/runtime.hpp>
#include <ATen/Parallel.h>

int main() {
  jyppx::liuding::initialize_runtime();
  at::set_num_threads(4);
  // 创建模型与管线，然后训练。
}
```

重复调用不会覆盖后续线程设置，OpenMP/MKL 仍按调用方配置计算。若其他代码已提前创建多线程辅助池，应调整启动顺序；该入口不能事后修复已创建的池。

### CUDA 显存不足

减小 batch，使用 `TrainerOptions::accumulate_steps`，或将 `SingleDeviceStrategy` 显式配置为 CPU。减少 `prefetch_batches` 可限制在途批次数；及时释放旧批次和不再使用的输出。预取容量与 Tensor 字节预算不是全部进程显存的硬上限。

### loss 为 NaN 或 Inf

核对输入、标签、归一化分母及学习率，可用 `jyppx::liuding::is_finite` 检查 Tensor。CUDA FP16 配置需要启用的 GradScaler 及其参与者；溢出周期跳过优化器更新并降低 scale，计数和调度不得当作成功更新推进。见[混合精度教程](../tutorials/05_mixed_precision.md)。

### 数据顺序或 worker 异常

DataPipeline 按 sampler 顺序提交。先用 `num_workers=0` 排查 factory，再以固定 seed 和显式 `set_epoch()` 检查并发路径；worker 访问的数据和捕获对象必须线程安全。非框架异常会由 `next()` 转为 `BackendError`。不要在多个 owner 中并发修改同一管线。

### 检查点恢复失败

新建与原任务兼容的域，保持模型形状、参与者 ID、优化器组、scheduler 树、精度、数据身份、worker 拓扑和 RNG 范围一致。将完整 `CheckpointStateSource` 交给 `restore_checkpoint()`；未知 schema、不兼容拓扑和缺失参与者会被拒绝。没有受支持的读取或迁移路径时，保留原应用和检查点用于回退，或建立新的训练任务。

## 性能排查

固定模型、数据、输入形状、dtype、设备、线程和依赖，每次只调整一个参数，分别记录预热与测量。先比较内联及少量 worker，再调整预取；CUDA 计时须明确同步边界。操作方法见[性能调优](performance_tuning.md)和[benchmark 指南](../benchmarks/README.md)。
