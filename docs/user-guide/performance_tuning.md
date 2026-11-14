# 性能调优

固定数据、模型、batch 形状、精度、设备、优化器和依赖版本，每次只修改一个参数；预热与正式采样分开，保存实际构建指纹和工作负载。基准运行方法见 [benchmark 指南](../benchmarks/README.md)。

## 数据管线

先用 `num_workers=0` 建立内联基线，再逐步增加少量 worker；按端到端吞吐判断收益。`prefetch_batches` 是已调度但未提交的总批次容量，不是每个 worker 的容量或内存硬上限。增加预取会占用更多内存并增加需要重放的工作。

固定 sampler seed，显式设置 epoch；随机 factory 使用稳定的 `DataWorkerContext` 和样本身份，避免可变全局 RNG。只读 Tensor 可共享，需要原位修改的样本由调用方复制。

CUDA 输入可使用 `pin_memory` 和显式 `target_device`。worker 负责验证/pin，`next()` 的调用线程负责传输；`non_blocking` 不创建隐式 stream，也不代替跨 stream 生命周期同步。边界见[数据 API](../api/data.md)与[批次变换](../api/batch_consumer_transform.md)。

## 训练配置

有效 batch 超过设备容量时，使用梯度累积，并按任务核对损失归一化：

```cpp
jyppx::liuding::training::TrainerOptions options;
options.accumulate_steps = 4;
options.gradient_clip_norm = 1.0;
```

通过 `TrainerOptions::precision` 显式选择精度。CUDA FP16 需要启用的 GradScaler 及其检查点参与者；BF16 不使用 loss scaling。溢出跳步不推进成功更新计数或 batch 调度器，见[混合精度教程](../tutorials/05_mixed_precision.md)。改变 batch、精度或参数布局时重新核对误差与任务质量。

## 选择测量入口

| CMake target | 用途 |
| --- | --- |
| `jyppx_bench_data_pipeline` | worker、factory 耗时、取批延迟和在途队列 |
| `jyppx_bench_trainer_throughput` | 固定训练负载吞吐 |
| `jyppx_bench_image_training` | 图像读取、缓存、变换、组批到真实训练更新 |
| `jyppx_bench_memory_usage` | 数据 worker 配置的驻留内存观测 |
| `jyppx_bench_optimizer_convergence` | 固定优化器工作负载 |
| `jyppx_bench_cuda_training` | CUDA FP32 或带 GradScaler 的 FP16 训练 |
| `jyppx_bench_cuda_optimizer` | 排除模型与数据加载的 CUDA 优化器计时 |

目标定义见 [tests/benchmarks/CMakeLists.txt](../../tests/benchmarks/CMakeLists.txt)，程序文件使用 `liuding-forge-bench-*` 前缀。启用 `JYPPX_LIUDING_BUILD_BENCHMARKS` 及对应组件后显式构建；CUDA 基准使用 `scripts/benchmarks/run_cuda_baseline.py` 采样。

分别记录源准备、完整 fit 与端到端耗时。producer 的累计并行时间不能相加为墙钟时间，`next()` 耗时也不等于纯队列等待。CUDA 计时明确同步边界；allocator allocated/reserved、进程 RSS 和逻辑 Tensor 字节是不同口径。只比较相同环境与工作负载的结果，微型随机数据不提供实际任务质量或跨机器性能结论。
