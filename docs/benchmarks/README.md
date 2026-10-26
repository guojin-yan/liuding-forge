# Benchmark 操作说明

本文档只说明开发人员如何构建和运行 benchmark target。历史测量、审计结论和原始结果文件统一保存在代码仓库外的方案记录中；不要把带日期的结果文件或运行日志放入 docs/。

## 构建和运行

启用测试和 benchmark 后配置构建目录，再使用当前平台的主构建目录编译 benchmark target：

~~~powershell
cmake -S . -B out/build/benchmark-release -DCMAKE_BUILD_TYPE=Release `
  -DJYPPX_LIUDING_BUILD_TESTS=ON `
  -DJYPPX_LIUDING_BUILD_BENCHMARKS=ON `
  -DCMAKE_PREFIX_PATH=third_party/install/libtorch
cmake --build out/build/benchmark-release --config Release --parallel 2
~~~

Windows 基线协议使用 scripts/benchmarks/Run-Baseline.ps1。脚本执行一次预热和多次独立采样，记录构建与环境身份，并将可替换的运行输出写入被忽略的 out/baseline/：

~~~powershell
pwsh -NoProfile -File scripts/benchmarks/Run-Baseline.ps1 `
  -BuildDir out/build/baseline-release `
  -OutputDir out/baseline/release
~~~

只有在确实需要隔离的全新构建测量时才使用 -FreshBuild；-ReferenceFile 只能指定与当前工作负载和环境匹配的 reference。runner 会拒绝不稳定样本和环境不匹配，不会把本地测量结果变成跨机器阈值。

资源 profile、存储身份、CUDA 要求、来源校验和证据布局见 scripts/benchmarks/RESOURCE_BASELINE.md。CUDA 训练和优化器套件使用 scripts/benchmarks/run_cuda_baseline.py。

## 图像读取到训练更新

启用 `JYPPX_LIUDING_BUILD_DATA_IMAGE=ON` 与 benchmark 后，编译 `jyppx_bench_image_training`。输入是至少两个非空类别的本地 ImageFolder 目录，每个类别下放 JPEG/PNG 文件。下面的 runner 先读取所有选中图像计算内容摘要，使本轮明确为 warm-cache 测量，再启动一个独立预热进程与五个独立采样进程；`--output` 必须指向代码仓库外的新文件。

~~~powershell
cmake --build out/build/config-windows --config Release --target jyppx_bench_image_training --parallel 2
python scripts/benchmarks/run_image_training.py `
  --binary out/build/config-windows/tests/benchmarks/Release/liuding-forge-bench-image-training.exe `
  --image-root F:/datasets/classification-train `
  --output F:/GitSpace/jyppx-training-framework/v1.0-plan/06_records/phase0_records/evidence/image-training-run.json `
  --batch-size 8 --image-size 32 --epochs 1 --threads 1
~~~

该工作负载使用 CPU 和单机 `Trainer::fit()`，实际执行文件读取、JPEG/PNG 解码、双线性 resize、ImageBatch 组批、分类损失、反向传播和 SGD 更新。`--workers 0` 是内联加载，`--workers 1..16` 使用有界 worker 预取，`--prefetch-batches 1..4` 控制在途批次数；比较不同配置时分别运行 runner 并保留各自结果。输出包含完整 fit 时间、吞吐、`DataPipeline::next()` 消费端耗时、各 producer 阶段累计工作时间和进程寿命内峰值 RSS。并行时 producer 阶段可能重叠，累计时间不能直接相加为训练耗时，`next()` 耗时也不是纯队列等待。当前不测冷缓存、CUDA 显存或其他视觉任务，也不提供跨 PyTorch 优势结论。

## 范围

benchmark 可执行文件位于 tests/benchmarks/。逻辑 CMake target 保留 jyppx_bench_* 技术身份，生成文件使用 liuding-forge 产品前缀。benchmark 输出属于本地证据，应写入被忽略的 out/ 或 CI artifact 存储。
