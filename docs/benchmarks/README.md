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

资源 profile、存储身份、设备要求、来源校验和证据布局见 [通用资源采样](../../scripts/benchmarks/RESOURCE_BASELINE.md)。CUDA 训练和优化器套件使用 scripts/benchmarks/run_cuda_baseline.py。

## ROCm 训练与优化器

选择已验证的 ROCm LibTorch 后，开启 `JYPPX_LIUDING_BUILD_BENCHMARKS=ON`，编译 `jyppx_bench_rocm_training` 和 `jyppx_bench_rocm_optimizer`。它们复用固定工作负载，使用 HIP 设备查询、同步和 LibTorch HIP caching allocator；CUDA 构建继续提供原有 CUDA 目标和模式，CPU 构建不生成这两类设备目标。

~~~powershell
cmake --build out/build/rocm --config Release --parallel 2 `
  --target jyppx_bench_rocm_training jyppx_bench_rocm_optimizer
python scripts/benchmarks/run_rocm_baseline.py --suite training `
  --benchmark out/build/rocm/tests/benchmarks/liuding-forge-bench-rocm-training.exe `
  --output-dir out/baseline/rocm-training
python scripts/benchmarks/run_rocm_baseline.py --suite optimizer `
  --benchmark out/build/rocm/tests/benchmarks/liuding-forge-bench-rocm-optimizer.exe `
  --output-dir out/baseline/rocm-optimizer
~~~

多配置生成器在可执行文件路径中增加 `Release/`。开发 runner 要求空输出目录，每模式至少一个独立预热进程和五个独立采样进程，默认每进程限时 180 秒；失败或超时保留原始 stdout、stderr 和退出记录。训练模式为 `rocm-fp32-single-sgd` 与 `rocm-fp16-multi-sgd-amp`，各 200 步，FP16 保留 FP32 主权重并检查两个优化器和 GradScaler 的更新边界；优化器套件覆盖八类算法的固定 FP32 稠密梯度 200 步。可用重复的 `--mode` 选择部分模式。

`--identity` 原生入口报告可见设备零的 HIP UUID、PCI 地址、架构、内存及编译/运行时/API 版本；runner 核对采样前后及每个样本的设备、构建和线程身份，另记录 OS、CPU、Windows 显示驱动适配器列表和可执行文件 SHA-256。HIP driver API 版本不等于 Windows 显示驱动版本，适配器列表也不建立 HIP UUID 映射。`samples.json` 保留原始指标，`statistics.json` 给出中位数、p95、CV 和稳定性判定；默认 CV 上限 10%，不稳定结果不能写成 reference。`--reference` 和 `--write-reference` 仅用于同一可执行文件、源码、工作负载与环境，不能跨 CUDA/ROCm、设备或构建比较。

训练计时覆盖 `Trainer::fit()` 与前后 HIP 同步，优化器计时覆盖更新循环与前后同步；不包含进程启动和模型准备。allocator 峰值包含计时起点已有分配，排除 driver/context 和此前准备期峰值。当前没有功耗、频率、温度及系统噪声采样，不能从本协议推导跨设备性能优势、任务准确率或长时间稳定性；图像和通用 resource runner 的设备支持分别说明，不由本入口推定。

## ROCm 稳定性与受控显存压力

启用 benchmark 后编译 `jyppx_bench_rocm_stability`，使用下面的开发脚本在三个独立原生进程中验证 FP32、FP16 autocast+GradScaler 和 BF16 autocast；参数保留 FP32。默认每精度 180 秒、每 15 秒执行一次文件检查点保存和恢复，最多一百万提交步，父进程另设请求时长加 120 秒的超时。提前达到步数上限、没有完成恢复或身份变化均失败；`--output-dir` 必须为空，脚本保留原始进程输出、逐次资源记录和最后一个检查点。

~~~powershell
cmake --build out/build/rocm --config Release --parallel 2 --target jyppx_bench_rocm_stability
python scripts/benchmarks/run_rocm_stability.py `
  --benchmark out/build/rocm/tests/benchmarks/liuding-forge-bench-rocm-stability.exe `
  --output-dir out/stability/rocm --seconds 180 --checkpoint-seconds 15 --pressure-mib 2048
~~~

固定 Linear/AdamW/StepLR 合成负载核对有限损失、梯度、实际参数更新和计数。每次通过有界 `FileCheckpointStore` 保存模型、优化器、调度器和 scaler，在全新对象中恢复，再与原对象各继续三步，逐步比较参数、梯度和控制状态；随后使用恢复对象继续。无随机训练算子，不验证 RNG 或数据管线恢复。检查点 artifact 上限 64 MiB，字段上限 16 MiB，固定参与者使用更小预算；同一文件反复原子替换，保留最后一次。

默认创建并实际写入 2 GiB 的设备张量，在每个恢复点释放并重新分配；请求上限 4 GiB，且不得超过当时空闲显存的四分之一，申请后至少保留 2 GiB，预算不足明确失败。报告记录 allocated/reserved 当前值与峰值、设备空闲显存、当前和峰值 RSS；核对每次释放的 live payload，固定负载允许最多 16 MiB 的活跃分配增长。缓存 reserved 允许保留，RSS 只观察、不设内存泄漏判定阈值，设备空闲值也不构成硬配额。

每个进程还将自身 allocator 配额设为 64 MiB，确认 128 MiB 分配以 `OutOfMemoryError` 拒绝，恢复配额后执行数值和训练检查；另验证同步矩阵形状拒绝后的继续执行，FP16 注入一次非有限梯度并核对跳步、scale 回退和后续有限更新。它不耗尽物理显存，不验证非法设备访问、驱动复位或挂起恢复。压力、检查点 I/O 和诊断计入持续墙钟时间；本入口不用于性能统计，资格限于实际完成的时长、精度和固定负载。`--precision` 可选择部分精度；短时冒烟必须按实际秒数记录，不能计为默认持续验证。

需要更长验证时，可显式设置 `--seconds 7200 --max-steps 20000000`，每个精度在独立进程持续两小时；两项上限分别为 7200 秒和 2000 万次更新，默认预算保持不变。步数预算先耗尽而未达到请求时长时仍失败，不会把提前结束计作完成。仅执行某一精度时使用 `--precision fp32` 等选择，并保留该精度的实际时长与文件恢复记录。

## 图像读取到训练更新

启用 `JYPPX_LIUDING_BUILD_DATA_IMAGE=ON` 与 benchmark 后，编译 `jyppx_bench_image_training`。输入是至少两个非空类别的本地 ImageFolder 目录，每个类别下放 JPEG/PNG 文件。下面的 runner 先读取所有选中图像计算内容摘要，使每次明确采用 warm-cache 测量，再启动一个独立预热进程与五个独立采样进程；`--output` 必须指向代码仓库外的新文件。

采样要求干净源码和与当前提交完全一致的二进制指纹，提交后先复用原配置增量重建。每个子进程默认限时 120 秒、每路输出 1 MiB，`--timeout-seconds` 可显式调整；原始日志、退出状态和样本保存在输出文件相邻的 `.evidence` 目录，失败也保留。规定的吞吐、fit/总时长、RSS 和 GPU allocator 重复指标要求 CV 不超过 10%，失败时仍写入报告并返回非零，不丢弃离群样本。

ROCm 继续采用 `--device cuda:0` 和既有 `cuda-fp16` 精度名，训练前后使用 HIP 同步。原生 `--identity` 查询与各进程绑定 HIP UUID、PCI、gfx、实际版本及线程设置；ROCm 显存字段为 `peak_rocm_allocated_bytes` / `peak_rocm_reserved_bytes`，NVIDIA 沿用 `peak_cuda_*`。报告另保留 OS/显示驱动独立观测及相关环境变量；本入口未建立 GPU 功耗/频率控制资格，不提供可比较 reference。

~~~powershell
cmake --build out/build/benchmark-release --config Release --target jyppx_bench_image_training --parallel 2
python scripts/benchmarks/run_image_training.py `
  --binary out/build/benchmark-release/tests/benchmarks/Release/liuding-forge-bench-image-training.exe `
  --image-root F:/datasets/classification-train `
  --output F:/benchmarks/image-training-run.json `
  --batch-size 8 --image-size 32 --epochs 1 --threads 1
~~~

该工作负载使用单机 `Trainer::fit()`，实际执行图像读取、双线性 resize、ImageBatch 组批、分类损失、反向传播和 SGD 更新。默认 `--source-mode image-folder` 每个 epoch 从文件解码；`--source-mode tensor-cache` 在训练前一次性解码为 CPU Tensor，再通过 `TensorImageDataset` 共享读取；`--source-mode tensor-cache-resized` 在缓存前执行相同的固定 resize，训练时直接读取较小的 Tensor，仅适用于该分类基准的固定输入形状。三种模式使用同一文件清单、像素策略和训练步骤；缓存模式可用 `--max-cached-bytes` 限制解码、预变换和常驻 Tensor 的已知逻辑字节占用，默认 512 MiB，超限失败。预变换成本计入源准备和总时间，不计入 fit 时间。缓存仍占进程内存，不能把局部变换节省直接算作端到端性能收益。

默认 `--device cpu --model linear64` 使用轻量分类器。启用可选 `models` 组件后，可通过 `--model` 选择以下随机初始化模型：

| 模型系列 | selector | 输入尺寸 |
| --- | --- | --- |
| ResNet18 | `resnet18` | 至少 64 |
| DINOv2 ViT-S/B/L/G | `vit_small_patch14_dinov2`、`vit_base_patch14_dinov2`、`vit_large_patch14_dinov2`、`vit_giant_patch14_dinov2` | 14 的正整数倍 |
| DINOv3 ViT-S/S+/B/L/L+/H+ | `dinov3_vits16`、`dinov3_vits16plus`、`dinov3_vitb16`、`dinov3_vitl16`、`dinov3_vitl16plus`、`dinov3_vith16plus` | 16 的正整数倍 |
| DINOv3 ConvNeXt-T/S/B/L | `dinov3_convnext_tiny`、`dinov3_convnext_small`、`dinov3_convnext_base`、`dinov3_convnext_large` | 至少 32；模型内部向下裁到 32 的倍数 |

CUDA LibTorch 构建可显式选择 `--device cuda:0`：CPU worker 组批后通过框架的 pin/transfer 阶段传输图像与标签，模型和优化器在同一 GPU 上更新；计时在 `Trainer::fit()` 前后同步 CUDA，并分别记录该区间 LibTorch allocator 的 allocated/reserved 显存峰值。CPU 模式的显存字段明确为不适用。模型均为随机初始化，不下载预训练权重，只用于性能测量和参数更新检查，不报告分类准确率或收敛质量。

精度默认 `--precision fp32`。CUDA 可选 `--precision cuda-fp16`（FP32 参数，FP16 autocast）或 `--precision cuda-fp16-storage`（FP16 参数，FP16 autocast）；两者均使用动态 GradScaler，初始 scale 为 128，并校验所有预期 SGD 更新实际发生。FP16 参数存储没有额外的 FP32 主参数，数值与内存语义不同，比较时应使用同一精度模式。大型变体需要显式选择并串行运行，按实际参数、精度、输入尺寸和优化器状态准备资源；7B 不在该基准入口中。报告记录完整参数元素数、精度、参数 dtype、初始/最终 scale 和优化步数；微型随机输入结果不代表真实任务质量或性能优势。

`--workers 0` 是内联加载，`--workers 1..16` 使用有界 worker 预取，`--prefetch-batches 1..4` 控制在途批次数；比较不同配置时分别运行 runner 并保留各自结果。输出分别包含源索引/可选缓存准备时间、完整 fit 时间、含运行时和模型准备的总工作负载时间，以及两种时间口径各自的吞吐、常驻 Tensor 逻辑字节、`DataPipeline::next()` 消费端耗时、各 producer 阶段累计工作时间和进程寿命内峰值 RSS；训练后参数摘要用于核对三种模式的计算结果。并行时 producer 阶段可能重叠，累计时间不能直接相加为训练耗时，`next()` 耗时也不是纯队列等待；Tensor 缓存模式的读取阶段不进行文件解码，预缩放缓存的训练阶段也不重复执行固定 resize。CUDA allocator 峰值包含计时开始时仍驻留的模型/优化器资源及训练期分配，不包含驱动/context 或模型准备阶段的峰值。当前不测冷缓存、其他视觉任务或等价 PyTorch 基线，不能据此宣称速度、内存或显存优势。

`--diagnostics off` 为默认基线；`counters` 仅记录既有检查的计数；`sampled` 每 16 次优化尝试测一次梯度范数并计时数据读取；`each-step` 每次尝试测范数并计时。报告保留采样次数、扫描元素、累计扫描时间和已知临时 payload 峰值。对照时保持模型、数据、设备和精度相同，同时核对最终参数摘要；这些模式用于测量诊断开销，不构成库的整体性能优势证据。

## 范围

benchmark 可执行文件位于 tests/benchmarks/。逻辑 CMake target 保留 jyppx_bench_* 技术身份，生成文件使用 liuding-forge 产品前缀。benchmark 输出属于本地证据，应写入被忽略的 out/ 或 CI artifact 存储。
