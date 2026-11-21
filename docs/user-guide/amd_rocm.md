# AMD GPU 与 ROCm

AMD 后端使用 ROCm 构建的 LibTorch 和匹配 HIP SDK。精确版本、编译器和验证范围见 [compatibility.json](../../compatibility.json) 的 `amd_rocm`；它与 NVIDIA CUDA、CPU 包独立配置。下文主要说明 Windows x64、Clang 24 clang-cl、LibTorch `2.14.0+rocm10.1.0`、ROCm SDK `10.1.0` 与 W7900 gfx1100 的开发基线，覆盖 FP32/FP16/BF16 训练、精确恢复、视觉流程及安装后 GPU 消费。[WSL2 核心与训练](#wsl2-原生核心与训练)有单独的验证范围，其他 Linux 和 GPU 仍需分别验证。

## 构建

在 x64 Visual Studio 开发命令提示符中，使用 SDK 的激活脚本配置 HIP 编译器、运行库和设备数据。设置 `LIBTORCH_ROOT` 为原生 LibTorch 前缀，`ROCM_PATH` 为 HIP SDK 前缀；这两个目录分别包含 Torch 与 HIP 的 CMake 配置，不使用 Conda/Python 包目录作为安装后的运行依赖。

```bat
set "PYTORCH_ROCM_ARCH=gfx1100"
cmake -S . -B out/build/rocm -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="%ROCM_PATH%/lib/llvm/bin/clang-cl.exe" -DCMAKE_CXX_COMPILER="%ROCM_PATH%/lib/llvm/bin/clang-cl.exe" -DTorch_DIR="%LIBTORCH_ROOT%/share/cmake/Torch" -DCMAKE_PREFIX_PATH="%ROCM_PATH%" -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build out/build/rocm --parallel 2
```

先按[贡献指南](../../CONTRIBUTING.md)准备所需的锁定依赖，配置阶段不下载。视觉组件按[视觉套件](../api/vision/index.md)选择。Windows 继续使用 Release CRT 和 `_ITERATOR_DEBUG_LEVEL=0`；clang-cl 和 MSVC 编译器身份分别记录。

## 设备与训练

ROCm LibTorch 沿用 `torch::kCUDA`、`torch::cuda` 和 `cuda:<index>`。C++ 使用 `parse_device("cuda:0")`、`SingleDeviceStrategy`、既有 Trainer 与优化器；视觉配方同样设置 `"device": "cuda:0"`。`hip:0`、`rocm:0` 不作为新设备别名。`cuda_is_available()` 在 ROCm 包中检查 AMD GPU；构建指纹以 `rocm-` 和 `rocm:available:` 标识真实后端。

在主线程首次使用 LibTorch、设置线程数或启动 worker 之前调用 `initialize_runtime()`。CLI 已调用；ROCm 构建通过实际 HIP 后端符号保留 Windows DLL 注册。`cuda_float16`、`cuda_bfloat16` 仍是 GPU 精度枚举名，是否适合任务由对应硬件、算子与实际验证决定。检查点与 RNG 使用既有接口，精确恢复要求匹配模型、数据、策略和运行身份，不承诺跨 CUDA/ROCm 逐位恢复。

本组合已验证 SGD、Adam、AdamW、Adagrad、Adadelta、RMSprop、NAdam 和 LAMB 与十五类学习率调度器的 FP32、FP16 autocast、BF16 autocast 组合，主权重均为 FP32，FP16 启用 GradScaler。验证覆盖标量前向/梯度参考、CPU/GPU 更新对照，以及优化器、调度器和 scaler 恢复后的逐步精确续训；不代表所有算法选项、模型或低精度权重存储都已验证。

代表任务已验证 ResNet18 分类/多标签、YOLO26n 检测/分类/实例分割/姿态/旋转框，以及 FCOS、ViT-Adapter、RT-DETR、语义分割、深度、SimMIM 和 ReID 的 FP32/FP16/BF16 单步训练。检查覆盖有限损失、梯度、实际参数更新及冻结参数保持；随机权重和合成输入不代表官方权重、任务准确率、逐模型续训或性能资格。

固定外部 ResNet50 a1 与 DINOv2-small 权重已验证 CPU/W7900 FP32 严格导入、独立前向参考及冻结主干后的新分类头更新；DINOv2-small 另通过两步梯度与分组 SGD 数值对照。两模型在固定 Fashion-MNIST 子集（训练/测试各 1000 张）上的冻结特征线性分类也通过原质量门槛，准确率分别为 0.81 和 0.855。权重、参考身份和输入范围见兼容矩阵；这些验证不授予完整数据、全量微调或训练恢复资格。额外的 DINOv2 真实首批次 CPU 梯度门禁在本组合仍有一处 patch embedding 梯度超出原容差，尚未通过。

真实空间任务已验证随机初始化 TinyUNet 的 FP32 固定配方训练，以及独立固定 ROCm 产物在 CPU/GPU 上的留出集质量、拆批统计和持久评估恢复。同设备训练另验证第 601 次更新的文件检查点在新进程恢复到第 1200 次更新、全部参与者状态与连续训练相同，以及完成态重放；数据、身份与门禁入口见[空间任务 fixture](../../tests/fixtures/metrics/penn_fudan/README.md)。质量资格限于该固定 25 图测试集。历史训练权重的 SHA 门禁仍失败，ROCm 产物使用单独身份；由预测掩码生成的候选框不代表独立检测模型。

DINOv2/DINOv3 的部分架构已验证 CPU FP32 与 GPU FP16 autocast 的完整文件检查点恢复；DINOv3 ViT-huge 另有 FP16 参数存储的独立验证。检查包含全部模型与优化器状态、数据游标、损失和下一 CPU/GPU 随机数。具体模型、输入和资源边界见兼容矩阵；未列容量、精度、预训练权重和任务准确率须分别验证。

本 ROCm 分发的 CPU FP32 路径另通过 DINOv2-small 与 DINOv3 Huge+ 的独立随机架构数值对照，覆盖严格状态导入、完整 tokens 和 CLS；Huge+ 还核对三层中间输出与原 12 GiB 峰值 RSS 上限。输入分别为 `2×3×224×280` 和 `1×3×35×50`，参考来源及固定身份见兼容矩阵。这些 CPU 对照不授予 GPU 数值、预训练权重或任务质量资格。

## 运行与安装

Windows 构建树会暂存所选 LibTorch 的 DLL、`.kpack` 和 AOTriton 图像；应用仍需提供匹配的 HIP 运行库、数学库和设备数据。保留 HIP SDK 自身的 `.kpack`、数学库数据目录，按 SDK 激活脚本设置 `HIPBLASLT_TENSILE_LIBPATH`、`ROCBLAS_TENSILE_LIBPATH` 及运行库搜索路径。只复制 `torch_hip.dll` 不能构成完整部署。

自行部署时，`torch_hip.dll` 上一级目录保留 `.kpack/torch_gfx1100.kpack`，`aotriton_v2.dll` 相邻目录保留 `aotriton.images`。不设置仅指向 Torch 归档的全局 `ROCM_KPACK_PATH_PREFIX`，它还会影响 HIP 数学库的归档选择。缺少设备数据时，设备发现和 BLAS 矩阵乘法仍可能成功，其他算子则会报 `hipErrorInvalidKernelFile`。

安装后的 `find_package(jyppx CONFIG REQUIRED COMPONENTS sdk)` 使用同一 ROCm LibTorch 版本；混用 CPU/NVIDIA 包会被拒绝。安装 SDK 不内置完整 ROCm 运行环境。

视觉内存预检在 ROCm 配置下使用 HIP 查询当前设备的空闲显存。它是采样值，不是显存硬配额。模型架构、官方权重、精度、吞吐和显存收益须分别验证。

## ONNX 导出与推理

本组合可启用 `JYPPX_LIUDING_BUILD_ONNX_EXPORT=ON`；需要原生推理时再启用 `JYPPX_LIUDING_BUILD_ONNX_RUNTIME=ON` 并通过 `JYPPX_LIUDING_ONNX_RUNTIME_ROOT` 指定官方 ORT 1.23.2 SDK，详见 [ONNX 导出与推理](../api/onnx.md)。导出配置还接受 Linux 的 ROCm 10.1.0 LibTorch 2.14.0，实际验证范围分别登记；其他 2.14.0 包仍会拒绝。

GPU 训练后的模型通过独立 CPU 副本导出，原训练模型继续保留设备、精度和梯度。视觉 CLI 使用 CPU FP32 eval 副本导出 opset 17；FP16/BF16 训练不代表 ONNX 采用同一精度。ONNX Runtime 的 Provider 独立于 LibTorch，默认 ORT 组合使用 CPU。

Windows 可显式选择独立的 ORT 1.24.4 / DirectML 1.15.4 组合；W7900 已验证 Linear、ResNet18 的 FP32 动态 batch 数值、实际 GPU 节点、并发调用、结果所有权及安装消费。本机完整 CIFAR-10 配方产生的 ResNet18 检查点，也已验证原训练记录关联的原生新导出、临时签名、可信加载与全部 10000 张验证图的 DirectML 评估，流程见 [已训练模型交付](../../examples/deployment/README.md#已训练的-cifar-10-模型交付)。设备参数采用 DXGI 索引，同一会话串行执行；精确配置和取消契约见 [ONNX 后端选择](../api/onnx.md#runtime-与后端选择)。这不授予其他图、精度、适配器或性能资格。

## 本地推理

可选 `serving` 组件已验证本组合的 native bundle 与显式可信 TorchScript GPU 加载、warmup 和数值前向，以及并发租约、取消、版本切换和安装消费。使用前仍须调用 `initialize_runtime()`。`onnx_serving` 已验证 ORT 1.23.2 CPU Provider，以及显式 DirectML 组合的合成多输入输出图加载、warmup、并发请求和关闭；它不会根据 ROCm LibTorch 自动选择 AMD GPU。组件仍为 Preview，接口与边界见 [Serving](../api/serving.md)。

可选原生 TensorBoard/OTLP、WinHTTP 发送、Ed25519 签名和 headless CPU 渲染已验证该组合的组件契约与安装消费，遥测另经锁定官方 reader 和 Collector 检查。签名的私钥/策略符号链接负例受当前环境限制仍未验证；完整范围见兼容矩阵。

## 本地进程执行

ROCm 构建支持 `local rocm-devices` 与 `rocm_device` 放置计划，以 HIP UUID 和 PCI 地址共同绑定设备身份，启动前重新检查 ordinal、架构和可用显存。Windows 使用 HIP 与 WDDM 全局专用显存统计的保守值，逻辑预约协调同一 STORE 的任务；子进程通过统一可见性变量将所选卡作为 `cuda:0` 使用。完整字段、恢复和采样限制见[本地进程运行器](../development/local_process_runner.md#amd-gpu-实际准入与-rocm-放置)。启动器须从未屏蔽 GPU 的环境启动；不要在 HIP 初始化后修改可见性变量。

## 开发资源测量

ROCm 构建可选择独立训练与八类优化器 benchmark，记录实际 HIP 设备身份、同步计时和 allocator 显存峰值，并由开发脚本运行独立预热及重复采样；命令、计时范围和环境绑定见 [benchmark 指南](../benchmarks/README.md#rocm-训练与优化器)。另有独立的[稳定性与受控压力入口](../benchmarks/README.md#rocm-稳定性与受控显存压力)，验证固定 Linear/AdamW/StepLR 的持续训练、文件恢复、显存 payload 释放和进程 allocator 配额 OOM 恢复；记录实际时长与精度，不扩大为其他负载或数小时稳定性资格。

模型与 Serving 的[通用资源入口](../../scripts/benchmarks/RESOURCE_BASELINE.md)使用 HIP 身份、同步与独立 `peak_rocm_*` 字段；缺少功耗/频率控制观测时保留独立样本并阻止写入可比较 reference。图像入口与真实任务性能须分别验证。

[图像读取到训练更新](../benchmarks/README.md#图像读取到训练更新)入口同样支持 HIP 身份、同步与独立显存指标，可选择文件读取、解码缓存或预缩放缓存；`cuda-fp16` 与 `cuda-fp16-storage` 分别使用 FP32 主权重和 FP16 参数存储。每种模型、数据、精度与加载配置须独立采样，训练完成不代表统计稳定或任务准确率。

## WSL2 原生核心与训练

W7900 另已验证 WSL2、Ubuntu 24.04.5、GCC 13.3、同版 ROCm/LibTorch 的原生 Linux C++11 ABI=1 组合。范围包括设备与数据管线、八类优化器与十五类调度器的 FP32/FP16/BF16 组合、AMP、Trainer、检查点，以及独立安装应用的 GPU 前向、反向和 HIP 物理设备身份检查。精确 LibTorch 2.5.1 历史检查点迁移用例仍条件跳过；原生版本查询应报告 `rocm:available:1`。

该组合的本地进程运行器另通过真实 AMD 子进程放置、显存准入与重试前复检、冻结搜索及完成态重放；GPU 显存预约仍是逻辑协调。CPU/RSS 计量与进程树清理已验证，操作系统内存硬限制须提供可写的 cgroup v2 委托目录，详见[本地进程运行器](../development/local_process_runner.md)；缺少该目录时不能启用此边界。

固定图像分类配方的 CPU 完整训练、精确重试与 image/paired 工作目录丢失恢复已通过；paired 任务另通过 CPU/GPU 真实训练和搜索在 2 GiB cgroup v2 硬内存限制下的恢复、进程清理和资源释放。CPU 图像任务还验证了取消、超时、搜索和 128 MiB 内核 OOM 后的失败持久化与租约释放。这些主存限制不构成显存硬配额。

模型接口另已验证 CPU/GPU 特征提取、分类头替换与冻结训练，以及 ResNet 前向、梯度、更新和 buffer 对照；原生模型 bundle 与显式可信 TorchScript 均通过实际 GPU 加载、预热、推理和关闭。模型、Serving 与图像组件的独立安装消费者也已通过，图像消费者执行真实 GPU 图像和 mask 拷贝。这些结果限于兼容清单列出的原始用例，不授予预训练权重、任务质量或完整视觉目录资格。

图像组件另通过标注、编解码、联合变换、组批、预算回收及 GPU 拷贝流契约；原生 CLI 的模型检查点、离线流程、本地任务生命周期、初始状态、update、AdamW 和 paired 套件也已通过。三个仅用于后端不可用的回收契约，在独立隐藏 GPU 的进程补验通过。默认未启用 BatchNorm 校准的 ResNet18 小样本配方仍未通过原质量门禁：固定 48 次更新后准确率为 0.5，低于 0.875；对应图像配方和本地重试套件保留失败，不包含在通过范围内。

视觉训练另已验证 14 个代表任务的 FP32、FP16、BF16 共 42 个 GPU 组合及 14 个 CPU 对照，覆盖分类、检测、分割、姿态、旋转框、深度、SimMIM 与 ReID 入口；视觉契约和独立安装消费者也已通过。范围限于原随机初始化模型、合成小输入的单步前向、loss、反向与实际更新，不推导完整架构目录、预训练权重、收敛、恢复或真实数据质量。

可选原生 TensorBoard/OTLP 离线映射、Ed25519 签名和 CPU headless 渲染已通过组件契约与三个独立安装消费者；Linux 私钥权限、链接和非普通文件拒绝用例也已执行。签名使用精确 OpenSSL 3.5.9 静态 SDK，保持私有 Crypto 符号隔离。

原生 TensorBoard 示例、CLI 和 worker 另通过独立官方 reader/writer 检查，OTLP 示例通过官方 protobuf 解码与固定映射对照；默认空环境 worker 与安装版 worker 的原契约也已验证。该范围尚不含真实 Collector、HTTP/TLS 发送或 GPU 渲染。

原生 ONNX 导出另已验证 FP32、FP16、BF16 GPU 训练状态经独立 CPU FP32 副本导出，以及 CPU ONNX Runtime 1.23.2、ONNX Serving 和独立安装消费者。固定参考图通过官方 checker、严格 shape inference 及原数值对照；这些结果仅覆盖兼容清单中的图与契约，不提供 AMD ORT GPU Execution Provider，也不继承 Windows DirectML 的验证范围。构建和使用见 [ONNX API](../api/onnx.md)。

原生合成稳定性负载已通过 FP32、FP16、BF16 各独立 180 秒、2 GiB GPU payload 的持续训练；每精度完成 11 次文件恢复和 33 步精确续训对照，初末活跃分配相同。进程 allocator 配额 OOM、非法 shape 和 FP16 注入溢出后的恢复也已通过；此范围不包含两小时 WSL 长测、物理显存耗尽或驱动重置。

资源入口的两种训练与八种优化器模式各完成一独立预热和五独立采样。训练模式通过原 10% CV 门槛，最大 CV 为 3.236%；优化器部分耗时指标超标，最大 CV 为 44.174%，保留原失败。缺少功耗和频率控制观测，不提供可比较性能 reference。

通用资源入口另完成三种 CPU 存储负载、五种 GPU 模型与 GPU Serving 的独立采样，RSS/HIP allocator 指标完整。原生 ext4 上的 storage、checkpoint、telemetry 通过原 10% CV 门槛；六种 GPU profile 的部分耗时指标未通过，最大 CV 为 36.695%，不授予完整统计稳定性。

该 Linux 分发的 profiler 会在进入 `main` 前枚举设备并解析 GPU 可见性变量，可能让只计算环境摘要的命令因占位值直接终止。无需 profiling 的启动器可在启动进程前设置 `ROCPROFILER_REGISTER_ENABLED=0`，关闭 profiler 注册；GPU 计算仍使用原 HIP 后端。框架不会修改调用者的环境；需要 profiling 时保留 SDK 默认设置，并确保可见性变量合法。`initialize_runtime()` 无法修复动态库加载前发生的终止。

在 WSL 原生文件系统中维护独立检出、构建和训练工作目录；模型包发布要求文件系统支持 Linux 原子不覆盖目录重命名，Windows 挂载目录可能拒绝该操作。激活原生 Linux SDK，让 `LIBTORCH_ROOT`、`ROCM_PATH` 分别指向 LibTorch 与 HIP 前缀。按[贡献指南](../../CONTRIBUTING.md#构建与验证)预置锁定依赖；核心 CLI 与安装清单中的原生转换工具可这样构建：

```sh
cmake -S . -B out/build/wsl-rocm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ \
  -DTorch_DIR="$LIBTORCH_ROOT/share/cmake/Torch" \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_ROOT;$ROCM_PATH" \
  -DJYPPX_LIUDING_BUILD_EXAMPLES=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
cmake --build out/build/wsl-rocm --parallel 2 \
  --target jyppx_cli jyppx_convert_dinov2_hf
./out/build/wsl-rocm/tools/cli/liuding-forge --version --format json
```

框架构建的 Linux ROCm 可执行文件使用 ELF RPATH 查找配置时选择的 LibTorch/HIP SDK 及其数学库，安装后也保留这些 SDK 路径，使默认空环境的原生遥测 worker 能正常启动。外部 SDK 必须保持在配置时的位置；迁移 SDK 后重新配置、构建和安装。该设置不复制 SDK、不修改系统搜索路径或 worker 的环境继承策略；应用自行构建可执行文件时仍须配置自己的运行库搜索路径。

该范围限于上述原生契约、任务、安装接口和合成资源负载；完整视觉配方、真实数据质量、其他资源负载与更长稳定性须在 WSL 中独立验证。Windows 的结果不扩大这些边界。
