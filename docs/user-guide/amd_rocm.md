# AMD GPU 与 ROCm

AMD 后端使用 ROCm 构建的 LibTorch 和匹配 HIP SDK。精确版本、编译器和验证范围见 [compatibility.json](../../compatibility.json) 的 `amd_rocm`；它与 NVIDIA CUDA、CPU 包独立配置。已验证的开发基线为 Windows x64、Clang 24 clang-cl、LibTorch `2.14.0+rocm10.1.0`、ROCm SDK `10.1.0` 与 W7900 gfx1100，覆盖 FP32/FP16/BF16 训练、精确恢复、视觉流程及安装后 GPU 消费。WSL/Linux 和其他 GPU 需独立验证。

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

## 运行与安装

Windows 构建树会暂存所选 LibTorch 的 DLL、`.kpack` 和 AOTriton 图像；应用仍需提供匹配的 HIP 运行库、数学库和设备数据。保留 HIP SDK 自身的 `.kpack`、数学库数据目录，按 SDK 激活脚本设置 `HIPBLASLT_TENSILE_LIBPATH`、`ROCBLAS_TENSILE_LIBPATH` 及运行库搜索路径。只复制 `torch_hip.dll` 不能构成完整部署。

自行部署时，`torch_hip.dll` 上一级目录保留 `.kpack/torch_gfx1100.kpack`，`aotriton_v2.dll` 相邻目录保留 `aotriton.images`。不设置仅指向 Torch 归档的全局 `ROCM_KPACK_PATH_PREFIX`，它还会影响 HIP 数学库的归档选择。缺少设备数据时，设备发现和 BLAS 矩阵乘法仍可能成功，其他算子则会报 `hipErrorInvalidKernelFile`。

安装后的 `find_package(jyppx CONFIG REQUIRED COMPONENTS sdk)` 使用同一 ROCm LibTorch 版本；混用 CPU/NVIDIA 包会被拒绝。安装 SDK 不内置完整 ROCm 运行环境。

视觉内存预检在 ROCm 配置下使用 HIP 查询当前设备的空闲显存。它是采样值，不是显存硬配额。模型架构、官方权重、精度、吞吐和显存收益须分别验证。

## ONNX 导出

本组合可启用 `JYPPX_LIUDING_BUILD_ONNX_EXPORT=ON`；需要原生推理时再启用 `JYPPX_LIUDING_BUILD_ONNX_RUNTIME=ON` 并通过 `JYPPX_LIUDING_ONNX_RUNTIME_ROOT` 指定官方 ORT 1.23.2 SDK，详见 [ONNX 导出与推理](../api/onnx.md)。此适配仅增加 Windows ROCm 10.1.0 LibTorch 2.14.0 精确组合，不扩展到其他 2.14.0 包。

GPU 训练后的模型通过独立 CPU 副本导出，原训练模型继续保留设备、精度和梯度。视觉 CLI 使用 CPU FP32 eval 副本导出 opset 17；FP16/BF16 训练不代表 ONNX 采用同一精度。ONNX Runtime 的 Provider 独立于 LibTorch，本组合使用 CPU Provider 验证导出图，不据此声明 ORT 的 AMD GPU 执行。

## 本地进程执行

ROCm 构建支持 `local rocm-devices` 与 `rocm_device` 放置计划，以 HIP UUID 和 PCI 地址共同绑定设备身份，启动前重新检查 ordinal、架构和可用显存。Windows 使用 HIP 与 WDDM 全局专用显存统计的保守值，逻辑预约协调同一 STORE 的任务；子进程通过统一可见性变量将所选卡作为 `cuda:0` 使用。完整字段、恢复和采样限制见[本地进程运行器](../development/local_process_runner.md#amd-gpu-实际准入与-rocm-放置)。启动器须从未屏蔽 GPU 的环境启动；不要在 HIP 初始化后修改可见性变量。
