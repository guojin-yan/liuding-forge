# ONNX 导出与推理

部署功能集中在 `deployment/`：`onnx_export` 将原生 C++ 模型导出为 ONNX，`onnx_runtime` 加载 ONNX 并执行推理。两个组件分别启用、默认关闭，安装后的导出与推理均无需 Python。

| 组件 | 用途 | 公共头文件 | CMake target |
| --- | --- | --- | --- |
| ONNX 导出 | 执行样例 forward，捕获计算图，按算子转换并序列化 | `<jyppx/liuding/deployment/onnx_export/exporter.hpp>` | `jyppx::liuding::onnx_export` |
| ONNX Runtime | 校验模型字节与输入，使用显式 Execution Provider 推理 | `<jyppx/liuding/deployment/onnx_runtime/session.hpp>` | `jyppx::liuding::onnx_runtime` |

## 构建与接入

准备 LibTorch 2.14.1；需要推理时再准备官方 ONNX Runtime 1.23.2 SDK。在现有 CMake 配置中启用需要的组件，两者互不强制依赖，也不要求启用项目的 `models` 组件。

```sh
cmake -S . -B out/build/your-profile \
  -DJYPPX_LIUDING_BUILD_ONNX_EXPORT=ON \
  -DJYPPX_LIUDING_BUILD_ONNX_RUNTIME=ON \
  -DJYPPX_LIUDING_ONNX_RUNTIME_ROOT=/path/to/onnxruntime-sdk
```

安装后的消费者显式请求组件：

```cmake
find_package(jyppx CONFIG REQUIRED COMPONENTS onnx_export onnx_runtime)
target_link_libraries(my_app PRIVATE
    jyppx::liuding::onnx_export jyppx::liuding::onnx_runtime)
```

Windows 应将安装目录 `bin/onnxruntime*.dll` 放在应用程序目录；仅调整 `PATH` 可能仍先加载系统自带的旧版 ORT。Linux 应确保加载安装包对应的共享库。

## 从 C++ 模型导出并执行

```cpp
#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <torch/nn/modules/linear.h>

#include <array>

torch::nn::Linear model(3, 2);  // 可使用已训练、已加载权重的模型。
model->eval();
const auto example = at::zeros({1, 3}, at::TensorOptions().dtype(at::kFloat));
const auto artifact = jyppx::liuding::onnx_export::export_model(model, example);

jyppx::liuding::onnx_runtime::SessionOptions options;
options.provider.name = "CPUExecutionProvider";
options.max_batch_size = 1024;
jyppx::liuding::onnx_runtime::Session session(
    artifact.onnx, artifact.digest, options);
const std::array<std::int64_t, 2> shape{1, 3};
const std::array<float, 3> input{1.0F, 2.0F, 3.0F};
const auto output = session.run(input, shape);
```

单 Tensor 输入输出的模型使用 `export_model()`；多个输入输出或自定义调用方式使用 `export_module(module, forward, examples, options)`，其中 `forward` 接收 `std::span<const at::Tensor>` 并返回 `std::vector<at::Tensor>`。注册的参数与 buffer 随图保存；输入输出名称由 `input_names`、`output_names` 指定。`export_linear()` 和 `export_resnet18()` 是调用相同通用入口的便利函数，不维护另一套模型整图实现。

导出以算子覆盖为边界，不按模型名称设置白名单。已支持算子的组合可直接复用；缺失或自定义算子通过 `custom_lowerings` 注册原生 C++ 转换，未注册时报告具体算子与原因。导出器使用 LibTorch 官方 ONNX serializer，当前输出为 opset 17、自包含权重，不支持 external data 或直接转换任意 `.pt` 文件。

导出会真实执行 forward，并可能为动态 batch 校验再次执行。调用方负责设定 eval 模式、CPU 设备、数据类型和预处理，保证 forward 确定且不修改输入、模型状态或外部资源。trace 只捕获实际执行路径；动态 batch 会用另一 batch 检查图结构，遇到形状常量或路径变化明确拒绝，不能据此推导任意数据相关控制流均可导出。当前动态轴仅支持 batch 维；其他维度须固定。

`ExportOptions` 设置 batch 上下界、命名和字节上限；`dynamic_batch=false` 导出固定形状。符号 batch 不会在 ONNX 中编码业务上下界，运行端还应设置 `max_batch_size` 并检查自己的最小 batch。输出大小限制不等于进程内存或显存硬配额。启用 `quantization` 后，可导出 canonical Q0 Linear 的 U8U8/int32 图，见[量化](quantization.md)。

## Runtime 与后端选择

当前 Runtime adapter 的用户接口接收单个 FP32 稠密输入并返回单个 FP32 输出，秩为 1–8，只允许共享的前导 batch 符号维；导出器的多输入输出能力另有独立 ORT 验证，不代表该 adapter 已开放多输入输出接口。I/O 位于主机内存，设备传输由所选 Provider 处理。

`available_providers()` 查询实际加载的 ORT 库，`provider_append_supported(name)` 查询 adapter 是否接入对应的追加 API。CPU、CUDA、TensorRT、OpenVINO 及通用 Provider 均显式指定名称和参数；实际可用性取决于 ORT 构建、驱动与 SDK。列出 Provider 或创建 session 成功，均不代表已验证模型实际使用该后端执行。

非 CPU 请求默认 `allow_cpu_fallback=false`，缺失后端或不支持的图明确失败；调用方显式允许后才接受 CPU 分区回退。`profile_prefix` 启用 ORT profiling，`end_profiling()` 返回可核对实际 Provider 与算子的文件。各后端的数值、kernel 与资源资格分别验证，CPU 结果不代表 GPU 结果。

Runtime 在解析前检查完整字节 SHA-256 和大小，执行前检查形状、batch、输入大小与有限数值，输出独立拥有；取消与 deadline 为协作式中断。I/O 上限不覆盖图中间量与设备分配，哈希不代表来源可信；不可信模型须额外隔离。
