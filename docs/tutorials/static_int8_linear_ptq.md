# 在 C++ 中校准、导出并执行静态 INT8 Linear

本教程将一个已知权重的 Linear 转为 Q0 便携制品，直接调用 C++ 原生 ONNX 导出 API，再交给独立 ONNX Runtime adapter 显式使用 CPU。整个生产流程不调用 Python。示例用于解释接口和数值契约；实际任务还须独立验证校准覆盖、准确率、资源和性能。

## 准备构建

准备与项目支持矩阵匹配的 LibTorch、编译器，以及官方 ONNX Runtime 1.23.2 SDK。ONNX 图序列化使用 LibTorch 自带的官方 C++ exporter/schema；运行时使用单独的 ORT SDK。配置阶段不下载依赖。复用你的兼容构建目录，显式启用以下组件：

```sh
cmake -S . -B out/build/my-profile -DJYPPX_LIUDING_BUILD_QUANTIZATION=ON -DJYPPX_LIUDING_BUILD_ONNX_EXPORT=ON -DJYPPX_LIUDING_BUILD_ONNX_RUNTIME=ON -DJYPPX_LIUDING_ONNX_RUNTIME_ROOT=/path/to/onnxruntime-1.23.2
cmake --build out/build/my-profile --config Release --target jyppx_int8_prepare jyppx_int8_export --parallel 2
```

SDK、LibTorch、编译器和 CRT 必须匹配实际平台；不能从一个平台的结果推导另一个平台支持。安装后的 C++ 消费者只链接所需组件：

```cmake
cmake_minimum_required(VERSION 3.18)
project(int8_linear_example LANGUAGES CXX)
find_package(jyppx REQUIRED COMPONENTS quantization onnx_export onnx_runtime)
add_executable(int8_linear_example main.cpp)
target_compile_features(int8_linear_example PRIVATE cxx_std_20)
target_link_libraries(int8_linear_example PRIVATE
    jyppx::liuding::quantization
    jyppx::liuding::onnx_export
    jyppx::liuding::onnx_runtime)
```

## 校准、导出与推理

保存下面完整程序为 `main.cpp`。真实应用应使用训练数据的独立校准子集，保留原始 sample IDs 和预处理身份；不得用测试集选择 qparams。这里使用精确整数 scale，便于手算核对。

```cpp
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/deployment/onnx_export/static_int8_linear.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/quantization/quantization.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <string>

int main() {
  namespace q = jyppx::liuding::quantization;
  namespace ex = jyppx::liuding::onnx_export;
  namespace rt = jyppx::liuding::onnx_runtime;

  q::CalibrationSession calibration({
      .dataset_version = "example-calibration/v1",
      .dataset_digest = {.hex = std::string(64, 'a')},
      .preprocessing_fingerprint = "identity-fp32/v1",
      .seed = 20261029U});
  const std::array<float, 2> calibration_values{-127.0F, 127.0F};
  calibration.observe("train/sample-0", calibration_values);

  const std::array<float, 6> weights{
      127.0F, -127.0F, 1.0F,
      -1.0F, 2.0F, -3.0F};
  const std::array<float, 2> bias{0.25F, -0.5F};
  const auto source = q::quantize_linear(
      {.in_features = 3, .out_features = 2,
       .weights = weights, .bias = bias},
      calibration.finalize());
  const auto source_bytes = q::serialize_linear_artifact(source);
  const jyppx::liuding::ArtifactDigest source_digest{
      .hex = jyppx::liuding::sha256_hex(source_bytes)};

  const std::array<std::int64_t, 2> shape{1, 3};
  ex::ExportOptions export_options;
  export_options.opset = 17;
  export_options.dynamic_batch = true;
  export_options.minimum_batch = 1;
  export_options.maximum_batch = 64;
  const auto exported = ex::export_static_int8_linear(
      source_bytes, source_digest, shape, export_options);

  rt::SessionOptions runtime_options;
  runtime_options.provider.name = "CPUExecutionProvider";
  runtime_options.allow_cpu_fallback = false;
  runtime_options.cpu_threads = 1;
  runtime_options.max_batch_size = 64;
  runtime_options.disable_graph_optimizations = true;
  runtime_options.profile_prefix = "int8-linear-profile";
  rt::Session session(exported.artifact.onnx,
                      exported.artifact.digest, runtime_options);

  const std::array<float, 3> input{-3.5F, 0.5F, 127.0F};
  const auto result = session.run(input, shape);
  std::cout << result.values[0] << ' ' << result.values[1] << '\n';
  std::cout << "ONNX SHA-256: " << exported.artifact.digest.hex << '\n';
  std::cout << "ORT profile: " << session.end_profiling().string() << '\n';
}
```

预期输出前两个数为 `-380.75 -377.5`：静态激活量化把 `-3.5` 取偶舍入为 `-4`，把 `0.5` 取偶舍入为 `0`，再执行整数点积及 FP32 bias。示例 digest 的 64 个 `a` 只是占位数据身份，实际应用必须填入真实校准数据清单 SHA-256；SHA-256 标识内容，不认证生产者。

导出函数首先核对原始 Q0 内容身份，再做规范格式、校准、shape、scale 和资源验证，返回自包含 ONNX 字节及来源身份。默认只有 batch 维可以动态变化，导出范围是声明性约束；运行时也须设置同一上限。Q0 存储 descriptor 继续声明 `compute_quantized=false`，不会因导出成功被改写。

## 通过命令行导出已有制品

已有 `linear.qint8` 时，可以直接调用原生工具：

```sh
liuding-forge-int8-prepare linear.qint8 > linear.preparation.json
liuding-forge-int8-export linear.qint8 linear.int8.onnx
```

准备工具输出诊断 JSON，实际导出工具直接读取并校验原始 Q0。导出工具使用动态 batch，拒绝覆盖已有输出；输出目录由调用方持有并保持稳定。生产流程不需要 Python 环境。

## 检查执行与任务质量

查看上面返回的 ORT profile，确认同一 ONNX SHA-256 对应的实际事件包含 CPU `MatMulInteger`，或经过单独核验的整数融合算子，并核对输入类型与整数累加。仅在模型里看见 INT8 字节或节点名称不足以证明执行。这个 CPU 配方不因为 runtime 能选择其他 provider 就自动获得 GPU INT8 资格。

校准范围之外的输入会饱和，权重采用每张量单一 scale；少量离群权重也可能影响任务准确率。分别记录浮点模型与 PTQ 模型在同一未参与校准的数据集上的指标，事前固定可接受门槛。失败后应保留结果并重新论证训练或量化方法，不把误差容忍度扩大到通过。

性能测量使用关闭 profiling 的独立进程，分别报告加载、首次推理、预热后的多次时延、吞吐、峰值工作集及制品体积，并保留相同线程、shape、输入、CPU 与库版本的浮点对照。小型 Linear 的量化和调度成本可能超过整数运算收益；实际整数执行不保证提速。

## 常见失败

| 情况 | 处理 |
| --- | --- |
| Q0 SHA-256、格式版本或校准身份不符 | 重新取得匹配的原始制品与来源身份，不忽略校验 |
| scale 太小、太大，或保守输出界溢出 | 重新论证输入/模型数值范围；此配方不隐式改用 FP32 |
| 维度、模型字节或 I/O 超过限制 | 明确调整有界资源配置，并重新执行资格测试 |
| ORT 版本、provider 或 operator 不可用 | 使用已验证的精确组合；不要把 provider 可枚举等同于模型可运行 |
| 真实任务精度下降超限 | 保留失败证据，审查校准覆盖和权重分布；不得用 synthetic 数值正确替代任务质量 |

Q0 详细格式和错误契约见[量化 API](../api/quantization.md)。
