# Linear 量化

`jyppx::liuding::quantization` 提供有界校准、对称 int8 参考运算和版本化 Linear 权重制品；该组件默认关闭，仅依赖 `foundation`。静态 PTQ 可以准备交给独立 ONNX Runtime adapter 执行的 U8U8/int32 图。制品存储、转换成功、实际整数执行和性能收益分别验证。

本项目的 C++ 代码负责校准、量化格式和制品校验。可选 `quantization_torch` 通过 LibTorch 官方量化算子执行受限的 CPU static INT8 Linear；ONNX 导出与 ONNX Runtime adapter 保留独立部署路径。两条执行路径的输出数值契约和平台资格分别验证。

| 路径 | 运算与边界 |
| --- | --- |
| Q0 便携制品 | `jyppx.qint8.symmetric.per_tensor/v1`；有符号 int8 权重、每张量单一 scale、零点 0、范围 `[-127,127]`、可选 FP32 bias及完整校准身份 |
| `quantization_torch::StaticLinear` | 原 Q0/QC1 权重经官方 `quantized::linear_prepack` 重建本机 packed state，由 `quantized::linear` 执行；输入和输出为静态 QUInt8，权重为 QInt8，每张量或按输出行 scale，输出单独校准 |
| `StorageQuantizedLinearRuntime` | 构造时反量化权重，随后执行 FP32 Linear；`compute_quantized=false`、`qat=false`，报告一个浮点 fallback 模块 |
| `prepare_static_int8_linear` | 将 Q0 的 `[out,in]` 权重无损转为 `[in,out]` 并加 128；生成独立的 `jyppx.linear.ptq.ort.u8u8/v1` 准备信息，不改变 Q0 格式或宣称已执行内核 |
| ORT 静态 PTQ 图 | 固定 ORT 1.23.2、ONNX opset 17，IR 版本由官方序列化器决定；FP32 输入经 `QuantizeLinear` 转 U8，U8U8 `MatMulInteger` 累加为 int32，随后 FP32 缩放及可选 bias；CPU 资格须由真实执行 trace 证明 |
| 按输出通道权重 | `quantize_linear_per_output_channel` 对 `[out,in]` 的每行独立取对称权重 scale；`export_static_int8_linear_per_output_channel` 导出一个 U8U8 `MatMulInteger` 加 `[out]` scale 向量，不执行浮点点积 |
| 显式 ORT 融合图 | `StaticInt8LinearGraph::ort_matmul_integer_to_float` 使用 `com.microsoft::MatMulIntegerToFloat` 合并整数点积、转换和缩放，保留独立 bias Add；需要支持该自定义算子的执行后端，单独验证数值、内核和性能 |

当前路径不提供 QAT、INT4、GPU 量化或任意整图自动转换。U8S8 的 AVX2 路径可能出现中间饱和，ORT 受限配方使用 U8U8，CPU 后端自己完成机器相关的 packing；便携制品不保存机器布局。

## 校准与格式

参考量化使用 `scale=max(abs(x))/127`；全零有限范围使用 scale 1，四舍六入、正中取偶，然后截到 `[-127,127]`。参考舍入不受调用方浮点舍入模式影响。空范围或全非有限范围不能生成 qparams；量化输入含 NaN/Inf 时拒绝。

`CalibrationSession` 保存数据集版本及 SHA-256、预处理 fingerprint、seed 和有序唯一 sample IDs。`MinMaxObserver` 统计全部调用和值，求极值时跳过 NaN/Inf；其 exact checkpoint 仅包含数值状态，恢复整个校准会话时还须保留同一批样本身份与 lineage。完整已完成的结果可用 `serialize_calibration_result` / `deserialize_calibration_result` 独立保存读取；`JYPPXCA1` v1 复用同一 canonical 校准编码，包含 lineage、有序 sample IDs、observer state 和 qparams，拒绝空会话、未知版本、畸形或尾随字节，并受原有元数据与完整字节上限约束。

Q0 二进制以 `JYPPXQ1\0` 开头，使用规范小端编码并拒绝未知版本、非法元数据、截断、非规范或尾随字节。默认限制为制品 16 MiB、每维 65,536、校准 sample IDs 1,000,000 个、每个 ID 256 字节、每个 lineage 字段 1,024 字节；维度与元数据同时受完整制品字节上限约束。

## LibTorch 官方静态 INT8

启用 `JYPPX_LIUDING_BUILD_QUANTIZATION_TORCH=ON`，同时启用 `CORE` 和 `QUANTIZATION`。安装后使用 `find_package(jyppx REQUIRED COMPONENTS quantization_torch)` 并链接 `jyppx::liuding::quantization_torch`，公共头为 `<jyppx/liuding/quantization_torch/static_linear.hpp>`。它使用现有 `Tensor` 类型，不需要 ONNX Runtime 或 Python。

`StaticLinear::load_per_tensor` / `load_per_output_channel` 同时接收权重制品字节与独立期望 SHA-256、输出校准字节与独立期望 SHA-256，以及资源选项。Q0/QC1 只含输入校准，因此不能单凭权重制品恢复此 runtime。输出须由另一 `CalibrationSession` 观测原 FP32 模型在同一训练校准子集上的输出，lineage 和 sample IDs 顺序必须与输入完全相同；不得根据测试标签或任务结果选择 output scale。`info()` 返回两份完整摘要、实际 packing engine、编译 LibTorch 版本和冻结 qparams。来源身份仍需应用审计，元数据一致不证明样本使用正确。

输入是连续、有限的 CPU FP32 `[batch,in_features]`；`quantize_input` 调用官方输入量化，`forward_quantized` 调用官方整数 Linear 并返回实际 QUInt8 `[batch,out_features]`，`forward` 仅在最终结果反量化。权重保持原有 int8 行序，zero point 0；按输出通道使用官方 axis 0。输入/输出 zero point 均为 128，输出使用独立冻结 scale，增加输出 requantization 舍入与饱和，不能直接套用 ORT 浮点输出参考。

加载时默认捕获并冻结当前 engine，只接受本轮对应的 `ONEDNN` 或 `X86`，同时检查 SDK 支持列表和官方内核注册；`StaticLinearOptions::expected_engine` 可选地约束当前值，其不一致时拒绝，其他 engine 明确返回不支持。adapter 不设置或临时切换全局 engine。应用须在并发前固定 engine，同一 runtime 串行保护 packed cache；外部并发修改全局 engine 不在其锁的保护范围内。packed state 仅驻留内存，加载时重建。非 nearest-even 模式、错配摘要/校准、shape/dtype/预算和非有限数均明确失败；输入预缩放绝对值和保守输出 requantization 界限不超过 `2^30`，不满足安全域时拒绝。量化计算、任务精度、速度和 RSS 分别验收，编译或小夹具结果不代表其他平台、engine 或真实任务资格。当前接口对应 LibTorch 2.14.1 的官方量化算子，量化 Tensor 创建 API 已有弃用提示，升级 SDK 须重新验证该组件。完整保存/重新读取示例见 [examples/quantization](../../examples/quantization/README.md)。

## 准备 CPU 静态 PTQ

先用 `quantize_linear` 和 `serialize_linear_artifact` 保存 Q0；完整流程见[静态 INT8 Linear 教程](../tutorials/static_int8_linear_ptq.md)。已安装的 `liuding-forge-int8-prepare` 会重新进行原生 Q0 校验并向标准输出写出有界 JSON。ONNX 图必须由 C++ `jyppx::liuding::onnx_export::export_static_int8_linear` 生成：

```sh
liuding-forge-int8-prepare linear.qint8 > linear.preparation.json
liuding-forge-int8-export linear.qint8 linear.int8.onnx
```

原生 API 使用 LibTorch 官方 C++ ONNX 序列化器，先核对调用方提供的 Q0 SHA-256，再校验规范格式和完整校准 lineage。CLI 将本次读取的文件作为源，跨文件流转时的来源身份须由调用方另行核对。输出文件完整发布且拒绝覆盖；导出成功只代表图制品生成，实际执行可使用单独的 `jyppx::liuding::onnx_runtime` adapter 显式选择 CPU。生产流程无需 Python。

输出行的权重幅度差异较大时，可用 `PerOutputChannelLinearArtifact` 和 `serialize_per_output_channel_linear_artifact` 保存新聚合格式 `JYPPXQC1`。它按输出顺序封装每行一个 canonical Q0/v1，校验完整通道数、显式索引、输入宽度、统一 bias 存在策略和逐字节一致的校准 lineage/sample IDs/observer/qparams。`max_features` 限制输入宽度与通道数，`max_artifact_bytes` 限制包含重复校准元数据的完整聚合。完整聚合 SHA-256 绑定顺序、bias 和全部通道；重新编码的合法不同模型仍须通过独立期望摘要识别。旧 Q0/v1 语义不变。激活仍使用共同的对称 scale 和 zero point 128，新 profile 为 `jyppx.linear.ptq.ort.u8u8.per_output_channel/v1`。

```sh
liuding-forge-int8-export --per-output-channel model.per-output-channel.qint8 model.int8.onnx
```

默认图选择是 `--graph portable`，只使用标准 ONNX 算子。要显式导出 ORT 融合图，可使用：

```sh
liuding-forge-int8-export --per-output-channel --graph ort-matmul-integer-to-float model.per-output-channel.qint8 model.ort-int8.onnx
```

该选项也支持每张量权重。融合图依赖 ORT 的 `com.microsoft` opset 1，返回的独立 profile 在原 `/v1` 前增加 `.matmul_integer_to_float`。运行时和执行后端必须支持该算子，数值、内核及性能须独立验证。C++ 两个导出函数的最后一个参数对应此图选择，默认值为 `StaticInt8LinearGraph::portable`。

完整 C++ 学习程序见 [examples/quantization](../../examples/quantization/README.md)。新路径仍需独立的真实 kernel trace、任务精度和性能资格，不能由通道数值单测推导通过。

静态配方将激活按校准范围取偶舍入并饱和到 `[0,255]`，不在运行时重新校准；超出校准范围的输入可能损失任务质量。每维固定不超过 65,536，int32 点积最坏绝对值不超过 1,065,353,216；scale 及其乘积须是正常、有限的 FP32 数，保守饱和输入输出界须能用有限 FP32 表示，否则准备失败。这个逻辑界限不是进程 RSS 上限。

## 证据与错误

`compare_reference` 报告 RMSE、MAE、最大绝对/相对误差、SQNR、cosine similarity 和非有限对数。相对误差分母为 `max(abs(reference),1e-12)`；JSON 将无穷 SQNR 表示为 `null`。存储误差不能代替真实任务准确率。

CPU 计算量化验收须核对同一 ONNX SHA-256 的实际 CPU `MatMulInteger` 或其整数融合算子 trace、独立整数累加参考及任务门槛。冷加载、首次推理、无 profiling 的热时延、进程工作集和制品体积分别测量；整数执行本身不意味着加速。缺少 ORT、版本不符、shape 不支持或数值无效均明确失败，不隐式切换浮点 Linear。现有显式存储 fallback 的语义保持独立。
