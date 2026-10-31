# Linear 量化

`jyppx::liuding::quantization` 提供有界校准、对称 int8 参考运算和版本化 Linear 权重制品；该组件默认关闭，仅依赖 `foundation`。静态 PTQ 可以准备交给独立 ONNX Runtime adapter 执行的 U8U8/int32 图。制品存储、转换成功、实际整数执行和性能收益分别验证。

本项目的 C++ 代码负责校准、量化格式和制品校验；LibTorch 用于训练和 ONNX 序列化，当前静态 INT8 图由 ONNX Runtime 执行。本模块尚未接入 LibTorch 原生量化执行。

| 路径 | 运算与边界 |
| --- | --- |
| Q0 便携制品 | `jyppx.qint8.symmetric.per_tensor/v1`；有符号 int8 权重、每张量单一 scale、零点 0、范围 `[-127,127]`、可选 FP32 bias及完整校准身份 |
| `StorageQuantizedLinearRuntime` | 构造时反量化权重，随后执行 FP32 Linear；`compute_quantized=false`、`qat=false`，报告一个浮点 fallback 模块 |
| `prepare_static_int8_linear` | 将 Q0 的 `[out,in]` 权重无损转为 `[in,out]` 并加 128；生成独立的 `jyppx.linear.ptq.ort.u8u8/v1` 准备信息，不改变 Q0 格式或宣称已执行内核 |
| ORT 静态 PTQ 图 | 固定 ORT 1.23.2、ONNX opset 17，IR 版本由官方序列化器决定；FP32 输入经 `QuantizeLinear` 转 U8，U8U8 `MatMulInteger` 累加为 int32，随后 FP32 缩放及可选 bias；CPU 资格须由真实执行 trace 证明 |
| 按输出通道权重 | `quantize_linear_per_output_channel` 对 `[out,in]` 的每行独立取对称权重 scale；`export_static_int8_linear_per_output_channel` 导出一个 U8U8 `MatMulInteger` 加 `[out]` scale 向量，不执行浮点点积 |
| 显式 ORT 融合图 | `StaticInt8LinearGraph::ort_matmul_integer_to_float` 使用 `com.microsoft::MatMulIntegerToFloat` 合并整数点积、转换和缩放，保留独立 bias Add；需要支持该自定义算子的执行后端，单独验证数值、内核和性能 |

当前路径不提供 QAT、INT4、GPU 量化或任意整图自动转换。U8S8 的 AVX2 路径可能出现中间饱和，本受限配方使用 U8U8，CPU 后端自己完成机器相关的 packing；便携制品不保存机器布局。

## 校准与格式

参考量化使用 `scale=max(abs(x))/127`；全零有限范围使用 scale 1，四舍六入、正中取偶，然后截到 `[-127,127]`。参考舍入不受调用方浮点舍入模式影响。空范围或全非有限范围不能生成 qparams；量化输入含 NaN/Inf 时拒绝。

`CalibrationSession` 保存数据集版本及 SHA-256、预处理 fingerprint、seed 和有序唯一 sample IDs。`MinMaxObserver` 统计全部调用和值，求极值时跳过 NaN/Inf；其 exact checkpoint 仅包含数值状态，恢复整个校准会话时还须保留同一批样本身份与 lineage。

Q0 二进制以 `JYPPXQ1\0` 开头，使用规范小端编码并拒绝未知版本、非法元数据、截断、非规范或尾随字节。默认限制为制品 16 MiB、每维 65,536、校准 sample IDs 1,000,000 个、每个 ID 256 字节、每个 lineage 字段 1,024 字节；维度与元数据同时受完整制品字节上限约束。

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
