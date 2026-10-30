# 按输出通道量化 Linear

`per_output_channel.cpp` 是完整 C++ 教学程序：校准激活、逐输出行量化权重、保存并重新解析 canonical 制品、验证完整 SHA-256、导出 ONNX，再用 ORT CPU 执行，与独立整数 golden 值比较。它只需要库和 ORT SDK，不依赖 Python。固定小夹具便于逐行学习，不能代表真实任务精度或性能。

构建时启用 `JYPPX_LIUDING_BUILD_QUANTIZATION`、`JYPPX_LIUDING_BUILD_ONNX_EXPORT`、`JYPPX_LIUDING_BUILD_ONNX_RUNTIME` 和 `JYPPX_LIUDING_BUILD_EXAMPLES`，并使用兼容的 LibTorch / ORT 1.23.2 配置：

```powershell
cmake --build out/build/config-windows --config Release --target jyppx_liuding_quantization_per_output_channel_example --parallel 2
out/build/config-windows/examples/Release/liuding-forge-example-int8-per-output-channel.exe
```

权重按 `[out_features,in_features]` 行序组织。每一行使用 `max(abs(row))/127` 的对称 int8 权重 scale；激活继续使用原先共同校准的对称 scale、U8 zero point 128。导出图只有一个 U8U8 `MatMulInteger`，int32 结果转换为 FP32 后乘 `[out_features]` scale 向量，再加可选 FP32 bias。缩放及 bias 为 FP32 不代表点积退回 FP32。

新聚合制品 `JYPPXQC1` 封装现有 Q0/v1 行制品，记录版本、输入/输出维度、通道数、逐行索引及完整长度。通道必须按输出顺序，宽度相同且各有一个输出；所有行的校准 lineage、sample IDs、observer state、activation qparams 必须完全相同，bias 必须全部存在或全部省略。`max_features` 同时限制输入宽度和通道数，`max_artifact_bytes` 限制完整聚合，包括重复校准元数据。旧 Q0/v1 字节与语义保持原样。

序列化后的完整 SHA-256 绑定行序、bias、权重与校准。解析器可以拒绝编码索引错序、丢失通道等结构错误；若有人重新编码一个合法的不同模型，只能靠独立提供的期望摘要检测。不要仅用文件名或现场计算的摘要作信任判断。

启用 tools 后，命令行显式选择新聚合格式：

```powershell
liuding-forge-int8-export --per-output-channel model.per-output-channel.qint8 model.onnx
```

旧命令仍只接受旧 Q0/v1。导出成功不能证明任务质量或速度；真实部署应保存原始 ONNX、摘要、CPU profiling trace、冻结数据与 FP32/INT8 任务指标，并检查没有浮点 `MatMul`/`Gemm` 执行。该路径不提供 QAT、全图自动转换、GPU 量化或加速承诺。
