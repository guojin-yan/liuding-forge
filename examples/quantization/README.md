# 静态 INT8 Linear 示例

## ONNX Runtime 按输出通道执行

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

## LibTorch 官方执行

`libtorch_static_linear.cpp` 用同一小型 FP32 Linear 和训练校准样本分别观察输入与输出，量化并保存原 Q0、QC1 和完整输出校准，再从文件有界读回，使用预先保留的两份期望摘要加载官方 runtime。它展示每张量与按输出行权重，显式打印实际 engine、版本、制品摘要和量化输出 qparams；最终反量化只用于显示。整个产品流程是 C++，不需要 ORT SDK 或 Python。

在兼容构建目录启用 `JYPPX_LIUDING_BUILD_CORE=ON`、`JYPPX_LIUDING_BUILD_QUANTIZATION=ON`、`JYPPX_LIUDING_BUILD_QUANTIZATION_TORCH=ON` 和 `JYPPX_LIUDING_BUILD_EXAMPLES=ON`，应用应在启动阶段固定全局 engine。加载默认捕获当前 `ONEDNN` 或 `X86`，`expected_engine` 可选地约束该值；其他 engine、支持列表或注册内核不匹配时明确失败，不自动切换。

```powershell
cmake --build out/build/config-windows --config Release --target jyppx_liuding_libtorch_static_linear_example --parallel 2
out/build/config-windows/examples/Release/liuding-forge-example-libtorch-static-int8.exe ./int8-demo-new
```

输出目录必须尚不存在，父目录须已存在；程序创建自己独占使用的目录并拒绝覆盖。再次运行时指定另一个新目录。生成 `linear.qint8`、`linear.per-output-channel.qint8` 和 `output.calibration`，其中最后一个文件保存完整 lineage、顺序 sample IDs、observer state 和 qparams；只保留前两个文件无法重建官方 runtime。读取受 `QuantizationLimits::max_artifact_bytes` 限制，加载前验证每个完整文件的 SHA-256。示例数据是本进程生成并预先取摘要的教学数据，外部文件的期望摘要须来自独立可信来源。

输入 QUInt8、权重 QInt8、输出 QUInt8，按输出通道权重使用 axis 0。输出 scale 来自原 FP32 模型对训练校准子集的输出，输入与输出的 lineage 和 sample IDs 顺序一致；禁止用测试标签或看过任务结果后调整 scale。输出会重新量化及饱和，不能与 ORT 浮点输出逐位等同。

这是接口与持久化教学例。实际整数执行、任务误差、速度与进程内存分别验证；不据此承诺提速或其他平台/engine 支持。并发调用同一 runtime 会串行保护官方 packed cache；全局 engine 必须由应用在并发开始前固定，packed state 每次加载重建。
