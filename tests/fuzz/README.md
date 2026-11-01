# 有界解析器模糊测试

基线包含 `device_parse`、`data_pipeline_state`、`iterable_pipeline_state` 和以下六个生产解析边界，共九个 libFuzzer target。解析器测试不访问网络或业务文件系统。

| Target 后缀 | 生产边界 | 成功输入的检查 |
| --- | --- | --- |
| `telemetry_codec` | 二进制 envelope 与指标定义 | 规范字节、SHA-256、编解码稳定性、精确字节限额 |
| `checkpoint_manifest` | 严格 checkpoint JSON | 解析、规范编码、再解析、精确字节限额 |
| `model_bundle_manifest` | 严格模型 bundle JSON | 规范 JSON、稳定签名载荷、精确字节限额 |
| `safetensors` | 完整头、形状与 extent 解析 | 元数据、Tensor 关联、不透明载荷、规范稳定性、文件限额 |
| `evaluation_json` | suite、measured-input、quality-report | 规范 JSON、字节限额、报告自比较、HTML 转义 |
| `backend_protocol` | `OfficialTelemetrySink` reply | emit/flush/close receipt、不可变重放、reply 限额、计数单调性、拒绝原子性 |

新增的六个 harness 接收不超过 64 KiB 的输入，并限制集合、字符串和形状大小。每个 harness 验证生产编码生成的种子，再对有效种子执行最多八次编辑；原始输入路径仍可修改全部字节。`generate_corpus.py` 生成这六类数据语料，C++ 可执行程序独立验证实际 codec 种子。

只捕获文档约定的 `jyppx::liuding::Error` 输入拒绝类别；内部错误、标准异常、不变量失败和 sanitizer 错误必须暴露。只有协议 harness 接受后端错误。输入解析成功后，后续编码或再解析失败不能重新归为非法输入。规范输出与转义 HTML 分别受限，并检查精确大小成功与少一个字节时的拒绝；safetensors 值保持不透明位模式。

`JYPPX_LIUDING_BUILD_FUZZERS=ON` 要求 Clang 和完整 SDK，并为框架实现归档启用 `fuzzer-no-link`，第三方库不重编译为 libFuzzer coverage。

```sh
CMAKE_PREFIX_PATH=/path/to/libtorch JYPPX_FUZZ_RUNS=10000 bash scripts/validation/run-fuzz.sh
```

runner 复用 `out/build/fuzz-linux`，在 `out/fuzz/linux` 保留发现的语料、日志与 reproducer。九个 target 使用 ASan/UBSan、固定种子、单输入超时和进程 RSS 限额；可设置 `JYPPX_LIUDING_BUILD_PARALLEL_LEVEL`、`JYPPX_FUZZ_RUNS`、`JYPPX_FUZZ_SEED`、`JYPPX_FUZZ_RSS_LIMIT_MB`。失败即停止并保留证据，不能跳过失败。该检查不替代独立的官方 sidecar 请求/协议集成验收。
