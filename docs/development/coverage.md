# 覆盖率 owner 与采集 profile

`coverage-policy.json` v2 将源码归属与已审阅的数值基线分开。现有 12 个 owner 的 covered、total、minimum_percent 以及历史观测保持不变；新增组件先登记 `baseline_status: pending-observation`，不填写 metrics。存在 pending owner 时，coverage gate 始终拒绝通过，即使该次报告的比例达到 100%；须保留真实观测并完成基线审阅，不能清零或降低旧 ratchet 来消除失败。

## 只检查源码归属

在源码根目录执行以下维护命令，不需要运行 CMake、产品、gcovr 或真实覆盖率采样：

```bash
python3 scripts/validation/coverage_gate.py coverage-policy.json \
  --audit-owners --source-root . --output out/coverage-owner-review/owners.json
```

该命令扫描 `include/jyppx` 与 `src/jyppx` 的 C/C++ 候选，每个文件须恰有一个组件 owner；报告未出现的新模块也不能逃过归属检查。`liuding.hpp` 是唯一登记的非执行例外，必须同时匹配完整 SHA-256 和纯 include/guard 语法；字节改变、出现可执行实体或报告含非零指标均拒绝。组件的公开头、私有 inline、生成代码仍留在各自 owner，生成文件不会因来源而自动排除。

owner audit 的 exit 0 仅表示静态归属有效，输出明确包含 `coverage_evaluated=false`、`release_qualified=false`。它不表示 CTest、基线或发布门槛通过。

## 实际选定采集 profile

Linux lane 使用 `v1-selected-cpu`，显式开启 Core/interop/data/nn/optim/training/experiment、data_image/vision_render/models/serving、onnx_export/onnx_runtime/onnx_serving、bundle_signing、telemetry_native/telemetry_http 和 tools。它对应该次选定的 19 个 owner；Windows 专用 WinHTTP 源码保持属于 telemetry_http，Linux 只编译其 curl 路径。该 CPU lane 不替代 Windows、声明 CUDA、实际网络、任务质量或资源资格。

ONNX 导出要求精确 LibTorch 2.14.1，旧 2.5.1 CPU coverage 不能冒充本 profile。ORT、签名 OpenSSL、nanopb、Linux 系统 curl 以及图像依赖均由协调者先盘点并提供显式的已审阅依赖输入。lane 不猜 SDK 根、版本或摘要，也不会因缺输入静默关闭组件。

```bash
export CMAKE_PREFIX_PATH='<已审阅的绝对 LibTorch 路径>'
export JYPPX_COVERAGE_DEPENDENCY_CACHE='<已审阅的绝对 CMake 初始缓存路径>'
export JYPPX_COVERAGE_DEPENDENCY_CACHE_SHA256='<已冻结的64位小写SHA-256>'
export JYPPX_COVERAGE_OFFLINE_INPUTS='<已审阅的绝对本地输入JSON路径>'
export JYPPX_COVERAGE_OFFLINE_INPUTS_SHA256='<该JSON已冻结的64位小写SHA-256>'
export JYPPX_LIUDING_BUILD_PARALLEL_LEVEL=2
bash scripts/validation/run-coverage.sh
```

依赖缓存是协调者审阅的 CMake 输入文件，SHA 必须事先冻结；它标识输入字节，不代替对应 SDK/二进制、许可或安全审查。另提供的本地输入 JSON 必须绑定同一 init-cache SHA，完整列出 nlohmann JSON、SQLite 和 GoogleTest 的锁定本地归档与预展开源码根：

```json
{
  "schema_version": 1,
  "kind": "jyppx-coverage-offline-inputs",
  "init_cache_sha256": "<实际init-cache的64位小写SHA-256>",
  "fetchcontent": {
    "jyppx_nlohmann_json": {"archive": "/reviewed/json.tar.xz", "source_root": "/reviewed/json"},
    "jyppx_sqlite": {"archive": "/reviewed/sqlite.zip", "source_root": "/reviewed/sqlite"},
    "googletest": {"archive": "/reviewed/googletest.zip", "source_root": "/reviewed/googletest"}
  }
}
```

上述是输入模板，路径和 init-cache SHA 须替换为协调者实际审阅的身份。归档摘要由 `coverage_dependencies.py` 固定，调用者不能另填摘要替换依赖。helper 在 CMake 前核 JSON/init-cache SHA、三个归档身份、完整安全成员边界，以及每个预展开树与相应归档的完整文件集合和字节；不解包、不修复缓存、不下载。缺失、修改、额外文件、链接或越界成员均前置拒绝。FULLY_DISCONNECTED 不负责首次 populate，所以只有归档而没有预展开源码根仍会拒绝。

lane 强制 `FETCHCONTENT_FULLY_DISCONNECTED:BOOL=ON` 和 `FETCHCONTENT_UPDATES_DISCONNECTED:BOOL=ON`，并显式传入三个已验证的 `FETCHCONTENT_SOURCE_DIR_*:PATH`。configure 后先核断网 BOOL 与精确 source PATH，再核原 18 个 profile 选项确为预期 BOOL/type/value，随后才构建。结束前再次核输入与实际 cache，记录初次和结束回执。该机制覆盖当前三条 FetchContent fallback；init-cache、SDK 配置和测试代码仍是需根审阅的代码输入，不宣称为任意代码提供网络沙箱。

共享平台构建由协调者串行安排，编译与 gcovr 并行仅允许 1 或 2，并覆盖 CMAKE_BUILD_PARALLEL_LEVEL。CTest 同时显式 `--parallel 1`、覆盖继承的 CTEST_PARALLEL_LEVEL；训练不扩并行。

lane 要求 clean committed source，使用独立 `out/coverage/linux/{build,report}`，保留依赖输入 SHA、实际配置缓存 SHA、源码身份、CTest 与 policy 分别的状态。CTest 失败后仍按原合同尝试采集；报告采集成功不等于 gate 通过。新 owner 未出现或没有三项实际分母也继续拒绝，不能为缺观测生成基线数字。

## 维护验证与基线审阅

`CoverageOwnerPolicyToolUnitTests` 使用小型明确标记的 synthetic fixture，检查 pending 拒绝、漏 owner、前缀交叠、聚合头例外及 profile 缓存拒绝。`CoverageOfflineDependenciesToolUnitTests` 以合成归档/源码检查锁定身份、缺失、篡改、成员边界、链接和实际 cache 的拒绝；CLI 的生产 pin 不接受调用者覆盖。Linux `CoverageLaneProtocolUnitTests` 使用假 CMake/CTest/gcovr 和明确替换为 synthetic pin 的私有 helper 副本，验证继承并行、输入缺失、断网 cache 改动、结束身份变化及错误路径。这些 fixture 数字和 pin 均不进入实际 policy 或发布证据。

真实采样后，先审阅每组件来源、完整 instrumentation 和观测身份，再登记新增 owner 的真实基线；旧 12 个 owner 的三重 ratchet 仍逐项检查。历史 data/optim 分母变化、Serving 比例缺口及既有失败必须独立审计，不因 schema 升级或新增 owner 而关闭。
