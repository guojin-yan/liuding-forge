# Linux 覆盖率采集

运行入口为 `scripts/validation/run-coverage.sh`；选定 CPU profile、完整依赖输入模板、串行规则、owner/ratchet 和严格计数校验统一见[覆盖率维护说明](../../docs/development/coverage.md)。仅设置 `CMAKE_PREFIX_PATH` 不足以启动：还须提供已冻结 SHA-256 的依赖缓存和离线输入 JSON，以及与锁定归档逐件一致的预展开源码。

脚本使用独立的 `out/coverage/linux/build` 和 `out/coverage/linux/report`，成功构建后只清除该构建树的旧 `.gcda`。复用固定报告名之前，先归档原观测及其源码身份；显式模型参考输入、条件跳过、打包合同的单列证据和各阶段退出码均按完整说明记录。

计数校验以 `coverage_hits.py` 的当前规则和保存收据为准，不使用历史工作负载的固定命中数推导上限。CTest 或 policy 失败仍使整体返回非零；配置、构建或报告采集失败会停止后续阶段，`not_run` 不代表通过。CPU 覆盖率不授予 Windows、CUDA、真实任务质量或正式发布资格。
