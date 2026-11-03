# 原生遥测 Collector 互操作夹具

此目录保留固定官方 Collector 0.135.0 的配置及制品摘要；该路径名称沿用历史身份。产品导出由 C++ TensorBoard/OTLP worker执行，旧 Python sidecar产品入口已退役。

`NativeOtlpCollectorInterchange` 使用 `tests/scripts/test_native_otlp_collector.py` 检查原生 API、SQLite CLI重放与真实 Collector；TensorBoard与离线 OTLP的独立官方读写参考位于 `tests/reference/telemetry`，不安装到 SDK。启用 `JYPPX_LIUDING_TEST_OFFICIAL_TELEMETRY` 时必须提供锁定的测试 CPython、Collector及原生 telemetry组件，不静默跳过。

历史运行结果仍绑定原源码和原协议，不转授当前 HTTPS、持久 ACK或完整章节资格。

`NativeOtlpCollectorDurableQueue` 使用同一锁定 Collector 和原生 HTTP harness，验证 `collector-durable-queue.yaml` 中无前置 batch processor、`file_storage`、`fsync: true` 与持久 sending queue 的组合。下游先持续返回 503，原生客户端确认接受并完成进程内重复检查后，测试直接强杀 Collector，再以原队列重启；不重放客户端，要求原始事件完整到达下游，并再次重启检查已完成项不再发送。相同配置去掉 queue 的 storage 后作为非持久对照，确认已获协议响应的事件在该崩溃观察中丢失。

启用上述 Collector 测试开关、构建 `jyppx_test_otlp_http_harness` 后，以 `ctest --test-dir <构建目录> -R '^NativeOtlpCollectorDurableQueue$' --output-on-failure` 运行；每次在构建目录 `tests/collector-queue/` 保留独立配置、原生输出、Collector 日志、强杀/回收状态及完整事件观察，失败不删除。Python 仅监督回环测试，不承担产品传输或安装运行功能。资格限于所选 Collector 配置的进程崩溃边界；不覆盖断电、磁盘损坏、任意 Collector 配置、通用 HTTP 200 或 exactly-once，未确认请求和下游结果不明时仍可能重复。
