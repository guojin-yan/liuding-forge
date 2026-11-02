# 原生遥测 Collector 互操作夹具

此目录保留固定官方 Collector 0.135.0 的配置及制品摘要；该路径名称沿用历史身份。产品导出由 C++ TensorBoard/OTLP worker执行，旧 Python sidecar产品入口已退役。

`NativeOtlpCollectorInterchange` 使用 `tests/scripts/test_native_otlp_collector.py` 检查原生 API、SQLite CLI重放与真实 Collector；TensorBoard与离线 OTLP的独立官方读写参考位于 `tests/reference/telemetry`，不安装到 SDK。启用 `JYPPX_LIUDING_TEST_OFFICIAL_TELEMETRY` 时必须提供锁定的测试 CPython、Collector及原生 telemetry组件，不静默跳过。

历史运行结果仍绑定原源码和原协议，不转授当前 HTTPS、持久 ACK或完整章节资格。
