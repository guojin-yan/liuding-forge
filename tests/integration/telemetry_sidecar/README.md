# 原生遥测 Collector 互操作夹具

此目录保留固定官方 Collector 0.135.0 的配置及制品摘要；该路径名称沿用历史身份。产品导出由 C++ TensorBoard/OTLP worker执行，旧 Python sidecar产品入口已退役。

`NativeOtlpCollectorInterchange` 使用 `tests/scripts/test_native_otlp_collector.py` 检查原生 API、SQLite CLI重放与真实 Collector；TensorBoard与离线 OTLP的独立官方读写参考位于 `tests/reference/telemetry`，不安装到 SDK。启用 `JYPPX_LIUDING_TEST_OFFICIAL_TELEMETRY` 时必须提供锁定的测试 CPython、Collector及原生 telemetry组件，不静默跳过。

这些测试分别检查原生协议与 Collector 互通；HTTPS 信任、远端持久 ACK 和跨进程去重不由进程测试代替。

`NativeOtlpCollectorDurableQueue` 使用同一锁定 Collector 和原生 HTTP harness，验证 `collector-durable-queue.yaml` 中无前置 batch processor、`file_storage`、`fsync: true` 与持久 sending queue 的组合。日志、指标及其源审计、追踪各有持久恢复和非持久丢失对照，共六场景；指标核对 gauge 值、定义和完整源审计，追踪核对时间锚点与确定性 trace/span ID。下游先持续返回 503，原生客户端确认接受并完成进程内重复检查后，测试直接强杀 Collector，再以原队列重启；不重放客户端，要求该事件的所有请求完整到达下游，并再次重启检查已完成项不再发送。相同配置去掉 queue 的 storage 后作为非持久对照，确认已获协议响应的事件在该崩溃观察中丢失。

启用上述 Collector 测试开关、构建 `jyppx_test_otlp_http_harness` 后，以 `ctest --test-dir <构建目录> -R '^NativeOtlpCollectorDurableQueue$' --output-on-failure` 运行；每次在构建目录 `tests/collector-queue/` 保留独立配置、原生输出、Collector 日志、强杀/回收状态及完整事件观察，失败不删除。Python 仅监督回环测试，不承担产品传输或安装运行功能。资格限于所选 Collector 配置的进程崩溃边界；不覆盖断电、磁盘损坏、任意 Collector 配置、通用 HTTP 200 或 exactly-once，未确认请求和下游结果不明时仍可能重复。

`NativeOtlpCollectorQueueFull` 使用相同持久配置，以每信号 16 个原生事件填满容量为 4、单消费者的队列。记录转发层保留原始 protobuf 和 Collector 实际拒绝状态；下游恢复并排空后，在同一原生 sink 中重试完整原批次，检查已确认前缀不再联网、未确认后缀按原字节补发，最终全批次为进程内重复。指标场景让指标下游可用、审计日志下游阻塞，明确验证指标请求成功而审计失败时，未确认事件的指标会在重试中重复；逐项核对三信号身份、完整审计、指标定义与追踪时钟。通过 `ctest --test-dir <构建目录> -R '^NativeOtlpCollectorQueueFull$' --output-on-failure` 运行，输出保留在 `tests/collector-queue-full/`；此项不授予磁盘故障、跨进程去重或 exactly-once 资格。

`NativeOtlpCollectorLostReply` 覆盖日志、追踪以及指标主请求/审计请求四个响应丢失边界。原生每批三个事件，转发层收到并保存 Collector 对第二个事件所选请求的真实成功响应后，在向原生客户端发送任何响应字节前断开连接；先验证原生失败且只确认前一个事件，再开放下游，要求这些已被 Collector 接受的请求在客户端重试前完整到达。原批次随后在同一 sink 中重试，前缀不再联网、未确认后缀保持原始 protobuf，第二个事件已产生的下游效果明确重复，最终三事件均确认且再次重放全为进程内重复。以 `ctest --test-dir <构建目录> -R '^NativeOtlpCollectorLostReply$' --output-on-failure` 运行，原始请求、被丢弃的 Collector 响应、进程终态和完整下游内容保留在 `tests/collector-lost-reply/`；此项不授予跨进程持久去重、磁盘故障或 exactly-once 资格。
