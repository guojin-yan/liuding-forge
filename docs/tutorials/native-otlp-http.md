# 原生 C++ 发送 OTLP 遥测

这个示例从四个预测值和目标值计算平均绝对误差 `0.125`，向已配置的 OpenTelemetry Collector 发送指标及其审计日志，再重放相同事件，演示进程内去重。示例不训练模型、不启动服务，也不自动下载依赖。

## 构建

在现有受支持的 LibTorch 构建配置上启用 `JYPPX_LIUDING_BUILD_TELEMETRY_NATIVE=ON`、`JYPPX_LIUDING_BUILD_TELEMETRY_HTTP=ON` 和 `JYPPX_LIUDING_BUILD_EXAMPLES=ON`，并显式提供已验证的 `JYPPX_LIUDING_NANOPB_SOURCE_DIR`。Windows 使用系统 WinHTTP；Linux 还需用 `JYPPX_LIUDING_TELEMETRY_HTTP_CURL_ROOT` 指定共享系统 curl SDK，例如 `/usr`，要求 curl ≥ 7.81、TLS 和异步 DNS 支持。构建目标是 `jyppx_liuding_otlp_http_example`。

安装包使用 `find_package(jyppx CONFIG REQUIRED COMPONENTS telemetry_http)`，链接 `jyppx::liuding::telemetry_http`。Linux HTTP 安装消费者也要显式选择自己的系统 curl SDK 前缀；只请求 `sdk` 或 `telemetry_native` 的消费者不发现这个网络 SDK。未指定组件或请求 `all` 时会发现安装包的全部依赖。

## 运行

先自行启动启用 OTLP HTTP/protobuf 接收器的 Collector，再指定主机、端口、传输方式和本次进程的资源实例标识：

```text
liuding-forge-example-otlp-http 127.0.0.1 4318 http local-demo-1
```

HTTPS 示例：

```text
liuding-forge-example-otlp-http collector.example.com 4318 https training-worker-1
```

主机参数只接受主机名或不带方括号的 IP 字面量，不接受完整 URL。明文 HTTP 仅允许 `127.0.0.1` 和 `::1`；HTTPS 始终验证系统信任链和主机名。此接口不使用环境代理、凭据、自动认证、Cookie 或重定向，也不支持自定义请求头。

程序成功时输出一个事件的协议响应已通过校验，以及本进程内重放已去重。Collector 应收到一次 `/v1/metrics` 和一次 `/v1/logs` 请求。重复事件不会再次联网。

## 接入应用

包含 `<jyppx/liuding/telemetry_http/otlp_http.hpp>`，填充 `OtlpHttpOptions` 后创建 `OtlpHttpTelemetrySink`；它实现公共 `TelemetrySink` 接口。完整调用见 [examples/experiment/otlp_http.cpp](../../examples/experiment/otlp_http.cpp)。生产训练仍应将持久化 run store 作为证据来源，由调用方串行使用 sink，并保留失败批次以便原样重试。

整个批次先验证和编码，然后才开始发送；一个事件的所有请求均收到 HTTP 200 并通过官方 protobuf 响应校验，才写入进程内去重索引。若指标发送成功但审计日志失败，重试可能再次发送指标。协议响应确认不表示 Collector 已持久化，索引也不跨进程保存。`flush` 没有待发送队列，析构不会发送网络请求。

## 失败与边界

HTTP 非 200、缺失或错误的 `Content-Type`、不支持的内容编码、部分成功响应、损坏或过大的 protobuf、证书错误都会使调用失败。响应应使用 `application/x-protobuf`，允许合法媒体类型参数，内容编码只接受未编码或 `identity`。先检查 Collector 接收器与证书配置，再原样重试；不要修改已发送事件的幂等标识或内容。

每次请求默认有 30 秒预算，并受调用方的更早绝对截止时间约束。取消是协作式的，Windows 同步系统调用可能在返回后才发现取消或超时；这不是进程级强制终止能力。需要这种隔离的调用方应保留独立进程边界。编码字节与索引字段限制不等于 RSS 或网络库的内部内存上限。
