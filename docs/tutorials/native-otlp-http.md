# 原生 C++ 发送 OTLP 遥测

这个示例从四个预测值和目标值计算平均绝对误差 `0.125`，向已配置的 OpenTelemetry Collector 发送指标、完整审计日志、计算说明及实际计算区间的 span，再重放相同事件，演示进程内去重。可选先将事件提交到 SQLite run，便于网络失败后重新导出。示例不训练模型、不启动服务，也不自动下载依赖。

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

程序成功时输出三个事件的协议响应已通过校验，以及本进程内重放已去重。Collector 应收到一次 `/v1/metrics`、两次 `/v1/logs` 和一次 `/v1/traces` 请求。重复事件不会再次联网。

Windows 与 Linux 已验证官方 OpenTelemetry Collector Contrib 0.135.0 的明文回环 HTTP/protobuf 接收：公共 C++ API、SQLite 事件经 CLI 重放，以及六类中立事件的原生 worker 均可被官方协议 reader 读取。此范围不包含 HTTPS 互操作、持久化确认或进程强取消。

需要保留本地记录时，追加一个父目录已存在、目标目录尚不存在的路径：

```text
liuding-forge-example-otlp-http 127.0.0.1 4318 http local-demo-1 ./mae-run
```

程序先把三个事件写入 `run.otlp-example`，再访问网络。网络失败会返回错误，但已提交的本地事件可以通过下面的 CLI 流程重新导出；再次运行示例应选择新的目录。

## 接入应用

包含 `<jyppx/liuding/telemetry_http/otlp_http.hpp>`，填充 `OtlpHttpOptions` 后创建 `OtlpHttpTelemetrySink`；它实现公共 `TelemetrySink` 接口。完整调用见 [examples/experiment/otlp_http.cpp](../../examples/experiment/otlp_http.cpp)。生产训练仍应将持久化 run store 作为证据来源，由调用方串行使用 sink，并保留失败批次以便原样重试。

整个批次先验证和编码，然后才开始发送；一个事件的所有请求均收到 HTTP 200 并通过官方 protobuf 响应校验，才写入进程内去重索引。若指标发送成功但审计日志失败，重试可能再次发送指标。协议响应确认不表示 Collector 已持久化，索引也不跨进程保存。`flush` 没有待发送队列，析构不会发送网络请求。

## 失败与边界

HTTP 非 200、缺失或错误的 `Content-Type`、不支持的内容编码、部分成功响应、损坏或过大的 protobuf、证书错误都会使调用失败。响应应使用 `application/x-protobuf`，允许合法媒体类型参数，内容编码只接受未编码或 `identity`。先检查 Collector 接收器与证书配置，再原样重试；不要修改已发送事件的幂等标识或内容。

每次请求默认有 30 秒预算，并受调用方的更早绝对截止时间约束。取消是协作式的，Windows 同步系统调用可能在返回后才发现取消或超时；这不是进程级强制终止能力。需要这种隔离的调用方应保留独立进程边界。编码字节与索引字段限制不等于 RSS 或网络库的内部内存上限。

## 导出已有训练 run（预览）

同时启用 `JYPPX_LIUDING_BUILD_TOOLS=ON` 并构建 `jyppx_cli`，可以显式选择 `opentelemetry-native`，由 C++ CLI 把 SQLite 中的原始事件交给独立 C++ worker，再调用同一个公共 HTTP sink。用户运行时不需要 Python。进程强取消和完整 TLS 互操作仍需独立验收。

以前面的 `mae-run` 为例，将 `$cli` 设为本次构建的 `liuding-forge` 可执行文件。span 使用生产时的时钟锚点，必须从原事件取值，不能用导出时的当前时间代替。创建独占工作目录和 UTF-8 配置，再导出原始事件：

```powershell
$store = (Resolve-Path './mae-run').Path
$events = (& $cli run events --root $store --id run.otlp-example --format json | ConvertFrom-Json).data.events
$span = ($events | Where-Object { $_.event.payload.kind -eq 4 }).event
$otlpWork = Join-Path (Get-Location) 'otlp-worker'
New-Item -ItemType Directory -Path $otlpWork | Out-Null
$otlpConfig = Join-Path (Get-Location) 'otlp-native.json'
$config = @{
    schema_version = 1
    endpoint = @{ host = '127.0.0.1'; port = 4318; tls = $false }
    resource_instance_id = 'mae-run-export'
    clock_mappings = @{ 'example.clock' = @{
        monotonic_ns = $span.payload.start_monotonic_ns
        unix_ns = $span.time.wall_time_unix_ns
    } }
    working_directory = $otlpWork
    inherit_environment = $true
    operation_timeout_ms = 10000
    request_timeout_ms = 10000
    cleanup_timeout_ms = 2000
} | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText($otlpConfig, $config, [Text.UTF8Encoding]::new($false))
& $cli run export --root $store --id run.otlp-example --backend opentelemetry-native --native-config $otlpConfig --timeout-ms 60000 --format json
```

`working_directory` 必须是已存在的独占绝对路径。示例显式继承环境以加载当前 LibTorch 动态库；默认不继承，也可以通过 `environment` 显式提供子进程环境。配置不接受可执行文件、额外启动参数或请求头；CLI 固定启动自身，并校验 worker 读取的配置字节与父进程一致。HTTPS 使用 `endpoint.tls: true`，沿用前述证书与主机约束。

成功结果的 `scan_completed` 表示快照已完整扫描，`fully_represented` 表示没有采样或不支持事件；应同时检查进程退出码和各类事件计数。没有匹配显式时钟锚点的 span 可能返回 `unsupported`；如需导出 span，在配置的 `clock_mappings` 中按生产者 `clock_id` 填入已捕获的 `monotonic_ns` 与 `unix_ns`，不要在导出时推测历史时钟。

每个 exchange 使用 `operation_timeout_ms`，整个命令还受 `--timeout-ms` 的更早截止约束。到达截止后，父进程请求终止 worker；`cleanup_timeout_ms`（默认 2000、最多 10000）约束首次清理开始后的排空和退出等待，不用于无限延长 flush。Windows 父端使用同步非阻塞管道，在清理时没有待完成的异步 I/O 或应用主动建立的无限完成等待；操作系统调用和调度仍不提供绝对墙钟上界，强取消资格仍需独立验收。Windows 使用 Job Object；Linux 使用进程组，后者不是针对自行脱组恶意进程的沙箱，也不能据此声称所有后代与内核资源已按时回收。

命令失败时返回非零，不自动重试。批次中的指标与审计日志成对确认；部分请求失败可能已在 Collector 形成数据，再执行导出可能重复。去重只属于当前 worker，重新执行命令会创建新 worker；始终保留 run store 作为证据来源，交付语义为至少一次。一次快照默认最多 100000 个事件，`max_seen` 可缩小这个上限，不会通过静默淘汰索引扩容；还会检查索引字节与批次准备字节上限。
