# OTLP 离线协议准备

`OtlpProjectionEncoder` 将中立遥测事件编码为官方 OTLP metrics、logs、traces protobuf 请求，并校验 collector 的 protobuf 响应。编码由 C++ 完成，返回的请求拥有自己的二进制数据；运行时不需要 Python、protoc 或 OpenTelemetry SDK。此接口只准备协议数据，不发送网络请求，也不确认事件送达。

## 构建与接入

按[构建指南](../../CONTRIBUTING.md#构建与验证)启用 `JYPPX_LIUDING_BUILD_EXPERIMENT` 和默认关闭的 `JYPPX_LIUDING_BUILD_TELEMETRY_NATIVE`，并通过 `JYPPX_LIUDING_NANOPB_SOURCE_DIR` 提供固定版本 nanopb 0.4.9.2 的源码目录。配置阶段只校验既有文件，不下载依赖。安装后通过 `find_package(jyppx REQUIRED COMPONENTS telemetry_native)` 和 `jyppx::liuding::telemetry_native` 接入；nanopb 头文件、宏及生成描述符不进入公共接口。

```cpp
#include <jyppx/liuding/telemetry_native/otlp_projection.hpp>

namespace forge = jyppx::liuding;

forge::TelemetryEnvelope event{
    .identity = {.stream_id = "training", .event_id = "started",
                 .idempotency_key = "started", .sequence = 1U},
    .time = {.wall_time_unix_ns = 1700000000000000000LL},
    .correlation = {.project_id = "image", .run_id = "run.1",
                    .attempt_id = "attempt.1"},
    .payload = forge::TelemetryLifecycle{.name = "training.started"}};

forge::experiment::OtlpProjectionEncoder encoder(
    {.resource_instance_id = "worker.1"});
auto requests = encoder.encode(event);
// requests 中每项包含 signal 和完整二进制 body。
// encoder.signal_path(request.signal) 返回固定的 /v1/... 路径。
```

事件、指标定义及单位遵循[训练遥测](training_telemetry.md)。同一 encoder 可在不同线程调用；调用期间不得修改它正在读取的事件。`encode` 接受取消令牌和绝对 deadline，失败通过框架异常报告。

## 映射与边界

| 事件 | 协议结果 | 必要条件与保留信息 |
|---|---|---|
| 指标、资源观测、含指标的生命周期 | metrics 请求与完整审计 log 各一份 | 必须有非负 Unix 时间；两份都编码完成后才返回，缺失值不改为零 |
| 普通生命周期、日志、制品 | logs 请求 | 可缺少 Unix 时间，缺失状态保留；负时间不支持 |
| span | traces 请求 | `clock_mappings` 必须给出对应 `clock_id` 的显式锚点；映射下溢或溢出不支持 |

`resource_instance_id` 必须显式提供。指标 resource 只包含服务名和实例 ID，项目、run、attempt 等完整信息保存在审计 log；metric 名由完整定义摘要生成，单位和数值语义不变。trace/span 的原始不透明 ID 使用域隔离摘要映射，原值仍保留在属性中。不支持的事件返回空请求列表，调用者可显式统计。

span 时钟映射使用 `OtlpClockAnchor{monotonic_ns, unix_ns}`，锚点应由同一生产者采集。最多配置 64 个时钟；不能用接收时刻或另一台机器的单调时钟代替。

默认每份请求最多 16 MiB，一个事件准备的全部请求最多 32 MiB，响应最多 1 MiB；这些选项及中立事件限制只能收窄。上限约束协议缓冲区，不代表整个进程的内存峰值。接口不保留后台队列。

`validate_response(signal, bytes)` 使用官方 schema 解码，拒绝损坏、超限及包含拒收数量或非空错误信息的响应。空响应是合法的 protobuf 确认；校验通过仅说明这一份响应满足协议条件，HTTP 状态、网络取消、重试、持久化和完整事件送达仍需由发送端保证。当前此组件尚未提供原生 HTTP 发送器，不能用离线编码成功替代真实 collector 验收。

`make_telemetry_metric_value` 和中立事件的规范持久化将负零归一为正零；直接构造的合法有限值可在实时编码中保留负零。需要在恢复后重放时，应保留同一份规范事件，不改变已经提交的投影内容。

## 学习示例

[OTLP 示例](../../examples/experiment/README.md#离线-otlp-协议准备)计算四个样本的平均绝对误差，输出中立 JSONL、三类官方请求及请求索引，展示指标与审计日志、缺失数据、时钟映射和无效时间的处理。示例仅写本地文件，目标目录必须尚不存在。
