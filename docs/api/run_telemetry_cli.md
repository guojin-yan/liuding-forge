# CLI：查询与导出运行遥测

`run events/metrics/export` 读取已有 `runs.sqlite3` 的规范事件，不创建缺失存储或伪造观测。路径使用 UTF-8；生产者通过 `SQLiteRunStore`/`SQLiteRunTelemetrySink` 写入，CLI 不提供未经校验的 JSON 注入入口。

所有命令接受 `--format json|text`、`--root PATH`、`--id RUN_ID`、`--timeout-ms N`（默认60,000、最大600,000）和 `--page-bytes N`（默认/最大16 MiB、最小1 MiB）。timeout 是覆盖打开、读取、导出与关闭的一条绝对单调时钟截止，不是观测时间戳；文件系统调用不提供无限停滞时的可中断保证。

## 固定快照查询

```text
liuding-forge run events --root local-runs --id run-001 --limit 128 --format json
liuding-forge run events --root local-runs --id run-001 --after 128 --snapshot 600 --limit 128 --format json
liuding-forge run metrics --root local-runs --id run-001 --metric validation.accuracy --limit 128 --format json
liuding-forge run metrics --root local-runs --id run-001 --metric validation.accuracy --after-sequence 80 --after-index 2 --snapshot 600 --format json
```

首次无游标查询返回 `data.snapshot_through`；后续保持 run、过滤器和该 `--snapshot`，使用返回的 `data.next_after`，直到 null。非零游标必须显式指定快照，不能超出边界；空历史快照0合法。这是追加日志上的逻辑快照，不跨命令持有 SQLite reader。

| 命令 | 额外选项 |
| --- | --- |
| `events` | `--after N`（exclusive、默认0）、`--snapshot N`、`--limit N`（默认256、最大1,024）、`--kind NAME` |
| `metrics` | 必需 `--metric ID`；`--after-sequence N`、`--after-index N`（默认0）；`--snapshot N`；`--limit N`（默认256、最大4,096）；`--definition-version N`、`--step-kind NAME` |

事件 kind 为 `lifecycle/metric_batch/log/span/resource_sample/artifact_reference`；step kind 为 `micro_step/global_step/epoch/sample/elapsed_nanoseconds`。游标是可用有符号64位表示的非负数据库坐标，原 metric step 保留 uint64 全范围。规范源字节与条数同时受限，JSON 响应另限64 MiB，超限时缩小分页。

`events` 返回 `schema_version/run_sequence/content_digest/canonical_hex/event`；SHA256和规范字节是持久化依据，JSON 数字格式不是规范编码。`metrics` 返回点游标、源事件摘要、完整不可变定义/上下文/step/status/value及可选 sample count，`data.source_events` 在每页去重保存完整来源。NaN、正负无穷和 missing 的 JSON value 为 null，状态分别保留；缺失时钟为 null，负 wall time 保持有符号值，不推断采样数、不重新聚合或拼接重置后的 step。

## 原生导出

```text
liuding-forge run export --root local-runs --id run-001 --backend jsonl --output exported.jsonl --format json
liuding-forge run export --root local-runs --id run-001 --backend jsonl --output exported.jsonl --resume true --snapshot 600 --format json
liuding-forge run export --root local-runs --id run-001 --backend tensorboard-native --output exports/run-001 --format json
liuding-forge run export --root local-runs --id run-001 --backend tensorboard-native --native-config tensorboard.json --format json
liuding-forge run export --root local-runs --id run-001 --backend opentelemetry-native --native-config otlp.json --format json
```

| backend | 配置与行为 |
| --- | --- |
| `jsonl` | `--output`；可用显式 `--resume true` 核验相同事件身份并修复不完整尾部 |
| `tensorboard-native` | 新的 `--output` 目录直接写入，或用 `--native-config` 启动隔离 C++ worker；二者互斥，不接受 resume |
| `opentelemetry-native` | 必需 `--native-config`，由隔离 C++ worker 使用原生 HTTP 发送 |

旧 `tensorboard/opentelemetry --sidecar-config` 路由和 Python runtime 安装入口已退役；命令明确拒绝，迁移到上表原生 backend。产品导出不需要 Python。原生组件未启用时明确返回 `not_implemented`，不回退到脚本。

TensorBoard worker 配置示例（路径须为绝对路径，输出目录必须尚不存在）：

```json
{"schema_version":1,"directory":"C:/exports/run-001","working_directory":"C:/exports/worker","inherit_environment":true}
```

OTLP worker 的 endpoint、资源实例、显式时钟锚点与容量配置见[原生 OTLP HTTP](../tutorials/native-otlp-http.md)；TensorBoard 映射范围和重放限制见[原生 TensorBoard](native_tensorboard.md)。不要从事件载荷推断执行程序、网络目的地、目录或环境。

导出先冻结快照再分页，`--after` 默认0，`--snapshot` 可显式指定，`--page-size` 默认256、最大1,024，`--max-events` 默认/最大1,000,000。超过容量在启动 worker 前拒绝；输出不能进入受管理的 SQLite/CAS 根。JSONL 不修改原事件字节，官方映射不能表示的内容计入 unsupported，required durable evidence 不允许作为 sampled 丢弃。

成功结果包含 `scan_completed/fully_represented`、读取和各类回执计数、快照边界及最后游标。`fully_represented` 只有 unsupported和sampled均为0时为 true。C++ worker 成功还必须取得 `data.transport.process_tree_cleanup_confirmed: true`；未确认清理返回 backend错误、无成功 data。进程隔离与截止合同见[进程传输](telemetry_sidecar_process.md)。

| 退出码 | 含义 |
| --- | --- |
| 0 | 扫描完成且全部表示 |
| 1 | 扫描完成，但有 unsupported或sampled |
| 2 | 参数、配置、协议或未启用组件错误 |
| 3 | 存储、后端或未确认进程清理错误 |
| 124 | 命令或进程操作截止到达 |

外部效果按至少一次处理：协议响应确认、HTTP200或进程成功均不等于持久 ACK、exactly-once或可信 HTTPS 的完整资格。失败保留 SQLite 原事件和真实错误，再按原身份决定重放；进程取消不依赖 worker 合作，但系统调用及清理宽限不承诺绝对墙钟上限。
