# 原生 TensorBoard 事件导出

`TensorBoardTelemetrySink` 直接用 C++ 写入 TensorBoard 事件文件，无需 Python 导出进程。它接收与训练和实验存储相同的 `TelemetryEnvelope`，支持生命周期文本、标量指标与资源测量，使用 binary64 保存数值。TensorBoard 查看器可独立安装和使用。

## 构建与链接

此后端默认关闭。准备 nanopb 0.4.9.2 官方源码后，向现有 CMake 配置追加：

```sh
-DJYPPX_LIUDING_BUILD_TELEMETRY_NATIVE=ON
-DJYPPX_LIUDING_NANOPB_SOURCE_DIR=/path/to/nanopb-0.4.9.2
```

构建会核对运行库源码的固定 SHA-256，不在配置时联网下载。安装包用户通过 `find_package(jyppx CONFIG REQUIRED COMPONENTS telemetry_native)` 发现组件，链接 `jyppx::liuding::telemetry_native`；消费安装包无需 nanopb SDK、protoc 或 Python。查看器与独立验收工具各自安装，不属于库的运行时依赖。

## 在 C++ 中使用

```cpp
#include <jyppx/liuding/telemetry_native/tensorboard.hpp>

namespace forge = jyppx::liuding;
forge::experiment::TensorBoardTelemetrySink observer({
    .directory = "exports/run-001"});  // 新目录，父目录需已存在

// events 是调用方保留的不可变 TelemetryEnvelope 批次。
const auto receipts = observer.emit(events);
const auto counters = observer.close();
```

链接 `jyppx::liuding::telemetry_native`。调用方串行使用同一实例，并独占输出目录；成功的 `emit` 已将字节交给操作系统，不另设后台队列。`close` 返回最终计数，多次成功关闭返回相同结果。析构只释放句柄，不能代替显式关闭。

## 导出已保存的运行

```sh
liuding-forge run export --root local-runs --id run-001 --backend tensorboard-native --output exports/run-001 --format json
tensorboard --logdir exports
```

`--output` 在当前 CLI 进程内直接写文件。输出目录必须是新目录，并位于实验存储目录之外。原生后端不接受 `--sidecar-config` 或 `--resume`；需要重新导出时，选择另一个新目录。`--after` 与 `--snapshot` 可选定原始存储快照中的一段事件。JSON 响应分别返回接受、重复和不支持的数量；扫描完成但含未表示事件时，退出码为 `1`，详情见 `fully_represented`。

### 使用受监督的隔离进程

需要由 CLI 管理导出进程的等待和清理预算时，改用 `--native-config`；它与 `--output` 互斥。先创建输出父目录和独立工作目录，再保存以下 UTF-8 JSON。路径均须绝对，`directory` 本身不得预先创建。

```json
{
  "schema_version": 1,
  "directory": "/absolute/path/exports/run-001",
  "working_directory": "/absolute/path/worker"
}
```

```sh
liuding-forge run export --root local-runs --id run-001 --backend tensorboard-native --native-config /absolute/path/tensorboard.json --timeout-ms 60000 --format json
```

隔离模式启动当前 CLI 自身，使用同一个 C++ writer，不调用 Python；配置和事件不能选择其他可执行文件。配置文件上限为 64 KiB，启动前后核对 SHA-256，拒绝未知字段、重复 JSON 键、无效版本和越界容量。请求帧上限为 16 MiB，响应帧上限为 128 KiB，均不含结尾换行；协议输出不混入普通 CLI 文本。JSON 结果中的 `transport` 给出已完成请求和传输字节数，直接模式中该字段为 `null`。

| 可选配置 | 默认值与上限 |
| --- | --- |
| `operation_timeout_ms`、`startup_timeout_ms` | 各 10,000 ms，最大 600,000 ms |
| `cleanup_timeout_ms` | 2,000 ms，最大 10,000 ms |
| `max_file_bytes`、`max_record_bytes`、`max_batch_encoded_bytes` | 默认且最大分别为 1 GiB、4 MiB、32 MiB；批次预算至少 16 字节 |
| `max_seen`、`max_seen_bytes` | 默认且最大分别为 100,000 个身份、32 MiB 逻辑索引内容 |
| `inherit_environment`、`environment` | 默认不继承；可显式继承或提供最多 128 个子进程环境覆盖项 |

数字配置必须为正整数。选定快照的事件数超过 `max_seen` 时，启动 worker 前即拒绝；字节预算仍在实际编码和写入前检查。CLI 检查输出位于 SQLite/CAS 存储之外，worker 启动时再次检查；调用方仍须独占工作和输出目录。父进程以请求等待和清理预算监督 worker；这些预算不构成对任意操作系统调用或调度的绝对墙钟保证。失败后保留原始运行存储，并改用新目录重新导出，不恢复旧事件文件。可运行的 PowerShell 步骤见[搜索与遥测示例](../../examples/experiment/README.md#原生-tensorboard)。

## 表示范围

| 输入 | TensorBoard 表示 |
| --- | --- |
| 生命周期 | 完整事件 JSON 文本，保留原始身份、时钟、决策及缺失指标；可表示的指标另写标量 |
| 指标批次、资源测量 | 带完整定义和 context 的 binary64 标量；NaN 和正负无穷按原状态写入 |
| 缺少 wall time、超出 int64 的 step 或生命周期 sequence | `unsupported`，不采样当前时间替代 |
| 独立指标/资源批次包含 missing | 整个事件返回 `unsupported`，不将缺失改成零或 NaN |
| 日志、span、制品引用 | `unsupported`，原始记录继续保存在实验存储中 |

系列名称包含完整定义、context 和 step 类型的 SHA-256，避免混合不同单位或进度坐标。生命周期用原始 `sequence` 作 step；标量使用各自声明的 step。精确 Unix 纳秒时间还保存在文本或标量元数据中。

实时显式构造的有限 `-0.0` 可按其符号写入；便捷工厂和 canonical 存储将正负零视为同一数值并归一成 `+0.0`，因此从持久运行恢复导出不保留负零符号。同一身份的重试按 canonical 语义判重，不更改首次写入的表示。

## 容量与失败处理

通过 options 限制记录大小、批次编码大小、文件大小和重放索引。默认单条 Event 记录上限为 4 MiB，单批编码上限为 32 MiB，事件文件上限为 1 GiB。索引容量按唯一事件计算，包括不支持的输入；已知输入重试不再占新名额。字节预算限制逻辑内容，不承诺进程 RSS 上限。

批次在写入前完成验证和容量检查。同一实例识别相同内容的重试，拒绝冲突的 operation、event 或 sequence；重启后的新实例使用新目录，不恢复原索引。写入失败或开始写入后的取消/超时会使实例不可继续使用，应保留持久事件并重放到新目录。同步系统文件调用可能晚于 deadline 返回，原生后端会检查这一情况并报错。

这是观察输出，不提供文件系统 `fsync`、崩溃恢复或外部效果恰好一次保证；持久运行事件仍由 SQLite/CAS 管理。原生 OTLP 的独立支持范围见[HTTP 导出教程](../tutorials/native-otlp-http.md)。

## 学习示例

构建 `jyppx_liuding_native_tensorboard_example` 后执行：

```sh
liuding-forge-example-native-tensorboard exports/learning-run
tensorboard --logdir exports
```

[C++ 示例](../../examples/experiment/tensorboard.cpp) 用四个样本的预测值和目标值计算平均绝对误差，导出生命周期、指标和缓冲区字节数，并重放同一批事件验证去重。它同时保存 `source-events.jsonl`，方便学习原始事件与 TensorBoard 表示的对应关系。示例不执行模型训练；接入真实训练时，将训练产生的不可变事件传给同一个 observer。
