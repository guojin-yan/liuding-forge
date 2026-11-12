# API：本地运行存储

链接目标为 `jyppx::liuding::experiment`；公共头位于 `<jyppx/liuding/experiment/artifact_store.hpp>`、`run.hpp`、`run_store.hpp`、`run_events.hpp` 和 `run_telemetry_sink.hpp`，共享制品身份位于 `<jyppx/liuding/foundation/artifact.hpp>`。

`SQLiteRunStore` 管理运行元数据、内容寻址存储（CAS）、运行到制品的引用及规范遥测。`RunSink` 负责生命周期和附件，`RunEventSink` 接收原子事件批次；`SQLiteRunStore` 实现两者并提供有界事件/指标查询。`SQLiteRunTelemetrySink` 将共享遥测接口接到已有 SQLite run，不依赖训练组件。

## 目录与所有权

每个存储使用独立的本地目录：

```text
local-runs/
  runs.sqlite3
  runs.sqlite3-wal             # 存在时由 SQLite 管理
  runs.sqlite3-shm             # 存在时由 SQLite 管理
  artifacts/
    objects/aa/<其余 62 位小写 SHA-256 十六进制字符>
    tmp/.upload-<nonce>-<sequence>.part
```

`RunMetadata` 包含稳定的 run/project/experiment ID、显示名、状态、调用方提供的 Unix 毫秒时间戳、不可变 lineage 和标签。project/experiment ID 是元数据字段，当前没有独立的项目/实验 CRUD 接口；时间戳和显示名不能替代 run ID。

共享类型 `jyppx::liuding::ArtifactDigest` 使用 SHA-256 的64位小写十六进制摘要；`ArtifactRef` 包含摘要、字节数、逻辑名、媒体类型和生产者。`RunArtifactReference` 另含 run 内唯一的 reference ID、`artifact/checkpoint` 类型、语义角色和附件时间。`checkpoint` 标签只标识内容，不执行训练状态序列化或恢复。

## 创建运行并保存制品

将下面完整示例保存为 `main.cpp`。它保存普通内容字节、完成 run 并检查全部引用内容；真实模型的序列化由生产者负责。再次用同一目录执行时，原请求身份和内容保持不变，因此重放不会增加引用。

```cpp
#include <jyppx/liuding/experiment/run_store.hpp>

#include <exception>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>

int main(int argc, char* argv[]) {
  if (argc > 2) {
    std::cerr << "usage: run-storage-example [STORE]\n";
    return 2;
  }
  try {
    using namespace jyppx::liuding::experiment;
    const std::filesystem::path root = argc == 2 ? argv[1] : "local-runs";
    SQLiteRunStore runs(root);

    RunMetadata request;
    request.run_id = "run-001";
    request.project_id = "project-main";
    request.experiment_id = "experiment-baseline";
    request.display_name = "Local baseline";
    request.status = RunStatus::running;
    request.created_at_unix_ms = 100;
    request.updated_at_unix_ms = 100;
    request.lineage.source_digest = "source-identity";
    request.lineage.build_fingerprint = "build-identity";
    request.lineage.config_digest = "redacted-config-identity";
    request.lineage.data_manifest_digest = "dataset-manifest-identity";
    request.tags.emplace("recipe", "baseline");
    runs.start(request);

    const std::string payload = "example-content";
    const auto bytes = std::as_bytes(std::span(payload.data(), payload.size()));
    ArtifactDescriptor descriptor;
    descriptor.logical_name = "result.bin";
    descriptor.producer = "run-storage-example";
    descriptor.expected_size_bytes = bytes.size();
    const auto reference = runs.put_artifact(
        request.run_id, "result-final", RunArtifactKind::artifact, "result",
        bytes, descriptor, 110);
    runs.transition(request.run_id, RunStatus::completed, 120);

    const auto metadata = runs.find(request.run_id);
    const auto references = runs.references(request.run_id);
    if (!metadata || metadata->status != RunStatus::completed ||
        references.size() != 1 ||
        references.front().artifact.digest != reference.artifact.digest ||
        !runs.integrity_check(true).ok()) {
      return 1;
    }
    std::cout << "completed " << request.run_id << " references=1\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
```

安装 SDK 后，在同一目录保存以下 `CMakeLists.txt`；只请求并链接 `experiment` 组件：

```cmake
cmake_minimum_required(VERSION 3.18)
project(run_storage_example LANGUAGES CXX)
find_package(jyppx CONFIG REQUIRED COMPONENTS experiment)
add_executable(run-storage-example main.cpp)
target_compile_features(run-storage-example PRIVATE cxx_std_20)
target_link_libraries(run-storage-example PRIVATE jyppx::liuding::experiment)
```

将路径替换为匹配兼容矩阵的安装 SDK 和 LibTorch。以下为单配置生成器的构建和重放步骤；多配置生成器使用 `--config Release` 并运行 `build/Release` 下的程序，Windows 加 `.exe`。运行前确保相应动态库可被系统找到；安装后使用此示例不需要 Python。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/sdk;/path/to/libtorch"
cmake --build build --config Release --parallel 1
./build/run-storage-example ./local-runs
./build/run-storage-example ./local-runs
```

两次均输出 `completed run-001 references=1`。新的实际任务使用自己的稳定 ID 和真实时间戳；修改相同 ID 下的原始请求会被拒绝。

`put_artifact` 接受内存字节 span。流式输入使用 `FileArtifactStore::begin`，多次调用 `ArtifactUploadSession::write`，最后把同一 session 交给 `commit`；`commit/put_artifact` 的可选预期摘要用于核对生产者断言。session 属于创建它的 store 句柄，store 必须活得更久；提交或中止后 session 关闭，未提交 session 析构时尝试删除临时文件。store 句柄不可复制或移动。

独立 CAS 可直接使用 `FileArtifactStore(root, limits)` 的 `put` 或上述流式协议。`open(digest)` 核验完整内容后返回拥有资源的 `ArtifactReader`；`read(destination)` 返回实际读取字节数，EOF 或空目标返回0。`exists(digest)` 检查内容有效性，缺失或损坏返回 false，畸形摘要和部分目录访问错误会抛异常。`verify()` 报告规范对象中的损坏与意外普通文件，不执行修复。

## 生命周期、来源与重放

lineage 的五个可选非空文本字段为 `parent_run_id/source_digest/build_fingerprint/config_digest/data_manifest_digest`，由调用方提供；存储不自动收集 Git/环境、不解析引用，也不强制父 run 外键。它们不是 `ArtifactDigest`，字段名含 digest 不意味着强制 SHA-256 校验。配置和来源信息应在保存前脱敏。

```text
created -> running -> completed | failed | cancelled | interrupted
```

`start` 接受 created 或 running；created 须先进入 running 才能终止。相同状态的 `transition` 返回已保存状态，不替换活动时间。新状态转换和 heartbeat 的时间不得早于最近活动，heartbeat 仅用于 running。新附件时间不得早于创建时间，并将活动时间推进到原值与附件时间的较大者。

run ID 是 start 的幂等身份。原始字段完全相同的 start 重放返回当前状态，包括后来发生的状态/活动时间变化；修改原始字段、初始状态/时间、lineage 或标签会失败，没有通用元数据更新接口。

reference ID 是 run 内附件幂等身份；重放比较类型、角色、内容、大小、描述符和附件时间。`attach` 一批新关系要么全部提交，要么全部回滚；终态只接受原关系重放。重放不重新核验内容，需重新核验时调用 `integrity_check()`。`last/best` 是文本角色，不是可变别名指针。`find` 对缺失 run 返回 `std::nullopt`；`references` 要求 run 存在，按 reference ID 返回独立值记录。

## 文件证据与联合提交

已发布本地文件可用 `record_published_file_evidence` 保存：请求包含稳定 run/reference ID、文件路径、精确大小和 SHA-256、描述符、附件时间、事件 batch key，以及 artifact 载荷仅含 role 的遥测 envelope。它先流式复制到 CAS 并提交引用，再追加完成事件；两次提交独立。事件失败后保留原请求身份，省略 `path` 重试；`find_reference(run_id, reference_id)` 可独立检查关系，无需原文件或加载全部引用。改变摘要、大小、描述符、角色、类型或附件时间会被拒绝，事件重试也须保留原 stream/sequence/batch/envelope 字段。一次成功操作占两个运行引用及其链接字节配额。

需要运行引用与制品事件同时持久化时，使用 `record_published_file_evidence_atomic(request, stop, deadline)`，请求字段与上述接口相同。该方法持有同一个 SQLite 写事务完成流式 CAS 复制、运行引用登记、事件及其 GC 引用追加；事件冲突、配额或取消导致的失败会回滚本次新增的全部数据库关系，已确认的旧引用保留。CAS 中可能留下未引用对象，可由该 store 的 GC 清理；原文件在调用前已经发布，不属于数据库事务。成功后即使原文件移走，也可保留全部请求身份并省略 `path` 重放；联合提交失败且引用未持久化时，重试仍需原文件。每次成功操作占用两个运行引用及相应字节配额。

提交阶段出现 I/O 或同步错误时，调用方可能没有收到成功回执，而完整事务已经写入 WAL；这类错误的提交结果未知，不能据异常认定数据库未写入。停止使用发生错误的句柄，重新打开存储，以原有 run、reference、batch、stream、event、sequence、摘要和描述符重试；已提交的请求会返回原回执，未提交的请求会重新提交，两种情况都不会产生半个联合提交或重复事件。在确认引用前保留原文件及完整请求身份。

## 可恢复的最终文件发布

需要在生产者进程和输入文件丢失后继续发布输出时，使用 `FilePublicationRequest`、`prepare_file_publication` 和 `recover_file_publication`。请求包含本 run 内稳定的 `publication_id`、最终 `destination` 和原 `PublishedFileEvidenceRequest`；首次准备时 `evidence.path` 指向输入文件，描述符包含精确大小和摘要。父目录须已存在、由调用方控制，最终路径位于 run 存储之外；已有不同内容、重定向路径或身份冲突会被拒绝。

`prepare_file_publication(request, stop, deadline)` 在同一 SQLite 写事务里将输入内容和完整意图各登记为一个普通 run 引用。意图保存规范事件、原描述符、时间戳、目标路径及结果身份；成功后可删除输入文件。准备阶段不会发布最终路径或结果事件，失败时新增的两个引用一起回滚；未引用 CAS 字节仍可由协调 GC 清理。意图最多 4 MiB，内容沿用制品字节上限；最终复制按 64 KiB 分块，支持取消和截止时间。重试准备须使用原意图；已成功准备后可省略 `evidence.path`。

编码结果已在内存时，将 `evidence.path` 留空，调用 `prepare_file_publication_bytes(request, encoded.bytes, stop, deadline)`。接口借用缓冲区到返回，按 64 KiB 分块写入 CAS，无需额外的输入文件或完整副本。首次准备须匹配精确大小和 SHA-256；相同已准备意图重放可传空 span，仍核验 CAS。文件与字节入口使用相同的持久格式和恢复接口。

重启后用 `file_publications(run_id, after_id, limit)` 分页发现意图 ID。每页 1–1024 项，按 ID 升序排列，`after_id` 为排他游标；已完成的意图也保留，适合统一幂等重放。恢复入口只需要 run ID 和意图 ID，直接读取原持久化请求，不重新执行生产者：

```cpp
auto ids = runs.file_publications("run-1");
for (const auto& id : ids) {
  auto confirmed = runs.recover_file_publication("run-1", id);
  // confirmed.path and confirmed.evidence describe the complete publication.
}
```

`recover_file_publication` 持有 SQLite 写锁，核验意图和 CAS，先在未提交事务里检查并登记结果引用与事件，再将 CAS 内容写入同目录专用暂存、同步并以不覆盖方式发布最终文件，最后提交 SQLite。现有文件必须完整匹配大小和摘要；匹配重放保留文件身份及修改时间。进程在文件发布后、数据库提交前退出时，意图和内容仍由原引用保护，下一次恢复确认文件并补交数据库。提交已成功但回执丢失时，相同身份返回原事件回执。不同连接按数据库写锁串行，协调 GC 无法进入复制和提交窗口。

准备阶段占两个 run 引用；完整提交后共占四个引用，分别为内容、意图、结果和事件，各自计入链接字节配额。内容即使去重也按每个引用计费，意图和内容保留供后续恢复、GC 和备份。首次完成沿用 run 的可写生命周期，调用方应先完成输出再把 run 标记为终态；终态只允许已登记证据的幂等重放。`backup` 包含这些引用及对应 CAS，`restore_backup` 不自动写入外部文件。意图保留原绝对目标路径，恢复不会重新定位目录；只对受信任的本地存储显式执行恢复。专用 `.liuding-publication-<意图摘要>.stage` 名称保留给该协议，进程中断后的文件仅按原意图恢复，不扫描删除其他暂存。

最终发布要求本地文件系统支持同目录硬链接；Windows 暂存句柄使用关闭时删除标志，成功创建的最终硬链接独立保留。成功回执确认文件与数据库证据均已完成。原始文件可能在 SQLite 提交前被直接观察到，任意文件读取者和数据库之间仍无瞬时共同可见性；数据游标不属于这个事务，进程强杀验证也不等同断电恢复资格。`ImageOutputSink` 回调可以只执行准备，此时回调 future 只确认准备，完整输出须等待恢复回执。可运行示例见[图像文件发布](../../examples/experiment/PUBLISHED_FILE.md)。

## 持久性、格式与备份

SQLite 使用 WAL、`synchronous=FULL`、外键、full-mutex 连接及句柄内互斥锁；多个句柄/进程通过 `BEGIN IMMEDIATE` 协调写入。默认 busy timeout 为5秒，耗尽返回 `NotReady`，可按原操作身份重试。目录必须允许 SQLite 创建 WAL/共享内存文件并支持 WAL；不保证网络文件系统或远端服务语义。

当前数据库 `user_version=2`，application ID 为 `0x4a595250`。空数据库事务初始化；外来 application ID、不支持/未来版本及非空无版本数据库被拒绝。打开或使用时暴露 schema/记录错误，不将损坏存储当新库。默认 `allow_v1_migration=true` 可事务升级已识别的 v1，保留元数据/引用并添加空事件表；不伪造历史遥测、时间或指标定义，关闭迁移会显式拒绝旧格式。

`backup(destination)` 向新目录发布一致的数据库与活跃 CAS 备份；静态 `SQLiteRunStore::restore_backup(backup_root, new_root)` 核验后发布新存储，均不覆盖已有根目录。WAL 由 SQLite 管理，不能只复制运行中的 `runs.sqlite3` 作为备份。JSONL/TensorBoard/OTLP 导出是观察投影，不是可完整恢复的存储包。

CAS 提交会刷新/关闭上传流、同步临时文件、核验完整摘要/大小，并用同文件系统原子 rename 发布。POSIX 同步相关目录条目，Windows 使用文件刷新与 write-through 发布；去重前必须验证已有同摘要对象。哈希使用1 MiB有界堆分块，不将完整制品或1 MiB缓冲区放在线程栈上。API 不可变性不将文件设置为 OS 只读；外部修改由 verified open 或完整性检查发现。

## 规范事件、指标与快照

`append(run_id, batch_key, envelopes, stop, deadline)` 在产生效果前验证完整有界输入，将事件、指标定义/索引、stream 状态和回执放在一个 SQLite 事务中提交。run 必须已存在，生产者身份、原时间、关联、载荷与指标语义由不可变 foundation envelope 提供；store 分配的 `run_sequence` 只定义提交分页，不替代生产者序列、metric step 或观测时间。

完全相同的 batch 重放返回原回执；相同生产者键/规范内容为重复，内容冲突或生产者 sequence 缺口报错。run/stream 终态拒绝新事件，允许原请求重放。提交成功但 ACK 丢失时须保留并重放整批原请求；SQLite 持久接收不为外部观察者提供 exactly-once 保证。

`events()` 首页固定 `snapshot_through`，按 `run_sequence > after_sequence` 查询至该边界。`metrics()` 按 `(run_sequence, point_index)` 排序，使用显式语义指标 ID，可筛选定义版本和 step kind。仅有后续结果时返回排他 `next_after`；后续页须沿用原边界和游标，并发新增不进入原快照。时钟缺失/乱序和 step 重置不改变此顺序，API 仅追加，直接修改 SQL 不受支持。

指标查询解码原始 envelope 并核对派生索引，保留单位、聚合、比较方向、参数、完整上下文、定义摘要/版本、step 域/值、可选样本数、finite/NaN/正负无穷/missing 状态及原因，同时保留生产者、时钟、关联、注解和资源身份；不隐式重新聚合、单位换算、填补缺失值或添加读取时刻。

产品 CLI 提供 `run events/metrics/export`，完整参数与退出码见[运行遥测命令](run_telemetry_cli.md)。查询含规范十六进制源字节和 SHA-256；导出读取固定 SQLite 快照，使用有界 JSONL、原生 TensorBoard 或原生 OTLP HTTP backend。后两者按配置使用隔离 C++ worker；旧 Python sidecar 路由已退役，产品运行不需要 Python。目录、worker 配置、环境和 Collector 设置由调用方显式提供，不从事件属性推断。

## CAS 与垃圾回收

`SQLiteRunStore::put_artifact` 在数据库写锁内完成 CAS 发布与引用事务；`attach` 持同一锁核验新对象并提交整批关系。`collect_garbage` 持锁读取活跃摘要并删除未引用对象，发现外键损坏时拒绝回收。

SQLiteRunStore 根目录的 GC 必须经该 store 执行；向原始 `FileArtifactStore::collect_garbage` 传入不完整活跃集合可能删除已引用内容。独立 FileArtifactStore 不提供线程/进程协调，所有者须串行化写入、GC 和临时文件恢复。

GC 默认保留24小时内修改的未引用对象；已引用对象始终保留，宽限期0允许立即删除未引用规范对象，意外普通路径保留。GC 不删除 run/引用历史；I/O 错误前可能已有部分对象删除，数据库回滚不能恢复这些文件。

`integrity_check(true)` 核验数据库、外键和每个不同活跃制品的摘要/大小；传 false 仅检查数据库、关系、规范遥测和派生索引。报告最多64项问题，不修复数据，数据库/查询错误也可能直接抛出。`FileArtifactStore::verify()` 检查整个 CAS，包括未引用对象。

## 陈旧状态与中断恢复

打开数据库不会修改 running run，其他进程可能仍拥有它；活跃所有者发送 `heartbeat(run_id, now_ms)`。显式 `recover_interrupted(cutoff_ms, interrupted_at_ms)` 只修改最近活动时间不晚于 cutoff 的 running run，替换时间不得早于 cutoff；created、新近活跃和终态保留，重复恢复无害。调用方判断任务是否已放弃，store 不提供租约所有者或自动存活探测。

interrupted 为终态；续训使用新 run ID、`parent_run_id` 和显式 checkpoint 引用，实际加载/恢复交给训练与检查点 API。

`FileArtifactStore::recover()` 只删除修改年龄达到 `stale_temp_age`（默认24小时）的普通 `.upload-*.part` 临时文件，不恢复 run 状态或回收已提交对象。年龄是唯一放弃依据，调用前须确认对应上传已不再活跃并持有根目录外部协调，年龄0允许立即删除。SQLiteRunStore 当前没有自动恢复临时上传的接口。

## 资源、权限与错误

默认值为可配置上限，不预留磁盘空间：

| 配置 | 默认值 | 范围 |
| --- | --- | --- |
| `ArtifactStoreLimits::max_artifact_bytes` | 1 TiB | 单个对象 |
| `max_total_bytes` / `max_object_files` | 4 TiB / 1,000,000 | CAS 普通对象文件 |
| `max_root_path_bytes` | 4,096 | 规范 CAS 根路径字节数 |
| logical name / media type / producer | 512 / 256 / 512字节 | 描述符字段 |
| `SQLiteRunStoreOptions::max_runs` | 1,000,000 | run 数 |
| `max_references_per_run` / `max_run_artifact_bytes` | 16,384 / 4 TiB | 每个 run 的引用数/链接字节 |
| reference ID / role | 256 / 128字节 | 引用字段 |
| run ID / display name | 256 / 512字节 | run 字段 |
| tag count / key / value | 128 / 128字节 / 512字节 | 标签 |
| lineage reference | 1,024字节 | 每个可选文本引用 |
| `event_limits.max_events_per_run` | 1,000,000 | 规范 envelope 数 |
| `max_metric_points_per_run` / `max_event_bytes_per_run` | 10,000,000 / 1 GiB | 指标点/规范事件字节 |
| `max_page_events` / `max_page_metric_points` | 1,024 / 4,096 | 每页条数 |
| `max_page_bytes` | 16 MiB | 每页规范源字节计费 |

同一去重内容的不同引用分别占用 run 链接字节额度。CAS 总字节不含临时上传、SQLite/WAL 或导出副本；SQL 时间戳/大小须是可表示的非负有符号64位整数，计数选项也有表示范围。SQLite 元数据须为不含 ASCII 控制字符的有效 UTF-8。逻辑名、媒体类型和 producer 不是路径；媒体类型不执行内容嗅探。CAS 总量不得小于单对象上限，必要容量/文本上限须为正数。

使用具有适当权限的专用本地根目录。受管理路径检查符号链接重定向，扫描不跟随链接，根目录规范化；访问遵循调用者权限，不另提供认证、加密、凭据保存或 ACL 管理。SHA-256 用于内容身份和变更检测，不授予访问权或替代签名。

| 错误 | 结果 |
| --- | --- |
| 畸形元数据、外来/关闭 session、冲突重放、非法生命周期 | `InvalidArgument` |
| 上传、字段、数量或字节超限 | `ResourceExhausted` |
| 预期摘要或内容哈希不匹配 | `SerializationError` |
| 文件访问、读写或刷新失败 | `IoError` |
| 等待后 SQLite 仍 busy/locked | `NotReady` |
| 不支持/外来数据库身份或版本 | `CompatibilityError` |
| 其他 SQLite/后端错误 | 结构化框架 `Error`，包括 `BackendError` |

超过声明上传大小的写入返回 `ResourceExhausted`，提交时不足声明精确大小返回 `InvalidArgument`；SQL 约束/类型失败映射为 `InvalidArgument`，数据库损坏为 `SerializationError`。元数据事务失败不确认新关系；CAS 已发布内容可能成为孤儿，供后续协调 GC 清理，已确认关系不作为失败清理对象。临时文件清理也可能因文件系统错误失败；保留操作身份和受损数据以便重试、诊断。

该 API 不提供完整训练速度、进程 RSS/显存或断电恢复的通用保证；组件、平台及任务支持范围见[当前限制](../user-guide/beta_limits.md)。需要执行本地程序、持久 worker 与 typed 输出时，继续阅读[本地 C++ 执行示例](../../examples/experiment/local_execution/README.md)。
