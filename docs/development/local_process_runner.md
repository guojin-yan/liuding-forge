# 本地进程运行器

`tools/cli/local_process.hpp` 提供工具层私有的 `LocalProcessRunner`，复用 CLI sidecar 的跨平台进程监督实现。它不属于安装 SDK；调用方负责提供受信任的可执行文件和参数。

## 调用契约

- 可执行文件和工作目录必须是现存的绝对路径，参数与环境值使用不含 NUL 的 UTF-8；直接传递 argv，不经过 shell，也不通过 PATH 查找程序。
- 子进程 stdin 立即关闭；默认不继承父环境，只使用显式环境映射。`inherit_environment=true` 时才复制父环境并应用覆盖，父进程的环境和工作目录不变。
- 启动参数与环境默认合计上限 256 KiB，stdout/stderr 默认各限制 1 MiB；输出按原始字节捕获，不要求 UTF-8。超过任一累计输出限额立即终止进程树，捕获内容不超过各自上限。
- 总时限默认 1 小时，最多 7 天，包含启动、运行与输出排空；启动默认另限 10 秒，清理最多额外使用配置的宽限期（默认 2 秒）。两条输出管道同时排空，避免互相阻塞。
- 结果区分普通退出、POSIX 信号退出、取消、超时与输出超限。普通退出保留原生退出码，信号退出保留信号号；取消、超时和超限不伪造退出码。非法配置、启动或管道错误通过框架异常报告。

Windows 采用挂起启动并加入禁止 breakaway 的 kill-on-close Job Object；POSIX 使用独立进程组。成功、失败、取消及超时都清理剩余子孙进程，POSIX 直接子进程由运行器回收。主动调用 `setsid()`/`setpgid()` 逃离进程组的程序不在 POSIX 回收契约内；进程组不是沙箱。

首次停止进程树时固定清理截止，输出排空和进程回收共用这段宽限期。只有 Windows Job 中没有活动进程，或 POSIX 主进程已回收且原进程组消失，才确认清理完成；POSIX 后代由实际父进程或系统回收，宽限期结束时仍可观察到僵尸进程则保持未确认，不释放逻辑资源占用。

该原语只拥有本次同步调用的进程生命周期；持久执行由下面的 CLI 组合现有 run/SQLite/CAS。输出上限不限制子进程本身的 RSS 或 GPU 显存。

## 原生资源观测

`LocalProcessResult::resource_usage` 与持久 `local.result` 的同名对象保存退出或强制停止后的操作系统计数：`user_cpu_time_us`、`system_cpu_time_us` 使用整数微秒，`peak_resident_memory_bytes` 使用整数字节。启动前取消、启动失败或最终计数不可用时对象为 `null`；单项读取失败时该项为 `null`，实际零计数仍保留为 `0`。旧结果可缺少整个字段，查询或恢复继续返回原已提交内容，不补写测量。

| `source` | CPU 时间范围 | 峰值驻留内存范围 |
| --- | --- | --- |
| `windows_process` | `GetProcessTimes` 返回的直接子进程用户态与内核态时间 | `GetProcessMemoryInfo` 的直接子进程 `PeakWorkingSetSize`；不包含 Job 中其他进程，不使用 Job 提交量替代工作集 |
| `linux_wait4` | 回收已拥有的直接子进程时，内核 `wait4` 返回的统计，可能包含该进程已等待回收的后代 | `ru_maxrss` 从 KiB 转换为字节，遵循内核对被回收进程及其已等待后代的峰值统计；不是并发进程树 RSS 的总和 |

计数在原有清理宽限内读取，不延长截止，不改变退出码、清理确认或逻辑资源释放。未回收后代、逃离进程组的进程、运行器自身和 GPU 显存不在完整观测范围；不同 `source` 不能直接当成相同口径比较。该观测不执行 RSS/CPU/显存硬限制，也不证明训练质量或端到端性能资格。

## 持久执行 CLI

`local execute --root STORE --id JOB --spec spec.json` 冻结一次本地执行计划；`local show` 查询，`local cancel` 幂等提交取消请求，`local recover` 修复已确认结果的生命周期并继续剩余的可重试 attempt。后三者使用相同的 `--root STORE --id JOB`。同一 ID 的相同计划可重放，改变计划必须使用新 ID。已确认的成功或失败结果重放只校验冻结计划与已提交的 CAS 结果，不要求旧程序或工作目录继续存在；任何实际新启动仍核验现存路径与可执行文件 SHA-256。

可执行文件的身份校验单独限 1 GiB，使用流式 SHA-256；配方训练制品各限 256 MiB。

```json
{
  "format": "jyppx.liuding.local_run_spec",
  "schema_version": 1,
  "project_id": "demo",
  "experiment_id": "training",
  "executable": "/absolute/path/to/training-program",
  "executable_sha256": "<64位小写SHA-256>",
  "arguments": ["--config", "/absolute/path/to/config.json"],
  "working_directory": "/absolute/path/to/work",
  "environment": {"OMP_NUM_THREADS": "2"},
  "inherit_environment": false,
  "timeout_ms": 3600000,
  "max_attempts": 2,
  "retry_exit_codes": [75],
  "retry_delay_ms": 1000,
  "max_stdout_bytes": 1048576,
  "max_stderr_bytes": 1048576
}
```

Windows 将示例中的程序和工作目录改成真实绝对路径，JSON 中的反斜杠需转义。目标程序每次启动前核对 SHA-256；计划最多 64 KiB、256 个参数、128 个环境项、16 次 attempt，单次时限最多 24 小时，双流各最多捕获 8 MiB，重试间隔最多 60 秒。只有声明的非零普通退出码可重试；信号、取消、超时、超限与启动失败不自动重试。`max_attempts` 包含首次执行。`environment` 中的值原样存入本地 CAS；需要只保存变量引用时使用下一节的 `environment_refs`。

本地计划和搜索 `execution` 的显式环境目标名必须在忽略 ASCII 大小写后唯一，包括 `environment` 内部及其与 `environment_refs` 的组合。`TOKEN` 与 `token` 同时出现会在创建存储或启动任务前被拒绝；单个目标名的原有拼写与值保持，仍可覆盖继承环境中的同名变量。

### 环境变量引用

`local_run_spec` 与搜索的 `execution` 可选 `environment_refs`，将目标环境名映射到启动进程环境中的变量引用：

```json
{
  "environment_refs": {
    "SERVICE_TOKEN": {"source": "process_environment", "name": "LOCAL_SERVICE_TOKEN"}
  },
  "inherit_environment": false
}
```

引用可额外声明 `"sha256": "<64位小写SHA-256>"`，绑定来源值的 UTF-8 原始字节；不去空白、不规范化 Unicode，空字符串也可绑定。调用者提供的摘要随引用冻结，C++ 运行器在准入及每个实际 worker 启动前解析并核验；值不符时拒绝启动，错误不包含值或摘要。未提供 `sha256` 的旧引用继续只冻结来源与名称，不自动持久化解析值或其摘要。

变量缺失、无效 UTF-8、摘要不符或解析后显式 argv/环境合计超过64 KiB时拒绝启动；存在的空值保留为空字符串。首次请求在准入前失败时不会创建 STORE；已持久资源预约或 worker 端的解析失败沿用原有 launch-error 与清理确认合同。重试前的准入失败保留已确认的 attempt 前缀；恢复匹配的来源值后可继续原有有限重试。

显式环境和引用合计最多128项；引用的目标与来源名限128字节的 ASCII 字母、数字与下划线，首字符不能是数字。目标名不得与显式环境或其他引用按 ASCII 大小写冲突，目标与来源均不得使用运行器保留前缀；不展开 shell、文件路径或搜索占位符。已终态任务的查询、恢复及重复执行只读重放已确认结果，不要求来源变量仍存在或继续匹配摘要；实际新 attempt 使用当次调用者传给 worker 的值，并验证已声明的摘要。

这是本机进程环境引用；显式摘要只约束对应引用的值，不认证完整环境快照或 secret manager。摘要会公开该值的内容指纹，敏感值是否绑定由调用者决定。`inherit_environment=false` 可避免整份父环境进入训练程序；训练程序若自行把变量写到 stdout/stderr 或产物，这些内容仍按原有合同保存，不自动脱敏。其他训练输入的可复现性仍须由应用自己的输入身份合同保证。

每次 attempt 是独立 child run，关联 parent job、冻结计划和原始输出。`JYPPX_LIUDING_LOCAL_` 前缀由运行器保留，不区分 ASCII 大小写；显式环境覆盖使用该前缀时，新任务在创建 Store 和启动进程前以 `invalid_argument` 拒绝。程序可从 `JYPPX_LIUDING_LOCAL_JOB_ID`、`JYPPX_LIUDING_LOCAL_ATTEMPT_ID`、`JYPPX_LIUDING_LOCAL_ATTEMPT_INDEX` 读取身份；工作目录由调用者提供并在重试间复用，应用负责按 attempt ID 隔离自己的可变文件。运行器不猜测训练程序的 checkpoint/resume 参数，也不解析退出码为训练完成或暂停语义。

同一 job 用独立 dispatch/worker 文件锁协调；锁文件保留，不能在运行时删除或替换。worker 在原生进程树清理完成后，使用 `put_artifacts()` 一次事务提交 stdout、stderr、result 三个引用，再推进 attempt/job 状态。调用者消失但 worker 仍在运行时，恢复返回 `active`，不会重复启动；已提交结果可跨进程修复并幂等重放。取消仅提交请求，已完成的成功结果优先；调用者应随后查询或等待执行命令确认进程树已清理。

已发起的 attempt 若缺少确认结果，即使文件锁已释放，也返回 `recovery_required`（退出码 7），不会单凭 PID、心跳或超时自动重试。强杀 worker 后，POSIX 子孙进程可能仍在运行；该状态必须由应用/操作者检查并处理，当前入口不提供强制解锁重启。正常取消与单次超时由 worker 的原生运行器清理子树后提交结果。[学习率搜索示例](../../examples/experiment/README.md)演示已实现的有界 grid/random、best trial 和实验备份。

未终态 job 即使已有完整结果，只要原生 `process_tree_cleanup_confirmed` 为 `false` 或缺失，就返回 `resource_cleanup_unconfirmed`（退出码 7），保留当前 attempt、结果和引用，不创建新 attempt 或将 job 推进至成功；该围栏适用于未声明资源和默认 retry 的任务。历史已终态 job 继续按原确认结果只读重放，不能据此推断其原生清理证明或资源可释放。

## 操作系统内存限制

本地计划和搜索的 `execution` 可显式增加 `os_memory_limit`。`bytes` 必须是正整数 MiB 的整倍数；Windows 使用 `{ "bytes": 1073741824 }`，Linux 另外要求 `{ "bytes": 1073741824, "cgroup_parent": "/sys/fs/cgroup/已委派的父组" }`。该限制独立于同 STORE 的逻辑预约，应用于本次原生程序及其后代；worker 和调用者不计入这份额度。

Windows 在恢复挂起的程序前设置并回读 Job Object 的整组提交量上限，禁止 breakaway。Linux 仅在调用者提供的 cgroup v2 父组内创建本次专属子组，回读 `memory.max`、`memory.swap.max=0` 和 `memory.oom.group=1`，然后让子进程在 exec 前加入；父组必须已经委派且启用 memory controller，内核必须提供 `cgroup.kill`。运行器不启用或修改祖先控制器，也不自动提权。后端或权限不可用时启动失败，不退化为逻辑额度或定时 RSS 采样。

Linux 调用者须位于同一委派子树，拥有目标 `cgroup.procs` 及当前组与目标组最近共同祖先的 `cgroup.procs` 写权限。仅能在父组创建目录并不证明可迁入进程；例如 SSH session 与 systemd 用户服务分属不同分支时，内核可能在 exec 前拒绝迁组。应由管理员预先建立委派布局，并从相应服务或 scope 内启动 worker。

结果中的 `memory_boundary` 记录实际启用的 `source`、`limit_bytes`、可读时的 `peak_bytes` 和 Linux 的 `oom_kills`，与 `resource_usage` 分开保存；旧结果可以缺少此字段，实际已启动的限额任务必须有对应收据。Windows 原生峰值计数可能高于配置上限，收据保留内核原值，不能把该峰值等同于成功提交的最大字节数。Linux 的 OOM 保留真实信号，Windows 分配拒绝保留程序真实退出码，不能仅凭非零退出就声称发生 OOM。取消和清理也对 Linux 专属组调用 `cgroup.kill`，清理确认还要求该组为空。

两个后端的计费口径不同：Windows 限制 Job 提交量，Linux 限制内核 cgroup 内存计费（包含被计费的匿名页、文件缓存等），共享页和进入组前已计到其他组的页面遵循内核归属。它们都不等于精确的整树 RSS，也不限制 GPU 显存。受信任程序及其父组配置须保持该边界；同身份程序主动修改 Job/cgroup 策略或自行迁组不属于沙箱保证。完整 RSS/VRAM、worker 被强杀后的外部确认及整体验收仍需各自证据。

## CPU/GPU 逻辑资源准入

`local_run_spec` 与搜索的 `execution` 可增加下面的 `resources`。首次准入会将同一机器、同一 STORE 的容量固定到 run/CAS；后续请求必须原样使用该容量。`gpu0` 是调用者约定的设备身份，运行器不探测硬件、不设置 CUDA 可见设备；训练程序负责将该身份映射到实际设备。

```json
{
"resources": {
  "capacity": {"cpu_slots": 8, "memory_bytes": 17179869184, "gpus": {"gpu0": 8589934592}},
  "request": {"cpu_slots": 4, "memory_bytes": 4294967296, "gpu": {"id": "gpu0", "memory_bytes": 6442450944}}
}
}
```

CPU 任务将 `request.gpu` 写为 `null`，容量中的 `gpus` 可以为 `{}`。CPU slots、主存与显存都是声明预算；请求必须为正且不超过容量。GPU 每次只接受一个未释放的 trial，即使其显存预算尚有余量；grid/random 保持串行。程序可从 `JYPPX_LIUDING_LOCAL_CPU_SLOTS`、`JYPPX_LIUDING_LOCAL_MEMORY_BYTES`、`JYPPX_LIUDING_LOCAL_GPU_ID`、`JYPPX_LIUDING_LOCAL_GPU_MEMORY_BYTES` 读取请求，但这些值不会限制线程数、RSS 或实际显存分配。

跨 job 的检查与准入用同一非继承文件锁串行化；占用先于 attempt/intent 持久化。预算不足返回 `resource_unavailable`（退出码 7），尚未创建 attempt 或启动程序；锁繁忙返回 `resource_busy`，可稍后用 `local recover` 继续。即使没有声明 `resources`、容量池尚未启用，同一 STORE 的新增 attempt 也参与该短临界区，避免首次启用容量时遗漏同时启动的任务；竞争失败不会创建 attempt 或启动程序，也不会静默等待或自动重试。恢复复用原占用身份，父调用者崩溃或未知 attempt 不释放容量。启用容量后，未声明资源的新增 attempt 返回 `resource_request_required`；启用前仍未确认的旧 attempt 返回 `resource_unmanaged_attempt`。旧结果没有原生清理确认时，请使用独立 STORE，不能据终态或空锁推断资源可释放。

释放记录同时绑定 lease、intent、worker、result 与双流摘要，并要求原生清理确认。Windows 要求 leader 已退出且 Job 活跃进程为零；Ubuntu 要求直接子进程已回收且原进程组不存在。组内尚未回收的僵尸也会保守保留占用；清理未确认返回 `resource_cleanup_unconfirmed`（退出码 7），不会强制解锁、重启或扩大容量。跨机器共享 STORE、其他 STORE、外部程序及逃离 POSIX 原进程组的进程不在该准入范围。系统级 RSS/显存硬限额与真实 GPU 资源资格仍待完成。

## 训练输入与显式恢复

`local_run_spec` 可选 `training` 对象，要求 `model_id`、`configuration`、`data_manifest` 和 `resume`。两个输入使用 `{ "run_id": "SOURCE", "reference_id": "REF", "sha256": "...", "size_bytes": 123 }` 引用同一存储中已提交的 CAS 内容；来源 run 必须同项目且为 completed/failed/cancelled。本地 attempt 来源还须通过既有的冻结计划、结果与双流引用检查，且结果必须明确包含 `process_tree_cleanup_confirmed: true`，无资源计量的旧任务也适用。历史终态结果仍可只读重放，但清理确认缺失或为 false 时，其配置、数据与 checkpoint 均不能作为新训练或搜索的输入。配置和数据清单各限 1 MiB；`resume` 为 `null` 或 `{ "checkpoint": { ...同样的引用字段... }, "restore_policy": "exact_only" }`，checkpoint 引用必须为 checkpoint 类型、最多 1 GiB。

运行器在创建 attempt 前验证并批量绑定 `local.training.configuration`、`local.training.data_manifest`、可选 `local.training.resume`，拥有执行权的 worker 启动程序前再次复核内容。训练程序通过 `JYPPX_LIUDING_LOCAL_STORE_ROOT`、`JYPPX_LIUDING_LOCAL_SPEC_SHA256` 及已有 job/attempt 环境读取冻结计划；输入绑定采用流式 CAS 验证，不解码模型张量。模型标识和输入摘要用于固定身份，实际模型支持、数据含义、checkpoint 完整性与恢复由 C++ 训练程序检查。示例使用原生 checkpoint coordinator 恢复模型、SGD momentum 和更新计数，再在新 job 中继续训练。

默认 retry 重用最初的冻结输入，在新的 attempt 中重新开始该请求。需要从已确认失败的 checkpoint 续训时，显式启用下一节的策略。已有未确认 attempt 仍返回 `recovery_required`；存在 checkpoint 不能证明旧进程树已经清理。系统级资源硬限制、未知 attempt 的外部确认和真实任务恢复联验仍开放；输入限额不代表 RSS/显存上限或训练提速。

## 训练输出确认

`training.outputs` 可选；省略时保持原执行语义。启用后声明1–64项 `{ "reference_id": "checkpoint.last", "kind": "checkpoint", "role": "last", "max_size_bytes": 65536 }`，kind仅为`artifact`或`checkpoint`。引用ID须唯一且不以`local.`开头，role为1–128字节非空文本；各项及上限合计最多1 GiB，完整确认收据最多64 KiB。

worker启动前观察本attempt的全部声明引用不存在，并将该事实写入不可变owner；启动前再次检查。仅原生正常退出0且进程树清理确认后，核对同attempt的引用、kind、role、真实文件大小和流式CAS摘要。`local.training.outputs`收据绑定冻结spec、job/attempt、实际有效输入、intent/owner及原生结果摘要，与stdout、stderr和result原子提交。结果保留真实原生退出事实，另用`training_outputs_confirmed`表示输出确认；缺失或不符使job失败，不自动重试。

输出核验使用最多30秒的协作预算，同时受原绝对deadline约束；取消、截止或存储检查失败保留原生exited/0并记录确认失败，最终完成前的job取消返回130。清理false或缺失始终先返回7，不读取或补造输出收据；这也适用于启用本合同的终态重放。已提交收据的重放使用独立30秒只读核验预算，不重启训练、不补写引用；后来补齐缺失输出不会改变原失败。不可返回的底层文件系统调用不受协作预算强制中断。

该确认只证明启动前的空引用观察和清理后的STORE不可变绑定/CAS内容，不能证明字节由哪个进程创建，也不能排除其他有写权限的STORE进程抢先发布。checkpoint类型不等于完整可恢复状态，训练质量与模型、优化器、采样器等状态仍由应用验收。可运行示例见[输出确认](../../examples/experiment/README.md#确认训练输出)。

## 已确认失败的 checkpoint 续训

`training.retry_resume` 可选。它要求 `max_attempts > 1`、非空 `retry_exit_codes`，并明确指定每个失败 attempt 必须发布的 checkpoint 引用：

```json
{
  "retry_resume": {
    "checkpoint_reference_id": "checkpoint.last",
    "restore_policy": "exact_only"
  }
}
```

运行器只从上一个已确认、普通非零退出且属于 `retry_exit_codes` 的 attempt 选择该引用；结果、双流、原始 intent、worker 归属和 `process_tree_cleanup_confirmed: true` 都须通过检查。checkpoint 必须属于该 attempt、类型为 checkpoint、大小不超过 1 GiB，且 CAS 内容有效。缺少引用、类型或来源错误、额度越界、清理未确认时停止推进；不会改为 fresh 重训、选择其他 checkpoint 或推断旧进程已结束。取消、超时、信号和启动失败继续遵守原有不重试规则。

启用后，每个 attempt 启动前在 job 保存 `local.training.inputs.N` 清单，固定 job/attempt/index、原始 spec 摘要、上一结果摘要、模型、原配置、原数据和本次 checkpoint 完整身份；原始 `local.spec` 保持不变。配置、数据和可选恢复内容批量绑定为 `local.training.configuration.N`、`local.training.data_manifest.N`、`local.training.resume.N`。清单已提交但尚未创建 attempt 时，恢复只修复相同内容的绑定；attempt 已存在而结果未确认时仍隔离。资源预约继续先于 attempt/intent，前一个预约仍须满足原清理与结果合同才能释放。

worker 在 attempt 上绑定 `local.training.inputs`，把其摘要写入 intent/result，并通过 `JYPPX_LIUDING_LOCAL_TRAINING_SHA256` 告知 C++ 训练程序。确认结果、终态重放和跨 job 释放资源时，运行器仍在 64 KiB 上限内读取并校验该清单的 CAS、规范内容和有效输入身份；引用摘要相同不能替代内容校验，缺失或损坏会停止推进。应用应按该摘要读取 attempt 清单，按 `.N` 的 job 引用加载输入，再用自己的 checkpoint coordinator 执行 `exact_only` 恢复。运行器只做身份、CAS 和生命周期校验；把任意字节标为 checkpoint 不能替代应用对模型、优化器、RNG 等完整状态的检查。[原生学习例](../../examples/experiment/README.md#失败后自动从-checkpoint-续训)演示 20 步 checkpoint、声明退出码和后续 40 步原生 SGD。

ImageFolder ResNet18、`paired-affine` 多输入输出和迁移学习配方已提供原生接线：`recipe local-spec` 封存实际配置/数据并生成三项 typed 输出的计划；`recipe local-run` 只在拥有运行权的 attempt 内执行，完成后流式发布完整 checkpoint、权重和结果。可选 `--pause-after` 用声明的普通退出码 75 接入本节既有 `retry_resume`，恢复前核对保留 workspace 的完整 checkpoint 身份。运行步骤和范围见[图像配方教程](../tutorials/image_classification_recipe.md#8-交给本地执行器并确认训练产物)；多输入输出步骤见[多输入输出教程](../tutorials/multi_io_recipe.md)；迁移学习的权重描述冻结及步骤见[迁移学习教程](../tutorials/transfer_learning_recipe.md)，当前接线验收使用CPU冻结ResNet50小任务，完整迁移设备/任务矩阵单独验收。声明配方的 workspace 丢失后可按原输入身份重建；显式环境引用及 Windows Job/Linux 委派 cgroup v2 内存限额已接通。完整环境认证及其他任务/设备的资源矩阵仍待验收。

训练超时保留 `timed_out` 与空原生退出码；只有进程树清理和结果均已确认，才释放逻辑预约。已有 `latest.jyck` 不会把超时改为可重试退出。若持久 owner worker 在提交结果前被强杀，attempt 保持 `recovery_required`，即使后来已无 OS 进程，`recover`、`execute` 或提交取消请求也不能补造结果或释放预约。Windows 的 kill-on-close Job 会随 worker 退出回收其 trainer；Linux worker 为同步启动的直接子进程启用父线程死亡信号，使受信任、同凭据执行的原生 trainer 随之终止。Linux 上 dispatch 调用者退出时，持久 worker 仍独立运行；该保护不覆盖任意 fork 孙进程、特权或凭据切换，以及主动清除信号的程序，也不能代替结果与清理确认。

## 显式放弃未知 attempt

缺少原生结果时，`local show` 或 `recover` 在无活跃 worker 的 attempt 中提供 `abandonment_request`。操作者须先独立确认原程序及其整个进程树已经清理，再复制该对象到确认文件，保持任务、attempt、spec、intent、owner 和资源预约摘要原样，填写 `confirmed_by`、`confirmed_at_unix_ms`、`reason`、`evidence`，并将 `external_process_tree_cleanup_confirmed` 显式设为 `true`。时间为 Unix 毫秒；证据文字应说明实际检查的环境、进程范围和清理结果。

```sh
liuding-forge local abandon --root local-runs --id trial-001 --confirmation abandonment.json --format json
```

该操作记录操作者的外部确认，工具不根据空闲锁、PID 消失或等待时间推定进程树已清理。活跃 dispatch/worker、摘要漂移、空证据、已有原生结果或其他终态会拒绝操作。确认文件不接受成功退出或已完成更新数声明：`native_outcome` 与 `applied_updates` 均保持 `unknown`，任务和 attempt 最终为 `failed`，命令返回退出码 1，`result` 仍为 `null`。资源释放收据明确引用外部确认，不补造原生 stdout/stderr、退出码、指标或输出成功收据。

相同确认可幂等重放；确认已提交但资源释放或生命周期提交中断时，`local recover` 只补齐这些提交，不启动新 attempt。放弃后同任务不再 retry，也不能成为已确认训练输出或 checkpoint 的来源。搜索会将该 trial 保留为失败，按原有 trial/失败预算继续，不能使用此前发布的训练指标将它升级为 best。需要再次训练时提交新任务，并使用已有独立确认的输入和 checkpoint。

## 搜索的训练输入

有界 grid/random 的 `execution.training` 使用相同字段，但 `configuration` 必须写成字符串 `"${trial_configuration}"`。搜索将每次实际配置写入 CAS，再生成确定 ID 的 `TRIAL.inputs` run，附上父搜索、索引和计划摘要并封存；该 run 的 `completed` 只表示输入已提交。配置输入的创建、绑定和封存可按原身份幂等恢复；同 ID 不同内容会被拒绝。随后 native attempt 复用 typed local 的终态来源、CAS、原生归属和取消检查。

整次搜索固定 `model_id`、`data_manifest` 和可选初始 `resume`；checkpoint 采用 `exact_only`。默认每个 trial 执行一次；显式配置 `training.retry_resume`、有界 `max_attempts > 1`、非空 `retry_exit_codes` 和 `training.outputs` 后，可以复用本地执行器的已确认 checkpoint 自动续训。`training.outputs` 除额外的 `max_size_bytes` 外须与搜索 `outputs` 按顺序相同，大小沿用合计 1 GiB 上限；成功退出后必须确认最终 attempt 的全部输出，取消、超时、信号、未知结果和未确认清理继续停止推进。

每个 trial 的 `training_inputs` 保留原始请求；启用自动续训时，`effective_training_inputs` 和 `confirmed_training_attempt` 另存最后一次已确认 attempt 的有效恢复输入、原生结果与输出收据摘要。训练程序从本地 `.N` 引用读取有效输入，仅在最终完成时发布外层训练 run，将最后 attempt ID 写入 `liuding.training.owner_attempt`，将模型标识写入 `liuding.training.model_id`、数据摘要写入 lineage，并绑定 `search.training.configuration`、`search.training.data_manifest` 和可选 `search.training.resume`。只有外层输入与最终有效输入相同、输出与最终收据相同且指标符合搜索目标时才参与 best；早期 attempt 的输出不能替代最终输出。参考[原生训练搜索教程](../../examples/experiment/README.md#训练输入与搜索衔接)中的 C++ 接线。

查询、恢复和重复执行会复核每个可选 trial 的原生执行终态、训练来源、实际已提交指标、输入输出引用及 CAS 内容；结果缓存、最终摘要和 best 彼此一致，仍不能替代这些来源。损坏的产物或不匹配的来源使命令失败，修复原证据后可只读重放；复核不要求旧可执行文件或工作目录存在，也不启动新的 attempt。已确认的不合格 trial 保留原失败结果，不因后来补齐产物而升级为候选。

## 验证

使用启用了测试与 CLI 的兼容构建目录，按顺序运行：

```sh
cmake --build out/build/config-windows --config Release --target jyppx_test_local_process jyppx_test_sidecar_transport jyppx_cli_recipe_gate --parallel 2
ctest --test-dir out/build/config-windows -C Release --output-on-failure --no-tests=error -R "^(LocalProcessTest|SidecarProcessTest)\.|^Cli_local_attempt$"
```

Linux 使用对应构建目录并省略多配置参数。原生测试覆盖进程生命周期和 sidecar 回归；独立 CLI 合同覆盖有限重试、输出 CAS、活跃归属、并发取消、调用者强杀和提交边界恢复。`Cli_local_recipe_lifecycle` 验证真实训练取消；`Cli_local_recipe_failure_lifecycle` 在完整 checkpoint 后挂起真实 trainer，验证原生超时和持久 worker 强杀；Linux 的 `Cli_local_recipe_parent_death` 额外要求夹具动手清理之前 trainer 已退出、未知预约仍被围栏。Python 仅注入故障与独立检查，各自的 `_CUDA` 合同需显式启用 CLI CUDA 测试，按顺序执行。强杀后的结果按平台原生生命周期判断，不把保守隔离误记为自动训练恢复。

Linux CPU 的 `Cli_local_recipe_memory` 在 2 GiB cgroup 限制下复用完整图像训练、精确续训、搜索、checkpoint 后取消和超时合同，并以独立 128 MiB 任务验证真实内核 OOM、失败结果、逻辑预约释放与无重训重放。调用者须先准备空的委派父组、启用其 memory 子控制器，并将绝对路径设为 `JYPPX_LIUDING_TEST_CGROUP_PARENT`；测试不修改祖先控制器，没有委派时明确跳过。可直接运行 `tests/cli/test_local_recipe_memory.py --cli <原生CLI> --evidence-dir <新证据目录> --temp-parent <临时目录>` 保留收据；OOM 发生前的更新数未知，不把它计作完成训练，Windows/CUDA 与精确整树 RSS/显存仍需各自验收。

### 多输入输出配方搜索

`paired-affine` 配方可显式启用闭合的 `recipe_bridge`，将已有 `recipe local-run`、typed 三输出及 `exact_only` retry 接入有界 grid/random。固定模板中的 `${config}` 指向本 trial 的有效配置，`${recipe_directory}` 指向尚不存在的 `TRIAL/recipe` 子目录；不可拼接模板片段或复用已存在的 trial 根作为训练目录。真实 resolver 必须确认有效配置与冻结配置完全一致、实际数据与冻结清单一致，才准入训练。

此模式的 `seed_policy` 为 `configuration`：配方使用有效配置中的 `seed` 训练，搜索派生的 `trial_seed` 只标识调度顺序，两者分别保留。完成态要求本地任务清理确认、完整三输出收据、有效 exact 恢复输入、原生 completed run 及其已提交的真实 accuracy；仅退出码 0 不足以选优。冻结桥接清单后，外层训练记录依次创建、原子绑定引用、提交相同指标，最后完成；整个流程跨多个 API，可恢复相同清单的部分发布，额外指标、部分引用或其他冲突会被拒绝。终态查询与实验导出依赖 CAS，不要求原生训练目录继续存在；实际续训遵守下一节的工作目录恢复合同。

配方桥接仍使用上述固定模板和至多一次 exact continuation；普通 typed 搜索的显式续训接线遵循前文的有效输入与最终输出确认合同。可运行配置见[配方搜索示例](../../examples/experiment/README.md#多输入输出配方搜索)。

### 配方工作目录丢失后的恢复

原生配方在本地暂停并提交 `checkpoint.last` 时，同时发布 `recipe.workspace.snapshot`，将该 checkpoint、构建来源、配置、数据清单和完整内层 run-store 备份绑定到来源 attempt。快照元数据最多128文件、合计32 MiB，manifest最多64 KiB；checkpoint沿用独立的256 MiB上限。备份和恢复使用原生SQLite/CAS校验，文件按64 KiB块流式复制，不将模型和备份整份装入内存。

`local recover` 接续已确认的75退出码时，若原工作目录已丢失，会从冻结 CAS 重建原目录并持有目录锁，保留原run身份、已有引用及训练记录，然后执行完整checkpoint的`exact_only`恢复。现存目录继续按原文件及checkpoint身份检查，自动恢复不会覆盖或修补它。外部数据、预训练权重和显式初态文件仍须存在并通过原身份检查；旧attempt没有工作目录快照时仍须保留原目录。快照缺失、来源/checkpoint/build不符、非法路径或CAS损坏会失败，不能回退为fresh训练。重建中发生存储错误时保留已创建目录供检查，可用原checkpoint和新计划选择新的目录；未知attempt仍受原结果、清理和资源预约围栏约束。

### 图像分类配方搜索

ImageFolder / ResNet18 配方使用相同的显式桥接合同，声明 `recipe=jyppx.liuding.imagefolder.resnet18/v1` 和 `seed_policy=configuration`。类别数与完整冻结数据清单一致，目标必须是原生验证集提交的 `classification.accuracy`；完整状态文件、模型、检查点和数据身份也须一致。可按固定模板暂停一次并从已确认的完整检查点精确继续，完成后的查询与发布恢复使用冻结 CAS。CPU 的 `train_cumulative_once` BatchNorm 校准还须证明完整遍历一次训练集、优化器计数不变，最终模型摘要与校准报告一致。配置与运行步骤见[配方搜索教程](../tutorials/paired_native_search.md#使用-imagefolder--resnet18-分类配方)。
