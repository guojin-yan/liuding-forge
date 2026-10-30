# 本地进程运行器

`tools/cli/local_process.hpp` 提供工具层私有的 `LocalProcessRunner`，复用 CLI sidecar 的跨平台进程监督实现。它不属于安装 SDK；调用方负责提供受信任的可执行文件和参数。

## 调用契约

- 可执行文件和工作目录必须是现存的绝对路径，参数与环境值使用不含 NUL 的 UTF-8；直接传递 argv，不经过 shell，也不通过 PATH 查找程序。
- 子进程 stdin 立即关闭；默认不继承父环境，只使用显式环境映射。`inherit_environment=true` 时才复制父环境并应用覆盖，父进程的环境和工作目录不变。
- 启动参数与环境默认合计上限 256 KiB，stdout/stderr 默认各限制 1 MiB；输出按原始字节捕获，不要求 UTF-8。超过任一累计输出限额立即终止进程树，捕获内容不超过各自上限。
- 总时限默认 1 小时，最多 7 天，包含启动、运行与输出排空；启动默认另限 10 秒，清理最多额外使用配置的宽限期（默认 2 秒）。两条输出管道同时排空，避免互相阻塞。
- 结果区分普通退出、POSIX 信号退出、取消、超时与输出超限。普通退出保留原生退出码，信号退出保留信号号；取消、超时和超限不伪造退出码。非法配置、启动或管道错误通过框架异常报告。

Windows 采用挂起启动并加入禁止 breakaway 的 kill-on-close Job Object；POSIX 使用独立进程组。成功、失败、取消及超时都清理剩余子孙进程，POSIX 直接子进程由运行器回收。主动调用 `setsid()`/`setpgid()` 逃离进程组的程序不在 POSIX 回收契约内；进程组不是沙箱。

该原语只拥有本次同步调用的进程生命周期；持久执行由下面的 CLI 组合现有 run/SQLite/CAS。输出上限不限制子进程本身的 RSS 或 GPU 显存。

## 持久执行 CLI

`local execute --root STORE --id JOB --spec spec.json` 冻结一次本地执行计划；`local show` 查询，`local cancel` 幂等提交取消请求，`local recover` 修复已确认结果的生命周期并继续剩余的可重试 attempt。后三者使用相同的 `--root STORE --id JOB`。同一 ID 的相同计划可重放，改变计划必须使用新 ID。

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

Windows 将示例中的程序和工作目录改成真实绝对路径，JSON 中的反斜杠需转义。目标程序每次启动前核对 SHA-256；计划最多 64 KiB、256 个参数、128 个环境覆盖、16 次 attempt，单次时限最多 24 小时，双流各最多捕获 8 MiB，重试间隔最多 60 秒。只有声明的非零普通退出码可重试；信号、取消、超时、超限与启动失败不自动重试。`max_attempts` 包含首次执行。环境覆盖原样存入本地 CAS；此入口尚无 secret-reference 解析或环境快照认证。

每次 attempt 是独立 child run，关联 parent job、冻结计划和原始输出。程序可从 `JYPPX_LIUDING_LOCAL_JOB_ID`、`JYPPX_LIUDING_LOCAL_ATTEMPT_ID`、`JYPPX_LIUDING_LOCAL_ATTEMPT_INDEX` 读取身份；工作目录由调用者提供并在重试间复用，应用负责按 attempt ID 隔离自己的可变文件。运行器不猜测训练程序的 checkpoint/resume 参数，也不解析退出码为训练完成或暂停语义。

同一 job 用独立 dispatch/worker 文件锁协调；锁文件保留，不能在运行时删除或替换。worker 在原生进程树清理完成后，使用 `put_artifacts()` 一次事务提交 stdout、stderr、result 三个引用，再推进 attempt/job 状态。调用者消失但 worker 仍在运行时，恢复返回 `active`，不会重复启动；已提交结果可跨进程修复并幂等重放。取消仅提交请求，已完成的成功结果优先；调用者应随后查询或等待执行命令确认进程树已清理。

已发起的 attempt 若缺少确认结果，即使文件锁已释放，也返回 `recovery_required`（退出码 7），不会单凭 PID、心跳或超时自动重试。强杀 worker 后，POSIX 子孙进程可能仍在运行；该状态必须由应用/操作者检查并处理，当前入口不提供强制解锁重启。正常取消与单次超时由 worker 的原生运行器清理子树后提交结果。[学习率搜索示例](../../examples/experiment/README.md)演示已实现的有界 grid/random、best trial 和实验备份。

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

`local_run_spec` 可选 `training` 对象，要求 `model_id`、`configuration`、`data_manifest` 和 `resume`。两个输入使用 `{ "run_id": "SOURCE", "reference_id": "REF", "sha256": "...", "size_bytes": 123 }` 引用同一存储中已提交的 CAS 内容；来源 run 必须同项目且为 completed/failed/cancelled。本地 attempt 来源还须通过既有的冻结计划、结果与双流引用检查，不能仅靠修改状态来证明已清理。配置和数据清单各限 1 MiB；`resume` 为 `null` 或 `{ "checkpoint": { ...同样的引用字段... }, "restore_policy": "exact_only" }`，checkpoint 引用必须为 checkpoint 类型、最多 1 GiB。

运行器在创建 attempt 前验证并批量绑定 `local.training.configuration`、`local.training.data_manifest`、可选 `local.training.resume`，拥有执行权的 worker 启动程序前再次复核内容。训练程序通过 `JYPPX_LIUDING_LOCAL_STORE_ROOT`、`JYPPX_LIUDING_LOCAL_SPEC_SHA256` 及已有 job/attempt 环境读取冻结计划；输入绑定采用流式 CAS 验证，不解码模型张量。模型标识和输入摘要用于固定身份，实际模型支持、数据含义、checkpoint 完整性与恢复由 C++ 训练程序检查。示例使用原生 checkpoint coordinator 恢复模型、SGD momentum 和更新计数，再在新 job 中继续训练。

默认 retry 重用最初的冻结输入，在新的 attempt 中重新开始该请求。需要从已确认失败的 checkpoint 续训时，显式启用下一节的策略。已有未确认 attempt 仍返回 `recovery_required`；存在 checkpoint 不能证明旧进程树已经清理。完整输出 schema、系统级资源硬限制、未知 attempt 的外部确认和真实任务恢复联验仍开放；输入限额不代表 RSS/显存上限或训练提速。

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

## 搜索的训练输入

有界 grid/random 的 `execution.training` 使用相同字段，但 `configuration` 必须写成字符串 `"${trial_configuration}"`。搜索将每次实际配置写入 CAS，再生成确定 ID 的 `TRIAL.inputs` run，附上父搜索、索引和计划摘要并封存；该 run 的 `completed` 只表示输入已提交。配置输入的创建、绑定和封存可按原身份幂等恢复；同 ID 不同内容会被拒绝。随后 native attempt 复用 typed local 的终态来源、CAS、原生归属和取消检查。

整次搜索固定 `model_id`、`data_manifest` 和可选 `resume`；checkpoint 必须显式指定且采用 `exact_only`，不会从失败 attempt 自动选择。每个 trial 结果的 `training_inputs` 包含实际配置、数据和恢复点的来源与摘要。训练程序须将模型标识写入 `liuding.training.model_id` tag、数据摘要写入 lineage，并将对应输入绑定为 `search.training.configuration`、`search.training.data_manifest` 和可选 `search.training.resume`；缺少或不匹配的结果不参与 best 选择。参考[原生训练搜索教程](../../examples/experiment/README.md#训练输入与搜索衔接)中的 C++ 接线。

## 验证

使用启用了测试与 CLI 的兼容构建目录，按顺序运行：

```sh
cmake --build out/build/config-windows --config Release --target jyppx_test_local_process jyppx_test_sidecar_transport jyppx_cli_recipe_gate --parallel 2
ctest --test-dir out/build/config-windows -C Release --output-on-failure --no-tests=error -R "^(LocalProcessTest|SidecarProcessTest)\.|^Cli_local_attempt$"
```

Linux 使用对应构建目录并省略多配置参数。原生测试覆盖进程生命周期和 sidecar 回归；独立 CLI 合同覆盖有限重试、输出 CAS、活跃归属、并发取消、调用者强杀和提交边界恢复。强杀后的结果按平台原生生命周期判断，不把保守隔离误记为自动训练恢复。
