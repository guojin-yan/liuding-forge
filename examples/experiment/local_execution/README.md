# 在 C++ 中执行本地原生程序

[main.cpp](main.cpp) 直接链接 `jyppx::liuding::experiment`，通过安装后的公共头启动本机程序。示例复用自己的可执行文件作为工作进程，验证字面量参数、独立环境、环境摘要、CPU放置、非零退出码、双流输出、取消、超时回收和两次JSON行交互，不依赖Python、命令行工具或shell。

在兼容构建目录启用 `experiment` 和 examples 后构建、运行：

```sh
cmake --build out/build/YOUR_PROFILE --target jyppx_liuding_local_execution_example --parallel 1
out/build/YOUR_PROFILE/examples/liuding-forge-example-local-execution
```

Windows 使用相应配置子目录及 `.exe`。成功时输出 `passed: true` 的JSON；示例设置的150毫秒超时用于演示工作进程回收，不是性能承诺。若运行库需要搜索路径，应在启动示例前设置好。

用户程序使用 `<jyppx/liuding/experiment/local_execution/process.hpp>` 中的 `LocalProcessOptions` 和 `LocalProcessRunner`，无需解析CLI输出。程序路径和工作目录必须是绝对路径；标准输入为EOF，参数直接传给程序，stdout/stderr分别设置累计字节上限。普通非零退出和POSIX信号保持原生语义；取消、超时和输出超限为独立状态，返回结果不再拥有运行中的子进程。清理证明、直接进程RSS和可选OS内存计数分开报告。

`inherit_environment` 默认关闭；显式环境只影响子进程。`local_process_environment_sha256()` 可绑定本平台完整有效环境，实际启动使用同一份已核对快照；调用者需要的动态库搜索路径也须包含在环境中。`cpu_affinity` 使用调用进程允许的CPU编号，启动用户代码前设置并严格回读；它是放置约束，不是CPU时间配额。OS内存限制和进程树边界见[资源合同](../../../docs/development/local_process_runner.md#操作系统内存限制)。

需要持久JSON行交互时，使用独立头 `<jyppx/liuding/experiment/local_execution/json_lines.hpp>` 和 `JsonLineProcessTransport`。工作进程按每请求一行返回一行；传输检查UTF-8、帧长度、截止和额外输出，首次协议或I/O失败后对象不可复用。对象在同一串行所有者下使用，销毁或 `terminate()` 回收原生进程。

上述单次进程接口负责执行和有界通道；持久attempt、run/CAS和有限重试使用下一节的编排接口，不能仅根据程序退出码认定训练成功。现有[本地执行和搜索CLI](../../../docs/user-guide/cli.md)与SDK复用持久协议。

## 持久任务与原生 worker

[durable.cpp](durable.cpp) 直接使用安装头 `<jyppx/liuding/experiment/local_execution/runner.hpp>`，调用 `execute_local_job()` 创建或重放冻结任务，`control_local_job()` 查询、取消或恢复。返回的 `LocalJobResult::status_code` 是编排状态：0成功或查询、1失败、7归属未确认、130取消；原生程序退出码保留在版本化JSON结果中。JSON采用与CLI相同的 `local_run_spec` v1，边界见[持久执行合同](../../../docs/development/local_process_runner.md#持久执行-cli)。它仍是完整训练RunSpec的有界子集。

调用者以 `LocalJobWorkerLaunch` 指定绝对worker程序、SHA-256和固定argv前缀；每个新attempt在写启动意图前核对程序身份。SDK追加 `--root STORE --id JOB --attempt N --intent-sha256 SHA`，程序在指定模式下转交 `run_local_job_worker()`。worker独立持有执行锁和提交结果，可与调度调用者分开存活；未知启动意图不因空闲锁、PID或时间而自动重试。确认终态的查询/重放不再依赖原程序或worker仍存在；`abandon_local_job()` 只接受原合同要求的显式外部清理确认，不猜测遗留进程。

示例第一次原生任务返回75，第二次提交具名结果；SDK核对同attempt的typed输出后成功，并在另一个原生进程中重放相同终态。它演示持久协议，不执行模型训练或授予任务质量资格；无需Python、CLI或第三方JSON头。

```sh
cmake --build out/build/YOUR_PROFILE --target jyppx_liuding_durable_local_execution_example --parallel 1
out/build/YOUR_PROFILE/examples/liuding-forge-example-durable-local-execution /absolute/unused-output
```

保留示例输出目录可检查原始run/CAS和 `example-spec.json`；每次新的演示使用未占用目录。安装消费者只需 `find_package(jyppx CONFIG REQUIRED COMPONENTS experiment)` 并链接 `jyppx::liuding::experiment`。

## 使用自己的程序和规格

[job.cpp](job.cpp) 是可复用的原生SDK调用示例，以自身作为独立worker，执行调用者提供的冻结 `local_run_spec` 文件。应用程序由规格的 `executable` 和字面量 `arguments` 指定，须自行读取attempt上下文并提交声明的typed输出；示例不替换应用的训练和检查点语义。它只链接experiment，读取规格最多64KiB，不依赖CLI或第三方JSON头。

```sh
cmake --build out/build/YOUR_PROFILE --target jyppx_liuding_local_job_example --parallel 1
out/build/YOUR_PROFILE/examples/liuding-forge-example-local-job environment /absolute/spec.json
out/build/YOUR_PROFILE/examples/liuding-forge-example-local-job execute /absolute/store my-job /absolute/spec.json
out/build/YOUR_PROFILE/examples/liuding-forge-example-local-job show /absolute/store my-job
out/build/YOUR_PROFILE/examples/liuding-forge-example-local-job recover /absolute/store my-job
out/build/YOUR_PROFILE/examples/liuding-forge-example-local-job cancel /absolute/store my-job
```

`environment` 只返回当前有效环境的 `environment_sha256`，可将该值写入规格绑定后再执行，不创建STORE。`execute` 创建或重放相同计划，`recover` 只在持久协议允许时推进剩余attempt；确认终态的重放不再执行应用。其余操作stdout是SDK返回的版本化JSON，进程退出码使用前述0/1/7/130编排状态。应用需要的运行库搜索路径、环境、时限、CPU放置和可选OS内存限额均应在规格中明确固定；Linux内存限额还要求调用者提供可写的委派cgroup父目录。
