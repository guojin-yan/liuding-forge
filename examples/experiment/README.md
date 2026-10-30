# 本地学习率搜索与最佳模型保存

这个离线示例使用 C++ 训练 `torch::nn::Linear`，通过网格或随机搜索比较三个学习率，将验证集均方误差、模型检查点和完整试验历史写入 SQLite 与内容寻址存储。训练、搜索、模型保存和恢复全部由 C++ 完成；运行时不下载数据，不调用 Python。源码见 [search_trial.cpp](search_trial.cpp)。

## 构建

按[构建指南](../../CONTRIBUTING.md#构建与验证)准备匹配的 LibTorch 和编译器，在现有兼容配置中启用 `JYPPX_LIUDING_BUILD_EXAMPLES`、`JYPPX_LIUDING_BUILD_TOOLS` 与八个 Core 组件；工作进程直接使用其中的 `experiment`、`nn`、`interop`、`training` 和 `optim`。构建原生工作进程和命令行程序：

```sh
cmake --build out/build/your-profile --config Release --target jyppx_liuding_search_trial jyppx_cli --parallel 2
```

以下 PowerShell 命令从仓库根目录运行，示例采用 Windows 多配置目录；按实际配置调整两个可执行文件的位置。Linux 单配置构建使用 `examples/liuding-forge-example-search-trial` 和 `tools/cli/liuding-forge`，其余参数相同。运行目录需要能加载当前构建使用的 LibTorch 动态库；示例 target 会复制 Windows 运行依赖。

## 运行网格搜索

先创建仓库外的独立工作目录，再由工作进程生成搜索配置。可执行文件必须使用绝对路径；生成的配置会固定该文件的 SHA-256 和当前工作目录，重新构建二进制后应重新生成配置。

```powershell
$worker = (Resolve-Path ./out/build/your-profile/examples/Release/liuding-forge-example-search-trial.exe).Path
$cli = (Resolve-Path ./out/build/your-profile/tools/cli/Release/liuding-forge.exe).Path
$work = Join-Path $env:TEMP ('liuding-search-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
Push-Location $work
try {
    $spec = Join-Path $work 'grid.json'
    $store = Join-Path $work 'runs'
    & $worker --write-spec $spec --strategy grid
    & $cli search execute --root $store --id linear-grid --spec $spec --format json
    & $cli search show --root $store --id linear-grid --format json
} finally {
    Pop-Location
}
```

工作进程固定从零权重和零偏置开始，用 32 个训练样本学习 `y = 3*x + 0.5`。每个试验执行 60 次全批次 SGD 更新，再用 10 个独立验证样本计算 `validation.mean_squared_error`。搜索依次尝试 `0.005`、`0.05`、`0.2`，每次只运行一个试验；选择指标定义、验证集上下文和步数均匹配的有限误差，误差相同时选择较早试验。

正常完成时，`status` 为 `completed`，`trials` 包含三个 `eligible: true` 的试验，`best` 指向学习率 `0.2` 的试验。每个试验的模型由 `ModelCheckpointParticipant` 和 `FileStateArchive` 保存为 `model-best.jyck`，随后在新模型实例中以 `exact_only` 恢复并逐值比较预测结果；只有通过该检查才发布 `checkpoint.best`。检查点保存参数状态，恢复时仍须构造相同 C++ 模型拓扑。

## 可选：为训练试验声明资源预算

在首次执行前，也可以给生成的配置增加 `execution.resources`，学习同一机器、同一存储中的资源准入。下面沿用 `$spec`，另用独立存储；模型训练仍由同一个 C++ 工作进程执行。

```powershell
$plan = Get-Content -LiteralPath $spec -Raw | ConvertFrom-Json
$resources = [ordered]@{
    capacity = [ordered]@{ cpu_slots = 2; memory_bytes = 8589934592; gpus = @{} }
    request = [ordered]@{ cpu_slots = 2; memory_bytes = 1073741824; gpu = $null }
}
$plan.execution | Add-Member -MemberType NoteProperty -Name resources -Value $resources
[IO.File]::WriteAllText($spec, ($plan | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
$resourceStore = Join-Path $work 'resource-runs'
& $cli search execute --root $resourceStore --id linear-resource-grid --spec $spec --format json
```

这里声明 2 个 CPU slots 和 1 GiB 主存请求；数值用于调度记账，不是测得的实际开销，也不限制 RSS。搜索仍依次执行三个试验；每次启动前保存占用，清理和双流结果均确认后保存释放记录。`local show --root $resourceStore --id TRIAL_ID` 的 attempt 中可查看 `resource_reserved` 和 `resource_released`。同一存储的容量固定，后续试验须重复相同容量；预算不足或未知清理状态返回退出码 7，不能据此强制重启。GPU 设备身份和边界见[资源准入说明](../../docs/development/local_process_runner.md#cpugpu-逻辑资源准入)。

## 随机搜索、查询与备份

生成另一份配置并使用新的搜索 ID 即可运行随机搜索。这里使用固定随机种子 `71`，从有限候选中不重复选择，三个试验仍会覆盖全部学习率；随机顺序不代表模型质量提升。

```powershell
$randomSpec = Join-Path $work 'random.json'
& $worker --write-spec $randomSpec --strategy random
& $cli search execute --root $store --id linear-random --spec $randomSpec --format json
```

从 `best.training_run_id` 获取最佳训练 ID，查询指标和模型引用。`checkpoint.best` 的 `artifact.sha256` 标识真正保存的检查点内容，`lineage` 将训练试验、冻结配置和父作业关联起来。

```powershell
$result = (& $cli search show --root $store --id linear-grid --format json | ConvertFrom-Json).data
$bestRun = $result.best.training_run_id
& $cli run show --root $store --id $bestRun --format json
& $cli run metrics --root $store --id $bestRun --metric validation.mean_squared_error --format json
& $cli run verify --root $store --format json
& $cli search export --root $store --id linear-grid --output (Join-Path $work 'backup') --format json
```

`search export` 导出完整实验存储备份，包括已有搜索和全部试验历史、SQLite 数据库、被引用的模型检查点及其内容校验清单。目标目录必须尚不存在，并且与活动存储目录分离。这份备份用于保留和迁移实验，不是可直接部署的 ONNX 模型包；检查点恢复与 ONNX 导出可继续参考[部署示例](../deployment/README.md)。

搜索配置给出三个试验、120 秒总时长、一次失败和每个原生进程 30 秒的上限。中断后先用 `search recover --root ... --id ...` 恢复已有搜索状态；`search cancel` 请求取消。已有终态搜索可以重新查询，使用同一搜索 ID 执行另一份配置会被拒绝。配置中的二进制哈希、进程限制和制品校验用于确认试验身份与结果，不代表训练吞吐或内存性能结论。

接入自己的训练程序时，保留模板参数接收、冻结配置校验、父试验 lineage、明确的验证指标语义和完成后的模型制品发布，再替换数据、模型、优化器与搜索空间。训练程序须在成功发布指标和检查点后将训练 run 标为完成；搜索只会把满足这些条件的试验纳入最佳结果。


## 新 job 显式恢复原生训练

同一个 `search_trial` 可执行文件也提供本地训练示例，固定模型、配置和生成式数据清单。先在独立目录中训练 20 步，保存模型、SGD momentum、已完成步数与输入身份，再指定该 attempt 的 checkpoint，在新 job 中继续到 60 步。这里使用带 momentum 的 SGD，与上面的学习率搜索配方分别执行。

沿用前文的 `$worker`、`$cli` 和独立 `$work`，另选一个存储：

```powershell
$localStore = Join-Path $work 'local-training-store'
$partialSpec = Join-Path $work 'partial-local.json'
& $worker --write-local-spec $partialSpec --store $localStore --until-step 20
$partial = (& $cli local execute --root $localStore --id partial --spec $partialSpec --format json | ConvertFrom-Json).data
$sourceAttempt = $partial.attempts[0].attempt_id
$resumeSpec = Join-Path $work 'resume-local.json'
& $worker --write-local-spec $resumeSpec --store $localStore --until-step 60 --resume-run $sourceAttempt
& $cli local execute --root $localStore --id resumed --spec $resumeSpec --format json
& $cli run show --root $localStore --id $sourceAttempt --format json
```

每个 attempt 的 `checkpoint.last` 保存三个原生参与者：`model.main`、`optimizer.main`、`run.inputs`；`training.result` 记录实际起止步数和验证均方误差。恢复要求相同模型拓扑、优化器选项和输入摘要，使用 `CheckpointCoordinator` 的 `exact_only`；只含模型参数的搜索快照不足以恢复这个训练过程。此示例没有随机批次、scheduler 或 AMP；接入这些能力时必须把它们的状态也加入 checkpoint。

应用检查和恢复 checkpoint，运行器校验 CAS 身份并管理进程生命周期。默认 `local recover` 修复已确认的执行状态；未知归属保持隔离。真实训练过程、配置/数据准备及恢复均由 C++ 实现；Python 仅用于独立测试读取和参考计算。示例规模不能用来证明训练速度或资源优势。

## 失败后自动从 checkpoint 续训

同一个 C++ 示例还可以展示已确认失败后的 checkpoint 续训。`--checkpoint-retry-after 20` 让首个 attempt 完成 20 步、提交完整 checkpoint 和训练摘要，再以声明的退出码 `23` 结束；运行器确认结果和进程树清理后，固定该 checkpoint 的 CAS 身份，启动第二个 attempt，使用相同配置和数据从第 20 步继续到第 60 步。这里的失败边界专为学习设置，实际异常不会自动变成可恢复状态。

```powershell
$retryStore = Join-Path $work 'checkpoint-retry-store'
$retrySpec = Join-Path $work 'checkpoint-retry.json'
& $worker --write-local-spec $retrySpec --store $retryStore --until-step 60 --checkpoint-retry-after 20
& $cli local execute --root $retryStore --id checkpoint-retry --spec $retrySpec --format json
$continued = (& $cli local show --root $retryStore --id checkpoint-retry --format json | ConvertFrom-Json).data
$continued.attempts | Select-Object index, run_status, training_sha256, result
& $cli run show --root $retryStore --id $continued.attempts[1].attempt_id --format json
& $cli local recover --root $retryStore --id checkpoint-retry --format json
```

预期 job 为 `completed`，两个 attempt 分别保留 `failed/23` 和 `completed/0` 的结果。第二个 attempt 的 `training.result` 应记录 `initial_step: 20`、`completed_steps: 60`、`resumed: true`；`checkpoint.last` 包含实际模型、SGD momentum、更新计数和输入身份。`local.training.inputs` 与 intent/result 中的 `training_sha256` 对应同一恢复清单，job 的 `local.training.inputs.2` 还记录首个结果摘要和 checkpoint 来源；原始配置中的 `resume` 仍为 `null`。重复 `recover` 查询相同结果，不再运行训练。

接入自己的 C++ 程序时，在 `training` 中显式声明 `retry_resume`，从 attempt 的清单和 job 的 `.N` 引用读取有效输入，然后执行完整 `exact_only` 恢复。保留原模型、配置和数据身份，确保退出前 checkpoint 已提交；运行器不会为应用推断 RNG、scheduler 或 AMP 状态。缺少 checkpoint、伪造或不完整的内容、身份变化、未知结果和未确认清理都不能转为 fresh 重试；详细约定见[本地运行器](../../docs/development/local_process_runner.md#已确认失败的-checkpoint-续训)。此策略仅用于 local job；typed grid/random 仍使用一次 attempt 和显式 checkpoint。

## 训练输入与搜索衔接

`--write-typed-search-spec` 演示搜索与 typed local 输入衔接。它使用带 momentum 的原生 SGD，三个学习率从相同零权重开始训练 60 步；模型标识和数据清单在整次搜索固定，每个 trial 的配置拥有独立、可校验的 CAS 身份。这是单独的搜索模式，前面的基础搜索仍展示只保存模型参数的流程。

```powershell
$typedStore = Join-Path $work 'typed-search-store'
$typedSpec = Join-Path $work 'typed-grid.json'
& $worker --write-typed-search-spec $typedSpec --store $typedStore --strategy grid
& $cli search execute --root $typedStore --id typed-grid --spec $typedSpec --format json
$typed = (& $cli search show --root $typedStore --id typed-grid --format json | ConvertFrom-Json).data
$typed.trials | Select-Object index, parameters, training_inputs, metric
& $cli run show --root $typedStore --id $typed.best.training_run_id --format json
```

将 `--strategy` 改为 `random` 并使用新的配置文件、搜索 ID，即可按固定种子无重复抽取候选。每个 trial 的 `training_inputs` 记录模型标识、真实配置、同源数据清单和可选 checkpoint；配置来源 `TRIAL.inputs` 的终态只表示输入已提交。训练 run 的 `checkpoint.last` 包含模型、SGD momentum、更新计数和输入身份，`training.result` 记录实际结果；两者与已确认的原生 attempt、验证指标和 best trial 一起保留在实验备份中。

需要继续一个已知 checkpoint 时，先在同一 `$typedStore` 中用 `--write-local-spec ... --until-step 20` 完成一次本地训练，再从 `local show` 的 `attempts[0].attempt_id` 明确取得 `$selectedAttempt`，生成新的搜索请求：

```powershell
$continuedSpec = Join-Path $work 'typed-continued.json'
& $worker --write-typed-search-spec $continuedSpec --store $typedStore --resume-run $selectedAttempt
& $cli search execute --root $typedStore --id typed-continued --spec $continuedSpec --format json
```

这个续训请求固定原学习率 `0.05`，只运行一次从 20 到 60 步的试验。`exact_only` 要求模型、优化器选项和配置/数据摘要一致；改变学习率会被拒绝。这里展示显式 checkpoint 与搜索结果追溯的接线，学习率调优从前一个 fresh 搜索模式开始。运行器不会自动选择 best 或失败 attempt 的 checkpoint；源 attempt 若没有已确认结果，即使 CAS 中已有 checkpoint，仍拒绝作为新的恢复来源。

## 原生 TensorBoard

启用可选 `telemetry_native` 组件后，构建 `jyppx_liuding_native_tensorboard_example` 并运行 `liuding-forge-example-native-tensorboard exports/learning-run`（父目录需已存在）。[示例源码](tensorboard.cpp) 使用四个样本的预测值和目标值计算平均绝对误差，并导出生命周期、指标和数组缓冲区大小；重复提交相同事件可观察去重结果。[使用与限制](../../docs/api/native_tensorboard.md) 说明安装、映射、容量和失败处理。该示例演示遥测接入，不把数组大小当成训练进程的内存峰值。

### 查看真实训练指标

也可以将前面搜索得到的 `$bestRun` 导入 TensorBoard。一次导出直接使用 `--output`；需要由 CLI 监督独立 C++ worker 时，使用显式 `--native-config`。两种方式共用相同 writer 和映射规则，分别选择新目录：

```powershell
$tbDirect = Join-Path $work 'tensorboard-direct'
& $cli run export --root $store --id $bestRun --backend tensorboard-native --output $tbDirect --format json

$tbWorker = Join-Path $work 'tensorboard-worker'
New-Item -ItemType Directory -Path $tbWorker | Out-Null
$tbOutput = Join-Path $work 'tensorboard-isolated'
$tbConfig = Join-Path $work 'tensorboard.json'
$tbSettings = [ordered]@{
    schema_version = 1
    directory = $tbOutput
    working_directory = $tbWorker
    operation_timeout_ms = 10000
    cleanup_timeout_ms = 2000
}
[IO.File]::WriteAllText($tbConfig, ($tbSettings | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
& $cli run export --root $store --id $bestRun --backend tensorboard-native --native-config $tbConfig --timeout-ms 60000 --format json
tensorboard --logdir $tbOutput
```

沿用前文的绝对 `$work` 路径，工作目录需要先创建，两个输出目录均不得预建或位于 `$store` 内；查看器单独安装。`--output` 与 `--native-config` 不能同时使用，也不接受 `--resume`。检查 `scan_completed`、`fully_represented` 和接受/不支持计数；制品引用、日志等仍保存在原始实验存储，不会伪装成 TensorBoard 标量。隔离导出失败后保留 `$store`，改写配置中的输出目录并重试。运行时全部由 C++ 完成；等待与清理预算、配置和帧容量见[原生 TensorBoard API](../../docs/api/native_tensorboard.md#使用受监督的隔离进程)。

## 离线 OTLP 协议准备

构建 `jyppx_liuding_otlp_projection_example`，运行 `liuding-forge-example-otlp-projection exports/otlp-learning`（父目录需已存在，目标目录须尚不存在）。[示例源码](otlp_projection.cpp) 将同一组评估事件保存为 JSONL 和官方 OTLP metrics、logs、traces 请求，并生成请求索引，便于学习中立事件到标准协议的映射。指标与完整审计日志成对准备，span 使用显式时钟锚点；无有效时钟或负时间的事件按支持边界处理。[API 与限制](../../docs/api/otlp_projection.md) 说明数据、容量及响应校验。该示例只写本地文件，不连接 collector，不表示网络导出已经完成。

## 原生 OTLP HTTP

启用独立可选 `telemetry_http` 组件后，构建 `jyppx_liuding_otlp_http_example`，运行 `liuding-forge-example-otlp-http 127.0.0.1 4318 http local-demo-1`，向自行启动的 Collector 发送一个指标及审计日志，并演示进程内重放去重。[完整教程](../../docs/tutorials/native-otlp-http.md) 说明显式依赖、HTTPS、取消与重试边界；[示例源码](otlp_http.cpp) 可直接用于学习公共 C++ sink 的接入。示例只在显式运行时联网，不作为默认自动网络测试。

## 将搜索结果导出到 Collector

上述搜索会产生真实训练指标和 checkpoint 引用。启用 `telemetry_http` 并构建 CLI 后，沿用 `$store`、`$bestRun`，按照[原生导出教程](../../docs/tutorials/native-otlp-http.md#导出已有训练-run预览)创建独占 worker 目录与固定 Collector 配置，然后显式运行：

```powershell
& $cli run export --root $store --id $bestRun --backend opentelemetry-native --native-config $otlpConfig --timeout-ms 60000 --format json
```

这条学习路径将 C++ 训练、持久化 run 查询和原生 OTLP 导出串起来。检查 `scan_completed`、`fully_represented` 和事件计数；网络失败后保留训练存储并按至少一次语义重试。原生入口属于预览，截止终止和清理边界见教程；运行时不调用 Python，也不会替用户启动 Collector。
