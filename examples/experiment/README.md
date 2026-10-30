# 本地学习率搜索与最佳模型保存

这个离线示例使用 C++ 训练 `torch::nn::Linear`，通过网格或随机搜索比较三个学习率，将验证集均方误差、模型检查点和完整试验历史写入 SQLite 与内容寻址存储。训练、搜索、模型保存和恢复全部由 C++ 完成；运行时不下载数据，不调用 Python。源码见 [search_trial.cpp](search_trial.cpp)。

## 构建

按[构建指南](../../CONTRIBUTING.md#构建与验证)准备匹配的 LibTorch 和编译器，在现有兼容配置中启用 `JYPPX_LIUDING_BUILD_EXAMPLES`、`JYPPX_LIUDING_BUILD_TOOLS` 与八个 Core 组件；工作进程直接使用其中的 `experiment`、`nn` 和 `interop`。构建原生工作进程和命令行程序：

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


## 原生 TensorBoard

启用可选 `telemetry_native` 组件后，构建 `jyppx_liuding_native_tensorboard_example` 并运行 `liuding-forge-example-native-tensorboard exports/learning-run`（父目录需已存在）。[示例源码](tensorboard.cpp) 使用四个样本的预测值和目标值计算平均绝对误差，并导出生命周期、指标和数组缓冲区大小；重复提交相同事件可观察去重结果。[使用与限制](../../docs/api/native_tensorboard.md) 说明安装、映射、容量和失败处理。该示例演示遥测接入，不把数组大小当成训练进程的内存峰值。

## 离线 OTLP 协议准备

构建 `jyppx_liuding_otlp_projection_example`，运行 `liuding-forge-example-otlp-projection exports/otlp-learning`（父目录需已存在，目标目录须尚不存在）。[示例源码](otlp_projection.cpp) 将同一组评估事件保存为 JSONL 和官方 OTLP metrics、logs、traces 请求，并生成请求索引，便于学习中立事件到标准协议的映射。指标与完整审计日志成对准备，span 使用显式时钟锚点；无有效时钟或负时间的事件按支持边界处理。[API 与限制](../../docs/api/otlp_projection.md) 说明数据、容量及响应校验。该示例只写本地文件，不连接 collector，不表示网络导出已经完成。
