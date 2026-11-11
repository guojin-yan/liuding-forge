# 用有界搜索运行原生配方

`search` 的 `recipe_bridge` 模式支持 `jyppx.liuding.paired-affine/v1`、`jyppx.liuding.imagefolder.resnet18/v1` 和有限的 `jyppx.liuding.imagefolder.transfer/v1` 迁移配方。迁移桥接目前仅接受 CPU、ResNet50、冻结骨干及 BN eval。每个 trial 调用真实 C++ 配方，确认训练、三类 typed 产物与持久提交的最终 accuracy 后，才参与 best 选择。

先按 `multi_io_recipe.md` 准备完整 paired-affine 数据与有效配置，再使用实际绝对路径：

```sh
liuding-forge recipe check --file /path/paired.json --format json > resolved.json
liuding-forge recipe local-spec --file /path/paired.json --root /path/unused-recipe --store /path/search-store --spec /path/local.json --executable /path/bin/liuding-forge --pause-after 259 --timeout-ms 360000 --format json > prepared.json
```

这两步检查并冻结输入，不训练。生成搜索规格时，复用 `prepared.json` 的`data.specification`，使用 `resolved.json` 的 `data.config` 作为完整`configuration`，然后设置以下搜索专属内容：

```json
{
  "recipe_bridge": {
    "format": "jyppx.liuding.local_search_recipe_bridge",
    "schema_version": 1,
    "recipe": "jyppx.liuding.paired-affine/v1",
    "seed_policy": "configuration"
  },
  "strategy": "grid",
  "seed": 8191,
  "max_trials": 1,
  "max_duration_ms": 600000,
  "max_failures": 1,
  "space": [
    {"name": "learning_rate", "config_pointer": "/learning_rate", "values": [0.1]}
  ]
}
```

完整搜索规格仍使用 `format=jyppx.liuding.local_search_spec`、`schema_version=1`、`project_id=liuding-forge`、`experiment_id=paired-affine`，并提供原有的 `configuration`、`execution`、`objective`、`inputs`、`outputs`。将执行规格的 `training.configuration` 改成完整字符串`${trial_configuration}`；`arguments` 必须是以下完整值模板：

```json
["recipe", "local-run", "--file", "${config}", "--root", "${recipe_directory}",
 "--format", "json", "--pause-after", "259"]
```

`working_directory` 可使用 `${trial_directory}`。`${config}` 是搜索冻结的完整有效配置；`${recipe_directory}` 是该 trial 内尚不存在的独立训练子目录。占位符只接受完整参数或环境值，不接受路径拼接、命令片段或未知占位符。

保留 local-spec 生成的三个 `training.outputs`，并在搜索的 `outputs` 中按相同顺序声明其 `reference_id`、`kind`、`role`，省略 `max_size_bytes`。这三个输出依次为 `checkpoint.last` / checkpoint / last、`model.weights` / artifact / model.weights、`training.result` / artifact / training.result。没有额外输入时，`inputs=[]`。

固定的 accuracy 目标如下，不能只提供同名指标而省略定义：

```json
{
  "definition": {
    "schema_version": 1, "metric_id": "classification.accuracy", "version": 1,
    "unit": 5, "aggregation": 3, "direction": 3,
    "parameters": {
      "classes": "2", "ignore_index": "none", "zero_division": "zero",
      "ties": "lowest_class_index", "macro_classes": "all"
    },
    "context_keys": ["slice", "source"]
  },
  "context": {"slice": "overall", "source": "all"},
  "step_kind": "sample", "selection": "last"
}
```

```sh
liuding-forge search execute --root /path/search-store --id paired-search --spec /path/search.json --format json
liuding-forge search show --root /path/search-store --id paired-search --format json
liuding-forge search recover --root /path/search-store --id paired-search --format json
liuding-forge search export --root /path/search-store --id paired-search --output /path/new-search-backup --format json
```

示例保留原任务的 1024/256 样本、配置种子 20261028、40 轮与 1280 次更新。第一个 attempt 在 259 次更新暂停，以 `exact_only` 从其已确认检查点完成剩余 1021 次更新。该模式仅允许单次运行或一次已声明的 75 退出码精确续训。普通搜索规格继续遵守其原有准入限制。

`seed_policy=configuration` 明确区分两个种子：配方实际训练消耗冻结配置中的`configuration.seed`；搜索 `seed` 控制 random 排序，并按原 SplitMix64 规则产生调度身份 `trial_seed`。桥接同时保存、核对 `training_seed` 和 `trial_seed`，不会把未消费的调度身份当作训练种子。多个 trial 可保持相同配置种子以比较超参数，也可以显式把 `/seed` 声明为搜索轴；每个实际训练种子仍须满足配方范围。

在 Ubuntu CPU 或 CUDA 上，可以把上例的 `max_trials` 设为 `3`、`values` 改成 `[0.05, 0.1, 0.2]`，分别用 `grid` 和 `random` 运行两个独立搜索；选择 CUDA 时使用对应构建的 CLI 和 `device=cuda` 配置。两种策略均逐个执行 trial，random 不重复抽取组合。多个 trial 的最后一条已提交 accuracy 相同时，按当前搜索的试验顺序选择最早的一条；两种策略的最佳学习率因此可能不同。每个搜索都可导出全部 trial 历史。

训练完成后，搜索检查原生 completed run、完整结果、配置/数据/模型/构建 lineage、真实提交的 accuracy 与三类 typed 输出。它先冻结桥接回执，再发布外层 trainingrun 的原子引用批和唯一原生指标，最后标记 completed。发布中断只能按同一回执重放；额外指标、额外或部分引用、身份漂移、未确认 owner/cleanup 都会保留证据并拒绝选优。

运行中的精确续训仍要求保留原训练目录。完成后的 search show/recover/export 读取已冻结的 SQLite/CAS 证据，导出后可通过 `run restore` 恢复并查看历史，无须原生训练工作目录。导出是包含 trial 历史的 run-store 备份，`deployable_model_bundle=false`。其他目标和更宽的搜索矩阵仍需独立验证。

## 使用 ImageFolder / ResNet18 分类配方

分类流程使用 CPU。先对完整分类配置执行 `recipe check --file <config> --format json`，将返回的 `data.config` 保存为配方文件；完整路径和配置值必须与搜索的 `configuration` 相同。用该文件调用 `recipe local-spec`，复用返回的 `data.specification` 作为 `execution`，再按上文替换完整参数占位符、`training.configuration` 和工作目录。生成规格的步骤只检查、冻结输入。

搜索的 `project_id=liuding-forge`、`experiment_id=image-classification`，`execution.training.model_id=resnet18`；桥接声明使用以下完整对象：

```json
{
  "format": "jyppx.liuding.local_search_recipe_bridge",
  "schema_version": 1,
  "recipe": "jyppx.liuding.imagefolder.resnet18/v1",
  "seed_policy": "configuration"
}
```

目标仍是最后一条 `classification.accuracy`，使用 `context={"slice":"overall","source":"all"}`、`step_kind=sample`、ratio / mean / maximize。定义中的 `parameters.classes` 必须是完整数据清单 `data.dataset.classes` 长度对应的十进制字符串，范围为 2–1024，例如二分类用 `"2"`；其余参数保留 `ignore_index=none`、`zero_division=zero`、`ties=lowest_class_index`、`macro_classes=all`。搜索在启动子任务前检查类别映射、原始图片内容、完整解析配置与冻结 typed 输入的一致性。

保持同样的三类输出和精确恢复模板。`--pause-after` 可在小于总更新数的位置暂停一次，接续使用该 trial 已确认的 `checkpoint.last`；学习率调度器可以按 epoch 或 update 计步，搜索按实际配置核对最终计数。分类模型固定为原生 ResNet18；完整初始状态文件继续遵守配方本身的字节、摘要和模型拓扑检查。

配置 `batch_norm_calibration=train_cumulative_once` 时必须同时设置 `calibration_max_snapshot_bytes`，且设备为 CPU。完成结果必须证明校准完整遍历一次训练集、没有额外 optimizer 更新，并将校准后的模型摘要绑定到最终 typed 模型。校准关闭时保留关闭状态。搜索目标来自原生 run-store 已提交的完整验证集准确率，模型、检查点、配置、数据和构建身份一致后才发布外层 training run。

运行和恢复命令沿用上文的 `search execute/show/recover`，导出流程也使用同一个 run-store 备份接口。分类质量仍按配置中的 `minimum_accuracy` 判断；小型自定义数据的通过结果不能替代完整 CIFAR10 等任务的质量要求。

## 使用冻结 ResNet50 迁移配方

先按[迁移学习配方](transfer_learning_recipe.md)准备本地完整预训练权重、描述文件、数据与配置，设置 `device=cpu`、`model=resnet50`、`backbone_mode=frozen`、`batch_norm=eval`。DINOv2、CUDA 和完整微调目前不属于搜索桥接支持范围；这些模式仍按迁移配方自己的接口和支持边界运行。

对完整配置执行 `recipe check --file <config> --format json`，保存返回的 `data.config`，再对该规范配置执行 `recipe local-spec`。搜索的 `project_id=liuding-forge`、`experiment_id=transfer-learning`、`execution.training.model_id=resnet50`；完整参数模板、一次 `exact_only` 续训和三类 typed 输出与上文相同。accuracy 定义中的类别数取完整数据清单的类别数。

迁移的 `recipe_bridge` 必须恰好包含 `format=jyppx.liuding.local_search_recipe_bridge`、`schema_version=1`、`recipe=jyppx.liuding.imagefolder.transfer/v1`、`seed_policy=configuration` 和对象字段 `transfer`。将 `recipe check` 返回的完整 `data.transfer` 原样复制到该字段，其中包含完整 `descriptor` 与 `weight_input_sha256`；省略描述字段或自行重算一个缩写身份都会被拒绝。`configuration` 仍是完整的 `data.config`，`execution.training.configuration` 仍使用完整字符串 `${trial_configuration}`。

搜索为每个候选冻结配置与完整迁移输入，启动前重新校验权重字节、描述、图片、类别及已解析配置。运行期间保持所有输入稳定；更改权重来源、预处理或描述时新建搜索。权重保留在显式本地路径，不复制进搜索输入 CAS，也不会自动下载。

每个候选按原配置完成全部更新和质量检查。搜索核对最终骨干摘要未变、BN 策略、权重输入身份与原生存储的完整描述，并使用该存储真实提交的完整验证集 accuracy；外层发布继续受原生 owner、进程清理、检查点、权重及结果回执约束。发布中断可复用同一冻结回执，不重训。完成后的历史查看与恢复只使用存储证据，不重新读取图片、权重或描述文件；正在执行的精确续训仍需保留原训练目录。小型四更新数据的结果不代表 Fashion-MNIST 等完整任务质量，也不能用于推导 DINOv2 搜索资格。
