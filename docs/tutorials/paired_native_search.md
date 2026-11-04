# 用有界搜索运行 paired 原生配方

`search` 的 `recipe_bridge` 模式目前仅支持`jyppx.liuding.paired-affine/v1`。每个 trial 调用真实 C++ 配方，确认训练、三类 typed 产物与持久提交的最终 accuracy 后，才参与 best 选择。

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

训练完成后，搜索检查原生 completed run、完整结果、配置/数据/模型/构建 lineage、真实提交的 accuracy 与三类 typed 输出。它先冻结桥接回执，再发布外层 trainingrun 的原子引用批和唯一原生指标，最后标记 completed。发布中断只能按同一回执重放；额外指标、额外或部分引用、身份漂移、未确认 owner/cleanup 都会保留证据并拒绝选优。

运行中的精确续训仍要求保留原训练目录。完成后的 search show/recover/export读取已冻结的 SQLite/CAS 证据，导出后可通过 `run restore` 恢复并查看历史，无须原生训练工作目录。导出是包含 trial 历史的 run-store 备份，`deployable_model_bundle=false`。分类配方、迁移学习、其他目标和更宽的搜索矩阵尚未获得本桥接的验证结论。
