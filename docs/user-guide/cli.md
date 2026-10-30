# 命令行工具

`liuding-forge`（Windows 为 `liuding-forge.exe`）提供环境检查、配置解析、本地运行管理、训练配方、实际评估及模型/检查点校验。启用 `JYPPX_LIUDING_BUILD_TOOLS=ON` 时构建并安装到 `bin/`；它需要完整的八核心 SDK，缺少组件时明确报错，不隐式开启其他组件。关闭 tools 不影响库的使用。

用 `liuding-forge --help --format text` 查看命令，用 `liuding-forge --version --format json` 查看实际构建身份。命令接受 `--format text|json`；JSON 成功响应使用下面的统一 envelope：

```json
{"schema_version":1,"status":"ok","data":{}}
```

错误响应为 `status: "error"` 和包含稳定错误码、名称、消息、组件、操作及提示的 `error` 对象。退出码 `0` 表示成功，`1` 表示已完成但未通过/不完整的检查，`2` 表示输入或功能不支持，`3` 表示操作失败，`4` 表示训练配方已完成但质量门槛未通过，`7` 表示本地任务仍活跃或需要确认恢复，`124` 表示超时，`130` 表示取消。

## 常用命令

```sh
liuding-forge doctor --device cpu --format text
liuding-forge doctor --device cuda --expected-libtorch 2.14.1 --format json
liuding-forge config lint --schema config-schema.json --file training.json --format text
liuding-forge config render --schema config-schema.json --set /training/rate=0.01 --format json
liuding-forge run list --root local-runs --format text
liuding-forge model verify --path weights.safetensors --format json
liuding-forge model bundle verify --path model-bundle --format json
liuding-forge checkpoint verify --path checkpoint.jyck --format json
```

`doctor` 实际执行 Tensor、autograd 和序列化操作；CUDA 不可用或 LibTorch 版本不符时返回失败，不安装驱动或自动换用 CPU。配置 schema 声明字段类型、默认值、环境绑定及可脱敏字段，解析器按默认值、配置文件、环境绑定及 CLI 覆盖的顺序合并。配置与运行的完整约定见[本地运行存储](../api/local_run_storage.md)和[评估](../api/evaluation.md)。

## 训练与恢复

```sh
liuding-forge data prepare --name paired-affine --data-root /path/to/datasets --format json
liuding-forge recipe schema --name paired-affine --format json
liuding-forge recipe check --file paired-affine.local.json --format json
liuding-forge recipe run --file paired-affine.local.json --root runs/example --pause-after 256 --format json
liuding-forge recipe resume --file paired-affine.local.json --root runs/example --format json
```

双输入分类/回归配方使用核心 SDK；[完整教程](../tutorials/multi_io_recipe.md)包括离线数据、配置、两类损失、评估及新进程恢复。图像分类配方额外需要 `models` 和 `data_image`，省略 `--name` 的 `recipe schema` 返回其配置 schema，详见[ResNet18 教程](../tutorials/image_classification_recipe.md)。配方命令按配置中的 `recipe` 标识选择入口，不会根据数据文件名猜测任务。

## 模型包与可选运行时

原生 C++ 的 `model onnx export` 可从训练检查点恢复模型，经通用导出与实际 ORT 数值验证后发布 bundle；已有 ONNX 使用 `model onnx pack`，包内数值复验使用 `model onnx smoke`。所需组件、输入格式和支持边界见 [ONNX 导出与推理](../api/onnx.md)。

`model bundle inspect|verify` 有界检查目录、文件大小、摘要和 safetensors 映射，分别报告内容完整性、签名信任及运行时兼容性。启用 `bundle_signing` 组件后，CLI 提供原生 `model bundle sign`，验签通过 `--trust-policy` 和 `--trust-policy-sha256` 绑定信任策略；关闭该组件时不提供受信验签。使用方法见[模型包签名与信任策略](bundle-signing.md)；schema 的实际解释和推理仍由对应运行时负责。

启用 `JYPPX_LIUDING_INSTALL_TELEMETRY_SIDECAR=ON` 可单独安装遥测 sidecar 源码与锁定依赖声明；CMake 不安装 Python 包，调用方显式提供解释器和配置，见[进程传输](../api/telemetry_sidecar_process.md)。更完整的本地组合示例见[离线工作流](../examples/offline_workflow.md)。

## 本地任务

```sh
liuding-forge local execute --root local-runs --id trial-001 --spec local-run.json --format json
liuding-forge local show --root local-runs --id trial-001 --format json
liuding-forge local recover --root local-runs --id trial-001 --format json
liuding-forge local cancel --root local-runs --id trial-001 --format json
```

任务按冻结的 spec 直接启动可信可执行文件，原始 stdout/stderr 和独立 attempt 保存到现有 run/CAS，最多尝试 16 次；只对声明的非零退出码重试。取消、超时和输出超限回收进程树，不自动重试。`recover` 可补齐已确认结果的生命周期提交；活跃 worker 或缺少确认结果的 attempt 返回退出码 7，避免重复运行。完整 [spec 示例与进程约定](../development/local_process_runner.md)包含环境、字节和时间限制；训练检查点续训需由调用方显式配置。

## 本地参数搜索

`search` 在本机按顺序执行有界 grid/random 搜索，直接启动 C++ 训练程序。可先使用[完整训练示例](../../examples/experiment/README.md)生成带可执行文件 SHA-256 的 spec；示例实际训练 Linear 模型、记录验证损失并保存可精确恢复的模型状态。

```sh
liuding-forge search execute --root local-runs --id learning-rate --spec search.json --format json
liuding-forge search show --root local-runs --id learning-rate --format json
liuding-forge search recover --root local-runs --id learning-rate --format json
liuding-forge search cancel --root local-runs --id learning-rate --format json
liuding-forge search export --root local-runs --id learning-rate --output /absolute/new-backup --format json
```

spec 使用 `jyppx.liuding.local_search_spec`、`schema_version: 1`，首次执行后固定搜索空间、顺序、seed、执行程序、目标指标和预算。同一 ID 只接受相同计划；更改计划应使用新 ID。`grid` 按 `space` 中的轴顺序遍历，最后一轴变化最快；`random` 在有限组合中均匀、无放回抽样。每次只运行一个 trial，最多 256 个 trial、16 个离散选择轴和 100 万个组合；最长 24 小时，并以 `max_failures` 限制失败数量，不支持连续分布或并行 HPO。

| 字段 | 用途 |
| --- | --- |
| `configuration`、`space` | 配置对象；每轴包含 `name`、已有标量字段的 `config_pointer` 和 `values`，取值须与原字段类型相同；数值类型可以互用。 |
| `strategy`、`seed`、`max_trials`、`max_duration_ms`、`max_failures` | 搜索方法、无符号随机种子和正整数预算；墙钟预算从搜索创建时开始，重启不重置。 |
| `execution` | 沿用本地任务 spec，固定可信程序、argv、环境、超时和输出限额；搜索要求 `max_attempts: 1` 和空 `retry_exit_codes`。 |
| `objective` | 完整 `definition`、精确 `context`、`step_kind` 和 `selection: "last"`；单位、聚合方式和优化方向必须明确。 |
| `inputs` | 输入数组；各项声明 `name`、已有 `run_id`、`reference_id`、`sha256`，开始搜索前核验同一存储中的引用与字节。无输入时使用空数组。 |
| `outputs` | 非空输出数组；各项声明 `reference_id`、`kind`（`artifact` 或 `checkpoint`）和 `role`，选择最佳 trial 时逐项核验。 |

argv、环境值或 `working_directory` 可整体使用 `${store}`、`${trial_id}`、`${training_run_id}`、`${config}`、`${config_sha256}`、`${trial_directory}`、`${trial_seed}`、`${param.NAME}` 和 `${input.NAME}`；不支持字符串内拼接或 shell 展开。配置文件是已确认 CAS 内容的本地视图，程序须核对其 SHA-256。`inputs` 传入的占位值是摘要，程序通过 C++ `SQLiteRunStore`/`FileArtifactStore` 读取内容，并在训练 run 上保留角色和引用 ID 均为 `search.input.NAME` 的同摘要引用。

训练程序必须创建 `${training_run_id}`，使用计划的 `project_id`/`experiment_id`、父 run `${trial_id}`、配置摘要 `${config_sha256}`，记录源码和构建身份，并通过 C++ run store 提交实际指标、声明的输出及 `completed` 状态。搜索从 SQLite/CAS 读取证据，不解析 stdout 推测训练结果。仅定义摘要、上下文和 step 类型完全匹配的最后提交点参加选择；最后一个匹配点为 NaN、无穷或缺失值时，该 trial 无效，不回退到旧值。同值保留最早的 trial。

`show` 返回 trial 参数、seed、训练 run、指标、输出与 `best`；`stop_reason` 说明 trial/空间已遍历，或墙钟、失败预算、取消触发停止。已有有效 trial 且正常停止时搜索为 `completed`；没有有效 trial 为 `failed`；取消为 `cancelled`。失败 trial 不覆盖旧结果，也不自动续训。恢复会重用已确认结果；活跃或结果未确认的 attempt 返回 7 并阻止后续 trial。需要训练续训时，用新搜索 ID 和显式检查点参数建立新的训练 run。

`export` 在复核最佳训练结果后，生成完整原生 SQLite/CAS 备份，包含所有 trial 历史和最佳选择，供 `run restore` 恢复；输出路径必须是不存在的绝对目录。该备份不是可部署 ModelBundle。恢复到新目录后可查看已有结果；尚未确认的 trial 保留原 admission 中的执行路径，不会自动迁移并重新启动。
