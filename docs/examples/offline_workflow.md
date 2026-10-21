# 离线训练、恢复与交付证据参考流程

`jyppx offline run` 在一个新目录内完成真实训练、持久化 checkpoint、销毁重建、恢复续训、模型导出、评测、运行存储审计及备份恢复。整个命令使用库的当前 Trainer、DataPipeline、优化器、调度器、CallbackDispatcher、EventOutbox 与 SQLiteRunStore，不依赖 Python Torch、网络或外部服务。

```text
jyppx offline run --root ./offline-evidence --device cpu --seed 20261021 --format json
```

CUDA 环境使用同一入口：

```text
jyppx offline run --root ./offline-cuda-evidence --device cuda --seed 20261021 --format json
```

`--root` 必填，父目录须已存在且可写，目标目录必须不存在。命令拒绝覆盖已有目录、文件或符号链接。`--device` 默认为 `cpu`，`cuda` 使用设备 0；`--seed` 默认为 `20261021`，接受 `0` 至 `9223372036854775807` 的十进制整数。省略 `--format json` 可查看文本输出。非法参数在创建输出目录前拒绝，CUDA 不可用也会在创建目录前失败。

## 实际执行内容

内置夹具包含 8 条训练样本和 8 条有固定偏移的验证样本，输入为 4 维 FP64 特征、输出为 3 类标签。网络为 `Linear(4,5) → tanh → dropout(0.25) → Linear(5,3)`，包括一个参与 checkpoint 的缩放 buffer。Trainer 使用带 momentum 和 weight decay 的 SGD、StepLR、可恢复的 F1 计数、随机 sampler、两个数据 worker 与预取，训练 3 个 epoch，共 12 次优化器更新。

流程先运行连续训练 baseline，再以相同 seed 从相同权重重做训练。第 6 次更新后的安全边界触发 callback checkpoint，完整保存模型、优化器、调度器、数据游标、callback、指标、事件 outbox 与 CPU/CUDA RNG 状态。

参考程序随后模拟“checkpoint 发布后进程丢失”：抛出内部标记，销毁整个训练对象域，通过 weak pointer 检查模型和优化器确已释放。它使用不同 seed 重建出不同初始权重，再从磁盘重开 checkpoint 恢复至第 6 步并训练至第 12 步。模拟阶段会取消旧域的后续事件投递，旧内存中的异常清理事件不会覆盖已保存事件游标。这是同一 CLI 进程内的销毁重建演示；真正的跨进程恢复仍由 FileCheckpointStore 的公开读写接口支持。

`recovery-check.json` 保留真实比较结果：

- baseline 与续训模型全部参数和 buffer 的 safetensors 字节必须完全一致。
- 最终训练/验证指标、样本消费顺序、epoch、optimizer/scheduler 与指标计数必须一致。
- 后续选定设备及 CPU 的实际随机数探针必须一致。
- checkpoint 中的遥测 stream 必须恢复，最终 outbox 必须排空。

任何比较失败都会使命令非零退出，并保留输出证据。这些检查针对同一次执行中的相同设备和软件环境，不能推导 CPU 与 CUDA、不同 GPU 或不同 LibTorch 版本之间的逐位一致性。

训练过程的 `training.f1` / `validation.f1` 是累计 confusion counts 导出的 running macro F1；Trainer 的 epoch 汇总是批次观测按样本数加权的均值。遥测 definition 保留该语义，不能将其误读成单次完整验证集的 macro F1。独立评测报告另行执行完整验证夹具推理，测量 accuracy 和平均 cross entropy。

## 运行来源和评测

两个真实 run 分别为 `run.offline.baseline` 与 `run.offline.recovery`。每个 RunMetadata 都记录编译产物的 Git/source identity、完整 BuildFingerprint、实际写出的 config 和 dataset 文件 SHA-256；同一内容通过 CAS 引用挂接到两个 run。`source_identity` 保留构建时 Git commit（可能包含 `+dirty`），不声称它是未提交源码树的完整内容摘要。

`evaluation-suite.json` 固定 suite ID/version、数据 manifest digest、设备/平台环境、seed、指标定义与回归阈值。评测通过 `run_evaluation` 调用程序内可信分类适配器，执行真实模型推理；没有读取或填入预设观察值。模型 digest 对应实际导出的 safetensors。

suite 检查 accuracy 的合法比例范围、cross entropy 非负，以及相对 baseline 的零绝对退化。其用途是验证恢复一致性与报告链路，夹具规模和阈值不代表生产分类质量或泛化能力。此入口是一个固定的库集成参考程序，不提供任意模型、数据或通用 recipe 引擎。

## 输出文件

| 文件或目录 | 内容 |
| --- | --- |
| `summary.json` | 完整成功结果，与 `--format json` 输出中的 `data` 相同 |
| `config.json`、`dataset.json`、`build.json` | 实际配置、夹具数值及构建来源 |
| `baseline-boundary.jyck`、`recovery-boundary.jyck` | 第 6 步的完整训练 checkpoint |
| `baseline-final.jyck`、`recovery-final.jyck` | 最终完整训练 checkpoint |
| `baseline-model.safetensors`、`recovery-model.safetensors` | 真实模型参数及 buffer |
| `recovery-check.json` | 销毁重建、参数、指标、数据游标和 RNG 比较 |
| `evaluation-suite.json`、`*-quality.json` | 版本化 suite 与真实推理报告 |
| `recovery-quality.html`、`quality-comparison.json` | 可读报告及 baseline 对比证据 |
| `*-run.json`、`*-references.json` | 实际 RunMetadata 与 checkpoint/model/report 等 CAS 关系 |
| `*-events.jsonl`、`*-stored-events.json` | JSONL 观测及从 SQLite 分页读取的中立事件证据 |
| `store/` | SQLite 运行数据库及 SHA-256 CAS |
| `backup/`、`restored-store/`、`storage-check.json` | 真实备份、恢复后的运行存储及内容一致性审计 |

SQLite evidence sink 是事件的持久化事实来源。JSONL observer 保存 canonical envelope bytes、SHA-256 和 JSON projection，并在重开时验证完整记录及去重；JSONL 的 flush 不等同于 fsync 持久性。备份前执行 SQLite、外键、事件及关联 CAS 完整性审计，恢复后再次审计，并比较每个 run 的全部元数据、产物引用与完整事件内容。

发生错误时目录保留，可能存在 `failure.json`，且不会生成成功的 `summary.json`。可通过 `jyppx doctor --device cpu|cuda` 检查环境，并对保留的文件使用 `checkpoint verify`、`model verify`、`run verify`。重新执行应选择另一个新目录。

## 章节统一验证

构建完成后可运行真正调用 CLI 的契约测试：

```text
python tests/cli/test_offline_workflow.py --cli /absolute/path/to/jyppx --device cpu
python tests/cli/test_offline_workflow.py --cli /absolute/path/to/jyppx --device cuda
```

Windows CPU 与远端 Ubuntu/CUDA 使用同一测试脚本和该环境已有的增量构建目录。脚本会生成临时的新输出目录，核对真实模型/checkpoint 读者、CAS 字节摘要、SQLite/JSONL 事件、一致性评测、备份恢复以及拒绝覆盖已有路径的行为。测试不重建程序，也不会为缺失的 CUDA 自动跳过验证。
