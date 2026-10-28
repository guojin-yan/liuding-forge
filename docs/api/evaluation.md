# 本地评估与质量报告

CMake target 为 `jyppx::liuding::experiment`，公共头文件为 `<jyppx/liuding/experiment/evaluation.hpp>`。实际模型推理接口见[训练与评估](training.md)。

`EvaluationSuite` 固定任务、数据集摘要、环境、seed 和有序规则。每条规则指定完整指标定义、切片上下文、最小样本数及阈值；修改规则时应递增 suite 版本，其规范 JSON 摘要也随之变化。生产者身份包含 evaluation/run ID、模型和配置 SHA-256、实际构建指纹及 adapter ID/version。

`evaluate_observations` 评判调用方提供的最终观测值；`run_evaluation` 先校验 suite 和身份，再执行受信任 C++ adapter，返回后再次检查取消和超时。它不隔离或强行中断任意代码。CLI 只读取实测输入文件，不从 JSON 加载可执行 adapter。

## 输入、报告与比较

`EvaluationInput` schema 1 包含 `schema_version`、`identity` 和 `observations`，通过 `evaluation_input_json` / `parse_evaluation_input_json` 编解码。观测值保存完整指标定义、类型化 step、数值状态、上下文及可选样本数；NaN、Inf 和缺失状态不会变成通过的零值。同一指标定义与上下文只允许一个最终观测值，未知观测值不能静默丢弃。

`QualityReport` 保存完整 suite、生产者身份及每条规则的结果。`quality_report_json` 和 `parse_quality_report_json` 重新计算规则及总判定，拒绝篡改的 verdict 或 reason。`quality_report_html` 将已校验的规范 JSON 转义后嵌入独立 HTML，不使用脚本或远程资源。报告编解码本身不证明推理已执行，也不认证生产者身份。

| 判定 | 含义 |
| --- | --- |
| `pass` | 所有适用的实测规则均达到包含边界的要求。 |
| `fail` | 有限实测值或方向性回归超过允许范围。 |
| `incomplete` | 缺少必要观测、有限数值或足够的已知样本。 |
| `warn` | 为带明确限制的可用证据预留。 |
| `invalid` | 为不兼容证据预留；非法契约直接抛类型化异常。 |

`compare_quality_reports` 要求 suite 摘要、adapter 契约和配对 step 含义相同，保留不同模型、配置及构建身份，检查绝对和相对方向性回归。基线为零且无法定义相对回归时结果为 incomplete；不自动给出统计显著性或置信区间。

所有 JSON 编解码拒绝重复、缺失或未知键、非法 UTF-8、负的无符号值和未知 schema。`EvaluationLimits` 限制字节、节点、容器深度及字符串字节；节点包括容器、键和标量，根容器深度为一。完整报告中的嵌入 suite、重复指标定义和全部观测共同占用预算，同限额下成功编码的内容可再次解码。

## CLI 与持久化

```sh
liuding-forge evaluation run --suite suite.json --input measured.json \
  --output quality.json --html quality.html --format json
liuding-forge evaluation show --report quality.json --format text
liuding-forge evaluation compare --baseline baseline.json --candidate quality.json --format json
```

将报告关联到已有本地 run 时添加 `--root RUNS --reference-id ID --attached-ms UNIX_MS`，run ID 取自输入身份。命令把规范报告写入 CAS 并提交 `evaluation.report` 关系；重试必须复用相同关系 ID、时间和内容，冲突内容会被拒绝，返回的 digest 标识确切报告。相同 evaluation 的调用方应固定同一关系 ID，不以新 ID 绕过冲突。

三类实际推理分别使用 `training::ClassificationEvaluationSession`、`SegmentationEvaluationSession`、`DetectionEvaluationSession` 恢复已提交的批次游标和任务统计；成功 EOF 后可从这些状态重建相同报告，约束见[评估保存与恢复](training.md#评估保存与恢复)。最终提交 hook 可先保存完成态检查点，再使用稳定关系 ID 发布报告，重建后重试该 hook。检查点文件与 CAS/SQLite 关系是可重试的分步操作，不构成跨存储原子事务。

输出文件要求父目录已存在；相同内容允许重放，已存在的不同内容会被拒绝。每次调用应独占输出路径；展示文件与 CAS 关系分别提交。退出码为成功 0、评判失败或不完整 1、非法输入 2、操作失败 3、取消 130、超时 124；text 与 schema-1 JSON 使用相同判定。

## 基础数据质量检查

`inspect_data_quality` 接受封闭的数值字段 schema 和有界投影行。每行提供 sample ID、split、内容摘要、数值字段及可选标签，统计缺失、非有限、越界、未知字段、重复 ID/内容、精确跨 split 内容泄漏和冲突标签。空数据结果为 incomplete，报告只保留汇总计数，不复制样本 ID、标签、原始数值或内容。

`liuding-forge data inspect|verify --input projection.json --format text|json` 使用 schema 1：`schema_version`、`schema`、`rows`。字段定义为 `required`、可空的 `minimum` 和 `maximum`；行定义为 `sample_id`、`split`、`content_digest`（`algorithm` / `hex`）、`fields` 和可空 `label`。JSON 数值必须有限或为 null，C++ API 另可接收显式 IEEE 非有限值。Tensor dtype/rank 由数据及模型 schema 校验；近重复、公平性、鲁棒性和统计推断不属于该基础检查器。
