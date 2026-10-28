# 训练双输入、分类与回归模型

本教程用一份可离线生成的数据，同时训练二分类和回归两个输出，完成配置检查、训练、评估、保存检查点和新进程续训。模型将 `left`、`right` 两组特征拼接后，分别送入分类和回归 Linear head；类别与回归目标只用于损失和评估，不会作为模型输入。

## 1. 构建 CLI

使用[支持矩阵](../../compatibility.json)中的 C++20 编译器和 LibTorch 组合，启用八个核心组件及 CLI。这份配方不依赖可选的 `models`、`data_image` 组件。本教程展示 CPU FP32 流程；CUDA 路径仍在验证中。

```sh
cmake -S . -B out/build/release -DCMAKE_PREFIX_PATH=/path/to/libtorch -DJYPPX_LIUDING_BUILD_TOOLS=ON
cmake --build out/build/release --config Release --target jyppx_cli --parallel 2
```

以下命令使用 PATH 中的 `liuding-forge`。Windows 产物位于 `out/build/release/tools/cli/Release/liuding-forge.exe`，单配置 Linux 产物位于 `out/build/release/tools/cli/liuding-forge`。

## 2. 准备数据

在源码根目录执行下面的命令，将数据根目录替换为仓库外的路径。工具只需要 Python 标准库，不下载数据，也不覆盖已有输出。

```sh
python scripts/datasets/prepare_paired_affine.py --data-root /path/to/datasets
```

生成的 `paired-affine-v1/` 包含 1,024 条训练样本、256 条验证样本及 `preparation.json` 内容清单，许可为 Apache-2.0。两个 split 的输入互不重复，数据、种子和摘要固定；详见[数据合同](../../recipes/multi-io/paired-affine.dataset.json)。这些合成样本用于学习多输入、多输出训练流程，不能证明真实业务任务质量或训练性能优势。

| 字段 | 模型中的用途 |
| --- | --- |
| `left`、`right` | 两个独立的 float32 `[batch,2]` positional 输入，特征范围 `[-1,1]` |
| `class_id` | int64 `[batch]`，取值 0 或 1；目标是特征线性组合的符号 |
| `regression` | float32 `[batch,1]`，目标是另一组仿射组合，范围 `[-1.1875,1.3125]` |

## 3. 配置与预检查

复制[配方模板](../../recipes/multi-io/paired-affine.json)到工作目录，将 `train_file`、`validation_file` 分别改为生成的两个 JSONL 文件。相对路径以配置文件所在目录为基准；Windows JSON 路径可写成 `F:/datasets/paired-affine-v1/train.jsonl`。训练和续训期间保持这些文件及其父目录稳定，路径不得经过符号链接或 Windows reparse point。

```sh
liuding-forge recipe schema --name paired-affine --format json
liuding-forge recipe check --file paired-affine.local.json --format json
```

`schema` 响应的 `data` 是可用于 `config lint` 的闭合 schema。所有字段都必须提供，未知字段、重复键、错误类型、越界值、错误输入/目标形状和非法类别 ID 会被拒绝。`check` 逐行校验两份数据，输出样本数、内容摘要、有效配置及总更新次数，不执行训练。

| 配置 | 含义 |
| --- | --- |
| `epochs`、`batch_size` | 模板固定 40 轮、每批 32，完整训练 1,280 次优化器更新 |
| `learning_rate`、`momentum`、`weight_decay`、`lr_gamma` | SGD momentum 参数和每个 epoch 执行一次的 StepLR |
| `regression_weight` | 总损失为 `cross_entropy(logits,class_id) + regression_weight * MSE(regression,target)` |
| `minimum_accuracy`、`maximum_mse` | 两个质量门槛均须通过；模板分别为 0.95 和 0.005 |
| `max_record_bytes`、`max_data_bytes`、`max_samples` | 单行含 LF 的字节上限、两个 split 合计文件字节上限、每个 split 的样本上限 |
| `max_batch_bytes`、`workers`、`prefetch_batches` | 单 batch 的已知 Tensor payload 上限，以及数据准备并发与预取限制 |

数据文件有界逐行解析为只读样本表，worker 共享该表；组批前校验每个样本共 28 字节的四路 Tensor payload。模板的整个数据文件上限为 1 MiB。逻辑上限不等于进程 RSS，也不包含 allocator、模型、优化器和检查点的全部开销。

## 4. 训练与评估

先创建输出目录的父目录，再使用一个尚不存在的运行目录。

```sh
liuding-forge recipe run --file paired-affine.local.json --root runs/paired-affine --format json
```

配方固定种子、单个 intra-op/inter-op 线程、确定性算法及 FP32 数值策略，每轮遍历完整验证集。完成后分别计算分类 accuracy/混淆矩阵和回归 MSE，并用同一份评估报告判断两个阈值。退出码 `0` 表示通过；`4` 表示训练和交付已完成，但质量门槛未通过。

| 输出 | 内容 |
| --- | --- |
| `config.json`、`dataset.json`、`lineage.json` | 有效配置、数据内容身份、数值策略和实际构建身份 |
| `latest.jyck`、`final.jyck` | 最新持久检查点和完成训练时的完整状态 |
| `model.safetensors` | 分类与回归两个 head 的全部参数 |
| `evaluation-suite.json`、`evaluation.json`、`diagnostics.json` | 冻结质量门槛、实际评估和训练诊断 |
| `result.json`、`run-store/` | 汇总结果和 SQLite/CAS 制品记录 |
| `bundle/` | 参数、双输入/双输出 schema、预处理、标签及交付描述文件 |

模型包当前未签名，完整传递 SBOM 仍待完成。它保存这份配方的参数和输入输出契约，推理应用需构造相同的模型并装载参数。

## 5. 暂停与独立进程恢复

在第 259 次更新先持久保存完整检查点，再结束当前进程；暂停点不必与 `checkpoint_every` 对齐，也不额外增加周期检查点或回调计数：

```sh
liuding-forge recipe run --file paired-affine.local.json --root runs/paired-resume --pause-after 259 --format json
liuding-forge recipe resume --file paired-affine.local.json --root runs/paired-resume --format json
```

恢复包括模型、SGD momentum、调度器、Trainer、训练/验证游标、回调、诊断、策略及 RNG。配置、数据内容或构建身份变化会被拒绝；需要变更时创建新运行。异常终止后可以从已成功发布的 `latest.jyck` 恢复；同目录同时运行会被锁拒绝。完成态重复 `resume` 返回原有结果，质量未通过的运行仍返回退出码 `4`。

## 6. 核对交付文件

```sh
liuding-forge model verify --path runs/paired-resume/model.safetensors --format json
liuding-forge checkpoint verify --path runs/paired-resume/final.jyck --format json
liuding-forge model bundle verify --path runs/paired-resume/bundle --format json
liuding-forge run verify --root runs/paired-resume/run-store --format json
```

遇到输入或续训拒绝时，先检查 `recipe check` 的摘要、路径、类别与字段范围。质量未通过时查看 `evaluation.json` 和 `diagnostics.json`，保留原运行结果；不要覆盖原证据或通过改低阈值重新解释同一次运行。
