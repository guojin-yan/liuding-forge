# C++ 分类评估、断点续评与质量报告

[classification.cpp](classification.cpp) 完整运行固定 CPU FP32 线性模型，演示实际推理、整体和来源切片统计、忽略标签、评估检查点、新进程恢复和 JSON/HTML 质量报告。示例不下载数据或权重，不进行训练；固定模型的 40% 准确率仅用于核对接口和计数，不代表模型质量。

## 构建

使用[兼容矩阵](../../../compatibility.json)列出的工具链与 LibTorch，安装 `training`、`interop`、`experiment` 组件。可在仓库现有 profile 中启用 examples，再构建 `jyppx_classification_evaluation`；也可直接消费已安装 SDK：

```sh
cmake -S examples/evaluation/classification \
  -B out/build/YOUR_PROFILE/_examples/classification-evaluation \
  -Djyppx_DIR=/path/to/install/lib/cmake/jyppx \
  -DCMAKE_PREFIX_PATH=/path/to/install
cmake --build out/build/YOUR_PROFILE/_examples/classification-evaluation \
  --config Release --target jyppx_classification_evaluation --parallel 1
```

Windows 选择与 SDK 匹配的 x64 Release 生成器；程序名为 `liuding-forge-example-classification-evaluation.exe`，多配置生成器通常将其放在构建目录的 `Release/`。运行前按[构建说明](../../../CONTRIBUTING.md)准备 LibTorch 和 SDK 运行库。Linux 程序名没有 `.exe`。

## 连续评估与新进程恢复

以下以 Linux shell 为例，将 `EVALUATOR` 替换为实际程序路径，并使用专属于示例的输出目录。Windows 可在 PowerShell 中用 `& '程序路径.exe' full '输出目录'` 调用相同模式。

```sh
EVALUATOR=out/build/YOUR_PROFILE/_examples/classification-evaluation/liuding-forge-example-classification-evaluation
"$EVALUATOR" full /runs/classification-full
"$EVALUATOR" pause /runs/classification-paused
"$EVALUATOR" resume /runs/classification-paused
"$EVALUATOR" resume /runs/classification-paused
```

每行都是独立进程。`full` 连续推理三个 batch，每批两行；`pause` 在第一批成功提交并保存 `evaluation.state` 后退出，只推理一次；第一次 `resume` 恢复检查点，只推理剩余两批；再次 `resume` 从完成态重建相同报告，模型 forward 和 source 调用均为零。程序会核对这些实际调用数，并在标准输出中打印 `forwards`、`source_calls`、已提交批次数及完成状态。

每批和 EOF 的提交 hook 使用 `CheckpointCoordinator` 与 `FileCheckpointStore` 保存 session；协调器只注册这个评估 participant。模型参数固定且 forward 无随机或可变状态，因此各进程重建同一模型。自己的随机模型、可变 buffer 或有副作用的数据源需要额外协调相应状态，不能直接套用这一简化假设。

```sh
"$EVALUATOR" cancel /runs/classification-cancelled
"$EVALUATOR" resume /runs/classification-cancelled
"$EVALUATOR" reject-identity /runs/classification-paused
```

`cancel` 在第一批持久提交后请求协作取消，确认只保留一批进度；下一条命令在新进程继续。`reject-identity` 故意改变配置摘要，确认 exact 恢复拒绝且新 session 游标保持零；该模式以退出码 0 表示“成功验证拒绝”。其他异常返回非零。

数据和模型摘要来自实际输入、标签、来源索引及实际参数字节，并包含固定形状、dtype 与线性模型结构标签；配置摘要绑定 batch、类别、忽略规则、并列规则和来源。原始 tensor 字节依赖所用架构与 dtype，示例要求同一平台、SDK 构建、数据、模型和配置恢复，不授予跨平台或跨设备恢复资格。库构建身份使用安装 SDK 自身的指纹，不将示例源码版本冒充 SDK 的完整重建来源。

## 核对指标与报告

线性模型为 `logits = [x0, x1, -x0-x1]`，bias 为零；相同最大值选择最小类别索引。六行标签为 `[0, 1, 1, 1, -1, 2]`，预测为 `[0, 1, 2, 0, 0, 0]`，标签 `-1` 被忽略。混淆矩阵行是真值、列是预测：

```text
1 0 0
1 1 1
1 0 0
```

最终 `rows_seen=6`、有效样本 5、accuracy 为 `2/5`、macro precision/recall 均为 `4/9`、macro F1 为 `1/3`。宏平均包含三个固定类别，直接从整体混淆计数计算，不能平均每批分数。来源 `a`、`b` 分别有 3、2 个有效样本；预先声明但没有样本的 `empty` 来源保留缺失分数。

| 文件 | 内容与预期判定 |
| --- | --- |
| `evaluation.state` | 完整评估身份、已提交游标和统计；暂停时为一批，完成时为三批加 EOF。 |
| `measured.json` | 全部整体与来源观测，包括缺失值、样本数及完整指标定义。 |
| `quality.json` / `quality.html` | 实际整体 accuracy 与 0.4 门槛比较，判定 `pass`。 |
| `strict.json` | 同一实测值与另一套 0.9 门槛比较，判定 `fail`。 |
| `empty-source.json` | 空来源要求至少一个有效样本，判定 `incomplete`；缺失不变成零或通过。 |

连续、暂停恢复及取消恢复的五份最终报告应逐字节相同；完成态重放也保持这些内容。示例允许重放相同报告，拒绝覆盖内容不同的报告。检查点与报告分别提交；此流程不构成跨文件原子事务，也不提供外部发布或 exactly-once 保证。输出目录应由当前调用独占。

`ClassificationEvaluationSession` 的资源选项限制批次数、行数、类别、来源及已知混淆矩阵 payload；示例使用 1MiB 检查点和协调器回滚预算。这些逻辑边界不等于模型 workspace、allocator 或进程 RSS 的硬上限。评估临时启用 eval/no-grad，返回或异常后恢复原训练模式，示例同时确认参数保持原字节。

构建目录中可用 `ctest -C Release --output-on-failure -j 1 -R '^ClassificationEvaluation-.*-Example$'` 运行八条示例合同；fixtures 自动安排暂停、恢复与重放依赖。实际图像训练、检查点和检测/分割任务见[评估示例目录](../README.md)，公共 API 见[训练与评估](../../../docs/api/training.md)及[质量报告](../../../docs/api/evaluation.md)。
