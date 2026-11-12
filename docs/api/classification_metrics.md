# 分类 Precision、Recall 与 F1

公共头为 `<jyppx/liuding/training/metrics/classification.hpp>`，target 为 `jyppx::liuding::training`。`Precision`、`Recall`、`F1Score` 在构造时固定指标 ID、类别数、聚合方式、零分母策略和可选忽略标签；更新接受形状为 `{batch, num_classes}` 的有限浮点 logits，以及同设备的整数标签。类别不会随数据增长，浮点标签不会被静默取整。

```cpp
using jyppx::liuding::training::metrics::ClassificationAverage;
using jyppx::liuding::training::metrics::F1Score;

F1Score validation_f1({.metric_id = "metric.validation.f1",
                       .num_classes = 10,
                       .average = ClassificationAverage::macro,
                       .ignore_index = -1});
validation_f1.update(logits, labels);
const double score = validation_f1.compute();
```

统计只保存 int64 混淆矩阵。`micro` 先合并计数再计算，`macro` 对全部固定类别等权平均，`weighted` 按真实类别样本数加权；`none` 通过 `per_class_scores()` 返回逐类结果，标量 `compute()` 会拒绝。零分母由 `ZeroDivisionPolicy` 显式选择零、一或抛 `NumericalError`。空集或全忽略状态抛 `NotReady`，不能把缺少数据当成通过的零分。

`merge()` 要求完整 descriptor 相同，从总计数精确合并 batch 或 rank 分片；不能用每批分数的平均值替代整体指标。每个对象也是 exact `CheckpointParticipant`：当前 schema 分别保存固定 topology 与有界 little-endian int64 计数，恢复先验证身份、字段、编码大小及计数范围，再替换状态。不同指标类型或数据 split 应使用不同 `metric_id`。

实际模型推理、来源切片和报告的完整可运行流程见[分类评估示例](../../examples/evaluation/classification/README.md)；评估 session 的数据、模型和配置绑定见[评估保存与恢复](training.md#评估保存与恢复)。
