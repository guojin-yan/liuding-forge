# 指标、batch 与自定义学习率调度器

包含 `<jyppx/liuding/optim/event_lr_scheduler.hpp>`，链接 `jyppx::liuding::optim`。

| 类型 | step mode | 可变状态 |
| --- | --- | --- |
| `ReduceLrOnPlateauScheduler` | metric | 最佳指标、未改善次数、cooldown |
| `OneCycleLrScheduler` | batch | 有界的成功事件计数 |
| `CyclicLrScheduler` | batch | 成功事件计数 |
| `LambdaLrScheduler` | epoch | 成功事件计数，配置保存稳定 callable 身份 |

每次转换先完成预览与校验，再一次提交所有组的学习率。callback 异常、无效 metric、非有限结果、OneCycle 越界、检查点不匹配或资源拒绝均保留原计数与学习率。

## Plateau 指标

`ReduceLrOnPlateauScheduler::step(metric)` 接受有限指标，方向和阈值模式使用枚举。最低学习率可为空（全部为0）、一个广播值或与稳定参数组一一对应的向量。相对阈值使用最佳指标的绝对值，负指标仍保留指定的改善方向。

## Batch 调度

OneCycle 构造时将各组设为 `max_lr/div_factor`，严格接受 `total_steps` 次调用，最后达到非负终值。Cyclic 支持 triangular、triangular2、exp-range 和带稳定身份的自定义 scale。`cycle_momentum=true` 抛 `NotImplemented`；当前优化器可变控制仅包含学习率。

## Callable 与恢复

Lambda 和自定义 Cyclic scale 要求稳定 callable ID，函数对象不序列化。精确恢复前由应用在同一 ID 下注册相同语义，ID 不同会导致拓扑不匹配。

`LrSchedulerCheckpointParticipant` schema3 保存完整描述符、各状态树节点的 typed control 和当前学习率；不读取旧 scalar map 或 participant schema2。Trainer 事件接入见[调度器适配](scheduler_adapter.md)，调用示例见[优化器教程](../tutorials/03_optimizers.md)。
