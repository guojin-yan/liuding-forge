# 训练调度器适配

公共头文件为 `<jyppx/liuding/training/scheduler_adapter.hpp>`。`training::SchedulerAdapter` 将训练事件转为 `optim::LrScheduler` 的 batch、epoch、metric 或 manual 调度；`SchedulerInput` 携带 trigger、一基连续 event_index 和可选指标。

```cpp
auto scheduler = std::make_shared<jyppx::liuding::optim::StepLrScheduler>(
    optimizer,
    jyppx::liuding::optim::StepLrSchedulerOptions{.step_size = 10, .gamma = 0.1});
auto scheduler_adapter =
    std::make_shared<jyppx::liuding::training::SchedulerAdapter>(scheduler);

scheduler_adapter->step({.trigger = jyppx::liuding::optim::SchedulerStepMode::epoch,
                         .event_index = 1});
checkpoint_registry.register_participant(scheduler_adapter);
```

错误 trigger、重复或跳过事件、缺少/非有限 metric、向非 metric 事件传入 metric，都会在修改前拒绝。metric 事件调用 `step(metric)`，其他事件调用无参数 `step()`。

适配器同时是 `CheckpointParticipant`，复用调度器的完整描述符、状态树和组学习率。接入当前 Trainer 时，把它放入对应的 `OptimizerExecutionBinding::scheduler`，训练器负责事件投递和参与者登记；不要在应用中重复调用 step。直接使用时由调用方串行管理事件和检查点 I/O，见[优化器教程](../tutorials/03_optimizers.md)。
