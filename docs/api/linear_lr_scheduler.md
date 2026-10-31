# 线性、预热、常量与多项式学习率调度器

头文件为 `<jyppx/liuding/optim/linear_lr_scheduler.hpp>`，类型位于 `jyppx::liuding::optim`。调度器统一管理所有参数组的学习率，状态转换先校验再提交；`LrSchedulerCheckpointParticipant` 保存完整描述符、计数与当前学习率。

| 调度器 | 构造与推进行为 |
| --- | --- |
| `LinearLrScheduler` | 构造时应用 `start_factor`，在 `total_iters` 个事件内线性变化到 `end_factor`，随后保持；`total_iters=0` 时在首个事件达到终值 |
| `WarmupLrScheduler` | 构造时应用正数 `start_factor`，在 `warmup_iters` 个事件内达到基础学习率，随后保持 |
| `ConstantLrScheduler` | 构造时应用 `factor`，前 `total_iters` 个事件保持，下一事件恢复基础学习率 |
| `PolynomialLrScheduler` | 构造时保持基础学习率，第 `n` 个事件应用 `pow(max(0, 1-n/total_iters), power)`，到期后保持终值；`power=0` 时保持常量 |

四种 options 都可设置 `step_mode`，默认 `SchedulerStepMode::epoch`；设置为 `SchedulerStepMode::batch` 后，Trainer 在成功完成一次优化器更新后推进。梯度累积中的微批次、丢弃的尾批及 AMP 溢出跳过均不推进；`metric`、`manual` 和无效枚举值被拒绝。直接调用 `step()` 时，由调用方保证事件时机。

```cpp
auto warmup = std::make_shared<jyppx::liuding::optim::WarmupLrScheduler>(
    optimizer,
    jyppx::liuding::optim::WarmupLrSchedulerOptions{
        .warmup_iters = 100,
        .start_factor = 0.01,
        .step_mode = jyppx::liuding::optim::SchedulerStepMode::batch});
auto adapter = std::make_shared<jyppx::liuding::training::SchedulerAdapter>(warmup);
// 将 adapter 绑定到 FitRequest 中对应优化器的 scheduler 字段。
```

构造后首次参数更新使用 `start_factor × 基础学习率`，第 100 次成功更新后达到基础学习率。接入 Trainer 时还需包含 `<jyppx/liuding/training/scheduler_adapter.hpp>`。完整可运行程序见 [basic_trainer.cpp](../../examples/basic_trainer.cpp)，构建 target 为 `jyppx_basic_trainer`，可执行文件为 `liuding-forge-example-basic-trainer`。

后续衰减可用 [SequentialLrScheduler](composite_lr_scheduler.md) 组合，例如按更新计数的预热加多项式衰减。子调度器必须使用同一个优化器与相同 `step_mode`，构造各子对象会立即应用其初始因子；先构造保持基础学习率的衰减对象，再构造预热对象，最后按预热、衰减的顺序交给组合器。

恢复要求调度器类型、事件模式、options、基础学习率、优化器身份及有序参数组一致。改变 epoch/batch 模式会在恢复预检阶段失败，不修改现有状态或学习率。默认 epoch 模式保持原有的描述符与推进语义。
