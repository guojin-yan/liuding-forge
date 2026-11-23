# StepLR 调度器

公共头文件为 `<jyppx/liuding/optim/step_lr_scheduler.hpp>`，链接 `jyppx::liuding::optim`。`StepLrScheduler` 通过 `OptimizerRateControl` 控制优化器；`step_size` 为正整数，`gamma` 为有限非负数，按完整 epoch 周期衰减。

```cpp
auto scheduler = std::make_shared<jyppx::liuding::optim::StepLrScheduler>(
    optimizer,
    jyppx::liuding::optim::StepLrSchedulerOptions{
        .scheduler_id = "scheduler.epochs",
        .step_size = 10,
        .gamma = 0.1});

// 每个 epoch 完成后调用一次。
scheduler->step();
```

描述符记录算法、step mode、typed 选项、基础学习率、optimizer/group 身份及完整子树。恢复比较完整规范描述符；短指纹只用于诊断。一次 `step()` 先计算候选学习率与计数，验证完成后统一提交；失败不留下部分更新。

`skip_initial_decay` 默认为 `false`。显式设为 `true` 时，首次事件保留初始学习率，后续事件使用 PyTorch `StepLR(last_epoch=-1)` 的启动序列；例如 `step_size=2`、`gamma=0.5`、初始学习率 `0.2` 的前六次事件为 `0.2, 0.2, 0.1, 0.1, 0.05, 0.05`。图像配方的 `epoch_step` 使用此模式；通用调用方应按自己的事件定义选择。

## 保存与恢复

`LrSchedulerCheckpointParticipant` 的 participant schema 为3，描述符 schema 为1。`topology` 保存完整描述符和身份，`control` 保存状态树和当前组学习率。读取检查 manifest、版本、资源上限、计数、拓扑及完整学习率后提交；只接受当前格式和 exact 策略。

启动模式属于描述符的一部分。相同模式可在首次事件前后及首次衰减前后恢复完整计数和各参数组学习率；`skip_initial_decay` 不同的检查点被拒绝，目标计数与优化器学习率保持不变。

参数更新、调度和检查点操作由同一 owner 串行执行。Trainer 集成使用[SchedulerAdapter](scheduler_adapter.md)，组合方式见[Sequential/Chained](composite_lr_scheduler.md)，其他算法见[优化器与调度总览](optim.md)。
