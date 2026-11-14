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

## 保存与恢复

`LrSchedulerCheckpointParticipant` 的 participant schema 为3，描述符 schema 为1。`topology` 保存完整描述符和身份，`control` 保存状态树和当前组学习率。读取检查 manifest、版本、资源上限、计数、拓扑及完整学习率后提交；只接受当前格式和 exact 策略。

参数更新、调度和检查点操作由同一 owner 串行执行。Trainer 集成使用[SchedulerAdapter](scheduler_adapter.md)，组合方式见[Sequential/Chained](composite_lr_scheduler.md)，其他算法见[优化器与调度总览](optim.md)。
