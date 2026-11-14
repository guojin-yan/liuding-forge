# 组合学习率调度器

包含 `<jyppx/liuding/optim/composite_lr_scheduler.hpp>`，链接 `jyppx::liuding::optim`。

## Sequential

`SequentialLrScheduler` 独占有序子列表。milestone 表示此前所有子调度器累计接收的父事件数；例如 milestone=2 时，第一个子调度器处理事件1、2，第二个从事件3开始。

```cpp
std::vector<std::unique_ptr<jyppx::liuding::optim::LrScheduler>> children;
children.push_back(std::make_unique<jyppx::liuding::optim::StepLrScheduler>(
    optimizer, jyppx::liuding::optim::StepLrSchedulerOptions{
                   .scheduler_id = "scheduler.warm",
                   .step_size = 1,
                   .gamma = 0.9}));
children.push_back(std::make_unique<jyppx::liuding::optim::StepLrScheduler>(
    optimizer, jyppx::liuding::optim::StepLrSchedulerOptions{
                   .scheduler_id = "scheduler.decay",
                   .step_size = 10,
                   .gamma = 0.1}));

auto scheduler = std::make_shared<jyppx::liuding::optim::SequentialLrScheduler>(
    optimizer, std::move(children),
    jyppx::liuding::optim::SequentialLrSchedulerOptions{
        .scheduler_id = "scheduler.epochs", .milestones = {5}});
```

## Chained

`ChainedLrScheduler` 按序预览全部子调度器，每个子对象接收上一个产生的候选学习率。全部成功后根对象只提交一次学习率，再无异常地提交父子计数；任一子对象失败均保留优化器及整个树的原状态。

全部子对象必须共享同一个优化器对象、使用唯一 scheduler ID 并声明同一个 step mode；当前组合接口拒绝 metric 子调度器。Linear/Warmup/Constant 构造会应用初始因子，组合顺序及构造注意事项见[线性调度器](linear_lr_scheduler.md)。

## 描述符与检查点

父描述符保存有序子 ID 和完整子描述符；Sequential 另存 typed milestones。`LrSchedulerCheckpointParticipant` 使用 participant schema3，分别保存 schema1 描述符树、可变状态树和当前优化器学习率。

恢复先检查完整 manifest、描述符、子数量及顺序、状态树形状、资源边界和数值，再一次提交全部组学习率及预校验的计数；仅接受当前版本的 storage-neutral 格式。事件接入见[调度器适配](scheduler_adapter.md)。
