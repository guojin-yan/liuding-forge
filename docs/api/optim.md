# 优化器与学习率调度

优化器和调度器位于 `jyppx::liuding::optim`，链接 `jyppx::liuding::optim`。完整训练和新进程恢复示例见[优化器教程](../tutorials/03_optimizers.md)。

## 优化器选择

| 优化器 | 数值实现 | 主要状态与语义 |
| --- | --- | --- |
| [SGD](sgd_adapter.md) | LibTorch SGD | momentum、dampening、Nesterov 与耦合衰减 |
| [Adam](adam_adapter.md) | LibTorch Adam | 每参数步数、一二阶矩及可选 AMSGrad |
| [AdamW](adamw_adapter.md) | LibTorch AdamW | 解耦权重衰减、一二阶矩及可选 AMSGrad |
| [RMSprop](rmsprop_adapter.md) | LibTorch RMSprop | 平方平均、可选 momentum 与 centered 均值 |
| [Adagrad](adagrad_adapter.md) | LibTorch Adagrad | 预先创建的累加器、独立参数步数与学习率衰减 |
| [NAdam](nadam_adapter.md) | 原生 Tensor 内核 | NAdam/NAdamW、累计动量乘积与 maximize |
| [Adadelta](adadelta_adapter.md) | 原生 Tensor 内核 | 平方梯度/更新平均与 maximize |
| [LAMB](lamb_adapter.md) | 原生 Tensor 内核 | bias correction、更新方向衰减与逐 Tensor trust ratio |

每个优化器使用稳定的 optimizer ID、有序参数组 ID 和全局唯一参数 ID。参数从模型拥有的 Tensor 绑定；支持稠密、非重叠 CPU/CUDA 浮点参数，梯度必须匹配参数的 shape、dtype 和设备。未定义梯度被跳过，稀疏或描述不匹配的梯度在更新前拒绝。

```cpp
#include <jyppx/liuding/optim/adamw_adapter.hpp>

auto optimizer = std::make_shared<jyppx::liuding::optim::AdamWOptimizerAdapter>(
    "optimizer.adamw",
    std::vector<jyppx::liuding::optim::AdamWParameterGroup>{
        {.group_id = "decay",
         .parameters = decay_parameter_bindings,
         .options = {.learning_rate = 1e-3, .weight_decay = 1e-2}},
        {.group_id = "no_decay",
         .parameters = no_decay_parameter_bindings,
         .options = {.learning_rate = 1e-3, .weight_decay = 0.0}}});
```

`set_learning_rates()` 校验并统一替换各组学习率；其他算法选项属于不可变拓扑。`SingleDeviceStrategy::prepare()` 可在首次更新前同步模型的 dtype/device 放置，开始更新后不得随意更换参数、组顺序或状态布局。优化器和参与者的调用需由训练 owner 串行执行。

## 学习率调度

| 调度器 | 配置与行为 |
| --- | --- |
| [StepLR](step_lr_scheduler.md) | `step_size`、`gamma`，按 epoch 周期衰减 |
| [Exponential/MultiStep/Cosine/WarmRestarts](epoch_lr_scheduler.md) | 指数、指定 epoch、余弦及带重启余弦 |
| [Linear/Warmup/Constant/Polynomial](linear_lr_scheduler.md) | 线性预热、固定因子与多项式衰减 |
| [Plateau/OneCycle/Cyclic/Lambda](event_lr_scheduler.md) | metric、batch 或显式 callable 驱动 |
| [Sequential/Chained](composite_lr_scheduler.md) | 按序选择或组合子调度器，统一提交学习率 |

调度器依赖 `OptimizerRateControl`。一次 step 先预览完整状态和各组学习率，校验成功后统一提交；失败保留原计数与学习率。epoch 调度在整轮完成后推进，batch 调度在成功更新后推进，metric 调度使用有限验证指标。

```cpp
#include <jyppx/liuding/optim/epoch_lr_scheduler.hpp>

auto scheduler = std::make_shared<jyppx::liuding::optim::ExponentialLrScheduler>(
    delegated_sgd,
    jyppx::liuding::optim::ExponentialLrSchedulerOptions{
        .scheduler_id = "scheduler.epochs", .gamma = 0.95});
scheduler->step();  // after one completed epoch
```

接入 Trainer 时，将调度器交给 [training::SchedulerAdapter](scheduler_adapter.md) 并绑定到对应的 `OptimizerExecutionBinding`；不要在外部重复 step。`cycle_momentum` 不受支持，因为当前优化器可变控制只包含学习率。

## 完整恢复

优化器使用各自的 `*OptimizerCheckpointParticipant`，保存参数/组拓扑、不可变选项、更新计数、当前学习率及算法状态。调度器使用 `LrSchedulerCheckpointParticipant`，保存完整描述符树、计数树和学习率。恢复先验证全部候选状态，再替换现存状态；要求匹配 schema、拓扑和声明的 LibTorch 版本。

将模型、优化器、调度器、数据、RNG 和训练控制状态一起登记到[检查点协调器](checkpoint_coordinator.md)。仅记录学习率或保存模型权重不足以继续原训练任务。
