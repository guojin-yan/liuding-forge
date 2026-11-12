# 教程 3：优化器、调度与状态恢复

优化器使用稳定的参数组描述符，调度器使用明确的 epoch、成功更新或指标事件。恢复时需要一起保存模型、优化器算法状态及调度器状态；只记录当前学习率会遗漏动量、计数和调度历史。本教程提供可独立消费 SDK 的 C++ 示例，核对真实梯度更新，并在新进程恢复有动量的 SGD。

## 1. 构建与运行

在[基础训练教程](01_basic_training.md#1-构建与运行)配置好的 `examples/training` 构建目录中运行：

```sh
cmake --build out/build/training-examples --config Release --target jyppx_optimizer_scheduling --parallel 1
ctest --test-dir out/build/training-examples -C Release --output-on-failure -R '^OptimizerScheduling-.*-Example$' --parallel 1
```

完整源文件为 [optimizer_scheduling.cpp](../../examples/training/optimizer_scheduling.cpp)。切换到可执行文件所在目录，分别运行：

```sh
./liuding-forge-example-optimizer-scheduling epoch
./liuding-forge-example-optimizer-scheduling warmup
./liuding-forge-example-optimizer-scheduling plateau
```

每次执行四个更新，显示实际使用的学习率、两个参数和下一次学习率，最后输出 `steps=4 scheduler_steps=4 checked=true`。程序用 float64 CPU 上的二次损失 `weight²+bias²`，初值为 1 和 2，两个参数组基础学习率为 0.125 和 0.0625，动量均为 0.5。每个更新后的参数和两个组的学习率都与独立有理数推导值精确比较，便于直接观察调度发生的边界。

## 2. 选择优化器与参数组

| 类型 | 主要配置 |
| --- | --- |
| `SgdOptimizerAdapter` | `learning_rate`、`momentum`、`dampening`、`weight_decay`、`nesterov` |
| `AdamOptimizerAdapter` | `beta1`、`beta2`、`epsilon`、`amsgrad` |
| `AdamWOptimizerAdapter` | 解耦的 `weight_decay`，其余 Adam 类参数 |
| `RmspropOptimizerAdapter` | `alpha`、`centered`、`momentum` |
| `AdagradOptimizerAdapter` | `learning_rate_decay`、`initial_accumulator_value` |
| `NAdamOptimizerAdapter` | `momentum_decay`、耦合或解耦权重衰减、`maximize` |
| `AdadeltaOptimizerAdapter` | `rho`、`epsilon`、`weight_decay`、`maximize` |
| `LambOptimizerAdapter` | `beta1`、`beta2`、`epsilon`、`weight_decay`，固定参数范数上限 10 |

前五类更新委派给支持的 LibTorch C++ 实现，后三类由原生 C++ 内核实现。此示例只验证 SGD、CPU 和所列调度路径；其他算法、设备和真实任务按各自支持证据使用。

`Domain` 构造函数展示完整的两个 `SgdParameterGroup`，每组声明稳定的 `group_id`，每个 Tensor 绑定稳定 `parameter_id`。通常从模型适配器的 `state().named_parameters()` 取得模型拥有的 Tensor，再决定哪些参数需要不同学习率或衰减。构造时检查选项、ID 和参数拓扑；`learning_rates()` 按稳定组顺序返回当前值，`set_learning_rates()` 校验完整向量后统一提交。

## 3. 在正确事件后调度

| 模式 | 本例事件 | weight 组四次更新实际使用的学习率 |
| --- | --- | --- |
| `epoch` | 每轮只有一个更新，完成整轮后 StepLR 前进一步；`step_size=2`、`gamma=0.5` | 0.125、0.125、0.0625、0.0625 |
| `warmup` | 构造时应用 0.5 倍基础率，前两个成功更新后递增到基础率 | 0.0625、0.09375、0.125、0.125 |
| `plateau` | 更新后传入四个显式指标 1、1、2、0.5，`patience=0`、`factor=0.5` | 0.125、0.125、0.0625、0.03125 |

bias 组始终使用对应 weight 组的一半学习率。指标序列是演示调度的固定输入，不是本例损失或实际验证准确率；第一次指标建立 best，随后的两个未改善指标分别降低下一次学习率。

手动更新的核心代码直接来自可编译程序：

```cpp
    domain.optimizer->zero_grad();
    const auto loss = domain.holder->weight.square().sum() +
                      domain.holder->bias.square().sum();
    loss.backward();
    domain.optimizer->step();
    if (mode == "plateau") {
      domain.scheduler->step(validation_losses.at(i));
    } else {
      // This example has one successful update per epoch in epoch mode.
      domain.scheduler->step();
    }
```

实际多个 batch 的 epoch 调度应放在整轮完成后；batch 模式仅在优化器成功更新后推进，不能按读取批数或被 AMP 跳过的更新推进。`ReduceLrOnPlateauScheduler` 使用 `step(metric)`，要求有限指标；它拒绝无指标调用，其他非 metric 调度器拒绝指标重载。

接入 Trainer 时，用 `<jyppx/liuding/training/scheduler_adapter.hpp>` 中的 `training::SchedulerAdapter` 包装调度器，连同优化器及其 checkpoint participant 放入同一 `OptimizerExecutionBinding`。Trainer 按模式递交有序事件并保存 adapter 状态；不要再在外部重复调度。metric 模式由 `TrainerOptions::scheduler_metric` 选择指标，默认 `validation.loss`，对应验证 step 必须提供该指标。

## 4. 保存与新进程恢复

在程序目录运行两条命令，每一行启动独立进程：

```sh
./liuding-forge-example-optimizer-scheduling save ./optimizer-state/state.bin
./liuding-forge-example-optimizer-scheduling restore ./optimizer-state/state.bin
```

`save` 使用 epoch 模式执行前两次更新，把模型参数、有动量的 SGD 算法状态和调度器共同交给 `CheckpointCoordinator`，通过 `FileCheckpointStore` 发布一个完整文件，输出 `mode=save steps=2 scheduler_steps=2 checked=true`。`restore` 创建全新域，按 exact 策略恢复三名参与者，继续第三、四次更新；最终参数应精确为 weight 0.2001953125、bias 1.0684814453125，下一次学习率为 0.03125 和 0.015625，与连续 epoch 路线一致。

模型使用 `nn::ModelCheckpointParticipant`，SGD 使用 `optim::SgdOptimizerCheckpointParticipant`，调度器使用 `optim::LrSchedulerCheckpointParticipant`。优化器保存组/参数身份、不可变选项、完成步数、当前学习率和动量；调度器保存完整描述符、计数和当前学习率。构造新域时应先构造调度器，再加载协调检查点，使构造时预热等初始率调整不会覆盖已恢复学习率。

其他算法须使用与其匹配的参与者：Adam/AdamW 包含各参数步数、一二阶矩和可选 AMSGrad 最大值；RMSprop 包含平方平均及配置启用的动量、中心均值；Adagrad 包含累加器与独立步数；NAdam 还包含累计动量乘积；Adadelta 包含平方梯度和平方更新平均；LAMB 包含矩状态及独立步数。恢复要求相同稳定拓扑、算法选项和声明的运行库版本，不能用换一个参与者类型恢复另一算法。

本例没有数据游标、随机变换或 Trainer 控制状态，只能继续这条确定性二次损失更新流程。完整任务须同时保存数据、RNG、训练器和其他状态，见[检查点协调器](../api/checkpoint_coordinator.md)及[持久检查点说明](../api/durable_checkpoint.md)。协调器在修改前预检并捕获回滚状态，避免模型、优化器和调度器分开恢复后留下部分成功的域；应用需在保存/恢复期间暂停修改。

## 5. 其他调度方式

`StepLrScheduler`、`ExponentialLrScheduler`、`MultiStepLrScheduler` 和 `CosineAnnealingLrScheduler` 提供按 epoch 的周期衰减、指数衰减、里程碑和余弦轨迹。`LinearLrScheduler`、`WarmupLrScheduler`、`ConstantLrScheduler` 和 `PolynomialLrScheduler` 的构造与边界见[线性调度 API](../api/linear_lr_scheduler.md)。预热后接其他调度可使用 `SequentialLrScheduler`，组合树保持统一事务和描述符。

One-cycle 与 cyclic 按成功更新触发；callable 调度必须提供稳定 `callable_id`，恢复时检查该身份。`completed_steps()` 表示调度事件计数，不普遍等于 epoch 数或优化器步数；两者何时相等由调用事件决定。继续阅读[回调与检查点教程](04_callbacks.md)了解训练器管理的状态边界。
