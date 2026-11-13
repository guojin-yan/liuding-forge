# 教程 1：基础训练

六丁炼算把模型调用、数据读取、目标提取和损失计算显式绑定到 `Trainer`，由训练器负责反向传播、更新、验证和调度事件。本教程运行一个完整的原生 C++ 线性回归示例，确认接入和生命周期；两轮微型数据训练不代表实际任务收敛资格。

## 1. 构建与运行

准备与编译器和 LibTorch 匹配的已安装 SDK，组合要求见[构建说明](../../CONTRIBUTING.md#构建与验证)。在仓库根目录运行，将依赖路径替换为实际位置；已有兼容的 `examples/training` cache 时继续复用：

```sh
cmake -S examples/training -B out/build/training-examples -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/sdk;/path/to/libtorch"
cmake --build out/build/training-examples --config Release --target jyppx_basic_trainer --parallel 1
ctest --test-dir out/build/training-examples -C Release --output-on-failure -R '^BasicTrainer-Example$' --parallel 1
```

完整程序为 [examples/training/basic_trainer.cpp](../../examples/training/basic_trainer.cpp)，只链接 `training` 及其必需依赖；同目录其他示例还使用 `interop`。Linux 程序位于上述构建目录，Windows 多配置位于其 `Release/` 子目录。将匹配的 LibTorch 运行库加入系统搜索路径后，直接运行 `liuding-forge-example-basic-trainer`，输出以 `status=1 global_step=4 learning_rate=0.01 epochs=2 scheduler_steps=4` 开头，并显示训练及验证损失。

## 2. 适配模型并绑定参数

示例创建输入 `[8,1]` 与目标 `2*x+1`，使用 `torch::nn::Linear(1,1)`，通过 `nn::adapt_tensor_holder` 包装为 `nn::ModelAdapter`。适配器保留底层模块，用 `state().named_parameters()` 取得模型拥有的 Tensor；为每个参数设置稳定 `parameter_id` 后，组成 `SgdParameterGroup` 并构造 `SgdOptimizerAdapter`。

参数组 ID、参数 ID 和顺序属于优化器恢复兼容性。应先建立这些绑定，再由 `SingleDeviceStrategy` 负责放置与训练；开始训练后不要再次自行搬迁模型。生产数据的 dtype、rank 和形状可由明确的模型调用 schema 约束，见[模型适配 API](../api/model_adapter.md)。

## 3. 组批与定义任务

`make_pipeline()` 用 `SequentialSampler` 和 batch size 4 组成训练数据流。factory 根据样本 ID 从输入和标签 Tensor 取行，返回二元 tuple。训练和验证使用两个独立的 pipeline ID 及游标，即使它们引用相同 Tensor，也不共享管线实例；实际任务还须使用独立验证数据，不能将本例的复用视为泛化评估。

`training_batch_mapper` 把 tuple 第一项变为 `nn::Invocation`，`training_target_mapper` 提取第二项，`training_step` 调用模型并返回标量 MSE 与真实样本数。验证复用相同的任务映射，由训练器切换评估与无梯度上下文。训练损失必须有限、可微且为标量，`sample_count` 必须等于已经提交的样本 ID 数量。worker、明确 schema 与数据游标恢复见[数据加载教程](02_data_loading.md)。

## 4. 优化器、调度器与必需状态

`make_request()` 绑定一个 SGD 优化器、与它匹配的 `SgdOptimizerCheckpointParticipant`、模型参与者和 CPU RNG 参与者。这些是训练域完整恢复所需的基础状态；程序没有写出检查点文件，不能把注册参与者视为已经完成了持久保存。

预热调度器的基础学习率为 0.01、`start_factor=0.2`、`warmup_iters=2`，事件模式是 `batch`。构造时首个更新使用 0.002，第一次成功更新后调到 0.006，第二次成功更新后达到 0.01；`training::SchedulerAdapter` 随 `OptimizerExecutionBinding` 交给 Trainer，后续事件由 Trainer 递交，不再由应用重复调用 `step()`。完整的手动调度、参数组与持久恢复流程见[优化器教程](03_optimizers.md)。

## 5. 准备、执行与关闭

程序创建最大两轮的 Trainer，并保留调度器以便核对实际推进次数，然后执行：

```cpp
    auto request = make_request(model, optimizer, train, validation);
    const auto scheduler = request.optimizers.front().scheduler;
    trainer.prepare(std::move(request));
    const auto result = trainer.fit();
```

每轮两个批次，共四次成功更新。示例断言 `FitStatus::completed`、两轮、`global_step=4`、优化器和调度器均推进四次、最终学习率为 0.01，并确认训练和验证损失有限，随后调用 `trainer.close()` 释放管线及策略资源。

`FitResult` 还可表示停止、取消或临时暂停；最终训练/验证指标属于最近完整阶段。异常会使 Trainer 进入 failed，此时应修复原因后在新训练域恢复检查点，避免继续使用可能已修改的后端状态。不要在 step 中再次执行 backward 或 optimizer step；它们由策略拥有。检查点协调与不可变回调见[教程 4](04_callbacks.md)。
