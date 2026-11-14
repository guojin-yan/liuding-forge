# Adadelta 优化器

公共头文件为 `<jyppx/liuding/optim/adadelta_adapter.hpp>`，类型为 `optim::AdadeltaOptimizerAdapter`，链接 `jyppx::liuding::optim`。

使用原生 Tensor 内核。rho 在 `[0,1]`，epsilon 为正，学习率与 weight_decay 非负；maximize 先反转目标梯度，耦合衰减在两项滑动平均之前加入梯度。

```cpp
auto optimizer = std::make_shared<jyppx::liuding::optim::AdadeltaOptimizerAdapter>(
    "optimizer.adadelta",
    std::vector<jyppx::liuding::optim::AdadeltaParameterGroup>{
        {.group_id = "main",
         .parameters = parameter_bindings,
         .options = {.learning_rate = 1.0,
                     .rho = 0.9,
                     .epsilon = 1e-6,
                     .weight_decay = 0.0}}});
```

参数绑定和调用生命周期遵循[优化器公共约定](optim.md#优化器选择)。构造时提供稳定 optimizer ID、组 ID 和唯一参数 ID；`step()` 先执行可选 closure，再检查完整梯度集合。未定义梯度跳过且保留对应参数步数；稀疏或 shape/dtype/device 不匹配的梯度在更新前拒绝。`completed_steps()` 只在更新成功后推进，`set_learning_rates()` 统一校验全部组后提交。

首次更新前可通过 `SingleDeviceStrategy::prepare()` 同步模型参数放置；参数身份、shape、stride、组顺序和选项保持固定。

## 检查点

`AdadeltaOptimizerCheckpointParticipant` 使用 schema1，固定完整参数拓扑、算法选项、运行库版本，保存成功更新计数、各组当前学习率及各参数状态存在位、独立步数、`square_avg` 和 `acc_delta`。

恢复先验证 manifest、拓扑、字段大小及全部候选 Tensor，成功后才替换状态；失败保留原状态。格式为有界的 storage-neutral 字段，不接受 pickle、LibTorch archive 或其他优化器状态。保存和恢复期间暂停训练修改；完整任务恢复还需模型、数据、RNG 与调度器，见[协调器](checkpoint_coordinator.md)。
