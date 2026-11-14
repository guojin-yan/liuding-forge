# Adagrad 优化器

公共头文件为 `<jyppx/liuding/optim/adagrad_adapter.hpp>`，类型为 `optim::AdagradOptimizerAdapter`，链接 `jyppx::liuding::optim`。

数值更新委派给 `torch::optim::Adagrad`。learning_rate_decay、weight_decay、initial_accumulator_value 与 epsilon 不可变。构造时即为每个参数创建累加器；首次更新前迁移 dtype/device 时重建零步累加器。

```cpp
auto optimizer = std::make_shared<jyppx::liuding::optim::AdagradOptimizerAdapter>(
    "optimizer.adagrad",
    std::vector<jyppx::liuding::optim::AdagradParameterGroup>{
        {.group_id = "embeddings",
         .parameters = embedding_parameter_bindings,
         .options = {.learning_rate = 1e-2,
                     .learning_rate_decay = 1e-3,
                     .initial_accumulator_value = 0.1}}});
```

参数绑定和调用生命周期遵循[优化器公共约定](optim.md#优化器选择)。构造时提供稳定 optimizer ID、组 ID 和唯一参数 ID；`step()` 先执行可选 closure，再检查完整梯度集合。未定义梯度跳过且保留对应参数步数；稀疏或 shape/dtype/device 不匹配的梯度在更新前拒绝。`completed_steps()` 只在更新成功后推进，`set_learning_rates()` 统一校验全部组后提交。

Adagrad 的零步累加器随首次放置同步重建；成功更新后禁止重新同步放置。

## 检查点

`AdagradOptimizerCheckpointParticipant` 使用 schema1，固定完整参数拓扑、算法选项、运行库版本，保存成功更新计数、各组当前学习率及每个参数的独立步数和累加器，包含零步的预初始化状态。

恢复先验证 manifest、拓扑、字段大小及全部候选 Tensor，成功后才替换状态；失败保留原状态。格式为有界的 storage-neutral 字段，不接受 pickle、LibTorch archive 或其他优化器状态。保存和恢复期间暂停训练修改；完整任务恢复还需模型、数据、RNG 与调度器，见[协调器](checkpoint_coordinator.md)。
