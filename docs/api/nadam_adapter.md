# NAdam 优化器

公共头文件为 `<jyppx/liuding/optim/nadam_adapter.hpp>`，类型为 `optim::NAdamOptimizerAdapter`，链接 `jyppx::liuding::optim`。

使用原生 Tensor 内核，支持 NAdam 与解耦衰减 NAdamW。beta、epsilon、weight_decay、momentum_decay、decoupled_weight_decay 和 maximize 不可变；maximize 在正则化和矩更新前反转梯度。

```cpp
auto optimizer = std::make_shared<jyppx::liuding::optim::NAdamOptimizerAdapter>(
    "optimizer.nadam",
    std::vector<jyppx::liuding::optim::NAdamParameterGroup>{
        {.group_id = "decay",
         .parameters = decay_parameter_bindings,
         .options = {.learning_rate = 2e-3,
                     .weight_decay = 1e-2,
                     .momentum_decay = 4e-3,
                     .decoupled_weight_decay = true}},
        {.group_id = "no_decay",
         .parameters = no_decay_parameter_bindings,
         .options = {.learning_rate = 2e-3}}});
```

参数绑定和调用生命周期遵循[优化器公共约定](optim.md#优化器选择)。构造时提供稳定 optimizer ID、组 ID 和唯一参数 ID；`step()` 先执行可选 closure，再检查完整梯度集合。未定义梯度跳过且保留对应参数步数；稀疏或 shape/dtype/device 不匹配的梯度在更新前拒绝。`completed_steps()` 只在更新成功后推进，`set_learning_rates()` 统一校验全部组后提交。

首次更新前可通过 `SingleDeviceStrategy::prepare()` 同步模型参数放置；参数身份、shape、stride、组顺序和选项保持固定。

## 检查点

`NAdamOptimizerCheckpointParticipant` 使用 schema1，固定完整参数拓扑、算法选项、运行库版本，保存成功更新计数、各组当前学习率及各参数状态存在位、独立步数、累计动量乘积及一二阶矩。

恢复先验证 manifest、拓扑、字段大小及全部候选 Tensor，成功后才替换状态；失败保留原状态。格式为有界的 storage-neutral 字段，不接受 pickle、LibTorch archive 或其他优化器状态。保存和恢复期间暂停训练修改；完整任务恢复还需模型、数据、RNG 与调度器，见[协调器](checkpoint_coordinator.md)。
