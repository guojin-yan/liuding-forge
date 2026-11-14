# SGD 优化器

公共头文件为 `<jyppx/liuding/optim/sgd_adapter.hpp>`，类型为 `optim::SgdOptimizerAdapter`，链接 `jyppx::liuding::optim`。

数值更新委派给 `torch::optim::SGD`。支持 momentum、dampening、weight_decay 与 Nesterov，只有学习率可在构造后修改。

```cpp
auto optimizer = std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
    "optimizer.main",
    std::vector<jyppx::liuding::optim::SgdParameterGroup>{
        {.group_id = "backbone",
         .parameters = {{.parameter_id = "encoder.weight",
                         .parameter = encoder_weight}},
         .options = {.learning_rate = 1e-2,
                     .momentum = 0.9,
                     .weight_decay = 1e-4}}});

optimizer->zero_grad();
loss.backward();
optimizer->step();
```

参数绑定和调用生命周期遵循[优化器公共约定](optim.md#优化器选择)。构造时提供稳定 optimizer ID、组 ID 和唯一参数 ID；`step()` 先执行可选 closure，再检查完整梯度集合。未定义梯度跳过且保留对应参数步数；稀疏或 shape/dtype/device 不匹配的梯度在更新前拒绝。`completed_steps()` 只在更新成功后推进，`set_learning_rates()` 统一校验全部组后提交。

首次更新前可通过 `SingleDeviceStrategy::prepare()` 同步模型参数放置；参数身份、shape、stride、组顺序和选项保持固定。

## 检查点

`SgdOptimizerCheckpointParticipant` 使用 schema1，固定完整参数拓扑、算法选项、运行库版本，保存成功更新计数、各组当前学习率及可选 momentum。schema1 的 `topology` 固定组/参数身份、选项、dtype/device/shape/stride 与 LibTorch 版本，`control` 保存更新次数、学习率和 momentum 存在位，`momentum.NNNNNNNNNN` 保存每个参数的缓冲区；未初始化时为空。

恢复先验证 manifest、拓扑、字段大小及全部候选 Tensor，成功后才替换状态；失败保留原状态。格式为有界的 storage-neutral 字段，不接受 pickle、LibTorch archive 或其他优化器状态。保存和恢复期间暂停训练修改；完整任务恢复还需模型、数据、RNG 与调度器，见[协调器](checkpoint_coordinator.md)。
