# 教程 5：混合精度与梯度缩放

六丁炼算提供显式 autocast、梯度缩放和 Trainer 精度契约。使用 `TrainerOptions::precision` 选择精度；CUDA float16 训练还需绑定启用的 `GradScaler` 及其检查点参与者，不使用旧的 `use_amp` 开关。

## 自动类型转换

`AutocastGuard` 限定 autocast 的作用域，析构时恢复此前的启用标志和 dtype，支持嵌套。进入作用域前确认实际设备和 dtype 受支持；前向与 loss 放在作用域内，反向在作用域结束后执行。

```cpp
jyppx::liuding::training::amp::AutocastGuard autocast{
    c10::DeviceType::CUDA, at::kHalf};
auto prediction = model.forward(input);
auto loss = torch::mse_loss(prediction, target);
```

实际混合精度训练须在目标模型和设备上比较 FP32 与混合精度的收敛结果，并记录设备、dtype、LibTorch 版本及缩放配置。下文的 CPU 标量示例用于学习缩放控制流程，不代表 CUDA/autocast 或完整模型训练已通过验收。

## 单优化器更新

`optimizer` 是 `std::shared_ptr<jyppx::liuding::optim::OptimizerAdapter>`，例如 `SgdOptimizerAdapter`，通过 `->` 调用优化器方法。`compute_loss()` 代表当前批次的前向和 loss 计算；每个更新周期开始先清梯度，缩放后的 loss 必须保留 autograd 图。`scaler` 是 `jyppx::liuding::training::amp::GradScaler` 实例。

```cpp
optimizer->zero_grad();
auto loss = compute_loss();
auto scaled = scaler.scale(loss);
scaled.backward();
scaler.unscale(optimizer);
const bool applied = scaler.step_allowed(optimizer);
if (applied) {
  optimizer->step();
}
scaler.mark_step(optimizer, applied);
scaler.update();
```

需要检查或裁剪梯度时，放在 `unscale()` 后、`step()` 前，操作真实梯度；若自定义操作产生新非有限值，不能沿用此前的有限性结论。`mark_step()` 报告实际执行结果，每个完整周期只调用一次 `update()`。溢出周期跳过参数更新并降低 scale；有限周期按配置累积增长计数。

## 多优化器统一决策

一个逻辑更新涉及多个优化器时，先对全部优化器 `unscale()`，再用 `cycle_allowed()` 统一决定。不能按单个优化器的 `step_allowed()` 分别更新，否则后续参与者溢出时可能只更新部分参数。

```cpp
encoder_optimizer->zero_grad();
head_optimizer->zero_grad();
auto loss = compute_loss();
auto scaled = scaler.scale(loss);
scaled.backward();
scaler.unscale(encoder_optimizer);
scaler.unscale(head_optimizer);
const bool apply_group = scaler.cycle_allowed();
if (apply_group) {
  encoder_optimizer->step();
  head_optimizer->step();
}
scaler.mark_step(encoder_optimizer, apply_group);
scaler.mark_step(head_optimizer, apply_group);
scaler.update();
```

任一参与者梯度非有限时整组跳过，每个参与者报告相同结果。整组决策针对梯度溢出；优化器自身的 `step()` 若抛出异常，不提供跨优化器事务回滚，应停止并按应用的检查点策略恢复。批次调度器和 Trainer 全局更新计数只在整组成功更新后推进；scaler 在成功或溢出跳过后均需结束周期。

## 编译并运行完整示例

[原生 C++ 示例](../../examples/training/grad_scaler.cpp) 包含优化器构造、真实 autograd 反向、结果断言和四种独立运行模式。它使用 CPU float64 标量和学习率 0.125，正常周期 scale 从 8 增至 16，溢出周期降至 4；双优化器模式可观察第二个梯度溢出时第一个有限梯度也跳过更新。

从仓库根目录使用已有 SDK 配置示例；`out/install/linux-release` 应指向已安装的兼容 SDK，保留构建缓存中的 LibTorch 和编译器路径：

```sh
cmake -S examples/training -B out/build/training-examples -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/out/install/linux-release"
cmake --build out/build/training-examples --target jyppx_grad_scaler --parallel 1
ctest --test-dir out/build/training-examples --output-on-failure -R '^GradScaler-.*-Example$' --parallel 1
```

四种模式为 `single`、`overflow`、`group`、`group-overflow`。正常单优化器输出 `parameter=0.75 applied=1 scale=16 steps=1`；正常双优化器输出 `encoder=0.75 head=1.5`；溢出模式保持初值和零更新计数。Windows 多配置构建还需 `--config Release` 与 CTest 的 `-C Release`，运行时库配置见[构建指南](../../CONTRIBUTING.md)。

## 检查点与梯度累积

将 `GradScalerCheckpointParticipant` 与模型、全部优化器、数据游标、RNG 和 Trainer 状态登记到同一[协调恢复域](../api/checkpoint_coordinator.md)。只有 `checkpoint_ready()` 为真时才能捕获或恢复 scaler 状态；仅恢复 scale、遗漏优化器或循环计数不能得到精确续训。

自定义含梯度累积的策略应先按累积窗口归一化 loss，再缩放并逐微批次反向；到优化器边界后对所有参与者取消缩放、检查与裁剪梯度，统一更新或跳过，逐个 `mark_step()`，最后 `update()` 一次。完整边界提交后才保存检查点。
