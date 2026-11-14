# Dice 损失

包含 `<jyppx/liuding/nn/loss/dice_loss.hpp>`，链接 `jyppx::liuding::nn`。输入始终是 logits，调用前不再应用 sigmoid 或 softmax。

| 项目 | 接口约束 |
| --- | --- |
| binary | logits 与二值标签同为 `(N,...)`，sigmoid 生成一个隐式类别通道 |
| multiclass | logits 为 `(N,C,...)`，整数类别标签为 `(N,...)`，使用 softmax |
| multilabel | logits 与二值标签同为 `(N,C,...)`，各通道独立 sigmoid |
| 统计范围 | 每个样本独立计算；类别轴后的空间维度展平 |
| macro | 先计算各选定类别损失，再在样本内取平均 |
| micro | 合并所选类别的交集和总量，再计算每个样本的损失 |
| 背景 | `include_background=false` 排除显式通道0；binary 不支持该设置 |
| 忽略 | `ignore_index` 仅用于 multiclass；忽略位置不计入统计且梯度为0 |
| 空类别 | `include_empty_classes=false` 排除目标总量为0的类别；无可选类别时该样本贡献0 |
| 数值参数 | `smooth` 有限非负，`epsilon` 有限且为正 |
| reduction | `none` 返回 `(N)`；`mean`、`sum` 仅归约样本维 |
| 输入 | 拒绝空 batch、类别或空间维；logits 和 target 必须位于相同 CPU/CUDA 设备 |
| 错误 | 无效契约抛 `InvalidArgument`，后端异常保留嵌套原因 |

对所选样本/类别计算：

```text
dice = (2 * sum(probability * target) + smooth)
       / clamp_min(sum(probability) + sum(target) + smooth, epsilon)
loss = 1 - dice
```

```cpp
jyppx::liuding::nn::DiceLossOptions options;
options.mode = jyppx::liuding::nn::DiceMode::multiclass;
options.average = jyppx::liuding::nn::DiceAverage::macro;
options.ignore_index = -1;
options.include_background = false;

auto loss = jyppx::liuding::nn::functional::dice_loss(logits, target, options);

jyppx::liuding::nn::DiceLoss criterion(options);
auto module_loss = criterion.forward(logits, target);
```
