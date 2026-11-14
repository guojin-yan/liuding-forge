# Focal 损失

包含 `<jyppx/liuding/nn/loss/focal_loss.hpp>`，链接 `jyppx::liuding::nn`。该原生实现使用稳定的 sigmoid/交叉熵 Tensor 运算，梯度传向浮点 logits。

| 项目 | 接口约束 |
| --- | --- |
| binary | 浮点 logits 与标签 shape 完全相同；整数、布尔或浮点标签值严格为0或1 |
| multiclass | logits 为 `(N,C,...)`，`C>0`；整数类别索引为 `(N,...)`，所有空间维一致 |
| alpha | 有限且在 `[0,1]` 内；用于二值正类权重及多类目标权重 |
| gamma | 有限非负；0对应带权交叉熵 |
| reduction | `none`、`mean` 或 `sum` |
| 设备 | input 与 target 位于相同 CPU/CUDA 设备 |
| 错误 | 无效配置、shape、dtype 或类别索引抛 `InvalidArgument`，后端异常保留嵌套原因 |

```cpp
namespace F = jyppx::liuding::nn::functional;

auto loss = F::focal_loss(logits, target, 0.25, 2.0, "mean");

jyppx::liuding::nn::FocalLoss criterion(0.25, 2.0);
auto module_loss = criterion.forward(logits, target);
```

不需要 focal 权重时，可直接使用 LibTorch 的交叉熵或二值 BCE。
