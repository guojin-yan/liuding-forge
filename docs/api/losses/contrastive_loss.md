# 对比损失

包含 `<jyppx/liuding/nn/loss/contrastive_loss.hpp>`，链接 `jyppx::liuding::nn`。`nn::functional::contrastive_loss` 实现成对度量学习目标：标签为1时最小化平方欧氏距离，标签为0时最小化相对正 margin 的平方 hinge 距离。

| 项目 | 接口约束 |
| --- | --- |
| 特征 | `x1`、`x2` 均为 `(N,D)`，shape、浮点 dtype 和 CPU/CUDA 设备完全一致 |
| 标签 | 同设备的 `(N)` Tensor；整数、布尔或浮点值必须严格为0或1 |
| margin | 有限正数 |
| reduction | `none`、`mean` 或 `sum` |
| 空输入 | 接受 `(0,D)`；`mean` 遵循 LibTorch 空归约语义 |
| 布局与梯度 | 接受后端支持的非连续二维输入，梯度传向两个特征 Tensor |
| 状态与错误 | 模块仅存配置；无效输入抛 `InvalidArgument`，后端异常保留嵌套原因 |

```cpp
namespace F = jyppx::liuding::nn::functional;

auto loss = F::contrastive_loss(x1, x2, pair_labels, 1.0, "mean");

jyppx::liuding::nn::ContrastiveLoss criterion(1.0, "mean");
auto module_loss = criterion.forward(x1, x2, pair_labels);
```

自定义距离、逐对 margin、软标签或其他归约方式需由调用方显式实现。
