# 标签平滑交叉熵

包含 `<jyppx/liuding/nn/loss/label_smoothing_cross_entropy.hpp>`，链接 `jyppx::liuding::nn`。`nn::functional::label_smoothing_cross_entropy` 将 Tensor 数学运算委派给 `torch::nn::functional::cross_entropy`，数值和梯度由所链接的 LibTorch 决定。

类别索引目标按 `label_smoothing=epsilon` 与均匀类别分布混合。`epsilon=0` 是普通交叉熵，`epsilon=1` 仅使用均匀分布。

| 项目 | 接口约束 |
| --- | --- |
| logits | 浮点 `(C)`、`(N,C)` 或 `(N,C,d1,...,dK)` |
| target | 类别索引 `()`、`(N)` 或 `(N,d1,...,dK)` |
| smoothing | 有限且在 `[0,1]` 内 |
| reduction | `none`、`mean` 或 `sum` |
| ignore_index | 固定为 LibTorch 默认值 `-100` |
| 类别权重 | 此适配器不暴露该选项 |
| 设备与 dtype | 遵循所链接 LibTorch 的交叉熵实现及实际 CPU/CUDA 能力 |
| 空输入与布局 | 由 LibTorch 决定；`none` 保留空目标形状，可接受后端支持的非连续输入 |
| 状态与错误 | 模块仅存配置；库校验配置及已定义 Tensor，后端异常保留嵌套原因 |

```cpp
namespace F = jyppx::liuding::nn::functional;

auto loss = F::label_smoothing_cross_entropy(logits, target, 0.1, "mean");

jyppx::liuding::nn::LabelSmoothingCrossEntropy criterion(0.1, "mean");
auto module_loss = criterion.forward(logits, target);
```

需要类别权重、自定义 ignore index 或概率目标时，直接使用 LibTorch 交叉熵接口。
