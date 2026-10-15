# Focal loss

Status: `native target` in the default `jyppx::nn` component.

Header: `#include <jyppx/nn/loss/focal_loss.hpp>`

Link target: `jyppx::nn`

## Semantic matrix

| Concern | Contract |
|---|---|
| Binary input | Floating logits and target with exactly identical shape |
| Binary target | Integral, boolean, or floating values exactly in `{0, 1}` |
| Multiclass input | Floating logits shaped `(N, C, ...)` with `C > 0` |
| Multiclass target | Integral class indices shaped `(N, ...)`, with every spatial dimension matching |
| Alpha | Finite value in the closed interval `[0, 1]`; weights the positive binary class and multiclass objective |
| Gamma | Finite non-negative focusing exponent; `gamma = 0` recovers weighted cross entropy |
| Reduction | `none`, `mean`, or `sum` |
| Device | Input and target use the same CPU or CUDA device |
| Gradient | Autograd flows to floating logits through stable sigmoid/cross-entropy tensor ops |
| Errors | Invalid contracts raise `jyppx::InvalidArgument`; backend failures retain a nested cause |

## API forms

```cpp
namespace F = jyppx::nn::functional;

auto loss = F::focal_loss(logits, target, 0.25, 2.0, "mean");

jyppx::nn::FocalLoss criterion(0.25, 2.0);
auto module_loss = criterion.forward(logits, target);
```

The implementation is native because LibTorch has no equivalent general focal
loss. Use direct LibTorch cross-entropy or binary BCE when focal weighting is
not part of the training objective.

## Verification

The focused target verifies the gamma-zero cross-entropy identity, hand-computed
binary weighting, all reductions, backward finiteness, invalid alpha/gamma,
shape/device/dtype/class-index contracts, module parity, and installed-package
consumer compilation through the exported `jyppx::nn` target.
