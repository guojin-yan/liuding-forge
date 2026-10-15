# Contrastive loss

Status: `native target` in the default `jyppx::nn` component.

Header: `#include <jyppx/nn/loss/contrastive_loss.hpp>`

Link target: `jyppx::nn`

## Purpose

`jyppx::nn::functional::contrastive_loss` computes the standard pairwise
metric-learning objective. Matching pairs (`target = 1`) minimize squared
Euclidean distance. Non-matching pairs (`target = 0`) minimize the squared
hinge distance against a positive margin. The implementation is deliberately
narrow and native to JYPPX so that invalid pair metadata is rejected before
backend kernels run.

## Semantic matrix

| Concern | Contract |
|---|---|
| Features | `x1` and `x2` are exactly rank-2 tensors shaped `(N, D)` |
| Pair shape | `x1` and `x2` have identical shape; `target` is rank-1 shaped `(N)` |
| Target | Integral, boolean, or floating tensor whose values are exactly `0` or `1` |
| Margin | Finite and strictly positive |
| Reduction | `none`, `mean`, or `sum` |
| Device | All three tensors use the same CPU or CUDA device |
| Feature dtype | `x1` and `x2` use the same floating-point dtype |
| Target dtype | May differ from feature dtype; labels are converted only for arithmetic |
| Empty input | `(0, D)` pairs are accepted; `mean` follows LibTorch's empty reduction semantics |
| Non-contiguous input | Supported when LibTorch accepts the logical rank-2 tensors |
| Gradient | Gradients flow to both feature tensors through distance and hinge operations |
| State | `ContrastiveLoss` stores only validated margin and reduction configuration |
| Errors | Invalid contracts raise `jyppx::InvalidArgument`; backend failures retain a nested cause |

## API forms

```cpp
namespace F = jyppx::nn::functional;

auto loss = F::contrastive_loss(x1, x2, pair_labels, 1.0, "mean");

jyppx::nn::ContrastiveLoss criterion(1.0, "mean");
auto module_loss = criterion.forward(x1, x2, pair_labels);
```

Use LibTorch directly when a different distance function, per-pair margin,
soft labels, or a custom reduction is required. The JYPPX contract intentionally
keeps those extensions explicit rather than silently broadening validation.

## Verification

The focused target verifies hand-computed matching and non-matching values, all
reductions, autograd, module parity, invalid shape/device/dtype/configuration
cases, binary-label rejection, and CUDA label validation when a CUDA runtime is
available. The installed-package consumer also compiles and executes the
functional API through the exported `jyppx::nn` target.
