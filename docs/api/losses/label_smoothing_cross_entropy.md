# Label-smoothed cross entropy

Status: `delegated adapter` in the default `jyppx::liuding::nn` component.

Header: `#include <jyppx/liuding/nn/loss/label_smoothing_cross_entropy.hpp>`

Link target: `jyppx::liuding::nn`

## Purpose

`jyppx::liuding::nn::functional::label_smoothing_cross_entropy` gives training code a
stable JYPPX entry point while delegating all tensor mathematics to
`torch::nn::functional::cross_entropy`. JYPPX does not maintain a duplicate
kernel or a separately evolving formula.

For `C` classes and smoothing value `epsilon`, the class-index target is mixed
with the uniform class distribution using LibTorch's `label_smoothing` option.
`epsilon = 0` is ordinary cross entropy; `epsilon = 1` uses only the uniform
distribution.

## Semantic matrix

| Concern | Contract |
|---|---|
| Classification | Delegated adapter; numerical authority is the linked LibTorch build |
| Input | Floating-point logits shaped `(C)`, `(N, C)`, or `(N, C, d1, ..., dK)` |
| Target | Class indices shaped `()`, `(N)`, or `(N, d1, ..., dK)` |
| Smoothing | Finite value in the closed interval `[0, 1]` |
| Reduction | `none`, `mean`, or `sum` |
| Ignore index | LibTorch default `-100`; not configurable in this focused adapter |
| Class weights | Not exposed; use LibTorch directly when weights are required |
| Device | CPU and CUDA when supported by the linked LibTorch build |
| Dtype | Follows LibTorch cross entropy; float32/float64 are baseline, CUDA float16 is tested when available |
| Empty input | Delegated to LibTorch; `none` preserves an empty target shape |
| Non-contiguous input | Supported when LibTorch accepts the same logical shape |
| Gradient | Autograd graph and backward values are those of LibTorch cross entropy |
| State | None; the module stores only smoothing and reduction configuration |
| Errors | JYPPX validates configuration and defined tensors; backend failures retain the nested LibTorch cause |

## API forms

```cpp
namespace F = jyppx::liuding::nn::functional;

auto loss = F::label_smoothing_cross_entropy(logits, target, 0.1, "mean");

jyppx::liuding::nn::LabelSmoothingCrossEntropy criterion(0.1, "mean");
auto module_loss = criterion.forward(logits, target);
```

Use LibTorch's cross-entropy API directly when class weights, a custom ignore
index, or probability targets are part of the training contract. The JYPPX
adapter intentionally stays narrow instead of partially duplicating those
options.

## Verification

The target test suite compares forward and backward results with LibTorch for
all reductions, unbatched and spatial inputs, non-contiguous tensors, boundary
smoothing values, empty unreduced input, invalid configuration, and CUDA
float16 when a CUDA runtime is available. The installed-package smoke test
also compiles and runs a consumer linked only to exported component targets.
