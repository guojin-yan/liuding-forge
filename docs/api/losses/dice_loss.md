# Dice loss

Status: `native target` in the default `jyppx::nn` component.

Header: `#include <jyppx/nn/loss/dice_loss.hpp>`

Link target: `jyppx::nn`

## Semantic matrix

| Concern | Contract |
|---|---|
| Binary | Logits and binary targets share shape `(N, ...)`; sigmoid creates one implicit class channel |
| Multiclass | Logits use `(N, C, ...)`; integral targets use `(N, ...)`; softmax creates mutually exclusive probabilities |
| Multilabel | Logits and binary targets share shape `(N, C, ...)`; sigmoid is applied independently per channel |
| Batch boundary | Dice statistics are computed independently per sample; batch items are never pooled together |
| Spatial boundary | Every dimension after the explicit class axis is flattened into the per-class spatial domain |
| Macro average | Compute one loss per selected class, then average selected classes within each sample |
| Micro average | Pool selected class intersections and cardinalities, then compute one loss per sample |
| Background | `include_background=false` removes explicit channel zero; binary mode rejects this option because it has no explicit class axis |
| Ignore | `ignore_index` is multiclass-only; ignored locations contribute neither probabilities nor targets, and receive zero gradient |
| Empty class | `include_empty_classes=false` excludes channels with zero target mass; a sample with no selected class contributes zero |
| Smooth | Finite non-negative value added to the numerator and denominator |
| Epsilon | Finite positive lower clamp for the complete denominator |
| Reduction | `none` returns `(N)`; `mean` and `sum` reduce only across batch items |
| Empty input | Any empty batch, class, or spatial dimension is rejected |
| Device | Input and target must use the same CPU or CUDA device |
| Errors | Invalid contracts raise `jyppx::InvalidArgument`; backend failures retain a nested cause |

## Formula

For each selected sample/class pair:

```text
dice = (2 * sum(probability * target) + smooth)
       / clamp_min(sum(probability) + sum(target) + smooth, epsilon)
loss = 1 - dice
```

Macro and micro aggregation use the boundaries in the semantic matrix. The
input is always interpreted as logits; callers must not pre-apply sigmoid or
softmax.

## API forms

```cpp
jyppx::nn::DiceLossOptions options;
options.mode = jyppx::nn::DiceMode::multiclass;
options.average = jyppx::nn::DiceAverage::macro;
options.ignore_index = -1;
options.include_background = false;

auto loss = jyppx::nn::functional::dice_loss(logits, target, options);

jyppx::nn::DiceLoss criterion(options);
auto module_loss = criterion.forward(logits, target);
```

## Verification

The focused target covers hand-computed binary, multiclass, and multilabel
references; macro/micro behavior; background and empty-class selection;
ignore masking and zero ignored gradients; all-ignored samples; every batch
reduction; non-contiguous spatial logits; backward finiteness; structured
configuration and tensor-contract failures; module parity; and installed
package consumption through `jyppx::nn`. A CUDA lane additionally compares
float16 multiclass output with a CPU float32 reference and verifies finite
backward gradients plus zero gradients at ignored locations.
