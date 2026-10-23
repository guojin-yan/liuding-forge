# API: Optimizers and Schedulers

All types live in the `jyppx::liuding::optim` namespace.

The migrated optimizer boundary includes independently exported
[`SgdOptimizerAdapter`](sgd_adapter.md) and
[`AdamOptimizerAdapter`](adam_adapter.md), plus
[`AdamWOptimizerAdapter`](adamw_adapter.md),
[`RmspropOptimizerAdapter`](rmsprop_adapter.md), and
[`AdagradOptimizerAdapter`](adagrad_adapter.md). They delegate numerical
updates to LibTorch. The current-only
[`NAdamOptimizerAdapter`](nadam_adapter.md) and
[`AdadeltaOptimizerAdapter`](adadelta_adapter.md), together with the value-added
[`LambOptimizerAdapter`](lamb_adapter.md), supply the same stable identity and
storage-neutral recovery contract around JYPPX-native kernels because the
supported LibTorch C++ candidates expose none of these algorithms.
The migrated scheduler boundary includes descriptor-first
[`StepLrScheduler`](step_lr_scheduler.md) plus transactional
[`SequentialLrScheduler` and `ChainedLrScheduler`](composite_lr_scheduler.md)
and the [epoch scheduler family](epoch_lr_scheduler.md), including the
[linear-family schedulers](linear_lr_scheduler.md) and
[event-driven schedulers](event_lr_scheduler.md).
They record typed semantics and stable optimizer/group identity and provide
exact storage-neutral recovery.
The old mutable-map optimizer base and all implementations that depended on it
have been removed. There is no legacy forwarding header or state conversion;
all supported optimizers use the typed current contract below.

```cpp
#include <jyppx/liuding/optim/adamw_adapter.hpp>

auto optimizer = std::make_shared<jyppx::liuding::optim::AdamWOptimizerAdapter>(
    "optimizer.adamw",
    std::vector<jyppx::liuding::optim::AdamWParameterGroup>{
        {.group_id = "decay",
         .parameters = decay_parameter_bindings,
         .options = {.learning_rate = 1e-3, .weight_decay = 1e-2}},
        {.group_id = "no_decay",
         .parameters = no_decay_parameter_bindings,
         .options = {.learning_rate = 1e-3, .weight_decay = 0.0}}});
```

Notes:

- **AdamW** applies decoupled weight decay (`param *= 1 - lr * wd`) like
  `torch.optim.AdamW`; the delegated adapter owns this behavior directly.
- **RMSprop momentum** follows the `torch::optim` C++ semantics: the buffer
  accumulates the normalized gradient `grad / avg` and the step is
  `-lr * buffer`. The delegated adapter owns that exact LibTorch behavior.
- **Adagrad** eagerly initializes each sum accumulator, preserves independent
  per-parameter step counters when gradients are absent, and checkpoints both
  values exactly.
- **NAdam** follows the documented momentum schedule, including `mu_next` and
  a float32-compatible cumulative momentum product. It supports coupled NAdam,
  decoupled NAdamW, and maximize mode with exact per-parameter recovery.
- **Adadelta** follows the documented two-accumulator update, coupled weight
  decay, and maximize semantics. Lazy per-parameter steps, squared-gradient
  averages, and accumulated-delta averages are recovered exactly.
- **LAMB** applies bias-corrected moments and update-direction weight decay
  before one tensor-level trust ratio. Its weight-norm cap, zero-norm fallback,
  lazy per-parameter steps, and two moment tensors are recovered exactly.
- Delegated SGD, Adam, AdamW, RMSprop, and Adagrad, plus native NAdam,
  Adadelta, and LAMB, use `set_learning_rates()` as their only mutable
  hyperparameter operation.

## Learning-rate schedulers

The migrated `jyppx::liuding::optim::LrScheduler` hierarchy depends only on
`OptimizerRateControl`. Its `step()` path previews and validates a complete
state/rate transition before one atomic optimizer update. The generic
`LrSchedulerCheckpointParticipant` persists a typed descriptor tree, complete
counter tree and all current group rates.

```cpp
#include <jyppx/liuding/optim/epoch_lr_scheduler.hpp>

auto scheduler = std::make_shared<jyppx::liuding::optim::ExponentialLrScheduler>(
    delegated_sgd,
    jyppx::liuding::optim::ExponentialLrSchedulerOptions{
        .scheduler_id = "scheduler.epochs", .gamma = 0.95});
scheduler->step();  // after one completed epoch
```

| Migrated scheduler | Typed options | Behavior |
|--------------------|---------------|----------|
| `StepLrScheduler` | `step_size`, `gamma` | periodic multiplicative decay |
| `ExponentialLrScheduler` | `gamma` | multiplies current rates every epoch |
| `MultiStepLrScheduler` | `milestones`, `gamma` | decays at positive strictly ordered epochs |
| `CosineAnnealingLrScheduler` | `t_max`, `eta_min` | chainable recursive cosine trajectory |
| `CosineAnnealingWarmRestartsLrScheduler` | `t_0`, `t_mult`, `eta_min` | cosine cycles with exact growing restart state |
| `LinearLrScheduler` | `start_factor`, `end_factor`, `total_iters` | construction-time factor interpolation |
| `WarmupLrScheduler` | `start_factor`, `warmup_iters` | linear warmup to captured base rates |
| `ConstantLrScheduler` | `factor`, `total_iters` | constant factor prefix then base rates |
| `PolynomialLrScheduler` | `total_iters`, `power` | closed-form decay toward zero |
| `ReduceLrOnPlateauScheduler` | metric direction, patience, cooldown, thresholds | metric-triggered per-group decay |
| `OneCycleLrScheduler` | maxima, total steps, phase geometry | bounded batch-level one-cycle schedule |
| `CyclicLrScheduler` | base/max rates, half-cycle sizes, scale profile | repeating batch-level triangular cycles |
| `LambdaLrScheduler` | stable callable ID and functions | identity-bearing user-defined epoch factors |
| `SequentialLrScheduler` | children, `milestones` | selects one ordered child per event |
| `ChainedLrScheduler` | ordered children | previews every child and commits once |

The obsolete scheduler headers and implementations have been removed. Their
direct descriptor-first replacements do not accept old map checkpoints.
Momentum cycling is explicitly rejected until optimizer hyperparameters have
an atomic control and participant contract.

```cpp
#include <jyppx/liuding/optim/event_lr_scheduler.hpp>

auto plateau = std::make_shared<jyppx::liuding::optim::ReduceLrOnPlateauScheduler>(
    delegated_sgd,
    jyppx::liuding::optim::ReduceLrOnPlateauSchedulerOptions{
        .factor = 0.5,
        .patience = 2,
        .min_learning_rates = {1e-6}});
// after each validation:
plateau->step(val_loss);
```

The migrated plateau scheduler exposes `has_best()`, `best()`,
`bad_observations()`, and `cooldown_remaining()`. Legacy `state_dict()` maps
are not accepted. The checkpoint participant only accepts the current
descriptor/control schema and restores optimizer rates atomically after
complete preflight.
