# API: Optimizers and Schedulers

All types live in the `jyppx::optim` namespace.

The migrated optimizer boundary includes independently exported
[`SgdOptimizerAdapter`](sgd_adapter.md) and
[`AdamOptimizerAdapter`](adam_adapter.md), plus
[`AdamWOptimizerAdapter`](adamw_adapter.md),
[`RmspropOptimizerAdapter`](rmsprop_adapter.md), and
[`AdagradOptimizerAdapter`](adagrad_adapter.md). They delegate numerical
updates to LibTorch. The current-only
[`NAdamOptimizerAdapter`](nadam_adapter.md) and
[`AdadeltaOptimizerAdapter`](adadelta_adapter.md) supply the same stable
identity and storage-neutral recovery contract around JYPPX-native kernels
because the supported LibTorch C++ candidates expose neither algorithm.
The migrated scheduler boundary includes descriptor-first
[`StepLrScheduler`](step_lr_scheduler.md) plus transactional
[`SequentialLrScheduler` and `ChainedLrScheduler`](composite_lr_scheduler.md)
and the [epoch scheduler family](epoch_lr_scheduler.md), including the
[linear-family schedulers](linear_lr_scheduler.md) and
[event-driven schedulers](event_lr_scheduler.md).
They record typed semantics and stable optimizer/group identity and provide
exact storage-neutral recovery.
The legacy hierarchy documented below remains only for optimizer
algorithms that have not yet migrated. It is not a compatibility wrapper for
the current API. The old hand-written SGD, Adam, AdamW, RMSprop, Adagrad,
NAdam, Adadelta, and `optim/schedulers` hierarchy have been removed.

## Legacy optimizer base

`#include <jyppx/optim/optimizer.hpp>`

```cpp
struct ParamGroup {
    TensorList params;                      // parameters in this group
    std::map<std::string, double> options;  // per-group hyperparameters
};

class Optimizer {
public:
    explicit Optimizer(const TensorList& params);  // throws on empty list
    virtual void step() = 0;
    void zero_grad(bool set_to_none = false);      // false zeroes, true clears
    std::vector<ParamGroup>& param_groups();
    void add_param_group(ParamGroup group);
    std::vector<double> get_lrs() const;           // throws if a group lacks "lr"
    void set_lr(double lr);
    void set_lrs(const std::vector<double>& lrs);
    int64_t step_count() const;
    virtual std::map<std::string, torch::Tensor> state_dict() const;
    virtual void load_state_dict(const std::map<std::string, torch::Tensor>& state);
};
```

`add_param_group()` rejects empty groups, undefined tensors, duplicate tensor
identities, non-finite options, and negative learning rates. Omitted options
inherit from the first group.

`state_dict()` round-trips hyperparameters (as float64) and per-parameter
optimizer state (e.g. momentum buffers), keyed by stable parameter order, so a
fresh optimizer can resume an interrupted run. Loading requires the current
format metadata, validates parameter-group and parameter counts plus every
declared state count, and replaces buffers atomically instead of merging them
with stale state.

## Remaining legacy optimizers

The hand-written base currently remains only for the experimental LAMB
implementation. That API is not the target contract and will be retired when
its delegated ledger closes.

```cpp
#include <jyppx/optim/adamw_adapter.hpp>

auto optimizer = std::make_shared<jyppx::optim::AdamWOptimizerAdapter>(
    "optimizer.adamw",
    std::vector<jyppx::optim::AdamWParameterGroup>{
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
- Remaining legacy optimizers expose mutable `options()`. Delegated SGD, Adam,
  AdamW, RMSprop, and Adagrad, plus native NAdam and Adadelta, use
  `set_learning_rates()` as their only mutable hyperparameter operation.

## Learning-rate schedulers

The migrated `jyppx::optim::LrScheduler` hierarchy depends only on
`OptimizerRateControl`. Its `step()` path previews and validates a complete
state/rate transition before one atomic optimizer update. The generic
`LrSchedulerCheckpointParticipant` persists a typed descriptor tree, complete
counter tree and all current group rates.

```cpp
#include <jyppx/optim/epoch_lr_scheduler.hpp>

auto scheduler = std::make_shared<jyppx::optim::ExponentialLrScheduler>(
    delegated_sgd,
    jyppx::optim::ExponentialLrSchedulerOptions{
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
#include <jyppx/optim/event_lr_scheduler.hpp>

auto plateau = std::make_shared<jyppx::optim::ReduceLrOnPlateauScheduler>(
    delegated_sgd,
    jyppx::optim::ReduceLrOnPlateauSchedulerOptions{
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
