# API: Optimizers and Schedulers

All types live in the `jyppx::optim` namespace.

The migrated optimizer boundary starts with the independently exported
[`SgdOptimizerAdapter`](sgd_adapter.md). It delegates numerical updates to
LibTorch and supplies stable parameter identity plus storage-neutral recovery.
The migrated scheduler boundary includes descriptor-first
[`StepLrScheduler`](step_lr_scheduler.md) plus transactional
[`SequentialLrScheduler` and `ChainedLrScheduler`](composite_lr_scheduler.md)
and the [epoch scheduler family](epoch_lr_scheduler.md), including the
[linear-family schedulers](linear_lr_scheduler.md) and
[event-driven schedulers](event_lr_scheduler.md).
They record typed semantics and stable optimizer/group identity and provide
exact storage-neutral recovery.
The hand-written hierarchy documented below is the legacy path still used by
unmigrated trainer and scheduler consumers; it is not a compatibility wrapper
for the delegated API.

## Optimizer base

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

## Optimizers

Every optimizer takes the parameter list plus an options struct and validates
the hyperparameters at construction. All of them are validated numerically
against `torch::optim` (float64, tolerance 1e-9) by the unit tests.

```cpp
struct SGDOptions   { double lr = 0.01; double momentum = 0.0; double dampening = 0.0;
                      double weight_decay = 0.0; bool nesterov = false; };
struct AdamOptions  { double lr = 0.001; double beta1 = 0.9; double beta2 = 0.999;
                      double eps = 1e-8; double weight_decay = 0.0; bool amsgrad = false; };
struct AdamWOptions { double lr = 0.001; double beta1 = 0.9; double beta2 = 0.999;
                      double eps = 1e-8; double weight_decay = 0.01; bool amsgrad = false; };
struct RMSpropOptions { double lr = 0.01; double alpha = 0.99; double eps = 1e-8;
                        double weight_decay = 0.0; double momentum = 0.0;
                        bool centered = false; };
struct AdagradOptions { double lr = 0.01; double lr_decay = 0.0;
                        double weight_decay = 0.0; double eps = 1e-10; };
```

```cpp
#include <jyppx/optim/sgd.hpp>       // also adam.hpp, adamw.hpp, rmsprop.hpp, adagrad.hpp

auto optimizer = std::make_shared<jyppx::optim::Adam>(
    model->parameters(), jyppx::optim::AdamOptions{.lr = 1e-3, .weight_decay = 1e-4});
```

Notes:

- **AdamW** applies decoupled weight decay (`param *= 1 - lr * wd`) like
  `torch.optim.AdamW`.
- **RMSprop momentum** follows the `torch::optim` C++ semantics: the buffer
  accumulates the normalized gradient `grad / avg` and the step is
  `-lr * buffer`. This coincides with `torch.optim.rmsprop` in Python on the
  first step and differs from the second step on.
- Mutating `options()` (e.g. `optimizer->options().lr = 0.01`) is picked up
  on the next `step()`; schedulers use `set_lr`/`set_lrs` instead.

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

These APIs do not wrap or forward to `jyppx::optim::schedulers`. The following
legacy algorithms remain available only for unmigrated Trainer/callback
consumers and are not checkpoint-safe under the new participant contract:

| Legacy scheduler | Parameters | Status |
|------------------|------------|--------|
| `ReduceLROnPlateau` | mode, factor, patience, threshold | retained only by the unmigrated Trainer contract |
| `CosineAnnealingWarmRestarts` | `T_0`, `T_mult`, `eta_min` | pending restart-state migration |

The obsolete OneCycle/Cyclic/Lambda forwarding headers and implementations
were removed. Their direct descriptor-first replacements do not accept old
map checkpoints. Momentum cycling is explicitly rejected until optimizer
hyperparameters have an atomic control and participant contract.

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
complete preflight. The similarly named legacy type remains solely because
the old Trainer contract still owns a legacy optimizer; it is not forwarded
from the new API.
