# Tutorial 3: Optimizers and Schedulers

JYPPX currently provides a legacy hand-written optimizer family alongside a
descriptor-first migration path backed by LibTorch. The compatibility tables
and quality records identify which algorithms have completed delegated
execution, typed checkpoint recovery, and numerical comparison; algorithms
that have not crossed those gates remain explicitly marked as legacy.

## Choosing an optimizer

| Optimizer | Use when | Key options |
|-----------|----------|-------------|
| `SGD` | large-scale vision, well-tuned lr schedules | `momentum`, `nesterov`, `weight_decay` |
| `Adam` | default first choice for most tasks | `beta1`, `beta2`, `eps`, `amsgrad` |
| `AdamW` | transformer/decoder training with weight decay | decoupled `weight_decay` |
| `RMSprop` | RNNs, non-stationary objectives | `alpha`, `centered`, `momentum` |
| `Adagrad` | sparse features, no manual schedule | `lr_decay` |

```cpp
auto optimizer = std::make_shared<jyppx::optim::AdamW>(
    model->parameters(),
    jyppx::optim::AdamWOptions{.lr = 3e-4, .weight_decay = 0.01});
```

Options validate at construction (negative lr, `beta1 >= 1`, non-positive
eps etc. throw `std::invalid_argument`). All optimizers expose
`step()`, `zero_grad(bool set_to_none = false)`, `get_lrs()/set_lr()`,
`step_count()` and `state_dict()/load_state_dict()`.

### Checkpointing an optimizer

Legacy optimizers expose `state_dict()`, but current Trainer accepts only an
`OptimizerAdapter` plus an explicit checkpoint participant. For migrated SGD,
register the participant in `FitRequest::optimizer_checkpoint`:

```cpp
auto optimizer_state =
    std::make_shared<jyppx::optim::SgdOptimizerCheckpointParticipant>(
        delegated_sgd);
```

The participant stores stable group/parameter identities, options, completed
steps, current rates and momentum without persisting native pointer keys.

## Epoch-based schedulers

Descriptor-first schedulers consume the narrow `OptimizerRateControl`
contract. They validate every group rate, preview the complete transition,
then replace all group rates atomically. Call `step()` once after each
successfully completed epoch:

```cpp
#include <jyppx/optim/epoch_lr_scheduler.hpp>

auto scheduler = std::make_shared<jyppx::optim::MultiStepLrScheduler>(
    delegated_sgd,
    jyppx::optim::MultiStepLrSchedulerOptions{
        .scheduler_id = "scheduler.epochs",
        .milestones = {30, 60, 90},
        .gamma = 0.1});
```

| Scheduler | Schedule |
|-----------|----------|
| `StepLrScheduler` | gamma every `step_size` epochs |
| `ExponentialLrScheduler` | multiply current rates by `gamma` every epoch |
| `MultiStepLrScheduler` | multiply at each strictly ordered milestone |
| `CosineAnnealingLrScheduler` | recursive cosine trajectory with `t_max` and `eta_min` |

The related construction-time schedulers are available from
`<jyppx/optim/linear_lr_scheduler.hpp>`:

| Scheduler | Construction and boundary |
|-----------|---------------------------|
| `LinearLrScheduler` | starts at `start_factor`, reaches `end_factor` after `total_iters`, then clamps |
| `WarmupLrScheduler` | starts at `start_factor`, reaches base rates after `warmup_iters`, then clamps |
| `ConstantLrScheduler` | starts at `factor`, keeps it for `total_iters` events, then reaches base rates |
| `PolynomialLrScheduler` | keeps base rates initially and decays with `power` toward a terminal factor |

Warmup is standalone by design. Use `SequentialLrScheduler` to append another
descriptor-first scheduler so that the complete tree remains transactional.

Read all current group rates with `scheduler->learning_rates()` and the
committed epoch counter with `completed_steps()`. Options, base rates, stable
optimizer/group identities and step mode are part of the typed descriptor.

### Checkpointing a scheduler

Every migrated scheduler uses the same exact participant:

```cpp
auto participant =
    std::make_shared<jyppx::optim::LrSchedulerCheckpointParticipant>(
        scheduler);
checkpoint_registry.register_participant(participant);
```

Restore requires the same scheduler type, typed options, base rates, stable
optimizer ID and ordered group IDs. Wrap the scheduler in `SchedulerAdapter`
and place it in `FitRequest::scheduler`; Trainer then owns the only ordered
batch, epoch or metric event stream and checkpoints the same adapter.

## Metric-driven scheduling

`ReduceLrOnPlateauScheduler` reacts to a validation metric instead of the
epoch count. It publishes metric step mode and stores best-value, patience,
cooldown, per-group minima, and current rates in the exact participant:

```cpp
#include <jyppx/optim/event_lr_scheduler.hpp>

auto plateau = std::make_shared<jyppx::optim::ReduceLrOnPlateauScheduler>(
    delegated_sgd,
    jyppx::optim::ReduceLrOnPlateauSchedulerOptions{
        .factor = 0.5,
        .patience = 3,
        .min_learning_rates = {1e-6}});

// Direct use in a manual loop:
plateau->step(validation_loss);

// Or bind SchedulerAdapter to Trainer and return the configured validation
// metric from the validation StepOutput.
```

Metric-driven schedulers use `step(metric)`; all other descriptor-first
schedulers reject that overload, and plateau rejects metric-less `step()`.
One-cycle and cyclic schedulers use batch step mode and must advance only
after a successful optimizer update. Callable schedules require a stable
`callable_id`; exact restore rejects a different identity.

## Manual training loop

If you outgrow `Trainer::fit`, the pieces compose directly:

```cpp
for (int epoch = 0; epoch < max_epochs; ++epoch) {
    optimizer->zero_grad();
    for (const auto& batch : loader) {
        auto prediction = model->forward(batch.data);
        auto loss = torch::mse_loss(prediction, batch.targets);
        loss.backward();
        optimizer->step();
        optimizer->zero_grad();
    }
    scheduler->step();
}
```

## What to read next

- Tutorial 4 adds checkpointing, early stopping and metrics around the loop.
