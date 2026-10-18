# Tutorial 3: Optimizers and Schedulers

JYPPX provides a current-only descriptor-first optimizer family. SGD, Adam,
AdamW, RMSprop, and Adagrad delegate their update kernels to LibTorch. NAdam,
Adadelta, and LAMB use audited native kernels because the supported LibTorch
C++ surface does not provide equivalent public implementations. Every current
algorithm has stable identities, typed options, bounded exact checkpoint state,
and the same Trainer-facing adapter contract; no legacy optimizer API is kept.

## Choosing an optimizer

| Optimizer | Use when | Key options |
|-----------|----------|-------------|
| `SGD` | large-scale vision, well-tuned lr schedules | `momentum`, `nesterov`, `weight_decay` |
| `Adam` | default first choice for most tasks | `beta1`, `beta2`, `eps`, `amsgrad` |
| `AdamW` | transformer/decoder training with weight decay | decoupled `weight_decay` |
| `RMSprop` | RNNs, non-stationary objectives | `alpha`, `centered`, `momentum` |
| `Adagrad` | adaptive decay for dense feature updates | `lr_decay`, `initial_accumulator_value` |
| `NAdam` | Adam-style adaptation with Nesterov momentum | `momentum_decay`, coupled or decoupled `weight_decay`, `maximize` |
| `Adadelta` | adaptive updates without a manually scaled gradient history | `rho`, `epsilon`, coupled `weight_decay`, `maximize` |
| `LAMB` | large-batch training with layer-wise trust ratios | `beta1`, `beta2`, `eps`, `weight_decay`, `trust_clip` |

```cpp
auto optimizer = std::make_shared<jyppx::optim::AdamWOptimizerAdapter>(
    "optimizer.adamw",
    std::vector<jyppx::optim::AdamWParameterGroup>{
        {.group_id = "decay",
         .parameters = decay_parameter_bindings,
         .options = {.learning_rate = 3e-4, .weight_decay = 0.01}},
        {.group_id = "no_decay",
         .parameters = no_decay_parameter_bindings,
         .options = {.learning_rate = 3e-4, .weight_decay = 0.0}}});
```

Current descriptors validate all options at construction. Invalid rates,
betas, epsilon, decay values, identities, or parameter topology raise typed
JYPPX errors before a native optimizer is created. The common adapter contract
provides `step()`, `zero_grad()`, `learning_rates()`,
`set_learning_rates()`, `completed_steps()`, and stable parameter bindings.
Checkpoint state is deliberately exposed only through the matching typed
participant; no native optimizer or legacy map state is public.

### Checkpointing an optimizer

Current Trainer accepts one or more `OptimizerExecutionBinding` values. Each
binding pairs an `OptimizerAdapter` with its matching checkpoint participant
and optional scheduler:

```cpp
auto optimizer_state =
    std::make_shared<jyppx::optim::AdamWOptimizerCheckpointParticipant>(
        optimizer);

request.optimizers.push_back({.optimizer = optimizer,
                              .checkpoint = optimizer_state,
                              .scheduler = scheduler_adapter});
```

The participant stores stable group and parameter identities, immutable
options, completed steps, current rates, and algorithm state without
persisting native pointer keys.

Adam uses `AdamOptimizerCheckpointParticipant`; it additionally stores every
initialized parameter's native step, first and second moments, and optional
AMSGrad maximum tensor. Restore requires identical stable topology and the
declared exact LibTorch version.

AdamW uses `AdamWOptimizerCheckpointParticipant` with the same complete moment
and step-state boundary while freezing decoupled weight decay in topology.

RMSprop uses `RmspropOptimizerCheckpointParticipant`. It always stores square
averages and conditionally stores momentum buffers and centered gradient
averages according to the immutable group options.

Adagrad uses `AdagradOptimizerCheckpointParticipant`. It stores each eagerly
initialized sum accumulator and its independent per-parameter native step, so
missing gradients resume with the same learning-rate-decay trajectory.

NAdam uses `NAdamOptimizerCheckpointParticipant`. It stores each initialized
parameter's first and second moments, independent update step, and cumulative
momentum product. That last scalar is required to resume the scheduled
Nesterov correction exactly when parameters have different missing-gradient
histories.

Adadelta uses `AdadeltaOptimizerCheckpointParticipant`. It stores each
initialized parameter's squared-gradient average, accumulated-delta average,
and independent update step. Fresh parameters retain lazy empty state, and
missing gradients do not advance their accumulators.

LAMB uses `LambOptimizerCheckpointParticipant`. It stores first and second
moments, independent parameter steps, stable group topology, and immutable
trust-ratio options required for exact continuation.

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
and place it in the same `OptimizerExecutionBinding` as its optimizer. Trainer
then owns an independent ordered batch, epoch or metric event cursor for every
scheduler and checkpoints each adapter.

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
