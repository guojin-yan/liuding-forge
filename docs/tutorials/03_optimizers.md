# Tutorial 3: Optimizers and Schedulers

JYPPX ships five optimizers and five learning-rate schedulers. Every
optimizer is validated numerically against `torch::optim` by the unit tests,
so results match LibTorch's own optimizers.

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

Optimizer state (momentum buffers, Adam moments, step counts) round-trips
through `state_dict()`, which is what `Trainer::save_checkpoint` captures:

```cpp
optimizer->load_state_dict(saved.state_dict());
```

## Epoch-based schedulers

Schedulers wrap an optimizer and rewrite its learning rate on every
`step()` — call once per epoch:

```cpp
#include <jyppx/optim/schedulers/step_lr.hpp>

auto scheduler = std::make_shared<jyppx::optim::schedulers::StepLR>(
    optimizer, /*step_size=*/10, /*gamma=*/0.1);
```

| Scheduler | Schedule |
|-----------|----------|
| `StepLR(optimizer, step_size, gamma)` | gamma every `step_size` epochs |
| `ExponentialLR(optimizer, gamma)` | `gamma^epoch` |
| `CosineAnnealingLR(optimizer, T_max, eta_min)` | cosine from base lr to `eta_min` |
| `MultiStepLR(optimizer, milestones, gamma)` | gamma at each milestone |

Read the current rate with `scheduler->get_last_lr()` and the epoch counter
with `get_step()`.

### Wiring a scheduler into the Trainer

Either hand the scheduler to the Trainer (stepped automatically at each
epoch end):

```cpp
trainer.set_scheduler(scheduler);
```

or wrap it in a callback — the Trainer de-duplicates when both reference the
same scheduler:

```cpp
trainer.add_callback(
    std::make_shared<jyppx::training::callbacks::LRSchedulerCallback>(scheduler));
```

## Metric-driven scheduling

`ReduceLROnPlateau` reacts to a validation metric instead of the epoch
count. Drive it manually after each validation:

```cpp
jyppx::optim::schedulers::ReduceLROnPlateauOptions options;
options.mode = "min";
options.factor = 0.5;
options.patience = 3;
options.min_lr = 1e-6;
auto plateau = std::make_shared<jyppx::optim::schedulers::ReduceLROnPlateau>(
    optimizer, options);

// inside a manual loop (not Trainer::fit):
double val_loss = trainer.validate(val_loader);
plateau->step(val_loss);   // decays when val_loss stagnates
```

Metric-driven schedulers use `step(metric)`; epoch-based schedulers reject
that overload, and `ReduceLROnPlateau::step()` rejects the metric-less call —
mixups fail loudly instead of silently.

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
