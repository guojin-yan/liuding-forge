# Tutorial 5: Mixed Precision (AMP)

Mixed-precision training runs the forward pass in float16 and keeps
accumulation, master weights and reductions in float32 — faster on GPUs with
Tensor Cores and roughly 30% less memory. JYPPX provides the same building
blocks as `torch.cuda.amp`.

## Requirements

- A CUDA device: `use_amp = true` is validated at construction and throws
  `std::runtime_error` on CPU builds of the training loop.

## With the Trainer

One flag — the trainer handles autocast, loss scaling, unscaling, gradient
clipping order and skip-on-overflow:

```cpp
jyppx::training::Trainer trainer(
    model,
    optimizer,
    jyppx::training::TrainerOptions{
        .max_epochs = 20,
        .device = torch::kCUDA,
        .gradient_clip_val = 1.0,           // recommended with AMP
        .use_amp = true});                  // enable mixed precision
trainer.fit(train_loader, &val_loader);
```

What the loop does per batch, in order:

1. forward pass inside an autocast region (float16 where beneficial);
2. `scaler.scale(loss / accumulation_steps).backward()`;
3. at each accumulation boundary: `scaler.unscale(optimizer)` (so gradient
   clipping sees true magnitudes), optional clipping,
   `scaler.step(optimizer)` (skipped entirely if inf/NaN gradients were
   found), `scaler.update()`, `optimizer->zero_grad()`.

The scaler backs its scale factor off on overflow and grows it again after
`growth_interval` clean steps — the same policy as `torch.cuda.amp`.

## Manual loop

Use the pieces directly when not using `Trainer::fit`:

```cpp
#include <jyppx/training/amp/autocast.hpp>
#include <jyppx/training/amp/grad_scaler.hpp>

jyppx::training::amp::GradScaler scaler;
for (const auto& batch : loader) {
    optimizer->zero_grad();

    jyppx::Tensor loss;
    {
        jyppx::training::amp::AutocastGuard autocast{c10::DeviceType::CUDA,
                                                     at::kHalf};
        loss = torch::mse_loss(model->forward(batch.data.to(torch::kCUDA)),
                               batch.targets.to(torch::kCUDA));
    }

    scaler.scale(loss).backward();
    scaler.step(optimizer);   // unscales, checks for inf/NaN, steps if clean
    scaler.update();
}
```

`AutocastGuard` restores the previous autocast state on destruction, so
nested or sequential scopes behave sensibly.

## GradScaler options

```cpp
struct GradScalerOptions {
    double init_scale = 65536.0;    // 2^16
    double growth_factor = 2.0;     // multiply on growth
    double backoff_factor = 0.5;    // multiply on overflow
    int growth_interval = 2000;     // clean steps between growths
    bool enabled = true;            // false = pass-through
};
```

Pass it through the trainer with `TrainerOptions::amp_options`, or construct
`GradScaler{options}` directly. The scaler state serializes via
`state_dict()`/`load_state_dict()` for long-running jobs.

## Best practices

- **Validate precision**: compare final AMP and FP32 validation losses on
  your task before committing; expect agreement to within a small fraction
  of a percent.
- **Keep gradient clipping on**: AMP can produce large early gradients.
- **Watch the scale**: `scaler.get_scale()` should climb over time; constant
  back-offs mean the initial scale is too high or the learning rate too
  large.
- **Batch size**: the memory savings usually allow a larger batch; adjust
  the learning rate together with it.
- **Small models**: autocast overhead can exceed its savings on tiny
  workloads — measure before enabling.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Loss becomes NaN | repeated overflow | lower the learning rate, lower `init_scale`, keep clipping on |
| No speedup | model too small for Tensor Cores | increase model/batch size, profile first |
| AMP throws at construction | CPU device or no CUDA driver | run on CUDA, or leave `use_amp = false` |
