# Tutorial 5: Mixed Precision Primitives

JYPPX currently provides explicit autocast and gradient-scaling primitives.
The current Trainer does not expose an implicit `use_amp` switch: precision,
scaler and multi-optimizer ordering will enter Trainer through a dedicated
precision contract rather than hidden option-dependent behavior.

## Autocast

`AutocastGuard` scopes backend autocast state and restores the prior enabled
flag and dtype on destruction, including nested use:

```cpp
jyppx::training::amp::AutocastGuard autocast{
    c10::DeviceType::CUDA, at::kHalf};
auto prediction = model.forward(input);
auto loss = torch::mse_loss(prediction, target);
```

Use CUDA Half/BFloat16 only on supported hardware. Validate device and dtype
before entering the region.

## GradScaler

`GradScaler` owns the scale, overflow history and growth/backoff policy:

```cpp
jyppx::training::amp::GradScaler scaler;
optimizer.zero_grad();
auto scaled = scaler.scale(loss);
scaled.backward();
scaler.unscale(optimizer);
// Inspect or clip true gradients here.
const bool applied = scaler.step(optimizer);
scaler.update();
```

The exact overloads depend on the optimizer surface being used. Overflow must
skip every optimizer that belongs to the same logical update; do not advance a
batch scheduler for a skipped update.

## Checkpoint state

Register `GradScalerCheckpointParticipant` with the same coordinated domain as
the model, every optimizer, data cursor, RNG and Trainer state. Restoring only
the scale while omitting optimizer or loop counters is not an exact resume.

## Integration rules

When implementing a custom precision-aware strategy, preserve this order:

1. run forward and loss under the selected autocast policy;
2. divide by the configured accumulation window and scale the loss;
3. backward each micro-batch;
4. at an optimizer boundary, unscale every participating optimizer;
5. detect non-finite gradients, then clip true gradients;
6. apply or atomically skip all optimizer updates;
7. update the scaler and advance schedulers only for committed updates;
8. checkpoint only after the complete boundary is committed.

Compare FP32 and mixed-precision convergence on the actual model. Record
device, dtype, LibTorch version and scaler configuration with benchmark results.
