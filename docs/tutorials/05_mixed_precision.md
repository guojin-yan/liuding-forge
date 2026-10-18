# Tutorial 5: Mixed Precision Primitives

JYPPX provides explicit autocast and gradient-scaling primitives plus a
current-only Trainer precision contract. There is no implicit `use_amp`
compatibility switch: select `TrainerOptions::precision`, and bind an enabled
`GradScaler` with its checkpoint participant for CUDA float16.

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
const bool applied = scaler.step_allowed(optimizer);
if (applied) {
  optimizer->step();
}
scaler.mark_step(optimizer, applied);
scaler.update();
```

For a manual multi-optimizer cycle, unscale every optimizer before asking for
the aggregate decision. Report the same group result for every participant and
update the scaler exactly once:

```cpp
scaler.unscale(encoder_optimizer);
scaler.unscale(head_optimizer);
const bool apply_group = scaler.cycle_allowed();
if (apply_group) {
  encoder_optimizer->step();
  head_optimizer->step();
}
scaler.mark_step(encoder_optimizer, apply_group);
scaler.mark_step(head_optimizer, apply_group);
scaler.update();
```

Overflow skips every optimizer in the logical update; batch schedulers and the
Trainer global step advance only after a committed group update.

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
