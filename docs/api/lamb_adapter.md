# API: Native LAMB Adapter

`#include <jyppx/liuding/optim/lamb_adapter.hpp>`

`LambOptimizerAdapter` provides a current-only Layer-wise Adaptive Moments
optimizer through a JYPPX-owned tensor kernel. The kernel uses bias-corrected
Adam moments, adds weight decay to the complete update direction, and then
applies a per-parameter-tensor trust ratio.

Construction requires a stable optimizer ID, stable group IDs, globally unique
parameter IDs, and typed `LambAdapterOptions`. Both beta values must be in
`[0, 1)`, epsilon must be positive, and learning rate and weight decay must be
non-negative. `maximize` negates the objective gradient before moments are
updated.

```cpp
auto optimizer = std::make_shared<jyppx::liuding::optim::LambOptimizerAdapter>(
    "optimizer.lamb",
    std::vector<jyppx::liuding::optim::LambParameterGroup>{
        {.group_id = "decay",
         .parameters = parameter_bindings,
         .options = {.learning_rate = 1e-3,
                     .beta1 = 0.9,
                     .beta2 = 0.999,
                     .epsilon = 1e-6,
                     .weight_decay = 1e-2}}});
```

The trust ratio is `min(||parameter||, 10) / ||direction||`. It falls back to
1 when either norm is zero. These details are frozen API semantics and covered
by independent reference tests. The ratio is calculated once per bound tensor,
so callers should bind tensors at the layer granularity required by their
training design.

Only learning rates are mutable after construction. `set_learning_rates()`
validates the complete replacement before changing any group. Parameters with
undefined gradients are skipped and retain independent state steps. Sparse or
descriptor-mismatched gradients are rejected before any parameter changes.

`LambOptimizerCheckpointParticipant` stores current-only schema version 1:

- stable group and parameter topology, immutable options, tensor descriptors,
  and the compile-time LibTorch version;
- completed adapter steps, mutable group learning rates, per-parameter state
  presence, and independent update steps;
- the `exp_avg` and `exp_avg_sq` tensors for every initialized parameter.

Fresh parameters have explicit empty lazy-state fields. Restore validates all
metadata and tensor bytes, materializes a complete replacement state map, and
commits with a nothrow swap. It does not accept legacy maps or another
algorithm's state.

The focused quality gate covers a 512-step multi-group tensor reference,
regularization, maximize mode, trust-ratio edge cases, missing-gradient resume,
malformed-state rollback, CPU bfloat16, and real CUDA float32/float16 when CUDA
is available. `bench_lamb_native_throughput` reports absolute median time per
step because LibTorch does not expose a comparable public C++ LAMB optimizer.
