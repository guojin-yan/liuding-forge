# API: Native Adadelta Adapter

`#include <jyppx/liuding/optim/adadelta_adapter.hpp>`

`AdadeltaOptimizerAdapter` provides current-only Adadelta updates through a
JYPPX-owned tensor kernel. The supported LibTorch C++ candidates do not publish
an Adadelta optimizer, so the kernel follows the documented
`torch.optim.Adadelta` single-tensor order and is verified against an
independent scalar reference.

Construction requires a stable optimizer ID, stable group IDs, globally unique
parameter IDs, and typed `AdadeltaAdapterOptions`. `rho` must be in `[0, 1]`,
epsilon must be positive, and learning rate and weight decay must be
non-negative. Coupled weight decay enters the gradient before the two running
averages; `maximize` negates the objective gradient first.

```cpp
auto optimizer = std::make_shared<jyppx::liuding::optim::AdadeltaOptimizerAdapter>(
    "optimizer.adadelta",
    std::vector<jyppx::liuding::optim::AdadeltaParameterGroup>{
        {.group_id = "main",
         .parameters = parameter_bindings,
         .options = {.learning_rate = 1.0,
                     .rho = 0.9,
                     .epsilon = 1e-6,
                     .weight_decay = 0.0}}});
```

Only learning rates are mutable after construction. `set_learning_rates()`
validates the complete replacement before changing any group. Parameters with
undefined gradients are skipped and keep independent state steps. Sparse or
descriptor-mismatched gradients are rejected before any parameter changes.

`AdadeltaOptimizerCheckpointParticipant` stores current-only schema version 1:

- stable group and parameter topology, immutable options, tensor descriptors,
  and the compile-time LibTorch version;
- completed adapter steps, mutable group learning rates, per-parameter state
  presence, and independent update steps;
- the `square_avg` and `acc_delta` tensor for every initialized parameter.

Fresh parameters have explicit empty lazy-state fields. Restore validates all
metadata and tensor bytes, materializes a complete replacement state map, and
commits with a nothrow swap. It does not accept legacy maps or another
algorithm's state.

The focused quality gate covers a 512-step, multi-group scalar reference,
coupled decay, maximize mode, missing-gradient resume, malformed-state
rollback, CPU bfloat16, and real CUDA float32/float16 when CUDA is available.
Because the candidate LibTorch C++ APIs have no comparable native class,
`bench_adadelta_native_throughput` records absolute median time per step rather
than presenting a misleading adapter/native ratio.
