# API: Native NAdam Adapter

`#include <jyppx/liuding/optim/nadam_adapter.hpp>`

`NAdamOptimizerAdapter` provides current-only NAdam and NAdamW updates through
a JYPPX-owned tensor kernel. LibTorch does not publish NAdam in its C++ optimizer API, so this adapter follows the documented `torch.optim.NAdam` single-tensor algorithm instead of exposing or forwarding the removed legacy implementation.

Construction requires a stable optimizer ID, ordered group IDs, globally unique parameter IDs, and typed `NAdamAdapterOptions`. Betas, epsilon, weight decay, momentum decay, decoupled-decay mode, and maximize mode are immutable topology. Only the complete vector of learning rates may change after construction.

```cpp
auto optimizer = std::make_shared<jyppx::liuding::optim::NAdamOptimizerAdapter>(
    "optimizer.nadam",
    std::vector<jyppx::liuding::optim::NAdamParameterGroup>{
        {.group_id = "decay",
         .parameters = decay_parameter_bindings,
         .options = {.learning_rate = 2e-3,
                     .weight_decay = 1e-2,
                     .momentum_decay = 4e-3,
                     .decoupled_weight_decay = true}},
        {.group_id = "no_decay",
         .parameters = no_decay_parameter_bindings,
         .options = {.learning_rate = 2e-3}}});
```

`step()` evaluates an optional closure and validates every gradient before the
first mutation. Undefined gradients are skipped and therefore retain their
own parameter step and cumulative momentum product. Dense gradients must
match parameter dtype, device, and shape; sparse gradients are rejected.
Setting `decoupled_weight_decay` selects NAdamW behavior, while `maximize`
negates gradients before regularization and moment updates.

`NAdamOptimizerCheckpointParticipant` stores current-only schema version 1:

- stable group and parameter topology, all immutable NAdam options, tensor descriptors, and the exact LibTorch version;
- successful adapter steps, current rates, per-parameter state presence, update steps, and cumulative momentum products;
- raw first- and second-moment tensors in stable parameter order.

Restore validates all metadata and materializes every candidate tensor before replacing state, so malformed input leaves the optimizer unchanged. No legacy map checkpoint, forwarding header, native archive, or pointer key is accepted.

Numerical coverage includes an independent scalar reference over long multi-group trajectories, NAdamW and maximize modes, missing-gradient resume, CPU bfloat16, and real CUDA float32/float16 when CUDA is available. Because LibTorch has no C++ NAdam baseline, `bench_nadam_native_throughput` records absolute median time per step rather than presenting a misleading adapter-to-LibTorch ratio.
