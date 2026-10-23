# API: Delegated Adam Adapter

`#include <jyppx/liuding/optim/adam_adapter.hpp>`

`AdamOptimizerAdapter` owns `torch::optim::Adam` and delegates every numerical
update to the supported LibTorch implementation. Construction requires a stable optimizer ID, ordered group IDs, globally unique parameter IDs, and typed `AdamAdapterOptions`. The adapter does not expose the native optimizer, mutable parameter groups, or pointer-keyed state.

During `SingleDeviceStrategy::prepare()`, the same model-owned tensors may be moved in place before the first update. The strategy synchronizes only fresh dtype/device placement metadata; identity, shape, stride, group order, and options remain fixed. Any later synchronization attempt is rejected once a step or native Adam state exists.

```cpp
auto optimizer = std::make_shared<jyppx::liuding::optim::AdamOptimizerAdapter>(
    "optimizer.adam",
    std::vector<jyppx::liuding::optim::AdamParameterGroup>{{
        .group_id = "main",
        .parameters = {{.parameter_id = "encoder.weight",
                        .parameter = encoder_weight}},
        .options = {.learning_rate = 1e-3,
                    .beta1 = 0.9,
                    .beta2 = 0.999,
                    .epsilon = 1e-8,
                    .weight_decay = 1e-4,
                    .amsgrad = false}}});
```

`step()` executes an optional closure first, then validates all gradients
before any update. Undefined gradients are skipped. Dense gradients must match
their parameter dtype, device, and shape; sparse gradients are explicitly
unsupported. `completed_steps()` advances only after LibTorch returns
successfully. `set_learning_rates()` validates the complete group-rate vector
before committing it atomically.

`AdamOptimizerCheckpointParticipant` stores current-only schema version 1:

- stable group and parameter topology, immutable Adam options, tensor descriptors, and exact LibTorch version;
- successful adapter steps, current group rates, state-presence flags, and each initialized parameter's native Adam step;
- raw first moment, second moment, and optional AMSGrad maximum tensors in stable parameter order.

Checkpoint fields are storage-neutral and bounded by explicit resource limits. Restore validates and allocates every candidate state tensor before swapping the native state map, so malformed or incompatible input leaves the optimizer unchanged. No native archive or legacy map reader is accepted.
