# API: Delegated AdamW Adapter

`#include <jyppx/liuding/optim/adamw_adapter.hpp>`

`AdamWOptimizerAdapter` owns `torch::optim::AdamW` and delegates every
numerical update to the supported LibTorch implementation. Construction requires a stable optimizer ID, ordered group IDs, globally unique parameter IDs, and typed `AdamWAdapterOptions`. Weight decay is decoupled from the gradient update. The adapter does not expose the native optimizer, mutable parameter groups, or pointer-keyed state.

During `SingleDeviceStrategy::prepare()`, the same model-owned tensors may be moved in place before the first update. The strategy synchronizes only fresh dtype/device placement metadata; identity, shape, stride, group order, and options remain fixed. Any later synchronization attempt is rejected once a step or native AdamW state exists.

```cpp
auto optimizer = std::make_shared<jyppx::liuding::optim::AdamWOptimizerAdapter>(
    "optimizer.adamw",
    std::vector<jyppx::liuding::optim::AdamWParameterGroup>{
        {.group_id = "decay",
         .parameters = decay_parameter_bindings,
         .options = {.learning_rate = 3e-4, .weight_decay = 0.01}},
        {.group_id = "no_decay",
         .parameters = no_decay_parameter_bindings,
         .options = {.learning_rate = 3e-4, .weight_decay = 0.0}}});
```

`step()` executes an optional closure first, then validates all gradients
before any update. Undefined gradients are skipped. Dense gradients must match
their parameter dtype, device, and shape; sparse gradients are explicitly
unsupported. `completed_steps()` advances only after LibTorch returns
successfully. `set_learning_rates()` validates the complete group-rate vector
before committing it atomically.

`AdamWOptimizerCheckpointParticipant` stores current-only schema version 1:

- stable group and parameter topology, immutable AdamW options, tensor descriptors, and exact LibTorch version;
- successful adapter steps, current group rates, state-presence flags, and each initialized parameter's native AdamW step;
- raw first moment, second moment, and optional AMSGrad maximum tensors in stable parameter order.

Checkpoint fields are storage-neutral and bounded by explicit resource limits. Restore validates and allocates every candidate state tensor before swapping the native state map, so malformed or incompatible input leaves the optimizer unchanged. No native archive or legacy map reader is accepted.
