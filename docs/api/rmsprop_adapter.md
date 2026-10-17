# API: Delegated RMSprop Adapter

`#include <jyppx/optim/rmsprop_adapter.hpp>`

`RmspropOptimizerAdapter` owns `torch::optim::RMSprop` and delegates every
numerical update to the supported LibTorch implementation. Construction
requires a stable optimizer ID, ordered group IDs, globally unique parameter
IDs, and typed `RmspropAdapterOptions`. Alpha, epsilon, coupled weight decay,
momentum, and centered mode are immutable topology; only learning rates may
change after construction. The adapter does not expose the native optimizer,
mutable parameter groups, or pointer-keyed state.

During `SingleDeviceStrategy::prepare()`, the same model-owned tensors may be
moved in place before the first update. The strategy synchronizes only fresh
dtype/device placement metadata; identity, shape, stride, group order, and
options remain fixed. Any later synchronization attempt is rejected once a
step or native RMSprop state exists.

```cpp
auto optimizer = std::make_shared<jyppx::optim::RmspropOptimizerAdapter>(
    "optimizer.rmsprop",
    std::vector<jyppx::optim::RmspropParameterGroup>{
        {.group_id = "recurrent",
         .parameters = recurrent_parameter_bindings,
         .options = {.learning_rate = 1e-3,
                     .alpha = 0.99,
                     .momentum = 0.9,
                     .centered = true}}});
```

`step()` executes an optional closure first, then validates all gradients
before any update. Undefined gradients are skipped. Dense gradients must match
their parameter dtype, device, and shape; sparse gradients are explicitly
unsupported. `completed_steps()` advances only after LibTorch returns
successfully. `set_learning_rates()` validates the complete group-rate vector
before committing it atomically.

`RmspropOptimizerCheckpointParticipant` stores current-only schema version 1:

- stable group and parameter topology, immutable RMSprop options, tensor
  descriptors, and exact LibTorch version;
- successful adapter steps, current group rates, state-presence flags, and
  each initialized parameter's native RMSprop step;
- raw square averages plus optional momentum buffers and centered gradient
  averages in stable parameter order.

Checkpoint fields are storage-neutral and bounded by explicit resource limits.
Restore validates and allocates every candidate state tensor before swapping
the native state map, so malformed or incompatible input leaves the optimizer
unchanged. No native archive, legacy map reader, or old RMSprop forwarding
header is accepted.
