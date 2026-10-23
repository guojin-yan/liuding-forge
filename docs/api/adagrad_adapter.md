# API: Delegated Adagrad Adapter

`#include <jyppx/liuding/optim/adagrad_adapter.hpp>`

`AdagradOptimizerAdapter` owns `torch::optim::Adagrad` and delegates every
numerical update to the supported LibTorch implementation. Construction requires a stable optimizer ID, ordered group IDs, globally unique parameter IDs, and typed `AdagradAdapterOptions`. Learning-rate decay, coupled weight decay, initial accumulator value, and epsilon are immutable topology; only learning rates may change after construction. The adapter does not expose the native optimizer, mutable parameter groups, or pointer-keyed state.

Adagrad initializes one sum accumulator for every parameter at construction. The adapter preserves that eager-state behavior, including distinct initial accumulator values per parameter group. During
`SingleDeviceStrategy::prepare()`, the same model-owned tensors may be moved in
place before the first update; synchronization rebuilds the zero-step
accumulators on the new dtype/device. Any later synchronization attempt is
rejected after a successful step.

```cpp
auto optimizer = std::make_shared<jyppx::liuding::optim::AdagradOptimizerAdapter>(
    "optimizer.adagrad",
    std::vector<jyppx::liuding::optim::AdagradParameterGroup>{
        {.group_id = "embeddings",
         .parameters = embedding_parameter_bindings,
         .options = {.learning_rate = 1e-2,
                     .learning_rate_decay = 1e-3,
                     .initial_accumulator_value = 0.1}}});
```

`step()` executes an optional closure first, then validates all gradients
before any update. Undefined gradients are skipped and each parameter keeps its own native step counter. Dense gradients must match their parameter dtype, device, and shape; sparse gradients are explicitly unsupported by this current contract. `completed_steps()` advances only after LibTorch returns successfully. `set_learning_rates()` validates the complete group-rate vector before committing it atomically.

`AdagradOptimizerCheckpointParticipant` stores current-only schema version 1:

- stable group and parameter topology, immutable Adagrad options, tensor
  descriptors, and exact LibTorch version;
- successful adapter steps, current group rates, and every parameter's native
  step counter, including zero-step eager state;
- raw sum accumulators in stable parameter order.

Checkpoint fields are storage-neutral and bounded by explicit resource limits.
Restore validates and allocates every candidate accumulator before swapping
the native state map, so malformed or incompatible input leaves the optimizer
unchanged. No native archive, legacy map reader, or old Adagrad forwarding
header is accepted.
