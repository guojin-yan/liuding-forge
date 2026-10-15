# API: Delegated SGD Adapter

The migrated SGD path lives in the `jyppx::optim` namespace and is exported by
the focused `jyppx::optim` CMake target.

```cpp
#include <jyppx/optim/sgd_adapter.hpp>
```

## Stable optimizer descriptor

`SgdOptimizerAdapter` delegates the numerical update to
`torch::optim::SGD`. JYPPX owns the stable descriptor around that backend:

- every group has a caller-selected `group_id`;
- every parameter has a globally unique `parameter_id`;
- tensor pointers and container positions are never persisted as identities;
- immutable SGD options are validated before native construction;
- learning rates are the only mutable group option exposed by the adapter.

```cpp
auto optimizer = std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
    std::vector<jyppx::optim::SgdParameterGroup>{
        {.group_id = "backbone",
         .parameters = {{.parameter_id = "encoder.weight",
                         .parameter = encoder_weight}},
         .options = {.learning_rate = 1e-2,
                     .momentum = 0.9,
                     .weight_decay = 1e-4}}});

optimizer->zero_grad();
loss.backward();
optimizer->step();
```

The adapter accepts dense, non-overlapping CPU or CUDA parameters with
float16, bfloat16, float32, or float64 dtype. Parameters must require
gradients. Group IDs, parameter IDs, tensor identities, and options must be
unique and valid when construction completes.

## Checkpoint participant

`SgdOptimizerCheckpointParticipant` provides exact, storage-neutral recovery:

```cpp
auto participant =
    std::make_shared<jyppx::optim::SgdOptimizerCheckpointParticipant>(
        optimizer,
        jyppx::optim::SgdOptimizerCheckpointOptions{
            .participant_id = "optimizer.main"});
registry.register_participant(participant);
```

Schema version 1 contains three field classes:

| Field | Contents |
|---|---|
| `topology` | stable group/parameter IDs, immutable options, dtype, device, shape, stride, and LibTorch version |
| `control` | completed step count, current group learning rates, and momentum-presence bits |
| `momentum.NNNNNNNNNN` | one fixed raw tensor field per parameter; empty before momentum exists |

The manifest does not change after the first step because momentum fields are
declared up front. Loading validates the stored manifest, full topology,
control metadata, all field sizes, and every candidate tensor before changing
native state. A successful commit swaps the complete pointer-keyed native state
map only after all data has been reconstructed against stable parameter IDs.

The format is not a LibTorch archive or pickle. Its exact guarantee currently
requires the same schema, stable topology, and compile-time LibTorch version.
The adapter and participant are not internally synchronized; training must be
quiescent during checkpoint describe, save, and load operations.

## Migration status

This is the first optimizer algorithm on the delegated path. The older
hand-written optimizer hierarchy remains separate while its consumers are
migrated algorithm by algorithm. There is no alias or forwarding bridge
between the two APIs; the old implementation will be deleted after all owned
consumers move to the delegated target and the optimizer ledger gates pass.
