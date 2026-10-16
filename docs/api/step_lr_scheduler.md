# API: Descriptor-First StepLR Scheduler

The migrated scheduler path lives directly in `jyppx::optim` and is exported
by the focused `jyppx::optim` CMake target.

```cpp
#include <jyppx/optim/step_lr_scheduler.hpp>
```

## Stable scheduler descriptor

`StepLrScheduler` controls any optimizer implementing `OptimizerRateControl`.
The current delegated SGD adapter implements that narrow interface. Scheduler
topology records the algorithm type, step mode, typed options, stable optimizer
identity, ordered group IDs, group signature, optional callable identity,
ordered child descriptors, and a diagnostic fingerprint.

```cpp
auto scheduler = std::make_shared<jyppx::optim::StepLrScheduler>(
    optimizer,
    jyppx::optim::StepLrSchedulerOptions{
        .scheduler_id = "scheduler.epochs",
        .step_size = 10,
        .gamma = 0.1});

// Call once after each completed epoch.
scheduler->step();
```

The descriptor stores configuration values with explicit types rather than a
`map<string, double>`. `StepLR` uses `SchedulerStepMode::epoch`; its typed
configuration contains `step_size`, `gamma`, and construction-time base rates.
The complete canonical descriptor is compared during restore. Its compact
fingerprint is diagnostic metadata and is never the sole compatibility check.

## Exact checkpoint participant

```cpp
auto participant = std::make_shared<
    jyppx::optim::StepLrSchedulerCheckpointParticipant>(scheduler);
registry.register_participant(participant);
```

Schema version 1 owns two bounded fields:

| Field | Contents |
|---|---|
| `topology` | complete canonical descriptor and scheduler identity |
| `control` | completed-step counter and every current group learning rate |

Restore validates the manifest, current-format magic, version, resource
limits, complete topology, counter, and all rates before mutation. Rate commit
uses the optimizer's atomic rate-control contract, and the scheduler counter is
changed only after that commit succeeds. Legacy native archives and untyped
scheduler maps are not accepted.

The scheduler and participant are not internally synchronized. Training must
serialize optimizer steps, scheduler steps, direct learning-rate changes, and
checkpoint operations.

## Migration status

This is the first scheduler on the descriptor-first path. The older
`jyppx::optim::schedulers` hierarchy remains separate while trainer consumers
are migrated. There is no alias or forwarding bridge between the two APIs.
