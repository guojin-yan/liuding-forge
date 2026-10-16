# Training scheduler adapter

`#include <jyppx/training/scheduler_adapter.hpp>`

`SchedulerAdapter` is the event boundary between a training loop and a
descriptor-first `LrScheduler`. It exposes the scheduler's declared `batch`,
`epoch`, `metric`, or `manual` mode and accepts a `SchedulerInput` containing
that trigger, a one-based event index, and an optional metric.

The adapter rejects a wrong trigger, duplicate event, skipped event, missing or
non-finite metric, and a metric attached to a non-metric event before any state
changes. A valid metric event calls `step(metric)`; every other valid event
calls the metric-less `step()`.

The adapter is also a `CheckpointParticipant`. Registration with
`CheckpointParticipantRegistry` delegates the exact scheduler descriptor,
typed state tree, and current rates without introducing another checkpoint
schema or compatibility reader.

```cpp
auto scheduler = std::make_shared<jyppx::optim::StepLrScheduler>(
    optimizer,
    jyppx::optim::StepLrSchedulerOptions{.step_size = 10, .gamma = 0.1});
auto scheduler_adapter =
    std::make_shared<jyppx::training::SchedulerAdapter>(scheduler);

scheduler_adapter->step({.trigger = jyppx::optim::SchedulerStepMode::epoch,
                         .event_index = 1});
checkpoint_registry.register_participant(scheduler_adapter);
```

Event delivery and checkpoint I/O must be serialized by the training loop.
The legacy `Trainer` is not adapted through this API; a new training loop can
consume the boundary directly without forwarding to legacy optimizer or
scheduler types.
