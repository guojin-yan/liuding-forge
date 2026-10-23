# Descriptor-first event schedulers

`#include <jyppx/liuding/optim/event_lr_scheduler.hpp>`

This family covers schedules whose event semantics are not a plain periodic
epoch decay:

| Type | Step mode | Mutable state |
|------|-----------|---------------|
| `ReduceLrOnPlateauScheduler` | metric | best metric, bad observations, cooldown |
| `OneCycleLrScheduler` | batch | bounded completed-step counter |
| `CyclicLrScheduler` | batch | completed-step counter |
| `LambdaLrScheduler` | epoch | completed-step counter and stable callable identity |

Every transition is previewed completely before the optimizer receives one
atomic all-group learning-rate update. A callback exception, invalid metric,
non-finite output, exhausted one-cycle schedule, checkpoint mismatch, or
resource violation leaves scheduler counters and optimizer rates unchanged.

## Plateau metrics

`ReduceLrOnPlateauScheduler::step(metric)` is the only metric overload.
Direction and threshold mode are enums rather than strings. Minimum learning rates accept an empty vector for zero, one broadcast value, or exactly one value per stable optimizer group. Relative thresholds use the magnitude of the best metric, so negative metrics retain the requested improvement direction.

## Batch schedules

One-cycle construction initializes each group to `max_lr / div_factor`. Exactly `total_steps` calls are accepted and the last reaches the non-negative final rate; the implementation does not reproduce PyTorch's historical post-boundary negative linear extrapolation. Cyclic scheduling supports triangular, triangular2, exp-range, and identity-bearing custom scales.

Momentum cycling is not silently emulated. Setting `cycle_momentum` throws
`NotImplemented` until the optimizer boundary can atomically update and
checkpoint learning rates plus momentum/beta values in one transaction.

## Callable identity

`LambdaLrScheduler` and custom cyclic scaling require stable callable IDs in
their descriptors. Function objects are process-local executable behavior and are not serialized. Applications must register the same semantics under the same ID before exact restore; a different ID produces a topology mismatch.

The generic `LrSchedulerCheckpointParticipant` schema version 3 stores typed mutable control values in every state-tree node. It is current-format only; old scalar maps and participant schema version 2 require an explicit external migration rather than an implicit reader.
