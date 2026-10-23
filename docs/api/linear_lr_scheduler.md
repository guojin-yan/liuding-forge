# Descriptor-first Linear-family Schedulers

`#include <jyppx/liuding/optim/linear_lr_scheduler.hpp>`

The linear family contains `LinearLrScheduler`, `WarmupLrScheduler`,
`ConstantLrScheduler`, and `PolynomialLrScheduler`. They all use the
descriptor-first `LrScheduler` contract, atomic all-group LR transitions, and
the exact `LrSchedulerCheckpointParticipant`.

## Construction semantics

- `LinearLrScheduler` applies `start_factor` immediately, interpolates toward
  `end_factor` for `total_iters` events, and holds the end factor afterward.
  `total_iters = 0` reaches the end factor on the first event.
- `WarmupLrScheduler` applies `start_factor` immediately, reaches the captured
  base rates at `warmup_iters`, and holds them afterward. Compose a later
  scheduler with `SequentialLrScheduler`; the warmup object does not own a
  mutable child.
- `ConstantLrScheduler` applies `factor` immediately, retains it through the
  first `total_iters` events, and reaches base rates on the next event.
- `PolynomialLrScheduler` keeps base rates at construction and applies
  `pow(max(0, 1 - step / total_iters), power)` after each event. It clamps at
  its terminal factor and accepts `power = 0` as a constant schedule.

All factors and exponents are finite and positive/non-negative as documented;
event intervals are stored as int64-compatible typed descriptor values. The
descriptor also contains the complete construction-time base-rate vector and
stable optimizer/group topology.

## Recovery

The scheduler participant stores the descriptor tree in `topology` and the
counter/current rates in `control`. Restore requires the same scheduler type,
typed options, base rates, optimizer identity, and ordered group IDs. The
current schema is exact-only and does not read legacy map state.
