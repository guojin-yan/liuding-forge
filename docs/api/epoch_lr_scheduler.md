# Descriptor-first epoch schedulers

`#include <jyppx/liuding/optim/epoch_lr_scheduler.hpp>`

The epoch scheduler family contains `ExponentialLrScheduler`,
`MultiStepLrScheduler`, `CosineAnnealingLrScheduler`, and
`CosineAnnealingWarmRestartsLrScheduler`. All four inherit
`LrScheduler`, consume a shared `OptimizerRateControl`, and expose the same
transaction and checkpoint behavior.

## Configuration

- `ExponentialLrSchedulerOptions` stores a stable `scheduler_id` and finite non-negative `gamma`.
- `MultiStepLrSchedulerOptions` additionally requires a nonempty sequence of positive, strictly increasing, int64-compatible milestones.
- `CosineAnnealingLrSchedulerOptions` stores positive int64-compatible
  `t_max` and a finite non-negative `eta_min` no greater than any captured base
  rate.
- `CosineAnnealingWarmRestartsLrSchedulerOptions` adds positive
  int64-compatible `t_0` and `t_mult`. Its exact state includes a cycle index,
  active cycle length, and position that must agree with `completed_steps`.

Every option, construction-time base rate, optimizer ID and ordered group ID
is encoded in the immutable typed descriptor. Changing any of them makes an
exact restore fail before state mutation.

## Transaction behavior

`step()` first reads and validates all current group rates. The scheduler then
calculates a complete candidate rate vector and next counter without mutation. Only after that preview succeeds does the optimizer receive one atomic all-group update; the scheduler counter is committed last through a non-throwing path.

The exponential and multi-step implementations transform the current rates, so they compose in `ChainedLrScheduler`. Cosine annealing uses the chainable recursive schedule, a stable half-angle weight, and an explicit transition away from each cosine minimum. Warm restarts calculate rates from captured base rates, reset every group at a cycle boundary, and grow the next cycle transactionally. A cycle-length or cycle-index overflow is rejected before optimizer or scheduler mutation.

## Recovery

Use `LrSchedulerCheckpointParticipant` for all four algorithms. Participant schema v3 stores the complete descriptor in `topology` and the typed leaf state plus current group rates in `control`. Only the current schema and exact restore policy are accepted; no legacy map reader or forwarding API exists.
