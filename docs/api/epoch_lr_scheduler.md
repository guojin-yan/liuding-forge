# Descriptor-first epoch schedulers

`#include <jyppx/optim/epoch_lr_scheduler.hpp>`

The epoch scheduler family contains `ExponentialLrScheduler`,
`MultiStepLrScheduler`, and `CosineAnnealingLrScheduler`. All three inherit
`LrScheduler`, consume a shared `OptimizerRateControl`, and expose the same
transaction and checkpoint behavior.

## Configuration

- `ExponentialLrSchedulerOptions` stores a stable `scheduler_id` and finite
  non-negative `gamma`.
- `MultiStepLrSchedulerOptions` additionally requires a nonempty sequence of
  positive, strictly increasing, int64-compatible milestones.
- `CosineAnnealingLrSchedulerOptions` stores positive int64-compatible
  `t_max` and a finite non-negative `eta_min` no greater than any captured base
  rate.

Every option, construction-time base rate, optimizer ID and ordered group ID
is encoded in the immutable typed descriptor. Changing any of them makes an
exact restore fail before state mutation.

## Transaction behavior

`step()` first reads and validates all current group rates. The scheduler then
calculates a complete candidate rate vector and next counter without mutation.
Only after that preview succeeds does the optimizer receive one atomic
all-group update; the scheduler counter is committed last through a
non-throwing path.

The exponential and multi-step implementations transform the current rates,
so they compose in `ChainedLrScheduler`. Cosine annealing uses the chainable
recursive schedule, a stable half-angle weight, and an explicit transition
away from each cosine minimum.

## Recovery

Use `LrSchedulerCheckpointParticipant` for all three algorithms. Participant
schema v2 stores the complete descriptor in `topology` and the leaf counter
plus current group rates in `control`. Only the current schema and exact
restore policy are accepted; no legacy map reader or forwarding API exists.
