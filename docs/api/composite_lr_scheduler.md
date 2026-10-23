# API: Transactional Composite LR Schedulers

The descriptor-first composite API lives in `jyppx::liuding::optim` and is exported by
the focused `jyppx::liuding::optim` CMake target.

```cpp
#include <jyppx/liuding/optim/composite_lr_scheduler.hpp>
```

## Sequential scheduling

`SequentialLrScheduler` uniquely owns an ordered child list. A milestone is the
number of successful parent events assigned to all preceding children. With
milestone `2`, child zero handles events one and two, then child one starts at
event three.

```cpp
std::vector<std::unique_ptr<jyppx::liuding::optim::LrScheduler>> children;
children.push_back(std::make_unique<jyppx::liuding::optim::StepLrScheduler>(
    optimizer, jyppx::liuding::optim::StepLrSchedulerOptions{
                   .scheduler_id = "scheduler.warm",
                   .step_size = 1,
                   .gamma = 0.9}));
children.push_back(std::make_unique<jyppx::liuding::optim::StepLrScheduler>(
    optimizer, jyppx::liuding::optim::StepLrSchedulerOptions{
                   .scheduler_id = "scheduler.decay",
                   .step_size = 10,
                   .gamma = 0.1}));

auto scheduler = std::make_shared<jyppx::liuding::optim::SequentialLrScheduler>(
    optimizer, std::move(children),
    jyppx::liuding::optim::SequentialLrSchedulerOptions{
        .scheduler_id = "scheduler.epochs", .milestones = {5}});
```

## Chained scheduling

`ChainedLrScheduler` previews every child in order. Each child receives the
preceding child's candidate rates. If all previews succeed, the root performs
one atomic all-group optimizer update and then commits every child counter
without throwing. A failing child leaves optimizer, parent, and all children
unchanged.

Children must share the exact optimizer object, use unique scheduler IDs, and
declare one common step mode. Metric-driven children are rejected until the
metric-aware transaction surface is implemented.

## Descriptor and checkpoint

The parent descriptor stores ordered child IDs and complete ordered child
descriptors. Sequential descriptors additionally store typed milestones.
`LrSchedulerCheckpointParticipant` uses participant schema version 3 and
persists the schema-version-1 descriptor tree separately from the mutable state
tree and current optimizer rates.

Restore checks the complete manifest, descriptor, child count/order, state-tree
shape, resource bounds, and numeric values before mutation. It then writes all
optimizer rates once and applies every prevalidated counter. Only the current
versioned storage-neutral format is accepted.
