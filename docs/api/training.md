# API: Training

Training types live in `jyppx::liuding::training`. The current API is contract-first: the loop does not infer model calls, targets, objectives, checkpoint contents, or scheduler timing.

## Trainer

Include `jyppx/training/trainer.hpp` and construct an unprepared Trainer from stable run options:

```cpp
jyppx::liuding::training::Trainer trainer({
    .trainer_id = "trainer.main",
    .run_id = "run.2026-10-17",
    .trace_id = "trace.001",
    .max_epochs = 20,
    .max_steps = 0,
    .accumulate_steps = 4,
    .accumulation_tail =
        jyppx::liuding::training::AccumulationTailPolicy::flush,
    .gradient_clip_norm = 1.0,
});
```

`max_steps == 0` means epoch-limited. A partial accumulation tail is either
explicitly rescaled and flushed or dropped. Checkpoint state records these
policies and rejects a different topology during restore.

## FitRequest

`prepare(FitRequest)` binds one complete execution and recovery domain:

- `ModelAdapter` validates invocation and result schemas.
- Each `OptimizerExecutionBinding` owns one optimizer, its exact checkpoint participant, and an optional scheduler.
- `BatchPipeline` accepts either map-style `DataPipeline` or
  `IterableDataPipeline` without inventing random access for streams.
- `BatchMapper`, optional `TargetMapper`, and `StepFunction` own task semantics.
- `TrainingStrategy` owns placement, backward, zero-grad, updates and barriers.
- model, every optimizer and RNG checkpoint participant are required.
- scheduler, callbacks, event outbox and additional task participants are
  optional but become part of the same coordinated checkpoint.

`SingleDeviceStrategy::prepare()` canonicalizes optimizers by stable identity,
then preflights their parameter ownership before moving model state. Optimizer parameter sets must be disjoint and together cover every named trainable model parameter exactly once. Each optimizer must have zero completed steps and no native state; after preparation and the first update, dtype/device placement is immutable.

```cpp
request.optimizers = {
    {.optimizer = encoder_optimizer,
     .checkpoint = encoder_optimizer_state,
     .scheduler = encoder_scheduler},
    {.optimizer = head_optimizer,
     .checkpoint = head_optimizer_state,
     .scheduler = head_scheduler},
};
```

One logical Trainer step applies every optimizer once in canonical identity order and advances `global_step` only after the complete group succeeds. Backend exceptions during an optimizer or scheduler call are fail-stop: the Trainer enters `failed` and cannot advertise a checkpoint-safe continuation; the framework does not claim rollback of arbitrary backend side effects.

The training step returns a differentiable scalar loss and the exact number of committed sample IDs. Validation may omit loss, but a metric scheduler then requires the configured `scheduler_metric` to be present in validation output.

```cpp
auto step = [](const jyppx::liuding::training::StepInput& input) {
    auto prediction = input.model.invoke(input.invocation).value.toTensor();
    auto target = input.target->toTensor();
    return jyppx::liuding::training::StepOutput{
        .loss = torch::mse_loss(prediction, target),
        .metrics = {{"task.mae", (prediction - target).abs().mean().item<double>()}},
        .sample_count = input.batch.sample_ids.size(),
    };
};
```

`fit()` runs once from a prepared or restored state and returns `FitResult`.
The loop rejects non-finite loss/gradients, empty training epochs, external
optimizer/scheduler steps, shared training/validation cursors and incomplete
checkpoint domains. `close()` explicitly drains events and closes data and
strategy resources.

## Checkpoints

Trainer checkpointing is storage-neutral:

```cpp
trainer.save_checkpoint(my_transactional_sink);

// A fresh, topology-identical domain: restored.prepare(make_request()); restored.restore_checkpoint(my_immutable_source); auto result = restored.fit();
```

The `CheckpointCoordinator` saves every registered participant in canonical ID
order. Restore preflights descriptors, snapshots rollback state, and restores
participants transactionally. Current schema readers do not route old Trainer
archives through compatibility forwarding. The configured restore policy is
used unless the caller explicitly supplies another policy.

## Callbacks and Events

Callbacks implement `CallbackHandler` and receive immutable `CallbackEvent`
values. They cannot mutate Trainer. A descriptor declares subscriptions,
ordering, statefulness and the only commands the callback may emit:

- `skip_validation`
- `request_checkpoint`
- `stop`

`CallbackDispatcher` resolves commands deterministically with fixed priority
`stop > request_checkpoint > skip_validation`. Stateful callbacks are separate
checkpoint participants. `EarlyStoppingCallback` is the built-in stateful
handler and monitors `validation.loss` by default.

`EventOutbox` stages committed callback audit records for one or more sinks.
Sink delivery happens only after callback state and the dispatcher cursor have committed.

## Schedulers

Wrap a descriptor-first `optim::LrScheduler` in `SchedulerAdapter`. The adapter accepts only its declared batch, epoch or metric event and rejects gaps, duplicates and unexpected metrics. Trainer is the sole scheduler event owner.

## Metrics

Focused metric headers live under `jyppx/training/metrics`. Classification and ranking metrics define explicit input, aggregation and checkpoint semantics. Register stateful metric objects through `FitRequest::additional_participants` when they must resume exactly. Per-batch scalar observations returned by
`StepOutput` are weighted by `sample_count` for the epoch result.

## Mixed Precision

The AMP module exposes `AutocastGuard`, `GradScaler`, and
`GradScalerCheckpointParticipant`. `TrainerOptions::precision` selects full,
CPU bfloat16, CUDA float16, or CUDA bfloat16 execution. CUDA float16 requires an enabled scaler plus its checkpoint participant. Trainer unscales every optimizer before deciding the cycle; one non-finite gradient skips the entire optimizer group, updates the scaler once, and does not advance `global_step` or batch schedulers. There is no legacy `use_amp` compatibility switch.

## Loss Functions

Focused loss headers live under `jyppx/nn/loss`. Focal, contrastive, Dice and label-smoothing cross entropy expose functional and module APIs with explicit shape, dtype, reduction and task-mode contracts. Knowledge distillation remains a task recipe because the caller must choose hard/soft composition, temperature, teacher gradient policy and aggregation.
