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

## 分类数据集评估

`<jyppx/liuding/training/classification_evaluation.hpp>` 提供 `training::evaluate_classification(model, source, options)`。`source` 每次返回一批模型参数和一维整数标签，完整读取后返回 `std::nullopt`；模型返回形状为 `N × C` 的有限浮点 logits。数据读取、预处理和设备放置由调用方明确完成，评估器不联网。

结果包含整体和显式来源切片的准确率、宏平均 precision/recall/F1、逐类 support/precision/recall/F1，以及行是真实类别、列是预测类别的混淆矩阵。宏平均包含所有固定类别；logits 并列时取最小类别索引，`ignore_index` 和零分母策略显式配置。空数据或全被忽略的切片返回缺失标量，不记为零分或通过。拆批后汇总使用整数计数，不平均每批指标。

```cpp
training::ClassificationEvaluationOptions options;
options.num_classes = 10;
options.max_samples = 10000;
auto result = training::evaluate_classification(model, next_batch, options);
// next_batch(std::stop_token) -> optional<ClassificationEvaluationBatch>
// 使用 result.observations 作为 experiment::run_evaluation 的实际观测值。
```

配置 `sources` 后，每行必须在 `source_indices` 中指定唯一来源；结果顺序与配置一致。`observations` 自带指标定义、切片上下文和有效样本数，可交给现有 `experiment::run_evaluation` 产生带阈值的质量报告；调用方必须绑定真实的模型、数据、配置和构建身份。

评估关闭梯度并临时切换为 eval，结束或异常后恢复各子模块原来的模式。取消、超时、非法输出和超限均抛异常，不返回部分数据的完整报告。评估期间应独占模型，并保持模块拓扑稳定；自定义 `train()` 必须支持恢复。批次数、样本数、类别、来源和已知混淆矩阵 payload 有显式上限，不保存全量 logits；这些限制不等于模型工作空间、allocator 或 RSS 的硬上限。

## 语义分割 IoU/Dice 与数据集评估

`<jyppx/liuding/training/metrics/segmentation.hpp>` 的 `metrics::SemanticSegmentationMetric` 接受 `N,C,H,W` 有限浮点 logits 或显式指定的 `N,H,W` 整数类别 mask，以及同设备的 `N,H,W` 整数标签。整数类型为 uint8 或 int8/int16/int32/int64。默认按数据集计算 macro IoU；`score=dice` 选择 Dice，`average` 支持 micro、macro、真实像素数加权和仅逐类结果。类别必须映射为 `[0,C)`；不执行阈值、resize、标签重映射或实例匹配。

```cpp
metrics::SemanticSegmentationMetric iou({
    .num_classes = 3,
    .ignore_index = 255,
    .include_background = false});
iou.update(logits, labels);
auto dataset_iou = iou.compute();
auto class_iou = iou.per_class_scores();
```

`ignore_index` 去除对应像素；`include_background=false` 仅从汇总排除 `background_index`，背景与前景之间的错误仍计入前景指标。真实和预测均不存在的类别按 `absent` 选择排除、零、一或报错。空集、全忽略或没有可汇总分母时抛 `NotReady`；逐类被排除的值使用 `std::nullopt`。统计使用每类三个 int64 计数，支持相同身份的 `merge()` 和当前格式的 exact 检查点；拆批、合并或恢复均从总计数计算，不能平均每批 IoU/Dice。每次更新按 `chunk_pixels` 分块，不创建整幅 one-hot；逻辑限额不等于 RSS/显存硬上限。

`<jyppx/liuding/training/segmentation_evaluation.hpp>` 的 `training::evaluate_semantic_segmentation(model, source, options)` 实际逐批推理，返回整体、逐类和显式来源统计；其 `average` 选择 micro、macro 或 weighted。`source` 返回 `SegmentationEvaluationBatch`，以 `std::nullopt` 结束；来源按图像指定，数据准备及设备放置由调用方负责。评估关闭梯度、恢复原模块模式，并在错误、取消、超时或超限时抛异常。`observations` 可接入 `experiment::run_evaluation`；选择与 suite 规则匹配的整体/类别/来源观测值，并绑定真实数据、模型、配置和构建身份。评估器不持久化中途结果，不提供崩溃恢复；检测 AP 与实例 mask AP 不属于此接口。
