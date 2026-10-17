# Tutorial 1: Basic Training

JYPPX separates task semantics from loop mechanics. A complete run explicitly
binds model invocation, data, target extraction, loss, strategy and recovery
state.

## 1. Adapt the model

Ordinary LibTorch modules are wrapped in `ModelAdapter` so input and output
schemas are checked at the framework boundary:

```cpp
auto holder = torch::nn::Linear(torch::nn::LinearOptions(2, 1));
auto model = std::make_shared<jyppx::nn::ModelAdapter>(
    jyppx::nn::adapt_tensor_holder(holder));
```

Use explicit `TensorSchema` values in production when rank, dtype or fixed
dimensions must be enforced.

## 2. Create a committed batch pipeline

`DataPipeline` combines a sampler, stable dataset/collator identity and a
factory. The factory returns an opaque `IValue`; Trainer consumes the common
`BatchPipeline` interface shared with `IterableDataPipeline`.

```cpp
auto pipeline = std::make_shared<jyppx::data::DataPipeline>(
    std::make_shared<jyppx::data::samplers::SequentialSampler>(sample_count),
    jyppx::data::DataPipelineOptions{
        .pipeline_id = "data.train",
        .dataset_version = "linear/v1",
        .dataset_size = sample_count,
        .batch_factory_id = "linear.tensor-slice/v1",
        .batch_size = 8,
    },
    batch_factory);
```

Create a separate pipeline instance for validation. Sharing the training cursor
is rejected because the two participant states cannot resume independently.

## 3. Bind optimizer, strategy and checkpoint state

Build `OptimizerParameterBinding` entries from the adapter's named parameters,
then create a descriptor-first optimizer. A `SingleDeviceStrategy` owns device
placement, backward and optimizer updates. The FitRequest also requires model,
optimizer and RNG checkpoint participants so exact recovery cannot omit primary
state.

## 4. Define task semantics

The batch mapper produces an `Invocation`; the optional target mapper extracts
the target; the step function performs forward and returns loss/metrics:

```cpp
auto step = [](const jyppx::training::StepInput& input) {
    auto prediction = input.model.invoke(input.invocation).value.toTensor();
    auto target = input.target->toTensor();
    return jyppx::training::StepOutput{
        .loss = (prediction - target).pow(2).mean(),
        .sample_count = input.batch.sample_ids.size(),
    };
};
```

Training loss must be a finite differentiable scalar. `sample_count` must match
the committed sample IDs exactly.

## 5. Prepare, fit and close

```cpp
jyppx::training::Trainer trainer({
    .trainer_id = "trainer.linear",
    .run_id = "run.linear",
    .trace_id = "trace.linear",
    .max_epochs = 20,
});
trainer.prepare(std::move(request));
auto result = trainer.fit();
std::cout << result.training_metrics.at("training.loss") << '\n';
trainer.close();
```

`FitResult` reports committed epochs, optimizer steps, final training and
validation metric maps, and whether the run completed, stopped, or was
cancelled. Exceptions put Trainer in `failed`; use a fresh domain or restore a
checkpoint rather than continuing mutated backend state.

The repository's `examples/basic_trainer.cpp` contains the complete compilable
program. Tutorial 2 covers data sources, Tutorial 3 covers schedulers, and
Tutorial 4 covers immutable callbacks and coordinated checkpoints.
