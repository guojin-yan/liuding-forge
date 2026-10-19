# Tutorial 2: Data Loading

JYPPX uses an explicit-epoch `DataPipeline`. A sampler selects sample IDs,
a `DataBatchFactory` fetches and collates those samples, and every committed
batch carries schema, lineage, epoch, and sequence metadata. The pipeline
never advances to another epoch implicitly.

## Tensor datasets

`TensorDataset` exposes paired input and target tensors as `Example` values.
The following factory turns its samples into the tuple consumed by a typical
training step:

```cpp
auto inputs = torch::rand({100, 3});
auto targets = torch::randint(0, 2, {100});
auto dataset =
    std::make_shared<jyppx::data::TensorDataset>(inputs, targets);
auto sampler =
    std::make_shared<jyppx::data::samplers::RandomSampler>(dataset->size(), 7);

jyppx::data::DataPipeline pipeline(
    std::move(sampler),
    {.pipeline_id = "data.train",
     .dataset_version = "example.train/v1",
     .dataset_size = dataset->size(),
     .schema = jyppx::data::BatchSchema::any("example.train.batch", 1),
     .batch_factory_id = "example.tensor-pair/v1",
     .batch_size = 16},
    [dataset](std::span<const std::size_t> indices,
              const jyppx::data::DataWorkerContext&,
              std::stop_token) {
      std::vector<torch::Tensor> data;
      std::vector<torch::Tensor> targets;
      data.reserve(indices.size());
      targets.reserve(indices.size());
      for (const auto index : indices) {
        const auto sample = dataset->get(index);
        data.push_back(sample.data);
        targets.push_back(sample.target);
      }
      return c10::ivalue::Tuple::create(
          {torch::stack(data), torch::stack(targets)});
    });

pipeline.set_epoch(0);
while (const auto batch = pipeline.next()) {
  const auto& values = batch->value.toTuple()->elements();
  train_step(values.at(0).toTensor(), values.at(1).toTensor());
}
```

The final batch is smaller unless `.drop_last = true`. `set_epoch(epoch)`
selects exactly one sampler order and resets the committed cursor. `next()`
returns an empty optional at exhaustion and does not start another epoch.

## Custom datasets

Derive from `jyppx::data::Dataset<T>` and implement `get()` and `size()`:

```cpp
class MyDataset final : public jyppx::data::Dataset<jyppx::data::Example> {
 public:
  jyppx::data::Example get(std::size_t index) const override {
    return {features_[index], labels_[index]};
  }

  std::size_t size() const override {
    return static_cast<std::size_t>(features_.size(0));
  }

 private:
  torch::Tensor features_ = torch::randn({500, 8});
  torch::Tensor labels_ = torch::randint(0, 3, {500});
};
```

The batch factory owns collation policy. It may return any `c10::IValue` that
matches the declared `BatchSchema`, including a tensor, tuple, list, or
dictionary. Its `batch_factory_id` must change whenever behavior that affects
replay compatibility changes.

## Train and validation splits

`random_split` produces deterministic subset views when given a seed:

```cpp
const auto subsets = jyppx::data::random_split(
    dataset, std::vector<std::size_t>{80, 20}, /*seed=*/42);
```

Create an independent sampler and pipeline for each subset. Give each pipeline
a unique `pipeline_id` and a dataset version that identifies the underlying
data lineage. Training and validation cursors can then participate in the same
checkpoint without identity collisions.

## Worker prefetch

Set `.num_workers` to use the in-process worker pool and set
`.prefetch_batches` to the total number of scheduled but uncommitted batches:

```cpp
jyppx::data::DataPipelineOptions options{
    .pipeline_id = "data.train",
    .dataset_version = "example.train/v1",
    .dataset_size = dataset->size(),
    .schema = jyppx::data::BatchSchema::any("example.train.batch", 1),
    .batch_factory_id = "example.tensor-pair/v1",
    .batch_size = 32,
    .num_workers = 4,
    .prefetch_batches = 8,
    .persistent_workers = true,
    .worker_seed = 7};
```

Worker completion may be out of order, but consumer-visible batches remain in
sampler order. Dataset access and captured factory state must be safe for
concurrent calls. Long-running factories must honor the supplied stop token so
epoch replacement and `close()` remain bounded.

A non-JYPPX exception from a worker is reported as a structured
`jyppx::BackendError` on the consuming thread. The failed batch is not
committed. Use `.num_workers = 0` to reproduce factory failures inline.

## Determinism and checkpointing

`DataWorkerContext` provides stable `epoch`, `sequence`, `worker_id`,
`shard_sequence`, and `worker_seed` values. With `n` workers, batch sequence
`s` belongs to logical shard `s % n` and has shard-local sequence `s / n`.
These identities are independent of physical thread scheduling. A factory
declared deterministic must derive random work from that context rather than
mutable process-global state. The pipeline saves only committed state;
prefetched batches are rebuilt after restore.

Use `SequentialSampler` for fixed order, `RandomSampler` for seeded epoch
permutations, `WeightedRandomSampler` for weighted draws,
`SubsetRandomSampler` for a subset, and `DistributedSampler` for rank-local
sharding. Sampler configuration and state are part of restore compatibility.

## Transforms

Apply transforms inside `Dataset::get()` or the batch factory. Tensor
transforms include normalization, resize, crop, flip, padding, rotation,
affine, grayscale, and color jitter. When workers are enabled, the transform
and all captured state must satisfy the same concurrency and deterministic
replay rules as the rest of the factory.
