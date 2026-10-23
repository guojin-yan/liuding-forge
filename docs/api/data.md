# API: Data

All data types live in the `jyppx::liuding::data` namespace. Include everything with
`#include <jyppx/liuding/liuding.hpp>` or the individual headers shown below.

## Dataset

`#include <jyppx/liuding/data/dataset.hpp>`

```cpp
template <typename T = Tensor>
class Dataset {
public:
    using value_type = T;
    virtual T get(size_t index) const = 0;   // one sample
    virtual size_t size() const = 0;         // sample count
    bool empty() const;                      // size() == 0
    T operator[](size_t index) const;        // convenience for get()
    virtual std::vector<T> get_batch(const std::vector<size_t>& indices) const;
};
```

`Dataset` is the map-style base class. `get_batch` can be overridden for
faster batched access; the default loops over `get`. A `DataPipeline` batch
factory may call `get` from multiple worker threads, so such implementations
must be thread-safe.

### TensorDataset

`#include <jyppx/liuding/data/tensor_dataset.hpp>`

```cpp
struct Example {
    Tensor data;
    Tensor target;
};

class TensorDataset : public Dataset<Example> {
public:
    TensorDataset(const Tensor& data, const Tensor& targets); // sizes must match on dim 0
    Example get(size_t index) const override;                 // row slices
    size_t size() const override;
    const Tensor& data() const;
    const Tensor& targets() const;
};
```

## Dataset utilities

### Subset

`#include <jyppx/liuding/data/subset.hpp>`

```cpp
template <typename T = Tensor> class Subset : public Dataset<T> { public:
    Subset(std::shared_ptr<Dataset<T>> dataset, std::vector<size_t> indices);
    T get(size_t index) const override;   // returns dataset->get(indices[index])
    size_t size() const override;
    const std::shared_ptr<Dataset<T>>& dataset() const;
    const std::vector<size_t>& indices() const;
};
```

### ConcatDataset

`#include <jyppx/liuding/data/concat_dataset.hpp>`

```cpp
template <typename T = Tensor>
class ConcatDataset : public Dataset<T> {
public:
    explicit ConcatDataset(std::vector<std::shared_ptr<Dataset<T>>> datasets);
    T get(size_t index) const override;   // continuous index across datasets
    size_t size() const override;
    const std::vector<size_t>& cumulative_sizes() const;
};
```

### random_split

```cpp
// Returns heap-allocated subsets ready for a DataPipeline batch factory.
template <typename DatasetT>
std::vector<std::shared_ptr<Subset<typename DatasetT::value_type>>>
random_split(std::shared_ptr<DatasetT> dataset,
             const std::vector<size_t>& lengths,
             Optional<int64_t> seed = std::nullopt);
```

Mirrors `torch.utils.data.random_split`: indices are shuffled (seeded when a seed is given) and assigned to consecutive subsets. `lengths` must sum to the dataset size.

```cpp
auto subsets = jyppx::liuding::data::random_split(
    dataset, std::vector<size_t>{80, 20}, /*seed=*/42);
// Capture subsets[0] and subsets[1] in independent DataPipeline factories.
```

## DataPipeline

`#include <jyppx/liuding/data/data_pipeline.hpp>`

`DataPipeline` is the explicit-epoch, cursor-aware contract used by the
checkpoint coordinator. `set_epoch(epoch)` is the only operation that creates an epoch order; `next()` never advances an epoch implicitly. Each successful
`next()` commits one batch and returns a `BatchEnvelope` containing the
validated value, complete versioned `BatchSchema`, sample IDs, dataset version,
epoch, and batch sequence.

```cpp
auto sampler = std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(100, 7); jyppx::liuding::data::DataPipeline pipeline(
    sampler,
    {.pipeline_id = "data.train",
     .dataset_version = "train-set/v1",
     .dataset_size = 100,
     .schema = jyppx::liuding::data::BatchSchema::tensor(
         "train-batch", {.dtype = torch::kFloat32, .rank = 2}, 1,
         "features"),
     .batch_factory_id = "stack/v1",
     .batch_size = 16},
    [dataset](std::span<const std::size_t> indices,
              const jyppx::liuding::data::DataWorkerContext& context,
              std::stop_token stop) {
        static_cast<void>(context);
        static_cast<void>(stop);
        std::vector<torch::Tensor> rows;
        for (const auto index : indices) {
            rows.push_back(dataset->get(index));
        }
        return c10::IValue(torch::stack(rows));
    });

pipeline.set_epoch(0); while (const auto batch = pipeline.next()) {
    // A checkpoint after this point resumes at the next uncommitted batch.
}
```

The map-style pipeline supports inline execution or an ordered worker pool. It
provides exact in-epoch restore when the sampler has checkpoint state and a
stable configuration identity, and when the batch factory declares
deterministic replay. The participant stores dataset identity, the complete
canonical batch schema, worker and memory-stage topology, sampler identity,
materialized epoch order, sampler state, next sample cursor, and next batch
sequence. Generated and restored sample IDs are checked against the declared
dataset size. Batch values are memory-staged and schema-validated before the
cursor is committed. A factory, staging, validation, or epoch-preparation
failure leaves committed pipeline state unchanged. The capability report
explicitly exposes worker model, schema, resource limits, and restore bounds.

### Multi-worker loading

Set `num_workers > 0` to load batches on a worker pool with prefetching:

- Output order is identical to single-threaded loading (results are sequenced).
- Logical shard assignment is stable: batch sequence `s` uses
  `worker_id = s % num_workers` and `shard_sequence = s / num_workers`.
These identities do not depend on which physical thread executes the task.
- Exceptions thrown inside `Dataset::get` or the batch factory become a structured `jyppx::liuding::BackendError` on the consuming thread.
- At most `prefetch_batches` batches are scheduled but uncommitted.
- `persistent_workers` controls whether worker threads survive explicit epoch replacement; `set_epoch` cancels any uncommitted work before replacement.
- `Dataset::get` must be thread-safe when `num_workers > 0`.

### Memory staging and CUDA streams

When `BatchMemoryOptions` configures pinning or a target device, background workers validate resource bounds and may pin CPU tensor leaves. They do not perform device transfer. The `next()` consumer transfers the prepared batch and validates its final schema before committing the cursor.

A configured target requires the batch factory to return CPU tensors. CUDA transfer uses the backend current stream of the thread calling `next()`; the pipeline creates no hidden stream or cross-stream event. Set the desired CUDA stream around `next()` when using a custom training stream. Non-blocking transfer requires both a target device and, for asynchronous host-to-device behavior, pinned source memory.

### Exception Handling

`DataPipeline` captures exceptions raised by a batch factory and reports a
stable `jyppx::liuding::BackendError` with component, operation, expected/actual values,
hint, and nested cause. Results remain ordered even when another worker
finishes first:

```cpp
try {
    while (const auto batch = pipeline.next()) {
        train_step(*batch);
    }
} catch (const jyppx::liuding::BackendError& error) {
    std::cerr << "data loading failed: " << error.what() << '\\n';
}
```

Set `num_workers = 0` to debug a failing factory on the calling thread. A
worker exception stops the current epoch without committing the failed batch;
select a fresh epoch or restore committed state after handling it. Dataset
access and custom factories must be safe for concurrent calls when workers are
enabled and must honor the supplied stop token.

### Iterable sources

`IterableDataPipeline` owns one isolated sequential source cursor per explicit
epoch. Its source descriptor declares finiteness, determinism, checkpoint support, exact/bounded/best-effort restore level, and any replay or omission bounds. The participant persists only the committed cursor boundary; prefetched items are rebuilt after restore.

The current schema-v3 writer stores the complete canonical `BatchSchema`. The reader also accepts the explicit adjacent schema-v2 format when its source descriptor and complete topology match. Because v2 stored only a schema label, migration is limited to the provably equivalent unconstrained
`BatchSchema::any(id, version)` contract. Structured schemas require an
explicit external migration. A migrated checkpoint is always rewritten as v3.

### Capability manifest integration

`describe_data_capability()` converts the configured pipeline's existing
`DataCapability` into the foundation schema-v1 `CapabilityDescriptor`. The
entry reports source style, worker/shard/stream policy, device and constrained
dtypes, batch schema identity and fingerprint, restore operation, and numeric
worker/prefetch/memory/schema/replay limits. It does not infer process workers
or backend features that the pipeline did not declare.

Use `make_current_capability_manifest()` to aggregate this entry with other
domain descriptors under one exact `BuildFingerprint`, then serialize it with
`capability_manifest_json()`. Output order is deterministic and numeric limits
remain JSON numbers. The data conversion does not create another checkpoint format or a second source of pipeline capability state.

## Collate functions

`DataBatchFactory` receives the ordered sample-index span, stable worker
context, and cooperative stop token. Capture a dataset to define collation:

```cpp
[](std::span<const std::size_t> indices, const jyppx::liuding::data::DataWorkerContext& context, std::stop_token stop) -> c10::IValue {
    // Fetch, collate, and return one IValue matching the declared BatchSchema.
}
```

## Transforms

`#include <jyppx/liuding/data/transforms/transform.hpp>` (base), `.../compose.hpp`,
`.../vision.hpp`

All transforms derive from `Transform<Input, Output>`; tensor transforms use
the `TensorTransform` alias and implement `Tensor operator()(const Tensor&)`
plus `name()`.

```cpp
using namespace jyppx::liuding::data::transforms;

Compose pipeline({
    std::make_shared<Resize>(std::vector<int64_t>{256, 256}),
    std::make_shared<RandomCrop>(std::vector<int64_t>{224, 224}),
    std::make_shared<RandomHorizontalFlip>(0.5),
    std::make_shared<Normalize>(
        std::vector<double>{0.485, 0.456, 0.406},
        std::vector<double>{0.229, 0.224, 0.225}),
}); auto transformed = pipeline(image);   // {C, H, W} float tensor
```

| Transform | Description |
|-----------|-------------|
| `Normalize(mean, std)` | `(x - mean) / std` per channel; integral inputs are promoted to float |
| `Resize(size, mode)` | bilinear (default) or nearest; `{n}` means square |
| `CenterCrop(size)` | central `{h, w}` region |
| `RandomCrop(size)` | random `{h, w}` region |
| `RandomHorizontalFlip(p)` / `RandomVerticalFlip(p)` | flip with probability `p` |
| `Pad(padding, fill)` | zero/custom-value padding; `{n}` all sides or `{l, r, t, b}` |
| `RandomRotation(degrees)` | random-angle rotation about the center (bilinear, zero fill) |
| `RandomAffine(degrees, translate, scale)` | combined rotate/scale/translate |
| `Grayscale(channels)` / `RandomGrayscale(p)` | ITU-R 601-2 luma conversion |
| `ColorJitter(brightness, contrast, saturation, hue)` | random color adjustments in random order |
| `Compose(transforms)` | apply in order; `name()` lists the chain |
| `RandomChoice(transforms)` | one transform per call, drawn from the global RNG |

`size` accepts `{n}` (square) or `{h, w}`. Random transforms draw from the
global torch RNG. Seed with `torch::manual_seed` for reproducible inline runs. A deterministic multi-worker factory should instead isolate random state through the supplied worker context. Crops and resizes require `{C, H, W}` inputs and throw `std::invalid_argument` on shape mismatches.

## Samplers

`#include <jyppx/liuding/data/samplers/...>`

```cpp
class Sampler { public:
    virtual std::vector<size_t> indices() const = 0;  // one epoch of indices
    virtual size_t size() const = 0;
};

class SequentialSampler : public Sampler;                       // 0..n-1 class RandomSampler : public Sampler;                           // seeded permutation class BatchSampler : public Sampler;                            // flattens + counts batches class WeightedRandomSampler : public Sampler;                   // weights, replacement opt. class SubsetRandomSampler : public Sampler;                     // permuted index subset class DistributedSampler : public Sampler;                      // rank/world_size sharding
```

`RandomSampler(size, seed)` produces a fresh permutation each call
(deterministic across loaders when a seed is given). `BatchSampler(base, batch_size, drop_last)` reports the number of batches via `size()`.

`WeightedRandomSampler(weights, num_samples, replacement, seed)` draws
indices with probability proportional to `weights`. `DistributedSampler(size,
rank, world_size, shuffle, seed, drop_last)` shards the epoch across ranks
(padding to even counts unless `drop_last`); call `set_epoch()` once per
epoch to reshuffle. Pass `rank` and `world_size` explicitly from the external
launcher or job configuration. v1.0 does not publish a process-group API.

Pass a custom sampler to the `DataPipeline` constructor for index strategies
the built-ins do not cover, such as weighted sampling.
