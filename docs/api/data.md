# API: Data

All data types live in the `jyppx::data` namespace. Include everything with
`#include <jyppx/jyppx.hpp>` or the individual headers shown below.

## Dataset

`#include <jyppx/data/dataset.hpp>`

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
faster batched access; the default loops over `get`. When a DataLoader runs
with `num_workers > 0`, `get` is called from multiple threads and **must be
thread-safe**.

### TensorDataset

`#include <jyppx/data/tensor_dataset.hpp>`

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

`#include <jyppx/data/subset.hpp>`

```cpp
template <typename T = Tensor>
class Subset : public Dataset<T> {
public:
    Subset(std::shared_ptr<Dataset<T>> dataset, std::vector<size_t> indices);
    T get(size_t index) const override;   // returns dataset->get(indices[index])
    size_t size() const override;
    const std::shared_ptr<Dataset<T>>& dataset() const;
    const std::vector<size_t>& indices() const;
};
```

### ConcatDataset

`#include <jyppx/data/concat_dataset.hpp>`

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
// Returns heap-allocated subsets, ready for DataLoader consumption.
template <typename DatasetT>
std::vector<std::shared_ptr<Subset<typename DatasetT::value_type>>>
random_split(std::shared_ptr<DatasetT> dataset,
             const std::vector<size_t>& lengths,
             Optional<int64_t> seed = std::nullopt);
```

Mirrors `torch.utils.data.random_split`: indices are shuffled (seeded when a
seed is given) and assigned to consecutive subsets. `lengths` must sum to the
dataset size.

```cpp
auto subsets = jyppx::data::random_split(dataset, std::vector<size_t>{80, 20}, /*seed=*/42);
jyppx::data::DataLoader train_loader(subsets[0], {.batch_size = 8});
jyppx::data::DataLoader val_loader(subsets[1], {.batch_size = 8});
```

## DataLoader

`#include <jyppx/data/dataloader.hpp>`

```cpp
struct DataLoaderOptions {
    size_t batch_size = 1;
    bool shuffle = false;
    size_t num_workers = 0;        // worker threads (0 = calling thread)
    bool drop_last = false;
    size_t prefetch_factor = 2;    // batches each worker keeps queued ahead
    Optional<int64_t> seed;        // seed for the shuffling sampler
};

template <typename T = Tensor>
class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset<T>> dataset,
               const DataLoaderOptions& options = {});
    DataLoader(std::shared_ptr<Dataset<T>> dataset,
               const DataLoaderOptions& options,
               CollateFn<T> collate_fn);
    DataLoader(std::shared_ptr<Dataset<T>> dataset,
               std::shared_ptr<samplers::Sampler> sampler,
               const DataLoaderOptions& options = {},
               CollateFn<T> collate_fn = {});

    class Iterator;               // input iterator over batches
    Iterator begin();             // starts a (fresh) epoch
    Iterator end();

    size_t size() const;          // number of batches per epoch
    size_t dataset_size() const;
    const DataLoaderOptions& options() const;
    void reset();                 // start a new epoch on the next begin()
    DataLoaderState state_dict() const;
    void load_state_dict(const DataLoaderState& state);
};
```

The sample type is deduced from the dataset's `value_type`
(`TensorDataset` yields `DataLoader<Example>`). Batches are `Batch<T>`:

- `Batch<Example>` has `.data` and `.targets` tensors;
- `Batch<Tensor>` has `.data` (default collate stacks samples).

```cpp
jyppx::data::DataLoader loader(dataset, {.batch_size = 4, .shuffle = true, .seed = 7});
for (const auto& batch : loader) {
    // batch.data {N, ...}, batch.targets {N, ...}
}
```

`reset()` prepares exactly one epoch; the following `begin()` consumes that
prepared order instead of advancing the sampler again. `state_dict()` is an
epoch-boundary snapshot of the current-version loader and sampler state.
`load_state_dict()` validates the dataset and batching topology, prepared
indices, sampler capability, and exact format version before applying the
state. A failed restore rolls the sampler, prepared epoch, and worker pipeline
back to their entry state. Custom samplers must explicitly implement
`supports_state()`, `save_state()`, and `load_state()` to participate.

Weighted sampling without replacement uses a bounded exponential-race draw.
It rejects non-finite or negative weights, a non-positive total, and requests
larger than the number of positive-weight entries, so zero-weight inputs cannot
cause a non-terminating retry loop.

## DataPipeline

`#include <jyppx/data/data_pipeline.hpp>`

`DataPipeline` is the new explicit-epoch, cursor-aware contract used by the
checkpoint coordinator. `set_epoch(epoch)` is the only operation that creates
an epoch order; `next()` never advances an epoch implicitly. Each successful
`next()` commits one batch and returns a `BatchEnvelope` containing the opaque
value, stable schema ID, sample IDs, dataset version, epoch, and batch sequence.

```cpp
auto sampler = std::make_shared<jyppx::data::samplers::RandomSampler>(100, 7);
jyppx::data::DataPipeline pipeline(
    sampler,
    {.pipeline_id = "data.train",
     .dataset_version = "train-set/v1",
     .dataset_size = 100,
     .schema_id = "train-batch/v1",
     .batch_size = 16},
    [dataset](std::span<const std::size_t> indices) {
        std::vector<torch::Tensor> rows;
        for (const auto index : indices) {
            rows.push_back(dataset->get(index));
        }
        return c10::IValue(torch::stack(rows));
    });

pipeline.set_epoch(0);
while (const auto batch = pipeline.next()) {
    // A checkpoint after this point resumes at the next uncommitted batch.
}
```

The current target is deliberately map-style and single-threaded. It provides
exact in-epoch restore when the sampler exposes both checkpoint state and a
stable configuration identity. The participant stores dataset/schema identity,
dataset size, sampler identity, the materialized epoch order, sampler state,
next sample cursor, and next batch sequence. Generated and restored sample IDs
are checked against the declared dataset size. A batch-factory exception or
failed epoch preparation does not advance committed pipeline state. Samplers
without a stable identity are reported as bounded restore rather than exact restore.
Worker seed, shard assignment, in-flight prefetch, iterable watermarks, and
bounded replay remain separate follow-up contracts; the capability report does
not claim them.

### Multi-worker loading

Set `num_workers > 0` to load batches on a worker pool with prefetching:

- Output order is identical to single-threaded loading (results are sequenced).
- Exceptions thrown inside `Dataset::get` or the collate function are
  rethrown on the consuming thread when their batch is reached.
- At most `num_workers * prefetch_factor` batches are in flight.
- Workers restart per epoch; breaking out of a range-for loop early is safe.
- `Dataset::get` must be thread-safe when `num_workers > 0`.

### Exception Handling

`DataLoader` captures exceptions raised by `Dataset::get` or the collate
function in a worker and rethrows the original exception on the consuming
thread when that batch reaches the front of the output stream. Results remain
ordered even when another worker finishes first:

```cpp
try {
    for (const auto& batch : loader) {
        train_step(batch);
    }
} catch (const std::runtime_error& error) {
    std::cerr << "data loading failed: " << error.what() << '\\n';
}
```

Standard exceptions and LibTorch `c10::Error` preserve their type and message.
Set `num_workers = 0` to debug a failing dataset in the calling thread. A
worker exception ends the current iteration; create a fresh epoch after
handling it. Dataset access and custom collate functions must be safe for
concurrent calls when workers are enabled.

## Collate functions

`CollateFn<T>` is `std::function<Batch<T>(const std::vector<T>&)>`. Provide
one to control batching for custom sample types:

```cpp
jyppx::data::DataLoader loader(
    dataset,
    {.batch_size = 4},
    [](const std::vector<MySample>& samples) {
        // build Batch<MySample> from the samples
    });
```

The default collate for `T = Tensor` and `T = Example` stacks the samples
with `torch::stack`.

## Transforms

`#include <jyppx/data/transforms/transform.hpp>` (base), `.../compose.hpp`,
`.../vision.hpp`

All transforms derive from `Transform<Input, Output>`; tensor transforms use
the `TensorTransform` alias and implement `Tensor operator()(const Tensor&)`
plus `name()`.

```cpp
using namespace jyppx::data::transforms;

Compose pipeline({
    std::make_shared<Resize>(std::vector<int64_t>{256, 256}),
    std::make_shared<RandomCrop>(std::vector<int64_t>{224, 224}),
    std::make_shared<RandomHorizontalFlip>(0.5),
    std::make_shared<Normalize>(
        std::vector<double>{0.485, 0.456, 0.406},
        std::vector<double>{0.229, 0.224, 0.225}),
});
auto transformed = pipeline(image);   // {C, H, W} float tensor
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
global torch RNG — seed with `torch::manual_seed` for reproducible runs; the
draws are safe under the multi-worker DataLoader. Crops and resizes require
`{C, H, W}` inputs and throw `std::invalid_argument` on shape mismatches.

## Samplers

`#include <jyppx/data/samplers/...>`

```cpp
class Sampler {
public:
    virtual std::vector<size_t> indices() const = 0;  // one epoch of indices
    virtual size_t size() const = 0;
};

class SequentialSampler : public Sampler;                       // 0..n-1
class RandomSampler : public Sampler;                           // seeded permutation
class BatchSampler : public Sampler;                            // flattens + counts batches
class WeightedRandomSampler : public Sampler;                   // weights, replacement opt.
class SubsetRandomSampler : public Sampler;                     // permuted index subset
class DistributedSampler : public Sampler;                      // rank/world_size sharding
```

`RandomSampler(size, seed)` produces a fresh permutation each call
(deterministic across loaders when a seed is given). `BatchSampler(base,
batch_size, drop_last)` reports the number of batches via `size()`.

`WeightedRandomSampler(weights, num_samples, replacement, seed)` draws
indices with probability proportional to `weights`. `DistributedSampler(size,
rank, world_size, shuffle, seed, drop_last)` shards the epoch across ranks
(padding to even counts unless `drop_last`); call `set_epoch()` once per
epoch to reshuffle, and wire `distributed::get_rank()`/`get_world_size()`
from the process group (single-process reference: rank 0, world size 1).

Pass a custom sampler to the fourth DataLoader constructor for index
strategies the built-ins do not cover (e.g. weighted sampling).

## ThreadSafeQueue

`#include <jyppx/data/thread_safe_queue.hpp>`

`ThreadSafeQueue<T>` is the bounded blocking queue used by the multi-worker
pipeline (blocking `put`/`get`, non-blocking `try_get`, `close` semantics
that keep already-queued items retrievable). It is public API and generally
useful for producer/consumer code.
