# Tutorial 2: Data Loading

This tutorial covers the data layer end to end: `TensorDataset`, custom
datasets, train/validation splits, custom collate functions and
multi-worker loading.

## TensorDataset

The quickest way to feed tensors:

```cpp
auto inputs = torch::rand({100, 3});
auto targets = torch::randint(0, 2, {100});

auto dataset = std::make_shared<jyppx::data::TensorDataset>(inputs, targets);
jyppx::data::DataLoader loader(dataset, {.batch_size = 16});

for (const auto& batch : loader) {
    // batch.data {16, 3}, batch.targets {16}
}
```

The last batch is smaller unless you set `.drop_last = true`. With
`.shuffle = true` and a `.seed`, every run shuffles identically.

## Custom datasets

Derive from `jyppx::data::Dataset<T>` and implement `get` and `size`. To use
a dataset directly with the Trainer, make its sample type
`TensorDataset::Example`:

```cpp
class MyDataset : public jyppx::data::Dataset<jyppx::data::Example> {
public:
    jyppx::data::Example get(size_t index) const override {
        return {features_[index], labels_[index]};   // input row, target row
    }
    size_t size() const override { return features_.size(0); }

private:
    torch::Tensor features_ = torch::randn({500, 8});
    torch::Tensor labels_ = torch::randint(0, 3, {500});
};

jyppx::data::DataLoader loader(std::make_shared<MyDataset>(), {.batch_size = 32});
// batch.data {32, 8}, batch.targets {32}
```

## Collate functions

Datasets with non-Tensor sample types pair with a **collate function**, which
turns a vector of samples into a batch of the same type:

```cpp
struct Sample {
    torch::Tensor features;
    int64_t label;
};

class SampleDataset : public jyppx::data::Dataset<Sample> { /* ... */ };

jyppx::data::DataLoader loader(
    std::make_shared<SampleDataset>(),
    {.batch_size = 32},
    [](const std::vector<Sample>& samples) {
        return jyppx::data::Batch<Sample>{samples};   // keep samples together
    });

for (const auto& batch : loader) {
    // batch.samples is std::vector<Sample> — useful in manual training loops
}
```

The default collate for `T = Tensor` and `T = Example` stacks the samples
with `torch::stack` (that is what `TensorDataset` loaders use). The Trainer's
built-in loop consumes `Tensor` and `Example` batches; other batch types work
in manual loops via `batch.samples`.

## Train/validation splits

`random_split` shuffles indices once (seeded) and hands back subset views:

```cpp
auto subsets = jyppx::data::random_split(
    dataset, std::vector<size_t>{80, 20}, /*seed=*/42);

jyppx::data::DataLoader train_loader(subsets[0], {.batch_size = 8, .shuffle = true});
jyppx::data::DataLoader val_loader(subsets[1], {.batch_size = 8});
```

Subsets are ordinary datasets: compose them with
`jyppx::data::ConcatDataset` to merge datasets, or index them directly for
inspection.

## Multi-worker loading

Set `num_workers` to load batches on a worker pool with prefetching:

```cpp
jyppx::data::DataLoader loader(
    dataset,
    {.batch_size = 64,
     .shuffle = true,
     .num_workers = 4,      // worker threads
     .prefetch_factor = 2,  // batches queued ahead per worker
     .seed = 7});
```

What to expect:

- **Identical output** to single-threaded loading — batches arrive in
  dataset order (or in the seeded shuffle order), so training results do not
  change when you add workers.
- Up to `num_workers * prefetch_factor` batches are prepared ahead of the
  training loop.
- Exceptions raised inside `Dataset::get` or the collate function surface on
  the consuming thread, at the point the failing batch would have arrived.
- Breaking out of a range-for loop early is safe; the next epoch starts
  clean.
- **Requirement**: `Dataset::get` is called from multiple threads, so it must
  be thread-safe. Tensor indexing is; hand-rolled datasets with shared
  mutable state need synchronization.

## Samplers

Built-in samplers (`SequentialSampler`, `RandomSampler`, `BatchSampler`)
cover sequential access, shuffling and batching. A custom sampler plugs into
the fourth DataLoader constructor:

```cpp
class EveryOtherSampler : public jyppx::data::samplers::Sampler {
public:
    explicit EveryOtherSampler(size_t size) : size_(size) {}
    std::vector<size_t> indices() const override {
        std::vector<size_t> result;
        for (size_t i = 0; i < size_; i += 2) result.push_back(i);
        return result;
    }
    size_t size() const override { return (size_ + 1) / 2; }

private:
    size_t size_;
};

jyppx::data::DataLoader loader(
    dataset,
    std::make_shared<EveryOtherSampler>(dataset->size()),
    {.batch_size = 8});
```

## What to read next

- Tutorial 3 pairs the loader with different optimizers and schedulers.
