# API：数据

数据类型位于 `jyppx::liuding::data` 命名空间。可包含 `#include <jyppx/liuding/liuding.hpp>` 或下列独立头文件。

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

`Dataset` is the map-style base class. `get_batch` can be overridden for faster batched access; the default loops over `get`. A `DataPipeline` batch factory may call `get` from multiple worker threads, so such implementations must be thread-safe.

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

## 数据集视图

`Subset` 按给定索引顺序共享源数据集，可保留重复索引；`ConcatDataset` 和 `ChainDataset` 拼接同类 map 数据集，合计长度超过 `size_t` 范围时在构造阶段抛出结构化 `InvalidArgument`，不会让累计偏移回绕。`StackDataset` 取各数据集最短公共前缀，拒绝空数据集；合法的最大 `size_t` 长度不会被当作空值。这些视图不按逻辑样本总数分配存储。

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

`random_split` 打乱全部索引，再按给定长度分配到各子集。`lengths` 的数学总和须等于数据集大小；在分配索引数组前逐项检查剩余数量，拒绝超量、少量及整数回绕输入，错误组件为 `data.random_split`、操作为 `lengths`。允许长度为零的子集。固定 seed 在同一标准库环境内复现顺序；这里使用 `std::shuffle`，不承诺与下文可移植 `RandomSampler` 相同的跨标准库顺序。该操作需要 O(N) 索引存储，逻辑数据集视图本身的轻量性不等于拆分可处理任意容量。

```cpp
auto subsets = jyppx::liuding::data::random_split(
    dataset, std::vector<size_t>{80, 20}, /*seed=*/42);
// Capture subsets[0] and subsets[1] in independent DataPipeline factories.
```

## DataPipeline

`#include <jyppx/liuding/data/data_pipeline.hpp>`

`DataPipeline` is the explicit-epoch, cursor-aware contract used by the checkpoint coordinator. `set_epoch(epoch)` is the only operation that creates an epoch order; `next()` never advances an epoch implicitly. Each successful `next()` commits one batch and returns a `BatchEnvelope` containing the validated value, complete versioned `BatchSchema`, sample IDs, dataset version, epoch, and batch sequence.

`BatchEnvelope` 的析构和整体赋值先释放原批次值，再归还对应的资源与生产准入租约；复制共享值和租约，移动转移所有权。自行保留 `value` 或其中 Tensor 的别名时，也须保留对应 envelope 或租约。该类型使用默认构造后逐字段设置，不能使用聚合初始化；检查点格式保持原样。

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

The map-style pipeline supports inline execution or an ordered worker pool. It provides exact in-epoch restore when the sampler has checkpoint state and a stable configuration identity, and when the batch factory declares deterministic replay. The participant stores dataset identity, the complete canonical batch schema, worker and memory-stage topology, sampler identity, materialized epoch order, sampler state, next sample cursor, and next batch sequence. Generated and restored sample IDs are checked against the declared dataset size. Batch values are memory-staged and schema-validated before the cursor is committed. A factory, staging, validation, or epoch-preparation failure leaves committed pipeline state unchanged. The capability report explicitly exposes worker model, schema, resource limits, and restore bounds.

### Multi-worker loading

Set `num_workers > 0` to load batches on a worker pool with prefetching:

- Output order is identical to single-threaded loading (results are sequenced).
- Logical shard assignment is stable: batch sequence `s` uses
`worker_id = s % num_workers` and `shard_sequence = s / num_workers`. These identities do not depend on which physical thread executes the task.
- Exceptions thrown inside `Dataset::get` or the batch factory become a structured `jyppx::liuding::BackendError` on the consuming thread.
- At most `prefetch_batches` batches are scheduled but uncommitted.
- `persistent_workers` controls whether worker threads survive explicit epoch replacement; `set_epoch` cancels any uncommitted work before replacement.
- `Dataset::get` must be thread-safe when `num_workers > 0`.

### Memory staging and CUDA streams

When `BatchMemoryOptions` configures pinning or a target device, background workers validate resource bounds and may pin CPU tensor leaves. They do not perform device transfer. The `next()` consumer transfers the prepared batch, runs the optional consumer transform, and validates its final schema before committing the cursor.

A configured target requires the batch factory to return CPU tensors. CUDA transfer uses the backend current stream of the thread calling `next()`; the pipeline creates no hidden stream or cross-stream event. Set the desired CUDA stream around `next()` when using a custom training stream. Non-blocking transfer requires both a target device and, for asynchronous host-to-device behavior, pinned source memory.

### Exception Handling

`DataPipeline` captures exceptions raised by a batch factory and reports a stable `jyppx::liuding::BackendError` with component, operation, expected/actual values, hint, and nested cause. Results remain ordered even when another worker finishes first:

```cpp
try {
    while (const auto batch = pipeline.next()) {
        train_step(*batch);
    }
} catch (const jyppx::liuding::BackendError& error) {
    std::cerr << "data loading failed: " << error.what() << '\\n';
}
```

Set `num_workers = 0` to debug a failing factory on the calling thread. A worker exception stops the current epoch without committing the failed batch; select a fresh epoch or restore committed state after handling it. Dataset access and custom factories must be safe for concurrent calls when workers are enabled and must honor the supplied stop token.

### Iterable sources

`IterableDataPipeline` owns one isolated sequential source cursor per explicit epoch. Its source descriptor declares finiteness, determinism, checkpoint support, exact/bounded/best-effort restore level, and any replay or omission bounds. The participant persists only the committed cursor boundary; prefetched items are rebuilt after restore.

当前 map 写 v5、iterable 写 v4，保存 consumer transform 身份、完整输入/输出 schema 和确定性声明；map v4 / iterable v3 的无 consumer 完整结构格式继续可读。更早的 map v3 / iterable v2 仅按已有约束迁移同身份的 `BatchSchema::any`。新启用的 transform 不会被静默加到旧 checkpoint。接口、失败终止、显式恢复和内存统计见[提交前批次变换](batch_consumer_transform.md)。

### Capability manifest integration

`describe_data_capability()` converts the configured pipeline's existing `DataCapability` into the foundation schema-v1 `CapabilityDescriptor`. The entry reports source style, worker/shard/stream policy, device and constrained dtypes, batch schema identity and fingerprint, restore operation, and numeric worker/prefetch/memory/schema/replay limits. It does not infer process workers or backend features that the pipeline did not declare.

Use `make_current_capability_manifest()` to aggregate this entry with other domain descriptors under one exact `BuildFingerprint`, then serialize it with `capability_manifest_json()`. Output order is deterministic and numeric limits remain JSON numbers. The data conversion does not create another checkpoint format or a second source of pipeline capability state.

## Collate functions

`DataBatchFactory` receives the ordered sample-index span, stable worker context, and cooperative stop token. Capture a dataset to define collation:

```cpp
[](std::span<const std::size_t> indices, const jyppx::liuding::data::DataWorkerContext& context, std::stop_token stop) -> c10::IValue {
    // Fetch, collate, and return one IValue matching the declared BatchSchema.
}
```

## Transforms

`#include <jyppx/liuding/data/transforms/transform.hpp>` (base), `.../compose.hpp`, `.../vision.hpp`

All transforms derive from `Transform<Input, Output>`; tensor transforms use the `TensorTransform` alias and implement `Tensor operator()(const Tensor&)` plus `name()`.

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

`size` accepts `{n}` (square) or `{h, w}`. Random transforms draw from the global torch RNG. Seed with `torch::manual_seed` for reproducible inline runs. A deterministic multi-worker factory should instead isolate random state through the supplied worker context. Crops and resizes require `{C, H, W}` inputs and throw `std::invalid_argument` on shape mismatches.

## Samplers

采样器位于 `jyppx::liuding::data::samplers`，按所用类型包含对应头文件；下列具体采样器均继承 `Sampler`。

| 头文件 | 类型 |
| --- | --- |
| `<jyppx/liuding/data/samplers/sequential_sampler.hpp>` | `SequentialSampler` |
| `<jyppx/liuding/data/samplers/random_sampler.hpp>` | `RandomSampler` |
| `<jyppx/liuding/data/samplers/batch_sampler.hpp>` | `BatchSampler` |
| `<jyppx/liuding/data/samplers/weighted_random_sampler.hpp>` | `WeightedRandomSampler`、`SubsetRandomSampler`、`DistributedSampler` |

```cpp
class Sampler { public:
    virtual std::vector<size_t> indices() const = 0;  // one epoch of indices
    virtual size_t size() const = 0;
};

class SequentialSampler;      // 0..n-1
class RandomSampler;          // seeded permutation
class BatchSampler;           // flat indices; size counts batches
class WeightedRandomSampler;  // weights, replacement opt.
class SubsetRandomSampler;    // permuted index subset
class DistributedSampler;     // rank/world_size sharding
```

`RandomSampler(size, seed)` 每次返回覆盖全部索引的随机排列。默认算法为 `splitmix64-fisher-yates-rejection/v1`：64 位无符号 SplitMix64、从后向前的 Fisher–Yates，以及拒绝取模偏差的有界采样；不依赖标准库的 `std::shuffle` 或 distribution。同一 population、seed、epoch 和调用顺序在不同标准库上产生相同索引；`set_epoch(e)` 将状态重置为 `uint64(seed) + e`，按 2⁶⁴ 取模。未传 seed 时使用系统随机源，并在状态中保存实际 base seed。该保证不包含不同设备或平台上的浮点训练结果。

`RandomSampler::algorithm_id()` 返回算法版本，`configuration_identity()` 同时包含算法和 population。`save_state()` 使用 `JYPPX-RANDOM-SAMPLER/v2`、population、base seed 和当前 64 位状态四行 ASCII（LF 分隔，无结尾换行）；`load_state()` 在完整验证后才修改实例，拒绝不同 population、非规范数值和旧 `std::shuffle` v1 状态。旧运行需要原版本继续恢复，新算法使用新的运行目录。该可移植保证仅适用于 `RandomSampler`，不自动扩展到 weighted、subset 或 distributed sampler。`BatchSampler(base, batch_size, drop_last)` 通过 `size()` 返回批次数。

`WeightedRandomSampler(weights, num_samples, replacement, seed)` 按权重抽样，权重须有限且非负、总和须有限且为正，`num_samples` 须为正；无放回抽样数不得超过正权重项数。零权重项不会被抽中，有放回模式在极小总权重的舍入边界仍保持这一约束。可独立包含 `<jyppx/liuding/data/samplers/weighted_random_sampler.hpp>`；相同标准库、配置和调用顺序下可用 `save_state()` / `load_state()` 精确续抽，不承诺跨标准库的加权序列或极端权重比下的精确实数概率。

`BatchSampler` 的 `indices()` 返回扁平样本索引，`size()` 保持返回批次数，`sample_count()` 返回应用尾批策略后的样本数；支持嵌套视图，并在截断前检查底层声明的样本数与实际索引长度。可直接传给 `DataPipeline`，管线的 `batch_size` / `drop_last` 必须与该视图一致，冲突在构造时拒绝；有效样本数为零仍拒绝构造。也可直接传基础 sampler，由管线组批。

`BatchSampler` 将 `set_epoch()`、epoch 能力和状态保存/恢复转发给底层 sampler；配置身份同时绑定底层类型/配置、批大小及尾批策略。底层缺少稳定身份时不提升恢复等级；底层支持状态、身份与确定性 batch factory 时，管线可按原合同精确恢复已提交游标及后续 epoch。旧版未记录该配置身份的 BatchSampler 管线状态会被视为拓扑不兼容，须使用原版本恢复。

`DistributedSampler(size, rank, world_size, shuffle, seed, drop_last)` 按显式 rank/world_size 划分 epoch，不丢弃尾部时会补齐各 rank 的样本数；每个 epoch 调用一次 `set_epoch()`。状态为 `save_state()` 生成的规范无符号十进制，接受 0 至最大 `uint64_t`，拒绝符号、空白、前导零、尾字节及溢出；拒绝时当前 epoch 和后续顺序保持不变。既有合法状态格式保持可读。rank/world_size 由外部启动器或任务配置提供，当前库未提供进程组 API。

内置策略不能满足需求时，可将自定义 `Sampler` 传给 `DataPipeline`；其配置身份、epoch 与状态接口需符合所需的恢复等级。
