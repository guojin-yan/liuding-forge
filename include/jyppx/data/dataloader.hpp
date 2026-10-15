#ifndef JYPPX_DATA_DATALOADER_HPP
#define JYPPX_DATA_DATALOADER_HPP

/**
 * @file dataloader.hpp
 * @brief Defines deterministic batching, worker prefetch, and epoch state.
 */

#include <jyppx/data/dataset.hpp>
#include <jyppx/data/samplers/batch_sampler.hpp>
#include <jyppx/data/samplers/random_sampler.hpp>
#include <jyppx/data/samplers/sampler.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/data/tensor_dataset.hpp>
#include <jyppx/data/thread_safe_queue.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace jyppx {
namespace data {

/**
 * @brief Batch container for arbitrary sample types.
 * @tparam T Value type returned by the source Dataset.
 */
template <typename T>
struct Batch {
  std::vector<T> samples;  ///< Sample values in stable sampler order.

  /**
   * @brief Returns the number of contained samples.
   * @return Length of `samples`.
   */
  size_t batch_size() const { return samples.size(); }

  /**
   * @brief Returns whether the batch contains no samples.
   * @return True when `samples` is empty.
   */
  bool empty() const { return samples.empty(); }
};

/** @brief Batched tensor samples stacked along dimension zero. */
template <>
struct Batch<Tensor> {
  Tensor data;  ///< Batched tensor; undefined represents an empty batch.

  /**
   * @brief Returns dimension zero, or zero for an undefined tensor.
   * @return Number of tensor samples.
   */
  size_t batch_size() const {
    return data.defined() ? static_cast<size_t>(data.size(0)) : 0;
  }

  /**
   * @brief Returns whether data is undefined or has an empty first dimension.
   * @return True when the batch contains no tensor samples.
   */
  bool empty() const { return !data.defined() || data.size(0) == 0; }
};

/** @brief Batched input and target tensors from TensorDataset. */
template <>
struct Batch<Example> {
  Tensor data;     ///< Input samples stacked along dimension zero.
  Tensor targets;  ///< Targets stacked in the same sample order as data.

  /**
   * @brief Returns input dimension zero, or zero for undefined data.
   * @return Number of paired examples.
   */
  size_t batch_size() const {
    return data.defined() ? static_cast<size_t>(data.size(0)) : 0;
  }

  /**
   * @brief Returns whether input data is undefined or empty.
   * @return True when the batch contains no examples.
   */
  bool empty() const { return !data.defined() || data.size(0) == 0; }
};

/**
 * @brief Function that converts ordered samples into one batch.
 * @tparam T Dataset sample type.
 */
template <typename T>
using CollateFn = std::function<Batch<T>(const std::vector<T>&)>;

/**
 * @brief Copies generic samples into a Batch.
 * @param samples Ordered values for one batch.
 * @return Batch containing the same sample values.
 */
template <typename T>
Batch<T> default_collate(const std::vector<T>& samples) {
  return Batch<T>{samples};
}

/**
 * @brief Stacks tensor samples along a new batch dimension.
 * @param samples Shape-compatible tensors.
 * @return Undefined data for an empty input, otherwise the stacked tensor.
 */
template <>
inline Batch<Tensor> default_collate(const std::vector<Tensor>& samples) {
  return Batch<Tensor>{samples.empty() ? Tensor{} : torch::stack(samples)};
}

/**
 * @brief Stacks TensorDataset inputs and targets separately.
 * @param samples Shape-compatible examples in sampler order.
 * @return Batched inputs and targets, or an empty batch for no samples.
 */
template <>
inline Batch<Example> default_collate(const std::vector<Example>& samples) {
  if (samples.empty()) {
    return {};
  }
  std::vector<Tensor> data;
  std::vector<Tensor> targets;
  data.reserve(samples.size());
  targets.reserve(samples.size());
  for (const auto& sample : samples) {
    data.push_back(sample.data);
    targets.push_back(sample.target);
  }
  return Batch<Example>{torch::stack(data), torch::stack(targets)};
}

/**
 * @brief Configures batching, ordering, and worker prefetch.
 */
struct DataLoaderOptions {
  size_t batch_size = 1;  ///< Samples per batch; must be positive.
  bool shuffle = false;  ///< Whether the built-in sampler permutes every epoch.
  size_t num_workers = 0;  ///< Worker threads; zero loads on the caller thread.
  bool drop_last = false;  ///< Whether to omit a final incomplete batch.
  size_t prefetch_factor = 2;  ///< Queued batches per worker; must be positive.
  Optional<int64_t>
      seed;  ///< Optional deterministic seed for built-in shuffling.
};

/**
 * @brief Serializable state required to reproduce future DataLoader epochs.
 *
 * The state records a prepared epoch but intentionally contains no in-epoch
 * cursor. It therefore provides exact recovery only at an epoch boundary.
 */
struct DataLoaderState {
  static constexpr std::uint32_t format_version =
      2;  ///< Latest supported schema version.

  std::uint32_t version = format_version;  ///< Serialized schema version.
  std::size_t dataset_size =
      0;  ///< Dataset cardinality used for compatibility checks.
  std::size_t batch_size = 0;  ///< Samples per batch used by the saved loader.
  bool drop_last = false;      ///< Saved incomplete-batch policy.
  bool epoch_prepared =
      false;  ///< Whether prepared_indices belongs to the next begin().
  bool sampler_state_supported =
      false;  ///< Whether sampler_state is exact and restorable.
  std::string sampler_state;  ///< Opaque state owned by the configured sampler.
  std::vector<std::size_t>
      prepared_indices;  ///< Exact next-epoch order when prepared.
};

/**
 * @brief Batches map-style Dataset samples in deterministic sampler order.
 *
 * With num_workers > 0 batches are fetched by a worker thread pool with
 * prefetching. Results are returned in submission order (identical to the
 * single-threaded loader), exceptions raised inside workers are rethrown on
 * the consuming thread, and Dataset::get must therefore be safe to call from
 * multiple threads when num_workers > 0.
 *
 * @tparam T Type of data samples
 *
 * A DataLoader owns worker threads and shares its Dataset and Sampler. It is
 * movable only through its owning pointer and is not safe for concurrent
 * iteration or state mutation.
 */
template <typename T = Tensor>
class DataLoader {
 public:
  /**
   * @brief Creates a loader with a built-in sequential or random sampler.
   * @param dataset Shared non-empty map-style dataset.
   * @param options Valid batching and worker options.
   * @throws std::invalid_argument Dataset or an option is invalid.
   */
  DataLoader(std::shared_ptr<Dataset<T>> dataset,
             const DataLoaderOptions& options = DataLoaderOptions{})
      : DataLoader(std::move(dataset), options, CollateFn<T>{}) {}

  /**
   * @brief Construct a loader with a custom collate function.
   *
   * The function receives the samples in the current batch and returns the
   * tensor consumed by the training loop. For non-Tensor sample types a
   * collate function is required.
   *
   * @param dataset Shared non-empty map-style dataset.
   * @param options Valid batching and worker options.
   * @param collate_fn Optional custom sample-to-batch function.
   * @throws std::invalid_argument Dataset or an option is invalid.
   */
  DataLoader(std::shared_ptr<Dataset<T>> dataset,
             const DataLoaderOptions& options, CollateFn<T> collate_fn)
      : dataset_(std::move(dataset)),
        options_(options),
        collate_fn_(std::move(collate_fn)) {
    if (!dataset_) {
      throw std::invalid_argument("DataLoader dataset cannot be null");
    }
    if (dataset_->size() == 0) {
      throw std::invalid_argument("DataLoader dataset cannot be empty");
    }
    if (options_.batch_size == 0) {
      throw std::invalid_argument("DataLoader batch_size must be positive");
    }
    if (options_.prefetch_factor == 0) {
      throw std::invalid_argument(
          "DataLoader prefetch_factor must be positive");
    }

    if (options_.shuffle) {
      sampler_ = std::make_shared<samplers::RandomSampler>(dataset_->size(),
                                                           options_.seed);
    } else {
      sampler_ =
          std::make_shared<samplers::SequentialSampler>(dataset_->size());
    }
    batch_sampler_ = std::make_shared<samplers::BatchSampler>(
        sampler_, options_.batch_size, options_.drop_last);
    if (options_.num_workers > 0) {
      start_workers();
    }
  }

  /**
   * @brief Constructs a loader from a caller-provided sampler.
   * @param dataset Shared non-empty map-style dataset.
   * @param sampler Shared non-null sampler for dataset indices.
   * @param options Valid batching and worker options.
   * @param collate_fn Optional custom sample-to-batch function.
   * @throws std::invalid_argument A required object or option is invalid.
   */
  DataLoader(std::shared_ptr<Dataset<T>> dataset,
             std::shared_ptr<samplers::Sampler> sampler,
             const DataLoaderOptions& options = DataLoaderOptions{},
             CollateFn<T> collate_fn = CollateFn<T>{})
      : dataset_(std::move(dataset)),
        sampler_(std::move(sampler)),
        options_(options),
        collate_fn_(std::move(collate_fn)) {
    if (!dataset_) {
      throw std::invalid_argument("DataLoader dataset cannot be null");
    }
    if (!sampler_) {
      throw std::invalid_argument("DataLoader sampler cannot be null");
    }
    if (dataset_->size() == 0) {
      throw std::invalid_argument("DataLoader dataset cannot be empty");
    }
    if (options_.batch_size == 0) {
      throw std::invalid_argument("DataLoader batch_size must be positive");
    }
    if (options_.prefetch_factor == 0) {
      throw std::invalid_argument(
          "DataLoader prefetch_factor must be positive");
    }
    batch_sampler_ = std::make_shared<samplers::BatchSampler>(
        sampler_, options_.batch_size, options_.drop_last);
    if (options_.num_workers > 0) {
      start_workers();
    }
  }

  /** @brief Stops owned worker threads and releases queued batches. */
  ~DataLoader() { shutdown_workers(); }

  /** @brief DataLoader cannot be copied because it owns worker threads. */
  DataLoader(const DataLoader&) = delete;

  /** @brief DataLoader cannot be copy-assigned because it owns worker threads.
   */
  DataLoader& operator=(const DataLoader&) = delete;

  /**
   * @brief Single-pass iterator over batches from one prepared epoch.
   *
   * The iterator borrows its DataLoader and must not outlive it. Dereference
   * may block for worker output and may rethrow a Dataset exception.
   */
  class Iterator {
   public:
    /**
     * @brief Creates an iterator at a batch position.
     * @param loader Borrowed non-null loader.
     * @param batch_idx Zero-based batch position, including end().
     * @throws std::invalid_argument @p loader is null.
     */
    Iterator(DataLoader* loader, size_t batch_idx);

    /**
     * @brief Returns the current batch, loading it at most once.
     * @return Cached or newly loaded batch at the iterator position.
     */
    Batch<T> operator*();

    /**
     * @brief Advances to the next batch and invalidates the cache.
     * @return This iterator after advancement.
     */
    Iterator& operator++();

    /**
     * @brief Compares loader identity and batch position.
     * @param other Iterator to compare against.
     * @return True when the iterators identify different positions.
     */
    bool operator!=(const Iterator& other) const;

   private:
    DataLoader* loader_;   ///< Borrowed loader; non-null for iterator lifetime.
    size_t batch_idx_;     ///< Zero-based batch position.
    bool cached_ = false;  ///< Whether cache_ contains the current batch.
    Batch<T> cache_;  ///< Loaded current batch retained across dereferences.
  };

  /**
   * @brief Begins a new or explicitly prepared epoch.
   * @return Iterator positioned at the first batch.
   */
  Iterator begin();

  /**
   * @brief Returns the sentinel iterator for the current loader.
   * @return Iterator positioned one past the final batch.
   */
  Iterator end();

  /**
   * @brief Returns the number of batches per epoch.
   * @return Batch count after applying the drop-last policy.
   */
  size_t size() const;

  /**
   * @brief Returns the number of samples in the underlying dataset.
   * @return Source dataset cardinality.
   */
  size_t dataset_size() const { return dataset_->size(); }

  /**
   * @brief Returns immutable loader options.
   * @return Configuration retained by this loader.
   */
  const DataLoaderOptions& options() const { return options_; }

  /**
   * @brief Snapshots state that determines future epoch ordering.
   * @return Epoch-boundary state; no current in-epoch cursor is included.
   */
  DataLoaderState state_dict() const {
    DataLoaderState state;
    state.dataset_size = dataset_->size();
    state.batch_size = options_.batch_size;
    state.drop_last = options_.drop_last;
    state.epoch_prepared = epoch_prepared_;
    state.sampler_state_supported = sampler_->supports_state();
    state.sampler_state = sampler_->save_state();
    if (epoch_prepared_) {
      state.prepared_indices = epoch_indices_;
    }
    return state;
  }

  /**
   * @brief Restores a state produced by state_dict().
   * @param state State from a compatible loader and checkpointable sampler.
   * @throws std::invalid_argument Schema, loader configuration, indices, or
   *     sampler state is incompatible.
   */
  void load_state_dict(const DataLoaderState& state) {
    if (state.version != DataLoaderState::format_version) {
      throw std::invalid_argument("Unsupported DataLoader state version");
    }
    if (state.dataset_size != dataset_->size() ||
        state.batch_size != options_.batch_size ||
        state.drop_last != options_.drop_last) {
      throw std::invalid_argument(
          "DataLoader state is incompatible with the current loader");
    }
    if (!state.sampler_state_supported || !sampler_->supports_state()) {
      throw std::invalid_argument(
          "DataLoader sampler does not support exact state restoration");
    }
    for (const auto index : state.prepared_indices) {
      if (index >= dataset_->size()) {
        throw std::invalid_argument(
            "DataLoader state contains an invalid index");
      }
    }
    const std::size_t sampler_size = sampler_->size();
    const std::size_t expected_indices =
        options_.drop_last
            ? (sampler_size / options_.batch_size) * options_.batch_size
            : sampler_size;
    if (state.epoch_prepared &&
        state.prepared_indices.size() != expected_indices) {
      throw std::invalid_argument(
          "Prepared DataLoader state has an incompatible index count");
    }
    if (!state.epoch_prepared && !state.prepared_indices.empty()) {
      throw std::invalid_argument(
          "Unprepared DataLoader state unexpectedly contains indices");
    }

    const std::string previous_sampler_state = sampler_->save_state();
    const bool previous_epoch_prepared = epoch_prepared_;
    const std::vector<size_t> previous_epoch_indices = epoch_indices_;
    try {
      sampler_->load_state(state.sampler_state);
      epoch_prepared_ = state.epoch_prepared;
      epoch_indices_ = state.prepared_indices;
      if (num_workers() > 0) {
        shutdown_workers();
        start_workers();
        if (epoch_prepared_) {
          start_epoch();
        }
      }
    } catch (...) {
      const std::exception_ptr restore_error = std::current_exception();
      try {
        sampler_->load_state(previous_sampler_state);
        epoch_prepared_ = previous_epoch_prepared;
        epoch_indices_ = previous_epoch_indices;
        if (num_workers() > 0) {
          shutdown_workers();
          start_workers();
          if (epoch_prepared_) {
            start_epoch();
          }
        }
      } catch (...) {
        std::throw_with_nested(std::runtime_error(
            "DataLoader state restore failed and rollback was incomplete"));
      }
      std::rethrow_exception(restore_error);
    }
  }

  /**
   * @brief Prepares exactly one new epoch for the next begin() call.
   *
   * Repeated begin() calls without reset() each prepare another epoch;
   * reset() followed by begin() consumes the already prepared order.
   */
  void reset() {
    prepare_iteration();
    epoch_prepared_ = true;
  }

 private:
  /** @brief Work item assigning one ordered sample-index batch to a worker. */
  struct TaggedIndices {
    std::uint64_t seq = 0;  ///< Monotonic batch sequence in the epoch.
    std::vector<std::size_t> indices;  ///< Dataset indices for this batch.
  };

  /** @brief Worker result paired with its submission sequence. */
  struct TaggedResult {
    std::uint64_t seq = 0;  ///< Sequence copied from the work item.
    Batch<T> batch;         ///< Collated result when loading succeeded.
    std::exception_ptr
        exception;  ///< Captured Dataset/collate failure, if any.
  };

  std::shared_ptr<Dataset<T>>
      dataset_;  ///< Shared sample source; non-null and non-empty.
  DataLoaderOptions options_;  ///< Immutable batching and worker configuration.
  std::shared_ptr<samplers::Sampler>
      sampler_;  ///< Shared sample-order provider.
  std::shared_ptr<samplers::Sampler>
      batch_sampler_;        ///< Batch truncation wrapper.
  CollateFn<T> collate_fn_;  ///< Optional custom sample-to-batch function.
  std::vector<size_t>
      epoch_indices_;  ///< Exact order for the active/prepared epoch.
  bool epoch_prepared_ =
      false;  ///< Whether next begin() consumes epoch_indices_.

  std::vector<std::thread>
      workers_;  ///< Owned prefetch workers; empty in inline mode.
  std::shared_ptr<ThreadSafeQueue<TaggedIndices>>
      index_queue_;  ///< Worker input queue.
  std::shared_ptr<ThreadSafeQueue<TaggedResult>>
      data_queue_;  ///< Worker output queue.
  std::map<std::uint64_t, TaggedResult>
      pending_;                      ///< Out-of-order results by sequence.
  std::uint64_t next_feed_seq_ = 0;  ///< Next batch sequence to submit.
  std::uint64_t next_consume_seq_ =
      0;  ///< Next result sequence required by consumer.
  std::uint64_t completed_seq_ =
      0;  ///< Results consumed or rethrown this epoch.

  /**
   * @brief Returns the configured worker count.
   * @return Number of owned worker threads.
   */
  std::size_t num_workers() const { return options_.num_workers; }

  /**
   * @brief Returns the positive maximum number of in-flight batches.
   * @return Worker count times prefetch factor, with a minimum of one.
   */
  std::size_t prefetch_limit() const {
    return std::max<std::size_t>(num_workers() * options_.prefetch_factor, 1);
  }

  /** @brief Generates and stores exactly one epoch of sample indices. */
  void prepare_epoch() { epoch_indices_ = batch_sampler_->indices(); }

  /** @brief Prepares ordering and resets the worker pipeline for iteration. */
  void prepare_iteration() {
    prepare_epoch();
    if (num_workers() > 0) {
      shutdown_workers();
      start_workers();
      start_epoch();
    }
  }

  /**
   * @brief Loads and collates one batch.
   * @param indices Dataset indices in output order.
   * @return Collated batch.
   */
  Batch<T> load_batch(const std::vector<size_t>& indices) const {
    auto samples = dataset_->get_batch(indices);
    if (collate_fn_) {
      return collate_fn_(samples);
    }
    return default_collate<T>(samples);
  }

  /**
   * @brief Loads one batch synchronously from epoch_indices_.
   * @param batch_idx Zero-based batch index.
   * @return Loaded batch.
   * @throws std::out_of_range @p batch_idx is outside the epoch.
   */
  Batch<T> fetch_batch(size_t batch_idx) const {
    const size_t start = batch_idx * options_.batch_size;
    const size_t total = epoch_indices_.size();
    if (start >= total) {
      throw std::out_of_range("DataLoader batch index out of range");
    }
    const size_t end = std::min(start + options_.batch_size, total);
    std::vector<size_t> indices(
        epoch_indices_.begin() + static_cast<std::ptrdiff_t>(start),
        epoch_indices_.begin() + static_cast<std::ptrdiff_t>(end));
    return load_batch(indices);
  }

  /**
   * @brief Loads queued batches until shutdown.
   *
   * Exceptions are captured per batch and rethrown by the consumer in
   * submission order. Every submitted item produces exactly one result.
   */
  void worker_loop() {
    TaggedIndices work;
    while (index_queue_->get(work)) {
      if (index_queue_->is_closed()) {
        break;  // shutdown: discard any remaining scheduled work
      }
      TaggedResult result;
      result.seq = work.seq;
      try {
        result.batch = load_batch(work.indices);
      } catch (...) {
        result.exception = std::current_exception();
      }
      if (!data_queue_->put(std::move(result))) {
        break;
      }
    }
  }

  /** @brief Allocates fresh queues and starts the configured worker threads. */
  void start_workers() {
    index_queue_ =
        std::make_shared<ThreadSafeQueue<TaggedIndices>>(prefetch_limit());
    data_queue_ =
        std::make_shared<ThreadSafeQueue<TaggedResult>>(prefetch_limit());
    workers_.reserve(num_workers());
    for (std::size_t i = 0; i < num_workers(); ++i) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  /**
   * @brief Stops workers and discards all in-flight results.
   *
   * The operation is idempotent and prevents a partially consumed epoch
   * from leaking into a later epoch.
   */
  void shutdown_workers() {
    if (workers_.empty()) {
      return;
    }
    index_queue_->close();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    workers_.clear();
    TaggedResult discard;
    while (data_queue_->try_get(discard)) {
    }
    pending_.clear();
    next_feed_seq_ = 0;
    next_consume_seq_ = 0;
    completed_seq_ = 0;
  }

  /** @brief Submits work until the positive prefetch limit is reached. */
  void feed_ahead() {
    if (num_workers() == 0) {
      return;
    }
    const std::uint64_t total = static_cast<std::uint64_t>(size());
    while (next_feed_seq_ < total &&
           next_feed_seq_ - completed_seq_ - pending_.size() <
               prefetch_limit()) {
      TaggedIndices work;
      work.seq = next_feed_seq_;
      const size_t start =
          static_cast<size_t>(next_feed_seq_) * options_.batch_size;
      const size_t end =
          std::min(start + options_.batch_size, epoch_indices_.size());
      work.indices.assign(
          epoch_indices_.begin() + static_cast<std::ptrdiff_t>(start),
          epoch_indices_.begin() + static_cast<std::ptrdiff_t>(end));
      ++next_feed_seq_;
      if (!index_queue_->put(std::move(work))) {
        --next_feed_seq_;
        break;
      }
    }
  }

  /** @brief Resets sequence counters and starts prefetch for an epoch. */
  void start_epoch() {
    next_feed_seq_ = 0;
    next_consume_seq_ = 0;
    completed_seq_ = 0;
    pending_.clear();
    feed_ahead();
  }

  /**
   * @brief Consumes ordered worker results through a target sequence.
   * @param target_seq Sequence number whose batch must be returned.
   * @return Batch for @p target_seq.
   * @throws std::runtime_error The worker pipeline closes prematurely.
   */
  Batch<T> receive_until(std::uint64_t target_seq) {
    while (true) {
      auto buffered = pending_.find(next_consume_seq_);
      if (buffered == pending_.end()) {
        TaggedResult result;
        if (!data_queue_->get(result)) {
          throw std::runtime_error(
              "DataLoader worker pipeline ended before all batches were "
              "produced");
        }
        if (result.seq == next_consume_seq_) {
          ++next_consume_seq_;
          ++completed_seq_;
          feed_ahead();
          if (result.exception) {
            std::rethrow_exception(result.exception);
          }
          if (result.seq == target_seq) {
            return std::move(result.batch);
          }
          continue;  // skipped sequence: consume and move on
        }
        pending_.emplace(result.seq, std::move(result));
        continue;
      }
      TaggedResult result = std::move(buffered->second);
      pending_.erase(buffered);
      ++next_consume_seq_;
      ++completed_seq_;
      feed_ahead();
      if (result.exception) {
        std::rethrow_exception(result.exception);
      }
      if (result.seq == target_seq) {
        return std::move(result.batch);
      }
    }
  }
};

template <typename T>
DataLoader<T>::Iterator::Iterator(DataLoader* loader, size_t batch_idx)
    : loader_(loader), batch_idx_(batch_idx) {
  if (!loader_) {
    throw std::invalid_argument("DataLoader iterator requires a loader");
  }
}

template <typename T>
Batch<T> DataLoader<T>::Iterator::operator*() {
  if (batch_idx_ >= loader_->size()) {
    throw std::out_of_range("Cannot dereference DataLoader end iterator");
  }
  if (!cached_) {
    cache_ =
        loader_->num_workers() == 0
            ? loader_->fetch_batch(batch_idx_)
            : loader_->receive_until(static_cast<std::uint64_t>(batch_idx_));
    cached_ = true;
  }
  return cache_;
}

template <typename T>
typename DataLoader<T>::Iterator& DataLoader<T>::Iterator::operator++() {
  if (batch_idx_ < loader_->size()) {
    ++batch_idx_;
    cached_ = false;
  }
  return *this;
}

template <typename T>
bool DataLoader<T>::Iterator::operator!=(const Iterator& other) const {
  return loader_ != other.loader_ || batch_idx_ != other.batch_idx_;
}

template <typename T>
typename DataLoader<T>::Iterator DataLoader<T>::begin() {
  if (!epoch_prepared_) {
    prepare_iteration();
  }
  epoch_prepared_ = false;
  return Iterator(this, 0);
}

template <typename T>
typename DataLoader<T>::Iterator DataLoader<T>::end() {
  return Iterator(this, size());
}

template <typename T>
size_t DataLoader<T>::size() const {
  return batch_sampler_->size();
}

/**
 * @brief Deduces a loader's sample type from a dataset pointer.
 * @tparam DatasetT Concrete Dataset implementation exposing value_type.
 */
template <typename DatasetT>
DataLoader(std::shared_ptr<DatasetT>,
           const DataLoaderOptions& = DataLoaderOptions{})
    -> DataLoader<typename DatasetT::value_type>;

/**
 * @brief Deduces a loader's sample type when a custom collate function is
 * supplied.
 * @tparam DatasetT Concrete Dataset implementation exposing value_type.
 */
template <typename DatasetT>
DataLoader(std::shared_ptr<DatasetT>, const DataLoaderOptions&,
           CollateFn<typename DatasetT::value_type>)
    -> DataLoader<typename DatasetT::value_type>;

}  // namespace data
}  // namespace jyppx

#endif  // JYPPX_DATA_DATALOADER_HPP
