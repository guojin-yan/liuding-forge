#ifndef JYPPX_DATA_DATALOADER_HPP
#define JYPPX_DATA_DATALOADER_HPP

#include <jyppx/data/dataset.hpp>
#include <jyppx/data/thread_safe_queue.hpp>
#include <jyppx/data/tensor_dataset.hpp>
#include <jyppx/data/samplers/sampler.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/data/samplers/random_sampler.hpp>
#include <jyppx/data/samplers/batch_sampler.hpp>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <functional>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace jyppx {
namespace data {

/** @brief Batch container for arbitrary sample types. */
template <typename T>
struct Batch {
    std::vector<T> samples;
    size_t batch_size() const { return samples.size(); }
    bool empty() const { return samples.empty(); }
};

/** @brief Batched tensor samples. */
template <>
struct Batch<Tensor> {
    Tensor data;
    size_t batch_size() const { return data.defined() ? static_cast<size_t>(data.size(0)) : 0; }
    bool empty() const { return !data.defined() || data.size(0) == 0; }
};

/** @brief Batched input and target tensors from TensorDataset. */
template <>
struct Batch<Example> {
    Tensor data;
    Tensor targets;
    size_t batch_size() const { return data.defined() ? static_cast<size_t>(data.size(0)) : 0; }
    bool empty() const { return !data.defined() || data.size(0) == 0; }
};

/**
 * @brief Collate function type
 */
template<typename T>
using CollateFn = std::function<Batch<T>(const std::vector<T>&)>;

/** @brief Default batching behavior for generic samples. */
template <typename T>
Batch<T> default_collate(const std::vector<T>& samples) {
    return Batch<T>{samples};
}

/** @brief Stack tensor samples along a new batch dimension. */
template <>
inline Batch<Tensor> default_collate(const std::vector<Tensor>& samples) {
    return Batch<Tensor>{samples.empty() ? Tensor{} : torch::stack(samples)};
}

/** @brief Stack TensorDataset inputs and targets separately. */
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
 * @brief DataLoader options
 */
struct DataLoaderOptions {
    size_t batch_size = 1;              ///< Batch size
    bool shuffle = false;                ///< Shuffle data
    size_t num_workers = 0;             ///< Worker threads (0 = load in the calling thread)
    bool drop_last = false;              ///< Drop last incomplete batch
    size_t prefetch_factor = 2;          ///< Batches each worker keeps queued ahead
    Optional<int64_t> seed;              ///< Random seed for shuffling
};

/**
 * @brief DataLoader for batching and loading data
 *
 * With num_workers > 0 batches are fetched by a worker thread pool with
 * prefetching. Results are returned in submission order (identical to the
 * single-threaded loader), exceptions raised inside workers are rethrown on
 * the consuming thread, and Dataset::get must therefore be safe to call from
 * multiple threads when num_workers > 0.
 *
 * @tparam T Type of data samples
 */
template<typename T = Tensor>
class DataLoader {
public:
    /**
     * @brief Constructor
     * @param dataset Dataset to load from
     * @param options DataLoader options
     */
    DataLoader(
        std::shared_ptr<Dataset<T>> dataset,
        const DataLoaderOptions& options = DataLoaderOptions{}
    ) : DataLoader(std::move(dataset), options, CollateFn<T>{}) {}

    /**
     * @brief Construct a loader with a custom collate function.
     *
     * The function receives the samples in the current batch and returns the
     * tensor consumed by the training loop. For non-Tensor sample types a
     * collate function is required.
     */
    DataLoader(
        std::shared_ptr<Dataset<T>> dataset,
        const DataLoaderOptions& options,
        CollateFn<T> collate_fn)
        : dataset_(std::move(dataset)), options_(options), collate_fn_(std::move(collate_fn)) {
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
            throw std::invalid_argument("DataLoader prefetch_factor must be positive");
        }

        if (options_.shuffle) {
            sampler_ = std::make_shared<samplers::RandomSampler>(dataset_->size(), options_.seed);
        } else {
            sampler_ = std::make_shared<samplers::SequentialSampler>(dataset_->size());
        }
        batch_sampler_ = std::make_shared<samplers::BatchSampler>(
            sampler_, options_.batch_size, options_.drop_last);
        if (options_.num_workers > 0) {
            start_workers();
        }
    }

    /** @brief Construct a loader from a caller-provided sampler. */
    DataLoader(
        std::shared_ptr<Dataset<T>> dataset,
        std::shared_ptr<samplers::Sampler> sampler,
        const DataLoaderOptions& options = DataLoaderOptions{},
        CollateFn<T> collate_fn = CollateFn<T>{})
        : dataset_(std::move(dataset)), sampler_(std::move(sampler)), options_(options),
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
            throw std::invalid_argument("DataLoader prefetch_factor must be positive");
        }
        batch_sampler_ = std::make_shared<samplers::BatchSampler>(
            sampler_, options_.batch_size, options_.drop_last);
        if (options_.num_workers > 0) {
            start_workers();
        }
    }

    ~DataLoader() {
        shutdown_workers();
    }

    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    /**
     * @brief Iterator for DataLoader
     */
    class Iterator {
    public:
        Iterator(DataLoader* loader, size_t batch_idx);

        Batch<T> operator*();
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;

    private:
        DataLoader* loader_;
        size_t batch_idx_;
        bool cached_ = false;
        Batch<T> cache_;
    };

    /**
     * @brief Begin iterator
     */
    Iterator begin();

    /**
     * @brief End iterator
     */
    Iterator end();

    /**
     * @brief Get number of batches
     */
    size_t size() const;

    /** @brief Number of samples in the underlying dataset. */
    size_t dataset_size() const { return dataset_->size(); }

    /** @brief Loader options. */
    const DataLoaderOptions& options() const { return options_; }

    /** @brief Start a new epoch; shuffled loaders generate a fresh order. */
    void reset() {
        prepare_epoch();
        if (num_workers() > 0) {
            shutdown_workers();
            start_workers();
            start_epoch();
        }
    }

private:
    struct TaggedIndices {
        std::uint64_t seq = 0;
        std::vector<std::size_t> indices;
    };

    struct TaggedResult {
        std::uint64_t seq = 0;
        Batch<T> batch;
        std::exception_ptr exception;
    };

    std::shared_ptr<Dataset<T>> dataset_;
    DataLoaderOptions options_;
    std::shared_ptr<samplers::Sampler> sampler_;
    std::shared_ptr<samplers::Sampler> batch_sampler_;
    CollateFn<T> collate_fn_;
    std::vector<size_t> epoch_indices_;

    // Multi-worker pipeline state (unused when options_.num_workers == 0).
    std::vector<std::thread> workers_;
    std::shared_ptr<ThreadSafeQueue<TaggedIndices>> index_queue_;
    std::shared_ptr<ThreadSafeQueue<TaggedResult>> data_queue_;
    std::map<std::uint64_t, TaggedResult> pending_;
    std::uint64_t next_feed_seq_ = 0;
    std::uint64_t next_consume_seq_ = 0;
    std::uint64_t completed_seq_ = 0;

    std::size_t num_workers() const { return options_.num_workers; }

    std::size_t prefetch_limit() const {
        return std::max<std::size_t>(num_workers() * options_.prefetch_factor, 1);
    }

    void prepare_epoch() {
        epoch_indices_ = batch_sampler_->indices();
    }

    /// Loads and collates one batch for the given sample indices.
    Batch<T> load_batch(const std::vector<size_t>& indices) const {
        auto samples = dataset_->get_batch(indices);
        if (collate_fn_) {
            return collate_fn_(samples);
        }
        return default_collate<T>(samples);
    }

    /// Single-threaded fetch: slice the epoch indices and load inline.
    Batch<T> fetch_batch(size_t batch_idx) const {
        const size_t start = batch_idx * options_.batch_size;
        const size_t total = epoch_indices_.size();
        if (start >= total) {
            throw std::out_of_range("DataLoader batch index out of range");
        }
        const size_t end = std::min(start + options_.batch_size, total);
        std::vector<size_t> indices(epoch_indices_.begin() + static_cast<std::ptrdiff_t>(start),
                                    epoch_indices_.begin() + static_cast<std::ptrdiff_t>(end));
        return load_batch(indices);
    }

    /// Worker body: pulls index batches, loads and collates, pushes results.
    /// Exceptions are captured per batch and rethrown by the consumer in
    /// submission order. Every submitted index batch produces exactly one
    /// result, so the consumer can never wait on a missing sequence number.
    void worker_loop() {
        TaggedIndices work;
        while (index_queue_->get(work)) {
            if (index_queue_->is_closed()) {
                break; // shutdown: discard any remaining scheduled work
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

    void start_workers() {
        index_queue_ = std::make_shared<ThreadSafeQueue<TaggedIndices>>(prefetch_limit());
        data_queue_ = std::make_shared<ThreadSafeQueue<TaggedResult>>(prefetch_limit());
        workers_.reserve(num_workers());
        for (std::size_t i = 0; i < num_workers(); ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    /// Stops the worker threads and discards any in-flight results. Safe to
    /// call repeatedly; used both between epochs and at destruction time so a
    /// partially consumed epoch never leaks into the next one.
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

    /// Feeds more index batches while fewer than prefetch_limit are in flight.
    void feed_ahead() {
        if (num_workers() == 0) {
            return;
        }
        const std::uint64_t total = static_cast<std::uint64_t>(size());
        while (next_feed_seq_ < total &&
               next_feed_seq_ - completed_seq_ - pending_.size() < prefetch_limit()) {
            TaggedIndices work;
            work.seq = next_feed_seq_;
            const size_t start = static_cast<size_t>(next_feed_seq_) * options_.batch_size;
            const size_t end = std::min(start + options_.batch_size, epoch_indices_.size());
            work.indices.assign(epoch_indices_.begin() + static_cast<std::ptrdiff_t>(start),
                                epoch_indices_.begin() + static_cast<std::ptrdiff_t>(end));
            ++next_feed_seq_;
            if (!index_queue_->put(std::move(work))) {
                --next_feed_seq_;
                break;
            }
        }
    }

    void start_epoch() {
        next_feed_seq_ = 0;
        next_consume_seq_ = 0;
        completed_seq_ = 0;
        pending_.clear();
        feed_ahead();
    }

    /// Consumes pipeline results in submission order up to target_seq and
    /// returns that batch. Out-of-order results are buffered; worker
    /// exceptions are rethrown when their sequence is reached.
    Batch<T> receive_until(std::uint64_t target_seq) {
        while (true) {
            auto buffered = pending_.find(next_consume_seq_);
            if (buffered == pending_.end()) {
                TaggedResult result;
                if (!data_queue_->get(result)) {
                    throw std::runtime_error(
                        "DataLoader worker pipeline ended before all batches were produced");
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
                    continue; // skipped sequence: consume and move on
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
        cache_ = loader_->num_workers() == 0
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
    prepare_epoch();
    if (num_workers() > 0) {
        shutdown_workers();
        start_workers();
        start_epoch();
    }
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

template <typename DatasetT>
DataLoader(std::shared_ptr<DatasetT>, const DataLoaderOptions& = DataLoaderOptions{})
    -> DataLoader<typename DatasetT::value_type>;

template <typename DatasetT>
DataLoader(
    std::shared_ptr<DatasetT>,
    const DataLoaderOptions&,
    CollateFn<typename DatasetT::value_type>) -> DataLoader<typename DatasetT::value_type>;

} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_DATALOADER_HPP
