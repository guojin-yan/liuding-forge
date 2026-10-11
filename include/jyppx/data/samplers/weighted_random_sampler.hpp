#ifndef JYPPX_DATA_SAMPLERS_WEIGHTED_RANDOM_SAMPLER_HPP
#define JYPPX_DATA_SAMPLERS_WEIGHTED_RANDOM_SAMPLER_HPP

#include <jyppx/data/samplers/sampler.hpp>

#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

namespace jyppx {
namespace data {
namespace samplers {

/**
 * @brief Draws num_samples indices with probability proportional to weights
 *
 * Mirrors torch.utils.data.WeightedRandomSampler. Each call to indices()
 * produces a fresh draw; with a fixed seed the sequence is reproducible.
 */
class WeightedRandomSampler : public Sampler {
public:
    WeightedRandomSampler(std::vector<double> weights, size_t num_samples,
                          bool replacement = true, Optional<uint64_t> seed = std::nullopt)
        : weights_(std::move(weights)),
          num_samples_(num_samples),
          replacement_(replacement),
          generator_(seed.has_value()
                         ? static_cast<std::mt19937_64::result_type>(*seed)
                         : static_cast<std::mt19937_64::result_type>(
                               std::random_device{}())) {
        if (weights_.empty()) {
            throw std::invalid_argument("WeightedRandomSampler requires weights");
        }
        if (num_samples_ == 0) {
            throw std::invalid_argument("WeightedRandomSampler num_samples must be positive");
        }
        for (const double weight : weights_) {
            if (weight < 0.0) {
                throw std::invalid_argument("WeightedRandomSampler weights must be non-negative");
            }
        }
        double total = 0.0;
        for (const double weight : weights_) {
            total += weight;
        }
        if (total <= 0.0) {
            throw std::invalid_argument("WeightedRandomSampler weights must sum to a positive value");
        }
        if (!replacement_ && num_samples_ > weights_.size()) {
            throw std::invalid_argument(
                "WeightedRandomSampler without replacement cannot exceed the population size");
        }
        // Precompute cumulative weights for the sampling walk.
        cumulative_.reserve(weights_.size());
        double running = 0.0;
        for (const double weight : weights_) {
            running += weight;
            cumulative_.push_back(running);
        }
    }

    std::vector<size_t> indices() const override {
        std::vector<size_t> result;
        result.reserve(num_samples_);
        std::uniform_real_distribution<double> uniform(0.0, cumulative_.back());

        if (replacement_) {
            for (size_t i = 0; i < num_samples_; ++i) {
                result.push_back(draw(uniform));
            }
        } else {
            // Rejection sampling: draw from the full distribution and keep
            // the first num_samples distinct indices.
            std::vector<bool> taken(weights_.size(), false);
            size_t taken_count = 0;
            while (taken_count < num_samples_) {
                const size_t candidate = draw(uniform);
                if (!taken[candidate]) {
                    taken[candidate] = true;
                    result.push_back(candidate);
                    ++taken_count;
                }
            }
        }
        return result;
    }

    size_t size() const override { return num_samples_; }

private:
    size_t draw(std::uniform_real_distribution<double>& uniform) const {
        const double value = uniform(generator_);
        const auto upper = std::lower_bound(cumulative_.begin(), cumulative_.end(), value);
        const size_t pos = static_cast<size_t>(
            std::distance(cumulative_.begin(), upper));
        return std::min(pos, weights_.size() - 1);
    }

    std::vector<double> weights_;
    std::vector<double> cumulative_;
    size_t num_samples_;
    bool replacement_;
    mutable std::mt19937_64 generator_;
};

/**
 * @brief Random permutation of a given subset of sample indices
 *
 * Mirrors torch.utils.data.SubsetRandomSampler; typically fed from the
 * training indices of a random_split.
 */
class SubsetRandomSampler : public Sampler {
public:
    SubsetRandomSampler(std::vector<size_t> indices, Optional<uint64_t> seed = std::nullopt)
        : indices_(std::move(indices)),
          generator_(seed.has_value()
                         ? static_cast<std::mt19937_64::result_type>(*seed)
                         : static_cast<std::mt19937_64::result_type>(
                               std::random_device{}())) {
        if (indices_.empty()) {
            throw std::invalid_argument("SubsetRandomSampler requires a non-empty index set");
        }
    }

    std::vector<size_t> indices() const override {
        auto shuffled = indices_;
        std::shuffle(shuffled.begin(), shuffled.end(), generator_);
        return shuffled;
    }

    size_t size() const override { return indices_.size(); }

private:
    std::vector<size_t> indices_;
    mutable std::mt19937_64 generator_;
};

/**
 * @brief Splits sample indices evenly across distributed ranks
 *
 * Mirrors torch.utils.data.DistributedSampler: indices are padded to a
 * multiple of world_size (dropped with drop_last), optionally shuffled with
 * seed + epoch, and each rank takes every world_size-th index starting from
 * its own offset. With world_size = 1 this behaves like a plain
 * (shuffled) sampler, which keeps single-process training code unchanged.
 */
class DistributedSampler : public Sampler {
public:
    DistributedSampler(size_t size, size_t rank, size_t world_size,
                       bool shuffle = true, Optional<uint64_t> seed = std::nullopt,
                       bool drop_last = false)
        : size_(size),
          rank_(rank),
          world_size_(world_size),
          shuffle_(shuffle),
          drop_last_(drop_last),
          seed_(seed.value_or(0)),
          generator_(seed.has_value()
                         ? static_cast<std::mt19937_64::result_type>(*seed)
                         : static_cast<std::mt19937_64::result_type>(
                               std::random_device{}())) {
        if (world_size_ == 0) {
            throw std::invalid_argument("DistributedSampler world_size must be positive");
        }
        if (rank_ >= world_size_) {
            throw std::invalid_argument("DistributedSampler rank must be < world_size");
        }
    }

    /** @brief Advances the shuffle epoch; call once per training epoch. */
    void set_epoch(int64_t epoch) { epoch_ = epoch; }
    int64_t epoch() const { return epoch_; }

    std::vector<size_t> indices() const override {
        // Pad so every rank receives the same number of samples.
        size_t padded = size_;
        if (!drop_last_) {
            padded = (size_ + world_size_ - 1) / world_size_ * world_size_;
        }
        std::vector<size_t> all(padded);
        for (size_t i = 0; i < padded; ++i) {
            all[i] = i % size_; // pad by repeating from the start (PyTorch semantics)
        }

        if (shuffle_) {
            std::mt19937_64 generator(static_cast<std::mt19937_64::result_type>(
                seed_ + static_cast<uint64_t>(epoch_)));
            std::shuffle(all.begin(), all.end(), generator);
        }
        // Drop the tail so every rank gets the same count under drop_last.
        const size_t per_rank = padded / world_size_;
        std::vector<size_t> mine;
        mine.reserve(per_rank);
        for (size_t i = rank_; i < per_rank * world_size_; i += world_size_) {
            mine.push_back(all[i]);
        }
        return mine;
    }

    size_t size() const override {
        const size_t padded = drop_last_
            ? (size_ / world_size_) * world_size_
            : (size_ + world_size_ - 1) / world_size_ * world_size_;
        return padded / world_size_;
    }

private:
    size_t size_;
    size_t rank_;
    size_t world_size_;
    bool shuffle_;
    bool drop_last_;
    uint64_t seed_;
    int64_t epoch_ = 0;
    mutable std::mt19937_64 generator_;
};

} // namespace samplers
} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_SAMPLERS_WEIGHTED_RANDOM_SAMPLER_HPP
