// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_DATA_SAMPLERS_WEIGHTED_RANDOM_SAMPLER_HPP
#define JYPPX_LIUDING_DATA_SAMPLERS_WEIGHTED_RANDOM_SAMPLER_HPP
/**
 * @file weighted_random_sampler.hpp
 * @brief Defines weighted, subset, and distributed sampling strategies.
 */

#include <jyppx/liuding/data/detail/error.hpp>
#include <jyppx/liuding/data/samplers/sampler.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

namespace jyppx::liuding {
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
  /**
   * @brief Creates a weighted sampler.
   * @param weights Finite non-negative weight per population entry.
   * @param num_samples Number of indices produced per epoch; positive.
   * @param replacement Whether the same index may be drawn more than once.
   * @param seed Optional deterministic engine seed.
   * @throws jyppx::liuding::InvalidArgument Weights or sample count cannot form a valid
   * draw.
   */
  WeightedRandomSampler(std::vector<double> weights, size_t num_samples,
                        bool replacement = true,
                        Optional<uint64_t> seed = std::nullopt)
      : weights_(std::move(weights)),
        num_samples_(num_samples),
        replacement_(replacement),
        generator_(seed.has_value()
                       ? static_cast<std::mt19937_64::result_type>(*seed)
                       : static_cast<std::mt19937_64::result_type>(
                             std::random_device{}())) {
    if (weights_.empty()) {
      ::jyppx::liuding::data::detail::throw_invalid(
          "data.weighted_random_sampler", "constructor.weights",
          "non-empty weights", "empty weights",
          "Provide one finite weight per population entry.");
    }
    if (num_samples_ == 0) {
      ::jyppx::liuding::data::detail::throw_invalid(
          "data.weighted_random_sampler", "constructor.num_samples",
          "positive sample count", "zero", "Set num_samples to at least one.");
    }
    size_t positive_weights = 0;
    double total = 0.0;
    for (const double weight : weights_) {
      if (!std::isfinite(weight) || weight < 0.0) {
        ::jyppx::liuding::data::detail::throw_invalid(
            "data.weighted_random_sampler", "constructor.weights",
            "finite non-negative weights", std::to_string(weight),
            "Replace NaN, infinity, or negative weights.");
      }
      positive_weights += weight > 0.0 ? 1 : 0;
      total += weight;
    }
    if (!std::isfinite(total) || total <= 0.0) {
      ::jyppx::liuding::data::detail::throw_invalid(
          "data.weighted_random_sampler", "constructor.weights",
          "finite positive weight sum", std::to_string(total),
          "Provide at least one positive finite weight.");
    }
    if (!replacement_ && num_samples_ > positive_weights) {
      ::jyppx::liuding::data::detail::throw_invalid(
          "data.weighted_random_sampler", "constructor.num_samples",
          "num_samples <= positive weight count",
          "num_samples=" + std::to_string(num_samples_) +
              ", positive_weights=" + std::to_string(positive_weights),
          "Enable replacement or reduce num_samples.");
    }
    // Precompute cumulative weights for the sampling walk.
    cumulative_.reserve(weights_.size());
    double running = 0.0;
    for (const double weight : weights_) {
      running += weight;
      cumulative_.push_back(running);
    }
  }

  /**
   * @brief Draws the next weighted epoch and advances generator state.
   * @return Configured number of weighted population indices.
   */
  std::vector<size_t> indices() const override {
    std::vector<size_t> result;
    result.reserve(num_samples_);
    std::uniform_real_distribution<double> uniform(0.0, cumulative_.back());

    if (replacement_) {
      for (size_t i = 0; i < num_samples_; ++i) {
        result.push_back(draw(uniform));
      }
    } else {
      // Exponential-race sampling is weighted, bounded, and cannot hang
      // when zero-weight entries are present.
      std::uniform_real_distribution<double> unit(0.0, 1.0);
      std::vector<std::pair<double, size_t>> keys;
      keys.reserve(weights_.size());
      for (size_t index = 0; index < weights_.size(); ++index) {
        if (weights_[index] == 0.0) {
          continue;
        }
        const double sample =
            std::max(unit(generator_), std::numeric_limits<double>::min());
        keys.emplace_back(-std::log(sample) / weights_[index], index);
      }
      std::partial_sort(keys.begin(), keys.begin() + num_samples_, keys.end());
      for (size_t i = 0; i < num_samples_; ++i) {
        result.push_back(keys[i].second);
      }
    }
    return result;
  }

  /**
   * @brief Returns the configured draws per epoch.
   * @return Positive sample count.
   */
  size_t size() const override { return num_samples_; }

  /**
   * @brief Reports exact random-engine checkpoint support.
   * @return Always true.
   */
  bool supports_state() const override { return true; }

  /**
   * @brief Serializes the standard-library random engine state.
   * @return Opaque textual engine representation.
   */
  std::string save_state() const override {
    std::ostringstream stream;
    stream << generator_;
    return stream.str();
  }

  /**
   * @brief Restores random-engine state atomically.
   * @param state Value returned by save_state().
   * @throws jyppx::liuding::SerializationError The state is malformed.
   */
  void load_state(const std::string& state) override {
    auto restored = generator_;
    std::istringstream stream(state);
    if (!(stream >> restored)) {
      ::jyppx::liuding::data::detail::throw_state_error(
          "data.weighted_random_sampler", "load_state", "invalid engine state");
    }
    stream >> std::ws;
    if (!stream.eof()) {
      ::jyppx::liuding::data::detail::throw_state_error("data.weighted_random_sampler",
                                               "load_state", "trailing bytes");
    }
    generator_ = restored;
  }

 private:
  /**
   * @brief Draws one replacement-mode index from @p uniform.
   * @param uniform Distribution spanning the cumulative weight range.
   * @return Population index selected by the sampled cumulative interval.
   */
  size_t draw(std::uniform_real_distribution<double>& uniform) const {
    const double value = uniform(generator_);
    const auto upper =
        std::upper_bound(cumulative_.begin(), cumulative_.end(), value);
    const size_t pos =
        static_cast<size_t>(std::distance(cumulative_.begin(), upper));
    return std::min(pos, weights_.size() - 1);
  }

  std::vector<double> weights_;  ///< Finite non-negative mass per entry.
  std::vector<double>
      cumulative_;      ///< Inclusive prefix sums ending above zero.
  size_t num_samples_;  ///< Number of output indices; positive.
  bool replacement_;    ///< Whether repeated indices are permitted.
  mutable std::mt19937_64 generator_;  ///< Engine advanced by indices().
};

/**
 * @brief Random permutation of a given subset of sample indices
 *
 * Mirrors torch.utils.data.SubsetRandomSampler; typically fed from the
 * training indices of a random_split.
 */
class SubsetRandomSampler : public Sampler {
 public:
  /**
   * @brief Creates a permutation sampler for a caller-provided subset.
   * @param indices Non-empty dataset indices to permute.
   * @param seed Optional deterministic engine seed.
   * @throws jyppx::liuding::InvalidArgument @p indices is empty.
   */
  SubsetRandomSampler(std::vector<size_t> indices,
                      Optional<uint64_t> seed = std::nullopt)
      : indices_(std::move(indices)),
        generator_(seed.has_value()
                       ? static_cast<std::mt19937_64::result_type>(*seed)
                       : static_cast<std::mt19937_64::result_type>(
                             std::random_device{}())) {
    if (indices_.empty()) {
      ::jyppx::liuding::data::detail::throw_invalid(
          "data.subset_random_sampler", "constructor.indices",
          "non-empty index set", "empty index set",
          "Provide at least one source index.");
    }
  }

  /**
   * @brief Returns the next random permutation of the configured subset.
   * @return Shuffled copy of caller-provided indices.
   */
  std::vector<size_t> indices() const override {
    auto shuffled = indices_;
    std::shuffle(shuffled.begin(), shuffled.end(), generator_);
    return shuffled;
  }

  /**
   * @brief Returns the number of indices in the subset.
   * @return Configured subset cardinality.
   */
  size_t size() const override { return indices_.size(); }

  /**
   * @brief Reports exact random-engine checkpoint support.
   * @return Always true.
   */
  bool supports_state() const override { return true; }

  /**
   * @brief Serializes the standard-library random engine state.
   * @return Opaque textual engine representation.
   */
  std::string save_state() const override {
    std::ostringstream stream;
    stream << generator_;
    return stream.str();
  }

  /**
   * @brief Restores random-engine state atomically.
   * @param state Value returned by save_state().
   * @throws jyppx::liuding::SerializationError The state is malformed.
   */
  void load_state(const std::string& state) override {
    auto restored = generator_;
    std::istringstream stream(state);
    if (!(stream >> restored)) {
      ::jyppx::liuding::data::detail::throw_state_error(
          "data.subset_random_sampler", "load_state", "invalid engine state");
    }
    stream >> std::ws;
    if (!stream.eof()) {
      ::jyppx::liuding::data::detail::throw_state_error("data.subset_random_sampler",
                                               "load_state", "trailing bytes");
    }
    generator_ = restored;
  }

 private:
  std::vector<size_t> indices_;  ///< Caller-provided dataset index subset.
  mutable std::mt19937_64 generator_;  ///< Engine advanced by indices().
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
  /**
   * @brief Creates one rank's deterministic distributed partition.
   * @param size Positive global dataset size.
   * @param rank Zero-based local rank, less than @p world_size.
   * @param world_size Positive number of participating ranks.
   * @param shuffle Whether to permute using seed plus epoch.
   * @param seed Base deterministic seed; defaults to zero.
   * @param drop_last Whether to drop the uneven global tail instead of padding.
   * @throws jyppx::liuding::InvalidArgument Size or rank topology is invalid.
   */
  DistributedSampler(size_t size, size_t rank, size_t world_size,
                     bool shuffle = true,
                     Optional<uint64_t> seed = std::nullopt,
                     bool drop_last = false)
      : size_(size),
        rank_(rank),
        world_size_(world_size),
        shuffle_(shuffle),
        drop_last_(drop_last),
        seed_(seed.value_or(0)) {
    if (size_ == 0) {
      ::jyppx::liuding::data::detail::throw_invalid(
          "data.distributed_sampler", "constructor.size", "positive size",
          "zero", "Provide a non-empty global dataset.");
    }
    if (world_size_ == 0) {
      ::jyppx::liuding::data::detail::throw_invalid(
          "data.distributed_sampler", "constructor.world_size",
          "positive world size", "zero",
          "Provide at least one participating rank.");
    }
    if (rank_ >= world_size_) {
      ::jyppx::liuding::data::detail::throw_invalid(
          "data.distributed_sampler", "constructor.rank", "rank < world_size",
          "rank=" + std::to_string(rank_) +
              ", world_size=" + std::to_string(world_size_),
          "Use a zero-based rank within the distributed topology.");
    }
  }

  /**
   * @brief Sets the non-negative epoch mixed into the shuffle seed.
   * @param epoch Non-negative logical epoch number.
   */
  void set_epoch(std::uint64_t epoch) override { epoch_ = epoch; }

  /**
   * @brief Reports deterministic epoch-derived ordering support.
   * @return Always true because seed and epoch fully determine the order.
   */
  bool supports_epoch_control() const override { return true; }

  /**
   * @brief Returns the stable immutable distributed topology identity.
   * @return Versioned identity covering partition and shuffle configuration.
   */
  std::string configuration_identity() const override {
    return "jyppx.sampler.distributed/v1:size=" + std::to_string(size_) +
           ":rank=" + std::to_string(rank_) +
           ":world_size=" + std::to_string(world_size_) +
           ":shuffle=" + (shuffle_ ? "1" : "0") +
           ":drop_last=" + (drop_last_ ? "1" : "0") +
           ":seed=" + std::to_string(seed_);
  }

  /**
   * @brief Returns the current non-negative shuffle epoch.
   * @return Epoch mixed into the deterministic shuffle seed.
   */
  std::uint64_t epoch() const { return epoch_; }

  /**
   * @brief Returns this rank's indices for the current epoch.
   * @return Deterministic rank-local partition.
   */
  std::vector<size_t> indices() const override {
    std::vector<size_t> all(size_);
    for (size_t i = 0; i < size_; ++i) {
      all[i] = i;
    }

    if (shuffle_) {
      std::mt19937_64 generator(static_cast<std::mt19937_64::result_type>(
          seed_ + static_cast<uint64_t>(epoch_)));
      std::shuffle(all.begin(), all.end(), generator);
    }

    // Virtual padding repeats the shuffled prefix without allocating a padded
    // global vector. This also keeps the count arithmetic valid at size_t's
    // upper bound.
    const size_t per_rank = size();
    std::vector<size_t> mine;
    mine.reserve(per_rank);
    for (size_t local_index = 0; local_index < per_rank; ++local_index) {
      const size_t global_index = rank_ + local_index * world_size_;
      mine.push_back(all[global_index % size_]);
    }
    return mine;
  }

  /**
   * @brief Returns the number of indices assigned to this rank.
   * @return Equalized rank-local sample count.
   */
  size_t size() const override {
    const size_t complete_samples = size_ / world_size_;
    const size_t remainder = size_ % world_size_;
    return drop_last_ || remainder == 0 ? complete_samples
                                        : complete_samples + 1;
  }

  /**
   * @brief Reports epoch checkpoint support.
   * @return Always true.
   */
  bool supports_state() const override { return true; }

  /**
   * @brief Serializes the current epoch as decimal text.
   * @return Non-negative epoch encoded as decimal text.
   */
  std::string save_state() const override { return std::to_string(epoch_); }

  /**
   * @brief Restores the shuffle epoch atomically.
   * @param state Decimal state returned by save_state().
   * @throws jyppx::liuding::SerializationError The state is malformed or negative.
   */
  void load_state(const std::string& state) override {
    size_t consumed = 0;
    try {
      if (state.empty() || state.front() == '-') {
        ::jyppx::liuding::data::detail::throw_state_error(
            "data.distributed_sampler", "load_state", "negative epoch");
      }
      const auto restored = std::stoull(state, &consumed);
      if (consumed != state.size()) {
        ::jyppx::liuding::data::detail::throw_state_error(
            "data.distributed_sampler", "load_state", "trailing bytes");
      }
      epoch_ = restored;
    } catch (const ::jyppx::liuding::SerializationError&) {
      throw;
    } catch (const std::exception&) {
      ::jyppx::liuding::data::detail::throw_state_error("data.distributed_sampler",
                                               "load_state", "invalid epoch");
    }
  }

 private:
  size_t size_;        ///< Positive global dataset size.
  size_t rank_;        ///< Zero-based local rank.
  size_t world_size_;  ///< Positive number of participating ranks.
  bool shuffle_;       ///< Whether each epoch is deterministically permuted.
  bool drop_last_;  ///< Whether to drop rather than pad the uneven global tail.
  uint64_t seed_;   ///< Base seed combined with epoch; defaults to zero.
  std::uint64_t epoch_ = 0;  ///< Non-negative shuffle epoch.
};

}  // namespace samplers
}  // namespace data
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_DATA_SAMPLERS_WEIGHTED_RANDOM_SAMPLER_HPP