// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_DATA_SAMPLERS_RANDOM_SAMPLER_HPP
#define JYPPX_LIUDING_DATA_SAMPLERS_RANDOM_SAMPLER_HPP
/**
 * @file random_sampler.hpp
 * @brief Defines a checkpointable full-population permutation sampler.
 */

#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/data/samplers/sampler.hpp>

#include <cstdint>
#include <random>

namespace jyppx::liuding {
namespace data {
namespace samplers {

/**
 * @brief Produces a fresh random permutation of a fixed population.
 *
 * Generator state is checkpointable. Instances are not thread-safe.
 */
class RandomSampler : public Sampler {
 public:
  /**
   * @brief Creates a permutation sampler.
   * @param size Number of dataset entries; must be positive.
   * @param seed Optional deterministic seed.
   * @throws jyppx::liuding::InvalidArgument @p size is zero.
   */
  explicit RandomSampler(size_t size, Optional<int64_t> seed = std::nullopt);

  /**
   * @brief Returns the next random permutation and advances generator state.
   * @return Permutation containing every population index exactly once.
   */
  std::vector<size_t> indices() const override;

  /**
   * @brief Returns the population size.
   * @return Number of indices in each permutation.
   */
  size_t size() const override;

  /**
   * @brief Selects a deterministic permutation epoch.
   * @param epoch Non-negative logical epoch number.
   */
  void set_epoch(std::uint64_t epoch) override;

  /**
   * @brief Reports deterministic epoch-derived ordering support.
   * @return Always true because each instance owns a persisted base seed.
   */
  bool supports_epoch_control() const override { return true; }

  /**
   * @brief Returns the stable immutable sampler configuration identity.
   * @return Versioned identity containing the fixed population size.
   */
  std::string configuration_identity() const override;

  /**
   * @brief Reports exact generator-state checkpoint support.
   * @return Always true.
   */
  bool supports_state() const override { return true; }

  /**
   * @brief Serializes the epoch base seed and random engine state.
   * @return Versioned opaque state representation.
   */
  std::string save_state() const override;

  /**
   * @brief Restores generator state atomically.
   * @param state Value returned by save_state().
   * @throws jyppx::liuding::SerializationError The engine state is malformed.
   */
  void load_state(const std::string& state) override;

 private:
  size_t size_;  ///< Number of indices in each permutation; positive.
  std::uint64_t base_seed_;  ///< Persisted seed used to derive every epoch.
  mutable std::mt19937_64 generator_;  ///< Engine advanced by indices().
};

}  // namespace samplers
}  // namespace data
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_DATA_SAMPLERS_RANDOM_SAMPLER_HPP