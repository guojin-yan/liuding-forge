// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_DATA_SAMPLERS_SEQUENTIAL_SAMPLER_HPP
#define JYPPX_LIUDING_DATA_SAMPLERS_SEQUENTIAL_SAMPLER_HPP
/**
 * @file sequential_sampler.hpp
 * @brief Defines a stateless identity-order sampler.
 */

#include <jyppx/liuding/data/samplers/sampler.hpp>

namespace jyppx::liuding {
namespace data {
namespace samplers {

/**
 * @brief Produces the identity index sequence for a fixed population.
 *
 * The sampler is stateless and safe for concurrent const calls.
 */
class SequentialSampler : public Sampler {
 public:
  /**
   * @brief Creates a sampler for a fixed population.
   * @param size Number of sequential indices returned per epoch.
   */
  explicit SequentialSampler(size_t size);

  /**
   * @brief Returns indices in the range `[0, size())`.
   * @return Identity index sequence.
   */
  std::vector<size_t> indices() const override;

  /**
   * @brief Returns the population size.
   * @return Number of sequential indices.
   */
  size_t size() const override;

  /**
   * @brief Reports deterministic epoch control for identity ordering.
   * @return Always true because every epoch has the same defined order.
   */
  bool supports_epoch_control() const override { return true; }

  /**
   * @brief Returns the stable immutable sampler configuration identity.
   * @return Versioned identity containing the fixed population size.
   */
  std::string configuration_identity() const override {
    return "jyppx.sampler.sequential/v1:size=" + std::to_string(size_);
  }

  /**
   * @brief Reports checkpoint support; the state payload is empty.
   * @return Always true because the sampler is stateless.
   */
  bool supports_state() const override { return true; }

 private:
  size_t size_;  ///< Number of sequential indices in each epoch.
};

}  // namespace samplers
}  // namespace data
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_DATA_SAMPLERS_SEQUENTIAL_SAMPLER_HPP