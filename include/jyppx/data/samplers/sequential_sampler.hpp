#ifndef JYPPX_DATA_SAMPLERS_SEQUENTIAL_SAMPLER_HPP
#define JYPPX_DATA_SAMPLERS_SEQUENTIAL_SAMPLER_HPP

#include <jyppx/data/samplers/sampler.hpp>

namespace jyppx {
namespace data {
namespace samplers {

/**
 * @brief Produces the identity index sequence for a fixed population.
 *
 * The sampler is stateless and safe for concurrent const calls.
 */
class SequentialSampler : public Sampler {
 public:
  /** @brief Creates a sampler for @p size entries. */
  explicit SequentialSampler(size_t size);

  /** @brief Returns indices in the range `[0, size())`. */
  std::vector<size_t> indices() const override;

  /** @brief Returns the population size. */
  size_t size() const override;

  /** @brief Reports checkpoint support; the state payload is empty. */
  bool supports_state() const override { return true; }

 private:
  size_t size_;  ///< Number of sequential indices in each epoch.
};

}  // namespace samplers
}  // namespace data
}  // namespace jyppx

#endif  // JYPPX_DATA_SAMPLERS_SEQUENTIAL_SAMPLER_HPP
