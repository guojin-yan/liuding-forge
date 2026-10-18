#ifndef JYPPX_DATA_SAMPLERS_BATCH_SAMPLER_HPP
#define JYPPX_DATA_SAMPLERS_BATCH_SAMPLER_HPP

/**
 * @file batch_sampler.hpp
 * @brief Defines batch-tail policy over an underlying sampler.
 */

#include <jyppx/data/samplers/sampler.hpp>
#include <jyppx/data/detail/error.hpp>

#include <memory>

namespace jyppx {
namespace data {
namespace samplers {

/**
 * @brief Truncates an underlying sample sequence to complete batches if
 * requested.
 *
 * The underlying sampler is shared and owns all mutable ordering state.
 */
class BatchSampler : public Sampler {
 public:
  /**
   * @brief Creates a batch-aware sampler view.
   * @param sampler Shared underlying sample-order provider; must not be null.
   * @param batch_size Samples per complete batch; must be positive.
   * @param drop_last Whether to omit the final incomplete batch.
   * @throws jyppx::InvalidArgument A required argument is invalid.
   */
  BatchSampler(std::shared_ptr<Sampler> sampler, size_t batch_size,
               bool drop_last = false);

  /**
   * @brief Returns one epoch of indices, truncated when drop-last is enabled.
   * @return Underlying sequence after applying the tail policy.
   */
  std::vector<size_t> indices() const override;

  /**
   * @brief Returns the number of batches yielded after truncation.
   * @return Complete and optional partial batch count.
   */
  size_t size() const override;

  /**
   * @brief Delegates checkpoint capability to the underlying sampler.
   * @return Underlying sampler checkpoint capability.
   */
  bool supports_state() const override { return sampler_->supports_state(); }

  /**
   * @brief Delegates state serialization to the underlying sampler.
   * @return Opaque underlying sampler state.
   */
  std::string save_state() const override { return sampler_->save_state(); }

  /**
   * @brief Delegates state restoration to the underlying sampler.
   * @param state Opaque state accepted by the underlying sampler.
   */
  void load_state(const std::string& state) override {
    sampler_->load_state(state);
  }

  /**
   * @brief Returns the number of samples in each full batch.
   * @return Positive configured batch size.
   */
  size_t batch_size() const { return batch_size_; }

 private:
  std::shared_ptr<Sampler> sampler_;  ///< Shared provider of sample ordering.
  size_t batch_size_;                 ///< Samples per complete batch; positive.
  bool drop_last_;  ///< Whether an incomplete final batch is removed.
};

}  // namespace samplers
}  // namespace data
}  // namespace jyppx

#endif  // JYPPX_DATA_SAMPLERS_BATCH_SAMPLER_HPP
