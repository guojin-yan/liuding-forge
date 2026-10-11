#ifndef JYPPX_DATA_SAMPLERS_BATCH_SAMPLER_HPP
#define JYPPX_DATA_SAMPLERS_BATCH_SAMPLER_HPP

#include <jyppx/data/samplers/sampler.hpp>
#include <memory>

namespace jyppx {
namespace data {
namespace samplers {

/**
 * @brief Batch sampler wrapping another sampler
 */
class BatchSampler : public Sampler {
public:
    BatchSampler(
        std::shared_ptr<Sampler> sampler,
        size_t batch_size,
        bool drop_last = false
    );

    std::vector<size_t> indices() const override;
    size_t size() const override;

    /** @brief Number of samples in each full batch. */
    size_t batch_size() const { return batch_size_; }

private:
    std::shared_ptr<Sampler> sampler_;
    size_t batch_size_;
    bool drop_last_;
};

} // namespace samplers
} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_SAMPLERS_BATCH_SAMPLER_HPP
