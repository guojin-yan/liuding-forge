#ifndef JYPPX_DATA_SAMPLERS_SEQUENTIAL_SAMPLER_HPP
#define JYPPX_DATA_SAMPLERS_SEQUENTIAL_SAMPLER_HPP

#include <jyppx/data/samplers/sampler.hpp>

namespace jyppx {
namespace data {
namespace samplers {

/**
 * @brief Sequential sampler
 */
class SequentialSampler : public Sampler {
public:
    explicit SequentialSampler(size_t size);

    std::vector<size_t> indices() const override;
    size_t size() const override;

private:
    size_t size_;
};

} // namespace samplers
} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_SAMPLERS_SEQUENTIAL_SAMPLER_HPP
