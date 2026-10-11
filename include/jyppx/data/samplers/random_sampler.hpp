#ifndef JYPPX_DATA_SAMPLERS_RANDOM_SAMPLER_HPP
#define JYPPX_DATA_SAMPLERS_RANDOM_SAMPLER_HPP

#include <jyppx/data/samplers/sampler.hpp>
#include <jyppx/core/types.hpp>
#include <random>

namespace jyppx {
namespace data {
namespace samplers {

/**
 * @brief Random sampler
 */
class RandomSampler : public Sampler {
public:
    explicit RandomSampler(size_t size, Optional<int64_t> seed = std::nullopt);

    std::vector<size_t> indices() const override;
    size_t size() const override;

private:
    size_t size_;
    mutable std::mt19937_64 generator_;
};

} // namespace samplers
} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_SAMPLERS_RANDOM_SAMPLER_HPP
