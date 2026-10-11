#ifndef JYPPX_DATA_SAMPLERS_SAMPLER_HPP
#define JYPPX_DATA_SAMPLERS_SAMPLER_HPP

#include <cstddef>
#include <vector>

namespace jyppx {
namespace data {
namespace samplers {

/**
 * @brief Base class for all samplers
 */
class Sampler {
public:
    virtual ~Sampler() = default;

    /**
     * @brief Get all indices
     * @return Vector of indices
     */
    virtual std::vector<size_t> indices() const = 0;

    /**
     * @brief Get number of samples
     */
    virtual size_t size() const = 0;
};

} // namespace samplers
} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_SAMPLERS_SAMPLER_HPP
