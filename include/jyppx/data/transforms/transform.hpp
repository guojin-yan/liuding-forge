#ifndef JYPPX_DATA_TRANSFORMS_TRANSFORM_HPP
#define JYPPX_DATA_TRANSFORMS_TRANSFORM_HPP

#include <jyppx/core/types.hpp>

#include <string>

namespace jyppx {
namespace data {
namespace transforms {

/**
 * @brief Base class for all transforms
 *
 * A transform is a callable that maps an input to an output; `name()` is used
 * for debugging and for the Compose/RandomChoice descriptions.
 *
 * @tparam InputType Transform input type
 * @tparam OutputType Transform output type (defaults to InputType)
 */
template <typename InputType, typename OutputType = InputType>
class Transform {
public:
    virtual ~Transform() = default;

    virtual OutputType operator()(const InputType& input) const = 0;
    virtual std::string name() const = 0;
};

/** @brief Convenience alias for tensor-to-tensor transforms. */
using TensorTransform = Transform<Tensor, Tensor>;

} // namespace transforms
} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_TRANSFORMS_TRANSFORM_HPP
