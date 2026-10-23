// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_DATA_TRANSFORMS_TRANSFORM_HPP
#define JYPPX_LIUDING_DATA_TRANSFORMS_TRANSFORM_HPP
/**
 * @file transform.hpp
 * @brief Defines the generic callable transform interface.
 */

#include <jyppx/liuding/core/types.hpp>

#include <string>

namespace jyppx::liuding {
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
  /** @brief Destroys a transform through the base interface. */
  virtual ~Transform() = default;

  /**
   * @brief Applies the transform to one value.
   * @param input Input value that remains owned by the caller.
   * @return Transformed value.
   */
  virtual OutputType operator()(const InputType& input) const = 0;

  /**
   * @brief Returns a stable human-readable transform description.
   * @return Transform name suitable for diagnostics.
   */
  virtual std::string name() const = 0;
};

/** @brief Convenience alias for tensor-to-tensor transforms. */
using TensorTransform = Transform<Tensor, Tensor>;

}  // namespace transforms
}  // namespace data
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_DATA_TRANSFORMS_TRANSFORM_HPP