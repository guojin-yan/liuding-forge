#ifndef JYPPX_DATA_TENSOR_DATASET_HPP
#define JYPPX_DATA_TENSOR_DATASET_HPP

/**
 * @file tensor_dataset.hpp
 * @brief Defines paired input/target samples backed by two tensors.
 */

#include <jyppx/data/dataset.hpp>

namespace jyppx {
namespace data {

/**
 * @brief One input tensor and its corresponding target tensor.
 */
struct Example {
  Tensor data;    ///< Input slice for one sample.
  Tensor target;  ///< Target slice with the same leading index.
};

/**
 * @brief Map-style dataset over matching leading dimensions of two tensors.
 */
class TensorDataset : public Dataset<Example> {
 public:
  /** Public sample type used by callers and deduction helpers. */
  using ExampleType = Example;

  /**
   * @brief Creates a paired tensor dataset.
   * @param data Defined tensor with a leading sample dimension.
   * @param targets Defined target tensor with the same leading size.
   * @throws jyppx::InvalidArgument A tensor is undefined, scalar, or has a
   * mismatched leading size.
   */
  TensorDataset(const Tensor& data, const Tensor& targets);

  /**
   * @brief Returns matching input and target slices.
   * @param index Zero-based sample index.
   * @return Paired slices at @p index.
   * @throws jyppx::InvalidArgument The index is outside the dataset.
   */
  Example get(size_t index) const override;

  /**
   * @brief Returns the shared leading dimension.
   * @return Number of paired samples.
   */
  size_t size() const override;

  /**
   * @brief Returns the underlying input tensor handle.
   * @return Retained input tensor.
   */
  const Tensor& data() const { return data_; }

  /**
   * @brief Returns the underlying target tensor handle.
   * @return Retained target tensor.
   */
  const Tensor& targets() const { return targets_; }

 private:
  Tensor data_;     ///< Input tensor retaining caller storage by handle.
  Tensor targets_;  ///< Target tensor retaining caller storage by handle.
};

}  // namespace data
}  // namespace jyppx

#endif  // JYPPX_DATA_TENSOR_DATASET_HPP
