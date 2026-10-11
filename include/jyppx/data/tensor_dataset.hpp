#ifndef JYPPX_DATA_TENSOR_DATASET_HPP
#define JYPPX_DATA_TENSOR_DATASET_HPP

#include <jyppx/data/dataset.hpp>

namespace jyppx {
namespace data {

/**
 * @brief Simple example structure for tensor dataset
 */
struct Example {
    Tensor data;
    Tensor target;
};

/**
 * @brief Dataset wrapping tensors
 */
class TensorDataset : public Dataset<Example> {
public:
    using ExampleType = Example;

    /**
     * @brief Constructor
     * @param data Input data tensor
     * @param targets Target tensor
     */
    TensorDataset(const Tensor& data, const Tensor& targets);

    /**
     * @brief Get a single sample
     */
    Example get(size_t index) const override;

    /**
     * @brief Get dataset size
     */
    size_t size() const override;

    /** @brief Access the underlying input tensor. */
    const Tensor& data() const { return data_; }

    /** @brief Access the underlying target tensor. */
    const Tensor& targets() const { return targets_; }

private:
    Tensor data_;
    Tensor targets_;
};

} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_TENSOR_DATASET_HPP
