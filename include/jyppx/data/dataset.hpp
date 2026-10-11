#ifndef JYPPX_DATA_DATASET_HPP
#define JYPPX_DATA_DATASET_HPP

#include <jyppx/core/types.hpp>
#include <cstddef>
#include <vector>

namespace jyppx {
namespace data {

/**
 * @brief Base class for all datasets
 *
 * @tparam T Type of data samples
 */
template<typename T = Tensor>
class Dataset {
public:
    using value_type = T;

    virtual ~Dataset() = default;

    /**
     * @brief Get a single sample
     * @param index Sample index
     * @return Sample at the given index
     */
    virtual T get(size_t index) const = 0;

    /**
     * @brief Get dataset size
     * @return Number of samples in the dataset
     */
    virtual size_t size() const = 0;

    /** @brief Whether the dataset contains no samples. */
    bool empty() const { return size() == 0; }

    /**
     * @brief Operator[] overload for convenient access
     */
    T operator[](size_t index) const {
        return get(index);
    }

    /**
     * @brief Get a batch of samples
     * @param indices List of indices
     * @return Vector of samples
     */
    virtual std::vector<T> get_batch(const std::vector<size_t>& indices) const {
        std::vector<T> batch;
        batch.reserve(indices.size());
        for (size_t idx : indices) {
            batch.push_back(get(idx));
        }
        return batch;
    }
};

} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_DATASET_HPP
