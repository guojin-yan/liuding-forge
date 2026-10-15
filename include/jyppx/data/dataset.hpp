#ifndef JYPPX_DATA_DATASET_HPP
#define JYPPX_DATA_DATASET_HPP

/**
 * @file dataset.hpp
 * @brief Defines the map-style dataset contract used by JYPPX data loaders.
 */

#include <jyppx/core/types.hpp>

#include <cstddef>
#include <vector>

namespace jyppx {
namespace data {

/**
 * @brief Random-access source of a finite sequence of samples.
 * @tparam T Value returned for one sample.
 * @note Implementations must keep `size()` stable while a DataLoader epoch is
 * active. Concurrent `get()` calls require an implementation to provide its
 * own synchronization.
 */
template <typename T = Tensor>
class Dataset {
 public:
  /** Value returned by `get()` and used for DataLoader type deduction. */
  using value_type = T;

  /** @brief Destroys a dataset through the base interface. */
  virtual ~Dataset() = default;

  /**
   * @brief Returns one sample.
   * @param index Zero-based sample index, less than `size()`.
   * @return Sample at @p index.
   * @throws std::out_of_range The index is outside the dataset.
   */
  virtual T get(size_t index) const = 0;

  /**
   * @brief Returns the number of addressable samples.
   * @return Stable sample count for the current epoch.
   */
  virtual size_t size() const = 0;

  /**
   * @brief Returns whether the dataset contains no samples.
   * @return True when `size()` is zero.
   */
  bool empty() const { return size() == 0; }

  /**
   * @brief Returns one sample using subscript syntax.
   * @param index Zero-based sample index, less than `size()`.
   * @return Sample at @p index.
   */
  T operator[](size_t index) const { return get(index); }

  /**
   * @brief Returns samples in caller-provided order.
   * @param indices Zero-based sample indices; duplicates are preserved.
   * @return Samples corresponding one-for-one with @p indices.
   * @throws std::out_of_range Any index is outside the dataset.
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

}  // namespace data
}  // namespace jyppx

#endif  // JYPPX_DATA_DATASET_HPP
