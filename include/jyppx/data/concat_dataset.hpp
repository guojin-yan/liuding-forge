#ifndef JYPPX_DATA_CONCAT_DATASET_HPP
#define JYPPX_DATA_CONCAT_DATASET_HPP

/**
 * @file concat_dataset.hpp
 * @brief Defines a logical concatenation of map-style datasets.
 */

#include <jyppx/data/dataset.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jyppx {
namespace data {

/**
 * @brief Concatenation of several datasets with the same sample type
 *
 * Samples are addressed continuously across the concatenated datasets, in the
 * order the datasets are given.
 *
 * @tparam T Sample type
 */
template <typename T = Tensor>
class ConcatDataset : public Dataset<T> {
 public:
  /**
   * @brief Creates a concatenated view in caller-provided order.
   * @param datasets Non-empty collection of non-null datasets.
   * @throws std::invalid_argument The collection is empty or contains null.
   */
  explicit ConcatDataset(std::vector<std::shared_ptr<Dataset<T>>> datasets)
      : datasets_(std::move(datasets)) {
    if (datasets_.empty()) {
      throw std::invalid_argument(
          "ConcatDataset requires at least one dataset");
    }
    cumulative_sizes_.reserve(datasets_.size());
    size_t cumulative = 0;
    for (const auto& dataset : datasets_) {
      if (!dataset) {
        throw std::invalid_argument("ConcatDataset datasets cannot be null");
      }
      cumulative += dataset->size();
      cumulative_sizes_.push_back(cumulative);
    }
  }

  /**
   * @brief Resolves and returns one logical sample.
   * @param index Zero-based index in the concatenated view.
   * @return Sample from the owning child dataset.
   * @throws std::out_of_range The index is outside the combined range.
   */
  T get(size_t index) const override {
    const auto [dataset, local_index] = locate(index);
    return dataset->get(local_index);
  }

  /**
   * @brief Returns the sum of all child dataset sizes.
   * @return Combined logical sample count.
   */
  size_t size() const override { return cumulative_sizes_.back(); }

  /**
   * @brief Returns cumulative sample counts, one entry per child dataset.
   * @return Inclusive child end offsets in dataset order.
   */
  const std::vector<size_t>& cumulative_sizes() const {
    return cumulative_sizes_;
  }

 private:
  /**
   * @brief Maps a concatenated index to a child dataset and local index.
   * @param index Zero-based index in the concatenated view.
   * @return Child dataset handle and its local index.
   * @throws std::out_of_range The index is outside the combined range.
   */
  std::pair<std::shared_ptr<Dataset<T>>, size_t> locate(size_t index) const {
    if (index >= size()) {
      throw std::out_of_range("ConcatDataset index out of range");
    }
    const auto upper = std::upper_bound(cumulative_sizes_.begin(),
                                        cumulative_sizes_.end(), index);
    const size_t dataset_index =
        static_cast<size_t>(std::distance(cumulative_sizes_.begin(), upper));
    const size_t previous =
        dataset_index == 0 ? 0 : cumulative_sizes_[dataset_index - 1];
    return {datasets_[dataset_index], index - previous};
  }

  std::vector<std::shared_ptr<Dataset<T>>>
      datasets_;  ///< Child datasets in logical sample order.
  std::vector<size_t> cumulative_sizes_;  ///< Inclusive child end offsets.
};

}  // namespace data
}  // namespace jyppx

#endif  // JYPPX_DATA_CONCAT_DATASET_HPP
