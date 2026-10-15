#ifndef JYPPX_DATA_SUBSET_HPP
#define JYPPX_DATA_SUBSET_HPP

/**
 * @file subset.hpp
 * @brief Defines indexed dataset views and deterministic random partitioning.
 */

#include <jyppx/data/dataset.hpp>

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace jyppx {
namespace data {

/**
 * @brief View of a dataset restricted to the given sample indices
 *
 * Wraps an existing dataset and exposes only the selected samples, in the
 * order the indices are given. Commonly used with random_split().
 *
 * @tparam T Sample type
 */
template <typename T = Tensor>
class Subset : public Dataset<T> {
 public:
  /**
   * @brief Creates an ordered indexed view over a dataset.
   * @param dataset Shared non-null source dataset.
   * @param indices Source indices in exposed subset order; duplicates allowed.
   * @throws std::invalid_argument @p dataset is null.
   * @throws std::out_of_range Any source index is invalid.
   */
  Subset(std::shared_ptr<Dataset<T>> dataset, std::vector<size_t> indices)
      : dataset_(std::move(dataset)), indices_(std::move(indices)) {
    if (!dataset_) {
      throw std::invalid_argument("Subset dataset cannot be null");
    }
    for (const size_t index : indices_) {
      if (index >= dataset_->size()) {
        throw std::out_of_range("Subset index is outside the dataset");
      }
    }
  }

  /**
   * @brief Returns the source sample selected by a subset position.
   * @param index Zero-based position in this view.
   * @return Source sample referenced by the selected index.
   * @throws std::out_of_range The subset position is invalid.
   */
  T get(size_t index) const override {
    if (index >= indices_.size()) {
      throw std::out_of_range("Subset index out of range");
    }
    return dataset_->get(indices_[index]);
  }

  /**
   * @brief Returns the number of selected positions.
   * @return Subset sample count.
   */
  size_t size() const override { return indices_.size(); }

  /**
   * @brief Returns the retained source dataset handle.
   * @return Shared source dataset.
   */
  const std::shared_ptr<Dataset<T>>& dataset() const { return dataset_; }

  /**
   * @brief Returns selected source indices in exposed order.
   * @return Source indices retained by this view.
   */
  const std::vector<size_t>& indices() const { return indices_; }

 private:
  std::shared_ptr<Dataset<T>> dataset_;  ///< Retained non-null source dataset.
  std::vector<size_t> indices_;  ///< Valid source indices in exposed order.
};

/**
 * @brief Split a dataset into random subsets of the given lengths
 *
 * Mirrors torch.utils.data.random_split: the sample indices are shuffled
 * (optionally with a seed) and then assigned to consecutive subsets whose
 * sizes are `lengths`. The returned subsets are heap-allocated so they can be
 * handed to a DataLoader directly. The sample type is deduced from the
 * dataset's value_type.
 *
 * @tparam DatasetT Concrete dataset type exposing `value_type`.
 * @tparam T Dataset sample type deduced from @p dataset.
 * @param dataset Non-null dataset to partition.
 * @param lengths Result sizes whose sum must equal the dataset size.
 * @param seed Optional deterministic shuffle seed.
 * @return Disjoint subsets in the same order as @p lengths.
 * @throws std::invalid_argument The dataset is null or lengths do not cover it.
 */
template <typename DatasetT, typename T = typename DatasetT::value_type>
std::vector<std::shared_ptr<Subset<T>>> random_split(
    std::shared_ptr<DatasetT> dataset, const std::vector<size_t>& lengths,
    Optional<int64_t> seed = std::nullopt) {
  if (!dataset) {
    throw std::invalid_argument("random_split dataset cannot be null");
  }
  size_t total = 0;
  for (const size_t length : lengths) {
    total += length;
  }
  if (total != dataset->size()) {
    throw std::invalid_argument(
        "random_split lengths must sum to the dataset size");
  }

  std::vector<size_t> indices(dataset->size());
  std::iota(indices.begin(), indices.end(), size_t{0});
  if (seed.has_value()) {
    std::mt19937_64 generator(static_cast<std::mt19937_64::result_type>(*seed));
    std::shuffle(indices.begin(), indices.end(), generator);
  } else {
    std::random_device device;
    std::mt19937_64 generator((static_cast<uint64_t>(device()) << 32) ^
                              static_cast<uint64_t>(device()));
    std::shuffle(indices.begin(), indices.end(), generator);
  }

  std::vector<std::shared_ptr<Subset<T>>> subsets;
  subsets.reserve(lengths.size());
  size_t offset = 0;
  for (const size_t length : lengths) {
    std::vector<size_t> portion(
        indices.begin() + static_cast<std::ptrdiff_t>(offset),
        indices.begin() + static_cast<std::ptrdiff_t>(offset + length));
    offset += length;
    subsets.push_back(std::make_shared<Subset<T>>(dataset, std::move(portion)));
  }
  return subsets;
}

}  // namespace data
}  // namespace jyppx

#endif  // JYPPX_DATA_SUBSET_HPP
