// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_DATA_SUBSET_HPP
#define JYPPX_LIUDING_DATA_SUBSET_HPP
/**
 * @file subset.hpp
 * @brief Defines indexed dataset views and deterministic random partitioning.
 */

#include <jyppx/liuding/data/dataset.hpp>
#include <jyppx/liuding/data/detail/error.hpp>

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace jyppx::liuding {
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
   * @throws jyppx::liuding::InvalidArgument @p dataset is null or a source
   * index is invalid.
   */
  Subset(std::shared_ptr<Dataset<T>> dataset, std::vector<size_t> indices)
      : dataset_(std::move(dataset)), indices_(std::move(indices)) {
    if (!dataset_) {
      detail::throw_invalid("data.subset", "constructor.dataset",
                            "non-null dataset", "null dataset",
                            "Provide a valid source dataset.");
    }
    for (const size_t index : indices_) {
      if (index >= dataset_->size()) {
        detail::throw_index_error("data.subset", "constructor.indices", index,
                                  dataset_->size());
      }
    }
  }

  /**
   * @brief Returns the source sample selected by a subset position.
   * @param index Zero-based position in this view.
   * @return Source sample referenced by the selected index.
   * @throws jyppx::liuding::InvalidArgument The subset position is invalid.
   */
  T get(size_t index) const override {
    if (index >= indices_.size()) {
      detail::throw_index_error("data.subset", "get", index, indices_.size());
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
 * captured by a DataPipeline batch factory. The sample type is deduced from
 * the dataset's value_type.
 *
 * @tparam DatasetT Concrete dataset type exposing `value_type`.
 * @tparam T Dataset sample type deduced from @p dataset.
 * @param dataset Non-null dataset to partition.
 * @param lengths Result sizes whose sum must equal the dataset size.
 * @param seed Optional deterministic shuffle seed.
 * @return Disjoint subsets in the same order as @p lengths.
 * @throws jyppx::liuding::InvalidArgument The dataset is null or lengths do not
 * cover the dataset.
 */
template <typename DatasetT, typename T = typename DatasetT::value_type>
std::vector<std::shared_ptr<Subset<T>>> random_split(
    std::shared_ptr<DatasetT> dataset, const std::vector<size_t>& lengths,
    Optional<int64_t> seed = std::nullopt) {
  if (!dataset) {
    detail::throw_invalid("data.random_split", "dataset", "non-null dataset",
                          "null dataset", "Provide a valid source dataset.");
  }
  size_t total = 0;
  for (const size_t length : lengths) {
    total += length;
  }
  if (total != dataset->size()) {
    detail::throw_invalid(
        "data.random_split", "lengths", "sum equal to dataset size",
        "sum=" + std::to_string(total) +
            ", dataset_size=" + std::to_string(dataset->size()),
        "Adjust split lengths to cover every sample exactly once.");
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
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_DATA_SUBSET_HPP