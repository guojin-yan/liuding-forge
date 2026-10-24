// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_DATA_STACK_DATASET_HPP
#define JYPPX_LIUDING_DATA_STACK_DATASET_HPP
/**
 * @file stack_dataset.hpp
 * @brief Defines a positional zip over multiple map-style datasets.
 */

#include <jyppx/liuding/data/dataset.hpp>
#include <jyppx/liuding/data/detail/error.hpp>

#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <variant>

namespace jyppx::liuding {
namespace data {

namespace detail {

/**
 * @brief Reads one sample from each tuple element at a shared index.
 * @tparam Ts Sample types owned by the input datasets.
 * @tparam I Tuple element indices expanded by the caller.
 * @param datasets Non-null dataset handles.
 * @param index Shared zero-based sample index.
 * @return Tuple containing one sample from every dataset.
 */
template <typename... Ts, size_t... I>
auto stack_get_at(const std::tuple<std::shared_ptr<Dataset<Ts>>...>& datasets,
                  size_t index, std::index_sequence<I...>) {
  return std::make_tuple(std::get<I>(datasets)->get(index)...);
}

/**
 * @brief Computes the addressable prefix shared by all tuple elements.
 * @tparam Ts Sample types owned by the input datasets.
 * @tparam I Tuple element indices expanded by the caller.
 * @param datasets Non-null dataset handles.
 * @return Smallest child size, or zero for an empty tuple.
 */
template <typename... Ts, size_t... I>
size_t stack_min_size(
    const std::tuple<std::shared_ptr<Dataset<Ts>>...>& datasets,
    std::index_sequence<I...>) {
  size_t smallest = std::numeric_limits<size_t>::max();
  ((smallest = std::min(smallest, std::get<I>(datasets)->size())), ...);
  return smallest == std::numeric_limits<size_t>::max() ? 0 : smallest;
}

}  // namespace detail

/**
 * @brief Zips multiple datasets into tuple samples
 *
 * Mirrors torch.utils.data.StackDataset for the map-style case: sample i is
 * `tuple(d1[i], d2[i], ...)`, where the length is the smallest dataset size.
 * Consume the tuple samples in manual loops or through a collate function
 * that reduces the tuple to the batch type your Trainer needs.
 *
 * @tparam Ts Sample types of the zipped datasets
 */
template <typename... Ts>
class StackDataset : public Dataset<std::tuple<Ts...>> {
 public:
  /**
   * @brief Creates a positional zip of non-empty datasets.
   * @param datasets Non-null dataset handles, one per sample type.
   * @throws jyppx::liuding::InvalidArgument No datasets are supplied, a handle
   * is null, or the smallest dataset is empty.
   */
  explicit StackDataset(std::shared_ptr<Dataset<Ts>>... datasets)
      : datasets_(std::move(datasets)...) {
    bool all_defined = true;
    std::apply(
        [&all_defined](const auto&... dataset) {
          ((all_defined = all_defined && static_cast<bool>(dataset)), ...);
        },
        datasets_);
    if (!all_defined) {
      detail::throw_invalid("data.stack_dataset", "constructor.datasets",
                            "non-null datasets", "null dataset",
                            "Provide a dataset handle for every sample type.");
    }
    if (sizeof...(Ts) == 0) {
      detail::throw_invalid("data.stack_dataset", "constructor.datasets",
                            "at least one dataset", "empty parameter pack",
                            "Provide at least one dataset.");
    }
    if (size() == 0) {
      detail::throw_invalid("data.stack_dataset", "constructor.datasets",
                            "non-empty datasets", "empty dataset",
                            "Provide datasets containing at least one sample.");
    }
  }

  /**
   * @brief Returns one tuple of positionally matched samples.
   * @param index Zero-based index less than `size()`.
   * @return Tuple containing one sample from every child dataset.
   * @throws jyppx::liuding::InvalidArgument The index is outside the shared
   * prefix.
   */
  std::tuple<Ts...> get(size_t index) const override {
    if (index >= size()) {
      detail::throw_index_error("data.stack_dataset", "get", index, size());
    }
    return detail::stack_get_at(datasets_, index,
                                std::index_sequence_for<Ts...>{});
  }

  /**
   * @brief Returns the smallest child dataset size.
   * @return Number of addressable tuple samples.
   */
  size_t size() const override {
    return detail::stack_min_size(datasets_, std::index_sequence_for<Ts...>{});
  }

 private:
  std::tuple<std::shared_ptr<Dataset<Ts>>...>
      datasets_;  ///< Child datasets retained in tuple order.
};

/**
 * @brief Deduces sample types from dataset pointer arguments.
 * @tparam Ts Dataset sample types.
 * @param datasets Dataset handles used solely for type deduction.
 */
template <typename... Ts>
StackDataset(std::shared_ptr<Dataset<Ts>>... datasets) -> StackDataset<Ts...>;

}  // namespace data
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_DATA_STACK_DATASET_HPP