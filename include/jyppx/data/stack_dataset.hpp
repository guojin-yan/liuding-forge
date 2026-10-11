#ifndef JYPPX_DATA_STACK_DATASET_HPP
#define JYPPX_DATA_STACK_DATASET_HPP

#include <jyppx/data/dataset.hpp>

#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>

namespace jyppx {
namespace data {

namespace detail {

// Zips one sample out of each dataset at `index` into a tuple.
template <typename... Ts, size_t... I>
auto stack_get_at(const std::tuple<std::shared_ptr<Dataset<Ts>>...>& datasets,
                  size_t index, std::index_sequence<I...>) {
    return std::make_tuple(std::get<I>(datasets)->get(index)...);
}

template <typename... Ts, size_t... I>
size_t stack_min_size(const std::tuple<std::shared_ptr<Dataset<Ts>>...>& datasets,
                      std::index_sequence<I...>) {
    size_t smallest = std::numeric_limits<size_t>::max();
    ((smallest = std::min(smallest, std::get<I>(datasets)->size())), ...);
    return smallest == std::numeric_limits<size_t>::max() ? 0 : smallest;
}

} // namespace detail

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
    explicit StackDataset(std::shared_ptr<Dataset<Ts>>... datasets)
        : datasets_(std::move(datasets)...) {
        bool all_defined = true;
        std::apply([&all_defined](const auto&... dataset) {
            ((all_defined = all_defined && static_cast<bool>(dataset)), ...);
        }, datasets_);
        if (!all_defined) {
            throw std::invalid_argument("StackDataset datasets cannot be null");
        }
        if (sizeof...(Ts) == 0) {
            throw std::invalid_argument("StackDataset requires at least one dataset");
        }
        if (size() == 0) {
            throw std::invalid_argument("StackDataset datasets must be non-empty");
        }
    }

    std::tuple<Ts...> get(size_t index) const override {
        if (index >= size()) {
            throw std::out_of_range("StackDataset index out of range");
        }
        return detail::stack_get_at(
            datasets_, index, std::index_sequence_for<Ts...>{});
    }

    size_t size() const override {
        return detail::stack_min_size(datasets_, std::index_sequence_for<Ts...>{});
    }

private:
    std::tuple<std::shared_ptr<Dataset<Ts>>...> datasets_;
};

// Deduction guide: StackDataset(ds1, ds2, ...) infers the sample types.
template <typename... Ts>
StackDataset(std::shared_ptr<Dataset<Ts>>... datasets) -> StackDataset<Ts...>;

} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_STACK_DATASET_HPP
