#ifndef JYPPX_DATA_SUBSET_HPP
#define JYPPX_DATA_SUBSET_HPP

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

    T get(size_t index) const override {
        if (index >= indices_.size()) {
            throw std::out_of_range("Subset index out of range");
        }
        return dataset_->get(indices_[index]);
    }

    size_t size() const override { return indices_.size(); }

    /** @brief Underlying dataset. */
    const std::shared_ptr<Dataset<T>>& dataset() const { return dataset_; }

    /** @brief The selected sample indices, in subset order. */
    const std::vector<size_t>& indices() const { return indices_; }

private:
    std::shared_ptr<Dataset<T>> dataset_;
    std::vector<size_t> indices_;
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
 * @param dataset Dataset to split
 * @param lengths Sizes of the resulting subsets; must sum to dataset size
 * @param seed Optional seed for the shuffle
 */
template <typename DatasetT, typename T = typename DatasetT::value_type>
std::vector<std::shared_ptr<Subset<T>>> random_split(
    std::shared_ptr<DatasetT> dataset,
    const std::vector<size_t>& lengths,
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
        std::mt19937_64 generator(
            (static_cast<uint64_t>(device()) << 32) ^ static_cast<uint64_t>(device()));
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

} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_SUBSET_HPP
