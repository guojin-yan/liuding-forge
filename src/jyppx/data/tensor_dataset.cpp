#include <jyppx/data/tensor_dataset.hpp>

#include <stdexcept>

namespace jyppx::data {

TensorDataset::TensorDataset(const Tensor& data, const Tensor& targets)
    : data_(data), targets_(targets) {
    if (!data_.defined() || !targets_.defined()) {
        throw std::invalid_argument("TensorDataset requires defined data and targets");
    }
    if (data_.dim() == 0 || targets_.dim() == 0) {
        throw std::invalid_argument("TensorDataset data and targets must have a batch dimension");
    }
    if (data_.size(0) != targets_.size(0)) {
        throw std::invalid_argument("TensorDataset data and targets must have the same size");
    }
}

Example TensorDataset::get(size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("TensorDataset index out of range");
    }
    return Example{data_[static_cast<int64_t>(index)], targets_[static_cast<int64_t>(index)]};
}

size_t TensorDataset::size() const {
    return static_cast<size_t>(data_.size(0));
}

} // namespace jyppx::data
