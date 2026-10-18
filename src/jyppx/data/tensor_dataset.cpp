#include <jyppx/data/tensor_dataset.hpp>
#include <jyppx/data/detail/error.hpp>

namespace jyppx::data {

TensorDataset::TensorDataset(const Tensor& data, const Tensor& targets)
    : data_(data), targets_(targets) {
  if (!data_.defined() || !targets_.defined()) {
    detail::throw_invalid("data.tensor_dataset", "constructor.tensors",
                          "defined data and targets", "undefined tensor",
                          "Provide defined input and target tensors.");
  }
  if (data_.dim() == 0 || targets_.dim() == 0) {
    detail::throw_invalid("data.tensor_dataset", "constructor.shape",
                          "data and targets with a leading batch dimension",
                          "scalar tensor",
                          "Provide tensors with dimension zero as samples.");
  }
  if (data_.size(0) != targets_.size(0)) {
    detail::throw_invalid(
        "data.tensor_dataset", "constructor.batch_size",
        "matching leading dimensions", "data=" + std::to_string(data_.size(0)) +
            ", targets=" + std::to_string(targets_.size(0)),
        "Provide one target for every input sample.");
  }
}

Example TensorDataset::get(size_t index) const {
  if (index >= size()) {
    detail::throw_index_error("data.tensor_dataset", "get", index, size());
  }
  return Example{data_[static_cast<int64_t>(index)],
                 targets_[static_cast<int64_t>(index)]};
}

size_t TensorDataset::size() const {
  return static_cast<size_t>(data_.size(0));
}

}  // namespace jyppx::data
