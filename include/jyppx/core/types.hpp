#ifndef JYPPX_CORE_TYPES_HPP
#define JYPPX_CORE_TYPES_HPP

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <optional>
#include <functional>

namespace jyppx {

// Tensor alias
using Tensor = torch::Tensor;

// TensorList alias
using TensorList = std::vector<Tensor>;

// Device type
using Device = torch::Device;

// Dtype alias
using Dtype = torch::Dtype;

// Optional types
template<typename T>
using Optional = std::optional<T>;

} // namespace jyppx

#endif // JYPPX_CORE_TYPES_HPP
