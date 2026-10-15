// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "jyppx/core/tensor_utils.hpp"

namespace jyppx {

bool has_nan(const Tensor& tensor) {
  return tensor.defined() && torch::isnan(tensor).any().item<bool>();
}

bool has_inf(const Tensor& tensor) {
  return tensor.defined() && torch::isinf(tensor).any().item<bool>();
}

bool is_finite(const Tensor& tensor) {
  return tensor.defined() && torch::isfinite(tensor).all().item<bool>();
}

TensorList clone_tensors(const TensorList& tensors) {
  TensorList clones;
  clones.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    clones.push_back(tensor.defined() ? tensor.clone() : Tensor{});
  }
  return clones;
}

TensorList to_device(const TensorList& tensors, const Device& device) {
  TensorList moved;
  moved.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    moved.push_back(tensor.defined() ? tensor.to(device) : Tensor{});
  }
  return moved;
}

}  // namespace jyppx
