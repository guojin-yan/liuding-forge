// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_CORE_TYPES_HPP
#define JYPPX_LIUDING_CORE_TYPES_HPP
/**
 * @file types.hpp
 * @brief Defines the public LibTorch-backed value types used by JYPPX APIs.
 */

#include <torch/torch.h>

#include <optional>
#include <vector>

namespace jyppx::liuding {

/** Tensor handle used by framework APIs. */
using Tensor = torch::Tensor;

/** Ordered collection of tensor handles. */
using TensorList = std::vector<Tensor>;

/** Compute device descriptor used for placement and transfer operations. */
using Device = torch::Device;

/** Tensor scalar type descriptor used by precision-sensitive APIs. */
using Dtype = torch::Dtype;

/**
 * @brief Optional value used by public framework contracts.
 * @tparam T Contained value type.
 */
template <typename T>
using Optional = std::optional<T>;

}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_CORE_TYPES_HPP