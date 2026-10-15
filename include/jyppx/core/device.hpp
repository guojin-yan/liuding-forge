// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_CORE_DEVICE_HPP
#define JYPPX_CORE_DEVICE_HPP

/**
 * @file device.hpp
 * @brief Provides bounded device parsing and runtime accelerator discovery.
 */

#include "jyppx/core/types.hpp"

#include <string>

namespace jyppx {

/**
 * @brief Selects the default compute device.
 * @return A CUDA device when the backend is available; otherwise a CPU
 * device.
 * @throws c10::Error If LibTorch cannot query CUDA availability.
 */
Device get_default_device();

/**
 * @brief Reports whether a usable CUDA backend is available.
 * @return `true` when LibTorch can use CUDA, or `false` otherwise.
 * @throws c10::Error If LibTorch cannot initialize the CUDA availability
 * query.
 */
bool cuda_is_available();

/**
 * @brief Returns the number of CUDA devices visible to LibTorch.
 * @return Non-negative visible CUDA device count.
 * @throws c10::Error If LibTorch cannot query the active CUDA runtime.
 */
int cuda_device_count();

/**
 * @brief Parses a bounded device specification.
 *
 * Accepted values follow LibTorch's device grammar, for example `cpu`,
 * `cuda`, and `cuda:0`. The input must contain between 1 and 64 bytes and
 * must not contain an embedded NUL byte. The size bound keeps malformed
 * configuration or protocol input from causing unbounded diagnostic
 * allocations.
 *
 * @param device_str Device specification to parse.
 * @return Parsed LibTorch device.
 * @throws std::invalid_argument The string is empty, too long, contains an
 * embedded NUL byte, or is not a valid device specification.
 */
Device parse_device(const std::string& device_str);

}  // namespace jyppx

#endif  // JYPPX_CORE_DEVICE_HPP
