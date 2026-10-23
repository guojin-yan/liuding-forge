// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "jyppx/liuding/core/device.hpp"

#include <torch/cuda.h>

#include <cstddef>
#include <exception>

namespace jyppx::liuding {

namespace {

constexpr std::size_t kMaxDeviceStringLength = 64;

}  // namespace

Device get_default_device() {
  return cuda_is_available() ? Device(torch::kCUDA) : Device(torch::kCPU);
}

bool cuda_is_available() { return torch::cuda::is_available(); }

int cuda_device_count() {
  return static_cast<int>(torch::cuda::device_count());
}

Device parse_device(const std::string& device_str) {
  if (device_str.empty()) {
    throw InvalidArgument(
        "Device string must not be empty",
        ErrorDetails{.component = "core.device",
                     .operation = "parse_device",
                     .expected = "a non-empty device specification",
                     .actual = "empty",
                     .hint = "Use cpu, cuda, or cuda:<index>"});
  }
  if (device_str.size() > kMaxDeviceStringLength) {
    throw InvalidArgument(
        "Device string exceeds the 64-byte limit",
        ErrorDetails{.component = "core.device",
                     .operation = "parse_device",
                     .expected = "at most 64 bytes",
                     .actual = "more than 64 bytes",
                     .hint = "Use a bounded device specification"});
  }
  if (device_str.find('\0') != std::string::npos) {
    throw InvalidArgument(
        "Device string must not contain an embedded NUL byte",
        ErrorDetails{.component = "core.device",
                     .operation = "parse_device",
                     .expected = "a NUL-free device specification",
                     .actual = "embedded NUL byte",
                     .hint = "Remove the embedded NUL byte"});
  }

  try {
    return Device(device_str);
  } catch (const std::exception& error) {
    throw InvalidArgument(
        "Invalid device string: " + std::string(error.what()),
        ErrorDetails{.component = "core.device",
                     .operation = "parse_device",
                     .expected = "cpu, cuda, or cuda:<index>",
                     .actual = "rejected by the LibTorch device parser",
                     .hint = "Use a valid LibTorch device specification"});
  }
}

}  // namespace jyppx::liuding
