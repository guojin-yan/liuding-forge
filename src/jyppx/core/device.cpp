#include <jyppx/core/device.hpp>

#include <torch/cuda.h>

#include <cstddef>
#include <stdexcept>

namespace jyppx {

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
    throw std::invalid_argument("Device string must not be empty");
  }
  if (device_str.size() > kMaxDeviceStringLength) {
    throw std::invalid_argument("Device string exceeds the 64-byte limit");
  }
  if (device_str.find('\0') != std::string::npos) {
    throw std::invalid_argument(
        "Device string must not contain an embedded NUL byte");
  }

  try {
    return Device(device_str);
  } catch (const std::exception& error) {
    throw std::invalid_argument("Invalid device string '" + device_str +
                                "': " + error.what());
  }
}

}  // namespace jyppx
