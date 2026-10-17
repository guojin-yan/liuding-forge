// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/sgd_adapter.hpp>

#include <torch/optim/sgd.h>
#include <torch/version.h>

#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <new>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace jyppx::optim {
namespace {

constexpr char kComponent[] = "optim.sgd_adapter";
constexpr std::uint32_t kSchemaVersion = 1;
constexpr std::uint32_t kTopologyVersion = 1;
constexpr std::uint32_t kControlVersion = 1;
constexpr std::size_t kMaximumRank = 64;
constexpr std::size_t kAdapterMaximumIdentifierBytes = 4096;
constexpr std::array<std::byte, 8> kTopologyMagic = {
    std::byte{'J'}, std::byte{'Y'}, std::byte{'P'}, std::byte{'P'},
    std::byte{'X'}, std::byte{'S'}, std::byte{'G'}, std::byte{'D'}};
constexpr std::array<std::byte, 8> kControlMagic = {
    std::byte{'J'}, std::byte{'Y'}, std::byte{'P'}, std::byte{'P'},
    std::byte{'X'}, std::byte{'S'}, std::byte{'G'}, std::byte{'C'}};

struct FlatParameter {
  std::size_t group_index;    ///< Owning stable group position.
  std::string parameter_id;   ///< Stable identity independent of pointers.
  Tensor parameter;           ///< Model-owned tensor handle used by LibTorch.
  std::uint8_t dtype_code;    ///< Versioned scalar-type encoding.
  std::uint8_t device_code;   ///< Versioned CPU or CUDA encoding.
  std::int32_t device_index;  ///< Concrete device ordinal, or -1 for CPU.
  std::vector<std::int64_t> sizes;    ///< Immutable construction-time shape.
  std::vector<std::int64_t> strides;  ///< Immutable construction-time strides.
  std::uint64_t byte_count;  ///< Logical dense bytes in one state tensor.
};

struct OptimizerSnapshot {
  std::vector<std::byte> topology;  ///< Immutable descriptor bytes.
  std::vector<std::byte> control;   ///< Mutable counters and presence bytes.
  std::vector<Tensor> momentum_buffers;  ///< Native buffers by stable order.
  std::vector<bool> has_momentum;  ///< Presence bit for every fixed field.
  StateManifest manifest;          ///< Complete storage-neutral state contract.
};

struct DecodedControl {
  std::uint64_t completed_steps;       ///< Restored successful-step count.
  std::vector<double> learning_rates;  ///< Mutable rates in group order.
  std::vector<bool> has_momentum;      ///< Presence bit in parameter order.
};

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid delegated SGD state",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Malformed delegated SGD checkpoint state",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_compatibility(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw CompatibilityError(
      "Delegated SGD checkpoint is incompatible with the current optimizer",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = std::move(expected),
                   .actual = std::move(actual),
                   .hint = std::move(hint)});
}

[[noreturn]] void throw_resource(std::string operation, std::uint64_t expected,
                                 std::uint64_t actual, std::string hint) {
  throw ResourceExhausted("Delegated SGD resource limit exceeded",
                          ErrorDetails{.component = kComponent,
                                       .operation = std::move(operation),
                                       .expected = std::to_string(expected),
                                       .actual = std::to_string(actual),
                                       .hint = std::move(hint)});
}

[[noreturn]] void translate_bad_alloc(std::string_view operation) {
  throw ResourceExhausted(
      "Unable to allocate delegated SGD state",
      ErrorDetails{.component = kComponent,
                   .operation = std::string(operation),
                   .expected = "memory within configured optimizer limits",
                   .actual = "allocation failed",
                   .hint = "Reduce optimizer state or the active workload."});
}

[[noreturn]] void translate_backend(const c10::Error& cause,
                                    std::string operation) {
  detail::rethrow_backend_error(
      cause, detail::BackendErrorContext{
                 .component = kComponent,
                 .operation = std::move(operation),
                 .failure_kind = detail::BackendFailureKind::unknown,
                 .expected = "a supported LibTorch SGD operation",
                 .hint = "Inspect the nested LibTorch diagnostic and device."});
}

void validate_identifier(std::string_view identifier, std::size_t maximum,
                         std::string_view operation) {
  if (identifier.empty() || identifier.size() > maximum ||
      identifier.find('\0') != std::string_view::npos) {
    throw_invalid(std::string(operation),
                  "a nonempty non-NUL stable ID within " +
                      std::to_string(maximum) + " bytes",
                  std::string(identifier),
                  "Use bounded logical IDs instead of paths or pointers.");
  }
}

void validate_sgd_options(const SgdAdapterOptions& options,
                          std::string_view group_id) {
  const auto require_non_negative = [&](double value,
                                        std::string_view option_name) {
    if (!std::isfinite(value) || value < 0.0) {
      throw_invalid(std::string(group_id) + "." + std::string(option_name),
                    "a finite non-negative value", std::to_string(value),
                    "Use an SGD option accepted by LibTorch.");
    }
  };
  require_non_negative(options.learning_rate, "learning_rate");
  require_non_negative(options.momentum, "momentum");
  require_non_negative(options.dampening, "dampening");
  require_non_negative(options.weight_decay, "weight_decay");
  if (options.nesterov &&
      (options.momentum <= 0.0 || options.dampening != 0.0)) {
    throw_invalid(std::string(group_id) + ".nesterov",
                  "positive momentum and zero dampening", "invalid options",
                  "Disable Nesterov or correct the momentum options.");
  }
}

std::uint8_t dtype_code(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Half:
      return 1;
    case c10::ScalarType::BFloat16:
      return 2;
    case c10::ScalarType::Float:
      return 3;
    case c10::ScalarType::Double:
      return 4;
    default:
      throw_invalid("parameter.dtype", "float16, bfloat16, float32, or float64",
                    c10::toString(dtype),
                    "Use a floating optimizer parameter dtype.");
  }
}

c10::ScalarType scalar_type(std::uint8_t code) {
  switch (code) {
    case 1:
      return c10::ScalarType::Half;
    case 2:
      return c10::ScalarType::BFloat16;
    case 3:
      return c10::ScalarType::Float;
    case 4:
      return c10::ScalarType::Double;
    default:
      throw_serialization("topology.dtype", "a supported floating dtype code",
                          std::to_string(code),
                          "Reject unknown optimizer tensor encodings.");
  }
}

std::uint64_t element_size(std::uint8_t code) {
  switch (code) {
    case 1:
    case 2:
      return 2;
    case 3:
      return 4;
    case 4:
      return 8;
    default:
      static_cast<void>(scalar_type(code));
      return 0;
  }
}

std::uint8_t device_code(const c10::Device& device) {
  if (device.is_cpu()) {
    return 1;
  }
  if (device.is_cuda()) {
    return 2;
  }
  throw_invalid("parameter.device", "CPU or CUDA", device.str(),
                "Move optimizer parameters to a supported device.");
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          std::uint64_t limit, std::string_view operation) {
  if (right > limit || left > limit - right) {
    throw_resource(std::string(operation), limit,
                   right > std::numeric_limits<std::uint64_t>::max() - left
                       ? std::numeric_limits<std::uint64_t>::max()
                       : left + right,
                   "Reduce optimizer state or raise the explicit limit.");
  }
  return left + right;
}

std::uint64_t checked_tensor_bytes(const Tensor& tensor, std::uint8_t code) {
  const auto count = tensor.numel();
  if (count < 0) {
    throw_invalid("parameter.numel", "a non-negative element count",
                  std::to_string(count), "Materialize the parameter shape.");
  }
  const auto unsigned_count = static_cast<std::uint64_t>(count);
  const auto width = element_size(code);
  if (unsigned_count > std::numeric_limits<std::uint64_t>::max() / width) {
    throw_resource("parameter.byte_count",
                   std::numeric_limits<std::uint64_t>::max(), unsigned_count,
                   "Reduce parameter dimensions.");
  }
  return unsigned_count * width;
}

torch::optim::SGDOptions make_native_options(const SgdAdapterOptions& options) {
  torch::optim::SGDOptions native(options.learning_rate);
  native.momentum(options.momentum)
      .dampening(options.dampening)
      .weight_decay(options.weight_decay)
      .nesterov(options.nesterov);
  return native;
}

const torch::optim::SGDOptions& native_options(
    const torch::optim::OptimizerParamGroup& group) {
  const auto* options = dynamic_cast<const torch::optim::SGDOptions*>(
      std::addressof(group.options()));
  if (options == nullptr) {
    throw_invalid("native.options", "torch::optim::SGDOptions",
                  "unexpected optimizer option type",
                  "Do not mutate the delegated native optimizer.");
  }
  return *options;
}

torch::optim::SGDOptions& native_options(
    torch::optim::OptimizerParamGroup& group) {
  auto* options =
      dynamic_cast<torch::optim::SGDOptions*>(std::addressof(group.options()));
  if (options == nullptr) {
    throw_invalid("native.options", "torch::optim::SGDOptions",
                  "unexpected optimizer option type",
                  "Do not mutate the delegated native optimizer.");
  }
  return *options;
}

std::string momentum_field_name(std::size_t index) {
  std::ostringstream output;
  output << "momentum." << std::setw(10) << std::setfill('0') << index;
  return output.str();
}

void append_u8(std::vector<std::byte>& output, std::uint8_t value) {
  output.push_back(static_cast<std::byte>(value));
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
  for (std::size_t shift = 0; shift < 32; shift += 8) {
    append_u8(output, static_cast<std::uint8_t>(value >> shift));
  }
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
  for (std::size_t shift = 0; shift < 64; shift += 8) {
    append_u8(output, static_cast<std::uint8_t>(value >> shift));
  }
}

void append_i32(std::vector<std::byte>& output, std::int32_t value) {
  append_u32(output, std::bit_cast<std::uint32_t>(value));
}

void append_i64(std::vector<std::byte>& output, std::int64_t value) {
  append_u64(output, std::bit_cast<std::uint64_t>(value));
}

void append_double(std::vector<std::byte>& output, double value) {
  append_u64(output, std::bit_cast<std::uint64_t>(value));
}

void append_string(std::vector<std::byte>& output, std::string_view value) {
  append_u32(output, static_cast<std::uint32_t>(value.size()));
  output.insert(
      output.end(), reinterpret_cast<const std::byte*>(value.data()),
      reinterpret_cast<const std::byte*>(value.data()) + value.size());
}

class ByteCursor final {
 public:
  explicit ByteCursor(std::span<const std::byte> bytes) : bytes_(bytes) {}

  std::uint8_t read_u8(std::string_view field) {
    require(1, field);
    return std::to_integer<std::uint8_t>(bytes_[position_++]);
  }

  std::uint32_t read_u32(std::string_view field) {
    std::uint32_t value = 0;
    for (std::size_t shift = 0; shift < 32; shift += 8) {
      value |= static_cast<std::uint32_t>(read_u8(field)) << shift;
    }
    return value;
  }

  std::uint64_t read_u64(std::string_view field) {
    std::uint64_t value = 0;
    for (std::size_t shift = 0; shift < 64; shift += 8) {
      value |= static_cast<std::uint64_t>(read_u8(field)) << shift;
    }
    return value;
  }

  std::int32_t read_i32(std::string_view field) {
    return std::bit_cast<std::int32_t>(read_u32(field));
  }

  std::int64_t read_i64(std::string_view field) {
    return std::bit_cast<std::int64_t>(read_u64(field));
  }

  double read_double(std::string_view field) {
    return std::bit_cast<double>(read_u64(field));
  }

  std::string read_string(std::size_t maximum, std::string_view field) {
    const auto length = read_u32(field);
    if (length == 0 || length > maximum) {
      throw_resource(std::string(field), maximum, length,
                     "Reject empty or oversized checkpoint identifiers.");
    }
    require(length, field);
    const auto* begin =
        reinterpret_cast<const char*>(bytes_.data() + position_);
    position_ += length;
    std::string value(begin, length);
    if (value.find('\0') != std::string::npos) {
      throw_serialization(std::string(field), "a non-NUL identifier", value,
                          "Reject ambiguous checkpoint identifiers.");
    }
    return value;
  }

  bool empty() const noexcept { return position_ == bytes_.size(); }

 private:
  void require(std::size_t count, std::string_view field) const {
    if (position_ > bytes_.size() || count > bytes_.size() - position_) {
      throw_serialization(std::string(field), "complete encoded bytes",
                          "truncated",
                          "Reject incomplete state before optimizer mutation.");
    }
  }

  std::span<const std::byte> bytes_;
  std::size_t position_ = 0;
};

std::vector<std::byte> encode_topology(
    const std::vector<SgdParameterGroup>& groups,
    const std::vector<FlatParameter>& parameters,
    const SgdOptimizerCheckpointOptions& limits) {
  std::vector<std::byte> output(kTopologyMagic.begin(), kTopologyMagic.end());
  append_u32(output, kTopologyVersion);
  append_u32(output, TORCH_VERSION_MAJOR);
  append_u32(output, TORCH_VERSION_MINOR);
  append_u32(output, TORCH_VERSION_PATCH);
  append_u32(output, static_cast<std::uint32_t>(groups.size()));
  append_u32(output, static_cast<std::uint32_t>(parameters.size()));

  std::size_t flat_index = 0;
  for (const auto& group : groups) {
    append_string(output, group.group_id);
    append_u32(output, static_cast<std::uint32_t>(group.parameters.size()));
    append_double(output, group.options.momentum);
    append_double(output, group.options.dampening);
    append_double(output, group.options.weight_decay);
    append_u8(output, group.options.nesterov ? 1 : 0);
    for (std::size_t reserved = 0; reserved < 7; ++reserved) {
      append_u8(output, 0);
    }

    for (std::size_t index = 0; index < group.parameters.size(); ++index) {
      const auto& parameter = parameters.at(flat_index++);
      append_string(output, parameter.parameter_id);
      append_u8(output, parameter.dtype_code);
      append_u8(output, parameter.device_code);
      append_u8(output, 1);
      append_u8(output, 0);
      append_i32(output, parameter.device_index);
      append_u32(output, static_cast<std::uint32_t>(parameter.sizes.size()));
      append_u64(output, parameter.byte_count);
      for (const auto size : parameter.sizes) {
        append_i64(output, size);
      }
      for (const auto stride : parameter.strides) {
        append_i64(output, stride);
      }
      if (output.size() > limits.max_topology_bytes) {
        throw_resource("topology.bytes", limits.max_topology_bytes,
                       output.size(),
                       "Reduce optimizer topology or raise its limit.");
      }
    }
  }
  return output;
}

std::vector<std::byte> encode_control(
    std::uint64_t completed_steps, const std::vector<double>& learning_rates,
    const std::vector<bool>& has_momentum,
    const SgdOptimizerCheckpointOptions& limits) {
  std::vector<std::byte> output(kControlMagic.begin(), kControlMagic.end());
  append_u32(output, kControlVersion);
  append_u64(output, completed_steps);
  append_u32(output, static_cast<std::uint32_t>(learning_rates.size()));
  append_u32(output, static_cast<std::uint32_t>(has_momentum.size()));
  for (const auto rate : learning_rates) {
    append_double(output, rate);
  }
  for (const auto present : has_momentum) {
    append_u8(output, present ? 1 : 0);
  }
  if (output.size() > limits.max_control_bytes) {
    throw_resource("control.bytes", limits.max_control_bytes, output.size(),
                   "Reduce optimizer groups or raise max_control_bytes.");
  }
  return output;
}

void validate_magic(ByteCursor& cursor,
                    const std::array<std::byte, 8>& expected,
                    std::string_view field, std::string_view rendered) {
  for (const auto byte : expected) {
    if (cursor.read_u8(field) != std::to_integer<std::uint8_t>(byte)) {
      throw_serialization(std::string(field), std::string(rendered), "mismatch",
                          "Use state produced by this optimizer schema.");
    }
  }
}

void validate_topology_blob(std::span<const std::byte> bytes,
                            const SgdOptimizerCheckpointOptions& limits) {
  if (bytes.size() > limits.max_topology_bytes) {
    throw_resource("topology.bytes", limits.max_topology_bytes, bytes.size(),
                   "Reject oversized topology before parsing.");
  }
  ByteCursor cursor(bytes);
  validate_magic(cursor, kTopologyMagic, "topology.magic", "JYPPXSGD");
  const auto version = cursor.read_u32("topology.version");
  if (version != kTopologyVersion) {
    throw_compatibility("topology.version", std::to_string(kTopologyVersion),
                        std::to_string(version),
                        "Migrate optimizer topology metadata explicitly.");
  }
  const auto major = cursor.read_u32("topology.torch_major");
  const auto minor = cursor.read_u32("topology.torch_minor");
  const auto patch = cursor.read_u32("topology.torch_patch");
  if (major != TORCH_VERSION_MAJOR || minor != TORCH_VERSION_MINOR ||
      patch != TORCH_VERSION_PATCH) {
    throw_compatibility(
        "topology.libtorch_version",
        std::to_string(TORCH_VERSION_MAJOR) + "." +
            std::to_string(TORCH_VERSION_MINOR) + "." +
            std::to_string(TORCH_VERSION_PATCH),
        std::to_string(major) + "." + std::to_string(minor) + "." +
            std::to_string(patch),
        "Use the declared exact LibTorch compatibility matrix.");
  }
  const auto group_count = cursor.read_u32("topology.group_count");
  const auto parameter_count = cursor.read_u32("topology.parameter_count");
  if (group_count == 0 || group_count > limits.max_groups) {
    throw_resource("topology.group_count", limits.max_groups, group_count,
                   "Reject empty or excessive optimizer groups.");
  }
  if (parameter_count == 0 || parameter_count > limits.max_parameters) {
    throw_resource("topology.parameter_count", limits.max_parameters,
                   parameter_count,
                   "Reject empty or excessive optimizer parameters.");
  }

  std::set<std::string> group_ids;
  std::set<std::string> parameter_ids;
  std::uint64_t total = bytes.size();
  std::uint64_t decoded_parameters = 0;
  for (std::uint32_t group_index = 0; group_index < group_count;
       ++group_index) {
    const auto group_id =
        cursor.read_string(limits.max_identifier_bytes, "topology.group_id");
    if (!group_ids.insert(group_id).second) {
      throw_serialization("topology.group_id", "unique group IDs", group_id,
                          "Reject ambiguous optimizer groups.");
    }
    const auto count = cursor.read_u32("topology.group_parameter_count");
    if (count == 0 || count > parameter_count - decoded_parameters) {
      throw_serialization("topology.group_parameter_count",
                          "a nonempty count within total parameters",
                          std::to_string(count),
                          "Reject inconsistent optimizer topology.");
    }
    decoded_parameters += count;
    SgdAdapterOptions options;
    options.momentum = cursor.read_double("topology.momentum");
    options.dampening = cursor.read_double("topology.dampening");
    options.weight_decay = cursor.read_double("topology.weight_decay");
    const auto nesterov = cursor.read_u8("topology.nesterov");
    if (nesterov > 1) {
      throw_serialization("topology.nesterov", "a boolean",
                          std::to_string(nesterov),
                          "Reject invalid optimizer options.");
    }
    options.nesterov = nesterov == 1;
    for (std::size_t reserved = 0; reserved < 7; ++reserved) {
      if (cursor.read_u8("topology.reserved") != 0) {
        throw_serialization("topology.reserved", "zero", "non-zero",
                            "Reject unknown topology extensions.");
      }
    }
    if (!std::isfinite(options.momentum) || options.momentum < 0.0 ||
        !std::isfinite(options.dampening) || options.dampening < 0.0 ||
        !std::isfinite(options.weight_decay) || options.weight_decay < 0.0 ||
        (options.nesterov &&
         (options.momentum <= 0.0 || options.dampening != 0.0))) {
      throw_serialization("topology.options", "valid finite SGD options",
                          group_id,
                          "Reject malformed immutable optimizer options.");
    }

    for (std::uint32_t index = 0; index < count; ++index) {
      const auto parameter_id = cursor.read_string(limits.max_identifier_bytes,
                                                   "topology.parameter_id");
      if (!parameter_ids.insert(parameter_id).second) {
        throw_serialization("topology.parameter_id", "unique parameter IDs",
                            parameter_id,
                            "Reject ambiguous optimizer parameters.");
      }
      const auto code = cursor.read_u8("topology.dtype");
      static_cast<void>(scalar_type(code));
      const auto device = cursor.read_u8("topology.device");
      if (device != 1 && device != 2) {
        throw_serialization("topology.device", "CPU or CUDA device code",
                            std::to_string(device),
                            "Reject unknown optimizer devices.");
      }
      if (cursor.read_u8("topology.requires_grad") != 1 ||
          cursor.read_u8("topology.flags") != 0) {
        throw_serialization("topology.parameter_flags",
                            "requires_grad and no unknown flags", "invalid",
                            "Reject incompatible parameter semantics.");
      }
      const auto device_index = cursor.read_i32("topology.device_index");
      if ((device == 1 && device_index != -1) ||
          (device == 2 && device_index < 0)) {
        throw_serialization("topology.device_index",
                            "CPU -1 or non-negative CUDA index",
                            std::to_string(device_index),
                            "Reject inconsistent device metadata.");
      }
      const auto rank = cursor.read_u32("topology.rank");
      const auto byte_count = cursor.read_u64("topology.byte_count");
      if (rank > kMaximumRank) {
        throw_resource("topology.rank", kMaximumRank, rank,
                       "Reject excessive tensor ranks before allocation.");
      }
      if (byte_count > limits.max_tensor_bytes) {
        throw_resource("topology.byte_count", limits.max_tensor_bytes,
                       byte_count,
                       "Reject oversized momentum state before allocation.");
      }
      std::uint64_t numel = 1;
      bool zero_dimension = false;
      for (std::uint32_t dimension = 0; dimension < rank; ++dimension) {
        const auto size = cursor.read_i64("topology.size");
        if (size < 0) {
          throw_serialization("topology.size", "non-negative dimensions",
                              std::to_string(size),
                              "Reject symbolic or malformed shapes.");
        }
        if (size == 0) {
          zero_dimension = true;
        } else if (!zero_dimension) {
          const auto value = static_cast<std::uint64_t>(size);
          if (numel > std::numeric_limits<std::uint64_t>::max() / value) {
            throw_resource("topology.numel",
                           std::numeric_limits<std::uint64_t>::max(), value,
                           "Reject shapes whose element count overflows.");
          }
          numel *= value;
        }
      }
      for (std::uint32_t dimension = 0; dimension < rank; ++dimension) {
        const auto stride = cursor.read_i64("topology.stride");
        if (stride < 0) {
          throw_serialization("topology.stride", "non-negative strides",
                              std::to_string(stride),
                              "Reject unsupported negative strides.");
        }
      }
      if (zero_dimension) {
        numel = 0;
      }
      const auto width = element_size(code);
      if (numel > std::numeric_limits<std::uint64_t>::max() / width ||
          numel * width != byte_count) {
        throw_serialization(
            "topology.byte_count", "shape multiplied by scalar width",
            std::to_string(byte_count), "Reject inconsistent tensor metadata.");
      }
      total = checked_add(total, std::max<std::uint64_t>(1, byte_count),
                          limits.max_total_bytes, "state.total_bytes");
    }
  }
  if (decoded_parameters != parameter_count || !cursor.empty()) {
    throw_serialization("topology.trailing_bytes",
                        "exact parameter count and no trailing metadata",
                        "mismatch", "Reject inconsistent topology extensions.");
  }
}

DecodedControl decode_control(std::span<const std::byte> bytes,
                              std::size_t expected_groups,
                              std::size_t expected_parameters,
                              const SgdOptimizerCheckpointOptions& limits) {
  if (bytes.size() > limits.max_control_bytes) {
    throw_resource("control.bytes", limits.max_control_bytes, bytes.size(),
                   "Reject oversized control metadata before parsing.");
  }
  ByteCursor cursor(bytes);
  validate_magic(cursor, kControlMagic, "control.magic", "JYPPXSGC");
  const auto version = cursor.read_u32("control.version");
  if (version != kControlVersion) {
    throw_compatibility("control.version", std::to_string(kControlVersion),
                        std::to_string(version),
                        "Migrate optimizer control metadata explicitly.");
  }
  DecodedControl decoded;
  decoded.completed_steps = cursor.read_u64("control.completed_steps");
  const auto group_count = cursor.read_u32("control.group_count");
  const auto parameter_count = cursor.read_u32("control.parameter_count");
  if (group_count != expected_groups ||
      parameter_count != expected_parameters) {
    throw_serialization(
        "control.counts",
        std::to_string(expected_groups) + ":" +
            std::to_string(expected_parameters),
        std::to_string(group_count) + ":" + std::to_string(parameter_count),
        "Reject control state from a different optimizer topology.");
  }
  decoded.learning_rates.reserve(group_count);
  for (std::uint32_t index = 0; index < group_count; ++index) {
    const auto rate = cursor.read_double("control.learning_rate");
    if (!std::isfinite(rate) || rate < 0.0) {
      throw_serialization("control.learning_rate",
                          "a finite non-negative value", std::to_string(rate),
                          "Reject invalid mutable optimizer options.");
    }
    decoded.learning_rates.push_back(rate);
  }
  decoded.has_momentum.reserve(parameter_count);
  for (std::uint32_t index = 0; index < parameter_count; ++index) {
    const auto present = cursor.read_u8("control.momentum_presence");
    if (present > 1) {
      throw_serialization("control.momentum_presence", "a boolean",
                          std::to_string(present),
                          "Reject invalid momentum presence metadata.");
    }
    decoded.has_momentum.push_back(present == 1);
  }
  if (!cursor.empty()) {
    throw_serialization("control.trailing_bytes", "no trailing metadata",
                        "present", "Reject ambiguous control extensions.");
  }
  return decoded;
}

bool state_field_equal(const StateFieldDescriptor& left,
                       const StateFieldDescriptor& right) {
  return left.name == right.name && left.media_type == right.media_type &&
         left.max_bytes == right.max_bytes && left.required == right.required;
}

bool manifest_equal(const StateManifest& left, const StateManifest& right) {
  return left.participant_id == right.participant_id &&
         left.schema_version == right.schema_version &&
         left.restore_level == right.restore_level &&
         left.degradation_reasons == right.degradation_reasons &&
         left.fields.size() == right.fields.size() &&
         std::equal(left.fields.begin(), left.fields.end(),
                    right.fields.begin(), state_field_equal);
}

void validate_checkpoint_options(const SgdOptimizerCheckpointOptions& options,
                                 std::string_view participant_id) {
  if (options.max_groups == 0 || options.max_parameters == 0 ||
      options.max_identifier_bytes == 0 || options.max_topology_bytes == 0 ||
      options.max_control_bytes == 0 || options.max_tensor_bytes == 0 ||
      options.max_total_bytes == 0) {
    throw_invalid("checkpoint.options", "positive resource limits", "zero",
                  "Set every optimizer checkpoint limit above zero.");
  }
  if (options.max_groups > std::numeric_limits<std::uint32_t>::max() ||
      options.max_parameters > std::numeric_limits<std::uint32_t>::max() ||
      options.max_identifier_bytes >
          std::numeric_limits<std::uint32_t>::max()) {
    throw_invalid("checkpoint.options", "limits within uint32 range",
                  "excessive",
                  "Use bounded optimizer topology and identifier limits.");
  }
  StateManifest probe{
      .participant_id = std::string(participant_id),
      .schema_version = kSchemaVersion,
      .restore_level = RestoreLevel::exact,
      .fields = {{.name = "topology", .max_bytes = 1}},
      .degradation_reasons = {},
  };
  validate_state_manifest(probe);
}

void validate_current_topology(const std::vector<SgdParameterGroup>& groups,
                               const std::vector<FlatParameter>& parameters,
                               const SgdOptimizerCheckpointOptions& limits) {
  if (groups.size() > limits.max_groups) {
    throw_resource("state.group_count", limits.max_groups, groups.size(),
                   "Reduce optimizer groups or raise max_groups.");
  }
  if (parameters.size() > limits.max_parameters) {
    throw_resource("state.parameter_count", limits.max_parameters,
                   parameters.size(),
                   "Reduce optimizer parameters or raise max_parameters.");
  }
  for (const auto& group : groups) {
    validate_identifier(group.group_id, limits.max_identifier_bytes,
                        "state.group_id");
  }
  for (const auto& parameter : parameters) {
    validate_identifier(parameter.parameter_id, limits.max_identifier_bytes,
                        "state.parameter_id");
    if (parameter.byte_count > limits.max_tensor_bytes) {
      throw_resource("state.tensor_bytes", limits.max_tensor_bytes,
                     parameter.byte_count,
                     "Reduce a parameter or raise max_tensor_bytes.");
    }
  }
}

std::vector<double> inspect_native_groups(
    const std::vector<SgdParameterGroup>& groups,
    const std::vector<FlatParameter>& parameters,
    const torch::optim::SGD& native) {
  const auto& native_groups = native.param_groups();
  if (native_groups.size() != groups.size()) {
    throw_invalid("native.group_count", std::to_string(groups.size()),
                  std::to_string(native_groups.size()),
                  "Do not mutate delegated native parameter groups.");
  }
  std::vector<double> learning_rates;
  learning_rates.reserve(groups.size());
  std::size_t flat_index = 0;
  for (std::size_t group_index = 0; group_index < groups.size();
       ++group_index) {
    const auto& expected = groups[group_index];
    const auto& actual = native_groups[group_index];
    if (actual.params().size() != expected.parameters.size()) {
      throw_invalid("native.parameter_count",
                    std::to_string(expected.parameters.size()),
                    std::to_string(actual.params().size()),
                    "Do not mutate delegated native parameter groups.");
    }
    const auto& options = native_options(actual);
    if (!std::isfinite(options.lr()) || options.lr() < 0.0 ||
        options.momentum() != expected.options.momentum ||
        options.dampening() != expected.options.dampening ||
        options.weight_decay() != expected.options.weight_decay ||
        options.nesterov() != expected.options.nesterov) {
      throw_invalid("native.options", "the stable SGD descriptor",
                    expected.group_id,
                    "Use set_learning_rates for the only mutable option.");
    }
    learning_rates.push_back(options.lr());
    for (std::size_t parameter_index = 0;
         parameter_index < actual.params().size(); ++parameter_index) {
      if (actual.params()[parameter_index].unsafeGetTensorImpl() !=
          parameters.at(flat_index++).parameter.unsafeGetTensorImpl()) {
        throw_invalid("native.parameter_order", "the stable binding order",
                      expected.parameters[parameter_index].parameter_id,
                      "Do not replace delegated native parameter handles.");
      }
      const auto& parameter = parameters[flat_index - 1];
      const auto& tensor = actual.params()[parameter_index];
      if (!tensor.defined() || !tensor.requires_grad() || tensor.is_meta() ||
          tensor.is_quantized() || tensor.layout() != c10::Layout::Strided ||
          !tensor.is_non_overlapping_and_dense() ||
          dtype_code(tensor.scalar_type()) != parameter.dtype_code ||
          device_code(tensor.device()) != parameter.device_code ||
          static_cast<std::int32_t>(tensor.device().index()) !=
              parameter.device_index ||
          tensor.sizes().vec() != parameter.sizes ||
          tensor.strides().vec() != parameter.strides ||
          checked_tensor_bytes(tensor, parameter.dtype_code) !=
              parameter.byte_count) {
        throw_invalid(
            "native.parameter_descriptor",
            "the construction-time dtype, device, shape, stride, and role",
            parameter.parameter_id,
            "Rebuild the optimizer after changing model parameter topology.");
      }
    }
  }
  return learning_rates;
}

void validate_step_gradients(const std::vector<FlatParameter>& parameters) {
  for (const auto& parameter : parameters) {
    const auto& gradient = parameter.parameter.grad();
    if (!gradient.defined()) {
      continue;
    }
    if (gradient.layout() != c10::Layout::Strided) {
      throw NotImplemented(
          "Delegated SGD does not support sparse gradients",
          ErrorDetails{
              .component = kComponent,
              .operation = "step.gradient_layout",
              .expected = "a dense strided gradient",
              .actual = parameter.parameter_id,
              .hint = "Use a dense gradient or a sparse-aware optimizer."});
    }
    if (gradient.is_meta() || gradient.is_quantized() ||
        gradient.scalar_type() != parameter.parameter.scalar_type() ||
        gradient.device() != parameter.parameter.device() ||
        gradient.sizes().vec() != parameter.sizes) {
      throw_invalid(
          "step.gradient_descriptor",
          "a dense gradient matching the parameter dtype, device, and shape",
          parameter.parameter_id,
          "Recompute gradients for the unchanged optimizer topology.");
    }
  }
}

OptimizerSnapshot inspect_optimizer(
    const std::vector<SgdParameterGroup>& groups,
    const std::vector<FlatParameter>& parameters,
    const torch::optim::SGD& native, std::uint64_t completed_steps,
    const SgdOptimizerCheckpointOptions& limits, std::string participant_id) {
  validate_current_topology(groups, parameters, limits);
  auto learning_rates = inspect_native_groups(groups, parameters, native);

  std::unordered_set<const void*> known_parameters;
  known_parameters.reserve(parameters.size());
  for (const auto& parameter : parameters) {
    known_parameters.insert(parameter.parameter.unsafeGetTensorImpl());
  }
  for (const auto& [key, state] : native.state()) {
    if (!known_parameters.contains(key) || state == nullptr) {
      throw_invalid("native.state", "state for a bound parameter", "unknown",
                    "Do not inject pointer-keyed native optimizer state.");
    }
  }

  std::vector<Tensor> momentum_buffers;
  std::vector<bool> has_momentum;
  momentum_buffers.reserve(parameters.size());
  has_momentum.reserve(parameters.size());
  for (const auto& parameter : parameters) {
    const auto state =
        native.state().find(parameter.parameter.unsafeGetTensorImpl());
    if (state == native.state().end()) {
      momentum_buffers.emplace_back();
      has_momentum.push_back(false);
      continue;
    }
    const auto* sgd_state =
        dynamic_cast<const torch::optim::SGDParamState*>(state->second.get());
    if (sgd_state == nullptr) {
      throw_invalid("native.state_type", "torch::optim::SGDParamState",
                    parameter.parameter_id,
                    "Do not inject state from another optimizer algorithm.");
    }
    const auto& buffer = sgd_state->momentum_buffer();
    if (!buffer.defined()) {
      momentum_buffers.emplace_back();
      has_momentum.push_back(false);
      continue;
    }
    const auto& descriptor = groups[parameter.group_index].options;
    if (descriptor.momentum <= 0.0 ||
        buffer.scalar_type() != parameter.parameter.scalar_type() ||
        buffer.device() != parameter.parameter.device() ||
        buffer.sizes().vec() != parameter.sizes ||
        buffer.strides().vec() != parameter.strides ||
        buffer.layout() != c10::Layout::Strided ||
        !buffer.is_non_overlapping_and_dense() ||
        checked_tensor_bytes(buffer, parameter.dtype_code) !=
            parameter.byte_count) {
      throw_invalid("native.momentum_buffer",
                    "dense state matching its stable parameter descriptor",
                    parameter.parameter_id,
                    "Recover from a valid optimizer checkpoint.");
    }
    momentum_buffers.push_back(buffer);
    has_momentum.push_back(true);
  }

  auto topology = encode_topology(groups, parameters, limits);
  auto control =
      encode_control(completed_steps, learning_rates, has_momentum, limits);
  std::uint64_t total =
      checked_add(topology.size(), control.size(), limits.max_total_bytes,
                  "state.total_bytes");
  StateManifest manifest{
      .participant_id = std::move(participant_id),
      .schema_version = kSchemaVersion,
      .restore_level = RestoreLevel::exact,
      .fields = {{.name = "topology",
                  .media_type = "application/vnd.jyppx.sgd-topology-v1",
                  .max_bytes = topology.size()},
                 {.name = "control",
                  .media_type = "application/vnd.jyppx.sgd-control-v1",
                  .max_bytes = control.size()}},
      .degradation_reasons = {},
  };
  manifest.fields.reserve(parameters.size() + 2);
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    const auto maximum =
        std::max<std::uint64_t>(1, parameters[index].byte_count);
    total = checked_add(total, maximum, limits.max_total_bytes,
                        "state.total_bytes");
    manifest.fields.push_back(
        {.name = momentum_field_name(index),
         .media_type = "application/vnd.jyppx.tensor-raw-v1",
         .max_bytes = maximum});
  }
  static_cast<void>(total);
  validate_state_manifest(manifest);
  return {.topology = std::move(topology),
          .control = std::move(control),
          .momentum_buffers = std::move(momentum_buffers),
          .has_momentum = std::move(has_momentum),
          .manifest = std::move(manifest)};
}

}  // namespace

struct SgdOptimizerAdapter::Impl {
  std::string optimizer_id;                      ///< Stable optimizer identity.
  std::vector<std::string> parameter_group_ids;  ///< Stable group identities.
  std::vector<SgdParameterGroup> groups;         ///< Stable public descriptor.
  std::vector<FlatParameter> parameters;      ///< Flattened checkpoint order.
  std::unique_ptr<torch::optim::SGD> native;  ///< Delegated algorithm owner.
  std::uint64_t completed_steps = 0;          ///< Successful native step count.
};

SgdOptimizerAdapter::SgdOptimizerAdapter(std::string optimizer_id,
                                         std::vector<SgdParameterGroup> groups)
    : impl_(std::make_unique<Impl>()) {
  try {
    validate_identifier(optimizer_id, kAdapterMaximumIdentifierBytes,
                        "optimizer_id");
    if (groups.empty()) {
      throw_invalid("groups", "at least one parameter group", "empty",
                    "Define stable optimizer groups before construction.");
    }
    if (groups.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw_invalid("groups", "at most uint32 maximum groups",
                    std::to_string(groups.size()),
                    "Use a bounded optimizer topology.");
    }

    std::set<std::string> group_ids;
    std::set<std::string> parameter_ids;
    std::unordered_set<const void*> tensor_identities;
    std::vector<torch::optim::OptimizerParamGroup> native_groups;
    impl_->parameter_group_ids.reserve(groups.size());
    native_groups.reserve(groups.size());
    for (std::size_t group_index = 0; group_index < groups.size();
         ++group_index) {
      auto& group = groups[group_index];
      validate_identifier(group.group_id, kAdapterMaximumIdentifierBytes,
                          "group_id");
      if (!group_ids.insert(group.group_id).second) {
        throw_invalid("group_id", "unique stable group IDs", group.group_id,
                      "Rename the duplicate optimizer group.");
      }
      impl_->parameter_group_ids.push_back(group.group_id);
      validate_sgd_options(group.options, group.group_id);
      if (group.parameters.empty()) {
        throw_invalid(group.group_id + ".parameters", "at least one parameter",
                      "empty", "Remove the group or bind model parameters.");
      }
      if (group.parameters.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw_invalid(group.group_id + ".parameters",
                      "at most uint32 maximum parameters",
                      std::to_string(group.parameters.size()),
                      "Use a bounded parameter group.");
      }

      std::vector<Tensor> native_parameters;
      native_parameters.reserve(group.parameters.size());
      for (auto& binding : group.parameters) {
        validate_identifier(binding.parameter_id,
                            kAdapterMaximumIdentifierBytes, "parameter_id");
        if (!parameter_ids.insert(binding.parameter_id).second) {
          throw_invalid("parameter_id", "globally unique stable parameter IDs",
                        binding.parameter_id,
                        "Use the model's fully qualified parameter name.");
        }
        auto& parameter = binding.parameter;
        if (!parameter.defined() || parameter.is_meta() ||
            parameter.is_quantized() ||
            parameter.layout() != c10::Layout::Strided ||
            !parameter.is_non_overlapping_and_dense() ||
            !parameter.requires_grad()) {
          throw_invalid(
              "parameter", "defined dense floating trainable parameter",
              binding.parameter_id,
              "Materialize and enable gradients before optimizer creation.");
        }
        if (parameter.dim() > static_cast<std::int64_t>(kMaximumRank)) {
          throw_invalid("parameter.rank", "at most 64 dimensions",
                        std::to_string(parameter.dim()),
                        "Reduce parameter rank before optimizer creation.");
        }
        if (!tensor_identities.insert(parameter.unsafeGetTensorImpl()).second) {
          throw_invalid("parameter.identity", "one group per tensor identity",
                        binding.parameter_id,
                        "Remove duplicate parameter bindings.");
        }
        const auto code = dtype_code(parameter.scalar_type());
        const auto device = device_code(parameter.device());
        impl_->parameters.push_back(
            {.group_index = group_index,
             .parameter_id = binding.parameter_id,
             .parameter = parameter,
             .dtype_code = code,
             .device_code = device,
             .device_index =
                 static_cast<std::int32_t>(parameter.device().index()),
             .sizes = parameter.sizes().vec(),
             .strides = parameter.strides().vec(),
             .byte_count = checked_tensor_bytes(parameter, code)});
        native_parameters.push_back(parameter);
      }
      native_groups.emplace_back(std::move(native_parameters));
    }
    if (impl_->parameters.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw_invalid("parameters", "at most uint32 maximum parameters",
                    std::to_string(impl_->parameters.size()),
                    "Use a bounded optimizer topology.");
    }

    auto defaults = make_native_options(groups.front().options);
    impl_->native =
        std::make_unique<torch::optim::SGD>(native_groups, defaults);
    for (std::size_t index = 0; index < groups.size(); ++index) {
      impl_->native->param_groups()[index].set_options(
          std::make_unique<torch::optim::SGDOptions>(
              make_native_options(groups[index].options)));
    }
    impl_->optimizer_id = std::move(optimizer_id);
    impl_->groups = std::move(groups);
    static_cast<void>(inspect_native_groups(impl_->groups, impl_->parameters,
                                            *impl_->native));
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("construct");
  } catch (const c10::Error& cause) {
    translate_backend(cause, "construct delegated SGD");
  }
}

SgdOptimizerAdapter::~SgdOptimizerAdapter() = default;

Tensor SgdOptimizerAdapter::step(LossClosure closure) {
  try {
    if (impl_->completed_steps == std::numeric_limits<std::uint64_t>::max()) {
      throw ResourceExhausted(
          "Delegated SGD step counter exhausted",
          ErrorDetails{.component = kComponent,
                       .operation = "step.completed_steps",
                       .expected = "a count below uint64 maximum",
                       .actual = "uint64 maximum",
                       .hint = "Start a new training run identity."});
    }
    Tensor result;
    if (closure) {
      auto validating_closure =
          [this, closure = std::move(closure)]() mutable -> Tensor {
        auto loss = closure();
        validate_step_gradients(impl_->parameters);
        return loss;
      };
      result = impl_->native->step(std::move(validating_closure));
    } else {
      validate_step_gradients(impl_->parameters);
      result = impl_->native->step();
    }
    ++impl_->completed_steps;
    return result;
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("step");
  } catch (const c10::Error& cause) {
    translate_backend(cause, "delegated SGD step");
  }
}

void SgdOptimizerAdapter::zero_grad(bool set_to_none) {
  try {
    impl_->native->zero_grad(set_to_none);
  } catch (const Error&) {
    throw;
  } catch (const c10::Error& cause) {
    translate_backend(cause, "zero delegated SGD gradients");
  }
}

std::vector<double> SgdOptimizerAdapter::learning_rates() const {
  try {
    return inspect_native_groups(impl_->groups, impl_->parameters,
                                 *impl_->native);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("learning_rates");
  }
}

void SgdOptimizerAdapter::set_learning_rates(
    const std::vector<double>& learning_rates) {
  if (learning_rates.size() != impl_->groups.size()) {
    throw_invalid("learning_rates.count", std::to_string(impl_->groups.size()),
                  std::to_string(learning_rates.size()),
                  "Provide one learning rate per stable parameter group.");
  }
  for (const auto rate : learning_rates) {
    if (!std::isfinite(rate) || rate < 0.0) {
      throw_invalid("learning_rate", "a finite non-negative value",
                    std::to_string(rate),
                    "Validate every rate before updating the optimizer.");
    }
  }
  auto& native_groups = impl_->native->param_groups();
  for (std::size_t index = 0; index < learning_rates.size(); ++index) {
    native_options(native_groups[index]).lr(learning_rates[index]);
  }
}

const std::string& SgdOptimizerAdapter::optimizer_id() const noexcept {
  return impl_->optimizer_id;
}

const std::vector<std::string>& SgdOptimizerAdapter::parameter_group_ids()
    const noexcept {
  return impl_->parameter_group_ids;
}

std::uint64_t SgdOptimizerAdapter::completed_steps() const noexcept {
  return impl_->completed_steps;
}

const std::vector<SgdParameterGroup>& SgdOptimizerAdapter::groups()
    const noexcept {
  return impl_->groups;
}

std::vector<OptimizerParameterBinding> SgdOptimizerAdapter::parameter_bindings()
    const {
  std::vector<OptimizerParameterBinding> result;
  result.reserve(impl_->parameters.size());
  for (const auto& parameter : impl_->parameters) {
    result.push_back({.parameter_id = parameter.parameter_id,
                      .parameter = parameter.parameter});
  }
  return result;
}

std::size_t SgdOptimizerAdapter::parameter_count() const noexcept {
  return impl_->parameters.size();
}

SgdOptimizerCheckpointParticipant::SgdOptimizerCheckpointParticipant(
    std::shared_ptr<SgdOptimizerAdapter> optimizer,
    SgdOptimizerCheckpointOptions options)
    : optimizer_(std::move(optimizer)), options_(options) {
  if (!optimizer_) {
    throw_invalid("optimizer", "a non-null shared SgdOptimizerAdapter", "null",
                  "Create the optimizer before its checkpoint participant.");
  }
  if constexpr (std::endian::native != std::endian::little) {
    throw NotImplemented(
        "Delegated SGD checkpoint byte order is unsupported",
        ErrorDetails{.component = kComponent,
                     .operation = "platform.endian",
                     .expected = "little endian",
                     .actual = "non-little endian",
                     .hint = "Add an explicit tensor byte-swap codec."});
  }
  validate_checkpoint_options(options_, optimizer_->optimizer_id());
  static_cast<void>(describe_state());
}

std::string SgdOptimizerCheckpointParticipant::participant_id() const {
  return optimizer_->optimizer_id();
}

StateManifest SgdOptimizerCheckpointParticipant::describe_state() const {
  try {
    return inspect_optimizer(
               optimizer_->impl_->groups, optimizer_->impl_->parameters,
               *optimizer_->impl_->native, optimizer_->impl_->completed_steps,
               options_, optimizer_->optimizer_id())
        .manifest;
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("describe");
  } catch (const c10::Error& cause) {
    translate_backend(cause, "describe delegated SGD state");
  }
}

void SgdOptimizerCheckpointParticipant::save_state(StateWriter& writer) const {
  try {
    const auto snapshot = inspect_optimizer(
        optimizer_->impl_->groups, optimizer_->impl_->parameters,
        *optimizer_->impl_->native, optimizer_->impl_->completed_steps,
        options_, optimizer_->optimizer_id());
    writer.write_blob("topology", snapshot.topology);
    writer.write_blob("control", snapshot.control);
    for (std::size_t index = 0; index < snapshot.momentum_buffers.size();
         ++index) {
      if (!snapshot.has_momentum[index]) {
        writer.write_blob(momentum_field_name(index),
                          std::span<const std::byte>{});
        continue;
      }
      const auto& parameter = optimizer_->impl_->parameters[index];
      auto cpu = snapshot.momentum_buffers[index]
                     .detach()
                     .to(torch::kCPU)
                     .contiguous();
      const auto bytes = checked_tensor_bytes(cpu, parameter.dtype_code);
      if (bytes != parameter.byte_count) {
        throw_serialization(
            "save.momentum_bytes", std::to_string(parameter.byte_count),
            std::to_string(bytes), "Quiesce optimizer mutation during save.");
      }
      const auto* data = static_cast<const std::byte*>(cpu.data_ptr());
      writer.write_blob(momentum_field_name(index),
                        std::span<const std::byte>(data, bytes));
    }
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("save");
  } catch (const c10::Error& cause) {
    translate_backend(cause, "save delegated SGD state");
  }
}

void SgdOptimizerCheckpointParticipant::load_state(const StateReader& reader,
                                                   RestorePolicy policy) {
  try {
    if (!restore_policy_accepts(policy, RestoreLevel::exact)) {
      throw_compatibility("restore.policy", "a policy accepting exact state",
                          std::string(restore_policy_name(policy)),
                          "Use a declared RestorePolicy value.");
    }
    const auto current = inspect_optimizer(
        optimizer_->impl_->groups, optimizer_->impl_->parameters,
        *optimizer_->impl_->native, optimizer_->impl_->completed_steps,
        options_, optimizer_->optimizer_id());
    const auto& stored_manifest = reader.manifest();
    validate_state_manifest(stored_manifest);
    if (!manifest_equal(stored_manifest, current.manifest)) {
      throw_compatibility("restore.manifest", optimizer_->optimizer_id(),
                          stored_manifest.participant_id,
                          "Use state produced by the same SGD schema.");
    }
    for (const auto& field : current.manifest.fields) {
      if (field.required && !reader.contains(field.name)) {
        throw_serialization("restore.field", field.name, "missing",
                            "Provide every required SGD state field.");
      }
    }

    const auto topology =
        reader.read_blob("topology", current.manifest.fields.front().max_bytes);
    validate_topology_blob(topology, options_);
    if (topology.size() != current.topology.size() ||
        !std::equal(topology.begin(), topology.end(),
                    current.topology.begin())) {
      throw_compatibility(
          "restore.topology",
          "identical stable IDs, immutable options, types, devices, shapes, "
          "strides, and LibTorch version",
          "topology differs",
          "Apply an explicit optimizer state migration before restore.");
    }

    const auto control_bytes =
        reader.read_blob("control", current.manifest.fields[1].max_bytes);
    const auto decoded =
        decode_control(control_bytes, optimizer_->impl_->groups.size(),
                       optimizer_->impl_->parameters.size(), options_);

    std::vector<Tensor> candidates;
    candidates.reserve(optimizer_->impl_->parameters.size());
    at::NoGradGuard no_grad;
    for (std::size_t index = 0; index < optimizer_->impl_->parameters.size();
         ++index) {
      const auto& parameter = optimizer_->impl_->parameters[index];
      const auto bytes =
          reader.read_blob(momentum_field_name(index),
                           std::max<std::uint64_t>(1, parameter.byte_count));
      if (!decoded.has_momentum[index]) {
        if (!bytes.empty()) {
          throw_serialization("restore.momentum_absent", "an empty field",
                              std::to_string(bytes.size()),
                              "Keep presence metadata and raw fields aligned.");
        }
        candidates.emplace_back();
        continue;
      }
      if (optimizer_->impl_->groups[parameter.group_index].options.momentum <=
          0.0) {
        throw_serialization("restore.momentum_presence",
                            "state only for momentum-enabled groups",
                            parameter.parameter_id,
                            "Reject state inconsistent with SGD options.");
      }
      if (bytes.size() != parameter.byte_count) {
        throw_serialization("restore.momentum_bytes",
                            std::to_string(parameter.byte_count),
                            std::to_string(bytes.size()),
                            "Reject truncated or extended optimizer state.");
      }
      auto cpu = torch::empty(parameter.sizes,
                              torch::TensorOptions()
                                  .dtype(scalar_type(parameter.dtype_code))
                                  .device(torch::kCPU));
      if (!bytes.empty()) {
        std::memcpy(cpu.data_ptr(), bytes.data(), bytes.size());
      }
      auto candidate =
          torch::empty_strided(parameter.sizes, parameter.strides,
                               torch::TensorOptions()
                                   .dtype(parameter.parameter.scalar_type())
                                   .device(parameter.parameter.device()));
      candidate.copy_(cpu.reshape(parameter.sizes));
      candidates.push_back(std::move(candidate));
    }

    using NativeStateMap =
        std::remove_reference_t<decltype(optimizer_->impl_->native->state())>;
    static_assert(std::is_nothrow_swappable_v<NativeStateMap>);
    NativeStateMap replacement;
    replacement.reserve(optimizer_->impl_->parameters.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      if (!decoded.has_momentum[index]) {
        continue;
      }
      auto state = std::make_unique<torch::optim::SGDParamState>();
      state->momentum_buffer(candidates[index]);
      replacement.emplace(
          optimizer_->impl_->parameters[index].parameter.unsafeGetTensorImpl(),
          std::move(state));
    }

    const auto previous_rates = optimizer_->learning_rates();
    auto& native_groups = optimizer_->impl_->native->param_groups();
    std::size_t updated_rates = 0;
    try {
      for (; updated_rates < decoded.learning_rates.size(); ++updated_rates) {
        native_options(native_groups[updated_rates])
            .lr(decoded.learning_rates[updated_rates]);
      }
    } catch (...) {
      for (std::size_t index = 0; index < updated_rates; ++index) {
        native_options(native_groups[index]).lr(previous_rates[index]);
      }
      throw;
    }
    optimizer_->impl_->native->state().swap(replacement);
    optimizer_->impl_->completed_steps = decoded.completed_steps;
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("restore");
  } catch (const c10::Error& cause) {
    translate_backend(cause, "restore delegated SGD state");
  }
}

SgdOptimizerAdapter& SgdOptimizerCheckpointParticipant::optimizer() noexcept {
  return *optimizer_;
}

const SgdOptimizerAdapter& SgdOptimizerCheckpointParticipant::optimizer()
    const noexcept {
  return *optimizer_;
}

const SgdOptimizerCheckpointOptions&
SgdOptimizerCheckpointParticipant::options() const noexcept {
  return options_;
}

}  // namespace jyppx::optim
