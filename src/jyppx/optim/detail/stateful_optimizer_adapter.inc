// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>

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

constexpr char kComponent[] = JYPPX_STATEFUL_OPTIMIZER_COMPONENT;
constexpr std::uint32_t kSchemaVersion = 1;
constexpr std::uint32_t kTopologyVersion = 1;
constexpr std::uint32_t kControlVersion = 1;
constexpr std::size_t kMaximumRank = 64;
constexpr std::size_t kAdapterMaximumIdentifierBytes = 4096;
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
constexpr std::array<std::string_view, 1> kStateFieldNames = {"sum"};
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
constexpr std::array<std::string_view, 3> kStateFieldNames = {
    "square_avg", "momentum_buffer", "grad_avg"};
#else
constexpr std::array<std::string_view, 3> kStateFieldNames = {
    "exp_avg", "exp_avg_sq", "max_exp_avg_sq"};
#endif
constexpr std::array<std::byte, 8> kTopologyMagic = {
    std::byte{'J'},
    std::byte{'Y'},
    std::byte{'P'},
    std::byte{'P'},
    std::byte{'X'},
    std::byte{'A'},
    std::byte{JYPPX_STATEFUL_OPTIMIZER_MAGIC_TAG},
    std::byte{'M'}};
constexpr std::array<std::byte, 8> kControlMagic = {
    std::byte{'J'},
    std::byte{'Y'},
    std::byte{'P'},
    std::byte{'P'},
    std::byte{'X'},
    std::byte{'A'},
    std::byte{JYPPX_STATEFUL_OPTIMIZER_MAGIC_TAG},
    std::byte{'C'}};

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
  std::vector<std::byte> topology;   ///< Immutable descriptor bytes.
  std::vector<std::byte> control;    ///< Mutable counters and presence bytes.
  std::vector<Tensor> first_state;   ///< Required first state by parameter.
  std::vector<Tensor> second_state;  ///< Optional second state by parameter.
  std::vector<Tensor> third_state;   ///< Optional third state by parameter.
  std::vector<bool> has_state;       ///< Native state presence by parameter.
  std::vector<std::int64_t> parameter_steps;  ///< Native step counters.
  StateManifest manifest;  ///< Complete storage-neutral state contract.
};

struct DecodedControl {
  std::uint64_t completed_steps;       ///< Restored successful-step count.
  std::vector<double> learning_rates;  ///< Mutable rates in group order.
  std::vector<bool> has_state;         ///< Presence bit in parameter order.
  std::vector<std::int64_t> parameter_steps;  ///< Native step counters.
};

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid delegated " JYPPX_STATEFUL_OPTIMIZER_NAME
                        " state",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Malformed delegated " JYPPX_STATEFUL_OPTIMIZER_NAME
                           " checkpoint state",
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
      "Delegated " JYPPX_STATEFUL_OPTIMIZER_NAME
      " checkpoint is incompatible with the current optimizer",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = std::move(expected),
                   .actual = std::move(actual),
                   .hint = std::move(hint)});
}

[[noreturn]] void throw_resource(std::string operation, std::uint64_t expected,
                                 std::uint64_t actual, std::string hint) {
  throw ResourceExhausted("Delegated " JYPPX_STATEFUL_OPTIMIZER_NAME
                          " resource limit exceeded",
                          ErrorDetails{.component = kComponent,
                                       .operation = std::move(operation),
                                       .expected = std::to_string(expected),
                                       .actual = std::to_string(actual),
                                       .hint = std::move(hint)});
}

[[noreturn]] void translate_bad_alloc(std::string_view operation) {
  throw ResourceExhausted(
      "Unable to allocate delegated " JYPPX_STATEFUL_OPTIMIZER_NAME " state",
      ErrorDetails{.component = kComponent,
                   .operation = std::string(operation),
                   .expected = "memory within configured optimizer limits",
                   .actual = "allocation failed",
                   .hint = "Reduce optimizer state or the active workload."});
}

[[noreturn]] void translate_backend(const c10::Error& cause,
                                    std::string operation) {
  detail::rethrow_backend_error(
      cause,
      detail::BackendErrorContext{
          .component = kComponent,
          .operation = std::move(operation),
          .failure_kind = detail::BackendFailureKind::unknown,
          .expected = "a supported LibTorch " JYPPX_STATEFUL_OPTIMIZER_NAME
                      " operation",
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

void validate_family_options(const JYPPX_STATEFUL_OPTIMIZER_OPTIONS& options,
                             std::string_view group_id) {
  const auto require_non_negative = [&](double value,
                                        std::string_view option_name) {
    if (!std::isfinite(value) || value < 0.0) {
      throw_invalid(std::string(group_id) + "." + std::string(option_name),
                    "a finite non-negative value", std::to_string(value),
                    "Use a " JYPPX_STATEFUL_OPTIMIZER_NAME
                    " option accepted by LibTorch.");
    }
  };
  require_non_negative(options.learning_rate, "learning_rate");
  require_non_negative(options.weight_decay, "weight_decay");
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
  require_non_negative(options.learning_rate_decay, "learning_rate_decay");
  require_non_negative(options.initial_accumulator_value,
                       "initial_accumulator_value");
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
  require_non_negative(options.momentum, "momentum");
  if (!std::isfinite(options.alpha) || options.alpha < 0.0 ||
      options.alpha > 1.0) {
    throw_invalid(std::string(group_id) + ".alpha", "a finite value in [0, 1]",
                  std::to_string(options.alpha),
                  "Use an RMSprop smoothing factor accepted by JYPPX.");
  }
#else
  if (!std::isfinite(options.beta1) || options.beta1 < 0.0 ||
      options.beta1 >= 1.0 || !std::isfinite(options.beta2) ||
      options.beta2 < 0.0 || options.beta2 >= 1.0) {
    throw_invalid(
        std::string(group_id) + ".betas", "finite values in [0, 1)",
        std::to_string(options.beta1) + ":" + std::to_string(options.beta2),
        "Use " JYPPX_STATEFUL_OPTIMIZER_NAME
        " decay factors accepted by LibTorch.");
  }
#endif
  if (!std::isfinite(options.epsilon) || options.epsilon <= 0.0) {
    throw_invalid(std::string(group_id) + ".epsilon", "a finite positive value",
                  std::to_string(options.epsilon),
                  "Use a positive denominator stability constant.");
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
  if (width == 0) {
    throw_invalid("parameter.dtype", "a supported floating dtype code",
                  std::to_string(code),
                  "Reject unknown optimizer tensor encodings.");
  }
  if (unsigned_count > std::numeric_limits<std::uint64_t>::max() / width) {
    throw_resource("parameter.byte_count",
                   std::numeric_limits<std::uint64_t>::max(), unsigned_count,
                   "Reduce parameter dimensions.");
  }
  return unsigned_count * width;
}

JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS make_native_options(
    const JYPPX_STATEFUL_OPTIMIZER_OPTIONS& options) {
  JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS native(options.learning_rate);
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
  native.lr_decay(options.learning_rate_decay)
      .weight_decay(options.weight_decay)
      .initial_accumulator_value(options.initial_accumulator_value)
      .eps(options.epsilon);
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
  native.alpha(options.alpha)
      .eps(options.epsilon)
      .weight_decay(options.weight_decay)
      .momentum(options.momentum)
      .centered(options.centered);
#else
  native.betas({options.beta1, options.beta2})
      .eps(options.epsilon)
      .weight_decay(options.weight_decay)
      .amsgrad(options.amsgrad);
#endif
  return native;
}

bool second_state_required(
    const JYPPX_STATEFUL_OPTIMIZER_OPTIONS& options) noexcept {
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
  static_cast<void>(options);
  return false;
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
  return options.momentum != 0.0;
#else
  static_cast<void>(options);
  return true;
#endif
}

bool third_state_required(
    const JYPPX_STATEFUL_OPTIMIZER_OPTIONS& options) noexcept {
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
  static_cast<void>(options);
  return false;
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
  return options.centered;
#else
  return options.amsgrad;
#endif
}

const Tensor& first_state_tensor(
    const JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE& state) noexcept {
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
  return state.sum();
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
  return state.square_avg();
#else
  return state.exp_avg();
#endif
}

const Tensor& second_state_tensor(
    const JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE& state) noexcept {
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
  static_cast<void>(state);
  static const Tensor undefined;
  return undefined;
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
  return state.momentum_buffer();
#else
  return state.exp_avg_sq();
#endif
}

const Tensor& third_state_tensor(
    const JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE& state) noexcept {
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
  static_cast<void>(state);
  static const Tensor undefined;
  return undefined;
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
  return state.grad_avg();
#else
  return state.max_exp_avg_sq();
#endif
}

void set_native_state_tensors(JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE& state,
                              const Tensor& first, const Tensor& second,
                              const Tensor& third) {
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
  state.sum(first);
  static_cast<void>(second);
  static_cast<void>(third);
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
  state.square_avg(first);
  state.momentum_buffer(second);
  state.grad_avg(third);
#else
  state.exp_avg(first);
  state.exp_avg_sq(second);
  state.max_exp_avg_sq(third);
#endif
}

#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
using NativeStateMap = std::remove_reference_t<
    decltype(std::declval<JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIMIZER&>()
                 .state())>;

NativeStateMap make_fresh_native_state(
    const std::vector<JYPPX_STATEFUL_OPTIMIZER_GROUP>& groups,
    const std::vector<FlatParameter>& parameters) {
  NativeStateMap replacement;
  replacement.reserve(parameters.size());
  at::NoGradGuard no_grad;
  for (const auto& parameter : parameters) {
    auto state = std::make_unique<JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE>();
    state->step(0);
    state->sum(torch::full_like(
        parameter.parameter.detach(),
        groups[parameter.group_index].options.initial_accumulator_value,
        at::MemoryFormat::Preserve));
    replacement.emplace(parameter.parameter.unsafeGetTensorImpl(),
                        std::move(state));
  }
  return replacement;
}

void validate_fresh_native_state(
    const std::vector<FlatParameter>& parameters,
    const JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIMIZER& native) {
  if (native.state().size() != parameters.size()) {
    throw_invalid("native.state_count", std::to_string(parameters.size()),
                  std::to_string(native.state().size()),
                  "Do not mutate eager Adagrad accumulator state.");
  }
  for (const auto& parameter : parameters) {
    const auto iterator =
        native.state().find(parameter.parameter.unsafeGetTensorImpl());
    const auto* state =
        iterator == native.state().end()
            ? nullptr
            : dynamic_cast<const JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE*>(
                  iterator->second.get());
    if (state == nullptr || state->step() != 0) {
      throw_invalid("native.fresh_state", "an eager zero-step Adagrad state",
                    parameter.parameter_id,
                    "Synchronize placement before the first optimizer step.");
    }
  }
}
#endif

const JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS& native_options(
    const torch::optim::OptimizerParamGroup& group) {
  const auto* options =
      dynamic_cast<const JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS*>(
          std::addressof(group.options()));
  if (options == nullptr) {
    throw_invalid("native.options",
                  JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS_NAME,
                  "unexpected optimizer option type",
                  "Do not mutate the delegated native optimizer.");
  }
  return *options;
}

JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS& native_options(
    torch::optim::OptimizerParamGroup& group) {
  auto* options = dynamic_cast<JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS*>(
      std::addressof(group.options()));
  if (options == nullptr) {
    throw_invalid("native.options",
                  JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS_NAME,
                  "unexpected optimizer option type",
                  "Do not mutate the delegated native optimizer.");
  }
  return *options;
}

std::string state_field_name(std::string_view state_name, std::size_t index) {
  std::ostringstream output;
  output << state_name << '.' << std::setw(10) << std::setfill('0') << index;
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
    const std::vector<JYPPX_STATEFUL_OPTIMIZER_GROUP>& groups,
    const std::vector<FlatParameter>& parameters,
    const JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS& limits) {
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
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
    append_double(output, group.options.learning_rate_decay);
    append_double(output, group.options.weight_decay);
    append_double(output, group.options.initial_accumulator_value);
    append_double(output, group.options.epsilon);
    append_u8(output, 0);
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
    append_double(output, group.options.alpha);
    append_double(output, group.options.epsilon);
    append_double(output, group.options.weight_decay);
    append_double(output, group.options.momentum);
    append_u8(output, group.options.centered ? 1 : 0);
#else
    append_double(output, group.options.beta1);
    append_double(output, group.options.beta2);
    append_double(output, group.options.epsilon);
    append_double(output, group.options.weight_decay);
    append_u8(output, group.options.amsgrad ? 1 : 0);
#endif
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
    const std::vector<bool>& has_state,
    const std::vector<std::int64_t>& parameter_steps,
    const JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS& limits) {
  std::vector<std::byte> output(kControlMagic.begin(), kControlMagic.end());
  append_u32(output, kControlVersion);
  append_u64(output, completed_steps);
  append_u32(output, static_cast<std::uint32_t>(learning_rates.size()));
  append_u32(output, static_cast<std::uint32_t>(has_state.size()));
  for (const auto rate : learning_rates) {
    append_double(output, rate);
  }
  for (std::size_t index = 0; index < has_state.size(); ++index) {
    const auto present = has_state[index];
    append_u8(output, present ? 1 : 0);
    append_i64(output, parameter_steps.at(index));
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

void validate_topology_blob(
    std::span<const std::byte> bytes,
    const JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS& limits) {
  if (bytes.size() > limits.max_topology_bytes) {
    throw_resource("topology.bytes", limits.max_topology_bytes, bytes.size(),
                   "Reject oversized topology before parsing.");
  }
  ByteCursor cursor(bytes);
  validate_magic(cursor, kTopologyMagic, "topology.magic",
                 JYPPX_STATEFUL_OPTIMIZER_TOPOLOGY_MAGIC);
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
    JYPPX_STATEFUL_OPTIMIZER_OPTIONS options;
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
    options.learning_rate_decay =
        cursor.read_double("topology.learning_rate_decay");
    options.weight_decay = cursor.read_double("topology.weight_decay");
    options.initial_accumulator_value =
        cursor.read_double("topology.initial_accumulator_value");
    options.epsilon = cursor.read_double("topology.epsilon");
    if (cursor.read_u8("topology.adagrad_flags") != 0) {
      throw_serialization("topology.adagrad_flags", "zero", "non-zero",
                          "Reject unknown Adagrad topology extensions.");
    }
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
    options.alpha = cursor.read_double("topology.alpha");
    options.epsilon = cursor.read_double("topology.epsilon");
    options.weight_decay = cursor.read_double("topology.weight_decay");
    options.momentum = cursor.read_double("topology.momentum");
    const auto centered = cursor.read_u8("topology.centered");
    if (centered > 1) {
      throw_serialization("topology.centered", "a boolean",
                          std::to_string(centered),
                          "Reject invalid optimizer options.");
    }
    options.centered = centered == 1;
#else
    options.beta1 = cursor.read_double("topology.beta1");
    options.beta2 = cursor.read_double("topology.beta2");
    options.epsilon = cursor.read_double("topology.epsilon");
    options.weight_decay = cursor.read_double("topology.weight_decay");
    const auto amsgrad = cursor.read_u8("topology.amsgrad");
    if (amsgrad > 1) {
      throw_serialization("topology.amsgrad", "a boolean",
                          std::to_string(amsgrad),
                          "Reject invalid optimizer options.");
    }
    options.amsgrad = amsgrad == 1;
#endif
    for (std::size_t reserved = 0; reserved < 7; ++reserved) {
      if (cursor.read_u8("topology.reserved") != 0) {
        throw_serialization("topology.reserved", "zero", "non-zero",
                            "Reject unknown topology extensions.");
      }
    }
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
    if (!std::isfinite(options.learning_rate_decay) ||
        options.learning_rate_decay < 0.0 ||
        !std::isfinite(options.weight_decay) || options.weight_decay < 0.0 ||
        !std::isfinite(options.initial_accumulator_value) ||
        options.initial_accumulator_value < 0.0 ||
        !std::isfinite(options.epsilon) || options.epsilon <= 0.0) {
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
    if (!std::isfinite(options.alpha) || options.alpha < 0.0 ||
        options.alpha > 1.0 || !std::isfinite(options.epsilon) ||
        options.epsilon <= 0.0 || !std::isfinite(options.weight_decay) ||
        options.weight_decay < 0.0 || !std::isfinite(options.momentum) ||
        options.momentum < 0.0) {
#else
    if (!std::isfinite(options.beta1) || options.beta1 < 0.0 ||
        options.beta1 >= 1.0 || !std::isfinite(options.beta2) ||
        options.beta2 < 0.0 || options.beta2 >= 1.0 ||
        !std::isfinite(options.epsilon) || options.epsilon <= 0.0 ||
        !std::isfinite(options.weight_decay) || options.weight_decay < 0.0) {
#endif
      throw_serialization(
          "topology.options",
          "valid finite " JYPPX_STATEFUL_OPTIMIZER_NAME " options", group_id,
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
                       "Reject oversized " JYPPX_STATEFUL_OPTIMIZER_NAME
                       " state before allocation.");
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
      for (std::size_t field = 0; field < kStateFieldNames.size(); ++field) {
        total = checked_add(total, std::max<std::uint64_t>(1, byte_count),
                            limits.max_total_bytes, "state.total_bytes");
      }
    }
  }
  if (decoded_parameters != parameter_count || !cursor.empty()) {
    throw_serialization("topology.trailing_bytes",
                        "exact parameter count and no trailing metadata",
                        "mismatch", "Reject inconsistent topology extensions.");
  }
}

DecodedControl decode_control(
    std::span<const std::byte> bytes, std::size_t expected_groups,
    std::size_t expected_parameters,
    const JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS& limits) {
  if (bytes.size() > limits.max_control_bytes) {
    throw_resource("control.bytes", limits.max_control_bytes, bytes.size(),
                   "Reject oversized control metadata before parsing.");
  }
  ByteCursor cursor(bytes);
  validate_magic(cursor, kControlMagic, "control.magic",
                 JYPPX_STATEFUL_OPTIMIZER_CONTROL_MAGIC);
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
  decoded.has_state.reserve(parameter_count);
  decoded.parameter_steps.reserve(parameter_count);
  for (std::uint32_t index = 0; index < parameter_count; ++index) {
    const auto present = cursor.read_u8("control.state_presence");
    if (present > 1) {
      throw_serialization("control.state_presence", "a boolean",
                          std::to_string(present),
                          "Reject invalid " JYPPX_STATEFUL_OPTIMIZER_NAME
                          " state presence metadata.");
    }
    const auto step = cursor.read_i64("control.parameter_step");
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
    if (present == 0 || step < 0 ||
        (step > 0 &&
         static_cast<std::uint64_t>(step) > decoded.completed_steps)) {
      throw_serialization(
          "control.parameter_step",
          "eager state with a non-negative count within completed steps",
          std::to_string(step),
          "Reject inconsistent Adagrad counters before optimizer mutation.");
    }
#else
    if ((present == 0 && step != 0) || (present == 1 && step <= 0) ||
        (step > 0 &&
         static_cast<std::uint64_t>(step) > decoded.completed_steps)) {
      throw_serialization(
          "control.parameter_step",
          "zero without state, or a positive count within completed steps",
          std::to_string(step),
          "Reject inconsistent " JYPPX_STATEFUL_OPTIMIZER_NAME
          " counters before optimizer mutation.");
    }
#endif
    decoded.has_state.push_back(present == 1);
    decoded.parameter_steps.push_back(step);
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

void validate_checkpoint_options(
    const JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS& options,
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

void validate_current_topology(
    const std::vector<JYPPX_STATEFUL_OPTIMIZER_GROUP>& groups,
    const std::vector<FlatParameter>& parameters,
    const JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS& limits) {
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
    const std::vector<JYPPX_STATEFUL_OPTIMIZER_GROUP>& groups,
    const std::vector<FlatParameter>& parameters,
    const JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIMIZER& native) {
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
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
    if (!std::isfinite(options.lr()) || options.lr() < 0.0 ||
        options.lr_decay() != expected.options.learning_rate_decay ||
        options.weight_decay() != expected.options.weight_decay ||
        options.initial_accumulator_value() !=
            expected.options.initial_accumulator_value ||
        options.eps() != expected.options.epsilon) {
#elif defined(JYPPX_STATEFUL_OPTIMIZER_RMSPROP)
    if (!std::isfinite(options.lr()) || options.lr() < 0.0 ||
        options.alpha() != expected.options.alpha ||
        options.eps() != expected.options.epsilon ||
        options.weight_decay() != expected.options.weight_decay ||
        options.momentum() != expected.options.momentum ||
        options.centered() != expected.options.centered) {
#else
    const auto [beta1, beta2] = options.betas();
    if (!std::isfinite(options.lr()) || options.lr() < 0.0 ||
        beta1 != expected.options.beta1 || beta2 != expected.options.beta2 ||
        options.eps() != expected.options.epsilon ||
        options.weight_decay() != expected.options.weight_decay ||
        options.amsgrad() != expected.options.amsgrad) {
#endif
      throw_invalid("native.options",
                    "the stable " JYPPX_STATEFUL_OPTIMIZER_NAME " descriptor",
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
          "Delegated " JYPPX_STATEFUL_OPTIMIZER_NAME
          " does not support sparse gradients",
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
    const std::vector<JYPPX_STATEFUL_OPTIMIZER_GROUP>& groups,
    const std::vector<FlatParameter>& parameters,
    const JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIMIZER& native,
    std::uint64_t completed_steps,
    const JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS& limits,
    std::string participant_id) {
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

  std::vector<Tensor> first_state;
  std::vector<Tensor> second_state;
  std::vector<Tensor> third_state;
  std::vector<bool> has_state;
  std::vector<std::int64_t> parameter_steps;
  first_state.reserve(parameters.size());
  second_state.reserve(parameters.size());
  third_state.reserve(parameters.size());
  has_state.reserve(parameters.size());
  parameter_steps.reserve(parameters.size());
  for (const auto& parameter : parameters) {
    const auto state =
        native.state().find(parameter.parameter.unsafeGetTensorImpl());
    if (state == native.state().end()) {
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
      throw_invalid("native.state", "eager state for every bound parameter",
                    parameter.parameter_id,
                    "Recover the missing Adagrad accumulator from a valid "
                    "checkpoint.");
#else
      first_state.emplace_back();
      second_state.emplace_back();
      third_state.emplace_back();
      has_state.push_back(false);
      parameter_steps.push_back(0);
      continue;
#endif
    }
    const auto* family_state =
        dynamic_cast<const JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE*>(
            state->second.get());
    if (family_state == nullptr) {
      throw_invalid("native.state_type",
                    JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE_NAME,
                    parameter.parameter_id,
                    "Do not inject state from another optimizer algorithm.");
    }
    const auto validate_state_tensor = [&](const Tensor& tensor,
                                           std::string_view state_name) {
      if (!tensor.defined() ||
          tensor.scalar_type() != parameter.parameter.scalar_type() ||
          tensor.device() != parameter.parameter.device() ||
          tensor.sizes().vec() != parameter.sizes ||
          tensor.strides().vec() != parameter.strides ||
          tensor.layout() != c10::Layout::Strided ||
          !tensor.is_non_overlapping_and_dense() ||
          checked_tensor_bytes(tensor, parameter.dtype_code) !=
              parameter.byte_count) {
        throw_invalid("native." + std::string(state_name),
                      "dense state matching its stable parameter descriptor",
                      parameter.parameter_id,
                      "Recover from a valid optimizer checkpoint.");
      }
    };
    validate_state_tensor(first_state_tensor(*family_state),
                          kStateFieldNames[0]);
#if !defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
    const auto& descriptor = groups[parameter.group_index].options;
    if (second_state_required(descriptor)) {
      validate_state_tensor(second_state_tensor(*family_state),
                            kStateFieldNames[1]);
    } else if (second_state_tensor(*family_state).defined()) {
      throw_invalid("native." + std::string(kStateFieldNames[1]),
                    "no disabled optional state", parameter.parameter_id,
                    "Recover from a checkpoint matching immutable options.");
    }
    if (third_state_required(descriptor)) {
      validate_state_tensor(third_state_tensor(*family_state),
                            kStateFieldNames[2]);
    } else if (third_state_tensor(*family_state).defined()) {
      throw_invalid("native." + std::string(kStateFieldNames[2]),
                    "no disabled optional state", parameter.parameter_id,
                    "Recover from a checkpoint matching immutable options.");
    }
#endif
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
    if (family_state->step() < 0 ||
        static_cast<std::uint64_t>(family_state->step()) > completed_steps) {
      throw_invalid("native.parameter_step",
                    "a non-negative count within completed adapter steps",
                    std::to_string(family_state->step()),
                    "Recover from a valid optimizer checkpoint.");
    }
#else
    if (family_state->step() <= 0 ||
        static_cast<std::uint64_t>(family_state->step()) > completed_steps) {
      throw_invalid("native.parameter_step",
                    "a positive count within completed adapter steps",
                    std::to_string(family_state->step()),
                    "Recover from a valid optimizer checkpoint.");
    }
#endif
    first_state.push_back(first_state_tensor(*family_state));
    second_state.push_back(second_state_tensor(*family_state));
    third_state.push_back(third_state_tensor(*family_state));
    has_state.push_back(true);
    parameter_steps.push_back(family_state->step());
  }

  auto topology = encode_topology(groups, parameters, limits);
  auto control = encode_control(completed_steps, learning_rates, has_state,
                                parameter_steps, limits);
  std::uint64_t total =
      checked_add(topology.size(), control.size(), limits.max_total_bytes,
                  "state.total_bytes");
  StateManifest manifest{
      .participant_id = std::move(participant_id),
      .schema_version = kSchemaVersion,
      .restore_level = RestoreLevel::exact,
      .fields =
          {{.name = "topology",
            .media_type =
                "application/vnd.jyppx." JYPPX_STATEFUL_OPTIMIZER_MEDIA_STEM
                "-topology-v1",
            .max_bytes = topology.size()},
           {.name = "control",
            .media_type =
                "application/vnd.jyppx." JYPPX_STATEFUL_OPTIMIZER_MEDIA_STEM
                "-control-v1",
            .max_bytes = control.size()}},
      .degradation_reasons = {},
  };
  manifest.fields.reserve(parameters.size() * kStateFieldNames.size() + 2);
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    const auto maximum =
        std::max<std::uint64_t>(1, parameters[index].byte_count);
    for (const auto state_name : kStateFieldNames) {
      total = checked_add(total, maximum, limits.max_total_bytes,
                          "state.total_bytes");
      manifest.fields.push_back(
          {.name = state_field_name(state_name, index),
           .media_type = "application/vnd.jyppx.tensor-raw-v1",
           .max_bytes = maximum});
    }
  }
  static_cast<void>(total);
  validate_state_manifest(manifest);
  return {.topology = std::move(topology),
          .control = std::move(control),
          .first_state = std::move(first_state),
          .second_state = std::move(second_state),
          .third_state = std::move(third_state),
          .has_state = std::move(has_state),
          .parameter_steps = std::move(parameter_steps),
          .manifest = std::move(manifest)};
}

}  // namespace

struct JYPPX_STATEFUL_OPTIMIZER_ADAPTER::Impl {
  std::string optimizer_id;                      ///< Stable optimizer identity.
  std::vector<std::string> parameter_group_ids;  ///< Stable group identities.
  std::vector<JYPPX_STATEFUL_OPTIMIZER_GROUP>
      groups;                             ///< Stable public descriptor.
  std::vector<FlatParameter> parameters;  ///< Flattened checkpoint order.
  std::unique_ptr<JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIMIZER>
      native;                         ///< Delegated algorithm owner.
  std::uint64_t completed_steps = 0;  ///< Successful native step count.
};

JYPPX_STATEFUL_OPTIMIZER_ADAPTER::JYPPX_STATEFUL_OPTIMIZER_ADAPTER(
    std::string optimizer_id,
    std::vector<JYPPX_STATEFUL_OPTIMIZER_GROUP> groups)
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
      validate_family_options(group.options, group.group_id);
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
    impl_->native = std::make_unique<JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIMIZER>(
        native_groups, defaults);
    for (std::size_t index = 0; index < groups.size(); ++index) {
      impl_->native->param_groups()[index].set_options(
          std::make_unique<JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS>(
              make_native_options(groups[index].options)));
    }
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
    auto initial_state = make_fresh_native_state(groups, impl_->parameters);
    impl_->native->state().swap(initial_state);
#endif
    impl_->optimizer_id = std::move(optimizer_id);
    impl_->groups = std::move(groups);
    static_cast<void>(inspect_native_groups(impl_->groups, impl_->parameters,
                                            *impl_->native));
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("construct");
  } catch (const c10::Error& cause) {
    translate_backend(cause,
                      "construct delegated " JYPPX_STATEFUL_OPTIMIZER_NAME);
  }
}

JYPPX_STATEFUL_OPTIMIZER_ADAPTER::~JYPPX_STATEFUL_OPTIMIZER_ADAPTER() = default;

Tensor JYPPX_STATEFUL_OPTIMIZER_ADAPTER::step(LossClosure closure) {
  try {
    if (impl_->completed_steps == std::numeric_limits<std::uint64_t>::max()) {
      throw ResourceExhausted(
          "Delegated " JYPPX_STATEFUL_OPTIMIZER_NAME " step counter exhausted",
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
    translate_backend(cause,
                      "delegated " JYPPX_STATEFUL_OPTIMIZER_NAME " step");
  }
}

void JYPPX_STATEFUL_OPTIMIZER_ADAPTER::zero_grad(bool set_to_none) {
  try {
    impl_->native->zero_grad(set_to_none);
  } catch (const Error&) {
    throw;
  } catch (const c10::Error& cause) {
    translate_backend(
        cause, "zero delegated " JYPPX_STATEFUL_OPTIMIZER_NAME " gradients");
  }
}

void JYPPX_STATEFUL_OPTIMIZER_ADAPTER::synchronize_parameter_placement() {
  try {
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
    if (impl_->completed_steps != 0) {
      throw NotReady(
          "Cannot synchronize placement after "
          "delegated " JYPPX_STATEFUL_OPTIMIZER_NAME " has stepped",
          ErrorDetails{
              .component = kComponent,
              .operation = "synchronize_parameter_placement",
              .expected = "a fresh zero-step optimizer",
              .actual =
                  std::to_string(impl_->completed_steps) + " completed steps",
              .hint = "Move the model through its strategy before the first "
                      "optimizer update."});
    }
    validate_fresh_native_state(impl_->parameters, *impl_->native);
#else
    if (impl_->completed_steps != 0 || !impl_->native->state().empty()) {
      throw NotReady(
          "Cannot synchronize placement after "
          "delegated " JYPPX_STATEFUL_OPTIMIZER_NAME " state exists",
          ErrorDetails{
              .component = kComponent,
              .operation = "synchronize_parameter_placement",
              .expected = "a fresh zero-step optimizer with no native state",
              .actual = std::to_string(impl_->completed_steps) +
                        " completed steps and " +
                        std::to_string(impl_->native->state().size()) +
                        " native state entries",
              .hint = "Move the model through its strategy before the first "
                      "optimizer update."});
    }
#endif

    auto candidate = impl_->parameters;
    for (auto& descriptor : candidate) {
      const auto& parameter = descriptor.parameter;
      if (!parameter.defined() || !parameter.requires_grad() ||
          parameter.is_meta() || parameter.is_quantized() ||
          parameter.layout() != c10::Layout::Strided ||
          !parameter.is_non_overlapping_and_dense() ||
          parameter.sizes().vec() != descriptor.sizes ||
          parameter.strides().vec() != descriptor.strides) {
        throw_invalid(
            "synchronize_parameter_placement",
            "the same dense trainable tensor identity, shape, and strides",
            descriptor.parameter_id,
            "Rebuild the optimizer after changing parameter topology.");
      }
      descriptor.dtype_code = dtype_code(parameter.scalar_type());
      descriptor.device_code = device_code(parameter.device());
      descriptor.device_index =
          static_cast<std::int32_t>(parameter.device().index());
      descriptor.byte_count =
          checked_tensor_bytes(parameter, descriptor.dtype_code);
    }
    static_cast<void>(
        inspect_native_groups(impl_->groups, candidate, *impl_->native));
#if defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
    auto replacement = make_fresh_native_state(impl_->groups, candidate);
    impl_->native->state().swap(replacement);
#endif
    impl_->parameters.swap(candidate);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("synchronize_parameter_placement");
  } catch (const c10::Error& cause) {
    translate_backend(cause,
                      "synchronize delegated " JYPPX_STATEFUL_OPTIMIZER_NAME
                      " placement");
  }
}

std::vector<double> JYPPX_STATEFUL_OPTIMIZER_ADAPTER::learning_rates() const {
  try {
    return inspect_native_groups(impl_->groups, impl_->parameters,
                                 *impl_->native);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("learning_rates");
  }
}

void JYPPX_STATEFUL_OPTIMIZER_ADAPTER::set_learning_rates(
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

const std::string& JYPPX_STATEFUL_OPTIMIZER_ADAPTER::optimizer_id()
    const noexcept {
  return impl_->optimizer_id;
}

const std::vector<std::string>&
JYPPX_STATEFUL_OPTIMIZER_ADAPTER::parameter_group_ids() const noexcept {
  return impl_->parameter_group_ids;
}

std::uint64_t JYPPX_STATEFUL_OPTIMIZER_ADAPTER::completed_steps()
    const noexcept {
  return impl_->completed_steps;
}

const std::vector<JYPPX_STATEFUL_OPTIMIZER_GROUP>&
JYPPX_STATEFUL_OPTIMIZER_ADAPTER::groups() const noexcept {
  return impl_->groups;
}

std::vector<OptimizerParameterBinding>
JYPPX_STATEFUL_OPTIMIZER_ADAPTER::parameter_bindings() const {
  std::vector<OptimizerParameterBinding> result;
  result.reserve(impl_->parameters.size());
  for (const auto& parameter : impl_->parameters) {
    result.push_back({.parameter_id = parameter.parameter_id,
                      .parameter = parameter.parameter});
  }
  return result;
}

std::size_t JYPPX_STATEFUL_OPTIMIZER_ADAPTER::parameter_count() const noexcept {
  return impl_->parameters.size();
}

JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT::JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT(
    std::shared_ptr<JYPPX_STATEFUL_OPTIMIZER_ADAPTER> optimizer,
    JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS options)
    : optimizer_(std::move(optimizer)), options_(options) {
  if (!optimizer_) {
    throw_invalid("optimizer", JYPPX_STATEFUL_OPTIMIZER_ADAPTER_DESCRIPTION,
                  "null",
                  "Create the optimizer before its checkpoint participant.");
  }
  if constexpr (std::endian::native != std::endian::little) {
    throw NotImplemented(
        "Delegated " JYPPX_STATEFUL_OPTIMIZER_NAME
        " checkpoint byte order is unsupported",
        ErrorDetails{.component = kComponent,
                     .operation = "platform.endian",
                     .expected = "little endian",
                     .actual = "non-little endian",
                     .hint = "Add an explicit tensor byte-swap codec."});
  }
  validate_checkpoint_options(options_, optimizer_->optimizer_id());
  static_cast<void>(describe_state());
}

std::string JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT::participant_id() const {
  return optimizer_->optimizer_id();
}

StateManifest JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT::describe_state() const {
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
    translate_backend(
        cause, "describe delegated " JYPPX_STATEFUL_OPTIMIZER_NAME " state");
  }
}

void JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT::save_state(
    StateWriter& writer) const {
  try {
    const auto snapshot = inspect_optimizer(
        optimizer_->impl_->groups, optimizer_->impl_->parameters,
        *optimizer_->impl_->native, optimizer_->impl_->completed_steps,
        options_, optimizer_->optimizer_id());
    writer.write_blob("topology", snapshot.topology);
    writer.write_blob("control", snapshot.control);
    for (std::size_t index = 0; index < snapshot.has_state.size(); ++index) {
      const auto& parameter = optimizer_->impl_->parameters[index];
      const auto write_tensor = [&](std::string_view state_name,
                                    const Tensor& tensor) {
        const auto field_name = state_field_name(state_name, index);
        if (!tensor.defined()) {
          writer.write_blob(field_name, std::span<const std::byte>{});
          return;
        }
        auto cpu = tensor.detach().to(torch::kCPU).contiguous();
        const auto bytes = checked_tensor_bytes(cpu, parameter.dtype_code);
        if (bytes != parameter.byte_count) {
          throw_serialization("save." + std::string(state_name) + "_bytes",
                              std::to_string(parameter.byte_count),
                              std::to_string(bytes),
                              "Quiesce optimizer mutation during save.");
        }
        const auto* data = static_cast<const std::byte*>(cpu.data_ptr());
        writer.write_blob(field_name, std::span<const std::byte>(data, bytes));
      };
      write_tensor(kStateFieldNames[0], snapshot.first_state[index]);
#if !defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
      write_tensor(kStateFieldNames[1], snapshot.second_state[index]);
      write_tensor(kStateFieldNames[2], snapshot.third_state[index]);
#endif
    }
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("save");
  } catch (const c10::Error& cause) {
    translate_backend(cause,
                      "save delegated " JYPPX_STATEFUL_OPTIMIZER_NAME " state");
  }
}

void JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT::load_state(const StateReader& reader,
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
      throw_compatibility(
          "restore.manifest", optimizer_->optimizer_id(),
          stored_manifest.participant_id,
          "Use state produced by the same " JYPPX_STATEFUL_OPTIMIZER_NAME
          " schema.");
    }
    for (const auto& field : current.manifest.fields) {
      if (field.required && !reader.contains(field.name)) {
        throw_serialization(
            "restore.field", field.name, "missing",
            "Provide every required " JYPPX_STATEFUL_OPTIMIZER_NAME
            " state field.");
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

    std::vector<Tensor> first_state_candidates;
    std::vector<Tensor> second_state_candidates;
    std::vector<Tensor> third_state_candidates;
    first_state_candidates.reserve(optimizer_->impl_->parameters.size());
    second_state_candidates.reserve(optimizer_->impl_->parameters.size());
    third_state_candidates.reserve(optimizer_->impl_->parameters.size());
    at::NoGradGuard no_grad;
    for (std::size_t index = 0; index < optimizer_->impl_->parameters.size();
         ++index) {
      const auto& parameter = optimizer_->impl_->parameters[index];
      const auto maximum = std::max<std::uint64_t>(1, parameter.byte_count);
      const auto first_state_bytes = reader.read_blob(
          state_field_name(kStateFieldNames[0], index), maximum);
      std::span<const std::byte> second_state_bytes;
      std::span<const std::byte> third_state_bytes;
#if !defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
      second_state_bytes = reader.read_blob(
          state_field_name(kStateFieldNames[1], index), maximum);
      third_state_bytes = reader.read_blob(
          state_field_name(kStateFieldNames[2], index), maximum);
      const auto& descriptor =
          optimizer_->impl_->groups[parameter.group_index].options;
#endif
      if (!decoded.has_state[index]) {
        if (!first_state_bytes.empty() || !second_state_bytes.empty() ||
            !third_state_bytes.empty()) {
          throw_serialization("restore.state_absent", "empty state fields",
                              parameter.parameter_id,
                              "Keep presence metadata and raw fields aligned.");
        }
        first_state_candidates.emplace_back();
        second_state_candidates.emplace_back();
        third_state_candidates.emplace_back();
        continue;
      }
      const auto decode_tensor = [&](std::string_view state_name,
                                     std::span<const std::byte> bytes) {
        if (bytes.size() != parameter.byte_count) {
          throw_serialization("restore." + std::string(state_name) + "_bytes",
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
        return candidate;
      };
      first_state_candidates.push_back(
          decode_tensor(kStateFieldNames[0], first_state_bytes));
#if !defined(JYPPX_STATEFUL_OPTIMIZER_ADAGRAD)
      if (second_state_required(descriptor)) {
        second_state_candidates.push_back(
            decode_tensor(kStateFieldNames[1], second_state_bytes));
      } else {
        if (!second_state_bytes.empty()) {
          throw_serialization(
              "restore." + std::string(kStateFieldNames[1]),
              "an empty field when the optional state is disabled",
              std::to_string(second_state_bytes.size()),
              "Use state matching immutable " JYPPX_STATEFUL_OPTIMIZER_NAME
              " options.");
        }
        second_state_candidates.emplace_back();
      }
      if (third_state_required(descriptor)) {
        third_state_candidates.push_back(
            decode_tensor(kStateFieldNames[2], third_state_bytes));
      } else {
        if (!third_state_bytes.empty()) {
          throw_serialization(
              "restore." + std::string(kStateFieldNames[2]),
              "an empty field when the optional state is disabled",
              std::to_string(third_state_bytes.size()),
              "Use state matching immutable " JYPPX_STATEFUL_OPTIMIZER_NAME
              " options.");
        }
        third_state_candidates.emplace_back();
      }
#else
      second_state_candidates.emplace_back();
      third_state_candidates.emplace_back();
#endif
    }

    using NativeStateMap =
        std::remove_reference_t<decltype(optimizer_->impl_->native->state())>;
    static_assert(std::is_nothrow_swappable_v<NativeStateMap>);
    NativeStateMap replacement;
    replacement.reserve(optimizer_->impl_->parameters.size());
    for (std::size_t index = 0; index < first_state_candidates.size();
         ++index) {
      if (!decoded.has_state[index]) {
        continue;
      }
      auto state = std::make_unique<JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE>();
      state->step(decoded.parameter_steps[index]);
      set_native_state_tensors(*state, first_state_candidates[index],
                               second_state_candidates[index],
                               third_state_candidates[index]);
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
    translate_backend(
        cause, "restore delegated " JYPPX_STATEFUL_OPTIMIZER_NAME " state");
  }
}

JYPPX_STATEFUL_OPTIMIZER_ADAPTER&
JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT::optimizer() noexcept {
  return *optimizer_;
}

const JYPPX_STATEFUL_OPTIMIZER_ADAPTER&
JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT::optimizer() const noexcept {
  return *optimizer_;
}

const JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS&
JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT::options() const noexcept {
  return options_;
}

}  // namespace jyppx::optim
