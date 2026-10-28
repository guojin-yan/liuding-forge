// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/optim/optimizer_adapter.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>

#include <torch/cuda.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::liuding::training {
namespace {

constexpr char kComponent[] = "training.single_device_strategy";
constexpr char kTopologyField[] = "topology";
constexpr char kControlField[] = "control";
constexpr char kTopologyMediaType[] =
    "application/vnd.jyppx.single-device-strategy-topology-v2";
constexpr char kControlMediaType[] =
    "application/vnd.jyppx.single-device-strategy-control-v2";
constexpr std::string_view kTopologyMagic =
    "JYPPX-SINGLE-DEVICE-STRATEGY-TOPOLOGY-V2";
constexpr std::string_view kControlMagic =
    "JYPPX-SINGLE-DEVICE-STRATEGY-CONTROL-V2";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid single-device strategy contract",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_incompatible(std::string operation,
                                     std::string expected, std::string actual,
                                     std::string hint) {
  throw CompatibilityError("Single-device strategy state is incompatible",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Malformed single-device strategy state",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_exhausted(std::string operation, std::string expected,
                                  std::string actual, std::string hint) {
  throw ResourceExhausted("Single-device strategy resource limit exceeded",
                          ErrorDetails{.component = kComponent,
                                       .operation = std::move(operation),
                                       .expected = std::move(expected),
                                       .actual = std::move(actual),
                                       .hint = std::move(hint)});
}

bool is_identifier(std::string_view value) {
  if (value.empty()) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    const auto byte = static_cast<unsigned char>(character);
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
           (byte >= '0' && byte <= '9') || byte == '.' || byte == '_' ||
           byte == '-';
  });
}

void validate_options(const SingleDeviceStrategyOptions& options) {
  if (!is_identifier(options.strategy_id) || options.max_optimizers == 0 ||
      options.max_parameters == 0 || options.max_metrics == 0 ||
      options.max_string_bytes == 0 || options.max_topology_bytes == 0 ||
      options.max_control_bytes == 0) {
    throw_invalid("options", "a stable ID and positive resource limits",
                  options.strategy_id,
                  "Configure explicit nonzero strategy bounds.");
  }
  if (options.strategy_id.size() > options.max_string_bytes) {
    throw_exhausted("options.strategy_id",
                    std::to_string(options.max_string_bytes),
                    std::to_string(options.strategy_id.size()),
                    "Use a shorter stable strategy identity.");
  }
  if (!options.device.is_cpu() && !options.device.is_cuda()) {
    throw_invalid("options.device", "cpu or an explicit CUDA device",
                  options.device.str(),
                  "Use a v1.0 single-device supported backend.");
  }
  if (options.device.is_cpu() && options.device.has_index()) {
    throw_invalid("options.device", "cpu without an index",
                  options.device.str(),
                  "Use the canonical cpu device descriptor.");
  }
  if (options.device.is_cuda() && !options.device.has_index()) {
    throw_invalid("options.device", "cuda with an explicit non-negative index",
                  options.device.str(),
                  "Use cuda:0 or another visible device for stable topology.");
  }
}

void ensure_ready(bool prepared, bool closed, std::string operation) {
  if (!prepared) {
    throw NotReady(
        "Single-device strategy is not prepared",
        ErrorDetails{.component = kComponent,
                     .operation = std::move(operation),
                     .expected = "successful prepare",
                     .actual = "unprepared",
                     .hint = "Prepare a matching model and optimizer "
                             "before execution."});
  }
  if (closed) {
    throw NotReady("Single-device strategy is closed",
                   ErrorDetails{.component = kComponent,
                                .operation = std::move(operation),
                                .expected = "an open strategy",
                                .actual = "closed",
                                .hint = "Create a new strategy for another "
                                        "training run."});
  }
}

void append_u8(std::vector<std::byte>& output, std::uint8_t value) {
  output.push_back(static_cast<std::byte>(value));
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    output.push_back(static_cast<std::byte>((value >> (index * 8U)) & 0xffU));
  }
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    output.push_back(static_cast<std::byte>((value >> (index * 8U)) & 0xffU));
  }
}

void append_i64(std::vector<std::byte>& output, std::int64_t value) {
  append_u64(output, std::bit_cast<std::uint64_t>(value));
}

void append_string(std::vector<std::byte>& output, std::string_view value) {
  append_u64(output, static_cast<std::uint64_t>(value.size()));
  for (const auto character : value) {
    output.push_back(
        static_cast<std::byte>(static_cast<unsigned char>(character)));
  }
}

class ByteReader final {
 public:
  explicit ByteReader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  std::uint8_t read_u8(std::string_view field) {
    require(sizeof(std::uint8_t), field);
    return std::to_integer<std::uint8_t>(bytes_[offset_++]);
  }

  std::uint64_t read_u64(std::string_view field) {
    require(sizeof(std::uint64_t), field);
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < sizeof(value); ++index) {
      value |= static_cast<std::uint64_t>(
                   std::to_integer<std::uint8_t>(bytes_[offset_ + index]))
               << (index * 8U);
    }
    offset_ += sizeof(value);
    return value;
  }

  std::string read_string(std::string_view field, std::size_t max_bytes) {
    const auto encoded_size = read_u64(field);
    if (encoded_size > max_bytes ||
        encoded_size > static_cast<std::uint64_t>(bytes_.size() - offset_)) {
      throw_serialization(std::string(field),
                          "a bounded length-prefixed string",
                          std::to_string(encoded_size),
                          "Reject truncated or oversized strategy state.");
    }
    const auto size = static_cast<std::size_t>(encoded_size);
    std::string value;
    value.reserve(size);
    for (std::size_t index = 0; index < size; ++index) {
      value.push_back(static_cast<char>(
          std::to_integer<unsigned char>(bytes_[offset_ + index])));
    }
    offset_ += size;
    return value;
  }

  void require_finished(std::string_view field) const {
    if (offset_ != bytes_.size()) {
      throw_serialization(
          std::string(field), "no trailing bytes",
          std::to_string(bytes_.size() - offset_) + " trailing bytes",
          "Reject state from an unknown schema.");
    }
  }

 private:
  void require(std::size_t count, std::string_view field) const {
    if (count > bytes_.size() - offset_) {
      throw_serialization(std::string(field), "a complete encoded value",
                          "truncated", "Reject incomplete strategy state.");
    }
  }

  std::span<const std::byte> bytes_;  ///< Borrowed complete field bytes.
  std::size_t offset_ = 0;            ///< Next unread byte position.
};

void validate_manifest(const StateManifest& stored,
                       const StateManifest& expected, RestorePolicy policy) {
  validate_state_manifest(stored);
  if (!restore_policy_accepts(policy, stored.restore_level)) {
    throw_incompatible("manifest.restore_level",
                       std::string(restore_policy_name(policy)),
                       std::string(restore_level_name(stored.restore_level)),
                       "Use exact restore policy for strategy state.");
  }
  if (stored.participant_id != expected.participant_id ||
      stored.schema_version != expected.schema_version ||
      stored.restore_level != RestoreLevel::exact ||
      stored.fields.size() != expected.fields.size()) {
    throw_incompatible("manifest", expected.participant_id,
                       stored.participant_id,
                       "Restore only current state for the same strategy.");
  }
  for (const auto& right : expected.fields) {
    const auto left = std::find_if(stored.fields.begin(), stored.fields.end(),
                                   [&](const StateFieldDescriptor& candidate) {
                                     return candidate.name == right.name;
                                   });
    if (left == stored.fields.end() || left->media_type != right.media_type ||
        left->required != right.required || left->max_bytes > right.max_bytes) {
      throw_incompatible("manifest.fields", right.name,
                         left == stored.fields.end() ? "missing" : left->name,
                         "Reject a changed strategy field contract.");
    }
  }
}

}  // namespace

std::string_view strategy_barrier_reason_name(
    StrategyBarrierReason reason) noexcept {
  switch (reason) {
    case StrategyBarrierReason::event_commit:
      return "event_commit";
    case StrategyBarrierReason::checkpoint:
      return "checkpoint";
    case StrategyBarrierReason::shutdown:
      return "shutdown";
  }
  return "unknown";
}

std::string_view strategy_step_status_name(StrategyStepStatus status) noexcept {
  switch (status) {
    case StrategyStepStatus::applied:
      return "applied";
  }
  return "unknown";
}

struct SingleDeviceStrategy::Impl {
  struct ParameterTopology {
    std::string parameter_id;         ///< Stable model and optimizer key.
    std::string device;               ///< Canonical parameter device.
    std::int64_t scalar_type = 0;     ///< LibTorch scalar type numeric value.
    std::vector<std::int64_t> shape;  ///< Fixed parameter dimensions.
  };

  /** @brief Tracks one canonical prepared optimizer and its cursor. */
  struct OptimizerState {
    optim::OptimizerAdapter* optimizer = nullptr;  ///< Borrowed exact adapter.
    std::string optimizer_id;  ///< Stable identity and canonical order key.
    std::vector<ParameterTopology>
        parameters;  ///< Disjoint parameter subset owned by this optimizer.
    std::uint64_t steps_at_prepare = 0;  ///< Adapter counter baseline.
    std::uint64_t completed_steps = 0;   ///< Strategy-owned updates after bind.
  };

  SingleDeviceStrategyOptions options;  ///< Immutable runtime limits.
  StrategyDescriptor descriptor;        ///< Immutable strategy identity.
  nn::ModelAdapter* model = nullptr;    ///< Borrowed prepared model adapter.
  std::vector<OptimizerState>
      optimizers;  ///< Canonical disjoint optimizer execution domain.
  std::uint64_t completed_backward_calls = 0;  ///< Successful backward count.
  bool prepared = false;  ///< Whether topology and placement committed.
  bool closed = false;    ///< Whether explicit finalization completed.
};

namespace {

template <typename ImplType>
std::vector<std::byte> encode_topology(const ImplType& impl) {
  std::vector<std::byte> output;
  append_string(output, kTopologyMagic);
  append_u32(output, impl.descriptor.schema_version);
  append_string(output, impl.descriptor.type_id);
  append_string(output, impl.descriptor.strategy_id);
  append_string(output, impl.descriptor.device);
  append_u32(output, impl.descriptor.world_size);
  append_u32(output, impl.descriptor.rank);
  append_u8(output, impl.options.non_blocking_transfer ? 1U : 0U);
  append_u64(output, static_cast<std::uint64_t>(impl.optimizers.size()));
  for (const auto& optimizer : impl.optimizers) {
    append_string(output, optimizer.optimizer_id);
    append_u64(output, static_cast<std::uint64_t>(optimizer.parameters.size()));
    for (const auto& parameter : optimizer.parameters) {
      append_string(output, parameter.parameter_id);
      append_string(output, parameter.device);
      append_i64(output, parameter.scalar_type);
      append_u64(output, static_cast<std::uint64_t>(parameter.shape.size()));
      for (const auto extent : parameter.shape) {
        append_i64(output, extent);
      }
    }
  }
  return output;
}

template <typename ImplType>
std::vector<std::byte> encode_control(const ImplType& impl) {
  std::vector<std::byte> output;
  append_string(output, kControlMagic);
  append_u8(output, impl.closed ? 1U : 0U);
  append_u64(output, impl.completed_backward_calls);
  append_u64(output, static_cast<std::uint64_t>(impl.optimizers.size()));
  for (const auto& optimizer : impl.optimizers) {
    append_u64(output, optimizer.steps_at_prepare);
    append_u64(output, optimizer.completed_steps);
  }
  return output;
}

template <typename OptimizerState>
void verify_optimizer_cursor(const OptimizerState& state,
                             std::string operation) {
  if (state.steps_at_prepare >
      std::numeric_limits<std::uint64_t>::max() - state.completed_steps) {
    throw InternalError(
        "Strategy optimizer counter overflow",
        ErrorDetails{.component = kComponent,
                     .operation = std::move(operation),
                     .expected = "representable optimizer step total",
                     .actual = "counter overflow",
                     .hint = "Start a new run before uint64 exhaustion."});
  }
  const auto expected = state.steps_at_prepare + state.completed_steps;
  const auto actual = state.optimizer->completed_steps();
  if (actual != expected) {
    throw CompatibilityError(
        "Optimizer was updated outside the prepared strategy",
        ErrorDetails{.component = kComponent,
                     .operation = std::move(operation),
                     .expected = std::to_string(expected),
                     .actual = std::to_string(actual),
                     .hint = "Route every optimizer update through the active "
                             "TrainingStrategy."});
  }
}

template <typename ImplType>
void verify_all_optimizer_cursors(const ImplType& impl,
                                  std::string_view operation) {
  for (const auto& optimizer : impl.optimizers) {
    verify_optimizer_cursor(
        optimizer, std::string(operation) + "." + optimizer.optimizer_id);
  }
}

}  // namespace

SingleDeviceStrategy::SingleDeviceStrategy(SingleDeviceStrategyOptions options)
    : impl_(std::make_unique<Impl>()) {
  if constexpr (std::endian::native != std::endian::little) {
    throw NotImplemented(
        "Single-device strategy checkpoint byte order is unsupported",
        ErrorDetails{.component = kComponent,
                     .operation = "platform.endian",
                     .expected = "little endian",
                     .actual = "non-little endian",
                     .hint = "Add an explicit byte-swap codec."});
  }
  validate_options(options);
  impl_->descriptor = {.schema_version = 2,
                       .type_id = "jyppx.strategy.single-device/v2",
                       .strategy_id = options.strategy_id,
                       .device = options.device.str(),
                       .world_size = 1,
                       .rank = 0};
  impl_->options = std::move(options);
}

SingleDeviceStrategy::~SingleDeviceStrategy() = default;

const StrategyDescriptor& SingleDeviceStrategy::descriptor() const noexcept {
  return impl_->descriptor;
}

void SingleDeviceStrategy::prepare(
    nn::ModelAdapter& model,
    std::span<optim::OptimizerAdapter* const> optimizers) {
  if (impl_->closed) {
    throw NotReady(
        "Cannot prepare a closed single-device strategy",
        ErrorDetails{.component = kComponent,
                     .operation = "prepare",
                     .expected = "an open strategy",
                     .actual = "closed",
                     .hint = "Create a new strategy for another run."});
  }
  if (optimizers.empty() || optimizers.size() > impl_->options.max_optimizers) {
    throw_invalid("optimizers",
                  "one to " + std::to_string(impl_->options.max_optimizers) +
                      " optimizer adapters",
                  std::to_string(optimizers.size()),
                  "Bind a bounded multi-optimizer execution domain.");
  }

  std::vector<optim::OptimizerAdapter*> canonical(optimizers.begin(),
                                                  optimizers.end());
  if (std::any_of(canonical.begin(), canonical.end(),
                  [](const auto* optimizer) { return optimizer == nullptr; })) {
    throw_invalid("optimizers", "non-null optimizer adapters", "null",
                  "Remove null entries before strategy preparation.");
  }
  std::sort(canonical.begin(), canonical.end(),
            [](const auto* left, const auto* right) {
              return left->optimizer_id() < right->optimizer_id();
            });
  if (impl_->prepared) {
    const bool same_domain =
        impl_->model == &model &&
        impl_->optimizers.size() == canonical.size() &&
        std::equal(impl_->optimizers.begin(), impl_->optimizers.end(),
                   canonical.begin(), [](const auto& state, const auto* value) {
                     return state.optimizer == value &&
                            state.optimizer_id == value->optimizer_id();
                   });
    if (!same_domain) {
      throw_incompatible("prepare", "the originally prepared adapters",
                         "different model or optimizer set",
                         "Do not rebind a live strategy to new state.");
    }
    return;
  }
  if (impl_->options.device.is_cuda()) {
    const auto index = impl_->options.device.index();
    if (!torch::cuda::is_available() || index < 0 ||
        index >= torch::cuda::device_count()) {
      throw DeviceError(
          "Configured CUDA device is unavailable",
          ErrorDetails{.component = kComponent,
                       .operation = "prepare.device",
                       .expected = impl_->options.device.str(),
                       .actual = "unavailable",
                       .hint = "Select a visible CUDA index or use cpu."});
    }
  }

  const auto named_parameters = model.state().named_parameters();
  std::map<std::string, Tensor> trainable;
  for (const auto& item : named_parameters) {
    if (item.value().requires_grad()) {
      trainable.emplace(item.key(), item.value());
    }
  }

  std::vector<Impl::OptimizerState> optimizer_states;
  optimizer_states.reserve(canonical.size());
  std::set<std::string> optimizer_ids;
  std::set<std::string> binding_ids;
  std::size_t parameter_count = 0;
  for (auto* optimizer : canonical) {
    const auto& optimizer_id = optimizer->optimizer_id();
    if (!is_identifier(optimizer_id) ||
        optimizer_id.size() > impl_->options.max_string_bytes ||
        !optimizer_ids.insert(optimizer_id).second) {
      throw_invalid("optimizer.optimizer_id",
                    "unique bounded stable optimizer identities", optimizer_id,
                    "Use a distinct descriptor-backed optimizer ID.");
    }
    const auto bindings = optimizer->parameter_bindings();
    if (bindings.empty() || bindings.size() != optimizer->parameter_count() ||
        bindings.size() > impl_->options.max_parameters ||
        parameter_count > impl_->options.max_parameters - bindings.size()) {
      throw_invalid("optimizer.parameters",
                    "a bounded nonempty stable parameter subset",
                    std::to_string(bindings.size()),
                    "Bind each optimizer to a disjoint model subset.");
    }
    parameter_count += bindings.size();
    Impl::OptimizerState state{.optimizer = optimizer,
                               .optimizer_id = optimizer_id,
                               .parameters = {},
                               .steps_at_prepare = optimizer->completed_steps(),
                               .completed_steps = 0};
    state.parameters.reserve(bindings.size());
    for (const auto& binding : bindings) {
      if (!is_identifier(binding.parameter_id) ||
          binding.parameter_id.size() > impl_->options.max_string_bytes ||
          !binding_ids.insert(binding.parameter_id).second) {
        throw_invalid("optimizer.parameter_id",
                      "globally unique bounded named model parameters",
                      binding.parameter_id,
                      "Do not bind one parameter to multiple optimizers.");
      }
      const auto expected = trainable.find(binding.parameter_id);
      if (expected == trainable.end() || !binding.parameter.defined() ||
          binding.parameter.unsafeGetTensorImpl() !=
              expected->second.unsafeGetTensorImpl()) {
        throw_incompatible(
            "prepare.parameter_binding", binding.parameter_id,
            "missing or different tensor",
            "Bind optimizer IDs and handles from named_parameters().");
      }
      state.parameters.push_back({.parameter_id = binding.parameter_id,
                                  .device = impl_->options.device.str(),
                                  .scalar_type = static_cast<std::int64_t>(
                                      binding.parameter.scalar_type()),
                                  .shape = binding.parameter.sizes().vec()});
    }
    optimizer_states.push_back(std::move(state));
  }
  if (parameter_count != trainable.size()) {
    throw_incompatible("prepare.parameter_count",
                       std::to_string(trainable.size()),
                       std::to_string(parameter_count),
                       "Cover each trainable model parameter exactly once.");
  }

  Impl candidate = *impl_;
  candidate.model = &model;
  candidate.optimizers = std::move(optimizer_states);
  candidate.prepared = true;
  const auto encoded = encode_topology(candidate);
  if (encoded.size() > impl_->options.max_topology_bytes) {
    throw_exhausted("topology",
                    std::to_string(impl_->options.max_topology_bytes),
                    std::to_string(encoded.size()),
                    "Reduce parameter topology or raise the explicit limit.");
  }

  try {
    model.state().to(impl_->options.device,
                     impl_->options.non_blocking_transfer);
  } catch (const c10::Error& error) {
    detail::rethrow_backend_error(
        error, {.component = kComponent,
                .operation = "prepare.model_to",
                .failure_kind = detail::BackendFailureKind::device,
                .expected = impl_->options.device.str(),
                .hint = "Verify backend availability and device memory."});
  }
  for (auto& optimizer : candidate.optimizers) {
    optimizer.optimizer->synchronize_parameter_placement();
    for (const auto& binding : optimizer.optimizer->parameter_bindings()) {
      if (binding.parameter.device() != impl_->options.device) {
        throw DeviceError(
            "Optimizer parameter did not move with its model",
            ErrorDetails{.component = kComponent,
                         .operation = "prepare.parameter_device",
                         .expected = impl_->options.device.str(),
                         .actual = binding.parameter.device().str(),
                         .hint = "Construct each optimizer from model-owned "
                                 "parameter handles before prepare()."});
      }
    }
  }
  for (const auto& buffer : model.state().buffers()) {
    if (buffer.defined() && buffer.device() != impl_->options.device) {
      throw DeviceError(
          "Model buffer did not reach the configured device",
          ErrorDetails{.component = kComponent,
                       .operation = "prepare.buffer_device",
                       .expected = impl_->options.device.str(),
                       .actual = buffer.device().str(),
                       .hint = "Use a module whose buffers support device "
                               "placement."});
    }
  }

  impl_->model = candidate.model;
  impl_->optimizers = std::move(candidate.optimizers);
  impl_->prepared = true;
}

void SingleDeviceStrategy::backward(const Tensor& loss) {
  ensure_ready(impl_->prepared, impl_->closed, "backward");
  if (!loss.defined() || loss.dim() != 0 || !loss.is_floating_point() ||
      !loss.requires_grad() || loss.device() != impl_->options.device) {
    throw_invalid("backward.loss",
                  "a defined differentiable floating scalar on " +
                      impl_->options.device.str(),
                  loss.defined() ? loss.device().str() : "undefined",
                  "Return a scalar training loss from the step function.");
  }
  if (impl_->completed_backward_calls ==
      std::numeric_limits<std::uint64_t>::max()) {
    throw_exhausted("backward.count", "below uint64 maximum", "uint64 maximum",
                    "Start a new training run.");
  }
  try {
    if (!torch::isfinite(loss).item<bool>()) {
      throw NumericalError(
          "Training loss is not finite",
          ErrorDetails{.component = kComponent,
                       .operation = "backward.loss",
                       .expected = "finite scalar",
                       .actual = "NaN or infinity",
                       .hint = "Inspect model outputs and precision policy."});
    }
    loss.backward();
  } catch (const Error&) {
    throw;
  } catch (const c10::Error& error) {
    detail::rethrow_backend_error(
        error, {.component = kComponent,
                .operation = "backward",
                .failure_kind = detail::BackendFailureKind::unknown,
                .expected = "successful scalar reverse-mode differentiation",
                .hint = "Inspect the nested LibTorch autograd failure."});
  }
  ++impl_->completed_backward_calls;
}

void SingleDeviceStrategy::zero_grad(optim::OptimizerAdapter& optimizer,
                                     bool set_to_none) {
  ensure_ready(impl_->prepared, impl_->closed, "zero_grad");
  const auto prepared =
      std::find_if(impl_->optimizers.begin(), impl_->optimizers.end(),
                   [&optimizer](const auto& state) {
                     return state.optimizer == &optimizer;
                   });
  if (prepared == impl_->optimizers.end() ||
      optimizer.optimizer_id() != prepared->optimizer_id) {
    throw_incompatible("zero_grad.optimizer", "a prepared optimizer",
                       optimizer.optimizer_id(),
                       "Use the exact optimizer supplied to prepare().");
  }
  verify_optimizer_cursor(*prepared, "zero_grad.cursor");
  optimizer.zero_grad(set_to_none);
}

StrategyStepOutcome SingleDeviceStrategy::optimizer_step(
    optim::OptimizerAdapter& optimizer) {
  ensure_ready(impl_->prepared, impl_->closed, "optimizer_step");
  const auto prepared =
      std::find_if(impl_->optimizers.begin(), impl_->optimizers.end(),
                   [&optimizer](const auto& state) {
                     return state.optimizer == &optimizer;
                   });
  if (prepared == impl_->optimizers.end() ||
      optimizer.optimizer_id() != prepared->optimizer_id) {
    throw_incompatible("optimizer_step.optimizer", "a prepared optimizer",
                       optimizer.optimizer_id(),
                       "Use the exact optimizer supplied to prepare().");
  }
  verify_optimizer_cursor(*prepared, "optimizer_step.cursor");
  if (prepared->completed_steps == std::numeric_limits<std::uint64_t>::max()) {
    throw_exhausted("optimizer_step.count", "below uint64 maximum",
                    "uint64 maximum", "Start a new training run.");
  }
  const auto before = optimizer.completed_steps();
  static_cast<void>(optimizer.step());
  const auto after = optimizer.completed_steps();
  if (before == std::numeric_limits<std::uint64_t>::max() ||
      after != before + 1U) {
    throw InternalError(
        "Optimizer adapter violated single-step accounting",
        ErrorDetails{.component = kComponent,
                     .operation = "optimizer_step.commit",
                     .expected = std::to_string(before) + " + 1",
                     .actual = std::to_string(after),
                     .hint = "Fix the optimizer adapter before continuing."});
  }
  ++prepared->completed_steps;
  return {.status = StrategyStepStatus::applied,
          .optimizer_id = prepared->optimizer_id,
          .optimizer_completed_steps = after};
}

StrategyMetricValues SingleDeviceStrategy::reduce_metrics(
    const StrategyMetricValues& values) const {
  ensure_ready(impl_->prepared, impl_->closed, "reduce_metrics");
  if (values.size() > impl_->options.max_metrics) {
    throw_exhausted("metrics", std::to_string(impl_->options.max_metrics),
                    std::to_string(values.size()),
                    "Reduce observation count or raise the explicit limit.");
  }
  for (const auto& [name, value] : values) {
    if (!is_identifier(name) || name.size() > impl_->options.max_string_bytes ||
        !std::isfinite(value)) {
      throw_invalid("metrics", "bounded names with finite values", name,
                    "Validate observations before strategy reduction.");
    }
  }
  return values;
}

void SingleDeviceStrategy::barrier(StrategyBarrierReason reason,
                                   std::stop_token stop) const {
  ensure_ready(impl_->prepared, impl_->closed, "barrier");
  if (strategy_barrier_reason_name(reason) == "unknown") {
    throw_invalid("barrier.reason", "a known strategy barrier", "unknown",
                  "Use a stable StrategyBarrierReason value.");
  }
  if (stop.stop_requested()) {
    throw Cancelled("Single-device barrier was cancelled",
                    ErrorDetails{.component = kComponent,
                                 .operation = std::string(
                                     strategy_barrier_reason_name(reason)),
                                 .expected = "operation allowed to continue",
                                 .actual = "stop requested",
                                 .hint = "Finalize the run at the current safe "
                                         "boundary."});
  }
}

void SingleDeviceStrategy::close() {
  if (impl_->closed) {
    return;
  }
  if (impl_->prepared) {
    verify_all_optimizer_cursors(*impl_, "close.cursor");
  }
  impl_->closed = true;
}

bool SingleDeviceStrategy::prepared() const noexcept { return impl_->prepared; }

bool SingleDeviceStrategy::closed() const noexcept { return impl_->closed; }

std::uint64_t SingleDeviceStrategy::completed_backward_calls() const noexcept {
  return impl_->completed_backward_calls;
}

std::uint64_t SingleDeviceStrategy::completed_optimizer_steps(
    std::string_view optimizer_id) const {
  const auto optimizer =
      std::find_if(impl_->optimizers.begin(), impl_->optimizers.end(),
                   [optimizer_id](const auto& state) {
                     return state.optimizer_id == optimizer_id;
                   });
  if (optimizer == impl_->optimizers.end()) {
    throw_invalid("completed_optimizer_steps.optimizer_id",
                  "a prepared optimizer identity", std::string(optimizer_id),
                  "Query an optimizer supplied to prepare().");
  }
  return optimizer->completed_steps;
}

const SingleDeviceStrategyOptions& SingleDeviceStrategy::options()
    const noexcept {
  return impl_->options;
}

std::string SingleDeviceStrategy::participant_id() const {
  return impl_->descriptor.strategy_id;
}

StateManifest SingleDeviceStrategy::describe_state() const {
  return {.participant_id = impl_->descriptor.strategy_id,
          .schema_version = 2,
          .restore_level = RestoreLevel::exact,
          .fields = {{.name = kTopologyField,
                      .media_type = kTopologyMediaType,
                      .max_bytes = impl_->options.max_topology_bytes},
                     {.name = kControlField,
                      .media_type = kControlMediaType,
                      .max_bytes = impl_->options.max_control_bytes}},
          .degradation_reasons = {}};
}

void SingleDeviceStrategy::save_state(StateWriter& writer) const {
  ensure_ready(impl_->prepared, impl_->closed, "save_state");
  verify_all_optimizer_cursors(*impl_, "save_state.cursor");
  const auto topology = encode_topology(*impl_);
  const auto control = encode_control(*impl_);
  if (topology.size() > impl_->options.max_topology_bytes ||
      control.size() > impl_->options.max_control_bytes) {
    throw_exhausted("save_state", "state within configured limits",
                    std::to_string(topology.size() + control.size()),
                    "Reduce topology or raise checkpoint limits.");
  }
  writer.write_blob(kTopologyField, std::span<const std::byte>(
                                        topology.data(), topology.size()));
  writer.write_blob(kControlField,
                    std::span<const std::byte>(control.data(), control.size()));
}

void SingleDeviceStrategy::load_state(const StateReader& reader,
                                      RestorePolicy policy) {
  if (!impl_->prepared || impl_->closed) {
    throw NotReady(
        "Strategy must be prepared and open before restore",
        ErrorDetails{.component = kComponent,
                     .operation = "load_state",
                     .expected = "matching prepared open adapters",
                     .actual = impl_->closed ? "closed" : "unprepared",
                     .hint = "Prepare model and optimizer before the "
                             "checkpoint coordinator restores state."});
  }
  const auto expected_manifest = describe_state();
  validate_manifest(reader.manifest(), expected_manifest, policy);
  const auto stored_topology =
      reader.read_blob(kTopologyField, impl_->options.max_topology_bytes);
  const auto expected_topology = encode_topology(*impl_);
  if (stored_topology.size() != expected_topology.size() ||
      !std::equal(stored_topology.begin(), stored_topology.end(),
                  expected_topology.begin())) {
    throw_incompatible("topology", "identical strategy and parameter topology",
                       "different",
                       "Restore with the same model, optimizer, and device.");
  }
  const auto stored_control =
      reader.read_blob(kControlField, impl_->options.max_control_bytes);
  ByteReader control(stored_control);
  if (control.read_string("control.magic", kControlMagic.size()) !=
      kControlMagic) {
    throw_serialization("control.magic", std::string(kControlMagic),
                        "different", "Reject unknown strategy state encoding.");
  }
  const auto closed = control.read_u8("control.closed");
  const auto backward_calls = control.read_u64("control.backward_calls");
  const auto optimizer_count = control.read_u64("control.optimizer_count");
  if (optimizer_count != impl_->optimizers.size()) {
    throw_serialization("control.optimizer_count",
                        std::to_string(impl_->optimizers.size()),
                        std::to_string(optimizer_count),
                        "Reject state for a different optimizer topology.");
  }
  std::vector<std::pair<std::uint64_t, std::uint64_t>> cursors;
  cursors.reserve(impl_->optimizers.size());
  for (std::size_t index = 0; index < impl_->optimizers.size(); ++index) {
    const auto baseline = control.read_u64("control.optimizer_baseline");
    const auto completed =
        control.read_u64("control.optimizer_completed_steps");
    if (baseline > std::numeric_limits<std::uint64_t>::max() - completed) {
      throw_serialization("control.optimizer_cursor",
                          "representable optimizer counters", "overflow",
                          "Reject corrupt strategy counters.");
    }
    const auto expected = baseline + completed;
    const auto actual = impl_->optimizers[index].optimizer->completed_steps();
    const auto prepared_baseline = impl_->optimizers[index].steps_at_prepare;
    // A coordinator may load this strategy before its optimizer, including
    // reverse-order rollback after a later participant rejects restoration.
    // Accept the live strategy cursor or the candidate cursor; an unrelated
    // external optimizer advance still matches neither.
    const auto live_expected =
        prepared_baseline + impl_->optimizers[index].completed_steps;
    if (baseline != prepared_baseline ||
        (actual != live_expected && actual != expected)) {
      throw_incompatible(
          "control.optimizer_steps." + impl_->optimizers[index].optimizer_id,
          std::to_string(live_expected) + " before optimizer restore or " +
              std::to_string(expected) + " after it",
          std::to_string(actual),
          "Prepare the same fresh optimizer domain and restore all checkpoint "
          "participants together.");
    }
    cursors.emplace_back(baseline, completed);
  }
  control.require_finished("control");
  if (closed != 0U) {
    throw_serialization(
        "control.closed", "zero for a restorable open strategy",
        std::to_string(closed),
        "Reject closed or invalid lifecycle state; closed strategies cannot "
        "be checkpointed.");
  }

  impl_->closed = false;
  impl_->completed_backward_calls = backward_calls;
  for (std::size_t index = 0; index < impl_->optimizers.size(); ++index) {
    impl_->optimizers[index].completed_steps = cursors[index].second;
  }
}

}  // namespace jyppx::liuding::training
