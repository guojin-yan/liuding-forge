// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/random_checkpoint_participant.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/lr_scheduler.hpp>
#include <jyppx/liuding/optim/optimizer_adapter.hpp>
#include <jyppx/liuding/training/amp/autocast.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>
#include <jyppx/liuding/training/amp/grad_scaler_checkpoint_participant.hpp>
#include <jyppx/liuding/training/callback_dispatcher.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/event_sink.hpp>
#include <jyppx/liuding/training/scheduler_adapter.hpp>
#include <jyppx/liuding/training/strategy.hpp>
#include <jyppx/liuding/training/trainer.hpp>

#include <torch/nn/utils.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::liuding::training {
namespace {

constexpr char kComponent[] = "training.trainer";
constexpr char kTopologyField[] = "topology";
constexpr char kControlField[] = "control";
constexpr char kTopologyMediaType[] =
    "application/vnd.jyppx.trainer-topology-v3";
constexpr char kControlMediaType[] = "application/vnd.jyppx.trainer-control-v3";
constexpr std::string_view kTopologyMagic = "JYPPX-TRAINER-TOPOLOGY-V3";
constexpr std::string_view kControlMagic = "JYPPX-TRAINER-CONTROL-V3";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid Trainer contract",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_not_ready(std::string operation, std::string expected,
                                  std::string actual, std::string hint) {
  throw NotReady("Trainer is not at the required lifecycle boundary",
                 ErrorDetails{.component = kComponent,
                              .operation = std::move(operation),
                              .expected = std::move(expected),
                              .actual = std::move(actual),
                              .hint = std::move(hint)});
}

[[noreturn]] void throw_exhausted(std::string operation, std::string expected,
                                  std::string actual, std::string hint) {
  throw ResourceExhausted("Trainer resource limit exceeded",
                          ErrorDetails{.component = kComponent,
                                       .operation = std::move(operation),
                                       .expected = std::move(expected),
                                       .actual = std::move(actual),
                                       .hint = std::move(hint)});
}

[[noreturn]] void throw_incompatible(std::string operation,
                                     std::string expected, std::string actual,
                                     std::string hint) {
  throw CompatibilityError("Trainer checkpoint is incompatible",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Malformed Trainer checkpoint state",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_numerical(std::string operation, std::string actual) {
  throw NumericalError(
      "Trainer observed a non-finite numeric value",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = "finite loss, metrics, and gradients",
                   .actual = std::move(actual),
                   .hint = "Inspect the task loss, inputs, and optimizer "
                           "configuration before retrying."});
}

bool is_identifier(std::string_view value) {
  if (value.empty()) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    const auto byte = static_cast<unsigned char>(character);
    return (byte >= static_cast<unsigned char>('a') &&
            byte <= static_cast<unsigned char>('z')) ||
           (byte >= static_cast<unsigned char>('A') &&
            byte <= static_cast<unsigned char>('Z')) ||
           (byte >= static_cast<unsigned char>('0') &&
            byte <= static_cast<unsigned char>('9')) ||
           byte == static_cast<unsigned char>('.') ||
           byte == static_cast<unsigned char>('_') ||
           byte == static_cast<unsigned char>('-');
  });
}

std::string phase_name(TrainerPhase phase) {
  switch (phase) {
    case TrainerPhase::created:
      return "created";
    case TrainerPhase::prepared:
      return "prepared";
    case TrainerPhase::running:
      return "running";
    case TrainerPhase::completed:
      return "completed";
    case TrainerPhase::cancelled:
      return "cancelled";
    case TrainerPhase::failed:
      return "failed";
    case TrainerPhase::closed:
      return "closed";
  }
  return "unknown";
}

bool valid_tail_policy(AccumulationTailPolicy policy) {
  return policy == AccumulationTailPolicy::flush ||
         policy == AccumulationTailPolicy::drop;
}

bool valid_precision_mode(PrecisionMode mode) {
  return mode == PrecisionMode::full || mode == PrecisionMode::cpu_bfloat16 ||
         mode == PrecisionMode::cuda_float16 ||
         mode == PrecisionMode::cuda_bfloat16;
}

bool is_cuda_device(std::string_view device) {
  return device == "cuda" || device.starts_with("cuda:");
}

bool is_cpu_device(std::string_view device) {
  return device == "cpu" || device.starts_with("cpu:");
}

bool restorable_phase(TrainerPhase phase) {
  return phase == TrainerPhase::prepared || phase == TrainerPhase::running ||
         phase == TrainerPhase::completed || phase == TrainerPhase::cancelled;
}

void validate_manifest(const StateManifest& stored,
                       const StateManifest& expected, RestorePolicy policy) {
  validate_state_manifest(stored);
  if (!restore_policy_accepts(policy, stored.restore_level)) {
    throw_incompatible("manifest.restore_level",
                       std::string(restore_policy_name(policy)),
                       std::string(restore_level_name(stored.restore_level)),
                       "Use a policy that accepts the stored guarantee.");
  }
  if (stored.participant_id != expected.participant_id ||
      stored.schema_version != expected.schema_version ||
      stored.restore_level != RestoreLevel::exact ||
      stored.fields.size() != expected.fields.size()) {
    throw_incompatible("manifest", expected.participant_id,
                       stored.participant_id,
                       "Restore only current state for the same Trainer.");
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
                         "Reject a changed Trainer field contract.");
    }
  }
}

void append_u8(std::vector<std::byte>& output, std::uint8_t value) {
  output.push_back(static_cast<std::byte>(value));
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    output.push_back(static_cast<std::byte>((value >> (index * 8U)) & 0xffU));
  }
}

void append_double(std::vector<std::byte>& output, double value) {
  append_u64(output, std::bit_cast<std::uint64_t>(value));
}

void append_string(std::vector<std::byte>& output, std::string_view value) {
  append_u64(output, static_cast<std::uint64_t>(value.size()));
  for (const auto character : value) {
    output.push_back(
        static_cast<std::byte>(static_cast<unsigned char>(character)));
  }
}

void append_metrics(std::vector<std::byte>& output,
                    const std::map<std::string, double>& values) {
  append_u64(output, static_cast<std::uint64_t>(values.size()));
  for (const auto& [name, value] : values) {
    append_string(output, name);
    append_double(output, value);
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
                   std::to_integer<std::uint8_t>(bytes_[offset_++]))
               << (index * 8U);
    }
    return value;
  }

  double read_double(std::string_view field) {
    return std::bit_cast<double>(read_u64(field));
  }

  std::string read_string(std::string_view field, std::size_t max_bytes) {
    const auto length = read_u64(field);
    if (length > max_bytes || length > remaining()) {
      throw_serialization(std::string(field),
                          "a bounded string within the remaining field",
                          std::to_string(length),
                          "Reject truncated or oversized Trainer state.");
    }
    if (length == 0U) {
      return {};
    }
    const auto start = bytes_.begin() + static_cast<std::ptrdiff_t>(offset_);
    offset_ += static_cast<std::size_t>(length);
    return {reinterpret_cast<const char*>(&*start),
            static_cast<std::size_t>(length)};
  }

  std::map<std::string, double> read_metrics(std::string_view field,
                                             std::size_t max_metrics,
                                             std::size_t max_name_bytes) {
    const auto count = read_u64(field);
    if (count > max_metrics) {
      throw_exhausted(std::string(field), std::to_string(max_metrics),
                      std::to_string(count),
                      "Reduce retained Trainer metric cardinality.");
    }
    std::map<std::string, double> result;
    for (std::uint64_t index = 0; index < count; ++index) {
      auto name = read_string(std::string(field) + ".name", max_name_bytes);
      const auto value = read_double(std::string(field) + ".value");
      if (!is_identifier(name) || !std::isfinite(value) ||
          !result.emplace(std::move(name), value).second) {
        throw_serialization(std::string(field),
                            "unique bounded names with finite values",
                            "invalid metric entry",
                            "Reject ambiguous Trainer aggregate state.");
      }
    }
    return result;
  }

  void expect_magic(std::string_view expected, std::size_t max_bytes,
                    std::string_view field) {
    const auto actual = read_string(field, max_bytes);
    if (actual != expected) {
      throw_serialization(std::string(field), std::string(expected), actual,
                          "Use only the current Trainer state schema.");
    }
  }

  void finish(std::string_view field) const {
    if (offset_ != bytes_.size()) {
      throw_serialization(std::string(field), "no trailing bytes",
                          std::to_string(bytes_.size() - offset_),
                          "Reject ambiguous state extensions without a new "
                          "schema version.");
    }
  }

 private:
  void require(std::size_t count, std::string_view field) const {
    if (count > remaining()) {
      throw_serialization(
          std::string(field), std::to_string(count) + " available bytes",
          std::to_string(remaining()), "Reject truncated Trainer state.");
    }
  }

  std::size_t remaining() const noexcept { return bytes_.size() - offset_; }

  std::span<const std::byte> bytes_;  ///< Borrowed bounded field bytes.
  std::size_t offset_ = 0;            ///< Next unread byte offset.
};

struct MetricAccumulator {
  std::map<std::string, double> weighted_sums;  ///< Weighted finite totals.
  std::uint64_t sample_count = 0;  ///< Observations represented by totals.

  void clear() {
    weighted_sums.clear();
    sample_count = 0;
  }
};

struct TrainerControlState {
  TrainerPhase phase = TrainerPhase::created;
  bool run_started = false;
  bool epoch_in_progress = false;
  bool validation_active = false;
  bool checkpoint_in_progress = false;
  bool callback_stop = false;
  bool skip_validation = false;
  bool checkpoint_pending = false;
  std::uint64_t active_epoch = 0;
  std::uint64_t next_epoch = 0;
  std::uint64_t completed_epochs = 0;
  std::uint64_t global_step = 0;
  std::map<std::string, std::uint64_t>
      scheduler_events;  ///< Per-scheduler committed event indices.
  std::uint64_t checkpoint_count = 0;
  std::size_t pending_micro_steps = 0;
  std::string stop_reason;
  MetricAccumulator training;
  std::map<std::string, double> last_training;
  std::map<std::string, double> last_validation;
};

void validate_options(const TrainerOptions& options) {
  const auto valid_identity = [&](std::string_view value) {
    return is_identifier(value) && value.size() <= options.max_identifier_bytes;
  };
  if (options.max_identifier_bytes == 0U || options.max_topology_bytes == 0U ||
      options.max_control_bytes == 0U || options.max_metrics == 0U ||
      options.max_optimizers == 0U ||
      options.checkpoint.max_participants == 0U ||
      options.checkpoint.max_rollback_bytes == 0U) {
    throw_invalid("options.limits", "positive resource limits", "zero",
                  "Configure explicit positive Trainer limits.");
  }
  if (!valid_identity(options.trainer_id) || !valid_identity(options.run_id) ||
      options.trace_id.empty() ||
      options.trace_id.size() > options.max_identifier_bytes ||
      !valid_identity(options.scheduler_metric)) {
    throw_invalid("options.identity", "bounded stable identities",
                  options.trainer_id + "/" + options.run_id,
                  "Use alphanumeric, dot, underscore, or dash identities.");
  }
  if (options.max_epochs == 0U || options.accumulate_steps == 0U ||
      !valid_tail_policy(options.accumulation_tail) ||
      !valid_precision_mode(options.precision)) {
    throw_invalid("options.loop", "positive epochs/accumulation and known tail",
                  std::to_string(options.max_epochs) + "/" +
                      std::to_string(options.accumulate_steps),
                  "Select a finite epoch limit and an explicit tail policy.");
  }
  if (options.gradient_clip_norm.has_value() &&
      (!std::isfinite(*options.gradient_clip_norm) ||
       *options.gradient_clip_norm <= 0.0)) {
    throw_invalid("options.gradient_clip_norm", "a positive finite norm",
                  std::to_string(*options.gradient_clip_norm),
                  "Unset clipping or provide a positive threshold.");
  }
}

void validate_metric_map(const std::map<std::string, double>& metrics,
                         const TrainerOptions& options,
                         std::string_view operation) {
  if (metrics.size() > options.max_metrics) {
    throw_exhausted(std::string(operation), std::to_string(options.max_metrics),
                    std::to_string(metrics.size()),
                    "Reduce task metric cardinality.");
  }
  for (const auto& [name, value] : metrics) {
    if (!is_identifier(name) || name.size() > options.max_identifier_bytes) {
      throw_invalid(std::string(operation), "bounded stable metric names", name,
                    "Use stable machine-readable metric identities.");
    }
    if (!std::isfinite(value)) {
      throw_numerical(std::string(operation), name);
    }
  }
}

void checked_increment(std::uint64_t& value, std::string_view operation) {
  if (value == std::numeric_limits<std::uint64_t>::max()) {
    throw_exhausted(std::string(operation), "a counter below uint64 maximum",
                    "uint64 maximum", "Start a new bounded training run.");
  }
  ++value;
}

void add_output(MetricAccumulator& accumulator, const StepOutput& output,
                std::optional<double> loss, std::string_view loss_name,
                const TrainerOptions& options) {
  validate_metric_map(output.metrics, options, "step.metrics");
  if (output.sample_count == 0U ||
      (loss.has_value() && !std::isfinite(*loss))) {
    if (loss.has_value() && !std::isfinite(*loss)) {
      throw_numerical("step.loss", "non-finite scalar");
    }
    throw_invalid("step.sample_count", "a positive observation count", "0",
                  "Return the batch cardinality from the step function.");
  }
  if (accumulator.sample_count >
      std::numeric_limits<std::uint64_t>::max() - output.sample_count) {
    throw_exhausted("step.sample_count", "a bounded cumulative count",
                    "uint64 overflow", "Use a shorter run or smaller epoch.");
  }
  auto add_value = [&](std::string name, double value) {
    const auto weighted = value * static_cast<double>(output.sample_count);
    auto& total = accumulator.weighted_sums[std::move(name)];
    total += weighted;
    if (!std::isfinite(total)) {
      throw_numerical("step.metric_accumulation", "overflowed aggregate");
    }
  };
  if (loss.has_value()) {
    add_value(std::string(loss_name), *loss);
  }
  for (const auto& [name, value] : output.metrics) {
    if (name == loss_name) {
      throw_invalid("step.metrics", "task metrics distinct from loss identity",
                    name, "Let the Trainer own the phase loss observation.");
    }
    add_value(name, value);
  }
  if (accumulator.weighted_sums.size() > options.max_metrics) {
    throw_exhausted("step.metric_accumulation",
                    std::to_string(options.max_metrics),
                    std::to_string(accumulator.weighted_sums.size()),
                    "Reduce task metric cardinality.");
  }
  accumulator.sample_count += output.sample_count;
}

std::map<std::string, double> finish_metrics(
    const MetricAccumulator& accumulator) {
  std::map<std::string, double> result;
  if (accumulator.sample_count == 0U) {
    return result;
  }
  for (const auto& [name, total] : accumulator.weighted_sums) {
    result.emplace(name, total / static_cast<double>(accumulator.sample_count));
  }
  return result;
}

void validate_step_output(const StepOutput& output,
                          const data::BatchEnvelope& batch, bool require_loss) {
  if (output.sample_count == 0U ||
      output.sample_count != batch.sample_ids.size()) {
    throw_invalid("step.sample_count", "the committed batch sample count",
                  std::to_string(output.sample_count),
                  "Report exactly one observation per committed sample ID.");
  }
  if (require_loss && (!output.loss.defined() || output.loss.numel() != 1 ||
                       !output.loss.requires_grad())) {
    throw_invalid(
        "step.loss", "one defined differentiable scalar training loss",
        output.loss.defined() ? std::to_string(output.loss.numel()) + " values"
                              : "undefined",
        "Return the scalar objective used for backward.");
  }
  if (output.loss.defined()) {
    if (output.loss.numel() != 1) {
      throw_invalid("step.loss", "one scalar value",
                    std::to_string(output.loss.numel()),
                    "Reduce task loss before returning it.");
    }
    if (!std::isfinite(output.loss.detach().item<double>())) {
      throw_numerical("step.loss", "non-finite scalar");
    }
  }
}

std::vector<std::byte> encode_topology(
    const TrainerOptions& options, const FitRequest& request,
    const std::vector<std::string>& participant_ids) {
  std::vector<std::byte> output;
  append_string(output, kTopologyMagic);
  append_string(output, options.trainer_id);
  append_string(output, options.run_id);
  append_string(output, options.trace_id);
  append_u64(output, options.max_epochs);
  append_u64(output, options.max_steps);
  append_u64(output, static_cast<std::uint64_t>(options.accumulate_steps));
  append_u8(output, static_cast<std::uint8_t>(options.accumulation_tail));
  append_u8(output, static_cast<std::uint8_t>(options.precision));
  append_u8(output, options.gradient_clip_norm.has_value() ? 1U : 0U);
  append_double(output, options.gradient_clip_norm.value_or(0.0));
  append_string(output, options.scheduler_metric);
  append_u8(output, static_cast<std::uint8_t>(options.restore_policy));
  append_u8(output, options.drain_events ? 1U : 0U);
  append_u64(output, static_cast<std::uint64_t>(options.max_metrics));
  append_u64(output, static_cast<std::uint64_t>(options.max_optimizers));
  append_u64(output, static_cast<std::uint64_t>(options.max_identifier_bytes));
  append_string(output, request.training_data->participant_id());
  append_string(output, request.validation_data
                            ? request.validation_data->participant_id()
                            : std::string{});
  append_string(output, request.strategy->participant_id());
  append_string(output, request.callbacks->participant_id());
  append_string(output, request.event_outbox
                            ? request.event_outbox->participant_id()
                            : std::string{});
  append_string(output, request.model_checkpoint->participant_id());
  append_string(output, request.rng_checkpoint->participant_id());
  append_string(output, request.gradient_scaler
                            ? request.gradient_scaler->options().scaler_id
                            : std::string{});
  append_u64(output, static_cast<std::uint64_t>(request.optimizers.size()));
  for (const auto& binding : request.optimizers) {
    append_string(output, binding.optimizer->optimizer_id());
    append_string(output, binding.checkpoint->participant_id());
    append_string(output, binding.scheduler
                              ? binding.scheduler->participant_id()
                              : std::string{});
  }
  append_u64(output, static_cast<std::uint64_t>(participant_ids.size()));
  for (const auto& participant_id : participant_ids) {
    append_string(output, participant_id);
  }
  return output;
}

std::vector<std::byte> encode_control(const TrainerControlState& state) {
  std::vector<std::byte> output;
  append_string(output, kControlMagic);
  append_u8(output, static_cast<std::uint8_t>(state.phase));
  append_u8(output, state.run_started ? 1U : 0U);
  append_u8(output, state.epoch_in_progress ? 1U : 0U);
  append_u8(output, state.validation_active ? 1U : 0U);
  append_u8(output, state.checkpoint_in_progress ? 1U : 0U);
  append_u8(output, state.callback_stop ? 1U : 0U);
  append_u8(output, state.skip_validation ? 1U : 0U);
  append_u8(output, state.checkpoint_pending ? 1U : 0U);
  append_u64(output, state.active_epoch);
  append_u64(output, state.next_epoch);
  append_u64(output, state.completed_epochs);
  append_u64(output, state.global_step);
  append_u64(output, static_cast<std::uint64_t>(state.scheduler_events.size()));
  for (const auto& [scheduler_id, event_index] : state.scheduler_events) {
    append_string(output, scheduler_id);
    append_u64(output, event_index);
  }
  append_u64(output, state.checkpoint_count);
  append_u64(output, static_cast<std::uint64_t>(state.pending_micro_steps));
  append_string(output, state.stop_reason);
  append_u64(output, state.training.sample_count);
  append_metrics(output, state.training.weighted_sums);
  append_metrics(output, state.last_training);
  append_metrics(output, state.last_validation);
  return output;
}

TrainerControlState decode_control(std::span<const std::byte> bytes,
                                   const TrainerOptions& options) {
  ByteReader reader(bytes);
  reader.expect_magic(kControlMagic, options.max_identifier_bytes,
                      "control.magic");
  TrainerControlState result;
  result.phase = static_cast<TrainerPhase>(reader.read_u8("control.phase"));
  auto read_bool = [&](std::string_view field) {
    const auto value = reader.read_u8(field);
    if (value > 1U) {
      throw_serialization(std::string(field), "0 or 1", std::to_string(value),
                          "Reject malformed Boolean state.");
    }
    return value == 1U;
  };
  result.run_started = read_bool("control.run_started");
  result.epoch_in_progress = read_bool("control.epoch_in_progress");
  result.validation_active = read_bool("control.validation_active");
  result.checkpoint_in_progress = read_bool("control.checkpoint_in_progress");
  result.callback_stop = read_bool("control.callback_stop");
  result.skip_validation = read_bool("control.skip_validation");
  result.checkpoint_pending = read_bool("control.checkpoint_pending");
  result.active_epoch = reader.read_u64("control.active_epoch");
  result.next_epoch = reader.read_u64("control.next_epoch");
  result.completed_epochs = reader.read_u64("control.completed_epochs");
  result.global_step = reader.read_u64("control.global_step");
  const auto scheduler_count = reader.read_u64("control.scheduler_event_count");
  if (scheduler_count > options.max_optimizers) {
    throw_serialization(
        "control.scheduler_event_count", std::to_string(options.max_optimizers),
        std::to_string(scheduler_count), "Reject oversized scheduler state.");
  }
  for (std::uint64_t index = 0; index < scheduler_count; ++index) {
    auto scheduler_id = reader.read_string("control.scheduler_id",
                                           options.max_identifier_bytes);
    const auto event_index = reader.read_u64("control.scheduler_event_index");
    if (!is_identifier(scheduler_id) || event_index == 0U ||
        !result.scheduler_events.emplace(std::move(scheduler_id), event_index)
             .second) {
      throw_serialization("control.scheduler_events",
                          "unique scheduler IDs with positive event indices",
                          "invalid entry",
                          "Reject malformed scheduler cursor state.");
    }
  }
  result.checkpoint_count = reader.read_u64("control.checkpoint_count");
  const auto pending = reader.read_u64("control.pending_micro_steps");
  if (pending > std::numeric_limits<std::size_t>::max()) {
    throw_serialization("control.pending_micro_steps", "a size_t value",
                        std::to_string(pending), "Reject foreign-width state.");
  }
  result.pending_micro_steps = static_cast<std::size_t>(pending);
  result.stop_reason =
      reader.read_string("control.stop_reason", options.max_identifier_bytes);
  result.training.sample_count = reader.read_u64("control.training.samples");
  result.training.weighted_sums =
      reader.read_metrics("control.training.sums", options.max_metrics,
                          options.max_identifier_bytes);
  result.last_training =
      reader.read_metrics("control.last_training", options.max_metrics,
                          options.max_identifier_bytes);
  result.last_validation =
      reader.read_metrics("control.last_validation", options.max_metrics,
                          options.max_identifier_bytes);
  reader.finish("control");
  return result;
}

}  // namespace

struct Trainer::Impl {
  explicit Impl(TrainerOptions trainer_options)
      : options(std::move(trainer_options)) {}

  TrainerOptions options;
  FitRequest request;
  bool prepared = false;
  TrainerControlState state;
  std::vector<std::string> participant_ids;
  std::unique_ptr<CheckpointParticipantRegistry> registry;
  std::unique_ptr<CheckpointCoordinator> coordinator;

  bool checkpoint_safe() const noexcept {
    return prepared && state.pending_micro_steps == 0U &&
           !state.validation_active && state.phase != TrainerPhase::created &&
           state.phase != TrainerPhase::failed &&
           state.phase != TrainerPhase::closed;
  }

  void accept_command(const CallbackDispatchResult& result,
                      bool accept_commands) {
    if (!accept_commands || !result.effective_command.has_value()) {
      return;
    }
    const auto& command = *result.effective_command;
    switch (command.kind) {
      case CallbackCommandKind::none:
        return;
      case CallbackCommandKind::skip_validation:
        state.skip_validation = true;
        return;
      case CallbackCommandKind::request_checkpoint:
        state.checkpoint_pending = true;
        return;
      case CallbackCommandKind::stop:
        state.callback_stop = true;
        state.stop_reason = command.callback_id + ":" + command.reason;
        return;
    }
  }

  CallbackDispatchResult emit(CallbackEventKind kind, CallbackPhase phase,
                              std::map<std::string, double> metrics,
                              std::stop_token stop, bool accept_commands = true,
                              bool drain = true) {
    validate_metric_map(metrics, options, "event.metrics");
    if (request.callbacks->last_event_index() ==
        std::numeric_limits<std::uint64_t>::max()) {
      throw_exhausted("event.index", "a value below uint64 maximum",
                      "uint64 maximum", "Start a new run.");
    }
    CallbackEvent event{
        .schema_version = 1,
        .kind = kind,
        .phase = phase,
        .event_index = request.callbacks->last_event_index() + 1U,
        .run_id = options.run_id,
        .trace_id = options.trace_id,
        .epoch = state.active_epoch,
        .global_step = state.global_step,
        .metrics = std::move(metrics)};
    auto result = request.event_outbox ? request.callbacks->dispatch(
                                             event, *request.event_outbox)
                                       : request.callbacks->dispatch(event);
    accept_command(result, accept_commands);
    if (request.event_outbox && options.drain_events && drain) {
      request.event_outbox->drain(stop);
    }
    return result;
  }

  void publish_checkpoint(std::stop_token stop) {
    if (!state.checkpoint_pending) {
      return;
    }
    if (!checkpoint_safe()) {
      return;
    }
    if (!request.checkpoint_action) {
      throw_not_ready(
          "checkpoint.action", "a configured CheckpointAction", "empty",
          "Provide an interop-owned checkpoint publication action.");
    }
    state.checkpoint_pending = false;
    state.checkpoint_in_progress = true;
    emit(CallbackEventKind::checkpoint_started, CallbackPhase::checkpointing,
         {}, stop, false);
    request.strategy->barrier(StrategyBarrierReason::checkpoint, stop);
    try {
      request.checkpoint_action(*coordinator);
    } catch (...) {
      const auto original = std::current_exception();
      state.checkpoint_in_progress = false;
      try {
        emit(CallbackEventKind::checkpoint_failed, CallbackPhase::checkpointing,
             {}, stop, false, false);
      } catch (...) {
      }
      std::rethrow_exception(original);
    }
    state.checkpoint_in_progress = false;
    checked_increment(state.checkpoint_count, "checkpoint.count");
    emit(CallbackEventKind::checkpoint_committed, CallbackPhase::checkpointing,
         {}, stop, false);
  }

  StepOutput invoke_step(const data::BatchEnvelope& batch,
                         const BatchMapper& mapper,
                         const TargetMapper& target_mapper,
                         const StepFunction& step, StepPhase phase) {
    try {
      auto invocation = mapper(batch);
      std::optional<c10::IValue> target;
      if (target_mapper) {
        target = target_mapper(batch);
      }
      std::unique_ptr<amp::AutocastGuard> autocast;
      if (options.precision == PrecisionMode::cpu_bfloat16) {
        autocast = std::make_unique<amp::AutocastGuard>(c10::DeviceType::CPU,
                                                        at::kBFloat16);
      } else if (options.precision == PrecisionMode::cuda_float16) {
        autocast = std::make_unique<amp::AutocastGuard>(c10::DeviceType::CUDA,
                                                        at::kHalf);
      } else if (options.precision == PrecisionMode::cuda_bfloat16) {
        autocast = std::make_unique<amp::AutocastGuard>(c10::DeviceType::CUDA,
                                                        at::kBFloat16);
      }
      const StepContext context{.phase = phase,
                                .epoch = state.active_epoch,
                                .global_step = state.global_step,
                                .batch_sequence = batch.sequence,
                                .micro_step = state.pending_micro_steps + 1U};
      return step(StepInput{.model = *request.model,
                            .batch = batch,
                            .invocation = invocation,
                            .target = target,
                            .context = context});
    } catch (const Error&) {
      throw;
    } catch (...) {
      std::throw_with_nested(BackendError(
          "Task step raised a non-JYPPX exception",
          ErrorDetails{.component = kComponent,
                       .operation = phase == StepPhase::training
                                        ? "training_step"
                                        : "validation_step",
                       .expected = "a valid StepOutput",
                       .actual = "user exception",
                       .hint = "Inspect the mapper and task step cause."}));
    }
  }

  void ensure_finite_gradients() const {
    for (const auto& optimizer : request.optimizers) {
      for (const auto& binding : optimizer.optimizer->parameter_bindings()) {
        const auto gradient = binding.parameter.grad();
        if (gradient.defined() &&
            !torch::isfinite(gradient).all().item<bool>()) {
          throw_numerical("optimizer.gradient", binding.parameter_id);
        }
      }
    }
  }

  void scale_gradients(double scale) const {
    torch::NoGradGuard guard;
    for (const auto& optimizer : request.optimizers) {
      for (const auto& binding : optimizer.optimizer->parameter_bindings()) {
        auto gradient = binding.parameter.grad();
        if (gradient.defined()) {
          gradient.mul_(scale);
        }
      }
    }
  }

  void clip_gradients() const {
    if (!options.gradient_clip_norm.has_value()) {
      return;
    }
    std::vector<Tensor> parameters;
    for (const auto& optimizer : request.optimizers) {
      for (const auto& binding : optimizer.optimizer->parameter_bindings()) {
        parameters.push_back(binding.parameter);
      }
    }
    torch::nn::utils::clip_grad_norm_(parameters, *options.gradient_clip_norm);
  }

  void zero_grad_all() {
    for (const auto& optimizer : request.optimizers) {
      request.strategy->zero_grad(*optimizer.optimizer);
    }
  }

  void step_schedulers(optim::SchedulerStepMode trigger,
                       std::optional<double> metric = std::nullopt) {
    for (const auto& optimizer : request.optimizers) {
      if (!optimizer.scheduler || optimizer.scheduler->mode() != trigger) {
        continue;
      }
      const auto scheduler_id = optimizer.scheduler->participant_id();
      auto [cursor, inserted] = state.scheduler_events.try_emplace(
          scheduler_id, static_cast<std::uint64_t>(0));
      static_cast<void>(inserted);
      checked_increment(cursor->second, "scheduler.events." + scheduler_id);
      optimizer.scheduler->step({.trigger = trigger,
                                 .event_index = cursor->second,
                                 .metric = metric});
    }
  }

  void optimizer_step(std::stop_token stop, bool partial_tail) {
    if (state.pending_micro_steps == 0U) {
      return;
    }
    if (partial_tail &&
        options.accumulation_tail == AccumulationTailPolicy::drop) {
      zero_grad_all();
      state.pending_micro_steps = 0;
      return;
    }
    emit(CallbackEventKind::optimizer_step_started, CallbackPhase::training, {},
         stop);
    if (partial_tail && state.pending_micro_steps < options.accumulate_steps) {
      scale_gradients(static_cast<double>(options.accumulate_steps) /
                      static_cast<double>(state.pending_micro_steps));
    }
    bool scaler_applied = true;
    if (request.gradient_scaler) {
      for (const auto& optimizer : request.optimizers) {
        request.gradient_scaler->unscale(optimizer.optimizer);
      }
      scaler_applied = request.gradient_scaler->cycle_allowed();
      if (scaler_applied) {
        ensure_finite_gradients();
        clip_gradients();
      }
    } else {
      ensure_finite_gradients();
      clip_gradients();
    }
    if (!scaler_applied) {
      for (const auto& optimizer : request.optimizers) {
        request.gradient_scaler->mark_step(optimizer.optimizer, false);
      }
      request.gradient_scaler->update();
      state.pending_micro_steps = 0;
      zero_grad_all();
      emit(
          CallbackEventKind::optimizer_step_finished, CallbackPhase::training,
          {{"optimizer.applied", 0.0},
           {"optimizer.count", static_cast<double>(request.optimizers.size())}},
          stop);
      return;
    }
    for (const auto& optimizer : request.optimizers) {
      const auto outcome =
          request.strategy->optimizer_step(*optimizer.optimizer);
      if (outcome.status != StrategyStepStatus::applied ||
          outcome.optimizer_id != optimizer.optimizer->optimizer_id() ||
          outcome.optimizer_completed_steps != state.global_step + 1U) {
        throw_incompatible(
            "optimizer.step",
            "one strategy-owned consecutive update for " +
                optimizer.optimizer->optimizer_id(),
            outcome.optimizer_id + ":" +
                std::to_string(outcome.optimizer_completed_steps),
            "Do not update any optimizer outside the strategy.");
      }
    }
    if (request.gradient_scaler) {
      for (const auto& optimizer : request.optimizers) {
        request.gradient_scaler->mark_step(optimizer.optimizer, true);
      }
      request.gradient_scaler->update();
    }
    checked_increment(state.global_step, "trainer.global_step");
    state.pending_micro_steps = 0;
    zero_grad_all();
    step_schedulers(optim::SchedulerStepMode::batch);
    emit(CallbackEventKind::optimizer_step_finished, CallbackPhase::training,
         {{"optimizer.applied", 1.0},
          {"optimizer.count", static_cast<double>(request.optimizers.size())}},
         stop);
  }

  std::map<std::string, double> validate_epoch(std::stop_token stop) {
    state.validation_active = true;
    MetricAccumulator aggregate;
    const bool was_training = request.model->state().module().is_training();
    request.model->state().module().eval();
    try {
      emit(CallbackEventKind::validation_started, CallbackPhase::validation, {},
           stop);
      request.validation_data->set_epoch(state.active_epoch);
      torch::NoGradGuard no_grad;
      while (!stop.stop_requested()) {
        auto batch = request.validation_data->next(stop);
        if (!batch.has_value()) {
          break;
        }
        emit(CallbackEventKind::batch_started, CallbackPhase::validation, {},
             stop);
        auto output =
            invoke_step(*batch, request.validation_batch_mapper,
                        request.validation_target_mapper,
                        request.validation_step, StepPhase::validation);
        validate_step_output(output, *batch, false);
        const auto loss =
            output.loss.defined()
                ? std::optional<double>(output.loss.detach().item<double>())
                : std::nullopt;
        add_output(aggregate, output, loss, "validation.loss", options);
        auto batch_metrics = output.metrics;
        if (loss.has_value()) {
          batch_metrics.emplace("validation.loss", *loss);
        }
        emit(CallbackEventKind::batch_finished, CallbackPhase::validation,
             std::move(batch_metrics), stop);
      }
      auto values = finish_metrics(aggregate);
      values = request.strategy->reduce_metrics(values);
      validate_metric_map(values, options, "validation.metrics");
      emit(CallbackEventKind::validation_finished, CallbackPhase::validation,
           values, stop);
      state.validation_active = false;
      request.model->state().module().train(was_training);
      return values;
    } catch (...) {
      state.validation_active = false;
      request.model->state().module().train(was_training);
      throw;
    }
  }

  void finalize_terminal_event(CallbackEventKind kind, CallbackPhase phase,
                               std::stop_token stop) {
    emit(kind, phase, {}, stop, false, false);
    if (request.event_outbox && options.drain_events) {
      request.event_outbox->drain(stop);
    }
  }
};

Trainer::Trainer(TrainerOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {
  validate_options(impl_->options);
}

Trainer::~Trainer() = default;

void Trainer::prepare(FitRequest request) {
  if (impl_->prepared || impl_->state.phase != TrainerPhase::created) {
    throw_not_ready("prepare", "a newly created Trainer",
                    phase_name(impl_->state.phase),
                    "Create one Trainer per execution domain.");
  }
  if (!request.model || request.optimizers.empty() ||
      request.optimizers.size() > impl_->options.max_optimizers ||
      !request.training_data || !request.training_batch_mapper ||
      !request.training_step || !request.strategy ||
      !request.model_checkpoint || !request.rng_checkpoint) {
    throw_invalid("prepare.required_objects",
                  "model, bounded optimizers, data, task, strategy, and state",
                  "one or more null objects",
                  "Bind a complete execution and checkpoint domain.");
  }
  std::sort(request.optimizers.begin(), request.optimizers.end(),
            [](const auto& left, const auto& right) {
              if (!left.optimizer) {
                return static_cast<bool>(right.optimizer);
              }
              if (!right.optimizer) {
                return false;
              }
              return left.optimizer->optimizer_id() <
                     right.optimizer->optimizer_id();
            });
  std::set<std::string> optimizer_ids;
  for (const auto& binding : request.optimizers) {
    if (!binding.optimizer || !binding.checkpoint) {
      throw_invalid("prepare.optimizer_binding",
                    "non-null optimizer and checkpoint participant", "null",
                    "Bind complete optimizer execution entries.");
    }
    const auto& optimizer_id = binding.optimizer->optimizer_id();
    if (!is_identifier(optimizer_id) ||
        optimizer_id.size() > impl_->options.max_identifier_bytes ||
        !optimizer_ids.insert(optimizer_id).second) {
      throw_invalid("prepare.optimizer_id",
                    "unique bounded stable optimizer identities", optimizer_id,
                    "Assign one distinct identity to each optimizer.");
    }
    if (binding.optimizer->completed_steps() != 0U) {
      throw_not_ready("prepare.optimizer", "a fresh optimizer with zero steps",
                      optimizer_id + ":" +
                          std::to_string(binding.optimizer->completed_steps()),
                      "Restore prior state only through the coordinator.");
    }
    if (binding.checkpoint->participant_id() != optimizer_id) {
      throw_incompatible("prepare.optimizer_participant", optimizer_id,
                         binding.checkpoint->participant_id(),
                         "Bind each checkpoint participant to its optimizer.");
    }
    if (binding.scheduler) {
      if (binding.scheduler->scheduler().completed_steps() != 0U) {
        throw_not_ready(
            "prepare.scheduler", "a fresh scheduler with zero steps",
            binding.scheduler->participant_id() + ":" +
                std::to_string(
                    binding.scheduler->scheduler().completed_steps()),
            "Restore prior state only through the coordinator.");
      }
      if (binding.scheduler->scheduler().descriptor().optimizer_id !=
          optimizer_id) {
        throw_incompatible(
            "prepare.scheduler_optimizer", optimizer_id,
            binding.scheduler->scheduler().descriptor().optimizer_id,
            "Bind each scheduler to the optimizer in the same entry.");
      }
    }
  }
  const bool any_validation =
      request.validation_data || request.validation_batch_mapper ||
      request.validation_target_mapper || request.validation_step;
  const bool complete_validation = request.validation_data &&
                                   request.validation_batch_mapper &&
                                   request.validation_step;
  if (any_validation && !complete_validation) {
    throw_invalid("prepare.validation", "data, batch mapper, and step together",
                  "partial validation configuration",
                  "Configure the complete validation domain or none of it.");
  }
  if (request.training_data->closed() ||
      (request.validation_data && request.validation_data->closed()) ||
      request.strategy->closed()) {
    throw_not_ready("prepare.lifecycle", "open data and strategy objects",
                    "closed object", "Create fresh execution resources.");
  }
  if (request.validation_data &&
      request.validation_data.get() == request.training_data.get()) {
    throw_invalid("prepare.validation_data",
                  "an independent validation pipeline",
                  "the training pipeline instance",
                  "Use separate cursor state for training and validation.");
  }
  if (!valid_precision_mode(impl_->options.precision)) {
    throw_invalid(
        "prepare.precision", "a known PrecisionMode value",
        std::to_string(static_cast<std::uint8_t>(impl_->options.precision)),
        "Select an explicit full, CPU bfloat16, or CUDA mode.");
  }
  const auto device = request.strategy->descriptor().device;
  const bool cuda = is_cuda_device(device);
  const bool cpu = is_cpu_device(device);
  if (cuda && !request.rng_checkpoint->options().include_cuda) {
    throw_invalid("prepare.rng_checkpoint", "CUDA RNG state enabled", device,
                  "Include CUDA generators for exact continuation.");
  }
  switch (impl_->options.precision) {
    case PrecisionMode::full:
      if (request.gradient_scaler || request.gradient_scaler_checkpoint) {
        throw_invalid("prepare.precision", "no GradScaler in full mode",
                      "scaler supplied",
                      "Bind a scaler only for CUDA float16 mode.");
      }
      break;
    case PrecisionMode::cpu_bfloat16:
      if (!cpu || request.gradient_scaler ||
          request.gradient_scaler_checkpoint) {
        throw_invalid("prepare.precision", "CPU device without GradScaler",
                      device, "Use CPU bfloat16 with a CPU strategy only.");
      }
      break;
    case PrecisionMode::cuda_float16:
      if (!cuda || !request.gradient_scaler ||
          !request.gradient_scaler_checkpoint ||
          !request.gradient_scaler->options().enabled) {
        throw_invalid(
            "prepare.precision",
            "CUDA device with an enabled GradScaler and its checkpoint", device,
            "Bind both scaler objects for CUDA float16 mode.");
      }
      if (&request.gradient_scaler_checkpoint->scaler() !=
          request.gradient_scaler.get()) {
        throw_incompatible(
            "prepare.gradient_scaler_participant",
            request.gradient_scaler->options().scaler_id,
            request.gradient_scaler_checkpoint->scaler().options().scaler_id,
            "Bind the checkpoint participant for this GradScaler.");
      }
      break;
    case PrecisionMode::cuda_bfloat16:
      if (!cuda || request.gradient_scaler ||
          request.gradient_scaler_checkpoint) {
        throw_invalid("prepare.precision", "CUDA bfloat16 without GradScaler",
                      device, "Use CUDA float16 for dynamic scaling.");
      }
      break;
  }
  for (const auto& binding : request.optimizers) {
    if (binding.scheduler &&
        binding.scheduler->mode() == optim::SchedulerStepMode::metric &&
        !request.validation_data) {
      throw_invalid("prepare.scheduler_metric",
                    "a validation domain for every metric scheduler",
                    binding.scheduler->participant_id(),
                    "Configure validation to publish monitored metrics.");
    }
  }
  if (request.model_checkpoint->state().shared_module().get() !=
      request.model->state().shared_module().get()) {
    throw_incompatible(
        "prepare.model_participant", "the adapted model module",
        "a different module",
        "Bind the checkpoint participant to the FitRequest model.");
  }
  if (!request.callbacks) {
    request.callbacks = std::make_shared<CallbackDispatcher>();
  }

  impl_->request = std::move(request);
  try {
    auto registry = std::make_unique<CheckpointParticipantRegistry>();
    registry->register_participants(
        {std::shared_ptr<CheckpointParticipant>(this,
                                                [](CheckpointParticipant*) {}),
         impl_->request.training_data, impl_->request.strategy,
         impl_->request.model_checkpoint, impl_->request.rng_checkpoint});
    for (const auto& binding : impl_->request.optimizers) {
      registry->register_participant(binding.checkpoint);
      if (binding.scheduler) {
        registry->register_participant(binding.scheduler);
      }
    }
    if (impl_->request.validation_data) {
      registry->register_participant(impl_->request.validation_data);
    }
    if (impl_->request.event_outbox) {
      registry->register_participant(impl_->request.event_outbox);
    }
    if (impl_->request.gradient_scaler_checkpoint) {
      registry->register_participant(impl_->request.gradient_scaler_checkpoint);
    }
    register_callback_checkpoint_participants(*registry,
                                              impl_->request.callbacks);
    if (!impl_->request.additional_participants.empty()) {
      registry->register_participants(impl_->request.additional_participants);
    }
    std::vector<optim::OptimizerAdapter*> optimizers;
    optimizers.reserve(impl_->request.optimizers.size());
    for (const auto& binding : impl_->request.optimizers) {
      optimizers.push_back(binding.optimizer.get());
    }
    impl_->request.strategy->prepare(*impl_->request.model, optimizers);
    impl_->zero_grad_all();
    impl_->participant_ids = registry->participant_ids();
    impl_->registry = std::move(registry);
    impl_->coordinator = std::make_unique<CheckpointCoordinator>(
        *impl_->registry, impl_->options.checkpoint);
    impl_->state = TrainerControlState{};
    impl_->state.phase = TrainerPhase::prepared;
    impl_->prepared = true;
  } catch (...) {
    impl_->request = {};
    impl_->participant_ids.clear();
    impl_->registry.reset();
    impl_->coordinator.reset();
    throw;
  }
}

FitResult Trainer::fit(std::stop_token stop) {
  if (!impl_->prepared || (impl_->state.phase != TrainerPhase::prepared &&
                           impl_->state.phase != TrainerPhase::running)) {
    throw_not_ready("fit", "a prepared or resumable Trainer",
                    phase_name(impl_->state.phase),
                    "Call prepare(), optionally restore(), then fit() once.");
  }
  if (impl_->request.event_outbox && impl_->options.drain_events) {
    impl_->request.event_outbox->drain(stop);
  }
  for (const auto& binding : impl_->request.optimizers) {
    if (binding.optimizer->completed_steps() != impl_->state.global_step) {
      throw_incompatible(
          "fit.optimizer_steps." + binding.optimizer->optimizer_id(),
          std::to_string(impl_->state.global_step),
          std::to_string(binding.optimizer->completed_steps()),
          "Restore Trainer and every optimizer state together.");
    }
    if (binding.scheduler) {
      const auto scheduler_id = binding.scheduler->participant_id();
      const auto cursor = impl_->state.scheduler_events.find(scheduler_id);
      const auto expected = cursor == impl_->state.scheduler_events.end()
                                ? static_cast<std::uint64_t>(0)
                                : cursor->second;
      if (binding.scheduler->scheduler().completed_steps() != expected) {
        throw_incompatible(
            "fit.scheduler_steps." + scheduler_id, std::to_string(expected),
            std::to_string(binding.scheduler->scheduler().completed_steps()),
            "Restore Trainer and every scheduler state together.");
      }
    }
  }

  try {
    if (!impl_->state.run_started) {
      impl_->state.phase = TrainerPhase::running;
      impl_->emit(CallbackEventKind::run_started, CallbackPhase::preparing, {},
                  stop);
      impl_->state.run_started = true;
    } else if (impl_->state.checkpoint_in_progress) {
      impl_->state.checkpoint_in_progress = false;
      checked_increment(impl_->state.checkpoint_count, "checkpoint.count");
      impl_->emit(CallbackEventKind::checkpoint_committed,
                  CallbackPhase::checkpointing, {}, stop, false);
    }

    impl_->request.model->state().module().train();
    bool cancelled = stop.stop_requested();
    bool reached_step_limit =
        impl_->options.max_steps != 0U &&
        impl_->state.global_step >= impl_->options.max_steps;

    while (!cancelled && !impl_->state.callback_stop && !reached_step_limit &&
           impl_->state.next_epoch < impl_->options.max_epochs) {
      if (!impl_->state.epoch_in_progress) {
        impl_->state.active_epoch = impl_->state.next_epoch;
        impl_->state.training.clear();
        impl_->state.last_validation.clear();
        impl_->state.skip_validation = false;
        impl_->request.training_data->set_epoch(impl_->state.active_epoch);
        impl_->state.epoch_in_progress = true;
        impl_->emit(CallbackEventKind::epoch_started, CallbackPhase::training,
                    {}, stop);
      }

      while (!cancelled && !impl_->state.callback_stop && !reached_step_limit) {
        auto batch = impl_->request.training_data->next(stop);
        if (!batch.has_value()) {
          cancelled = stop.stop_requested();
          break;
        }
        impl_->emit(CallbackEventKind::batch_started, CallbackPhase::training,
                    {}, stop);
        auto output = impl_->invoke_step(
            *batch, impl_->request.training_batch_mapper,
            impl_->request.training_target_mapper, impl_->request.training_step,
            StepPhase::training);
        validate_step_output(output, *batch, true);
        const auto loss_value = output.loss.detach().item<double>();
        add_output(impl_->state.training, output, loss_value, "training.loss",
                   impl_->options);
        auto backward_loss =
            output.loss / static_cast<double>(impl_->options.accumulate_steps);
        if (impl_->request.gradient_scaler) {
          backward_loss = impl_->request.gradient_scaler->scale(backward_loss);
        }
        impl_->request.strategy->backward(backward_loss);
        ++impl_->state.pending_micro_steps;
        if (impl_->state.pending_micro_steps ==
            impl_->options.accumulate_steps) {
          impl_->optimizer_step(stop, false);
        }
        auto batch_metrics = output.metrics;
        batch_metrics.emplace("training.loss", loss_value);
        impl_->emit(CallbackEventKind::batch_finished, CallbackPhase::training,
                    std::move(batch_metrics), stop);
        reached_step_limit =
            impl_->options.max_steps != 0U &&
            impl_->state.global_step >= impl_->options.max_steps;
        cancelled = stop.stop_requested();
        if (!cancelled && impl_->state.pending_micro_steps == 0U) {
          impl_->publish_checkpoint(stop);
        }
      }

      if (cancelled) {
        if (impl_->state.pending_micro_steps != 0U) {
          impl_->zero_grad_all();
          impl_->state.pending_micro_steps = 0;
        }
        break;
      }
      if (impl_->state.callback_stop) {
        if (impl_->state.pending_micro_steps != 0U) {
          impl_->zero_grad_all();
          impl_->state.pending_micro_steps = 0;
        }
        break;
      }
      if (impl_->state.pending_micro_steps != 0U) {
        impl_->optimizer_step(stop, true);
        reached_step_limit =
            impl_->options.max_steps != 0U &&
            impl_->state.global_step >= impl_->options.max_steps;
      }
      if (impl_->state.training.sample_count == 0U) {
        throw_not_ready("training.epoch", "at least one committed batch",
                        "empty data stream",
                        "Provide training data for every configured epoch.");
      }
      impl_->state.last_training = impl_->request.strategy->reduce_metrics(
          finish_metrics(impl_->state.training));
      validate_metric_map(impl_->state.last_training, impl_->options,
                          "training.metrics");

      if (impl_->request.validation_data && !impl_->state.skip_validation &&
          !stop.stop_requested()) {
        impl_->state.last_validation = impl_->validate_epoch(stop);
      }
      impl_->step_schedulers(optim::SchedulerStepMode::epoch);
      const bool has_metric_scheduler = std::any_of(
          impl_->request.optimizers.begin(), impl_->request.optimizers.end(),
          [](const auto& binding) {
            return binding.scheduler && binding.scheduler->mode() ==
                                            optim::SchedulerStepMode::metric;
          });
      if (has_metric_scheduler) {
        const auto metric =
            impl_->state.last_validation.find(impl_->options.scheduler_metric);
        if (metric == impl_->state.last_validation.end()) {
          throw_not_ready("scheduler.metric", impl_->options.scheduler_metric,
                          "missing",
                          "Run validation and publish the configured metric.");
        }
        impl_->step_schedulers(optim::SchedulerStepMode::metric,
                               metric->second);
      }

      auto epoch_metrics = impl_->state.last_training;
      epoch_metrics.insert(impl_->state.last_validation.begin(),
                           impl_->state.last_validation.end());
      impl_->state.epoch_in_progress = false;
      checked_increment(impl_->state.completed_epochs,
                        "trainer.completed_epochs");
      impl_->state.next_epoch = impl_->state.active_epoch + 1U;
      impl_->emit(CallbackEventKind::epoch_finished, CallbackPhase::training,
                  std::move(epoch_metrics), stop);
      impl_->publish_checkpoint(stop);
      cancelled = stop.stop_requested();
      reached_step_limit = impl_->options.max_steps != 0U &&
                           impl_->state.global_step >= impl_->options.max_steps;
    }

    FitResult result{.status = FitStatus::completed,
                     .completed_epochs = impl_->state.completed_epochs,
                     .global_step = impl_->state.global_step,
                     .training_metrics = impl_->state.last_training,
                     .validation_metrics = impl_->state.last_validation,
                     .reason = {}};
    if (cancelled) {
      impl_->state.phase = TrainerPhase::cancelled;
      result.status = FitStatus::cancelled;
      result.reason = "stop token requested";
      impl_->finalize_terminal_event(CallbackEventKind::run_cancelled,
                                     CallbackPhase::cancelled, stop);
    } else {
      impl_->state.phase = TrainerPhase::completed;
      if (impl_->state.callback_stop) {
        result.status = FitStatus::stopped;
        result.reason = impl_->state.stop_reason;
      }
      impl_->finalize_terminal_event(CallbackEventKind::run_finished,
                                     CallbackPhase::completed, stop);
    }
    return result;
  } catch (...) {
    const auto original = std::current_exception();
    if (impl_->state.phase == TrainerPhase::running) {
      if (impl_->state.pending_micro_steps != 0U) {
        try {
          impl_->zero_grad_all();
        } catch (...) {
        }
        impl_->state.pending_micro_steps = 0;
      }
      impl_->state.validation_active = false;
      impl_->state.phase = TrainerPhase::failed;
      try {
        impl_->finalize_terminal_event(CallbackEventKind::run_failed,
                                       CallbackPhase::failed, stop);
      } catch (...) {
      }
    }
    std::rethrow_exception(original);
  }
}

CheckpointDescriptor Trainer::save_checkpoint(CheckpointStateSink& sink) {
  if (!impl_->checkpoint_safe()) {
    throw_not_ready("save_checkpoint", "an empty accumulation safe boundary",
                    phase_name(impl_->state.phase),
                    "Wait for an optimizer or epoch boundary.");
  }
  impl_->request.strategy->barrier(StrategyBarrierReason::checkpoint);
  return impl_->coordinator->save(sink);
}

CheckpointDescriptor Trainer::restore_checkpoint(
    const CheckpointStateSource& source, std::optional<RestorePolicy> policy) {
  if (!impl_->prepared || impl_->state.phase != TrainerPhase::prepared ||
      impl_->state.run_started) {
    throw_not_ready("restore_checkpoint", "prepared state before fit",
                    phase_name(impl_->state.phase),
                    "Prepare a fresh matching domain before restore.");
  }
  impl_->request.strategy->barrier(StrategyBarrierReason::checkpoint);
  return impl_->coordinator->restore(
      source, policy.value_or(impl_->options.restore_policy));
}

void Trainer::close(std::stop_token stop) {
  if (impl_->state.phase == TrainerPhase::closed) {
    return;
  }
  if (!impl_->prepared) {
    impl_->state.phase = TrainerPhase::closed;
    return;
  }
  if (impl_->request.event_outbox) {
    impl_->request.event_outbox->close(stop);
  }
  impl_->request.training_data->close();
  if (impl_->request.validation_data) {
    impl_->request.validation_data->close();
  }
  impl_->request.strategy->barrier(StrategyBarrierReason::shutdown, stop);
  impl_->request.strategy->close();
  impl_->state.phase = TrainerPhase::closed;
}

const TrainerOptions& Trainer::options() const noexcept {
  return impl_->options;
}

TrainerPhase Trainer::phase() const noexcept { return impl_->state.phase; }

std::uint64_t Trainer::global_step() const noexcept {
  return impl_->state.global_step;
}

std::uint64_t Trainer::completed_epochs() const noexcept {
  return impl_->state.completed_epochs;
}

const std::map<std::string, double>& Trainer::training_metrics()
    const noexcept {
  return impl_->state.last_training;
}

const std::map<std::string, double>& Trainer::validation_metrics()
    const noexcept {
  return impl_->state.last_validation;
}

const CheckpointCoordinator& Trainer::checkpoint_coordinator() const {
  if (!impl_->coordinator) {
    throw_not_ready("checkpoint_coordinator", "a prepared Trainer", "created",
                    "Call prepare() before accessing checkpoint state.");
  }
  return *impl_->coordinator;
}

std::string Trainer::participant_id() const {
  return impl_->options.trainer_id;
}

StateManifest Trainer::describe_state() const {
  StateManifest manifest{
      .participant_id = participant_id(),
      .schema_version = 3,
      .restore_level = RestoreLevel::exact,
      .fields = {{.name = kTopologyField,
                  .media_type = kTopologyMediaType,
                  .max_bytes = impl_->options.max_topology_bytes,
                  .required = true},
                 {.name = kControlField,
                  .media_type = kControlMediaType,
                  .max_bytes = impl_->options.max_control_bytes,
                  .required = true}},
      .degradation_reasons = {}};
  validate_state_manifest(manifest);
  return manifest;
}

void Trainer::save_state(StateWriter& writer) const {
  if (!impl_->checkpoint_safe()) {
    throw_not_ready(
        "save_state", "a prepared safe consistency boundary",
        phase_name(impl_->state.phase),
        "Checkpoint only with no pending micro-step or validation.");
  }
  const auto topology =
      encode_topology(impl_->options, impl_->request, impl_->participant_ids);
  const auto control = encode_control(impl_->state);
  if (topology.size() > impl_->options.max_topology_bytes ||
      control.size() > impl_->options.max_control_bytes) {
    throw_exhausted("save_state", "state within configured byte limits",
                    std::to_string(topology.size() + control.size()),
                    "Reduce identities or retained metric state.");
  }
  writer.write_blob(kTopologyField, topology);
  writer.write_blob(kControlField, control);
}

void Trainer::load_state(const StateReader& reader, RestorePolicy policy) {
  if (!impl_->prepared || impl_->state.phase != TrainerPhase::prepared) {
    throw_not_ready("load_state", "fresh prepared state",
                    phase_name(impl_->state.phase),
                    "Restore before starting the run.");
  }
  validate_manifest(reader.manifest(), describe_state(), policy);
  const auto stored_topology =
      reader.read_blob(kTopologyField, impl_->options.max_topology_bytes);
  const auto expected_topology =
      encode_topology(impl_->options, impl_->request, impl_->participant_ids);
  if (stored_topology.size() != expected_topology.size() ||
      !std::equal(stored_topology.begin(), stored_topology.end(),
                  expected_topology.begin())) {
    throw_incompatible("load_state.topology", "identical Trainer topology",
                       "different encoded topology",
                       "Restore with the same options and participant domain.");
  }
  auto candidate = decode_control(
      reader.read_blob(kControlField, impl_->options.max_control_bytes),
      impl_->options);
  std::set<std::string> configured_schedulers;
  for (const auto& binding : impl_->request.optimizers) {
    if (binding.scheduler) {
      configured_schedulers.insert(binding.scheduler->participant_id());
    }
  }
  for (const auto& [scheduler_id, event_index] : candidate.scheduler_events) {
    static_cast<void>(event_index);
    if (!configured_schedulers.contains(scheduler_id)) {
      throw_serialization("load_state.scheduler_events",
                          "only configured scheduler identities", scheduler_id,
                          "Reject state from another optimizer topology.");
    }
  }
  if (!restorable_phase(candidate.phase) || candidate.validation_active ||
      candidate.pending_micro_steps != 0U ||
      candidate.next_epoch > impl_->options.max_epochs ||
      candidate.completed_epochs > candidate.next_epoch ||
      (impl_->options.max_steps != 0U &&
       candidate.global_step > impl_->options.max_steps) ||
      (!candidate.run_started && candidate.phase != TrainerPhase::prepared) ||
      (candidate.epoch_in_progress &&
       candidate.active_epoch != candidate.next_epoch)) {
    throw_serialization("load_state.control",
                        "a safe internally consistent Trainer boundary",
                        "invalid lifecycle or cursor relationship",
                        "Reject incomplete micro-step and validation state.");
  }
  validate_metric_map(candidate.training.weighted_sums, impl_->options,
                      "load_state.training");
  validate_metric_map(candidate.last_training, impl_->options,
                      "load_state.last_training");
  validate_metric_map(candidate.last_validation, impl_->options,
                      "load_state.last_validation");
  impl_->state = std::move(candidate);
}

}  // namespace jyppx::liuding::training
