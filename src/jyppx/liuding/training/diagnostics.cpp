// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/diagnostics.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace jyppx::liuding::training {
namespace {

constexpr std::uint64_t kTopologyMagic = 0x31504f5447414944ULL;
constexpr std::uint64_t kStateMagic = 0x3154415447414944ULL;
constexpr std::uint64_t kTopologyBytes = 1152;
constexpr std::uint64_t kStateBytes = 184;

[[noreturn]] void invalid(std::string operation) {
  throw InvalidArgument(
      "Invalid training diagnostics contract",
      ErrorDetails{
          .component = "training.diagnostics",
          .operation = std::move(operation),
          .expected = "valid identity, positive bounds and real strided "
                      "CPU/CUDA tensors",
          .actual = "invalid diagnostic input",
          .hint = "Check the diagnostic options and supplied gradients."});
}

[[noreturn]] void exhausted(std::string operation) {
  throw ResourceExhausted(
      "Training diagnostics resource bound exceeded",
      ErrorDetails{
          .component = "training.diagnostics",
          .operation = std::move(operation),
          .expected = "scan and statistics within explicit bounds",
          .actual = "configured bound or numeric counter exceeded",
          .hint =
              "Reduce sampling work or explicitly increase the scan bounds."});
}

[[noreturn]] void malformed() {
  throw SerializationError("Malformed training diagnostics state");
}

void validate_limits(const GradientNormLimits& limits) {
  if (limits.max_tensors == 0 || limits.max_elements == 0 ||
      limits.tile_elements == 0 ||
      limits.tile_elements >
          (std::numeric_limits<std::uint64_t>::max() - 16) / 8 ||
      limits.max_scratch_bytes < 24) {
    invalid("gradient_norm.limits");
  }
}

std::uint64_t added(std::uint64_t left, std::uint64_t right) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    exhausted("counter");
  }
  return left + right;
}

double duration_sum(double left, double right) {
  const auto result = left + right;
  if (!std::isfinite(right) || right < 0.0 || !std::isfinite(result)) {
    exhausted("duration");
  }
  return result;
}

using Bytes = std::vector<std::byte>;

void append(Bytes& bytes, std::uint64_t value) {
  for (unsigned index = 0; index < 8; ++index) {
    bytes.push_back(static_cast<std::byte>((value >> (index * 8U)) & 0xffU));
  }
}

void append_double(Bytes& bytes, double value) {
  append(bytes, std::bit_cast<std::uint64_t>(value));
}

class Reader {
 public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}
  std::uint64_t integer() {
    if (bytes_.size() < 8) malformed();
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
      value |= static_cast<std::uint64_t>(bytes_[index]) << (index * 8U);
    }
    bytes_ = bytes_.subspan(8);
    return value;
  }
  double number() {
    const auto value = std::bit_cast<double>(integer());
    if (!std::isfinite(value) || value < 0.0) malformed();
    return value;
  }

 private:
  std::span<const std::byte> bytes_;
};

Bytes topology(const TrainingDiagnosticsOptions& options) {
  Bytes bytes;
  append(bytes, kTopologyMagic);
  append(bytes, options.participant_id.size());
  for (const auto character : options.participant_id) {
    bytes.push_back(static_cast<std::byte>(character));
  }
  append(bytes, options.gradient_norm_every);
  append(bytes, options.measure_data_wait ? 1 : 0);
  append(bytes, options.gradient_norm.max_tensors);
  append(bytes, options.gradient_norm.max_elements);
  append(bytes, options.gradient_norm.tile_elements);
  append(bytes, options.gradient_norm.max_scratch_bytes);
  return bytes;
}

Bytes encode(const TrainingDiagnosticsSnapshot& state) {
  Bytes bytes;
  bytes.reserve(kStateBytes);
  append(bytes, kStateMagic);
  for (const auto& counts : {state.training_loss, state.validation_loss}) {
    append(bytes, counts.observations);
    append(bytes, counts.nan);
    append(bytes, counts.infinity);
  }
  for (const auto& wait : {state.training_wait, state.validation_wait}) {
    append(bytes, wait.batches);
    append_double(bytes, wait.seconds);
    append_double(bytes, wait.max_seconds);
  }
  append(bytes, state.update_attempts);
  append(bytes, state.nonfinite_gradient_attempts);
  append(bytes, state.amp_skipped_attempts);
  append(bytes, state.norm_samples);
  append(bytes, state.norm_elements);
  append_double(bytes, state.last_gradient_l2);
  append_double(bytes, state.mean_gradient_l2);
  append_double(bytes, state.max_gradient_l2);
  append_double(bytes, state.norm_seconds);
  append(bytes, state.known_peak_scratch_bytes);
  return bytes;
}

TrainingDiagnosticsSnapshot decode(std::span<const std::byte> bytes,
                                   const TrainingDiagnosticsOptions& options) {
  if (bytes.size() != kStateBytes) malformed();
  Reader reader(bytes);
  if (reader.integer() != kStateMagic) malformed();
  TrainingDiagnosticsSnapshot state;
  for (auto* counts : {&state.training_loss, &state.validation_loss}) {
    counts->observations = reader.integer();
    counts->nan = reader.integer();
    counts->infinity = reader.integer();
    if (counts->nan > counts->observations ||
        counts->infinity > counts->observations - counts->nan)
      malformed();
  }
  for (auto* wait : {&state.training_wait, &state.validation_wait}) {
    wait->batches = reader.integer();
    wait->seconds = reader.number();
    wait->max_seconds = reader.number();
    if (wait->max_seconds > wait->seconds ||
        (wait->batches == 0 && wait->seconds != 0.0) ||
        (!options.measure_data_wait && wait->batches != 0))
      malformed();
  }
  state.update_attempts = reader.integer();
  state.nonfinite_gradient_attempts = reader.integer();
  state.amp_skipped_attempts = reader.integer();
  state.norm_samples = reader.integer();
  state.norm_elements = reader.integer();
  state.last_gradient_l2 = reader.number();
  state.mean_gradient_l2 = reader.number();
  state.max_gradient_l2 = reader.number();
  state.norm_seconds = reader.number();
  state.known_peak_scratch_bytes = reader.integer();
  if (state.nonfinite_gradient_attempts > state.update_attempts ||
      state.amp_skipped_attempts > state.nonfinite_gradient_attempts ||
      state.norm_samples >
          state.update_attempts - state.nonfinite_gradient_attempts ||
      state.last_gradient_l2 > state.max_gradient_l2 ||
      state.mean_gradient_l2 > state.max_gradient_l2 ||
      state.known_peak_scratch_bytes >
          options.gradient_norm.max_scratch_bytes ||
      (options.gradient_norm_every == 0 && state.norm_samples != 0) ||
      (options.gradient_norm_every != 0 &&
       state.norm_samples >
           state.update_attempts / options.gradient_norm_every) ||
      (state.norm_samples == 0 &&
       (state.norm_elements != 0 || state.last_gradient_l2 != 0.0 ||
        state.mean_gradient_l2 != 0.0 || state.max_gradient_l2 != 0.0 ||
        state.norm_seconds != 0.0 || state.known_peak_scratch_bytes != 0))) {
    malformed();
  }
  return state;
}

void scan_tile(const Tensor& tensor, std::uint64_t tile_elements,
               double& norm) {
  if (static_cast<std::uint64_t>(tensor.numel()) > tile_elements) {
    std::int64_t dimension = 0;
    for (std::int64_t index = 1; index < tensor.dim(); ++index) {
      if (tensor.size(index) > tensor.size(dimension)) dimension = index;
    }
    const auto split = tensor.size(dimension) / 2;
    scan_tile(tensor.narrow(dimension, 0, split), tile_elements, norm);
    scan_tile(tensor.narrow(dimension, split, tensor.size(dimension) - split),
              tile_elements, norm);
    return;
  }
  if (tensor.numel() == 0) return;
  auto work = tensor.detach().to(tensor.options().dtype(torch::kFloat64), false,
                                 true, torch::MemoryFormat::Contiguous);
  work.abs_();
  const auto maximum = work.max().item<double>();
  if (!std::isfinite(maximum)) {
    throw NumericalError("Gradient norm received nonfinite values");
  }
  if (maximum != 0.0) {
    work.div_(maximum);
    work.mul_(work);
    norm = std::hypot(norm, maximum * std::sqrt(work.sum().item<double>()));
    if (!std::isfinite(norm)) {
      throw NumericalError("Gradient L2 norm exceeds finite FP64 range");
    }
  }
}

}  // namespace

GradientNormMeasurement measure_gradient_l2(std::span<const Tensor> gradients,
                                            const GradientNormLimits& limits) {
  validate_limits(limits);
  if (gradients.size() > limits.max_tensors) exhausted("gradient_norm.tensors");
  GradientNormMeasurement result;
  std::uint64_t largest = 0;
  for (const auto& gradient : gradients) {
    if (!gradient.defined() || !gradient.is_floating_point() ||
        gradient.layout() != torch::kStrided ||
        (!gradient.device().is_cpu() && !gradient.device().is_cuda())) {
      invalid("gradient_norm.tensor");
    }
    const auto elements = static_cast<std::uint64_t>(gradient.numel());
    if (elements > limits.max_elements - result.elements)
      exhausted("gradient_norm.elements");
    result.elements += elements;
    largest = std::max(largest, elements);
  }
  if (largest != 0) {
    result.known_scratch_bytes =
        8 * std::min(largest, limits.tile_elements) + 16;
    if (result.known_scratch_bytes > limits.max_scratch_bytes)
      exhausted("gradient_norm.scratch");
  }
  const auto started = std::chrono::steady_clock::now();
  torch::NoGradGuard guard;
  for (const auto& gradient : gradients)
    scan_tile(gradient, limits.tile_elements, result.l2);
  result.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  return result;
}

TrainingDiagnostics::TrainingDiagnostics(TrainingDiagnosticsOptions options)
    : options_(std::move(options)) {
  validate_limits(options_.gradient_norm);
  if (options_.participant_id.empty() ||
      options_.participant_id.size() > 1024 ||
      !std::all_of(options_.participant_id.begin(),
                   options_.participant_id.end(), [](unsigned char c) {
                     return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                            (c >= '0' && c <= '9') || c == '.' || c == '-' ||
                            c == '_';
                   }))
    invalid("participant_id");
  validate_state_manifest(describe_state());
}

const TrainingDiagnosticsOptions& TrainingDiagnostics::options()
    const noexcept {
  return options_;
}

TrainingDiagnosticsSnapshot TrainingDiagnostics::snapshot() const noexcept {
  return state_;
}

void TrainingDiagnostics::observe_loss(double loss, bool validation) {
  auto counts = validation ? state_.validation_loss : state_.training_loss;
  counts.observations = added(counts.observations, 1);
  if (std::isnan(loss)) counts.nan = added(counts.nan, 1);
  if (std::isinf(loss)) counts.infinity = added(counts.infinity, 1);
  (validation ? state_.validation_loss : state_.training_loss) = counts;
}

void TrainingDiagnostics::observe_wait(double seconds, bool validation) {
  auto wait = validation ? state_.validation_wait : state_.training_wait;
  wait.batches = added(wait.batches, 1);
  wait.seconds = duration_sum(wait.seconds, seconds);
  wait.max_seconds = std::max(wait.max_seconds, seconds);
  (validation ? state_.validation_wait : state_.training_wait) = wait;
}

bool TrainingDiagnostics::begin_gradient_attempt() {
  state_.update_attempts = added(state_.update_attempts, 1);
  return options_.gradient_norm_every != 0 &&
         state_.update_attempts % options_.gradient_norm_every == 0;
}

void TrainingDiagnostics::observe_nonfinite_gradient(bool amp_skipped) {
  auto candidate = state_;
  candidate.nonfinite_gradient_attempts =
      added(candidate.nonfinite_gradient_attempts, 1);
  if (amp_skipped)
    candidate.amp_skipped_attempts = added(candidate.amp_skipped_attempts, 1);
  state_ = candidate;
}

void TrainingDiagnostics::observe_norm(
    const GradientNormMeasurement& measurement) {
  auto candidate = state_;
  candidate.norm_samples = added(candidate.norm_samples, 1);
  candidate.norm_elements =
      added(candidate.norm_elements, measurement.elements);
  candidate.last_gradient_l2 = measurement.l2;
  candidate.mean_gradient_l2 += (measurement.l2 - candidate.mean_gradient_l2) /
                                static_cast<double>(candidate.norm_samples);
  candidate.max_gradient_l2 =
      std::max(candidate.max_gradient_l2, measurement.l2);
  candidate.norm_seconds =
      duration_sum(candidate.norm_seconds, measurement.seconds);
  candidate.known_peak_scratch_bytes = std::max(
      candidate.known_peak_scratch_bytes, measurement.known_scratch_bytes);
  state_ = candidate;
}

std::map<std::string, double> TrainingDiagnostics::terminal_metrics() const {
  return {{"diagnostics.training.loss_nan",
           state_.training_loss.nan != 0 ? 1.0 : 0.0},
          {"diagnostics.training.loss_infinity",
           state_.training_loss.infinity != 0 ? 1.0 : 0.0},
          {"diagnostics.validation.loss_nan",
           state_.validation_loss.nan != 0 ? 1.0 : 0.0},
          {"diagnostics.validation.loss_infinity",
           state_.validation_loss.infinity != 0 ? 1.0 : 0.0},
          {"diagnostics.gradient.nonfinite",
           state_.nonfinite_gradient_attempts != 0 ? 1.0 : 0.0},
          {"diagnostics.amp.overflow",
           state_.amp_skipped_attempts != 0 ? 1.0 : 0.0}};
}

std::string TrainingDiagnostics::participant_id() const {
  return options_.participant_id;
}

StateManifest TrainingDiagnostics::describe_state() const {
  return {.participant_id = participant_id(),
          .schema_version = 1,
          .restore_level = RestoreLevel::exact,
          .fields =
              {{.name = "topology",
                .media_type =
                    "application/vnd.jyppx.training-diagnostics-topology-v1",
                .max_bytes = kTopologyBytes,
                .required = true},
               {.name = "statistics",
                .media_type =
                    "application/vnd.jyppx.training-diagnostics-statistics-v1",
                .max_bytes = kStateBytes,
                .required = true}},
          .degradation_reasons = {}};
}

void TrainingDiagnostics::save_state(StateWriter& writer) const {
  writer.write_blob("topology", topology(options_));
  writer.write_blob("statistics", encode(state_));
}

void TrainingDiagnostics::load_state(const StateReader& reader,
                                     RestorePolicy policy) {
  const auto expected = describe_state();
  const auto& stored = reader.manifest();
  validate_state_manifest(stored);
  if (stored.participant_id != expected.participant_id ||
      stored.schema_version != 1 ||
      stored.restore_level != RestoreLevel::exact ||
      !stored.degradation_reasons.empty() ||
      !restore_policy_accepts(policy, stored.restore_level) ||
      !state_fields_equal(stored.fields, expected.fields)) {
    throw CompatibilityError("Incompatible training diagnostics manifest");
  }
  const auto stored_topology = reader.read_blob("topology", kTopologyBytes);
  const auto expected_topology = topology(options_);
  if (!std::equal(stored_topology.begin(), stored_topology.end(),
                  expected_topology.begin(), expected_topology.end())) {
    throw CompatibilityError(
        "Incompatible training diagnostics enablement or bounds");
  }
  state_ = decode(reader.read_blob("statistics", kStateBytes), options_);
}

std::map<std::string, TrainingTelemetryMetricBinding>
training_diagnostic_metric_bindings() {
  std::map<std::string, TrainingTelemetryMetricBinding> result;
  for (const auto* key :
       {"diagnostics.training.loss_nan", "diagnostics.training.loss_infinity",
        "diagnostics.validation.loss_nan",
        "diagnostics.validation.loss_infinity",
        "diagnostics.gradient.nonfinite", "diagnostics.amp.overflow",
        "diagnostics.gradient.l2", "diagnostics.gradient.scan_seconds",
        "diagnostics.training.data_wait_seconds",
        "diagnostics.validation.data_wait_seconds"}) {
    const std::string name(key);
    const bool seconds = name.ends_with("seconds");
    const bool norm = name.ends_with(".l2");
    result.emplace(
        name,
        TrainingTelemetryMetricBinding{
            .definition =
                {.metric_id = "jyppx.liuding." + name,
                 .version = 1,
                 .unit = seconds ? TelemetryMetricUnit::seconds
                                 : TelemetryMetricUnit::dimensionless,
                 .aggregation = TelemetryAggregation::last,
                 .direction = TelemetryMetricDirection::none,
                 .parameters =
                     {{"measurement",
                       norm      ? "preclip-unscaled-tail-corrected-global-l2"
                       : seconds ? (name.find("data_wait") != std::string::npos
                                        ? "successful-nonempty-next-host-wall"
                                        : "norm-scan-host-wall")
                                 : "run-to-date-boolean-issue"}},
                 .context_keys = {}},
            .context = {}});
  }
  return result;
}

}  // namespace jyppx::liuding::training
