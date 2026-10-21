// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "fuzz_support.hpp"

#include <jyppx/quantization/quantization.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace q = jyppx::quantization;
namespace fuzz = jyppx::fuzz;

constexpr std::size_t kMaxObserverBytes = 128U;

q::QuantizationLimits limits() {
  return {.max_artifact_bytes = fuzz::kMaxInputBytes,
          .max_features = 64U,
          .max_samples = 32U,
          .max_sample_id_bytes = 64U,
          .max_lineage_text_bytes = 256U};
}

std::uint64_t observer_budget(const jyppx::StateManifest& manifest) {
  fuzz::require(manifest.fields.size() == 1U &&
                    manifest.fields.front().name == "state" &&
                    manifest.fields.front().max_bytes > 1U &&
                    manifest.fields.front().max_bytes <= kMaxObserverBytes,
                "observer published an unbounded or unexpected state contract");
  return manifest.fields.front().max_bytes;
}

class ObserverWriter final : public jyppx::StateWriter {
 public:
  explicit ObserverWriter(std::uint64_t limit) : limit_(limit) {}

  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    fuzz::require(name == "state" && !written_ && bytes.size() <= limit_,
                  "observer wrote an unknown, repeated or oversized field");
    written_ = true;
    bytes_.assign(bytes.begin(), bytes.end());
  }

  std::vector<std::byte> finish() && {
    fuzz::require(written_ && bytes_.size() == limit_,
                  "observer state changed its fixed byte-size contract");
    return std::move(bytes_);
  }

 private:
  std::uint64_t limit_;
  bool written_ = false;
  std::vector<std::byte> bytes_;
};

class ObserverReader final : public jyppx::StateReader {
 public:
  ObserverReader(jyppx::StateManifest manifest,
                 std::span<const std::byte> bytes, std::uint64_t limit,
                 bool present = true)
      : manifest_(std::move(manifest)),
        bytes_(bytes),
        limit_(limit),
        present_(present) {}

  const jyppx::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  bool contains(std::string_view name) const override {
    return name == "state" && present_;
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    fuzz::require(contains(name) && max_bytes > 0U && max_bytes <= limit_,
                  "observer requested an undeclared or unbounded state read");
    requested_limit_ = max_bytes;
    if (bytes_.size() > max_bytes)
      throw jyppx::ResourceExhausted("Fuzz reader rejected oversized state");
    return bytes_;
  }

  std::uint64_t requested_limit() const noexcept { return requested_limit_; }

 private:
  jyppx::StateManifest manifest_;
  std::span<const std::byte> bytes_;
  std::uint64_t limit_;
  bool present_;
  mutable std::uint64_t requested_limit_ = 0U;
};

std::vector<std::byte> observer_bytes(const q::MinMaxObserver& observer) {
  ObserverWriter writer(observer_budget(observer.describe_state()));
  observer.save_state(writer);
  return std::move(writer).finish();
}

bool states_equal(const q::MinMaxObserverState& left,
                  const q::MinMaxObserverState& right) {
  return left.sample_count == right.sample_count &&
         left.value_count == right.value_count &&
         left.has_finite_value == right.has_finite_value &&
         left.minimum == right.minimum && left.maximum == right.maximum;
}

void exercise_artifact(std::span<const std::byte> input) {
  std::optional<q::QuantizedLinearArtifact> parsed;
  try {
    parsed = fuzz::parse_input(
        [&] { return q::deserialize_linear_artifact(input, limits()); });
  } catch (const jyppx::NumericalError& error) {
    // The v1 decoder reports nonfinite FP32 bias as a NumericalError. This
    // exception applies only to the initial untrusted input, never a roundtrip.
    if (error.details().component == "quantization" &&
        error.details().operation == "linear.bias")
      return;
    throw;
  }
  if (!parsed) return;

  q::validate_linear_artifact(*parsed, limits());
  const auto canonical = q::serialize_linear_artifact(*parsed, limits());
  fuzz::require(
      canonical.size() > 1U && canonical.size() == input.size() &&
          std::equal(canonical.begin(), canonical.end(), input.begin()),
      "quantization decoder accepted noncanonical artifact bytes");
  auto exact = limits();
  exact.max_artifact_bytes = canonical.size();
  fuzz::require(q::serialize_linear_artifact(*parsed, exact) == canonical,
                "quantization artifact changes at its exact byte budget");
  const auto reparsed = q::deserialize_linear_artifact(canonical, exact);
  fuzz::require(
      reparsed.in_features == parsed->in_features &&
          reparsed.out_features == parsed->out_features &&
          reparsed.quantized_weights == parsed->quantized_weights &&
          reparsed.calibration.sample_ids == parsed->calibration.sample_ids &&
          states_equal(reparsed.calibration.activation_state,
                       parsed->calibration.activation_state) &&
          q::serialize_linear_artifact(reparsed, exact) == canonical,
      "quantization artifact roundtrip changed canonical state");
  --exact.max_artifact_bytes;
  fuzz::require_resource_rejection([&] {
    static_cast<void>(q::deserialize_linear_artifact(canonical, exact));
  });
  fuzz::require_resource_rejection(
      [&] { static_cast<void>(q::serialize_linear_artifact(*parsed, exact)); });
}

void exercise_restore(std::span<const std::byte> input,
                      jyppx::StateManifest manifest,
                      jyppx::RestorePolicy policy, bool present) {
  q::MinMaxObserver observer;
  const std::array<float, 2U> sentinel{-3.0F, 4.0F};
  observer.observe(sentinel);
  const auto before = observer_bytes(observer);
  const auto before_state = observer.state();
  const auto participant = observer.participant_id();
  const auto budget = observer_budget(observer.describe_state());
  const ObserverReader reader(std::move(manifest), input, budget, present);
  const auto accepted = fuzz::parse_input([&] {
    observer.load_state(reader, policy);
    return true;
  });
  if (!accepted) {
    const auto after = observer.state();
    fuzz::require(observer.participant_id() == participant &&
                      states_equal(after, before_state) &&
                      std::bit_cast<std::uint32_t>(after.minimum) ==
                          std::bit_cast<std::uint32_t>(before_state.minimum) &&
                      std::bit_cast<std::uint32_t>(after.maximum) ==
                          std::bit_cast<std::uint32_t>(before_state.maximum) &&
                      observer_bytes(observer) == before,
                  "rejected observer restore partially mutated existing state");
    return;
  }

  const auto canonical = observer_bytes(observer);
  fuzz::require(input.size() == canonical.size() &&
                    reader.requested_limit() == budget &&
                    observer.participant_id() == participant,
                "observer restore ignored its exact bounded state contract");
  q::MinMaxObserver restored;
  const ObserverReader canonical_reader(restored.describe_state(), canonical,
                                        budget);
  // Signed zero in accepted state is normalized by the production encoder.
  // Any subsequent restore/encode failure must escape to libFuzzer.
  restored.load_state(canonical_reader, jyppx::RestorePolicy::exact_only);
  fuzz::require(states_equal(restored.state(), observer.state()) &&
                    observer_bytes(restored) == canonical,
                "observer canonical state roundtrip was not stable");
}

void exercise_observer(std::span<const std::byte> input) {
  const q::MinMaxObserver observer;
  auto manifest = observer.describe_state();
  exercise_restore(input, manifest, jyppx::RestorePolicy::exact_only, true);
  if (input.empty()) return;

  bool present = true;
  const auto control = std::to_integer<unsigned>(input.front());
  switch (control % 16U) {
    case 0U:
      break;
    case 1U:
      manifest.participant_id = "quantization.observer.other";
      break;
    case 2U:
      manifest.schema_version = 2U;
      break;
    case 3U:
      manifest.restore_level = jyppx::RestoreLevel::bounded;
      manifest.degradation_reasons = {"fuzz bounded restore"};
      break;
    case 4U:
      --manifest.fields.front().max_bytes;
      break;
    case 5U:
      manifest.fields.front().media_type = "application/octet-stream";
      break;
    case 6U:
      manifest.fields.front().required = false;
      break;
    case 7U:
      manifest.fields.clear();
      break;
    case 8U: {
      const auto field = manifest.fields.front();
      manifest.fields.push_back(field);
      break;
    }
    case 9U:
      manifest.fields.front().max_bytes = 0U;
      break;
    case 10U:
      manifest.degradation_reasons = {"unexpected exact degradation"};
      break;
    case 11U:
      present = false;
      break;
    case 12U:
      manifest.participant_id.assign(
          reinterpret_cast<const char*>(input.data()),
          (std::min)(std::size_t{16U}, input.size()));
      break;
    case 13U:
      manifest.fields.front().name = "other";
      break;
    case 14U:
      manifest.restore_level = static_cast<jyppx::RestoreLevel>(255U);
      break;
    default:
      ++manifest.fields.front().max_bytes;
      break;
  }
  const auto policy = input.size() < 2U
                          ? jyppx::RestorePolicy::exact_only
                          : static_cast<jyppx::RestorePolicy>(
                                std::to_integer<unsigned>(input[1U]) % 5U);
  exercise_restore(input, std::move(manifest), policy, present);
}

void exercise(std::span<const std::byte> input) {
  exercise_artifact(input);
  exercise_observer(input);
}

const std::vector<std::vector<std::byte>>& seeds() {
  static const auto value = [] {
    q::CalibrationSession calibration(
        {.dataset_version = "fuzz.calibration/v1",
         .dataset_digest = {.hex = std::string(64U, 'a')},
         .preprocessing_fingerprint = "identity-fp32/v1",
         .seed = 20261021U},
        limits());
    const std::array<float, 3U> first{-127.0F, -1.0F, -0.0F};
    const std::array<float, 2U> second{1.0F, 127.0F};
    calibration.observe("sample-a", first);
    calibration.observe("sample.\xe4\xb8\xad", second);
    const auto evidence = calibration.finalize();
    const std::array<float, 6U> weights{-127.0F, -64.0F, 0.0F,
                                        64.0F,   127.0F, 32.0F};
    const std::array<float, 2U> bias{-0.0F, 1.25F};
    const std::array<float, 1U> zeros{0.0F};
    std::vector<std::vector<std::byte>> result;
    result.push_back(
        q::serialize_linear_artifact(q::quantize_linear({.in_features = 3U,
                                                         .out_features = 2U,
                                                         .weights = weights,
                                                         .bias = bias},
                                                        evidence, limits()),
                                     limits()));
    result.push_back(
        q::serialize_linear_artifact(q::quantize_linear({.in_features = 1U,
                                                         .out_features = 1U,
                                                         .weights = zeros,
                                                         .bias = {}},
                                                        evidence, limits()),
                                     limits()));
    result.push_back(observer_bytes(q::MinMaxObserver{}));
    result.push_back(observer_bytes(calibration.observer()));
    q::MinMaxObserver nonfinite;
    const std::array<float, 3U> nonfinite_values{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    nonfinite.observe(nonfinite_values);
    result.push_back(observer_bytes(nonfinite));
    q::MinMaxObserver empty_sample;
    empty_sample.observe({});
    result.push_back(observer_bytes(empty_sample));
    q::MinMaxObserver finite_zero;
    finite_zero.observe(zeros);
    result.push_back(observer_bytes(finite_zero));
    for (const auto& seed : result) exercise(seed);
    return result;
  }();
  return value;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      std::size_t size) {
  return fuzz::exercise_raw_and_seed(data, size, seeds(), exercise, 8U);
}
