// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_CLI_RECIPE_BN_CALIBRATION_HPP
#define JYPPX_LIUDING_CLI_RECIPE_BN_CALIBRATION_HPP
/** @file recipe_bn_calibration.hpp
 * @brief Recipe-only whole-stage recovery, with no partial BN cursor.
 */
#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <utility>

namespace jyppx::liuding::cli {

class RecipeBatchNormCalibration final : public CheckpointParticipant {
 public:
  enum class Phase { training, ready, completed };
  explicit RecipeBatchNormCalibration(std::string contract)
      : contract_(std::move(contract)) {
    if (contract_.empty() || contract_.size() > 65536 ||
        nlohmann::json::parse(contract_).dump() != contract_)
      throw InvalidArgument("Invalid canonical BN calibration contract");
    validate_state_manifest(describe_state());
  }
  std::string participant_id() const override {
    return "recipe.batch-norm-calibration";
  }
  StateManifest describe_state() const override {
    return {.participant_id = participant_id(),
            .schema_version = 1,
            .restore_level = RestoreLevel::exact,
            .fields = {{.name = "contract",
                        .media_type = "application/json",
                        .max_bytes = contract_.size()},
                       {.name = "control",
                        .media_type = "application/json",
                        .max_bytes = 2048}},
            .degradation_reasons = {}};
  }
  void save_state(StateWriter& writer) const override {
    writer.write_blob("contract", bytes(contract_));
    const auto control = report().dump();
    writer.write_blob("control", bytes(control));
  }
  void load_state(const StateReader& reader, RestorePolicy policy) override {
    const auto expected = describe_state();
    const auto& stored = reader.manifest();
    validate_state_manifest(stored);
    if (!restore_policy_accepts(policy, RestoreLevel::exact) ||
        stored.participant_id != expected.participant_id ||
        stored.schema_version != expected.schema_version ||
        stored.restore_level != RestoreLevel::exact ||
        !stored.degradation_reasons.empty() ||
        !state_fields_equal(stored.fields, expected.fields))
      throw CompatibilityError("Incompatible BN calibration participant");
    const auto contract = reader.read_blob("contract", contract_.size());
    if (text(contract) != contract_)
      throw CompatibilityError("BN calibration input or method has changed");
    const auto encoded = reader.read_blob("control", 2048);
    try {
      const auto raw_control = text(encoded);
      const auto control = nlohmann::json::parse(raw_control);
      if (!control.is_object() || control.size() != 7 ||
          !control.at("phase").is_string() ||
          !control.at("source_model_sha256").is_string() ||
          !control.at("calibrated_model_sha256").is_string() ||
          !control.at("optimizer_updates").is_number_unsigned() ||
          !control.at("batches").is_number_unsigned() ||
          !control.at("samples").is_number_unsigned() ||
          !control.at("schema_version").is_number_unsigned() ||
          control.at("schema_version") != 1)
        throw SerializationError("Malformed BN calibration control");
      if (control.dump() != raw_control)
        throw SerializationError(
            "BN calibration control must be canonical JSON");
      const auto name = control.at("phase").get<std::string>();
      const auto phase =
          name == "training" ? Phase::training
          : name == "ready"  ? Phase::ready
          : name == "completed"
              ? Phase::completed
              : throw SerializationError("Unknown calibration phase");
      auto source = control.at("source_model_sha256").get<std::string>();
      auto calibrated =
          control.at("calibrated_model_sha256").get<std::string>();
      const auto updates = control.at("optimizer_updates").get<std::uint64_t>();
      const auto batches = control.at("batches").get<std::uint64_t>();
      const auto samples = control.at("samples").get<std::uint64_t>();
      if (phase == Phase::training) {
        if (!source.empty() || !calibrated.empty() || updates || batches ||
            samples)
          throw SerializationError("Training phase contains calibration state");
      } else {
        validate_artifact_digest({.hex = source});
        if (updates == 0)
          throw SerializationError(
              "Calibration phase lacks training completion");
        if (phase == Phase::ready) {
          if (!calibrated.empty() || batches || samples)
            throw SerializationError("READY contains a partial calibration");
        } else {
          validate_artifact_digest({.hex = calibrated});
          if (!batches || samples < batches)
            throw SerializationError("COMPLETED lacks finite pass counts");
        }
      }
      // Candidate decoding/allocation/validation completed before mutation.
      source_ = std::move(source);
      calibrated_ = std::move(calibrated);
      updates_ = updates;
      batches_ = batches;
      samples_ = samples;
      phase_ = phase;
    } catch (const nlohmann::json::exception&) {
      throw SerializationError("Malformed BN calibration JSON control");
    }
  }
  Phase phase() const noexcept { return phase_; }
  const std::string& source_model_sha256() const noexcept { return source_; }
  const std::string& calibrated_model_sha256() const noexcept {
    return calibrated_;
  }
  std::uint64_t optimizer_updates() const noexcept { return updates_; }
  void ready(std::string source, std::uint64_t updates) {
    if (phase_ != Phase::training || updates == 0)
      throw InvalidArgument("Calibration READY requires completed training");
    validate_artifact_digest({.hex = source});
    source_ = std::move(source);
    updates_ = updates;
    phase_ = Phase::ready;
  }
  void completed(std::string model, std::uint64_t batches,
                 std::uint64_t samples) {
    if (phase_ != Phase::ready || !batches || samples < batches)
      throw InvalidArgument("Calibration COMPLETED requires one complete pass");
    validate_artifact_digest({.hex = model});
    calibrated_ = std::move(model);
    batches_ = batches;
    samples_ = samples;
    phase_ = Phase::completed;
  }
  void discard_completion() noexcept {
    calibrated_.clear();
    batches_ = samples_ = 0;
    phase_ = Phase::ready;
  }
  nlohmann::json report() const {
    return {{"schema_version", 1},
            {"phase", phase_ == Phase::training ? "training"
                      : phase_ == Phase::ready  ? "ready"
                                                : "completed"},
            {"source_model_sha256", source_},
            {"calibrated_model_sha256", calibrated_},
            {"optimizer_updates", updates_},
            {"batches", batches_},
            {"samples", samples_}};
  }

 private:
  static std::span<const std::byte> bytes(const std::string& value) {
    return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
  }
  static std::string text(std::span<const std::byte> value) {
    return {reinterpret_cast<const char*>(value.data()), value.size()};
  }
  std::string contract_, source_, calibrated_;
  Phase phase_ = Phase::training;
  std::uint64_t updates_ = 0, batches_ = 0, samples_ = 0;
};
}  // namespace jyppx::liuding::cli
#endif
