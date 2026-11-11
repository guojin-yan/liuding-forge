// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "fixture.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/training/diagnostics.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <bit>
#include <fstream>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace diagnostic_recovery {
namespace training = jyppx::liuding::training;
namespace {
training::TrainingDiagnosticsOptions options() {
  return {.gradient_norm_every = 2, .measure_data_wait = true};
}

// A valid, deliberately nonzero serialized fixture, not a training result.
// Preserve the participant's own topology, magic and manifest validation.
class FixtureState final : public jyppx::liuding::StateWriter,
                           public jyppx::liuding::StateReader {
 public:
  explicit FixtureState(const training::TrainingDiagnostics& source)
      : manifest_(source.describe_state()) {}
  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }
  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    fields_[std::string(name)] = {bytes.begin(), bytes.end()};
  }
  bool contains(std::string_view name) const override {
    return fields_.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t maximum) const override {
    const auto& bytes = fields_.at(std::string(name));
    require(bytes.size() <= maximum, "fixture exceeds reader limit");
    return bytes;
  }
  void integer(std::size_t word, std::uint64_t value) {
    auto& bytes = fields_.at("statistics");
    require(bytes.size() == 184 && word < 23, "invalid fixture state size");
    for (unsigned i = 0; i != 8; ++i)
      bytes.at(word * 8 + i) = static_cast<std::byte>((value >> (i * 8)) & 255);
  }
  void number(std::size_t word, double value) {
    integer(word, std::bit_cast<std::uint64_t>(value));
  }

 private:
  jyppx::liuding::StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> fields_;
};
}  // namespace

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(input.is_open(), "required input cannot be opened");
  const auto size = input.tellg();
  require(size >= 0 && size <= 65536, "fixture input exceeds bound");
  std::string bytes(static_cast<std::size_t>(size), '\0');
  input.seekg(0);
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  require(static_cast<bool>(input), "fixture input could not be read");
  return bytes;
}

void write_file(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.close();
  require(static_cast<bool>(output), "fixture output could not be written");
}

void seed_checkpoint(const std::filesystem::path& root) {
  auto source = std::make_shared<training::TrainingDiagnostics>(options());
  FixtureState state(*source);
  source->save_state(state);
  state.integer(1, (std::uint64_t{1} << 53U) + 17);
  state.integer(2, 2);
  state.integer(3, 1);
  state.integer(4, 11);
  state.integer(5, 1);
  state.integer(6, 2);
  state.integer(7, 7);
  state.number(8, 0.75);
  state.number(9, 0.25);
  state.integer(10, 3);
  state.number(11, 0.5);
  state.number(12, 0.25);
  state.integer(13, 9);
  state.integer(14, 3);
  state.integer(15, 2);
  state.integer(16, 3);
  state.integer(17, 12);
  state.number(18, 4);
  state.number(19, 3);
  state.number(20, 4);
  state.number(21, 0.125);
  state.integer(22, 48);
  source->load_state(state, jyppx::liuding::RestorePolicy::exact_only);
  training::CheckpointParticipantRegistry registry;
  registry.register_participant(source);
  training::CheckpointCoordinator coordinator(registry);
  training::FileCheckpointStore file(root / "diagnostics.jyck");
  coordinator.save(file);
}

std::string checkpoint_report(const std::filesystem::path& root,
                              bool conflicting_model) {
  auto restored = std::make_shared<training::TrainingDiagnostics>(options());
  training::CheckpointParticipantRegistry registry;
  registry.register_participant(restored);
  training::CheckpointCoordinator coordinator(registry);
  training::FileCheckpointStore file(root / "diagnostics.jyck");
  coordinator.restore(file);
  const auto checkpoint = read_file(root / "diagnostics.jyck");
  const training::TrainingDiagnosticIdentity identity{
      .report_id = "diagnostics-1",
      .run_id = "run-1",
      .attempt_id = "attempt-1",
      .model = {.hex = std::string(64, conflicting_model ? 'd' : 'a')},
      .configuration = {.hex = std::string(64, 'b')},
      .dataset = {.hex = std::string(64, 'c')},
      .checkpoint = jyppx::liuding::ArtifactDigest{
          .hex = jyppx::liuding::sha256_hex(
              std::as_bytes(std::span(checkpoint)))}};
  return training::training_diagnostics_report_json(
      training::capture_training_diagnostics_report(
          *restored, identity, training::TrainingDiagnosticOutcome::failed));
}
}  // namespace diagnostic_recovery
