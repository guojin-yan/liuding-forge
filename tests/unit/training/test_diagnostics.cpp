// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/core/device.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/diagnostics.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <vector>

#ifdef JYPPX_LIUDING_TEST_DIAGNOSTIC_JSON
#include <nlohmann/json.hpp>
#endif
#ifdef JYPPX_LIUDING_TEST_DIAGNOSTIC_STORE
#include <jyppx/liuding/experiment/artifact_store.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#endif

namespace {
namespace training = jyppx::liuding::training;
using jyppx::liuding::Tensor;

class State final : public jyppx::liuding::StateWriter,
                    public jyppx::liuding::StateReader {
 public:
  explicit State(const training::TrainingDiagnostics& participant)
      : descriptor(participant.describe_state()) {}
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    fields[std::string(name)] = {data.begin(), data.end()};
  }
  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return descriptor;
  }
  bool contains(std::string_view name) const override {
    return fields.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t limit) const override {
    const auto& bytes = fields.at(std::string(name));
    if (bytes.size() > limit)
      throw jyppx::liuding::ResourceExhausted("test field exceeds limit");
    return bytes;
  }
  jyppx::liuding::StateManifest descriptor;
  std::map<std::string, std::vector<std::byte>> fields;
};

TEST(TrainingDiagnostics,
     ScaledDoubleTilesMatchIndependentHypotAndPreserveStrides) {
  for (const auto tile : {1U, 3U, 64U}) {
    auto base = torch::tensor({3.0, 0.0, 4.0, 0.0, 12.0, 0.0}, torch::kFloat64);
    const auto strided = base.slice(0, 0, 6, 2);
    ASSERT_FALSE(strided.is_contiguous());
    const std::array<Tensor, 3> values{strided,
                                       torch::zeros({0}, torch::kFloat32),
                                       torch::tensor({-5.0}, torch::kFloat32)};
    const auto before = base.clone();
    const auto result =
        training::measure_gradient_l2(values, {.tile_elements = tile});
    EXPECT_NEAR(result.l2, std::hypot(13.0, 5.0), 1e-13);
    EXPECT_EQ(result.elements, 4U);
    EXPECT_LE(result.known_scratch_bytes, 8U * tile + 16U);
    EXPECT_TRUE(torch::equal(base, before));
    EXPECT_EQ(strided.stride(0), 2);
  }
}

TEST(TrainingDiagnostics, AvoidsFloat32AndFloat64SquareOverflowAndUnderflow) {
  for (const auto scale : {1e-300, 1e30, 1e300}) {
    const std::array<Tensor, 1> values{
        torch::tensor({3.0 * scale, 4.0 * scale}, torch::kFloat64)};
    const auto result =
        training::measure_gradient_l2(values, {.tile_elements = 1});
    EXPECT_NEAR(result.l2 / scale, 5.0, 1e-14);
  }
  const std::array<Tensor, 1> fp32{torch::tensor({3e30F, 4e30F})};
  EXPECT_NEAR(training::measure_gradient_l2(fp32).l2 / 1e30, 5.0, 1e-6);
}

TEST(TrainingDiagnostics, EmptyAndZeroGradientsHaveZeroNorm) {
  EXPECT_DOUBLE_EQ(training::measure_gradient_l2({}).l2, 0.0);
  const std::array<Tensor, 1> zeros{torch::zeros({2, 3})};
  EXPECT_DOUBLE_EQ(training::measure_gradient_l2(zeros).l2, 0.0);
}

TEST(TrainingDiagnostics, RejectsCompleteInputBoundsAndUnsupportedValues) {
  const std::array<Tensor, 2> values{torch::ones({5}), torch::ones({6})};
  EXPECT_THROW(training::measure_gradient_l2(values, {.max_tensors = 1}),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(training::measure_gradient_l2(values, {.max_elements = 10}),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(training::measure_gradient_l2(
                   values, {.tile_elements = 3, .max_scratch_bytes = 39}),
               jyppx::liuding::ResourceExhausted);
  EXPECT_NO_THROW(training::measure_gradient_l2(
      values,
      {.max_elements = 11, .tile_elements = 3, .max_scratch_bytes = 40}));
  const std::array<Tensor, 1> integral{torch::ones({2}, torch::kInt64)};
  EXPECT_THROW(training::measure_gradient_l2(integral),
               jyppx::liuding::InvalidArgument);
  const std::array<Tensor, 1> undefined{Tensor{}};
  EXPECT_THROW(training::measure_gradient_l2(undefined),
               jyppx::liuding::InvalidArgument);
  const std::array<Tensor, 1> sparse{torch::eye(2).to_sparse()};
  EXPECT_THROW(training::measure_gradient_l2(sparse),
               jyppx::liuding::InvalidArgument);
  for (const auto value : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
    const std::array<Tensor, 1> nonfinite{
        torch::tensor({value}, torch::kFloat64)};
    EXPECT_THROW(training::measure_gradient_l2(nonfinite),
                 jyppx::liuding::NumericalError);
  }
  const std::array<Tensor, 1> too_large{
      torch::full({2}, std::numeric_limits<double>::max(), torch::kFloat64)};
  EXPECT_THROW(training::measure_gradient_l2(too_large),
               jyppx::liuding::NumericalError);
}

TEST(TrainingDiagnostics, OptionsRejectInvalidIdentityAndLimits) {
  EXPECT_THROW(training::TrainingDiagnostics({.participant_id = "../escape"}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      training::TrainingDiagnostics({.gradient_norm = {.tile_elements = 0}}),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(training::TrainingDiagnostics(
                   {.gradient_norm = {.max_scratch_bytes = 23}}),
               jyppx::liuding::InvalidArgument);
}

TEST(TrainingDiagnostics,
     CurrentCheckpointValidatesTopologyManifestAndStatisticsAtomically) {
  training::TrainingDiagnostics source;
  State original(source);
  source.save_state(original);
  ASSERT_EQ(original.fields.at("statistics").size(), 184U);
  for (int corruption = 0; corruption < 6; ++corruption) {
    auto changed = original;
    if (corruption == 0) changed.descriptor.schema_version = 2;
    if (corruption == 1) changed.fields.at("topology").back() ^= std::byte{1};
    if (corruption == 2) changed.fields.at("statistics").pop_back();
    if (corruption == 3)
      changed.fields.at("statistics")[16] =
          std::byte{1};  // NaN > observations.
    if (corruption == 4)
      changed.fields.at("statistics")[128] =
          std::byte{1};  // Samples with no attempts.
    if (corruption == 5) {
      const auto value = std::bit_cast<std::array<std::byte, 8>>(
          std::numeric_limits<double>::infinity());
      std::copy(value.begin(), value.end(),
                changed.fields.at("statistics").begin() + 144);
    }
    training::TrainingDiagnostics target;
    EXPECT_THROW(
        target.load_state(changed, jyppx::liuding::RestorePolicy::exact_only),
        jyppx::liuding::Error);
    State after(target);
    target.save_state(after);
    EXPECT_EQ(after.fields, original.fields);
  }
  training::TrainingDiagnostics changed_options({.gradient_norm_every = 2});
  EXPECT_THROW(changed_options.load_state(
                   original, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_NO_THROW(
      source.load_state(original, jyppx::liuding::RestorePolicy::exact_only));
}

TEST(TrainingDiagnostics, MetricDefinitionsHaveExplicitStableSemantics) {
  const auto bindings = training::training_diagnostic_metric_bindings();
  ASSERT_EQ(bindings.size(), 10U);
  for (const auto& [key, binding] : bindings) {
    SCOPED_TRACE(key);
    EXPECT_NO_THROW(jyppx::liuding::validate_telemetry_metric_definition(
        binding.definition));
    EXPECT_EQ(binding.definition.aggregation,
              jyppx::liuding::TelemetryAggregation::last);
    EXPECT_FALSE(binding.definition.parameters.empty());
  }
}

TEST(TrainingDiagnosticsCuda, NoncontiguousHalfAndDoubleTilesUseRealCuda) {
  if (!jyppx::liuding::cuda_is_available()) GTEST_SKIP() << "CUDA unavailable";
  for (const auto dtype :
       {torch::kHalf, torch::kBFloat16, torch::kFloat32, torch::kFloat64}) {
    auto base =
        torch::tensor({3.0, 0.0, 4.0, 0.0},
                      torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
    const std::array<Tensor, 1> values{base.slice(0, 0, 4, 2)};
    const auto before = base.clone();
    EXPECT_DOUBLE_EQ(
        training::measure_gradient_l2(values, {.tile_elements = 1}).l2, 5.0);
    EXPECT_TRUE(torch::equal(base, before));
  }
}

training::TrainingDiagnosticIdentity report_identity() {
  return {.report_id = "diagnostics.fixture",
          .run_id = "run.fixture",
          .attempt_id = "attempt.fixture",
          .model = {.hex = std::string(64, 'a')},
          .configuration = {.hex = std::string(64, 'b')},
          .dataset = {.hex = std::string(64, 'c')}};
}

training::TrainingDiagnosticsReport empty_report() {
  training::TrainingDiagnostics participant;
  return training::capture_training_diagnostics_report(participant,
                                                       report_identity());
}

void emit_report(std::string_view name, const std::string& bytes) {
  if (const auto* path = std::getenv("JYPPX_LIUDING_DIAGNOSTIC_REPORT_DIR")) {
    std::filesystem::create_directories(path);
    std::ofstream output(
        std::filesystem::path(path) / (std::string(name) + ".json"),
        std::ios::binary);
    output << bytes;
    ASSERT_TRUE(output.good());
  }
}

TEST(TrainingDiagnosticReport, CaptureOwnsDisabledStateAndActualBuild) {
  auto report = empty_report();
  EXPECT_EQ(report.build.git_commit,
            jyppx::liuding::current_build_fingerprint().git_commit);
  EXPECT_EQ(report.outcome, training::TrainingDiagnosticOutcome::incomplete);
  EXPECT_EQ(report.statistics.training_loss.observations, 0U);
  const auto bytes = training::training_diagnostics_report_json(report);
  EXPECT_EQ(bytes, training::training_diagnostics_report_json(report));
  EXPECT_NE(bytes.find("\"state\":\"disabled\""), std::string::npos);
  EXPECT_NE(bytes.find("\"last_l2\":null"), std::string::npos);
  EXPECT_NE(bytes.find("\"checkpoint\":null"), std::string::npos);
#ifdef JYPPX_LIUDING_TEST_DIAGNOSTIC_JSON
  const auto parsed = nlohmann::json::parse(bytes);
  EXPECT_EQ(parsed.at("schema_version"), 1);
  EXPECT_EQ(parsed.at("gradient").at("state"), "disabled");
  EXPECT_EQ(parsed.at("loss").at("training").at("state"), "no_samples");
  EXPECT_TRUE(
      parsed.at("data_wait").at("training").at("host_seconds").is_null());
#endif
  emit_report("disabled", bytes);
}

TEST(TrainingDiagnosticReport, DistinguishesNoSamplesFromMeasuredZero) {
  training::TrainingDiagnostics participant(
      {.gradient_norm_every = 1, .measure_data_wait = true});
  auto report = training::capture_training_diagnostics_report(
      participant, report_identity(),
      training::TrainingDiagnosticOutcome::completed);
  const auto pending = training::training_diagnostics_report_json(report);
  EXPECT_EQ(report.statistics.norm_samples, 0U);
  auto& state = report.statistics;
  state.update_attempts = state.norm_samples = 1;
  state.norm_elements = 2;
  state.known_peak_scratch_bytes = 32;
  state.training_wait.batches = 1;
  state.training_loss.observations = 1;
  state.norm_seconds = -0.0;
  const auto measured = training::training_diagnostics_report_json(report);
  EXPECT_NE(pending, measured);
  EXPECT_NE(measured.find("\"last_l2\":0"), std::string::npos);
  EXPECT_EQ(measured.find(":-0"), std::string::npos);
#ifdef JYPPX_LIUDING_TEST_DIAGNOSTIC_JSON
  const auto absent = nlohmann::json::parse(pending);
  const auto present = nlohmann::json::parse(measured);
  EXPECT_EQ(absent.at("gradient").at("state"), "no_samples");
  EXPECT_TRUE(absent.at("gradient").at("last_l2").is_null());
  EXPECT_EQ(present.at("gradient").at("state"), "measured");
  EXPECT_EQ(present.at("gradient").at("last_l2"), 0.0);
  EXPECT_EQ(present.at("data_wait").at("training").at("state"), "measured");
  EXPECT_EQ(present.at("data_wait").at("training").at("host_seconds"), 0.0);
  EXPECT_EQ(present.at("data_wait").at("validation").at("state"), "no_samples");
#endif
  emit_report("no-samples", pending);
  emit_report("measured-zero", measured);
}

TEST(TrainingDiagnosticReport, PreservesUint64CountsAndEscapesUtf8) {
  auto report = empty_report();
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  report.statistics.training_loss = {maximum, maximum - 1, 1};
  report.statistics.update_attempts = maximum;
  report.statistics.nonfinite_gradient_attempts = maximum - 1;
  report.statistics.amp_skipped_attempts = maximum - 2;
  report.outcome = training::TrainingDiagnosticOutcome::failed;
  report.identity.checkpoint =
      jyppx::liuding::ArtifactDigest{.hex = std::string(64, 'd')};
  report.build.compiler =
      std::string("compiler\"\\\n\t\0", 13) + "\xe5\x85\xad\xe4\xb8\x81";
  const auto bytes = training::training_diagnostics_report_json(report);
  EXPECT_NE(bytes.find("18446744073709551615"), std::string::npos);
#ifdef JYPPX_LIUDING_TEST_DIAGNOSTIC_JSON
  const auto parsed = nlohmann::json::parse(bytes);
  EXPECT_EQ(
      parsed.at("loss").at("training").at("observations").get<std::uint64_t>(),
      maximum);
  EXPECT_EQ(
      parsed.at("gradient").at("amp_skipped_attempts").get<std::uint64_t>(),
      maximum - 2);
  EXPECT_EQ(parsed.at("build").at("compiler").get<std::string>(),
            report.build.compiler);
  EXPECT_EQ(parsed.at("identity").at("checkpoint").at("hex"),
            std::string(64, 'd'));
#endif
  emit_report("large-counts-utf8", bytes);
}

TEST(TrainingDiagnosticReport, EnforcesOutputByteLimitBeforeEveryAppend) {
  const auto report = empty_report();
  const auto bytes = training::training_diagnostics_report_json(report);
  EXPECT_EQ(training::training_diagnostics_report_json(report, bytes.size()),
            bytes);
  EXPECT_EQ(training::training_diagnostics_report_json(
                report, std::numeric_limits<std::size_t>::max()),
            bytes);
  for (const auto limit : {std::size_t{0}, std::size_t{1}, bytes.size() - 1})
    EXPECT_THROW(training::training_diagnostics_report_json(report, limit),
                 jyppx::liuding::ResourceExhausted);
}

TEST(TrainingDiagnosticReport, RejectsInvalidIdentityOptionsAndSnapshot) {
  const auto original = empty_report();
  for (int corruption = 0; corruption < 16; ++corruption) {
    SCOPED_TRACE(corruption);
    auto report = original;
    switch (corruption) {
      case 0:
        report.schema_version = 2;
        break;
      case 1:
        report.identity.report_id = "../escape";
        break;
      case 2:
        report.identity.run_id.clear();
        break;
      case 3:
        report.identity.attempt_id = std::string(1025, 'x');
        break;
      case 4:
        report.identity.model.hex[0] = 'G';
        break;
      case 5:
        report.identity.configuration.algorithm =
            static_cast<jyppx::liuding::ArtifactDigestAlgorithm>(99);
        break;
      case 6:
        report.identity.dataset.hex.pop_back();
        break;
      case 7:
        report.identity.checkpoint = jyppx::liuding::ArtifactDigest{};
        break;
      case 8:
        report.outcome = static_cast<training::TrainingDiagnosticOutcome>(99);
        break;
      case 9:
        report.options.gradient_norm.tile_elements = 0;
        break;
      case 10:
        report.statistics.training_loss.nan = 1;
        break;
      case 11:
        report.statistics.amp_skipped_attempts = 1;
        break;
      case 12:
        report.statistics.norm_seconds =
            std::numeric_limits<double>::infinity();
        break;
      case 13:
        report.statistics.training_wait.batches = 1;
        break;
      case 14:
        report.build.git_commit.clear();
        break;
      case 15:
        report.build.compiler = std::string(4097, 'a');
        break;
    }
    EXPECT_THROW(training::training_diagnostics_report_json(report),
                 jyppx::liuding::Error);
  }
  training::TrainingDiagnostics participant;
  auto identity = report_identity();
  identity.run_id = "run/invalid";
  EXPECT_THROW(
      training::capture_training_diagnostics_report(participant, identity),
      jyppx::liuding::InvalidArgument);
}

TEST(TrainingDiagnosticReport, RejectsMalformedUtf8AndAcceptsScalarBoundaries) {
  const auto original = empty_report();
  for (const auto invalid : {"\x80", "\xc0\x80", "\xe0\x80\x80", "\xed\xa0\x80",
                             "\xf4\x90\x80\x80", "\xf0\x90", "\xe2\x28\xa1"}) {
    auto report = original;
    report.build.compiler = invalid;
    EXPECT_THROW(training::training_diagnostics_report_json(report),
                 jyppx::liuding::InvalidArgument);
  }
  auto valid = original;
  valid.build.compiler =
      "\xc2\x80\xe0\xa0\x80\xed\x9f\xbf\xee\x80\x80\xf0\x90\x80\x80\xf4\x8f\xbf"
      "\xbf";
  EXPECT_NO_THROW(training::training_diagnostics_report_json(valid));
}

TEST(TrainingDiagnosticReport, CheckpointReconstructionProducesIdenticalBytes) {
  training::TrainingDiagnostics source({.gradient_norm_every = 2});
  State saved(source);
  source.save_state(saved);
  // A valid checkpoint with a real exact uint64 count above double precision.
  const std::uint64_t count = (std::uint64_t{1} << 53U) + 17;
  for (unsigned i = 0; i < 8; ++i)
    saved.fields.at("statistics")[8 + i] =
        static_cast<std::byte>((count >> (8U * i)) & 0xffU);
  source.load_state(saved, jyppx::liuding::RestorePolicy::exact_only);
  const auto expected =
      training::capture_training_diagnostics_report(source, report_identity());
  training::TrainingDiagnostics reconstructed({.gradient_norm_every = 2});
  reconstructed.load_state(saved, jyppx::liuding::RestorePolicy::exact_only);
  const auto actual = training::capture_training_diagnostics_report(
      reconstructed, report_identity());
  EXPECT_EQ(actual.statistics.training_loss.observations, count);
  EXPECT_EQ(training::training_diagnostics_report_json(expected),
            training::training_diagnostics_report_json(actual));
}

#if JYPPX_LIUDING_COMPONENT_INTEROP_ENABLED
TEST(TrainingDiagnosticReport, FileCheckpointReopenReconstructsReport) {
  const auto root =
      std::filesystem::temp_directory_path() /
      ("liuding-diagnostic-state-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  ASSERT_TRUE(std::filesystem::create_directory(root));
  struct Cleanup {
    std::filesystem::path root;
    ~Cleanup() {
      std::error_code error;
      std::filesystem::remove_all(root, error);
    }
  } cleanup{root};
  std::string expected;
  {
    auto source = std::make_shared<training::TrainingDiagnostics>();
    State state(*source);
    source->save_state(state);
    state.fields.at("statistics")[8] = std::byte{7};
    state.fields.at("statistics")[16] = std::byte{2};
    source->load_state(state, jyppx::liuding::RestorePolicy::exact_only);
    expected = training::training_diagnostics_report_json(
        training::capture_training_diagnostics_report(
            *source, report_identity(),
            training::TrainingDiagnosticOutcome::failed));
    training::CheckpointParticipantRegistry registry;
    registry.register_participant(source);
    training::CheckpointCoordinator coordinator(registry);
    training::FileCheckpointStore file(root / "diagnostics.state");
    coordinator.save(file);
  }
  auto restored = std::make_shared<training::TrainingDiagnostics>();
  training::CheckpointParticipantRegistry registry;
  registry.register_participant(restored);
  training::CheckpointCoordinator coordinator(registry);
  training::FileCheckpointStore file(root / "diagnostics.state");
  coordinator.restore(file);
  EXPECT_EQ(restored->snapshot().training_loss.observations, 7U);
  EXPECT_EQ(restored->snapshot().training_loss.nan, 2U);
  EXPECT_EQ(training::training_diagnostics_report_json(
                training::capture_training_diagnostics_report(
                    *restored, report_identity(),
                    training::TrainingDiagnosticOutcome::failed)),
            expected);
}
#endif

#ifdef JYPPX_LIUDING_TEST_DIAGNOSTIC_STORE
TEST(TrainingDiagnosticReport, RunCasReopenRetryAndConflictPreserveExactBytes) {
  namespace experiment = jyppx::liuding::experiment;
  const auto root =
      std::filesystem::temp_directory_path() /
      ("liuding-diagnostic-report-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  ASSERT_TRUE(std::filesystem::create_directory(root));
  struct Cleanup {
    std::filesystem::path root;
    ~Cleanup() {
      std::error_code error;
      std::filesystem::remove_all(root, error);
    }
  } cleanup{root};
  auto report = empty_report();
  report.outcome = training::TrainingDiagnosticOutcome::cancelled;
  const auto bytes = training::training_diagnostics_report_json(report);
  const auto publish = [&](const std::string& content) {
    experiment::SQLiteRunStore store(root);
    store.start({.run_id = report.identity.run_id,
                 .project_id = "project.fixture",
                 .experiment_id = "experiment.fixture",
                 .display_name = "diagnostics",
                 .created_at_unix_ms = 10,
                 .updated_at_unix_ms = 10});
    return store.put_artifact(
        report.identity.run_id, report.identity.report_id,
        experiment::RunArtifactKind::artifact, "training.diagnostics",
        std::as_bytes(std::span(content)),
        {.logical_name = "training-diagnostics.json",
         .media_type =
             "application/vnd.jyppx.training-diagnostics+json;version=1",
         .producer = "training.diagnostics/v1",
         .expected_size_bytes = content.size()},
        20);
  };
  const auto reference = publish(bytes);
  EXPECT_EQ(publish(bytes).artifact.digest, reference.artifact.digest);
  {
    experiment::SQLiteRunStore reopened(root);
    ASSERT_EQ(reopened.references(report.identity.run_id).size(), 1U);
    EXPECT_TRUE(reopened.integrity_check().ok());
    experiment::FileArtifactStore cas(root / "artifacts");
    auto reader = cas.open(reference.artifact.digest);
    std::vector<std::byte> stored(
        static_cast<std::size_t>(reader->size_bytes()));
    std::size_t offset = 0;
    while (offset < stored.size()) {
      const auto read = reader->read(std::span(stored).subspan(offset));
      ASSERT_NE(read, 0U);
      offset += read;
    }
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(stored.data()),
                          stored.size()),
              bytes);
  }
  report.identity.model.hex[0] = 'f';
  EXPECT_THROW(publish(training::training_diagnostics_report_json(report)),
               jyppx::liuding::InvalidArgument);
  experiment::SQLiteRunStore reopened(root);
  ASSERT_EQ(reopened.references(report.identity.run_id).size(), 1U);
  EXPECT_EQ(reopened.references(report.identity.run_id).front().artifact.digest,
            reference.artifact.digest);
}
#endif
}  // namespace
