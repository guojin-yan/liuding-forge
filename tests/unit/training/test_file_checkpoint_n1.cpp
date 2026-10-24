// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::vector<std::byte> read_historical_fixture(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  std::string encoded;
  std::string trailing;
  if (!(input >> encoded) || (input >> trailing) || encoded.empty() ||
      encoded.size() % 4U != 0U) {
    throw jyppx::liuding::SerializationError(
        "Invalid historical SGD checkpoint fixture");
  }
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const auto decode = [&](char character) -> std::uint32_t {
    const auto position = alphabet.find(character);
    if (position == std::string_view::npos) {
      throw jyppx::liuding::SerializationError("Invalid base64 fixture character");
    }
    return static_cast<std::uint32_t>(position);
  };
  std::vector<std::byte> result;
  result.reserve(encoded.size() / 4U * 3U);
  for (std::size_t offset = 0; offset != encoded.size(); offset += 4U) {
    const bool final = offset + 4U == encoded.size();
    const bool third_padding = encoded[offset + 2U] == '=';
    const bool fourth_padding = encoded[offset + 3U] == '=';
    if (encoded[offset] == '=' || encoded[offset + 1U] == '=' ||
        (!final && (third_padding || fourth_padding)) ||
        (third_padding && !fourth_padding)) {
      throw jyppx::liuding::SerializationError("Invalid base64 fixture padding");
    }
    const auto packed =
        (decode(encoded[offset]) << 18U) |
        (decode(encoded[offset + 1U]) << 12U) |
        (third_padding ? 0U : decode(encoded[offset + 2U]) << 6U) |
        (fourth_padding ? 0U : decode(encoded[offset + 3U]));
    result.push_back(static_cast<std::byte>((packed >> 16U) & 0xffU));
    if (!third_padding) {
      result.push_back(static_cast<std::byte>((packed >> 8U) & 0xffU));
    }
    if (!fourth_padding) {
      result.push_back(static_cast<std::byte>(packed & 0xffU));
    }
  }
  return result;
}

jyppx::liuding::StateManifest historical_sgd_manifest() {
  return jyppx::liuding::StateManifest{
      .participant_id = "optimizer.main",
      .schema_version = 1,
      .restore_level = jyppx::liuding::RestoreLevel::exact,
      .fields = {{.name = "topology",
                  .media_type = "application/vnd.jyppx.sgd-topology-v1",
                  .max_bytes = 128,
                  .required = true},
                 {.name = "control",
                  .media_type = "application/vnd.jyppx.sgd-control-v1",
                  .max_bytes = 37,
                  .required = true},
                 {.name = "momentum.0000000000",
                  .media_type = "application/vnd.jyppx.tensor-raw-v1",
                  .max_bytes = 16,
                  .required = true}},
      .degradation_reasons = {},
  };
}

void write_historical_sgd_checkpoint(
    jyppx::liuding::training::FileCheckpointStore& store) {
  // These bytes came from the schema-v1 writer at commit 4b325732, rather
  // than being regenerated or relabeled by the current schema-v2 writer.
  const auto manifest = historical_sgd_manifest();
  store.begin({.restore_level = jyppx::liuding::RestoreLevel::exact,
               .participants = {manifest}});
  const auto fixture_directory =
      std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) / "checkpoint" /
      "sgd_schema_v1_libtorch_2.5.1";
  for (const auto& field : manifest.fields) {
    const auto bytes =
        read_historical_fixture(fixture_directory / (field.name + ".bin.b64"));
    store.writer_for(manifest.participant_id).write_blob(field.name, bytes);
  }
  store.commit();
}

std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter> make_sgd(
    const torch::Tensor& parameter, double learning_rate) {
  return std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
      "optimizer.main", std::vector<jyppx::liuding::optim::SgdParameterGroup>{
                            {.group_id = "main",
                             .parameters = {{.parameter_id = "layer.weight",
                                             .parameter = parameter}},
                             .options = {.learning_rate = learning_rate,
                                         .momentum = 0.8,
                                         .weight_decay = 0.02}}});
}

class FileCheckpointN1Test : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    path_ = std::filesystem::temp_directory_path() /
            (std::string("jyppx-file-checkpoint-n1-") + info->name() + ".bin");
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
    std::filesystem::remove(path_.string() + ".tmp", ignored);
  }

  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
    std::filesystem::remove(path_.string() + ".tmp", ignored);
  }

  std::filesystem::path path_;
};

TEST_F(FileCheckpointN1Test,
       MigratesHistoricalSgdFixtureAfterReopenAndWritesCurrentSchema) {
  if (TORCH_VERSION_MAJOR != 2 || TORCH_VERSION_MINOR != 5 ||
      TORCH_VERSION_PATCH != 1) {
    GTEST_SKIP() << "historical fixture requires exact LibTorch 2.5.1";
  }

  {
    jyppx::liuding::training::FileCheckpointStore store(path_);
    write_historical_sgd_checkpoint(store);
  }
  jyppx::liuding::training::FileCheckpointStore reopened(path_);
  ASSERT_EQ(reopened.descriptor().participants.size(), 1U);
  EXPECT_EQ(reopened.descriptor().participants.front().schema_version, 1U);
  EXPECT_FALSE(reopened.reader_for("optimizer.main").contains("producer"));

  auto reference_parameter = torch::tensor(
      {2.0, -1.0},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto reference = make_sgd(reference_parameter, 0.04);
  for (std::int64_t step = 1; step <= 2; ++step) {
    reference_parameter.mutable_grad() =
        torch::tensor({0.5 * step, -0.25 * step}, torch::kFloat64);
    reference->step();
  }
  reference->set_learning_rates({0.015});

  auto resumed_parameter =
      reference_parameter.detach().clone().set_requires_grad(true);
  auto resumed = make_sgd(resumed_parameter, 0.9);
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  const auto participant =
      std::make_shared<jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
          resumed);
  registry.register_participant(participant);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  coordinator.restore(reopened);
  EXPECT_EQ(resumed->completed_steps(), 2U);
  EXPECT_EQ(resumed->learning_rates(), (std::vector<double>{0.015}));

  const auto gradient = torch::tensor({-0.3, 0.7}, torch::kFloat64);
  reference_parameter.mutable_grad() = gradient.clone();
  resumed_parameter.mutable_grad() = gradient.clone();
  reference->step();
  resumed->step();
  EXPECT_TRUE(
      torch::allclose(reference_parameter, resumed_parameter, 1e-12, 1e-12));

  coordinator.save(reopened);
  jyppx::liuding::training::FileCheckpointStore migrated(path_);
  ASSERT_EQ(migrated.descriptor().participants.size(), 1U);
  EXPECT_EQ(migrated.descriptor().participants.front().schema_version, 2U);
  EXPECT_TRUE(migrated.reader_for("optimizer.main").contains("producer"));
}

TEST_F(FileCheckpointN1Test,
       PersistsExactHistoricalFixtureBytesBeforeMigration) {
  jyppx::liuding::training::FileCheckpointStore store(path_);
  write_historical_sgd_checkpoint(store);
  jyppx::liuding::training::FileCheckpointStore reopened(path_);
  const auto& reader = reopened.reader_for("optimizer.main");
  const auto directory = std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) /
                         "checkpoint" / "sgd_schema_v1_libtorch_2.5.1";
  for (const auto& field : historical_sgd_manifest().fields) {
    const auto expected =
        read_historical_fixture(directory / (field.name + ".bin.b64"));
    const auto actual = reader.read_blob(field.name, field.max_bytes);
    EXPECT_EQ(std::vector<std::byte>(actual.begin(), actual.end()), expected);
  }
}

}  // namespace
