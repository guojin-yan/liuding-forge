// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/data/data_pipeline.hpp>
#include <jyppx/data/dataset.hpp>
#include <jyppx/data/samplers/random_sampler.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/data/samplers/weighted_random_sampler.hpp>
#include <jyppx/foundation/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class TensorRowsDataset final : public jyppx::data::Dataset<jyppx::Tensor> {
 public:
  explicit TensorRowsDataset(jyppx::Tensor rows) : rows_(std::move(rows)) {}

  jyppx::Tensor get(std::size_t index) const override {
    return rows_[static_cast<std::int64_t>(index)];
  }

  std::size_t size() const override {
    return static_cast<std::size_t>(rows_.size(0));
  }

 private:
  jyppx::Tensor rows_;  ///< Tensor rows indexed by the sampler.
};

class MemoryStateStore final : public jyppx::StateWriter,
                               public jyppx::StateReader {
 public:
  explicit MemoryStateStore(jyppx::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::validate_state_manifest(manifest_);
  }

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const jyppx::StateFieldDescriptor& candidate) {
                       return candidate.name == name;
                     });
    if (field == manifest_.fields.end() || data.size() > field->max_bytes) {
      throw std::runtime_error("invalid DataPipeline state field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw std::runtime_error("duplicate DataPipeline state field");
    }
  }

  const jyppx::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto found = blobs_.find(std::string(name));
    if (found == blobs_.end() || found->second.size() > max_bytes) {
      throw std::runtime_error("missing DataPipeline state field");
    }
    return found->second;
  }

  /**
   * @brief Appends one byte to a stored field for corruption testing.
   * @param name Existing field name to corrupt.
   */
  void append_byte(std::string_view name) {
    blobs_.at(std::string(name)).push_back(std::byte{0});
  }

  /**
   * @brief Rewrites a little-endian integer relative to a field's end.
   * @param name Existing field name to mutate.
   * @param trailing_bytes Bytes following the integer in the field.
   * @param value Replacement unsigned integer value.
   */
  void overwrite_u64_from_end(std::string_view name, std::size_t trailing_bytes,
                              std::uint64_t value) {
    auto& blob = blobs_.at(std::string(name));
    if (blob.size() < sizeof(value) + trailing_bytes) {
      throw std::runtime_error("DataPipeline test field is too short");
    }
    const auto offset = blob.size() - sizeof(value) - trailing_bytes;
    for (unsigned int shift = 0; shift < 64; shift += 8) {
      blob[offset + shift / 8] =
          static_cast<std::byte>(static_cast<std::uint8_t>(value >> shift));
    }
  }

 private:
  jyppx::StateManifest manifest_;  ///< Manifest enforced by this store.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Stored participant fields.
};

/** @brief Stateful sampler used to verify failed epoch preparation rollback. */
class FaultingSampler final : public jyppx::data::samplers::Sampler {
 public:
  /** @brief Returns either a valid epoch or an intentionally short epoch. */
  std::vector<std::size_t> indices() const override {
    return fail_ ? std::vector<std::size_t>{0U, 1U}
                 : std::vector<std::size_t>{0U, 1U, 2U, 3U, 4U, 5U};
  }

  /** @brief Returns the declared epoch cardinality. */
  std::size_t size() const override { return 6U; }

  /** @brief Records the epoch before index materialization. */
  void set_epoch(std::uint64_t epoch) override { epoch_ = epoch; }

  /** @brief Reports deterministic epoch selection. */
  bool supports_epoch_control() const override { return true; }

  /** @brief Returns a stable test sampler identity. */
  std::string configuration_identity() const override {
    return "test.faulting-sampler/v1:size=6";
  }

  /** @brief Reports exact state support for the current epoch value. */
  bool supports_state() const override { return true; }

  /** @brief Serializes the selected epoch. */
  std::string save_state() const override { return std::to_string(epoch_); }

  /** @brief Restores the selected epoch. */
  void load_state(const std::string& state) override {
    epoch_ = std::stoull(state);
  }

  /** @brief Enables or disables the short-epoch failure. */
  void set_failure(bool fail) { fail_ = fail; }

  /** @brief Returns the epoch currently held by the sampler. */
  std::uint64_t epoch() const { return epoch_; }

 private:
  std::uint64_t epoch_ = 0;  ///< Epoch mutated by set_epoch and checkpointing.
  bool fail_ = false;        ///< Whether indices() violates its size contract.
};

std::shared_ptr<TensorRowsDataset> make_dataset() {
  return std::make_shared<TensorRowsDataset>(
      torch::arange(0, 12, torch::TensorOptions().dtype(torch::kFloat32))
          .view({6, 2}));
}

jyppx::data::DataBatchFactory make_factory(
    const std::shared_ptr<TensorRowsDataset>& dataset) {
  return [dataset](std::span<const std::size_t> indices) {
    std::vector<jyppx::Tensor> rows;
    rows.reserve(indices.size());
    for (const auto index : indices) {
      rows.push_back(dataset->get(index));
    }
    return c10::IValue(torch::stack(rows));
  };
}

MemoryStateStore save_state(const jyppx::data::DataPipeline& pipeline) {
  MemoryStateStore store(pipeline.describe_state());
  pipeline.save_state(store);
  return store;
}

TEST(DataPipelineTest, RequiresExplicitEpochAndDoesNotImplicitlyAdvance) {
  auto dataset = make_dataset();
  auto pipeline = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::RandomSampler>(6, 42),
      {.pipeline_id = "data.pipeline.test",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      make_factory(dataset));

  EXPECT_THROW(pipeline.next(), jyppx::InvalidArgument);
  pipeline.set_epoch(3);
  ASSERT_TRUE(pipeline.next().has_value());
  ASSERT_TRUE(pipeline.next().has_value());
  ASSERT_TRUE(pipeline.next().has_value());
  EXPECT_FALSE(pipeline.next().has_value());
  EXPECT_FALSE(pipeline.next().has_value());

  auto peer = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::RandomSampler>(6, 42),
      {.pipeline_id = "data.pipeline.test",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      make_factory(dataset));
  peer.set_epoch(3);
  EXPECT_EQ(pipeline.state().epoch_indices, peer.state().epoch_indices);
}

TEST(DataPipelineCheckpointTest, RestoresExactInEpochCursor) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::RandomSampler>(6, 17),
      {.pipeline_id = "data.pipeline.restore",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(4);
  ASSERT_TRUE(source.next().has_value());
  auto store = save_state(source);

  auto target = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::RandomSampler>(6, 91),
      {.pipeline_id = "data.pipeline.restore",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  target.load_state(store, jyppx::RestorePolicy::exact_only);
  EXPECT_EQ(target.state().next_cursor, 2U);
  EXPECT_EQ(target.state().next_sequence, 1U);
  while (true) {
    const auto expected = source.next();
    const auto actual = target.next();
    ASSERT_EQ(actual.has_value(), expected.has_value());
    if (!expected.has_value()) {
      break;
    }
    EXPECT_EQ(actual->sample_ids, expected->sample_ids);
    EXPECT_EQ(actual->sequence, expected->sequence);
    EXPECT_TRUE(
        torch::equal(actual->value.toTensor(), expected->value.toTensor()));
  }

  source.set_epoch(9);
  target.set_epoch(9);
  EXPECT_EQ(target.state().epoch_indices, source.state().epoch_indices);
}

TEST(DataPipelineTest, FactoryFailureDoesNotAdvanceCursor) {
  auto dataset = make_dataset();
  bool fail = true;
  auto pipeline = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.failure",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      [&fail, dataset](std::span<const std::size_t> indices) {
        if (fail) {
          fail = false;
          throw std::runtime_error("factory failure");
        }
        std::vector<jyppx::Tensor> rows;
        for (const auto index : indices) {
          rows.push_back(dataset->get(index));
        }
        return c10::IValue(torch::stack(rows));
      });
  pipeline.set_epoch(0);
  EXPECT_THROW(pipeline.next(), std::runtime_error);
  EXPECT_EQ(pipeline.state().next_cursor, 0U);
  ASSERT_TRUE(pipeline.next().has_value());
  EXPECT_EQ(pipeline.state().next_cursor, 2U);
}

TEST(DataPipelineTest, CancellationAndDropLastDoNotOverCommit) {
  auto dataset = make_dataset();
  auto pipeline = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.cancel",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 4,
       .drop_last = true},
      make_factory(dataset));
  pipeline.set_epoch(0);

  std::stop_source cancelled;
  cancelled.request_stop();
  EXPECT_FALSE(pipeline.next(cancelled.get_token()).has_value());
  EXPECT_EQ(pipeline.state().next_cursor, 0U);

  const auto batch = pipeline.next();
  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(batch->sample_ids, std::vector<std::uint64_t>({0U, 1U, 2U, 3U}));
  EXPECT_EQ(pipeline.state().next_cursor, 4U);
  EXPECT_FALSE(pipeline.next().has_value());
}

TEST(DataPipelineTest, FailedEpochPreparationRollsBackSamplerAndCursor) {
  auto dataset = make_dataset();
  auto sampler = std::make_shared<FaultingSampler>();
  auto pipeline =
      jyppx::data::DataPipeline(sampler,
                                {.pipeline_id = "data.pipeline.rollback",
                                 .dataset_version = "dataset/v1",
                                 .dataset_size = 6,
                                 .schema_id = "tensor.rows/v1",
                                 .batch_size = 2},
                                make_factory(dataset));
  pipeline.set_epoch(3);
  ASSERT_TRUE(pipeline.next().has_value());
  const auto before = pipeline.state();

  sampler->set_failure(true);
  EXPECT_THROW(pipeline.set_epoch(8), jyppx::InvalidArgument);
  EXPECT_EQ(sampler->epoch(), 3U);
  EXPECT_EQ(pipeline.state().epoch, before.epoch);
  EXPECT_EQ(pipeline.state().next_cursor, before.next_cursor);
  EXPECT_EQ(pipeline.state().epoch_indices, before.epoch_indices);
}

TEST(DataPipelineTest, MissingSamplerIdentityDowngradesRestoreGuarantee) {
  auto dataset = make_dataset();
  auto pipeline = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::WeightedRandomSampler>(
          std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 6, true, 7),
      {.pipeline_id = "data.pipeline.bounded",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      make_factory(dataset));

  const auto capability = pipeline.capability();
  EXPECT_EQ(capability.restore_level, jyppx::RestoreLevel::bounded);
  EXPECT_FALSE(capability.degradation_reasons.empty());
}

TEST(DataPipelineCheckpointTest, RejectsTopologyMismatchWithoutMutation) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::RandomSampler>(6, 8),
      {.pipeline_id = "data.pipeline.topology",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(0);
  ASSERT_TRUE(source.next().has_value());
  auto store = save_state(source);

  auto target = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::RandomSampler>(6, 8),
      {.pipeline_id = "data.pipeline.topology",
       .dataset_version = "dataset/v2",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  const auto before = target.state();
  EXPECT_THROW(target.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);
  EXPECT_EQ(target.state().pipeline_id, before.pipeline_id);
  EXPECT_EQ(target.state().prepared, before.prepared);
  EXPECT_EQ(target.state().next_cursor, before.next_cursor);
}

TEST(DataPipelineCheckpointTest,
     RejectsSamplerConfigurationMismatchWithoutMutation) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::RandomSampler>(6, 8),
      {.pipeline_id = "data.pipeline.sampler-mismatch",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(0);
  auto store = save_state(source);

  auto target = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.sampler-mismatch",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  const auto before = target.state();
  EXPECT_THROW(target.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);
  EXPECT_EQ(target.state().sampler_identity, before.sampler_identity);
  EXPECT_EQ(target.state().prepared, before.prepared);
}

TEST(DataPipelineCheckpointTest, RejectsCorruptCursorWithoutMutation) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::RandomSampler>(6, 8),
      {.pipeline_id = "data.pipeline.corrupt",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(2);
  auto store = save_state(source);
  store.append_byte("cursor");

  auto target = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::RandomSampler>(6, 8),
      {.pipeline_id = "data.pipeline.corrupt",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  target.set_epoch(5);
  const auto before = target.state();
  EXPECT_THROW(target.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  EXPECT_EQ(target.state().epoch, before.epoch);
  EXPECT_EQ(target.state().epoch_indices, before.epoch_indices);
  EXPECT_EQ(target.state().next_cursor, before.next_cursor);
}

TEST(DataPipelineCheckpointTest, RejectsCursorInsideUncommittedBatch) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.cursor-boundary",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(0);
  ASSERT_TRUE(source.next().has_value());
  auto store = save_state(source);
  store.overwrite_u64_from_end("cursor", sizeof(std::uint64_t), 1U);

  auto target = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.cursor-boundary",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  EXPECT_THROW(target.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  EXPECT_FALSE(target.state().prepared);
}

TEST(DataPipelineCheckpointTest, RejectsOutOfRangeRestoredSampleId) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.sample-bound",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(0);
  auto store = save_state(source);
  store.overwrite_u64_from_end("topology", 0, 6U);

  auto target = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.sample-bound",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema_id = "tensor.rows/v1",
       .batch_size = 2},
      factory);
  EXPECT_THROW(target.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  EXPECT_FALSE(target.state().prepared);
}

}  // namespace
