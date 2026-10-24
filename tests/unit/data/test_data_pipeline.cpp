// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/dataset.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/data/samplers/weighted_random_sampler.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

class TensorRowsDataset final
    : public jyppx::liuding::data::Dataset<jyppx::liuding::Tensor> {
 public:
  explicit TensorRowsDataset(jyppx::liuding::Tensor rows)
      : rows_(std::move(rows)) {}

  jyppx::liuding::Tensor get(std::size_t index) const override {
    return rows_[static_cast<std::int64_t>(index)];
  }

  std::size_t size() const override {
    return static_cast<std::size_t>(rows_.size(0));
  }

 private:
  jyppx::liuding::Tensor rows_;  ///< Tensor rows indexed by the sampler.
};

class MemoryStateStore final : public jyppx::liuding::StateWriter,
                               public jyppx::liuding::StateReader {
 public:
  explicit MemoryStateStore(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::liuding::validate_state_manifest(manifest_);
  }

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field = std::find_if(
        manifest_.fields.begin(), manifest_.fields.end(),
        [name](const jyppx::liuding::StateFieldDescriptor& candidate) {
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

  const jyppx::liuding::StateManifest& manifest() const noexcept override {
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

  /** @brief Returns a mutable manifest for compatibility-negative tests. */
  jyppx::liuding::StateManifest& mutable_manifest() noexcept {
    return manifest_;
  }

  /**
   * @brief Replaces one same-length ASCII fragment in a stored field.
   * @param name Existing field containing the fragment.
   * @param original Exact byte sequence to locate once.
   * @param replacement Same-length byte sequence to install.
   */
  void replace_ascii(std::string_view name, std::string_view original,
                     std::string_view replacement) {
    if (original.size() != replacement.size()) {
      throw std::runtime_error("DataPipeline replacement length changed");
    }
    auto& blob = blobs_.at(std::string(name));
    const auto begin = std::search(
        blob.begin(), blob.end(), original.begin(), original.end(),
        [](std::byte left, char right) {
          return left ==
                 static_cast<std::byte>(static_cast<unsigned char>(right));
        });
    if (begin == blob.end()) {
      throw std::runtime_error("DataPipeline test fragment is absent");
    }
    std::transform(
        replacement.begin(), replacement.end(), begin, [](char value) {
          return static_cast<std::byte>(static_cast<unsigned char>(value));
        });
  }

  /** @brief Returns all stored fields for writer-version assertions. */
  const std::map<std::string, std::vector<std::byte>>& blobs() const noexcept {
    return blobs_;
  }

  /** @brief Replaces one stored field with intentionally malformed bytes. */
  void replace_blob(std::string_view name, std::vector<std::byte> bytes) {
    blobs_.at(std::string(name)) = std::move(bytes);
  }

 private:
  jyppx::liuding::StateManifest
      manifest_;  ///< Manifest enforced by this store.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Stored participant fields.
};

/** @brief Stateful sampler used to verify failed epoch preparation rollback. */
class FaultingSampler final : public jyppx::liuding::data::samplers::Sampler {
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

/**
 * @brief Stateless sampler used to verify restore-capability degradation.
 *
 * The sampler exposes a stable immutable identity but deliberately retains the
 * base class state contract. This isolates missing mutable-state support from
 * missing configuration identity when DataPipeline reports restore guarantees.
 */
class StatelessSampler final : public jyppx::liuding::data::samplers::Sampler {
 public:
  /** @brief Returns the fixed six-sample epoch used by the test dataset. */
  std::vector<std::size_t> indices() const override {
    return {0U, 1U, 2U, 3U, 4U, 5U};
  }

  /** @brief Returns the fixed epoch cardinality. */
  std::size_t size() const override { return 6U; }

  /** @brief Returns a stable identity while leaving state support disabled. */
  std::string configuration_identity() const override {
    return "test.stateless-sampler/v1:size=6";
  }
};

std::shared_ptr<TensorRowsDataset> make_dataset() {
  return std::make_shared<TensorRowsDataset>(
      torch::arange(0, 12, torch::TensorOptions().dtype(torch::kFloat32))
          .view({6, 2}));
}

jyppx::liuding::data::DataBatchFactory make_factory(
    const std::shared_ptr<TensorRowsDataset>& dataset) {
  return [dataset](std::span<const std::size_t> indices,
                   const jyppx::liuding::data::DataWorkerContext&,
                   std::stop_token) {
    std::vector<jyppx::liuding::Tensor> rows;
    rows.reserve(indices.size());
    for (const auto index : indices) {
      rows.push_back(dataset->get(index));
    }
    return c10::IValue(torch::stack(rows));
  };
}

MemoryStateStore save_state(
    const jyppx::liuding::data::DataPipeline& pipeline) {
  MemoryStateStore store(pipeline.describe_state());
  pipeline.save_state(store);
  return store;
}

/** @brief Strictly decodes one base64 field from a canonical fixture. */
std::vector<std::byte> read_base64_fixture(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::string encoded;
  std::string trailing;
  if (!(input >> encoded) || (input >> trailing) || encoded.empty() ||
      encoded.size() % 4U != 0U) {
    throw std::runtime_error("invalid base64 DataPipeline fixture: " +
                             path.string());
  }

  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const auto decode = [&](char value) -> std::uint32_t {
    const auto position = alphabet.find(value);
    if (position == std::string_view::npos) {
      throw std::runtime_error("invalid base64 DataPipeline byte: " +
                               path.string());
    }
    return static_cast<std::uint32_t>(position);
  };

  std::vector<std::byte> decoded;
  decoded.reserve((encoded.size() / 4U) * 3U);
  for (std::size_t offset = 0; offset < encoded.size(); offset += 4U) {
    const auto final_group = offset + 4U == encoded.size();
    const auto third_padding = encoded[offset + 2U] == '=';
    const auto fourth_padding = encoded[offset + 3U] == '=';
    if (encoded[offset] == '=' || encoded[offset + 1U] == '=' ||
        (!final_group && (third_padding || fourth_padding)) ||
        (third_padding && !fourth_padding)) {
      throw std::runtime_error("invalid base64 DataPipeline padding: " +
                               path.string());
    }
    const auto packed =
        (decode(encoded[offset]) << 18U) |
        (decode(encoded[offset + 1U]) << 12U) |
        (third_padding ? 0U : decode(encoded[offset + 2U]) << 6U) |
        (fourth_padding ? 0U : decode(encoded[offset + 3U]));
    decoded.push_back(
        std::byte{static_cast<unsigned char>((packed >> 16U) & 0xFFU)});
    if (!third_padding) {
      decoded.push_back(
          std::byte{static_cast<unsigned char>((packed >> 8U) & 0xFFU)});
    }
    if (!fourth_padding) {
      decoded.push_back(std::byte{static_cast<unsigned char>(packed & 0xFFU)});
    }
  }
  return decoded;
}

/** @brief Loads state emitted by the historical schema-v3 writer. */
MemoryStateStore load_canonical_schema_v3_fixture() {
  MemoryStateStore fixture({
      .participant_id = "data.pipeline.fixture",
      .schema_version = 3,
      .restore_level = jyppx::liuding::RestoreLevel::exact,
      .fields = {{.name = "topology",
                  .media_type = "application/octet-stream",
                  .max_bytes = 32U * 1024U * 1024U,
                  .required = true},
                 {.name = "cursor",
                  .media_type = "application/octet-stream",
                  .max_bytes = 128U,
                  .required = true}},
      .degradation_reasons = {},
  });
  const auto fixture_dir =
      std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) / "checkpoint" /
      "data_pipeline_schema_v3_libtorch_2.5.1";
  for (const auto& field : fixture.manifest().fields) {
    fixture.write_blob(
        field.name,
        read_base64_fixture(fixture_dir / (field.name + ".bin.b64")));
  }
  return fixture;
}

/** @brief Creates one deterministic worker topology for pipeline tests. */
jyppx::liuding::data::DataPipelineOptions make_worker_options(
    std::string pipeline_id, std::size_t workers = 3,
    std::size_t prefetch_batches = 2, bool persistent_workers = true) {
  return {.pipeline_id = std::move(pipeline_id),
          .dataset_version = "dataset.worker/v1",
          .dataset_size = 6,
          .schema = jyppx::liuding::data::BatchSchema::any("worker.batch", 1),
          .batch_factory_id = "factory.worker/v1",
          .batch_size = 2,
          .drop_last = false,
          .num_workers = workers,
          .prefetch_batches = prefetch_batches,
          .persistent_workers = persistent_workers,
          .deterministic_batch_factory = true,
          .worker_seed = 1977};
}

/** @brief Drains integer-valued worker batches in committed order. */
std::vector<std::pair<std::vector<std::uint64_t>, std::int64_t>>
consume_worker_pipeline(jyppx::liuding::data::DataPipeline& pipeline) {
  std::vector<std::pair<std::vector<std::uint64_t>, std::int64_t>> batches;
  while (const auto batch = pipeline.next()) {
    batches.emplace_back(batch->sample_ids, batch->value.toInt());
  }
  return batches;
}

/** @brief Blocks test factories until cooperative cancellation. */
class StoppableGate final {
 public:
  /** @brief Enters the gate and waits for the supplied stop token. */
  void enter(std::stop_token stop) {
    std::unique_lock<std::mutex> lock(mutex_);
    ++entered_;
    maximum_entered_ = std::max(maximum_entered_, entered_);
    changed_.notify_all();
    std::stop_callback notify_stop(stop, [this] { changed_.notify_all(); });
    changed_.wait(lock, [&] { return stop.stop_requested(); });
    --entered_;
    changed_.notify_all();
  }

  /** @brief Waits until at least @p expected factories entered the gate. */
  bool wait_for_entered(std::size_t expected,
                        std::chrono::milliseconds timeout) {
    // Explicit lock/unlock polling remains observable to the GCC 11 TSan
    // runtime, which does not reliably model timed condition waits.
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entered_ >= expected) {
          return true;
        }
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  /** @brief Returns the largest simultaneous factory count observed. */
  std::size_t maximum_entered() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return maximum_entered_;
  }

 private:
  mutable std::mutex mutex_;         ///< Protects gate concurrency counters.
  std::condition_variable changed_;  ///< Announces entry or cancellation.
  std::size_t entered_ = 0;          ///< Factories currently inside the gate.
  std::size_t maximum_entered_ = 0;  ///< Peak simultaneous factory count.
};

TEST(DataPipelineTest, RequiresExplicitEpochAndDoesNotImplicitlyAdvance) {
  auto dataset = make_dataset();
  auto pipeline = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(6, 42),
      {.pipeline_id = "data.pipeline.test",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      make_factory(dataset));

  EXPECT_EQ(pipeline.options().pipeline_id, "data.pipeline.test");
  EXPECT_EQ(pipeline.options().batch_size, 2U);
  EXPECT_THROW(pipeline.next(), jyppx::liuding::InvalidArgument);
  pipeline.set_epoch(3);
  ASSERT_TRUE(pipeline.next().has_value());
  ASSERT_TRUE(pipeline.next().has_value());
  ASSERT_TRUE(pipeline.next().has_value());
  EXPECT_FALSE(pipeline.next().has_value());
  EXPECT_FALSE(pipeline.next().has_value());

  auto peer = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(6, 42),
      {.pipeline_id = "data.pipeline.test",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      make_factory(dataset));
  peer.set_epoch(3);
  EXPECT_EQ(pipeline.state().epoch_indices, peer.state().epoch_indices);
}

TEST(DataPipelineCheckpointTest, RestoresExactInEpochCursor) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(6, 17),
      {.pipeline_id = "data.pipeline.restore",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(4);
  ASSERT_TRUE(source.next().has_value());
  auto store = save_state(source);

  auto target = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(6, 91),
      {.pipeline_id = "data.pipeline.restore",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  target.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
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

TEST(DataPipelineCheckpointTest,
     MigratesCanonicalSchemaV3AndReplaysWorkersExactly) {
  auto factory = [](std::span<const std::size_t>,
                    const jyppx::liuding::data::DataWorkerContext& context,
                    std::stop_token) {
    return c10::IValue(
        static_cast<std::int64_t>(context.sequence * 10U + context.worker_id));
  };
  const auto options =
      make_worker_options("data.pipeline.fixture", 3U, 3U, true);

  jyppx::liuding::data::DataPipeline uninterrupted(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      options, factory);
  uninterrupted.set_epoch(8U);
  ASSERT_TRUE(uninterrupted.next().has_value());
  const auto expected = consume_worker_pipeline(uninterrupted);

  jyppx::liuding::data::DataPipeline restored(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      options, factory);
  const auto fixture = load_canonical_schema_v3_fixture();
  restored.load_state(fixture, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(restored.state().version,
            jyppx::liuding::data::DataPipelineState::format_version);
  EXPECT_EQ(restored.state().epoch, 8U);
  EXPECT_EQ(restored.state().next_cursor, 2U);
  EXPECT_EQ(restored.state().next_sequence, 1U);
  EXPECT_EQ(consume_worker_pipeline(restored), expected);

  const auto rewritten = save_state(restored);
  EXPECT_EQ(rewritten.manifest().schema_version,
            jyppx::liuding::data::DataPipelineState::format_version);
  const auto& topology = rewritten.blobs().at("topology");
  const std::string topology_text(
      reinterpret_cast<const char*>(topology.data()), topology.size());
  EXPECT_NE(topology_text.find("JYPPX-DATA-PIPELINE-TOPOLOGY/v5"),
            std::string::npos);
  EXPECT_EQ(topology_text.find("JYPPX-DATA-PIPELINE-TOPOLOGY/v3"),
            std::string::npos);
}

TEST(DataPipelineCheckpointTest,
     RejectsUnsupportedOrRelabelledSchemaV3WithoutMutation) {
  auto factory = [](std::span<const std::size_t>,
                    const jyppx::liuding::data::DataWorkerContext& context,
                    std::stop_token) {
    return c10::IValue(static_cast<std::int64_t>(context.sequence));
  };
  const auto options =
      make_worker_options("data.pipeline.fixture", 3U, 3U, true);
  jyppx::liuding::data::DataPipeline target(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      options, factory);
  target.set_epoch(11U);
  ASSERT_TRUE(target.next().has_value());
  const auto before = target.state();

  auto obsolete = load_canonical_schema_v3_fixture();
  obsolete.mutable_manifest().schema_version = 2U;
  EXPECT_THROW(
      target.load_state(obsolete, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.state().epoch, before.epoch);
  EXPECT_EQ(target.state().next_cursor, before.next_cursor);

  auto future = load_canonical_schema_v3_fixture();
  future.mutable_manifest().schema_version =
      jyppx::liuding::data::DataPipelineState::format_version + 1U;
  EXPECT_THROW(
      target.load_state(future, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.state().epoch, before.epoch);
  EXPECT_EQ(target.state().next_cursor, before.next_cursor);

  auto relabelled = load_canonical_schema_v3_fixture();
  relabelled.mutable_manifest().fields.front().media_type =
      "application/vnd.jyppx.relabelled";
  EXPECT_THROW(
      target.load_state(relabelled, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.state().epoch, before.epoch);
  EXPECT_EQ(target.state().next_cursor, before.next_cursor);
}

TEST(DataPipelineCheckpointTest,
     RejectsSchemaV3IdentityOrUnprovenStructureWithoutMutation) {
  auto factory = [](std::span<const std::size_t>,
                    const jyppx::liuding::data::DataWorkerContext& context,
                    std::stop_token) {
    return c10::IValue(static_cast<std::int64_t>(context.sequence));
  };
  auto options = make_worker_options("data.pipeline.fixture", 3U, 3U, true);
  jyppx::liuding::data::DataPipeline target(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      options, factory);
  target.set_epoch(12U);
  const auto before = target.state();

  auto mismatched_id = load_canonical_schema_v3_fixture();
  mismatched_id.replace_ascii("topology", "worker.batch/v1", "worker.other/v1");
  EXPECT_THROW(target.load_state(mismatched_id,
                                 jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.state().epoch, before.epoch);
  EXPECT_EQ(target.state().next_cursor, before.next_cursor);

  options.schema = jyppx::liuding::data::BatchSchema::tensor("worker.batch");
  jyppx::liuding::data::DataPipeline structured(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      options, factory);
  structured.set_epoch(13U);
  const auto structured_before = structured.state();
  const auto fixture = load_canonical_schema_v3_fixture();
  EXPECT_THROW(
      structured.load_state(fixture, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(structured.state().epoch, structured_before.epoch);
  EXPECT_EQ(structured.state().next_cursor, structured_before.next_cursor);
}

TEST(DataPipelineTest, FactoryFailureDoesNotAdvanceCursor) {
  auto dataset = make_dataset();
  bool fail = true;
  auto pipeline = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.failure",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      [&fail, dataset](std::span<const std::size_t> indices,
                       const jyppx::liuding::data::DataWorkerContext&,
                       std::stop_token) {
        if (fail) {
          fail = false;
          throw std::runtime_error("factory failure");
        }
        std::vector<jyppx::liuding::Tensor> rows;
        for (const auto index : indices) {
          rows.push_back(dataset->get(index));
        }
        return c10::IValue(torch::stack(rows));
      });
  pipeline.set_epoch(0);
  EXPECT_THROW(pipeline.next(), std::runtime_error);
  EXPECT_EQ(pipeline.state().next_cursor, 0U);
  const auto checkpoint = save_state(pipeline);
  EXPECT_THROW(pipeline.next(), jyppx::liuding::BackendError);
  pipeline.load_state(checkpoint, jyppx::liuding::RestorePolicy::exact_only);
  ASSERT_TRUE(pipeline.next().has_value());
  EXPECT_EQ(pipeline.state().next_cursor, 2U);
}

TEST(DataPipelineTest, CancellationAndDropLastDoNotOverCommit) {
  auto dataset = make_dataset();
  auto pipeline = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.cancel",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 4,
       .drop_last = true},
      make_factory(dataset));
  pipeline.set_epoch(0);

  std::stop_source cancelled;
  cancelled.request_stop();
  EXPECT_FALSE(pipeline.next(cancelled.get_token()).has_value());
  EXPECT_EQ(pipeline.state().next_cursor, 0U);

  EXPECT_THROW(pipeline.next(), jyppx::liuding::Cancelled);
  pipeline.set_epoch(0);
  const auto batch = pipeline.next();
  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(batch->sample_ids, std::vector<std::uint64_t>({0U, 1U, 2U, 3U}));
  EXPECT_EQ(pipeline.state().next_cursor, 4U);
  EXPECT_FALSE(pipeline.next().has_value());
}

TEST(DataPipelineTest, FailedEpochPreparationRollsBackSamplerAndCursor) {
  auto dataset = make_dataset();
  auto sampler = std::make_shared<FaultingSampler>();
  auto pipeline = jyppx::liuding::data::DataPipeline(
      sampler,
      {.pipeline_id = "data.pipeline.rollback",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      make_factory(dataset));
  pipeline.set_epoch(3);
  ASSERT_TRUE(pipeline.next().has_value());
  const auto before = pipeline.state();

  sampler->set_failure(true);
  EXPECT_THROW(pipeline.set_epoch(8), jyppx::liuding::InvalidArgument);
  EXPECT_EQ(sampler->epoch(), 3U);
  EXPECT_EQ(pipeline.state().epoch, before.epoch);
  EXPECT_EQ(pipeline.state().next_cursor, before.next_cursor);
  EXPECT_EQ(pipeline.state().epoch_indices, before.epoch_indices);
}

TEST(DataPipelineTest, MissingSamplerIdentityDowngradesRestoreGuarantee) {
  auto dataset = make_dataset();
  auto pipeline = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::WeightedRandomSampler>(
          std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 6, true, 7),
      {.pipeline_id = "data.pipeline.bounded",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      make_factory(dataset));

  const auto capability = pipeline.capability();
  EXPECT_EQ(capability.restore_level, jyppx::liuding::RestoreLevel::bounded);
  EXPECT_FALSE(capability.degradation_reasons.empty());
}

TEST(DataPipelineTest, StatelessSamplerDowngradesRestoreGuarantee) {
  auto dataset = make_dataset();
  auto pipeline = jyppx::liuding::data::DataPipeline(
      std::make_shared<StatelessSampler>(),
      {.pipeline_id = "data.pipeline.stateless",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      make_factory(dataset));

  const auto capability = pipeline.capability();
  EXPECT_EQ(capability.restore_level, jyppx::liuding::RestoreLevel::bounded);
  EXPECT_EQ(capability.degradation_reasons,
            std::vector<std::string>{"sampler does not expose exact state"});
}

TEST(DataPipelineTest, PublishesStructuredCrossDomainCapability) {
  auto dataset = make_dataset();
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
          dataset->size()),
      {.pipeline_id = "data.pipeline.capability",
       .dataset_version = "dataset/v1",
       .dataset_size = dataset->size(),
       .schema = jyppx::liuding::data::BatchSchema::tensor(
           "tensor.rows", {.dtype = torch::kFloat32, .rank = 2}, 1, "features"),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2,
       .num_workers = 2,
       .prefetch_batches = 3,
       .persistent_workers = true},
      make_factory(dataset));

  const auto descriptor =
      jyppx::liuding::data::describe_data_capability(pipeline.capability());
  EXPECT_EQ(descriptor.feature_id, "jyppx.data.pipeline.map");
  EXPECT_EQ(descriptor.scope_id, "data.pipeline.capability");
  EXPECT_EQ(descriptor.status, jyppx::liuding::CapabilityStatus::available);
  EXPECT_EQ(descriptor.implementation,
            jyppx::liuding::CapabilityImplementation::native);
  EXPECT_EQ(descriptor.source,
            jyppx::liuding::CapabilitySource::configured_instance);
  EXPECT_EQ(descriptor.backend, "libtorch");
  EXPECT_NE(
      std::find(descriptor.dtypes.begin(), descriptor.dtypes.end(), "Float"),
      descriptor.dtypes.end());
  EXPECT_NE(std::find(descriptor.operations.begin(),
                      descriptor.operations.end(), "worker.thread-pool"),
            descriptor.operations.end());
  EXPECT_NE(std::find(descriptor.operations.begin(),
                      descriptor.operations.end(), "restore.exact"),
            descriptor.operations.end());
  const auto worker_limit =
      std::find_if(descriptor.limits.begin(), descriptor.limits.end(),
                   [](const jyppx::liuding::CapabilityLimit& limit) {
                     return limit.name == "worker_count";
                   });
  ASSERT_NE(worker_limit, descriptor.limits.end());
  EXPECT_EQ(worker_limit->value, 2U);
  EXPECT_EQ(worker_limit->unit, "workers");

  const auto manifest =
      jyppx::liuding::make_current_capability_manifest({descriptor});
  const auto json = jyppx::liuding::capability_manifest_json(manifest);
  EXPECT_NE(json.find("\"feature_id\":\"jyppx.data.pipeline.map\""),
            std::string::npos);
  EXPECT_NE(json.find("\"scope_id\":\"data.pipeline.capability\""),
            std::string::npos);
  EXPECT_NE(json.find("\"source\":\"configured-instance\""), std::string::npos);
  EXPECT_NE(json.find("\"value\":2"), std::string::npos);
}

TEST(DataPipelineTest, PublishesDegradedRestoreCapabilityOperations) {
  auto dataset = make_dataset();
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
          dataset->size()),
      {.pipeline_id = "data.pipeline.degraded-restore",
       .dataset_version = "dataset/v1",
       .dataset_size = dataset->size(),
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      make_factory(dataset));
  auto capability = pipeline.capability();
  capability.restore_level = jyppx::liuding::RestoreLevel::bounded;
  capability.degradation_reasons = {"bounded replay"};

  auto descriptor = jyppx::liuding::data::describe_data_capability(capability);
  EXPECT_NE(std::find(descriptor.operations.begin(),
                      descriptor.operations.end(), "restore.bounded"),
            descriptor.operations.end());

  capability.restore_level = jyppx::liuding::RestoreLevel::best_effort;
  capability.degradation_reasons = {"source position is approximate"};
  descriptor = jyppx::liuding::data::describe_data_capability(capability);
  EXPECT_NE(std::find(descriptor.operations.begin(),
                      descriptor.operations.end(), "restore.best-effort"),
            descriptor.operations.end());
}

TEST(DataPipelineTest, RejectsTamperedCrossDomainCapability) {
  auto dataset = make_dataset();
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
          dataset->size()),
      {.pipeline_id = "data.pipeline.capability-negative",
       .dataset_version = "dataset/v1",
       .dataset_size = dataset->size(),
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      make_factory(dataset));
  auto capability = pipeline.capability();
  capability.pipeline_id.clear();
  EXPECT_THROW(jyppx::liuding::data::describe_data_capability(capability),
               jyppx::liuding::InvalidArgument);

  capability = pipeline.capability();
  capability.batch_schema_fingerprint = "relabelled";
  EXPECT_THROW(jyppx::liuding::data::describe_data_capability(capability),
               jyppx::liuding::InvalidArgument);

  capability = pipeline.capability();
  capability.worker_shard_policy =
      jyppx::liuding::data::DataWorkerShardPolicy::round_robin_batch;
  EXPECT_THROW(jyppx::liuding::data::describe_data_capability(capability),
               jyppx::liuding::InvalidArgument);
}

TEST(DataPipelineTest, RejectsEveryCrossDomainCapabilityInvariant) {
  auto dataset = make_dataset();
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
          dataset->size()),
      {.pipeline_id = "data.pipeline.capability-invariants",
       .dataset_version = "dataset/v1",
       .dataset_size = dataset->size(),
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      make_factory(dataset));
  const auto valid = pipeline.capability();
  const auto expect_invalid = [&valid](const auto& mutate) {
    auto candidate = valid;
    mutate(candidate);
    EXPECT_THROW(jyppx::liuding::data::describe_data_capability(candidate),
                 jyppx::liuding::InvalidArgument);
  };

  expect_invalid([](auto& value) { value.iterable_style = true; });
  expect_invalid([](auto& value) { value.worker_count = 1; });
  expect_invalid([](auto& value) {
    value.worker_model = jyppx::liuding::data::DataWorkerModel::thread_pool;
    value.worker_shard_policy =
        jyppx::liuding::data::DataWorkerShardPolicy::round_robin_batch;
  });
  expect_invalid([](auto& value) {
    value.worker_model = jyppx::liuding::data::DataWorkerModel::producer_thread;
  });
  expect_invalid([](auto& value) {
    value.worker_model =
        static_cast<jyppx::liuding::data::DataWorkerModel>(255);
  });
  expect_invalid([](auto& value) {
    value.worker_shard_policy =
        static_cast<jyppx::liuding::data::DataWorkerShardPolicy>(255);
  });
  expect_invalid([](auto& value) {
    value.degradation_reasons = {"unexpected degradation"};
  });
  expect_invalid([](auto& value) {
    value.restore_level = jyppx::liuding::RestoreLevel::bounded;
  });
  expect_invalid([](auto& value) {
    value.restore_level = static_cast<jyppx::liuding::RestoreLevel>(255);
    value.degradation_reasons = {"invalid restore level"};
  });
  expect_invalid([](auto& value) { value.non_blocking_transfer = true; });
  expect_invalid([](auto& value) { value.pin_memory = true; });
  expect_invalid([](auto& value) { value.memory_stage = true; });
  expect_invalid([](auto& value) { value.memory_stage_capacity = 1; });
  expect_invalid([](auto& value) {
    value.transfer_stream_policy =
        static_cast<jyppx::liuding::data::BatchTransferStreamPolicy>(255);
  });

  auto excessive_metadata = valid;
  excessive_metadata.restore_level = jyppx::liuding::RestoreLevel::bounded;
  excessive_metadata.degradation_reasons.assign(
      jyppx::liuding::kMaxCapabilityValues + 1U, "bounded replay");
  EXPECT_THROW(
      jyppx::liuding::data::describe_data_capability(excessive_metadata),
      jyppx::liuding::ResourceExhausted);
}

TEST(DataPipelineTest, PublishesEveryOptionalCrossDomainCapabilityField) {
  auto dataset = make_dataset();
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
          dataset->size()),
      {.pipeline_id = "data.pipeline.capability-complete",
       .dataset_version = "dataset/v1",
       .dataset_size = dataset->size(),
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      make_factory(dataset));
  auto capability = pipeline.capability();
  capability.batch_schema.value = jyppx::liuding::ValueSchema::dictionary(
      {{"payload",
        jyppx::liuding::ValueSchema::tuple({jyppx::liuding::ValueSchema::tensor(
            {.dtype = torch::kFloat32,
             .device = jyppx::liuding::Device(torch::kCPU)},
            "features")})}});
  capability.batch_schema_fingerprint =
      jyppx::liuding::data::batch_schema_fingerprint(capability.batch_schema,
                                                     capability.schema_limits);
  capability.process_workers = true;
  capability.batch_timeout = true;
  capability.memory_stage = true;
  capability.pin_memory = true;
  capability.non_blocking_transfer = true;
  capability.target_device = jyppx::liuding::Device(torch::kCPU);
  capability.memory_stage_capacity = 1;
  capability.max_replayed_samples = 4;
  capability.max_omitted_samples = 2;

  const auto descriptor =
      jyppx::liuding::data::describe_data_capability(capability);
  const auto has_operation = [&descriptor](std::string_view operation) {
    return std::find(descriptor.operations.begin(), descriptor.operations.end(),
                     operation) != descriptor.operations.end();
  };
  EXPECT_EQ(descriptor.devices, (std::vector<std::string>{"cpu"}));
  EXPECT_EQ(descriptor.dtypes, (std::vector<std::string>{"Float"}));
  EXPECT_TRUE(has_operation("worker.process"));
  EXPECT_TRUE(has_operation("batch.timeout"));
  EXPECT_TRUE(has_operation("memory.stage"));
  EXPECT_TRUE(has_operation("memory.pin"));
  EXPECT_TRUE(has_operation("memory.transfer"));
  EXPECT_TRUE(has_operation("memory.transfer.non-blocking"));
  EXPECT_TRUE(has_operation("memory.stream.caller-current"));
  EXPECT_NE(std::find_if(descriptor.limits.begin(), descriptor.limits.end(),
                         [](const jyppx::liuding::CapabilityLimit& limit) {
                           return limit.name == "max_replayed_samples" &&
                                  limit.value == 4;
                         }),
            descriptor.limits.end());
  EXPECT_NE(std::find_if(descriptor.limits.begin(), descriptor.limits.end(),
                         [](const jyppx::liuding::CapabilityLimit& limit) {
                           return limit.name == "max_omitted_samples" &&
                                  limit.value == 2;
                         }),
            descriptor.limits.end());
}

TEST(DataPipelineTest, DescribesIterableProducerCapability) {
  auto dataset = make_dataset();
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
          dataset->size()),
      {.pipeline_id = "data.pipeline.capability-iterable",
       .dataset_version = "dataset/v1",
       .dataset_size = dataset->size(),
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      make_factory(dataset));
  auto capability = pipeline.capability();
  capability.map_style = false;
  capability.iterable_style = true;
  capability.worker_model =
      jyppx::liuding::data::DataWorkerModel::producer_thread;
  capability.worker_count = 1;
  capability.worker_shard_policy =
      jyppx::liuding::data::DataWorkerShardPolicy::single_shard;

  const auto descriptor =
      jyppx::liuding::data::describe_data_capability(capability);
  EXPECT_EQ(descriptor.feature_id, "jyppx.data.pipeline.iterable");
  EXPECT_EQ(descriptor.scope_id, "data.pipeline.capability-iterable");
  EXPECT_NE(std::find(descriptor.operations.begin(),
                      descriptor.operations.end(), "worker.producer-thread"),
            descriptor.operations.end());
}

TEST(DataPipelineCheckpointTest,
     RejectsStructuralSchemaMismatchWithoutMutation) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(6, 8),
      {.pipeline_id = "data.pipeline.topology",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(0);
  ASSERT_TRUE(source.next().has_value());
  auto store = save_state(source);

  auto target = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(6, 8),
      {.pipeline_id = "data.pipeline.topology",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::tensor(
           "tensor.rows", {.dtype = torch::kFloat32, .rank = 2}, 1, "features"),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  const auto before = target.state();
  EXPECT_THROW(
      target.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.state().pipeline_id, before.pipeline_id);
  EXPECT_EQ(target.state().prepared, before.prepared);
  EXPECT_EQ(target.state().next_cursor, before.next_cursor);
}

TEST(DataPipelineCheckpointTest,
     RejectsSamplerConfigurationMismatchWithoutMutation) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(6, 8),
      {.pipeline_id = "data.pipeline.sampler-mismatch",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(0);
  auto store = save_state(source);

  auto target = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.sampler-mismatch",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  const auto before = target.state();
  EXPECT_THROW(
      target.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.state().sampler_identity, before.sampler_identity);
  EXPECT_EQ(target.state().prepared, before.prepared);
}

TEST(DataPipelineCheckpointTest, RejectsCorruptCursorWithoutMutation) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(6, 8),
      {.pipeline_id = "data.pipeline.corrupt",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(2);
  auto store = save_state(source);
  store.append_byte("cursor");

  auto target = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(6, 8),
      {.pipeline_id = "data.pipeline.corrupt",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  target.set_epoch(5);
  const auto before = target.state();
  EXPECT_THROW(
      target.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  EXPECT_EQ(target.state().epoch, before.epoch);
  EXPECT_EQ(target.state().epoch_indices, before.epoch_indices);
  EXPECT_EQ(target.state().next_cursor, before.next_cursor);
}

TEST(DataPipelineCheckpointTest,
     RejectsMalformedCursorEncodingsWithoutMutation) {
  auto dataset = make_dataset();
  auto make_pipeline = [&dataset] {
    return jyppx::liuding::data::DataPipeline(
        std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(6, 8),
        {.pipeline_id = "data.pipeline.cursor-encoding",
         .dataset_version = "dataset/v1",
         .dataset_size = 6,
         .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
         .batch_factory_id = "factory.tensor-rows/v1",
         .batch_size = 2},
        make_factory(dataset));
  };
  auto source = make_pipeline();
  source.set_epoch(2);
  ASSERT_TRUE(source.next().has_value());
  const auto saved = save_state(source);
  const auto& valid_cursor = saved.blobs().at("cursor");

  auto target = make_pipeline();
  target.set_epoch(5);
  ASSERT_TRUE(target.next().has_value());
  const auto before = save_state(target).blobs();

  std::map<std::string, std::vector<std::byte>> malformed;
  // The v4 cursor ends with a one-byte prepared flag and three uint64 values.
  const auto prepared_offset =
      valid_cursor.size() - 3U * sizeof(std::uint64_t) - 1U;
  malformed["invalid prepared boolean"] = valid_cursor;
  malformed.at("invalid prepared boolean")[prepared_offset] = std::byte{2};
  malformed["missing prepared boolean"] = valid_cursor;
  malformed.at("missing prepared boolean").resize(prepared_offset);
  malformed["truncated integer"] = valid_cursor;
  malformed.at("truncated integer").pop_back();
  malformed["unknown encoding"] = valid_cursor;
  malformed.at("unknown encoding")[sizeof(std::uint64_t)] ^= std::byte{1};
  malformed["truncated string length"] = valid_cursor;
  malformed.at("truncated string length").resize(sizeof(std::uint64_t) - 1U);
  malformed["truncated string data"] = valid_cursor;
  malformed.at("truncated string data").resize(sizeof(std::uint64_t) + 1U);

  for (const auto& [label, bytes] : malformed) {
    SCOPED_TRACE(label);
    auto corrupted = saved;
    corrupted.replace_blob("cursor", bytes);
    EXPECT_THROW(
        target.load_state(corrupted, jyppx::liuding::RestorePolicy::exact_only),
        jyppx::liuding::SerializationError);
    EXPECT_EQ(save_state(target).blobs(), before);
  }

  auto oversized = saved;
  auto oversized_cursor = valid_cursor;
  std::fill_n(oversized_cursor.begin(), sizeof(std::uint64_t), std::byte{0xff});
  oversized.replace_blob("cursor", std::move(oversized_cursor));
  EXPECT_THROW(
      target.load_state(oversized, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(save_state(target).blobs(), before);
}

TEST(DataPipelineCheckpointTest, RejectsCursorInsideUncommittedBatch) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.cursor-boundary",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(0);
  ASSERT_TRUE(source.next().has_value());
  auto store = save_state(source);
  store.overwrite_u64_from_end("cursor", sizeof(std::uint64_t), 1U);

  auto target = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.cursor-boundary",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  EXPECT_THROW(
      target.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  EXPECT_FALSE(target.state().prepared);
}

TEST(DataPipelineCheckpointTest, RejectsOutOfRangeRestoredSampleId) {
  auto dataset = make_dataset();
  auto factory = make_factory(dataset);
  auto source = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.sample-bound",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  source.set_epoch(0);
  auto store = save_state(source);
  store.overwrite_u64_from_end("topology", 17U, 6U);

  auto target = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      {.pipeline_id = "data.pipeline.sample-bound",
       .dataset_version = "dataset/v1",
       .dataset_size = 6,
       .schema = jyppx::liuding::data::BatchSchema::any("tensor.rows", 1),
       .batch_factory_id = "factory.tensor-rows/v1",
       .batch_size = 2},
      factory);
  EXPECT_THROW(
      target.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  EXPECT_FALSE(target.state().prepared);
}

TEST(DataPipelineWorkerTest, PreservesOrderAndStableWorkerContext) {
  std::mutex contexts_mutex;
  std::map<std::uint64_t, jyppx::liuding::data::DataWorkerContext> contexts;
  auto factory = [&contexts_mutex, &contexts](
                     std::span<const std::size_t>,
                     const jyppx::liuding::data::DataWorkerContext& context,
                     std::stop_token) {
    if (context.sequence == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    } else if (context.sequence == 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    {
      std::lock_guard<std::mutex> lock(contexts_mutex);
      contexts.emplace(context.sequence, context);
    }
    return c10::IValue(static_cast<std::int64_t>(context.sequence));
  };
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      make_worker_options("data.pipeline.worker-order"), factory);

  pipeline.set_epoch(4);
  EXPECT_EQ(pipeline.outstanding_batches(), 2U);
  const auto batches = consume_worker_pipeline(pipeline);
  ASSERT_EQ(batches.size(), 3U);
  EXPECT_EQ(batches[0].first, std::vector<std::uint64_t>({0U, 1U}));
  EXPECT_EQ(batches[1].first, std::vector<std::uint64_t>({2U, 3U}));
  EXPECT_EQ(batches[2].first, std::vector<std::uint64_t>({4U, 5U}));
  EXPECT_EQ(batches[0].second, 0);
  EXPECT_EQ(batches[1].second, 1);
  EXPECT_EQ(batches[2].second, 2);
  EXPECT_EQ(pipeline.outstanding_batches(), 0U);

  std::lock_guard<std::mutex> lock(contexts_mutex);
  ASSERT_EQ(contexts.size(), 3U);
  for (std::uint64_t sequence = 0; sequence < 3; ++sequence) {
    const auto& context = contexts.at(sequence);
    EXPECT_EQ(context.epoch, 4U);
    EXPECT_EQ(context.sequence, sequence);
    EXPECT_EQ(context.worker_count, 3U);
    EXPECT_EQ(context.worker_id, sequence % 3U);
    EXPECT_EQ(context.shard_sequence, sequence / 3U);
  }
  EXPECT_NE(contexts.at(0).worker_seed, contexts.at(1).worker_seed);
}

TEST(DataPipelineWorkerTest, AssignsStableRoundRobinBatchShards) {
  std::mutex contexts_mutex;
  std::map<std::uint64_t, jyppx::liuding::data::DataWorkerContext> contexts;
  auto factory = [&contexts_mutex, &contexts](
                     std::span<const std::size_t>,
                     const jyppx::liuding::data::DataWorkerContext& context,
                     std::stop_token) {
    std::lock_guard<std::mutex> lock(contexts_mutex);
    contexts.emplace(context.sequence, context);
    return c10::IValue(static_cast<std::int64_t>(context.shard_sequence));
  };
  auto options = make_worker_options("data.pipeline.worker-shards", 3U, 6U);
  options.dataset_size = 12U;
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(12U),
      options, factory);

  pipeline.set_epoch(5U);
  const auto capability = pipeline.capability();
  EXPECT_EQ(capability.worker_shard_policy,
            jyppx::liuding::data::DataWorkerShardPolicy::round_robin_batch);
  const auto batches = consume_worker_pipeline(pipeline);
  ASSERT_EQ(batches.size(), 6U);

  std::lock_guard<std::mutex> lock(contexts_mutex);
  ASSERT_EQ(contexts.size(), 6U);
  for (std::uint64_t sequence = 0; sequence < 6U; ++sequence) {
    const auto& context = contexts.at(sequence);
    EXPECT_EQ(context.worker_id, sequence % 3U);
    EXPECT_EQ(context.shard_sequence, sequence / 3U);
  }
}

TEST(DataPipelineWorkerTest, BoundsTotalPrefetchAndCloseCancelsFactories) {
  StoppableGate gate;
  auto factory = [&gate](std::span<const std::size_t>,
                         const jyppx::liuding::data::DataWorkerContext& context,
                         std::stop_token stop) {
    gate.enter(stop);
    return c10::IValue(static_cast<std::int64_t>(context.sequence));
  };
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      make_worker_options("data.pipeline.worker-bound", 4, 2), factory);

  pipeline.set_epoch(0);
  ASSERT_TRUE(gate.wait_for_entered(2, std::chrono::seconds(1)));
  EXPECT_EQ(pipeline.outstanding_batches(), 2U);
  EXPECT_EQ(gate.maximum_entered(), 2U);
  pipeline.close();
  EXPECT_TRUE(pipeline.closed());
  EXPECT_EQ(gate.maximum_entered(), 2U);
  EXPECT_THROW(pipeline.next(), jyppx::liuding::NotReady);
  EXPECT_THROW(pipeline.set_epoch(1), jyppx::liuding::NotReady);
  EXPECT_THROW(static_cast<void>(save_state(pipeline)),
               jyppx::liuding::NotReady);
}

TEST(DataPipelineWorkerTest, CheckpointReplaysOnlyUncommittedBatchesExactly) {
  auto factory = [](std::span<const std::size_t>,
                    const jyppx::liuding::data::DataWorkerContext& context,
                    std::stop_token) {
    return c10::IValue(
        static_cast<std::int64_t>(context.sequence * 10U + context.worker_id));
  };
  const auto options =
      make_worker_options("data.pipeline.worker-restore", 3, 3);
  jyppx::liuding::data::DataPipeline source(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      options, factory);
  source.set_epoch(8);
  ASSERT_EQ(source.outstanding_batches(), 3U);
  const auto first = source.next();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->sequence, 0U);
  auto store = save_state(source);
  const auto expected = consume_worker_pipeline(source);

  jyppx::liuding::data::DataPipeline restored(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      options, factory);
  restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(restored.state().next_sequence, 1U);
  EXPECT_EQ(consume_worker_pipeline(restored), expected);
}

TEST(DataPipelineWorkerTest,
     NonDeterministicFactoryDeclaresBoundedReplayWindow) {
  auto options = make_worker_options("data.pipeline.worker-bounded", 2, 4);
  options.deterministic_batch_factory = false;
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      options,
      [](std::span<const std::size_t>,
         const jyppx::liuding::data::DataWorkerContext& context,
         std::stop_token) {
        return c10::IValue(static_cast<std::int64_t>(context.sequence));
      });

  const auto capability = pipeline.capability();
  EXPECT_TRUE(capability.worker_prefetch);
  EXPECT_EQ(capability.worker_shard_policy,
            jyppx::liuding::data::DataWorkerShardPolicy::round_robin_batch);
  EXPECT_EQ(capability.worker_count, 2U);
  EXPECT_EQ(capability.prefetch_batches, 4U);
  EXPECT_TRUE(capability.persistent_workers);
  EXPECT_EQ(capability.max_replayed_batches, 4U);
  EXPECT_EQ(capability.restore_level, jyppx::liuding::RestoreLevel::bounded);
  EXPECT_FALSE(capability.degradation_reasons.empty());
}

TEST(DataPipelineWorkerTest, CancellationStopsEpochWithoutCommittingBatch) {
  auto factory = [](std::span<const std::size_t>,
                    const jyppx::liuding::data::DataWorkerContext& context,
                    std::stop_token) {
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    return c10::IValue(static_cast<std::int64_t>(context.sequence));
  };
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      make_worker_options("data.pipeline.worker-cancel", 1, 1), factory);
  pipeline.set_epoch(0);

  std::stop_source cancelled;
  std::jthread request_cancel([&cancelled] {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    cancelled.request_stop();
  });
  EXPECT_FALSE(pipeline.next(cancelled.get_token()).has_value());
  EXPECT_EQ(pipeline.state().next_cursor, 0U);
  EXPECT_THROW(pipeline.next(), jyppx::liuding::Cancelled);
  pipeline.set_epoch(1);
  const auto resumed = pipeline.next();
  ASSERT_TRUE(resumed.has_value());
  EXPECT_EQ(resumed->epoch, 1U);
}

TEST(DataPipelineWorkerTest, TimeoutCancelsEpochWithoutAdvancingCursor) {
  auto options = make_worker_options("data.pipeline.worker-timeout", 1, 1);
  options.batch_timeout = std::chrono::milliseconds(20);
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      options,
      [](std::span<const std::size_t>,
         const jyppx::liuding::data::DataWorkerContext& context,
         std::stop_token stop) {
        while (!stop.stop_requested()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return c10::IValue(static_cast<std::int64_t>(context.sequence));
      });
  pipeline.set_epoch(0);

  EXPECT_THROW(pipeline.next(), jyppx::liuding::DeadlineExceeded);
  EXPECT_EQ(pipeline.state().next_cursor, 0U);
  EXPECT_THROW(pipeline.next(), jyppx::liuding::DeadlineExceeded);
  pipeline.close();
  EXPECT_TRUE(pipeline.closed());
}

TEST(DataPipelineWorkerTest, WorkerFailureIsStructuredAndStopsTheEpoch) {
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      make_worker_options("data.pipeline.worker-error", 2, 2),
      [](std::span<const std::size_t>,
         const jyppx::liuding::data::DataWorkerContext& context,
         std::stop_token) {
        if (context.sequence == 0) {
          throw std::runtime_error("synthetic worker failure");
        }
        return c10::IValue(static_cast<std::int64_t>(context.sequence));
      });
  pipeline.set_epoch(0);

  EXPECT_THROW(pipeline.next(), jyppx::liuding::BackendError);
  EXPECT_EQ(pipeline.state().next_cursor, 0U);
  EXPECT_THROW(pipeline.next(), jyppx::liuding::BackendError);
}

TEST(DataPipelineWorkerTest, PersistentWorkersAcceptASecondExplicitEpoch) {
  std::atomic<std::uint64_t> calls{0};
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      make_worker_options("data.pipeline.worker-persistent", 2, 2, true),
      [&calls](std::span<const std::size_t>,
               const jyppx::liuding::data::DataWorkerContext& context,
               std::stop_token) {
        calls.fetch_add(1, std::memory_order_relaxed);
        return c10::IValue(static_cast<std::int64_t>(context.epoch));
      });

  pipeline.set_epoch(2);
  const auto first_epoch = consume_worker_pipeline(pipeline);
  pipeline.set_epoch(3);
  const auto second_epoch = consume_worker_pipeline(pipeline);
  ASSERT_EQ(first_epoch.size(), 3U);
  ASSERT_EQ(second_epoch.size(), 3U);
  EXPECT_EQ(first_epoch.front().second, 2);
  EXPECT_EQ(second_epoch.front().second, 3);
  EXPECT_EQ(calls.load(std::memory_order_relaxed), 6U);
}

TEST(DataPipelineWorkerTest, RejectsWorkerTopologyMismatchWithoutMutation) {
  auto factory = [](std::span<const std::size_t>,
                    const jyppx::liuding::data::DataWorkerContext& context,
                    std::stop_token) {
    return c10::IValue(static_cast<std::int64_t>(context.sequence));
  };
  auto source_options =
      make_worker_options("data.pipeline.worker-topology", 2, 2);
  jyppx::liuding::data::DataPipeline source(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      source_options, factory);
  source.set_epoch(1);
  ASSERT_TRUE(source.next().has_value());
  auto store = save_state(source);

  auto target_options = source_options;
  target_options.num_workers = 3;
  jyppx::liuding::data::DataPipeline target(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      target_options, factory);
  target.set_epoch(9);
  const auto before = target.state();
  EXPECT_THROW(
      target.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.state().epoch, before.epoch);
  EXPECT_EQ(target.state().next_cursor, before.next_cursor);
}

TEST(DataPipelineWorkerTest, RejectsInvalidWorkerOptionsBeforeStarting) {
  auto sampler =
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6);
  auto factory = [](std::span<const std::size_t>,
                    const jyppx::liuding::data::DataWorkerContext& context,
                    std::stop_token) {
    return c10::IValue(static_cast<std::int64_t>(context.sequence));
  };

  auto missing_identity = make_worker_options("data.pipeline.invalid-id");
  missing_identity.batch_factory_id.clear();
  EXPECT_THROW(
      jyppx::liuding::data::DataPipeline(sampler, missing_identity, factory),
      jyppx::liuding::InvalidArgument);

  auto oversized_identity = make_worker_options("data.pipeline.invalid-size");
  oversized_identity.pipeline_id = std::string(4097, 'x');
  EXPECT_THROW(
      jyppx::liuding::data::DataPipeline(sampler, oversized_identity, factory),
      jyppx::liuding::ResourceExhausted);

  auto empty_dataset = make_worker_options("data.pipeline.empty-dataset");
  empty_dataset.dataset_size = 0;
  EXPECT_THROW(
      jyppx::liuding::data::DataPipeline(sampler, empty_dataset, factory),
      jyppx::liuding::InvalidArgument);

  auto empty_batch = make_worker_options("data.pipeline.empty-batch");
  empty_batch.batch_size = 0;
  EXPECT_THROW(
      jyppx::liuding::data::DataPipeline(sampler, empty_batch, factory),
      jyppx::liuding::InvalidArgument);

  auto excessive_workers = make_worker_options("data.pipeline.worker-bound");
  excessive_workers.num_workers = 257;
  EXPECT_THROW(
      jyppx::liuding::data::DataPipeline(sampler, excessive_workers, factory),
      jyppx::liuding::ResourceExhausted);

  auto no_workers = make_worker_options("data.pipeline.invalid-persistent");
  no_workers.num_workers = 0;
  EXPECT_THROW(jyppx::liuding::data::DataPipeline(sampler, no_workers, factory),
               jyppx::liuding::InvalidArgument);

  auto no_prefetch = make_worker_options("data.pipeline.invalid-prefetch");
  no_prefetch.prefetch_batches = 0;
  EXPECT_THROW(
      jyppx::liuding::data::DataPipeline(sampler, no_prefetch, factory),
      jyppx::liuding::InvalidArgument);

  auto excessive_prefetch = make_worker_options("data.pipeline.prefetch-bound");
  excessive_prefetch.prefetch_batches = 4097;
  EXPECT_THROW(
      jyppx::liuding::data::DataPipeline(sampler, excessive_prefetch, factory),
      jyppx::liuding::InvalidArgument);

  auto negative_timeout = make_worker_options("data.pipeline.negative-timeout");
  negative_timeout.batch_timeout = std::chrono::milliseconds(-1);
  EXPECT_THROW(
      jyppx::liuding::data::DataPipeline(sampler, negative_timeout, factory),
      jyppx::liuding::InvalidArgument);

  auto excessive_timeout = make_worker_options("data.pipeline.timeout-bound");
  excessive_timeout.batch_timeout = std::chrono::milliseconds(86400001);
  EXPECT_THROW(
      jyppx::liuding::data::DataPipeline(sampler, excessive_timeout, factory),
      jyppx::liuding::InvalidArgument);

  auto inline_timeout = make_worker_options("data.pipeline.invalid-timeout");
  inline_timeout.num_workers = 0;
  inline_timeout.persistent_workers = false;
  inline_timeout.batch_timeout = std::chrono::milliseconds(1);
  EXPECT_THROW(
      jyppx::liuding::data::DataPipeline(sampler, inline_timeout, factory),
      jyppx::liuding::InvalidArgument);
}

}  // namespace

namespace {
MemoryStateStore load_canonical_schema_v4_fixture() {
  auto manifest = load_canonical_schema_v3_fixture().manifest();
  manifest.schema_version = 4;
  MemoryStateStore fixture(manifest);
  const auto dir = std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) /
                   "checkpoint" / "data_pipeline_schema_v4_spec";
  for (const auto& field : manifest.fields) {
    fixture.write_blob(field.name,
                       read_base64_fixture(dir / (field.name + ".bin.b64")));
  }
  return fixture;
}
jyppx::liuding::data::BatchSchema pipeline_integer_schema(std::string id) {
  return {.id = std::move(id),
          .value = {.name = "batch",
                    .semantic_tag = "sequence_or_sum",
                    .kind = jyppx::liuding::ValueKind::integer}};
}
TEST(DataPipelineCheckpointTest,
     MigratesSchemaV4FullContractAndRejectsAddedTransform) {
  auto config = make_worker_options("data.pipeline.fixture", 3, 3, true);
  config.schema = pipeline_integer_schema("worker.batch");
  auto factory = [](std::span<const std::size_t>,
                    const jyppx::liuding::data::DataWorkerContext& c,
                    std::stop_token) {
    return c10::IValue(
        static_cast<std::int64_t>(c.sequence * 10U + c.worker_id));
  };
  const auto fixture = load_canonical_schema_v4_fixture();
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      config, factory);
  pipeline.load_state(fixture, jyppx::liuding::RestorePolicy::exact_only);
  ASSERT_EQ(pipeline.state().next_cursor, 2U);
  const auto next = pipeline.next();
  ASSERT_TRUE(next);
  EXPECT_EQ(next->value.toInt(), 11);
  EXPECT_EQ(save_state(pipeline).manifest().schema_version, 5U);
  config.consumer_transform = jyppx::liuding::data::BatchConsumerTransform{
      .id = "new-stage/v1",
      .input_schema = config.schema,
      .apply = [](const c10::IValue& v,
                  const jyppx::liuding::data::BatchConsumerContext&,
                  std::stop_token) { return v; }};
  jyppx::liuding::data::DataPipeline changed(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      config, factory);
  changed.set_epoch(10);
  EXPECT_THROW(
      changed.load_state(fixture, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(changed.state().epoch, 10U);
}
TEST(DataPipelineCheckpointTest,
     RejectsMalformedOrChangedSchemaV4WithoutMutation) {
  auto config = make_worker_options("data.pipeline.fixture", 3, 3, true);
  config.schema = pipeline_integer_schema("worker.batch");
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(6),
      config,
      [](std::span<const std::size_t>,
         const jyppx::liuding::data::DataWorkerContext&,
         std::stop_token) { return c10::IValue(std::int64_t{1}); });
  pipeline.set_epoch(3);
  for (int corruption = 0; corruption < 3; ++corruption) {
    auto bad = load_canonical_schema_v4_fixture();
    if (corruption == 0)
      bad.replace_ascii("topology", "sequence_or_sum", "sequence_or_bad");
    if (corruption == 1) bad.append_byte("topology");
    if (corruption == 2) bad.mutable_manifest().schema_version = 5;
    EXPECT_THROW(
        pipeline.load_state(bad, jyppx::liuding::RestorePolicy::exact_only),
        jyppx::liuding::Error);
    EXPECT_EQ(pipeline.state().epoch, 3U);
    EXPECT_EQ(pipeline.state().next_cursor, 0U);
  }
}
}  // namespace
