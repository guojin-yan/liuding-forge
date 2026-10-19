// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/data/iterable_data_pipeline.hpp>
#include <jyppx/foundation/error.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using jyppx::data::IterableDataPipeline;
using jyppx::data::IterableDataPipelineOptions;
using jyppx::data::IterableSample;
using jyppx::data::IterableSource;
using jyppx::data::IterableSourceCursor;
using jyppx::data::IterableSourceDescriptor;

/** @brief Storage-neutral in-memory checkpoint used by iterable tests. */
class MemoryStateStore final : public jyppx::StateWriter,
                               public jyppx::StateReader {
 public:
  /** @brief Creates an empty store governed by one participant manifest. */
  explicit MemoryStateStore(jyppx::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::validate_state_manifest(manifest_);
  }

  /** @brief Stores one declared checkpoint field. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const jyppx::StateFieldDescriptor& candidate) {
                       return candidate.name == name;
                     });
    if (field == manifest_.fields.end() || data.size() > field->max_bytes) {
      throw std::runtime_error("undeclared iterable checkpoint field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw std::runtime_error("duplicate iterable checkpoint field");
    }
  }

  /** @brief Returns the stored participant manifest. */
  const jyppx::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  /** @brief Reports whether one named field is present. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Returns one bounded field view to the participant. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto found = blobs_.find(std::string(name));
    if (found == blobs_.end() || found->second.size() > max_bytes) {
      throw std::runtime_error("missing iterable checkpoint field");
    }
    return found->second;
  }

  /** @brief Returns the mutable manifest for compatibility-negative tests. */
  jyppx::StateManifest& mutable_manifest() noexcept { return manifest_; }

  /** @brief Removes one field to exercise required-field validation. */
  void erase_blob(std::string_view name) { blobs_.erase(std::string(name)); }

  /** @brief Appends one byte to exercise trailing-payload validation. */
  void append_byte(std::string_view name) {
    blobs_.at(std::string(name)).push_back(std::byte{0});
  }

  /** @brief Returns all stored fields for writer-version assertions. */
  const std::map<std::string, std::vector<std::byte>>& blobs() const noexcept {
    return blobs_;
  }

 private:
  jyppx::StateManifest manifest_;  ///< Immutable participant descriptor.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Encoded participant fields.
};

/** @brief Shared counters used to assert bounded producer behavior. */
struct CursorStats {
  mutable std::mutex mutex;  ///< Protects the pull counter and close state.
  std::size_t pulls = 0;     ///< Number of values returned by `next()`.
  bool closed = false;       ///< Whether `close()` was invoked.
};

/** @brief Deterministic vector cursor with a serializable item position. */
class VectorCursor final : public IterableSourceCursor {
 public:
  /** @brief Creates a cursor for one immutable epoch value vector. */
  VectorCursor(std::vector<std::int64_t> values, std::uint64_t epoch,
               std::shared_ptr<CursorStats> stats,
               std::optional<std::string> watermark_override)
      : values_(std::move(values)),
        epoch_(epoch),
        stats_(std::move(stats)),
        watermark_override_(std::move(watermark_override)) {}

  /** @brief Returns the next value unless stopped or exhausted. */
  std::optional<IterableSample> next(std::stop_token stop) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || stop.stop_requested() || position_ == values_.size()) {
      return std::nullopt;
    }
    const auto position = position_++;
    if (stats_) {
      std::lock_guard<std::mutex> stats_lock(stats_->mutex);
      ++stats_->pulls;
    }
    return IterableSample{.value = c10::IValue(values_[position]),
                          .sample_id = epoch_ * 1000U + position,
                          .watermark = watermark_override_.value_or(
                              std::to_string(position + 1U))};
  }

  /** @brief Encodes the next position as bounded decimal text. */
  std::string save_state() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::to_string(position_);
  }

  /** @brief Validates and applies a previously saved position. */
  void load_state(std::string_view state) override {
    std::size_t parsed = 0;
    try {
      parsed = static_cast<std::size_t>(std::stoull(std::string(state)));
    } catch (const std::exception&) {
      throw jyppx::SerializationError("invalid vector cursor position");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (parsed > values_.size()) {
      throw jyppx::CompatibilityError("vector cursor position is out of range");
    }
    position_ = parsed;
  }

  /** @brief Makes future pulls return and records closure for tests. */
  void close() noexcept override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    if (stats_) {
      std::lock_guard<std::mutex> stats_lock(stats_->mutex);
      stats_->closed = true;
    }
  }

 private:
  std::vector<std::int64_t> values_;    ///< Immutable epoch values.
  std::uint64_t epoch_ = 0;             ///< Logical epoch used in sample IDs.
  std::shared_ptr<CursorStats> stats_;  ///< Optional observation counters.
  std::optional<std::string>
      watermark_override_;    ///< Optional invalid watermark used by tests.
  mutable std::mutex mutex_;  ///< Protects position and close state.
  std::size_t position_ = 0;  ///< Next unread value position.
  bool closed_ = false;       ///< Whether this cursor has been closed.
};

/** @brief Immutable source factory returning isolated vector cursors. */
class VectorSource final : public IterableSource {
 public:
  /** @brief Creates a source with values, guarantees, and test counters. */
  VectorSource(std::vector<std::int64_t> values,
               IterableSourceDescriptor descriptor,
               std::shared_ptr<CursorStats> stats = nullptr,
               std::optional<std::string> watermark_override = std::nullopt)
      : values_(std::move(values)),
        descriptor_(std::move(descriptor)),
        stats_(std::move(stats)),
        watermark_override_(std::move(watermark_override)) {}

  /** @brief Returns the frozen source descriptor. */
  IterableSourceDescriptor descriptor() const override { return descriptor_; }

  /** @brief Opens a fresh cursor whose state is independent of other epochs. */
  std::unique_ptr<IterableSourceCursor> open_epoch(
      std::uint64_t epoch) const override {
    if (fail_epoch_ && *fail_epoch_ == epoch) {
      throw std::runtime_error("synthetic epoch open failure");
    }
    return std::make_unique<VectorCursor>(values_, epoch, stats_,
                                          watermark_override_);
  }

  /** @brief Causes one future epoch open to fail for rollback testing. */
  void fail_epoch(std::uint64_t epoch) { fail_epoch_ = epoch; }

 private:
  std::vector<std::int64_t> values_;     ///< Immutable source values.
  IterableSourceDescriptor descriptor_;  ///< Frozen source guarantees.
  std::shared_ptr<CursorStats> stats_;   ///< Optional pull/close counters.
  std::optional<std::string>
      watermark_override_;  ///< Optional watermark returned for every sample.
  std::optional<std::uint64_t> fail_epoch_;  ///< Optional failing epoch.
};

/** @brief Cursor that blocks until the pipeline requests cancellation. */
class BlockingCursor final : public IterableSourceCursor {
 public:
  /** @brief Waits for stop or close and then reports end-of-source. */
  std::optional<IterableSample> next(std::stop_token stop) override {
    std::unique_lock<std::mutex> lock(mutex_);
    std::stop_callback notify_stop(stop, [this] { changed_.notify_all(); });
    changed_.wait(lock, [&] { return closed_ || stop.stop_requested(); });
    return std::nullopt;
  }

  /** @brief Returns an empty position because no item can be consumed. */
  std::string save_state() const override { return {}; }

  /** @brief Accepts only the empty initial position. */
  void load_state(std::string_view state) override {
    if (!state.empty()) {
      throw jyppx::SerializationError("blocking cursor state is not empty");
    }
  }

  /** @brief Unblocks the waiting source operation. */
  void close() noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    changed_.notify_all();
  }

 private:
  mutable std::mutex mutex_;         ///< Protects blocking state.
  std::condition_variable changed_;  ///< Announces stop or close.
  bool closed_ = false;              ///< Whether close was requested.
};

/** @brief Source factory that supplies one blocking cursor per epoch. */
class BlockingSource final : public IterableSource {
 public:
  /** @brief Creates a checkpointable exact blocking source. */
  IterableSourceDescriptor descriptor() const override {
    return {.source_id = "test.blocking/v1",
            .finite_epoch = false,
            .deterministic_epoch = true,
            .checkpointable = true,
            .restore_level = jyppx::RestoreLevel::exact,
            .max_replayed_samples = 0,
            .max_omitted_samples = 0};
  }

  /** @brief Opens a cursor that waits for cancellation or close. */
  std::unique_ptr<IterableSourceCursor> open_epoch(
      std::uint64_t) const override {
    return std::make_unique<BlockingCursor>();
  }
};

/** @brief Cursor that exposes a source-side failure to the producer. */
class ThrowingCursor final : public IterableSourceCursor {
 public:
  /** @brief Throws a non-framework exception for translation testing. */
  std::optional<IterableSample> next(std::stop_token) override {
    throw std::runtime_error("synthetic source failure");
  }

  /** @brief Encodes the unconsumed initial position. */
  std::string save_state() const override { return "0"; }

  /** @brief Accepts only the initial position. */
  void load_state(std::string_view state) override {
    if (state != "0") {
      throw jyppx::SerializationError("invalid throwing cursor state");
    }
  }

  /** @brief Needs no wakeup because `next()` never blocks. */
  void close() noexcept override {}
};

/** @brief Exact source factory whose cursor always fails on pull. */
class ThrowingSource final : public IterableSource {
 public:
  /** @brief Returns a stable exact descriptor for the failing source. */
  IterableSourceDescriptor descriptor() const override {
    return {.source_id = "test.throwing/v1",
            .finite_epoch = true,
            .deterministic_epoch = true,
            .checkpointable = true,
            .restore_level = jyppx::RestoreLevel::exact,
            .max_replayed_samples = 0,
            .max_omitted_samples = 0};
  }

  /** @brief Opens an independent failing cursor. */
  std::unique_ptr<IterableSourceCursor> open_epoch(
      std::uint64_t) const override {
    return std::make_unique<ThrowingCursor>();
  }
};

IterableSourceDescriptor exact_descriptor() {
  return {.source_id = "test.vector/exact/v1",
          .finite_epoch = true,
          .deterministic_epoch = true,
          .checkpointable = true,
          .restore_level = jyppx::RestoreLevel::exact,
          .max_replayed_samples = 0,
          .max_omitted_samples = 0};
}

IterableDataPipelineOptions options(std::string id,
                                    std::size_t batch_size = 2) {
  return {.pipeline_id = std::move(id),
          .dataset_version = "dataset.iterable/v1",
          .schema = {.id = "batch.iterable",
                     .version = 1,
                     .value =
                         jyppx::ValueSchema{.semantic_tag = "aggregate",
                                            .kind = jyppx::ValueKind::integer}},
          .batch_factory_id = "factory.sum/v1",
          .batch_size = batch_size,
          .prefetch_batches = 2};
}

std::shared_ptr<VectorSource> make_source(
    std::vector<std::int64_t> values,
    IterableSourceDescriptor descriptor = exact_descriptor(),
    std::shared_ptr<CursorStats> stats = nullptr,
    std::optional<std::string> watermark_override = std::nullopt) {
  return std::make_shared<VectorSource>(std::move(values),
                                        std::move(descriptor), std::move(stats),
                                        std::move(watermark_override));
}

auto sum_factory() {
  return [](std::span<const IterableSample> samples, std::uint64_t,
            std::uint64_t, std::stop_token) {
    std::int64_t sum = 0;
    for (const auto& sample : samples) {
      sum += sample.value.toInt();
    }
    return c10::IValue(sum);
  };
}

MemoryStateStore save_state(const IterableDataPipeline& pipeline) {
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
    throw std::runtime_error("invalid iterable pipeline fixture: " +
                             path.string());
  }

  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const auto decode = [&](char value) -> std::uint32_t {
    const auto position = alphabet.find(value);
    if (position == std::string_view::npos) {
      throw std::runtime_error("invalid iterable fixture byte: " +
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
      throw std::runtime_error("invalid iterable fixture padding: " +
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

/** @brief Returns the historical fixture descriptor for one restore level. */
IterableSourceDescriptor fixture_descriptor(std::string_view mode) {
  if (mode == "exact") {
    return {.source_id = "fixture.vector/exact/v1",
            .finite_epoch = true,
            .deterministic_epoch = true,
            .checkpointable = true,
            .restore_level = jyppx::RestoreLevel::exact,
            .max_replayed_samples = 0,
            .max_omitted_samples = 0};
  }
  if (mode == "bounded") {
    return {.source_id = "fixture.vector/bounded/v1",
            .finite_epoch = true,
            .deterministic_epoch = true,
            .checkpointable = true,
            .restore_level = jyppx::RestoreLevel::bounded,
            .max_replayed_samples = 5,
            .max_omitted_samples = 1,
            .degradation_reasons = {"fixture bounded source"}};
  }
  if (mode == "best-effort") {
    return {.source_id = "fixture.vector/best-effort/v1",
            .finite_epoch = true,
            .deterministic_epoch = false,
            .checkpointable = false,
            .restore_level = jyppx::RestoreLevel::best_effort,
            .degradation_reasons = {"fixture best-effort source"}};
  }
  throw std::runtime_error("unknown iterable fixture mode");
}

/** @brief Creates options matching one historical schema-v2 fixture. */
IterableDataPipelineOptions fixture_options(std::string_view mode) {
  return {.pipeline_id = "data.iterable.fixture." + std::string(mode),
          .dataset_version = "dataset.iterable/v1",
          .schema = jyppx::data::BatchSchema::any("batch.iterable", 1),
          .batch_factory_id = "factory.sum/v1",
          .batch_size = 2,
          .prefetch_batches = 3};
}

/** @brief Loads canonical state emitted by the historical schema-v2 writer. */
MemoryStateStore load_canonical_schema_v2_fixture(std::string_view mode) {
  const auto descriptor = fixture_descriptor(mode);
  MemoryStateStore fixture({
      .participant_id = "data.iterable.fixture." + std::string(mode),
      .schema_version = 2,
      .restore_level = descriptor.restore_level,
      .fields = {{.name = "topology",
                  .media_type = "application/octet-stream",
                  .max_bytes = 64U * 1024U,
                  .required = true},
                 {.name = "cursor",
                  .media_type = "application/octet-stream",
                  .max_bytes = 8U * 1024U * 1024U,
                  .required = true}},
      .degradation_reasons = descriptor.degradation_reasons,
  });
  const auto fixture_mode =
      mode == "best-effort" ? std::string("best_effort") : std::string(mode);
  const auto fixture_dir =
      std::filesystem::path(JYPPX_TEST_FIXTURE_ROOT) / "checkpoint" /
      ("iterable_pipeline_schema_v2_" + fixture_mode + "_libtorch_2.5.1");
  for (const auto& field : fixture.manifest().fields) {
    fixture.write_blob(
        field.name,
        read_base64_fixture(fixture_dir / (field.name + ".bin.b64")));
  }
  return fixture;
}

/** @brief Creates the immutable vector source used by schema-v2 fixtures. */
std::shared_ptr<VectorSource> make_fixture_source(std::string_view mode) {
  return make_source({10, 20, 30, 40, 50, 60}, fixture_descriptor(mode));
}

bool wait_for_pulls(const std::shared_ptr<CursorStats>& stats,
                    std::size_t expected, std::chrono::milliseconds timeout) {
  // Explicit lock/unlock polling remains observable to the GCC 11 TSan
  // runtime, which does not reliably model timed condition waits.
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    {
      std::lock_guard<std::mutex> lock(stats->mutex);
      if (stats->pulls >= expected) {
        return true;
      }
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

TEST(IterableDataPipelineTest, ValidatesRestoreDescriptorMatrix) {
  EXPECT_NO_THROW(
      jyppx::data::validate_iterable_source_descriptor(exact_descriptor()));

  auto bounded = exact_descriptor();
  bounded.restore_level = jyppx::RestoreLevel::bounded;
  bounded.max_replayed_samples = 3;
  bounded.max_omitted_samples = 2;
  bounded.degradation_reasons = {"upstream partition movement"};
  EXPECT_NO_THROW(jyppx::data::validate_iterable_source_descriptor(bounded));

  auto best_effort = exact_descriptor();
  best_effort.checkpointable = false;
  best_effort.deterministic_epoch = false;
  best_effort.restore_level = jyppx::RestoreLevel::best_effort;
  best_effort.max_replayed_samples.reset();
  best_effort.max_omitted_samples.reset();
  best_effort.degradation_reasons = {"source has no stable cursor"};
  EXPECT_NO_THROW(
      jyppx::data::validate_iterable_source_descriptor(best_effort));

  auto invalid = exact_descriptor();
  invalid.degradation_reasons = {"unexpected degradation"};
  EXPECT_THROW(jyppx::data::validate_iterable_source_descriptor(invalid),
               jyppx::InvalidArgument);
}

TEST(IterableDataPipelineTest, CommitsBatchesAndPreservesWatermarks) {
  auto source = make_source({1, 2, 3, 4, 5});
  IterableDataPipeline pipeline(source, options("data.iterable.exact"),
                                sum_factory());
  pipeline.set_epoch(3);

  const auto first = pipeline.next();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->value.toInt(), 3);
  EXPECT_EQ(first->sample_ids, (std::vector<std::uint64_t>{3000, 3001}));
  EXPECT_EQ(first->source_watermark, "2");
  EXPECT_EQ(pipeline.state().committed_samples, 2U);
  EXPECT_EQ(pipeline.state().source_state, "2");

  const auto second = pipeline.next();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->value.toInt(), 7);
  EXPECT_EQ(second->source_watermark, "4");
  const auto tail = pipeline.next();
  ASSERT_TRUE(tail.has_value());
  EXPECT_EQ(tail->value.toInt(), 5);
  EXPECT_EQ(tail->source_watermark, "5");
  EXPECT_FALSE(pipeline.next().has_value());
  EXPECT_TRUE(pipeline.state().exhausted);
  EXPECT_EQ(pipeline.state().committed_samples, 5U);
}

TEST(IterableDataPipelineTest, HandlesEmptyAndSingleItemEpochs) {
  IterableDataPipeline empty(make_source({}), options("data.iterable.empty"),
                             sum_factory());
  empty.set_epoch(0);
  EXPECT_FALSE(empty.next().has_value());
  EXPECT_TRUE(empty.state().exhausted);
  EXPECT_EQ(empty.state().committed_samples, 0U);

  IterableDataPipeline single(
      make_source({42}), options("data.iterable.single", 4), sum_factory());
  single.set_epoch(5);
  const auto batch = single.next();
  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(batch->value.toInt(), 42);
  EXPECT_EQ(batch->sample_ids, (std::vector<std::uint64_t>{5000}));
  EXPECT_FALSE(single.next().has_value());
}

TEST(IterableDataPipelineTest, DropLastConsumesAndCheckpointsFiniteTail) {
  auto source = make_source({1, 2, 3, 4, 5});
  auto config = options("data.iterable.drop-last");
  config.drop_last = true;
  IterableDataPipeline pipeline(source, config, sum_factory());
  pipeline.set_epoch(0);

  ASSERT_TRUE(pipeline.next().has_value());
  ASSERT_TRUE(pipeline.next().has_value());
  EXPECT_FALSE(pipeline.next().has_value());
  EXPECT_EQ(pipeline.state().committed_samples, 4U);
  EXPECT_EQ(pipeline.state().source_watermark, "5");
  EXPECT_EQ(pipeline.state().source_state, "5");
}

TEST(IterableDataPipelineTest, PrefetchStopsAtDeclaredCapacityUntilCommit) {
  auto stats = std::make_shared<CursorStats>();
  auto source = make_source({1, 2, 3, 4, 5}, exact_descriptor(), stats);
  auto config = options("data.iterable.backpressure", 1);
  config.prefetch_batches = 2;
  IterableDataPipeline pipeline(source, config, sum_factory());
  pipeline.set_epoch(0);

  ASSERT_TRUE(wait_for_pulls(stats, 2, std::chrono::seconds(1)));
  EXPECT_EQ(pipeline.outstanding_batches(), 2U);
  EXPECT_FALSE(wait_for_pulls(stats, 3, std::chrono::milliseconds(80)));
  ASSERT_TRUE(pipeline.next().has_value());
  EXPECT_TRUE(wait_for_pulls(stats, 3, std::chrono::seconds(1)));
}

TEST(IterableDataPipelineTest,
     MemoryStageHasIndependentCapacityAndPublishesMetadata) {
  auto stats = std::make_shared<CursorStats>();
  auto source = make_source({1, 2, 3}, exact_descriptor(), stats);
  auto config = options("data.iterable.memory-capacity", 1);
  config.prefetch_batches = 3;
  config.memory.target_device = jyppx::Device(torch::kCPU);
  config.memory.stage_capacity = 1;
  config.schema =
      jyppx::data::BatchSchema::tensor("batch.iterable",
                                       {.dtype = torch::kFloat32,
                                        .device = jyppx::Device(torch::kCPU),
                                        .rank = 1},
                                       1, "features");
  IterableDataPipeline pipeline(
      source, config,
      [](std::span<const IterableSample> samples, std::uint64_t, std::uint64_t,
         std::stop_token) {
        return c10::IValue(
            torch::tensor({samples.front().value.toInt()}, torch::kFloat32));
      });
  pipeline.set_epoch(0);

  ASSERT_TRUE(wait_for_pulls(stats, 1, std::chrono::seconds(1)));
  EXPECT_FALSE(wait_for_pulls(stats, 2, std::chrono::milliseconds(80)));
  const auto capability = pipeline.capability();
  EXPECT_TRUE(capability.memory_stage);
  EXPECT_TRUE(capability.iterable_style);
  EXPECT_TRUE(capability.finite_epoch);
  EXPECT_FALSE(capability.process_workers);
  EXPECT_EQ(capability.worker_model,
            jyppx::data::DataWorkerModel::producer_thread);
  EXPECT_EQ(capability.worker_shard_policy,
            jyppx::data::DataWorkerShardPolicy::single_shard);
  EXPECT_EQ(capability.transfer_stream_policy,
            jyppx::data::BatchTransferStreamPolicy::caller_current);
  EXPECT_TRUE(capability.schema_validation);
  EXPECT_EQ(capability.batch_schema.id, "batch.iterable");
  EXPECT_FALSE(capability.batch_schema_fingerprint.empty());
  EXPECT_EQ(capability.prefetch_batches, 3U);
  EXPECT_EQ(capability.memory_stage_capacity, 1U);
  EXPECT_EQ(capability.max_replayed_batches, 1U);

  const auto batch = pipeline.next();
  ASSERT_TRUE(batch.has_value());
  EXPECT_TRUE(batch->memory.staged);
  EXPECT_EQ(batch->memory.tensor_count, 1U);
  EXPECT_TRUE(batch->value.toTensor().device().is_cpu());
  EXPECT_TRUE(wait_for_pulls(stats, 2, std::chrono::seconds(1)));
}

TEST(IterableDataPipelineTest, MemoryStageFailureDoesNotCommitSourceCursor) {
  auto config = options("data.iterable.memory-failure", 1);
  config.memory.target_device = jyppx::Device(torch::kCPU);
  config.memory.max_bytes = 1;
  config.schema = jyppx::data::BatchSchema::tensor(
      "batch.iterable", {.dtype = torch::kFloat32, .rank = 1});
  IterableDataPipeline pipeline(
      make_source({1, 2}), config,
      [](std::span<const IterableSample>, std::uint64_t, std::uint64_t,
         std::stop_token) {
        return c10::IValue(torch::ones({2}, torch::kFloat32));
      });
  pipeline.set_epoch(4);
  const auto before = pipeline.state();

  EXPECT_THROW(pipeline.next(), jyppx::ResourceExhausted);
  EXPECT_EQ(pipeline.state().next_sequence, before.next_sequence);
  EXPECT_EQ(pipeline.state().committed_samples, before.committed_samples);
  EXPECT_EQ(pipeline.state().source_state, before.source_state);
}

TEST(IterableDataPipelineTest,
     SchemaFailureDoesNotCommitSourceCursorOrSequence) {
  auto config = options("data.iterable.schema-failure", 1);
  config.schema = {
      .id = "batch.iterable",
      .version = 1,
      .value = jyppx::ValueSchema{.kind = jyppx::ValueKind::integer}};
  IterableDataPipeline pipeline(
      make_source({1, 2}), config,
      [](std::span<const IterableSample>, std::uint64_t, std::uint64_t,
         std::stop_token) {
        return c10::IValue(torch::ones({1}, torch::kFloat32));
      });
  pipeline.set_epoch(4);
  const auto before = pipeline.state();

  EXPECT_THROW(pipeline.next(), jyppx::InvalidArgument);
  EXPECT_EQ(pipeline.state().next_sequence, before.next_sequence);
  EXPECT_EQ(pipeline.state().committed_samples, before.committed_samples);
  EXPECT_EQ(pipeline.state().source_state, before.source_state);
}

TEST(IterableDataPipelineTest, RestoresExactStateAtLastCommittedBatch) {
  auto stats = std::make_shared<CursorStats>();
  auto source =
      make_source({10, 20, 30, 40, 50, 60}, exact_descriptor(), stats);
  auto config = options("data.iterable.restore");
  config.prefetch_batches = 3;
  IterableDataPipeline original(source, config, sum_factory());
  original.set_epoch(7);
  ASSERT_TRUE(wait_for_pulls(stats, 6, std::chrono::seconds(1)));
  EXPECT_EQ(original.outstanding_batches(), 3U);
  ASSERT_TRUE(original.next().has_value());
  const auto checkpoint = save_state(original);
  EXPECT_EQ(original.state().source_state, "2");

  IterableDataPipeline restored(source, config, sum_factory());
  EXPECT_NO_THROW(
      restored.load_state(checkpoint, jyppx::RestorePolicy::exact_only));
  const auto next = restored.next();
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(next->sample_ids, (std::vector<std::uint64_t>{7002, 7003}));
  EXPECT_EQ(next->value.toInt(), 70);
}

TEST(IterableDataPipelineCheckpointTest,
     MigratesCanonicalSchemaV2AcrossSourceGuarantees) {
  IterableDataPipeline exact(make_fixture_source("exact"),
                             fixture_options("exact"), sum_factory());
  exact.load_state(load_canonical_schema_v2_fixture("exact"),
                   jyppx::RestorePolicy::exact_only);
  EXPECT_EQ(exact.state().version,
            jyppx::data::IterableDataPipelineState::format_version);
  EXPECT_EQ(exact.state().epoch, 7U);
  EXPECT_EQ(exact.state().committed_samples, 2U);
  const auto exact_next = exact.next();
  ASSERT_TRUE(exact_next.has_value());
  EXPECT_EQ(exact_next->sample_ids, (std::vector<std::uint64_t>{7002U, 7003U}));
  EXPECT_EQ(exact_next->value.toInt(), 70);

  const auto rewritten = save_state(exact);
  EXPECT_EQ(rewritten.manifest().schema_version,
            jyppx::data::IterableDataPipelineState::format_version);
  const auto& topology = rewritten.blobs().at("topology");
  const std::string topology_text(
      reinterpret_cast<const char*>(topology.data()), topology.size());
  EXPECT_NE(topology_text.find("JYPPX-ITERABLE-PIPELINE-TOPOLOGY/v3"),
            std::string::npos);
  EXPECT_EQ(topology_text.find("JYPPX-ITERABLE-PIPELINE-TOPOLOGY/v2"),
            std::string::npos);

  IterableDataPipeline bounded(make_fixture_source("bounded"),
                               fixture_options("bounded"), sum_factory());
  const auto bounded_fixture = load_canonical_schema_v2_fixture("bounded");
  EXPECT_THROW(
      bounded.load_state(bounded_fixture, jyppx::RestorePolicy::exact_only),
      jyppx::CompatibilityError);
  bounded.load_state(bounded_fixture, jyppx::RestorePolicy::allow_bounded);
  const auto bounded_next = bounded.next();
  ASSERT_TRUE(bounded_next.has_value());
  EXPECT_EQ(bounded_next->sample_ids,
            (std::vector<std::uint64_t>{7002U, 7003U}));

  IterableDataPipeline best_effort(make_fixture_source("best-effort"),
                                   fixture_options("best-effort"),
                                   sum_factory());
  const auto best_fixture = load_canonical_schema_v2_fixture("best-effort");
  EXPECT_THROW(
      best_effort.load_state(best_fixture, jyppx::RestorePolicy::allow_bounded),
      jyppx::CompatibilityError);
  best_effort.load_state(best_fixture, jyppx::RestorePolicy::allow_best_effort);
  const auto replayed = best_effort.next();
  ASSERT_TRUE(replayed.has_value());
  EXPECT_EQ(replayed->sample_ids, (std::vector<std::uint64_t>{7000U, 7001U}));
}

TEST(IterableDataPipelineCheckpointTest,
     RejectsUnsupportedOrMalformedSchemaV2WithoutMutation) {
  IterableDataPipeline target(make_fixture_source("exact"),
                              fixture_options("exact"), sum_factory());
  target.set_epoch(11U);
  ASSERT_TRUE(target.next().has_value());
  const auto before = target.state();

  auto obsolete = load_canonical_schema_v2_fixture("exact");
  obsolete.mutable_manifest().schema_version = 1U;
  EXPECT_THROW(target.load_state(obsolete, jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);

  auto future = load_canonical_schema_v2_fixture("exact");
  future.mutable_manifest().schema_version =
      jyppx::data::IterableDataPipelineState::format_version + 1U;
  EXPECT_THROW(target.load_state(future, jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);

  auto relabelled = load_canonical_schema_v2_fixture("exact");
  relabelled.mutable_manifest().fields.front().max_bytes = 2U * 1024U * 1024U;
  EXPECT_THROW(target.load_state(relabelled, jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);

  auto missing = load_canonical_schema_v2_fixture("exact");
  missing.erase_blob("cursor");
  EXPECT_THROW(target.load_state(missing, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);

  auto trailing = load_canonical_schema_v2_fixture("exact");
  trailing.append_byte("cursor");
  EXPECT_THROW(target.load_state(trailing, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);

  EXPECT_EQ(target.state().epoch, before.epoch);
  EXPECT_EQ(target.state().next_sequence, before.next_sequence);
  EXPECT_EQ(target.state().committed_samples, before.committed_samples);
  EXPECT_EQ(target.state().source_state, before.source_state);
  EXPECT_TRUE(target.next().has_value());
}

TEST(IterableDataPipelineCheckpointTest,
     RejectsSchemaV2IdentityStructureOrSourceMismatchWithoutMutation) {
  auto mismatched_options = fixture_options("exact");
  mismatched_options.schema = jyppx::data::BatchSchema::any("batch.other", 1);
  IterableDataPipeline mismatched(make_fixture_source("exact"),
                                  mismatched_options, sum_factory());
  mismatched.set_epoch(12U);
  const auto mismatched_before = mismatched.state();
  EXPECT_THROW(mismatched.load_state(load_canonical_schema_v2_fixture("exact"),
                                     jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);
  EXPECT_EQ(mismatched.state().epoch, mismatched_before.epoch);
  EXPECT_EQ(mismatched.state().source_state, mismatched_before.source_state);

  auto structured_options = fixture_options("exact");
  structured_options.schema =
      jyppx::data::BatchSchema::tensor("batch.iterable");
  IterableDataPipeline structured(make_fixture_source("exact"),
                                  structured_options, sum_factory());
  structured.set_epoch(13U);
  const auto structured_before = structured.state();
  EXPECT_THROW(structured.load_state(load_canonical_schema_v2_fixture("exact"),
                                     jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);
  EXPECT_EQ(structured.state().epoch, structured_before.epoch);
  EXPECT_EQ(structured.state().source_state, structured_before.source_state);

  auto changed_descriptor = fixture_descriptor("exact");
  changed_descriptor.source_id = "fixture.vector/other/v1";
  IterableDataPipeline changed_source(
      make_source({10, 20, 30, 40, 50, 60}, changed_descriptor),
      fixture_options("exact"), sum_factory());
  changed_source.set_epoch(14U);
  const auto changed_before = changed_source.state();
  EXPECT_THROW(
      changed_source.load_state(load_canonical_schema_v2_fixture("exact"),
                                jyppx::RestorePolicy::exact_only),
      jyppx::CompatibilityError);
  EXPECT_EQ(changed_source.state().epoch, changed_before.epoch);
  EXPECT_EQ(changed_source.state().source_state, changed_before.source_state);
}

TEST(IterableDataPipelineTest,
     RejectsStructuralSchemaMismatchWithoutMutatingPreparedState) {
  auto source = make_source({1, 2, 3, 4});
  auto original_config = options("data.iterable.schema-restore");
  IterableDataPipeline original(source, original_config, sum_factory());
  original.set_epoch(2);
  ASSERT_TRUE(original.next().has_value());
  const auto checkpoint = save_state(original);

  auto changed_config = original_config;
  changed_config.schema = {
      .id = "batch.iterable",
      .version = 1,
      .value = jyppx::ValueSchema{.semantic_tag = "class-index",
                                  .kind = jyppx::ValueKind::integer}};
  IterableDataPipeline changed(source, changed_config, sum_factory());
  changed.set_epoch(9);
  const auto before = changed.state();

  EXPECT_THROW(changed.load_state(checkpoint, jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);
  EXPECT_EQ(changed.state().epoch, before.epoch);
  EXPECT_EQ(changed.state().committed_samples, before.committed_samples);
  EXPECT_EQ(changed.state().schema_descriptor, before.schema_descriptor);
}

TEST(IterableDataPipelineTest, BoundedAndBestEffortPoliciesAreExplicit) {
  auto bounded_descriptor = exact_descriptor();
  bounded_descriptor.restore_level = jyppx::RestoreLevel::bounded;
  bounded_descriptor.max_replayed_samples = 5;
  bounded_descriptor.max_omitted_samples = 1;
  bounded_descriptor.degradation_reasons = {"remote shard rebalance"};
  auto bounded_source = make_source({1, 2, 3}, bounded_descriptor);
  IterableDataPipeline bounded(bounded_source, options("data.iterable.bounded"),
                               sum_factory());
  EXPECT_EQ(bounded.capability().max_replayed_batches, 2U);
  const auto bounded_state = save_state(bounded);
  EXPECT_THROW(
      bounded.load_state(bounded_state, jyppx::RestorePolicy::exact_only),
      jyppx::CompatibilityError);
  EXPECT_NO_THROW(
      bounded.load_state(bounded_state, jyppx::RestorePolicy::allow_bounded));

  auto best_descriptor = exact_descriptor();
  best_descriptor.checkpointable = false;
  best_descriptor.deterministic_epoch = false;
  best_descriptor.restore_level = jyppx::RestoreLevel::best_effort;
  best_descriptor.max_replayed_samples.reset();
  best_descriptor.max_omitted_samples.reset();
  best_descriptor.degradation_reasons = {"external cursor is unavailable"};
  auto best_source = make_source({8, 9, 10}, best_descriptor);
  auto best_config = options("data.iterable.best-effort", 1);
  IterableDataPipeline best(best_source, best_config, sum_factory());
  best.set_epoch(1);
  ASSERT_TRUE(best.next().has_value());
  auto checkpoint = save_state(best);
  IterableDataPipeline reopened(best_source, best_config, sum_factory());
  EXPECT_NO_THROW(
      reopened.load_state(checkpoint, jyppx::RestorePolicy::allow_best_effort));
  const auto replayed = reopened.next();
  ASSERT_TRUE(replayed.has_value());
  EXPECT_EQ(replayed->sample_ids.front(), 1000U);
}

TEST(IterableDataPipelineTest, CancellationAndTimeoutDoNotCommitSourceState) {
  auto source = std::make_shared<BlockingSource>();
  auto config = options("data.iterable.cancel", 1);
  IterableDataPipeline cancelled(source, config, sum_factory());
  cancelled.set_epoch(0);
  std::stop_source stop;
  std::jthread request_stop([&stop] {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stop.request_stop();
  });
  EXPECT_FALSE(cancelled.next(stop.get_token()).has_value());
  EXPECT_EQ(cancelled.state().committed_samples, 0U);
  EXPECT_THROW(cancelled.next(), jyppx::Cancelled);

  auto timeout_config = options("data.iterable.timeout", 1);
  timeout_config.batch_timeout = std::chrono::milliseconds(25);
  IterableDataPipeline timeout(source, timeout_config, sum_factory());
  timeout.set_epoch(0);
  EXPECT_THROW(timeout.next(), jyppx::DeadlineExceeded);
  EXPECT_EQ(timeout.state().committed_samples, 0U);
  EXPECT_THROW(timeout.next(), jyppx::DeadlineExceeded);
}

TEST(IterableDataPipelineTest, UserFailureIsStructuredAndStopsEpoch) {
  auto source = make_source({1, 2, 3});
  IterableDataPipeline pipeline(
      source, options("data.iterable.failure"),
      [](std::span<const IterableSample>, std::uint64_t, std::uint64_t,
         std::stop_token) -> c10::IValue {
        throw std::runtime_error("synthetic iterable failure");
      });
  pipeline.set_epoch(0);
  EXPECT_THROW(pipeline.next(), jyppx::BackendError);
  EXPECT_EQ(pipeline.state().committed_samples, 0U);
  EXPECT_THROW(pipeline.next(), jyppx::BackendError);
}

TEST(IterableDataPipelineTest, SourceFailureIsStructuredAndStopsEpoch) {
  IterableDataPipeline pipeline(std::make_shared<ThrowingSource>(),
                                options("data.iterable.source-failure"),
                                sum_factory());
  pipeline.set_epoch(0);
  EXPECT_THROW(pipeline.next(), jyppx::BackendError);
  EXPECT_EQ(pipeline.state().committed_samples, 0U);
  EXPECT_THROW(pipeline.next(), jyppx::BackendError);
}

TEST(IterableDataPipelineTest, RejectsInvalidWatermarksWithoutCommittingState) {
  IterableDataPipeline oversized(
      make_source({1}, exact_descriptor(), nullptr, std::string(4097, 'x')),
      options("data.iterable.oversized-watermark"), sum_factory());
  oversized.set_epoch(0);
  EXPECT_THROW(oversized.next(), jyppx::ResourceExhausted);
  EXPECT_EQ(oversized.state().committed_samples, 0U);

  IterableDataPipeline empty(
      make_source({1}, exact_descriptor(), nullptr, std::string{}),
      options("data.iterable.empty-watermark"), sum_factory());
  empty.set_epoch(0);
  EXPECT_THROW(empty.next(), jyppx::InvalidArgument);
  EXPECT_EQ(empty.state().committed_samples, 0U);
}

TEST(IterableDataPipelineTest,
     FailedEpochPreparationLeavesPreviousRuntimeUsable) {
  auto source = make_source({1, 2, 3, 4});
  source->fail_epoch(2);
  IterableDataPipeline pipeline(source, options("data.iterable.rollback"),
                                sum_factory());
  pipeline.set_epoch(1);
  ASSERT_TRUE(pipeline.next().has_value());
  const auto before = pipeline.state();
  EXPECT_THROW(pipeline.set_epoch(2), jyppx::BackendError);
  EXPECT_EQ(pipeline.state().epoch, before.epoch);
  EXPECT_EQ(pipeline.state().committed_samples, before.committed_samples);
  EXPECT_TRUE(pipeline.next().has_value());
}

TEST(IterableDataPipelineTest, InvalidCheckpointDoesNotMutatePreparedPipeline) {
  auto source = make_source({1, 2, 3});
  auto config = options("data.iterable.atomic");
  IterableDataPipeline pipeline(source, config, sum_factory());
  pipeline.set_epoch(4);
  const auto before = pipeline.state();

  MemoryStateStore malformed(pipeline.describe_state());
  EXPECT_THROW(pipeline.load_state(malformed, jyppx::RestorePolicy::exact_only),
               std::exception);
  const auto after = pipeline.state();
  EXPECT_EQ(after.epoch, before.epoch);
  EXPECT_EQ(after.prepared, before.prepared);
  EXPECT_EQ(after.source_state, before.source_state);
}

TEST(IterableBatchMemoryCudaTest, PinsTransfersAndRestoresExactCursor) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA runtime is unavailable";
  }
  auto config = options("data.iterable.cuda-memory", 2);
  config.prefetch_batches = 3;
  config.memory = {.pin_memory = true,
                   .target_device = jyppx::Device(torch::kCUDA, 0),
                   .non_blocking = true,
                   .stage_capacity = 1};
  config.schema = jyppx::data::BatchSchema::tensor(
      "batch.iterable",
      {.dtype = torch::kFloat32,
       .device = jyppx::Device(torch::kCUDA, 0),
       .rank = 1},
      1, "features");
  const auto factory = [](std::span<const IterableSample> samples,
                          std::uint64_t, std::uint64_t, std::stop_token) {
    std::vector<std::int64_t> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) {
      values.push_back(sample.value.toInt());
    }
    return c10::IValue(torch::tensor(values, torch::kFloat32));
  };
  auto source = make_source({1, 2, 3, 4, 5, 6});
  IterableDataPipeline original(source, config, factory);
  original.set_epoch(9);
  const auto first = original.next();
  ASSERT_TRUE(first.has_value());
  EXPECT_TRUE(first->value.toTensor().is_cuda());
  EXPECT_EQ(first->memory.pinned_tensor_count, 1U);
  EXPECT_EQ(first->memory.transferred_tensor_count, 1U);

  const auto checkpoint = save_state(original);
  const auto expected = original.next();
  ASSERT_TRUE(expected.has_value());
  IterableDataPipeline restored(source, config, factory);
  restored.load_state(checkpoint, jyppx::RestorePolicy::exact_only);
  const auto actual = restored.next();
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(actual->sample_ids, expected->sample_ids);
  EXPECT_TRUE(actual->value.toTensor().is_cuda());
  EXPECT_TRUE(
      torch::equal(actual->value.toTensor(), expected->value.toTensor()));
  EXPECT_EQ(actual->memory.pinned_tensor_count, 1U);
  EXPECT_EQ(actual->memory.transferred_tensor_count, 1U);
}

}  // namespace
