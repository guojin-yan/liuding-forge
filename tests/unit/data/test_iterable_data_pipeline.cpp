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

 private:
  jyppx::StateManifest manifest_;  ///< Immutable participant descriptor.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Encoded participant fields.
};

/** @brief Shared counters used to assert bounded producer behavior. */
struct CursorStats {
  mutable std::mutex mutex;         ///< Protects counter and close state.
  std::condition_variable changed;  ///< Announces pulls and closure.
  std::size_t pulls = 0;            ///< Number of values returned by `next()`.
  bool closed = false;              ///< Whether `close()` was invoked.
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
      stats_->changed.notify_all();
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
      stats_->changed.notify_all();
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
          .schema_id = "batch.iterable/v1",
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

bool wait_for_pulls(const std::shared_ptr<CursorStats>& stats,
                    std::size_t expected, std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(stats->mutex);
  return stats->changed.wait_for(lock, timeout,
                                 [&] { return stats->pulls >= expected; });
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

}  // namespace
