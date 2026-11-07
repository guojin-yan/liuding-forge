// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/image_output_sink.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <latch>
#include <memory>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

namespace {

using namespace jyppx::liuding;
using namespace jyppx::liuding::data::image;

class SinkDirectory final {
 public:
  SinkDirectory() {
    static std::atomic<std::uint64_t> next{0};
    path = std::filesystem::temp_directory_path() /
           ("liuding-image-sink-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(next.fetch_add(1)));
    std::filesystem::create_directories(path);
  }
  ~SinkDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  std::filesystem::path path;
};

EncodedImage Png(std::uint8_t value = 0) {
  PixelSpec spec;
  spec.layout = ImageLayout::hwc;
  spec.color = PixelColorModel::rgb;
  spec.alpha = PixelAlphaMode::none;
  spec.dtype = torch::kUInt8;
  spec.range = PixelValueRange::unsigned_full_scale;
  return encode_image(torch::full({2, 3, 3}, value, torch::kUInt8), spec);
}

std::vector<std::byte> Read(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  const std::string bytes{std::istreambuf_iterator<char>{input},
                          std::istreambuf_iterator<char>{}};
  const auto view = std::as_bytes(std::span(bytes));
  return {view.begin(), view.end()};
}

TEST(ImageOutputSinkTest, PublishesCompleteBytesAndReleasesBudgetLeases) {
  SinkDirectory directory;
  auto image = Png();
  const auto expected = image.bytes;
  ImageResourceBudgetLimits limits;
  limits.max_encoded_queue_bytes = expected.size();
  limits.max_disk_spool_bytes = expected.size();
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  image.output_lease =
      budget->reserve(ImageResourceKind::encoded_bytes, image.bytes.capacity());
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).current,
            image.bytes.capacity());
  ImageOutputSinkOptions options;
  options.resource_budget = budget;
  options.max_pending_bytes = expected.size();
  options.max_item_bytes = expected.size();
  ImageOutputSink sink(options);
  const auto path = directory.path / "result.png";
  auto ticket = sink.submit(path, std::move(image));
  ASSERT_EQ(ticket.disposition, ImageSinkDisposition::queued);
  const auto result = ticket.completion.get();
  sink.flush();
  sink.close();
  EXPECT_EQ(result.path, path);
  EXPECT_EQ(result.bytes, expected.size());
  EXPECT_EQ(result.sha256, sha256_hex(expected));
  EXPECT_FALSE(ticket.evidence_completion.valid());
  EXPECT_EQ(Read(path), expected);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).current, 0U);
  EXPECT_EQ(sink.stats().completed, 1U);
  EXPECT_EQ(sink.stats().pending_bytes, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_queue_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::disk_spool_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::disk_spool_bytes).peak,
            expected.size());
  for (const auto& entry : std::filesystem::directory_iterator(directory.path))
    EXPECT_EQ(entry.path(), path);
}

TEST(ImageOutputSinkTest,
     NoReplaceFailurePreservesExistingOutputAndPropagates) {
  SinkDirectory directory;
  const auto path = directory.path / "result.png";
  const auto old = Png(3).bytes;
  {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(old.data()), old.size());
  }
  ImageOutputSink sink;
  auto ticket = sink.submit(path, Png(9));
  EXPECT_THROW(ticket.completion.get(), IoError);
  EXPECT_THROW(sink.flush(), IoError);
  EXPECT_THROW(sink.close(), IoError);
  EXPECT_EQ(Read(path), old);
  EXPECT_EQ(sink.stats().failed, 1U);
  ASSERT_EQ(sink.failures().size(), 1U);
  EXPECT_EQ(sink.failures().front().path, path);
  for (const auto& entry : std::filesystem::directory_iterator(directory.path))
    EXPECT_EQ(entry.path(), path);
}

TEST(ImageOutputSinkTest, ReplaceExistingPublishesLastCompleteImage) {
  SinkDirectory directory;
  ImageOutputSinkOptions options;
  options.replace_existing = true;
  ImageOutputSink sink(options);
  const auto path = directory.path / "result.png";
  auto first = sink.submit(path, Png(1));
  first.completion.get();
  auto second_image = Png(2);
  const auto expected = second_image.bytes;
  auto second = sink.submit(path, std::move(second_image));
  second.completion.get();
  sink.close();
  EXPECT_EQ(Read(path), expected);
  EXPECT_EQ(sink.stats().completed, 2U);
}

TEST(ImageOutputSinkTest, IdenticalReplayPreservesFileAndRunsEvidenceAgain) {
  SinkDirectory directory;
  const auto path = directory.path / "result.png";
  auto image = Png(7);
  // Exercise comparisons across multiple bounded reads, including the tail.
  image.bytes.resize(196613U, std::byte{37});
  const auto expected = image.bytes;
  {
    ImageOutputSink first;
    first.submit(path, std::move(image)).completion.get();
    first.close();
  }
  const auto alias = directory.path / "same-inode.png";
  std::filesystem::create_hard_link(path, alias);
  const auto modified = std::filesystem::last_write_time(path);
  auto budget = std::make_shared<ImageResourceBudget>();
  ImageOutputSinkOptions options;
  options.replay_identical_existing = true;
  options.resource_budget = budget;
  ImageOutputSink replay(options);
  std::size_t callbacks = 0;
  auto retry = Png(7);
  retry.bytes = expected;
  auto ticket = replay.submit_with_evidence(
      path, std::move(retry), [&](const ImageSinkWriteResult& result) {
        ++callbacks;
        EXPECT_EQ(result.sha256, sha256_hex(expected));
        EXPECT_EQ(result.bytes, expected.size());
      });
  EXPECT_EQ(ticket.completion.get().path, path);
  EXPECT_NO_THROW(ticket.evidence_completion.get());
  replay.close();
  EXPECT_EQ(callbacks, 1U);
  EXPECT_TRUE(std::filesystem::equivalent(path, alias));
  EXPECT_EQ(std::filesystem::last_write_time(path), modified);
  EXPECT_EQ(Read(path), expected);
  EXPECT_EQ(replay.stats().completed, 1U);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_queue_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::disk_spool_bytes).current, 0U);
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory.path),
                          std::filesystem::directory_iterator{}),
            2);
}

TEST(ImageOutputSinkTest, IdenticalReplayRejectsConflictingContentAndLength) {
  SinkDirectory directory;
  const auto path = directory.path / "result.png";
  auto original = Png(7);
  original.bytes.resize(196613U, std::byte{37});
  const auto expected = original.bytes;
  {
    ImageOutputSink first;
    first.submit(path, std::move(original)).completion.get();
    first.close();
  }
  const auto modified = std::filesystem::last_write_time(path);
  for (const auto index : {8U, 65536U, 196612U, 196613U, 196614U}) {
    auto different = Png(7);
    different.bytes = expected;
    if (index < expected.size())
      different.bytes[index] ^= std::byte{1};
    else
      different.bytes.resize(index == expected.size() ? expected.size() - 1U
                                                      : expected.size() + 1U);
    ImageOutputSinkOptions options;
    options.replay_identical_existing = true;
    ImageOutputSink sink(options);
    bool called = false;
    auto ticket = sink.submit_with_evidence(
        path, std::move(different), [&](const auto&) { called = true; });
    EXPECT_THROW(ticket.completion.get(), IoError);
    EXPECT_THROW(ticket.evidence_completion.get(), IoError);
    EXPECT_THROW(sink.close(), IoError);
    EXPECT_FALSE(called);
    EXPECT_EQ(Read(path), expected);
    EXPECT_EQ(std::filesystem::last_write_time(path), modified);
    EXPECT_EQ(sink.stats().failed, 1U);
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory.path),
                            std::filesystem::directory_iterator{}),
              1);
  }
}

TEST(ImageOutputSinkTest, IdenticalReplayDoesNotFollowSymlink) {
  SinkDirectory directory;
  const auto original = directory.path / "original.png";
  {
    ImageOutputSink first;
    first.submit(original, Png(7)).completion.get();
    first.close();
  }
  const auto path = directory.path / "symlink.png";
  std::error_code error;
  std::filesystem::create_symlink(original, path, error);
  if (error)
    GTEST_SKIP() << "symlink creation unavailable: " << error.message();
  ImageOutputSinkOptions options;
  options.replay_identical_existing = true;
  ImageOutputSink sink(options);
  auto ticket = sink.submit(path, Png(7));
  EXPECT_THROW(ticket.completion.get(), IoError);
  EXPECT_THROW(sink.close(), IoError);
  EXPECT_TRUE(std::filesystem::is_symlink(path));
  EXPECT_EQ(Read(original), Png(7).bytes);
}

TEST(ImageOutputSinkTest, IdenticalReplaySerializesConcurrentPublishers) {
  SinkDirectory directory;
  ImageOutputSinkOptions options;
  options.replay_identical_existing = true;
  ImageOutputSink first(options), second(options);
  const auto path = directory.path / "result.png";
  std::latch ready(2), proceed(1);
  auto publish = [&](ImageOutputSink& sink) {
    auto image = Png(7);
    ready.count_down();
    proceed.wait();
    return sink.submit(path, std::move(image)).completion.get();
  };
  auto a = std::async(std::launch::async, [&] { return publish(first); });
  auto b = std::async(std::launch::async, [&] { return publish(second); });
  ready.wait();
  proceed.count_down();
  const auto left = a.get(), right = b.get();
  first.close();
  second.close();
  EXPECT_EQ(left.sha256, right.sha256);
  EXPECT_EQ(left.bytes, right.bytes);
  EXPECT_EQ(Read(path), Png(7).bytes);
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory.path),
                          std::filesystem::directory_iterator{}),
            1);
}

TEST(ImageOutputSinkTest, ExistingFilePoliciesAreExplicitAndExclusive) {
  ImageOutputSinkOptions options;
  options.replay_identical_existing = true;
  options.replace_existing = true;
  EXPECT_THROW(ImageOutputSink{options}, InvalidArgument);
  SinkDirectory directory;
  const auto path = directory.path / "result.png";
  {
    ImageOutputSink first;
    first.submit(path, Png(7)).completion.get();
    first.close();
  }
  ImageOutputSink strict;
  auto duplicate = strict.submit(path, Png(7));
  EXPECT_THROW(duplicate.completion.get(), IoError);
  EXPECT_THROW(strict.close(), IoError);
}

TEST(ImageOutputSinkTest, DropNewestKeepsOutstandingWorkBounded) {
  SinkDirectory directory;
  ImageOutputSinkOptions options;
  options.max_pending_items = 1;
  options.overflow = ImageSinkOverflowPolicy::drop_newest;
  ImageOutputSink sink(options);
  const auto reference = Png();
  std::vector<std::future<ImageSinkWriteResult>> accepted;
  for (int index = 0; index < 128; ++index) {
    auto image =
        EncodedImage{reference.bytes,  reference.format,   reference.width,
                     reference.height, reference.channels, {}};
    auto ticket = sink.submit(
        directory.path / ("output-" + std::to_string(index) + ".png"),
        std::move(image));
    EXPECT_LE(sink.stats().pending_items, 1U);
    if (ticket.disposition == ImageSinkDisposition::queued)
      accepted.push_back(std::move(ticket.completion));
    else
      EXPECT_FALSE(ticket.completion.valid());
  }
  for (auto& completion : accepted)
    EXPECT_TRUE(std::filesystem::exists(completion.get().path));
  sink.close();
  EXPECT_GT(sink.stats().dropped, 0U);
  EXPECT_EQ(sink.stats().queued, accepted.size());
  EXPECT_EQ(sink.stats().submitted, sink.stats().queued + sink.stats().dropped);
}

TEST(ImageOutputSinkTest, ContinueAfterErrorPublishesLaterItem) {
  SinkDirectory directory;
  ImageOutputSinkOptions options;
  options.continue_after_error = true;
  ImageOutputSink sink(options);
  auto failing = sink.submit(directory.path / "missing" / "bad.png", Png());
  const auto good_path = directory.path / "good.png";
  auto good = sink.submit(good_path, Png());
  EXPECT_THROW(failing.completion.get(), IoError);
  EXPECT_EQ(good.completion.get().path, good_path);
  EXPECT_THROW(sink.flush(), IoError);
  EXPECT_THROW(sink.close(), IoError);
  EXPECT_EQ(sink.stats().completed, 1U);
  EXPECT_EQ(sink.stats().failed, 1U);
}

TEST(ImageOutputSinkTest, SamplingIsStableAcrossSubmissionOrder) {
  SinkDirectory directory;
  ImageOutputSinkOptions options;
  options.sample_every = 3;
  ImageOutputSink first(options);
  ImageOutputSink second(options);
  std::vector<ImageSinkDisposition> decisions;
  for (int index = 0; index < 24; ++index) {
    auto key = "sample-" + std::to_string(index);
    auto ticket =
        first.submit(directory.path / ("first-" + key + ".png"), Png(), key);
    decisions.push_back(ticket.disposition);
    if (ticket.disposition == ImageSinkDisposition::queued)
      ticket.completion.get();
  }
  for (int index = 23; index >= 0; --index) {
    auto key = "sample-" + std::to_string(index);
    auto ticket =
        second.submit(directory.path / ("second-" + key + ".png"), Png(), key);
    EXPECT_EQ(ticket.disposition, decisions[index]);
    if (ticket.disposition == ImageSinkDisposition::queued)
      ticket.completion.get();
  }
  first.close();
  second.close();
  EXPECT_GT(first.stats().sampled_out, 0U);
  EXPECT_GT(first.stats().completed, 0U);
  EXPECT_EQ(first.stats().sampled_out, second.stats().sampled_out);
  EXPECT_EQ(first.stats().completed, second.stats().completed);
}

TEST(ImageOutputSinkTest, RejectsOversizeAndCancelledAdmission) {
  SinkDirectory directory;
  auto image = Png();
  ImageOutputSinkOptions options;
  options.max_item_bytes = image.bytes.size() - 1;
  ImageOutputSink sink(options);
  EXPECT_THROW(sink.submit(directory.path / "large.png", Png()),
               ResourceExhausted);
  std::stop_source cancelled;
  cancelled.request_stop();
  ImageOutputSink normal;
  EXPECT_THROW(normal.submit(directory.path / "cancelled.png", Png(), {},
                             cancelled.get_token()),
               Cancelled);
  EXPECT_FALSE(std::filesystem::exists(directory.path / "cancelled.png"));
}

TEST(ImageOutputSinkTest,
     EvidenceRunsAfterPublicationWithReleasedEncodedBytesAndHeldCapacity) {
  using namespace std::chrono_literals;
  SinkDirectory directory;
  auto image = Png();
  const auto bytes = image.bytes.size();
  ImageResourceBudgetLimits limits;
  limits.max_encoded_queue_bytes = bytes;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  ImageOutputSinkOptions options;
  options.resource_budget = budget;
  options.max_pending_items = 1;
  options.max_pending_bytes = bytes;
  options.overflow = ImageSinkOverflowPolicy::drop_newest;
  ImageOutputSink sink(options);
  std::promise<void> entered;
  std::promise<void> release;
  auto released = release.get_future().share();
  const auto path = directory.path / "result.png";
  auto ticket = sink.submit_with_evidence(
      path, std::move(image), [&](const ImageSinkWriteResult& published) {
        EXPECT_EQ(published.path, path);
        EXPECT_TRUE(std::filesystem::exists(path));
        entered.set_value();
        released.wait();
      });
  const auto entered_status = entered.get_future().wait_for(5s);
  const auto file_status = ticket.completion.wait_for(5s);
  const auto evidence_status = ticket.evidence_completion.wait_for(0s);
  const auto pending = sink.stats();
  const auto encoded_usage =
      budget->usage(ImageResourceKind::encoded_queue_bytes).current;
  auto dropped = sink.submit_with_evidence(
      directory.path / "dropped.png", Png(),
      [](const ImageSinkWriteResult&) { FAIL() << "dropped callback ran"; });
  release.set_value();
  EXPECT_EQ(entered_status, std::future_status::ready);
  EXPECT_EQ(file_status, std::future_status::ready);
  EXPECT_EQ(evidence_status, std::future_status::timeout);
  EXPECT_EQ(pending.pending_items, 1U);
  EXPECT_EQ(pending.pending_bytes, bytes);
  EXPECT_EQ(encoded_usage, 0U);
  EXPECT_EQ(dropped.disposition, ImageSinkDisposition::dropped);
  EXPECT_FALSE(dropped.completion.valid());
  EXPECT_FALSE(dropped.evidence_completion.valid());
  EXPECT_EQ(ticket.completion.get().path, path);
  EXPECT_NO_THROW(ticket.evidence_completion.get());
  sink.close();
  EXPECT_EQ(sink.stats().completed, 1U);
}

TEST(ImageOutputSinkTest, EvidenceFailureKeepsFileResultForRetry) {
  SinkDirectory directory;
  ImageOutputSink sink;
  const auto path = directory.path / "result.png";
  auto ticket =
      sink.submit_with_evidence(path, Png(), [](const ImageSinkWriteResult&) {
        throw std::runtime_error("evidence unavailable");
      });
  const auto published = ticket.completion.get();
  EXPECT_EQ(published.path, path);
  EXPECT_EQ(published.sha256, sha256_hex(Read(path)));
  EXPECT_THROW(ticket.evidence_completion.get(), std::runtime_error);
  EXPECT_THROW(sink.flush(), std::runtime_error);
  EXPECT_THROW(sink.close(), std::runtime_error);
  EXPECT_EQ(sink.stats().failed, 1U);
  EXPECT_EQ(sink.stats().completed, 0U);
  ASSERT_EQ(sink.failures().size(), 1U);
  EXPECT_EQ(sink.failures().front().path, path);
}

TEST(ImageOutputSinkTest, EvidenceSkipsFailedAndSampledPublications) {
  SinkDirectory directory;
  ImageOutputSink failing;
  std::atomic<int> invoked{0};
  auto failed = failing.submit_with_evidence(
      directory.path / "missing" / "result.png", Png(),
      [&](const ImageSinkWriteResult&) { ++invoked; });
  EXPECT_THROW(failed.completion.get(), IoError);
  EXPECT_THROW(failed.evidence_completion.get(), IoError);
  EXPECT_THROW(failing.close(), IoError);
  EXPECT_EQ(invoked.load(), 0);

  ImageOutputSinkOptions options;
  options.sample_every = 17;
  ImageOutputSink sampled(options);
  std::size_t queued = 0;
  std::size_t skipped = 0;
  for (int index = 0; index < 40; ++index) {
    auto ticket = sampled.submit_with_evidence(
        directory.path / ("sample-" + std::to_string(index) + ".png"), Png(),
        [&](const ImageSinkWriteResult&) { ++invoked; },
        "sample-" + std::to_string(index));
    if (ticket.disposition == ImageSinkDisposition::queued) {
      ++queued;
      ticket.completion.get();
      ticket.evidence_completion.get();
    } else {
      ++skipped;
      EXPECT_EQ(ticket.disposition, ImageSinkDisposition::sampled_out);
      EXPECT_FALSE(ticket.completion.valid());
      EXPECT_FALSE(ticket.evidence_completion.valid());
    }
  }
  sampled.close();
  EXPECT_GT(skipped, 0U);
  EXPECT_EQ(invoked.load(), queued);
  EXPECT_THROW(
      sampled.submit_with_evidence(directory.path / "empty.png", Png(), {}),
      InvalidArgument);
}

TEST(ImageOutputSinkTest, ContinuesPublishingAfterEvidenceFailureWhenEnabled) {
  SinkDirectory directory;
  ImageOutputSinkOptions options;
  options.continue_after_error = true;
  ImageOutputSink sink(options);
  auto failed = sink.submit_with_evidence(
      directory.path / "first.png", Png(), [](const ImageSinkWriteResult&) {
        throw std::runtime_error("evidence unavailable");
      });
  EXPECT_TRUE(std::filesystem::exists(failed.completion.get().path));
  EXPECT_THROW(failed.evidence_completion.get(), std::runtime_error);
  auto later = sink.submit(directory.path / "second.png", Png(7));
  EXPECT_TRUE(std::filesystem::exists(later.completion.get().path));
  EXPECT_THROW(sink.close(), std::runtime_error);
  EXPECT_EQ(sink.stats().failed, 1U);
  EXPECT_EQ(sink.stats().completed, 1U);
}

TEST(ImageOutputSinkTest,
     BlockedAdmissionAndFlushCancelWithoutLosingAcceptedFile) {
  using namespace std::chrono_literals;
  SinkDirectory directory;
  auto budget = std::make_shared<ImageResourceBudget>();
  ImageOutputSinkOptions options;
  options.resource_budget = budget;
  options.max_pending_items = 1;
  ImageOutputSink sink(options);
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  const auto expected = Png(7).bytes;
  const auto path = directory.path / "accepted.png";
  auto accepted = sink.submit_with_evidence(path, Png(7), [&](const auto&) {
    entered.set_value();
    released.wait();
  });
  const auto status = entered.get_future().wait_for(5s);
  // Release before any fatal assertion or exception can destroy a blocked sink.
  if (status != std::future_status::ready) {
    release.set_value();
    FAIL() << "writer callback did not start";
  }
  std::stop_source admission_stop, flush_stop;
  std::promise<void> admission_started, flush_started;
  auto rejected = std::async(std::launch::async, [&] {
    admission_started.set_value();
    EXPECT_THROW(sink.submit(directory.path / "cancelled.png", Png(), {},
                             admission_stop.get_token()),
                 Cancelled);
  });
  auto flushed = std::async(std::launch::async, [&] {
    flush_started.set_value();
    EXPECT_THROW(sink.flush(flush_stop.get_token()), Cancelled);
  });
  admission_started.get_future().wait();
  flush_started.get_future().wait();
  admission_stop.request_stop();
  flush_stop.request_stop();
  const auto admission_status = rejected.wait_for(5s);
  const auto flush_status = flushed.wait_for(5s);
  const auto pending = sink.stats();
  release.set_value();
  EXPECT_EQ(admission_status, std::future_status::ready);
  EXPECT_EQ(flush_status, std::future_status::ready);
  rejected.get();
  flushed.get();
  EXPECT_EQ(pending.pending_items, 1U);
  EXPECT_EQ(pending.queued, 1U);
  EXPECT_EQ(accepted.completion.get().sha256, sha256_hex(expected));
  EXPECT_NO_THROW(accepted.evidence_completion.get());
  EXPECT_NO_THROW(sink.flush());
  EXPECT_NO_THROW(sink.close());
  EXPECT_NO_THROW(sink.close());
  EXPECT_EQ(Read(path), expected);
  EXPECT_FALSE(std::filesystem::exists(directory.path / "cancelled.png"));
  EXPECT_EQ(sink.stats().completed, 1U);
  EXPECT_EQ(sink.stats().pending_items, 0U);
  EXPECT_EQ(sink.stats().pending_bytes, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_queue_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::disk_spool_bytes).current, 0U);
  EXPECT_THROW(sink.submit(directory.path / "closed.png", Png()),
               InvalidArgument);
}

TEST(ImageOutputSinkTest, RejectsInvalidPolicyImageAndKeyBeforePublication) {
  SinkDirectory directory;
  for (int field = 0; field < 5; ++field) {
    ImageOutputSinkOptions options;
    if (field == 0) options.max_pending_items = 0;
    if (field == 1) options.max_pending_bytes = 0;
    if (field == 2) options.max_item_bytes = 0;
    if (field == 3) options.sample_every = 0;
    if (field == 4) options.overflow = static_cast<ImageSinkOverflowPolicy>(99);
    EXPECT_THROW(ImageOutputSink{options}, InvalidArgument) << field;
  }
  ImageOutputSink sink;
  EXPECT_THROW(sink.submit({}, Png()), InvalidArgument);
  EXPECT_THROW(sink.submit(directory.path / "wrong.jpg", Png()),
               InvalidArgument);
  auto invalid = Png();
  invalid.width = 0;
  EXPECT_THROW(sink.submit(directory.path / "invalid.png", std::move(invalid)),
               InvalidArgument);
  invalid = Png();
  invalid.format = static_cast<ImageFormat>(99);
  EXPECT_THROW(sink.submit(directory.path / "invalid.png", std::move(invalid)),
               InvalidArgument);
  invalid = Png();
  invalid.format = ImageFormat::jpeg;
  EXPECT_THROW(sink.submit(directory.path / "invalid.jpg", std::move(invalid)),
               InvalidArgument);
  ImageOutputSinkOptions options;
  options.sample_every = 2;
  ImageOutputSink sampled(options);
  EXPECT_THROW(sampled.submit(directory.path / "no-key.png", Png()),
               InvalidArgument);
  EXPECT_THROW(sampled.submit(directory.path / "long-key.png", Png(),
                              std::string(4097, 'x')),
               InvalidArgument);
  sink.close();
  sampled.close();
  EXPECT_EQ(sink.stats().submitted, 0U);
  EXPECT_TRUE(std::filesystem::is_empty(directory.path));
}

TEST(ImageOutputSinkTest,
     FailedWriterRejectsLaterAdmissionAndBoundsDiagnostics) {
  SinkDirectory directory;
  ImageOutputSinkOptions options;
  options.max_reported_failures = 0;
  auto budget = std::make_shared<ImageResourceBudget>();
  options.resource_budget = budget;
  ImageOutputSink sink(options);
  auto failed = sink.submit(directory.path / "missing" / "bad.png", Png());
  EXPECT_THROW(failed.completion.get(), IoError);
  EXPECT_THROW(sink.flush(), IoError);
  EXPECT_THROW(sink.submit(directory.path / "later.png", Png()), IoError);
  EXPECT_THROW(sink.close(), IoError);
  EXPECT_TRUE(sink.failures().empty());
  EXPECT_EQ(sink.stats().failed, 1U);
  EXPECT_EQ(sink.stats().queued, 1U);
  EXPECT_EQ(sink.stats().pending_bytes, 0U);
  EXPECT_FALSE(std::filesystem::exists(directory.path / "later.png"));
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_queue_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::disk_spool_bytes).current, 0U);
}

}  // namespace
