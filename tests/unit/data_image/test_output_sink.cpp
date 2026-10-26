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
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <span>
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
  EXPECT_EQ(Read(path), expected);
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

}  // namespace
