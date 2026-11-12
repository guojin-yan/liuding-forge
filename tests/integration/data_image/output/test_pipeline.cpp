// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");

#include "pipeline_fixture.hpp"

namespace {
using namespace image_output_pipeline_test;

TEST(ImageOutputPipelineTest,
     SlowEvidenceRetainsCountAndBytesBeforeNextDecode) {
  Fixture f;
  auto first = f.pipeline->next();
  ASSERT_TRUE(first);
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  auto release_future = release.get_future();
  image::ImageOutputSink sink;
  auto ticket = sink.produce_and_submit(
      f.root / "preview.png", kBatchBytes,
      [&](std::stop_token stop) { return f.preview(*first, stop); }, {}, {},
      [&, lease = first->admission_lease](const image::ImageSinkWriteResult&) {
        (void)lease;
        entered.set_value();
        if (release_future.wait_for(10s) != std::future_status::ready)
          throw std::runtime_error("Test evidence gate timed out");
      });
  first.reset();
  auto next =
      std::async(std::launch::async, [&] { return f.pipeline->next(); });
  EXPECT_EQ(entered_future.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(f.blocked_attempt.get_future().wait_for(5s),
            std::future_status::ready);
  EXPECT_EQ(next.wait_for(0s), std::future_status::timeout);
  EXPECT_EQ(f.decoded, 2U);
  EXPECT_EQ(f.acquired, 1U);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::pipeline_inflight_batches)
                .current,
            1U);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::pipeline_inflight_bytes)
                .current,
            kBatchBytes);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::decoded_bytes).current,
            0U);
  release.set_value();
  ticket.evidence_completion.get();
  auto second = next.get();
  EXPECT_TRUE(second);
  EXPECT_EQ(f.decoded, 4U);
  second.reset();
  sink.close();
  f.expect_released();
}

TEST(ImageOutputPipelineTest,
     CancellationWhileEvidenceOwnsAdmissionDoesNotDecode) {
  Fixture f;
  auto first = f.pipeline->next();
  ASSERT_TRUE(first);
  auto lease = first->admission_lease;
  first.reset();
  EXPECT_EQ(f.blocked_attempt.get_future().wait_for(5s),
            std::future_status::ready);
  std::stop_source stop;
  auto next = std::async(std::launch::async,
                         [&] { return f.pipeline->next(stop.get_token()); });
  EXPECT_EQ(next.wait_for(0s), std::future_status::timeout);
  stop.request_stop();
  EXPECT_FALSE(next.get());
  f.pipeline->close();
  EXPECT_EQ(f.decoded, 2U);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::pipeline_inflight_bytes)
                .current,
            kBatchBytes);
  lease.reset();
  f.expect_released();
}

TEST(ImageOutputPipelineTest,
     FailedEvidenceReleasesAdmissionAndPreservesPublishedFile) {
  Fixture f;
  auto first = f.pipeline->next();
  ASSERT_TRUE(first);
  image::ImageOutputSink sink;
  auto ticket = sink.produce_and_submit(
      f.root / "preview.png", kBatchBytes,
      [&](std::stop_token stop) { return f.preview(*first, stop); }, {}, {},
      [lease = first->admission_lease](const image::ImageSinkWriteResult&) {
        (void)lease;
        throw std::runtime_error("Evidence rejected");
      });
  first.reset();
  EXPECT_NO_THROW(ticket.completion.get());
  EXPECT_THROW(ticket.evidence_completion.get(), std::runtime_error);
  EXPECT_THROW(sink.close(), std::runtime_error);
  EXPECT_TRUE(std::filesystem::is_regular_file(f.root / "preview.png"));
  auto second = f.pipeline->next();
  EXPECT_TRUE(second);
  second.reset();
  f.expect_released();
}

TEST(ImageOutputPipelineTest,
     FailedProducerReleasesOutputReservationWithoutEvidence) {
  Fixture f;
  auto first = f.pipeline->next();
  ASSERT_TRUE(first);
  image::ImageOutputSink sink({.resource_budget = f.budget});
  bool recorded = false;
  EXPECT_THROW(sink.produce_and_submit(
                   f.root / "preview.png", kBatchBytes,
                   [](std::stop_token) -> image::EncodedImage {
                     throw std::runtime_error("Encoder rejected");
                   },
                   {}, {},
                   [&, lease = first->admission_lease](
                       const image::ImageSinkWriteResult&) {
                     (void)lease;
                     recorded = true;
                   }),
               std::runtime_error);
  first.reset();
  sink.close();
  EXPECT_FALSE(recorded);
  EXPECT_FALSE(std::filesystem::exists(f.root / "preview.png"));
  EXPECT_EQ(sink.stats().producing_items, 0U);
  f.expect_released();
}
}  // namespace
