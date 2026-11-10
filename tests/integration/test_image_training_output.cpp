// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");

#include "../../examples/training/image_output_support.hpp"

#include <gtest/gtest.h>

namespace {
using namespace image_output_example;

struct Fixture {
  fs::path root;
  std::shared_ptr<image::ImageResourceBudget> budget;
  Fixture() {
    static std::atomic<unsigned> serial{0};
    root = fs::temp_directory_path() /
           ("liuding-training-output-" +
            std::to_string(Clock::now().time_since_epoch().count()) + "-" +
            std::to_string(serial++));
    image::PixelSpec spec;
    spec.layout = image::ImageLayout::chw;
    for (unsigned c = 0; c < 2; ++c) {
      fs::create_directories(root / "images" / std::to_string(c));
      for (unsigned i = 0; i < 3; ++i) {
        const auto pixels = torch::arange(3 * 9 * 11, torch::kFloat32)
                                .add(c * 37 + i * 11)
                                .remainder(256)
                                .to(torch::kUInt8)
                                .reshape({3, 9, 11});
        image::write_image_file(
            root / "images" / std::to_string(c) / (std::to_string(i) + ".png"),
            pixels, spec);
      }
    }
    image::ImageResourceBudgetLimits limits;
    limits.max_pipeline_inflight_batches = 1;
    limits.max_pipeline_inflight_bytes = kBatchBytes;
    budget = std::make_shared<image::ImageResourceBudget>(limits);
  }
  ~Fixture() {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  }
  Options options(const char* name, bool cuda = false,
                  bool copy = false) const {
    return {.images = root / "images",
            .output = root / name,
            .device = cuda ? forge::Device(torch::kCUDA, 0)
                           : forge::Device(torch::kCPU),
            .explicit_copy = copy,
            .batch_size = 2,
            .budget = budget};
  }
  void released() const {
    for (unsigned i = 0; i < 14; ++i)
      EXPECT_EQ(budget->usage(static_cast<image::ImageResourceKind>(i)).current,
                0)
          << i;
  }
};

TEST(ImageTrainingOutputTest, CpuTrainerCommitsBeforeBoundedEvidence) {
  Fixture fixture;
  const auto report = image_output_example::Run(fixture.options("cpu"));
  EXPECT_EQ(report["updates"], 3);
  EXPECT_EQ(report["decoded"], 6);
  EXPECT_EQ(report["steps"].size(), 3);
  EXPECT_EQ(report["files"].size(), 3);
  EXPECT_EQ(report["pressure"]["decoded_samples"], 2);
  EXPECT_EQ(report["pressure"]["acquired_batches"], 1);
  EXPECT_TRUE(report["pressure"]["input_storage_released"].get<bool>());
  for (unsigned i = 0; i < 3; ++i)
    EXPECT_EQ(report["steps"][i]["global_step"], i + 1);
  fixture.released();
}

TEST(ImageTrainingOutputTest, FailedEvidenceReleasesTrainingOwners) {
  Fixture fixture;
  std::atomic<bool> injected{false};
  auto options = fixture.options("failed");
  options.evidence_observer = [&](std::uint64_t) {
    injected = true;
    throw std::runtime_error("rejected evidence");
  };
  EXPECT_ANY_THROW(image_output_example::Run(options));
  EXPECT_TRUE(injected);
  EXPECT_TRUE(fs::is_regular_file(options.output / "step-1.png"));
  forge::experiment::SQLiteRunStore store(options.output / "run-store");
  EXPECT_TRUE(store.references("image-training").empty());
  EXPECT_TRUE(store.events("image-training").events.empty());
  fixture.released();
}

TEST(ImageTrainingOutputTest, CancelledBlockedReadRetainsOnlyCommittedUpdate) {
  Fixture fixture;
  std::stop_source stop;
  auto options = fixture.options("cancelled");
  options.stop = stop.get_token();
  options.evidence_observer = [&](std::uint64_t) { stop.request_stop(); };
  const auto report = image_output_example::Run(options);
  EXPECT_TRUE(report["cancelled"].get<bool>());
  EXPECT_EQ(report["updates"], 1);
  EXPECT_EQ(report["steps"].size(), 1);
  EXPECT_EQ(report["files"].size(), 1);
  fixture.released();
}

TEST(ImageTrainingOutputTest,
     CudaCopyAndCurrentTrainIdenticallyThroughEvidence) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is unavailable";
  Fixture fixture;
  const auto current =
      image_output_example::Run(fixture.options("current", true));
  fixture.released();
  const auto copy =
      image_output_example::Run(fixture.options("copy", true, true));
  EXPECT_EQ(current["initial_state_sha256"], copy["initial_state_sha256"]);
  EXPECT_EQ(current["steps"], copy["steps"]);
  EXPECT_EQ(current["final_state_sha256"], copy["final_state_sha256"]);
  EXPECT_EQ(current["files"], copy["files"]);
  ASSERT_EQ(copy["timelines"].size(), 3);
  for (const auto& timeline : copy["timelines"])
    EXPECT_NE(timeline["copy_stream_id"], timeline["consumer_stream_id"]);
  EXPECT_GT(copy["resources"][6]["peak"], 0);
  EXPECT_GT(copy["resources"][7]["peak"], 0);
  fixture.released();
}

TEST(ImageTrainingOutputTest, CudaCopyFailureReleasesDeviceAndSourceOwners) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is unavailable";
  Fixture fixture;
  std::atomic<bool> injected{false};
  auto options = fixture.options("cuda-failed", true, true);
  options.evidence_observer = [&](std::uint64_t) {
    injected = true;
    throw std::runtime_error("rejected evidence");
  };
  EXPECT_ANY_THROW(image_output_example::Run(options));
  EXPECT_TRUE(injected);
  EXPECT_TRUE(fs::is_regular_file(options.output / "step-1.png"));
  fixture.released();
}

TEST(ImageTrainingOutputTest, CudaCopyCancellationPreservesCommittedEvidence) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is unavailable";
  Fixture fixture;
  std::stop_source stop;
  auto options = fixture.options("cuda-cancelled", true, true);
  options.stop = stop.get_token();
  options.evidence_observer = [&](std::uint64_t) { stop.request_stop(); };
  const auto report = image_output_example::Run(options);
  EXPECT_TRUE(report["cancelled"].get<bool>());
  EXPECT_EQ(report["updates"], 1);
  EXPECT_EQ(report["files"].size(), 1);
  fixture.released();
}
}  // namespace
