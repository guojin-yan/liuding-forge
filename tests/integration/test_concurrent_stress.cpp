// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/** @file test_concurrent_stress.cpp
 * @brief Stress-tests current DataPipeline worker and epoch isolation.
 */

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/dataset.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

namespace {

using jyppx::liuding::Tensor;

/** @brief Deterministic row source that records every materialization. */
class CountingDataset final : public jyppx::liuding::data::Dataset<Tensor> {
 public:
  /** @brief Creates a source exposing indices in `[0, count)`. */
  explicit CountingDataset(std::size_t count) : count_(count) {}

  /** @brief Returns one scalar row equal to its dataset index. */
  Tensor get(std::size_t index) const override {
    calls_.fetch_add(1, std::memory_order_relaxed);
    return torch::tensor({static_cast<float>(index)});
  }

  /** @brief Returns the immutable dataset cardinality. */
  std::size_t size() const override { return count_; }

  /** @brief Returns the number of rows materialized so far. */
  std::size_t calls() const { return calls_.load(std::memory_order_relaxed); }

 private:
  std::size_t count_;  ///< Immutable number of addressable rows.
  mutable std::atomic<std::size_t> calls_{0};  ///< Materialized row count.
};

/** @brief Creates a worker-backed map pipeline over a counting dataset. */
std::unique_ptr<jyppx::liuding::data::DataPipeline> make_pipeline(
    const std::shared_ptr<CountingDataset>& dataset, std::string pipeline_id,
    std::size_t batch_size, std::size_t workers, bool shuffle = false) {
  std::shared_ptr<jyppx::liuding::data::samplers::Sampler> sampler;
  if (shuffle) {
    sampler = std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(
        dataset->size(), /*seed=*/17);
  } else {
    sampler = std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
        dataset->size());
  }
  return std::make_unique<jyppx::liuding::data::DataPipeline>(
      std::move(sampler),
      jyppx::liuding::data::DataPipelineOptions{
          .pipeline_id = std::move(pipeline_id),
          .dataset_version = "test.counting_dataset/v1",
          .dataset_size = dataset->size(),
          .schema = jyppx::liuding::data::BatchSchema::any("test.concurrent.batch", 1),
          .batch_factory_id = "test.counting_dataset.stack/v1",
          .batch_size = batch_size,
          .num_workers = workers,
          .prefetch_batches = std::max<std::size_t>(workers, 2U),
          .persistent_workers = workers > 0,
          .worker_seed = 17},
      [dataset](std::span<const std::size_t> indices,
                const jyppx::liuding::data::DataWorkerContext&, std::stop_token) {
        std::vector<Tensor> rows;
        rows.reserve(indices.size());
        for (const auto index : indices) {
          rows.push_back(dataset->get(index));
        }
        return c10::IValue(torch::stack(rows));
      });
}

TEST(ConcurrentStressTest, SixtyFourWorkersPreserveOrderAndCompleteness) {
  auto dataset = std::make_shared<CountingDataset>(512);
  auto pipeline = make_pipeline(dataset, "data.concurrent.sixty_four", 8, 64);
  pipeline->set_epoch(0);

  std::size_t samples = 0;
  float expected = 0.0F;
  while (const auto batch = pipeline->next()) {
    const auto rows = batch->value.toTensor();
    ASSERT_EQ(rows.size(0), 8);
    for (std::int64_t row = 0; row < rows.size(0); ++row) {
      EXPECT_FLOAT_EQ(rows[row].item<float>(), expected++);
    }
    samples += batch->sample_ids.size();
  }
  EXPECT_EQ(samples, 512U);
  EXPECT_EQ(dataset->calls(), 512U);
}

TEST(ConcurrentStressTest, MixedPipelineInstancesCanRunConcurrently) {
  const auto run_pipeline = [](float offset, std::string pipeline_id) {
    auto dataset = std::make_shared<CountingDataset>(128);
    auto pipeline =
        make_pipeline(dataset, std::move(pipeline_id), 4, /*workers=*/8);
    pipeline->set_epoch(0);
    float checksum = 0.0F;
    while (const auto batch = pipeline->next()) {
      checksum += batch->value.toTensor().sum().item<float>() + offset;
    }
    return std::pair<float, std::size_t>{checksum, dataset->calls()};
  };

  auto first = std::async(std::launch::async, run_pipeline, 1.0F,
                          "data.concurrent.first");
  auto second = std::async(std::launch::async, run_pipeline, 2.0F,
                           "data.concurrent.second");
  const auto first_result = first.get();
  const auto second_result = second.get();
  EXPECT_EQ(first_result.second, 128U);
  EXPECT_EQ(second_result.second, 128U);
  EXPECT_GT(second_result.first, first_result.first);
}

TEST(ConcurrentStressTest, PartialEpochReplacementRemainsIsolated) {
  auto dataset = std::make_shared<CountingDataset>(96);
  auto pipeline = make_pipeline(dataset, "data.concurrent.epochs", 6,
                                /*workers=*/16, /*shuffle=*/true);

  for (std::uint64_t epoch = 0; epoch < 12; ++epoch) {
    pipeline->set_epoch(epoch);
    for (int batch = 0; batch < 2; ++batch) {
      const auto value = pipeline->next();
      ASSERT_TRUE(value.has_value());
      ASSERT_EQ(value->sample_ids.size(), 6U);
      ASSERT_EQ(value->epoch, epoch);
    }
  }
  pipeline->close();
  EXPECT_GE(dataset->calls(), 12U * 2U * 6U);
}

}  // namespace
