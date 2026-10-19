// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/** @file test_large_dataset.cpp
 * @brief Exercises current map pipeline geometry with a logical 10 GiB source.
 */

#include <jyppx/core/tensor_utils.hpp>
#include <jyppx/data/data_pipeline.hpp>
#include <jyppx/data/dataset.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

namespace {

using jyppx::Tensor;

/** @brief Represents more than 10 GiB without allocating backing storage. */
class VirtualLargeDataset final : public jyppx::data::Dataset<Tensor> {
 public:
  static constexpr std::size_t kRows = 2'000'000;  ///< Logical row count.
  static constexpr std::size_t kElementsPerRow =
      1536;  ///< Float elements in one 6 KiB row.
  static constexpr std::uint64_t kLogicalBytes =
      kRows * kElementsPerRow * sizeof(float);  ///< Total logical capacity.

  static_assert(kLogicalBytes > 10ULL * 1024ULL * 1024ULL * 1024ULL);

  /** @brief Materializes one 6 KiB logical row and records the access. */
  Tensor get(std::size_t index) const override {
    calls_.fetch_add(1, std::memory_order_relaxed);
    if (index >= kRows) {
      throw std::out_of_range("virtual dataset index out of range");
    }
    return torch::full({static_cast<std::int64_t>(kElementsPerRow)},
                       static_cast<double>(index),
                       torch::TensorOptions().dtype(torch::kFloat32));
  }

  /** @brief Returns the fixed logical row count. */
  std::size_t size() const override { return kRows; }

  /** @brief Returns the number of rows materialized by batch factories. */
  std::size_t calls() const { return calls_.load(std::memory_order_relaxed); }

 private:
  mutable std::atomic<std::size_t> calls_{0};  ///< Materialized row count.
};

/** @brief Creates a current map pipeline over the logical dataset. */
std::unique_ptr<jyppx::data::DataPipeline> make_pipeline(
    const std::shared_ptr<VirtualLargeDataset>& dataset, std::size_t workers,
    std::size_t prefetch_batches) {
  return std::make_unique<jyppx::data::DataPipeline>(
      std::make_shared<jyppx::data::samplers::SequentialSampler>(
          dataset->size()),
      jyppx::data::DataPipelineOptions{
          .pipeline_id = "data.integration.large",
          .dataset_version = "test.virtual_10gib/v1",
          .dataset_size = dataset->size(),
          .schema =
              jyppx::data::BatchSchema::any("test.large_dataset.batch", 1),
          .batch_factory_id = "test.large_dataset.stack/v1",
          .batch_size = workers == 0 ? 1024U : 256U,
          .num_workers = workers,
          .prefetch_batches = prefetch_batches,
          .persistent_workers = workers > 0},
      [dataset](std::span<const std::size_t> indices,
                const jyppx::data::DataWorkerContext&, std::stop_token) {
        std::vector<Tensor> rows;
        rows.reserve(indices.size());
        for (const auto index : indices) {
          rows.push_back(dataset->get(index));
        }
        return c10::IValue(torch::stack(rows));
      });
}

TEST(LargeDatasetTest, LogicalTenGiBDatasetHasStableBatchGeometry) {
  auto dataset = std::make_shared<VirtualLargeDataset>();
  auto pipeline = make_pipeline(dataset, /*workers=*/0, /*prefetch_batches=*/2);
  pipeline->set_epoch(0);

  const auto state = pipeline->state();
  EXPECT_EQ(state.dataset_size, VirtualLargeDataset::kRows);
  EXPECT_EQ(state.epoch_indices.size(), VirtualLargeDataset::kRows);
  EXPECT_EQ((state.dataset_size + state.batch_size - 1) / state.batch_size,
            (VirtualLargeDataset::kRows + 1023) / 1024);

  const auto first = pipeline->next();
  ASSERT_TRUE(first.has_value());
  const auto first_rows = first->value.toTensor();
  EXPECT_EQ(first_rows.sizes(), torch::IntArrayRef({1024, 1536}));
  EXPECT_EQ(first_rows[0][0].item<int>(), 0);
  EXPECT_EQ(first_rows[1023][0].item<int>(), 1023);

  const auto second = pipeline->next();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->value.toTensor()[0][0].item<int>(), 1024);
  EXPECT_TRUE(jyppx::is_finite(second->value.toTensor()));
  EXPECT_EQ(dataset->calls(), 2048U);
}

TEST(LargeDatasetTest, RepeatedPartialEpochsDoNotRetainBatches) {
  auto dataset = std::make_shared<VirtualLargeDataset>();
  auto pipeline = make_pipeline(dataset, /*workers=*/4, /*prefetch_batches=*/8);

  for (std::uint64_t epoch = 0; epoch < 8; ++epoch) {
    pipeline->set_epoch(epoch);
    for (int batch = 0; batch < 3; ++batch) {
      const auto value = pipeline->next();
      ASSERT_TRUE(value.has_value());
      ASSERT_EQ(value->value.toTensor().size(0), 256);
      ASSERT_TRUE(jyppx::is_finite(value->value.toTensor()));
    }
  }
  pipeline->close();

  // Prefetch may finish extra batches before the next explicit epoch. Every
  // committed prefix must nevertheless have been materialized completely.
  EXPECT_GE(dataset->calls(), 8U * 3U * 256U);
}

}  // namespace
