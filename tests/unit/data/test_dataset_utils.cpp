// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/**
 * @file test_dataset_utils.cpp
 * @brief Tests dataset views, splitting, and current pipeline integration.
 */

#include <jyppx/data/concat_dataset.hpp>
#include <jyppx/data/data_pipeline.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/data/subset.hpp>
#include <jyppx/data/tensor_dataset.hpp>
#include <jyppx/foundation/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <span>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace jyppx;

/** @brief Builds a tuple-producing pipeline for an Example dataset view. */
std::unique_ptr<data::DataPipeline> make_example_pipeline(
    std::shared_ptr<const data::Dataset<data::Example>> dataset,
    std::string pipeline_id, std::size_t batch_size) {
  const auto dataset_size = dataset->size();
  auto sampler =
      std::make_shared<data::samplers::SequentialSampler>(dataset_size);
  return std::make_unique<data::DataPipeline>(
      std::move(sampler),
      data::DataPipelineOptions{
          .pipeline_id = std::move(pipeline_id),
          .dataset_version = "test.dataset_utils/v1",
          .dataset_size = dataset_size,
          .schema = data::BatchSchema::any("test.dataset_utils.batch", 1),
          .batch_factory_id = "test.dataset_utils.example_collate/v1",
          .batch_size = batch_size},
      [dataset = std::move(dataset)](std::span<const std::size_t> indices,
                                     const data::DataWorkerContext&,
                                     std::stop_token) {
        std::vector<Tensor> inputs;
        std::vector<Tensor> targets;
        inputs.reserve(indices.size());
        targets.reserve(indices.size());
        for (const auto index : indices) {
          const auto sample = dataset->get(index);
          inputs.push_back(sample.data);
          targets.push_back(sample.target);
        }
        return c10::ivalue::Tuple::create(
            {torch::stack(inputs), torch::stack(targets)});
      });
}

class DatasetUtilsTest : public ::testing::Test {
 protected:
  /** @brief Creates ten deterministic feature and target rows. */
  void SetUp() override {
    inputs_ =
        torch::arange(0, 20, torch::TensorOptions().dtype(torch::kFloat32))
            .view({10, 2});
    targets_ =
        torch::arange(0, 10, torch::TensorOptions().dtype(torch::kFloat32));
    dataset_ = std::make_shared<data::TensorDataset>(inputs_, targets_);
  }

  Tensor inputs_;   ///< Two-feature input rows shared by every test.
  Tensor targets_;  ///< Scalar target row paired with each input.
  std::shared_ptr<data::TensorDataset>
      dataset_;  ///< Dataset under test and source for derived views.
};

TEST_F(DatasetUtilsTest, SubsetExposesSelectedSamples) {
  const data::Subset<data::Example> subset(dataset_, {3, 0, 7});

  ASSERT_EQ(subset.size(), 3);
  EXPECT_TRUE(torch::equal(subset.get(0).data, inputs_[3]));
  EXPECT_TRUE(torch::equal(subset.get(1).data, inputs_[0]));
  EXPECT_TRUE(torch::equal(subset.get(2).data, inputs_[7]));
  EXPECT_TRUE(torch::equal(subset.get(2).target, targets_[7]));
}

TEST_F(DatasetUtilsTest, DefaultBatchLookupPreservesOrderAndDuplicates) {
  const auto samples = dataset_->get_batch({2, 0, 2});

  ASSERT_EQ(samples.size(), 3U);
  EXPECT_TRUE(torch::equal(samples[0].data, inputs_[2]));
  EXPECT_TRUE(torch::equal(samples[1].data, inputs_[0]));
  EXPECT_TRUE(torch::equal(samples[2].target, targets_[2]));
}

TEST_F(DatasetUtilsTest, SubsetValidatesArguments) {
  EXPECT_THROW(data::Subset<data::Example>(nullptr, {0}),
               jyppx::InvalidArgument);
  EXPECT_THROW(data::Subset<data::Example>(dataset_, {10}),
               jyppx::InvalidArgument);
  const data::Subset<data::Example> subset(dataset_, {0});
  EXPECT_THROW(subset.get(1), jyppx::InvalidArgument);
}

TEST_F(DatasetUtilsTest, SubsetFeedsDataPipeline) {
  auto subset = std::make_shared<data::Subset<data::Example>>(
      dataset_, std::vector<size_t>{1, 3, 5, 7, 9});
  auto pipeline = make_example_pipeline(subset, "data.dataset_utils.subset", 2);
  pipeline->set_epoch(0);

  std::vector<float> first_column;
  std::size_t batch_count = 0;
  while (const auto batch = pipeline->next()) {
    const auto inputs = batch->value.toTuple()->elements().at(0).toTensor();
    for (std::int64_t row = 0; row < inputs.size(0); ++row) {
      first_column.push_back(inputs[row][0].item<float>());
    }
    ++batch_count;
  }
  EXPECT_EQ(batch_count, 3U);
  // Odd samples only, in subset order.
  const std::vector<float> expected{2.f, 6.f, 10.f, 14.f, 18.f};
  EXPECT_EQ(first_column, expected);
}

TEST_F(DatasetUtilsTest, ConcatDatasetSpansDatasets) {
  auto first = std::make_shared<data::Subset<data::Example>>(
      dataset_, std::vector<size_t>{0, 1, 2});
  auto second = std::make_shared<data::Subset<data::Example>>(
      dataset_, std::vector<size_t>{8, 9});
  const data::ConcatDataset<data::Example> concatenated({first, second});

  EXPECT_EQ(concatenated.size(), 5);
  EXPECT_EQ(concatenated.cumulative_sizes(), (std::vector<size_t>{3, 5}));
  EXPECT_TRUE(torch::equal(concatenated.get(0).data, inputs_[0]));
  EXPECT_TRUE(torch::equal(concatenated.get(3).data, inputs_[8]));
  EXPECT_TRUE(torch::equal(concatenated.get(4).target, targets_[9]));
}

TEST_F(DatasetUtilsTest, ConcatDatasetValidatesArguments) {
  EXPECT_THROW(data::ConcatDataset<data::Example>({}), jyppx::InvalidArgument);

  std::shared_ptr<data::Dataset<data::Example>> null_dataset;
  auto valid = std::make_shared<data::Subset<data::Example>>(
      dataset_, std::vector<size_t>{0});
  EXPECT_THROW(data::ConcatDataset<data::Example>({valid, null_dataset}),
               jyppx::InvalidArgument);

  const data::ConcatDataset<data::Example> concatenated({valid});
  EXPECT_THROW(concatenated.get(1), jyppx::InvalidArgument);
}

TEST_F(DatasetUtilsTest, RandomSplitPartitionsEverySample) {
  const auto subsets =
      data::random_split(dataset_, std::vector<size_t>{6, 4}, /*seed=*/123);

  ASSERT_EQ(subsets.size(), 2);
  EXPECT_EQ(subsets[0]->size(), 6);
  EXPECT_EQ(subsets[1]->size(), 4);

  std::vector<size_t> seen;
  for (const auto& subset : subsets) {
    for (size_t i = 0; i < subset->size(); ++i) {
      seen.push_back(subset->indices()[i]);
    }
  }
  std::sort(seen.begin(), seen.end());
  for (size_t i = 0; i < seen.size(); ++i) {
    EXPECT_EQ(seen[i], i);  // partition covers each sample exactly once
  }
}

TEST_F(DatasetUtilsTest, RandomSplitIsDeterministicWithSeed) {
  const auto first =
      data::random_split(dataset_, std::vector<size_t>{5, 5}, /*seed=*/7);
  const auto second =
      data::random_split(dataset_, std::vector<size_t>{5, 5}, /*seed=*/7);

  for (size_t s = 0; s < first.size(); ++s) {
    EXPECT_EQ(first[s]->indices(), second[s]->indices());
    EXPECT_TRUE(torch::equal(first[s]->get(0).data, second[s]->get(0).data));
  }
}

TEST_F(DatasetUtilsTest, RandomSplitValidatesLengths) {
  EXPECT_THROW(data::random_split(dataset_, std::vector<size_t>{5, 6}, 1),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      data::random_split(std::shared_ptr<data::Dataset<data::Example>>(),
                         std::vector<size_t>{10}, 1),
      jyppx::InvalidArgument);
}

TEST_F(DatasetUtilsTest, RandomSplitTrainValPipelinesWork) {
  // The canonical train/validation split flows through current pipelines.
  const auto subsets =
      data::random_split(dataset_, std::vector<size_t>{8, 2}, /*seed=*/42);
  auto train = make_example_pipeline(subsets[0], "data.dataset_utils.train", 4);
  auto validation =
      make_example_pipeline(subsets[1], "data.dataset_utils.validation", 2);
  train->set_epoch(0);
  validation->set_epoch(0);

  std::size_t train_batches = 0;
  while (const auto batch = train->next()) {
    EXPECT_EQ(batch->value.toTuple()->elements().at(0).toTensor().size(0), 4);
    ++train_batches;
  }
  std::size_t validation_batches = 0;
  while (const auto batch = validation->next()) {
    EXPECT_EQ(batch->value.toTuple()->elements().at(0).toTensor().size(0), 2);
    ++validation_batches;
  }
  EXPECT_EQ(train_batches, 2U);
  EXPECT_EQ(validation_batches, 1U);
}

}  // namespace
