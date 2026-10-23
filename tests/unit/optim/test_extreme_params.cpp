// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/**
 * @file test_extreme_params.cpp
 * @brief Tests optimizer and pipeline behavior at extreme parameter sizes.
 */

#include <jyppx/liuding/core/tensor_utils.hpp>
#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/optim/adam_adapter.hpp>
#include <jyppx/liuding/optim/adamw_adapter.hpp>

#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <stop_token>
#include <vector>

namespace {

using jyppx::liuding::Tensor;

TEST(ExtremeOptimizerParamsTest, AdamVariantsRemainFiniteAtLargeLearningRate) {
  for (const bool decoupled : {false, true}) {
    auto parameter = torch::ones(
        {128},
        torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    parameter.mutable_grad() = torch::full_like(parameter, 0.5);
    if (decoupled) {
      jyppx::liuding::optim::AdamWOptimizerAdapter optimizer(
          "optimizer.adamw",
          {{.group_id = "main",
            .parameters = {{.parameter_id = "parameter",
                            .parameter = parameter}},
            .options = {.learning_rate = 1e2, .weight_decay = 1e-1}}});
      optimizer.step();
    } else {
      jyppx::liuding::optim::AdamOptimizerAdapter optimizer(
          "optimizer.adam",
          {{.group_id = "main",
            .parameters = {{.parameter_id = "parameter",
                            .parameter = parameter}},
            .options = {.learning_rate = 1e2, .weight_decay = 1e-1}}});
      optimizer.step();
    }
    EXPECT_TRUE(jyppx::liuding::is_finite(parameter));
  }
}

TEST(ExtremeOptimizerParamsTest, VeryLargeBatchIsBatchedWithoutOverflow) {
  const auto data =
      torch::arange(0, 8192, torch::TensorOptions().dtype(torch::kFloat32))
          .view({8192, 1});
  const auto targets = data * 2.0F;
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(8192),
      {.pipeline_id = "data.extreme.large_batch",
       .dataset_version = "test.extreme.large_batch/v1",
       .dataset_size = 8192,
       .schema = jyppx::liuding::data::BatchSchema::any("test.extreme.batch", 1),
       .batch_factory_id = "test.extreme.tensor_pair/v1",
       .batch_size = 8192},
      [data, targets](std::span<const std::size_t> indices,
                      const jyppx::liuding::data::DataWorkerContext&, std::stop_token) {
        std::vector<std::int64_t> rows;
        rows.reserve(indices.size());
        for (const auto index : indices) {
          rows.push_back(static_cast<std::int64_t>(index));
        }
        const auto row_tensor = torch::tensor(rows, torch::kLong);
        return c10::ivalue::Tuple::create(
            {data.index_select(0, row_tensor),
             targets.index_select(0, row_tensor)});
      });
  pipeline.set_epoch(0);

  const auto batch = pipeline.next();
  ASSERT_TRUE(batch.has_value());
  const auto values = batch->value.toTuple()->elements();
  ASSERT_EQ(values.at(0).toTensor().size(0), 8192);
  EXPECT_TRUE(jyppx::liuding::is_finite(values.at(0).toTensor()));
  EXPECT_TRUE(jyppx::liuding::is_finite(values.at(1).toTensor()));
  EXPECT_FALSE(pipeline.next().has_value());
}

}  // namespace
