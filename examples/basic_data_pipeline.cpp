// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/** @file basic_data_pipeline.cpp
 * @brief Demonstrates the current explicit-epoch map-style data pipeline.
 */

#include <jyppx/liuding/liuding.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stop_token>
#include <utility>
#include <vector>

int main() {
  const auto data =
      torch::arange(0, 10, torch::TensorOptions().dtype(torch::kFloat32))
          .view({10, 1});
  const auto targets =
      torch::arange(0, 10, torch::TensorOptions().dtype(torch::kFloat32));
  auto sampler =
      std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(
          10, /*seed=*/7);
  jyppx::liuding::data::DataPipeline pipeline(
      std::move(sampler),
      jyppx::liuding::data::DataPipelineOptions{
          .pipeline_id = "example.basic_data_pipeline",
          .dataset_version = "example.basic_data/v1",
          .dataset_size = 10,
          .schema =
              jyppx::liuding::data::BatchSchema::any("example.basic.batch", 1),
          .batch_factory_id = "example.basic.tensor-pair/v1",
          .batch_size = 4},
      [data, targets](std::span<const std::size_t> indices,
                      const jyppx::liuding::data::DataWorkerContext&,
                      std::stop_token) {
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
  while (const auto batch = pipeline.next()) {
    const auto values = batch->value.toTuple()->elements();
    std::cout << "batch: " << values.at(0).toTensor().size(0)
              << " samples, target mean="
              << values.at(1).toTensor().mean().item<double>() << '\n';
  }
  pipeline.close();
  return 0;
}
