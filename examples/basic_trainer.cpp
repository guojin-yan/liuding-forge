// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/liuding.hpp>

#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace {

using jyppx::liuding::Tensor;

/** @brief Creates a linear regression batch stream over in-memory tensors. */
std::shared_ptr<jyppx::liuding::data::DataPipeline> make_pipeline(
    const Tensor& inputs, const Tensor& targets, std::string pipeline_id) {
  auto sampler = std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
      static_cast<std::size_t>(inputs.size(0)));
  return std::make_shared<jyppx::liuding::data::DataPipeline>(
      std::move(sampler),
      jyppx::liuding::data::DataPipelineOptions{
          .pipeline_id = std::move(pipeline_id),
          .dataset_version = "example.linear/v1",
          .dataset_size = static_cast<std::size_t>(inputs.size(0)),
          .schema = jyppx::liuding::data::BatchSchema::any("example.linear.batch", 1),
          .batch_factory_id = "example.linear.tensor-slice/v1",
          .batch_size = 4},
      [inputs, targets](std::span<const std::size_t> indices,
                        const jyppx::liuding::data::DataWorkerContext&,
                        std::stop_token) {
        std::vector<std::int64_t> rows;
        rows.reserve(indices.size());
        for (const auto index : indices) {
          rows.push_back(static_cast<std::int64_t>(index));
        }
        const auto row_tensor = torch::tensor(rows, torch::kLong);
        return c10::ivalue::Tuple::create(
            {inputs.index_select(0, row_tensor),
             targets.index_select(0, row_tensor)});
      });
}

/** @brief Builds the new type-erased model/optimizer training domain. */
jyppx::liuding::training::FitRequest make_request(
    const std::shared_ptr<jyppx::liuding::nn::ModelAdapter>& model,
    const std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter>& optimizer,
    const std::shared_ptr<jyppx::liuding::data::BatchPipeline>& training_data,
    const std::shared_ptr<jyppx::liuding::data::BatchPipeline>& validation_data) {
  auto strategy = std::make_shared<jyppx::liuding::training::SingleDeviceStrategy>();
  auto model_checkpoint =
      std::make_shared<jyppx::liuding::nn::ModelCheckpointParticipant>(*model);
  auto optimizer_checkpoint =
      std::make_shared<jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
          optimizer);
  auto rng_checkpoint = std::make_shared<jyppx::liuding::core::RngCheckpointParticipant>(
      jyppx::liuding::core::RngCheckpointParticipantOptions{.include_cuda = false});
  const auto mapper = [](const jyppx::liuding::data::BatchEnvelope& batch) {
    const auto& values = batch.value.toTuple()->elements();
    return jyppx::liuding::nn::Invocation{.positional = {values.at(0)}};
  };
  const auto targets = [](const jyppx::liuding::data::BatchEnvelope& batch)
      -> std::optional<c10::IValue> {
    return batch.value.toTuple()->elements().at(1);
  };
  const auto step = [](const jyppx::liuding::training::StepInput& input) {
    const auto prediction =
        input.model.invoke(input.invocation).value.toTensor();
    const auto target = input.target->toTensor();
    return jyppx::liuding::training::StepOutput{
        .loss = (prediction - target).pow(2).mean(),
        .sample_count = input.batch.sample_ids.size()};
  };
  return {.model = model,
          .optimizers = {{.optimizer = optimizer,
                          .checkpoint = std::move(optimizer_checkpoint)}},
          .training_data = training_data,
          .training_batch_mapper = mapper,
          .training_target_mapper = targets,
          .training_step = step,
          .validation_data = validation_data,
          .validation_batch_mapper = mapper,
          .validation_target_mapper = targets,
          .validation_step = step,
          .strategy = std::move(strategy),
          .model_checkpoint = std::move(model_checkpoint),
          .rng_checkpoint = std::move(rng_checkpoint)};
}

}  // namespace

int main() {
  try {
    torch::manual_seed(7);
    const auto inputs =
        torch::arange(0, 8, torch::TensorOptions().dtype(torch::kFloat32))
            .view({8, 1});
    const auto targets = inputs * 2.0 + 1.0;
    auto holder = torch::nn::Linear(torch::nn::LinearOptions(1, 1));
    auto model = std::make_shared<jyppx::liuding::nn::ModelAdapter>(
        jyppx::liuding::nn::adapt_tensor_holder(holder));
    std::vector<jyppx::liuding::optim::OptimizerParameterBinding> bindings;
    for (const auto& item : model->state().named_parameters()) {
      bindings.push_back(
          {.parameter_id = item.key(), .parameter = item.value()});
    }
    auto optimizer = std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
        "optimizer.example", std::vector<jyppx::liuding::optim::SgdParameterGroup>{
                                 {.group_id = "main",
                                  .parameters = std::move(bindings),
                                  .options = {.learning_rate = 0.01}}});
    auto train = make_pipeline(inputs, targets, "data.example.train");
    auto validation = make_pipeline(inputs, targets, "data.example.validation");
    jyppx::liuding::training::Trainer trainer({.trainer_id = "trainer.example",
                                      .run_id = "run.example",
                                      .trace_id = "trace.example",
                                      .max_epochs = 2});
    trainer.prepare(make_request(model, optimizer, train, validation));
    const auto result = trainer.fit();
    std::cout << "status=" << static_cast<int>(result.status)
              << " global_step=" << result.global_step << '\n';
    trainer.close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
