// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/liuding.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace {

/** @brief Dense classifier used by the CPU Trainer throughput benchmark. */
class ThroughputModel final : public torch::nn::Module {
 public:
  /** @brief Registers the benchmark network. */
  ThroughputModel()
      : network_(register_module(
            "network", torch::nn::Sequential(
                           torch::nn::Linear(3 * 32 * 32, 512),
                           torch::nn::ReLU(), torch::nn::Linear(512, 256),
                           torch::nn::ReLU(), torch::nn::Linear(256, 10)))) {}

  /** @brief Produces logits for one image batch. */
  jyppx::liuding::Tensor forward(const jyppx::liuding::Tensor& input) {
    return network_->forward(input.view({input.size(0), -1}));
  }

 private:
  torch::nn::Sequential network_{nullptr};  ///< Dense classification stack.
};

/** @brief Measures samples processed per second for one batch size. */
double run_configuration(std::size_t batch_size, std::uint64_t epochs,
                         std::size_t samples) {
  const auto data =
      torch::randn({static_cast<std::int64_t>(samples), 3, 32, 32});
  const auto targets = torch::randn({static_cast<std::int64_t>(samples), 10});
  auto pipeline = std::make_shared<jyppx::liuding::data::DataPipeline>(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
          samples),
      jyppx::liuding::data::DataPipelineOptions{
          .pipeline_id = "data.benchmark",
          .dataset_version = "benchmark.random/v1",
          .dataset_size = samples,
          .batch_factory_id = "benchmark.tensor-slice/v1",
          .batch_size = batch_size},
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
  auto holder = std::make_shared<ThroughputModel>();
  auto model = std::make_shared<jyppx::liuding::nn::ModelAdapter>(
      jyppx::liuding::nn::adapt_tensor_module(holder));
  std::vector<jyppx::liuding::optim::OptimizerParameterBinding> bindings;
  for (const auto& item : model->state().named_parameters()) {
    bindings.push_back({.parameter_id = item.key(), .parameter = item.value()});
  }
  auto optimizer = std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
      "optimizer.benchmark",
      std::vector<jyppx::liuding::optim::SgdParameterGroup>{
          {.group_id = "main",
           .parameters = std::move(bindings),
           .options = {.learning_rate = 1e-3}}});
  const auto mapper = [](const jyppx::liuding::data::BatchEnvelope& batch) {
    return jyppx::liuding::nn::Invocation{
        .positional = {batch.value.toTuple()->elements().at(0)}};
  };
  const auto target = [](const jyppx::liuding::data::BatchEnvelope& batch)
      -> std::optional<c10::IValue> {
    return batch.value.toTuple()->elements().at(1);
  };
  const auto step = [](const jyppx::liuding::training::StepInput& input) {
    const auto prediction =
        input.model.invoke(input.invocation).value.toTensor();
    return jyppx::liuding::training::StepOutput{
        .loss = (prediction - input.target->toTensor()).pow(2).mean(),
        .sample_count = input.batch.sample_ids.size()};
  };
  jyppx::liuding::training::Trainer trainer({.trainer_id = "trainer.benchmark",
                                             .run_id = "run.benchmark",
                                             .trace_id = "trace.benchmark",
                                             .max_epochs = epochs});
  trainer.prepare(
      {.model = model,
       .optimizers =
           {{.optimizer = optimizer,
             .checkpoint = std::make_shared<
                 jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
                 optimizer)}},
       .training_data = pipeline,
       .training_batch_mapper = mapper,
       .training_target_mapper = target,
       .training_step = step,
       .strategy =
           std::make_shared<jyppx::liuding::training::SingleDeviceStrategy>(),
       .model_checkpoint =
           std::make_shared<jyppx::liuding::nn::ModelCheckpointParticipant>(
               *model),
       .rng_checkpoint =
           std::make_shared<jyppx::liuding::core::RngCheckpointParticipant>(
               jyppx::liuding::core::RngCheckpointParticipantOptions{
                   .include_cuda = false})});
  const auto start = std::chrono::steady_clock::now();
  static_cast<void>(trainer.fit());
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
  trainer.close();
  return static_cast<double>(samples * epochs) / elapsed.count();
}

}  // namespace

int main() {
  constexpr std::size_t kSamples = 50'000;
  constexpr std::uint64_t kEpochs = 1;
  std::cout << "batch_size,samples,epochs,samples_per_second\n";
  std::cout << std::fixed << std::setprecision(2);
  for (const auto batch_size : {16U, 32U, 64U, 128U}) {
    const auto throughput = run_configuration(batch_size, kEpochs, kSamples);
    std::cout << batch_size << ',' << kSamples << ',' << kEpochs << ','
              << throughput << '\n';
  }
}
