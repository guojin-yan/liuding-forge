// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "mnist_data.hpp"

#include <jyppx/jyppx.hpp>

#include <torch/nn/init.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using jyppx::Tensor;

/** @brief Small convolutional classifier used by the MNIST example. */
class MnistCnn final : public torch::nn::Module {
 public:
  /** @brief Registers convolutional and dense classifier layers. */
  MnistCnn()
      : conv1_(register_module(
            "conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 8, 3).padding(1)))),
        conv2_(register_module(
            "conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 3).padding(1)))),
        fc1_(register_module("fc1", torch::nn::Linear(16 * 7 * 7, 64))),
        fc2_(register_module("fc2", torch::nn::Linear(64, 10))) {
    torch::nn::init::kaiming_uniform_(conv1_->weight, 0.0, torch::kFanIn,
                                      torch::kLeakyReLU);
    torch::nn::init::kaiming_uniform_(conv2_->weight, 0.0, torch::kFanIn,
                                      torch::kLeakyReLU);
  }

  /** @brief Maps flattened MNIST images to ten class logits. */
  Tensor forward(const Tensor& input) {
    auto value = input.view({-1, 1, 28, 28});
    value = torch::max_pool2d(torch::relu(conv1_->forward(value)), 2);
    value = torch::max_pool2d(torch::relu(conv2_->forward(value)), 2);
    value = torch::relu(fc1_->forward(value.view({value.size(0), -1})));
    return fc2_->forward(value);
  }

 private:
  torch::nn::Conv2d conv1_;  ///< First image feature projection.
  torch::nn::Conv2d conv2_;  ///< Second image feature projection.
  torch::nn::Linear fc1_;    ///< Hidden classifier projection.
  torch::nn::Linear fc2_;    ///< Ten-class logit projection.
};

/** @brief Downloads one MNIST archive when it is not already available. */
void download_if_missing(const std::string& path, const std::string& url) {
  if (std::filesystem::exists(path)) {
    return;
  }
  std::cout << "downloading " << url << '\n';
  const auto command =
      "curl -L --retry 3 --fail -o \"" + path + "\" \"" + url + "\"";
  if (std::system(command.c_str()) != 0 || !std::filesystem::exists(path)) {
    throw std::runtime_error("failed to download " + url);
  }
}

/** @brief Creates one checkpointable tensor-backed MNIST pipeline. */
std::shared_ptr<jyppx::data::DataPipeline> make_pipeline(
    const Tensor& images, const Tensor& labels, std::size_t batch_size,
    std::string pipeline_id, bool shuffle) {
  const auto size = static_cast<std::size_t>(images.size(0));
  std::shared_ptr<jyppx::data::samplers::Sampler> sampler;
  if (shuffle) {
    sampler =
        std::make_shared<jyppx::data::samplers::RandomSampler>(size, 2026);
  } else {
    sampler = std::make_shared<jyppx::data::samplers::SequentialSampler>(size);
  }
  return std::make_shared<jyppx::data::DataPipeline>(
      std::move(sampler),
      jyppx::data::DataPipelineOptions{
          .pipeline_id = std::move(pipeline_id),
          .dataset_version = "mnist.idx/v1",
          .dataset_size = size,
          .schema_id = "mnist.batch/v1",
          .batch_factory_id = "mnist.tensor-slice/v1",
          .batch_size = batch_size,
          .num_workers = shuffle ? 2U : 0U,
          .worker_seed = 2026},
      [images, labels](std::span<const std::size_t> indices,
                       const jyppx::data::DataWorkerContext&, std::stop_token) {
        std::vector<std::int64_t> rows;
        rows.reserve(indices.size());
        for (const auto index : indices) {
          rows.push_back(static_cast<std::int64_t>(index));
        }
        const auto row_tensor = torch::tensor(rows, torch::kLong);
        return c10::ivalue::Tuple::create(
            {images.index_select(0, row_tensor).view({-1, 784}),
             labels.index_select(0, row_tensor)});
      });
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto data_dir =
        argc > 1
            ? std::string(argv[1])
            : (std::filesystem::temp_directory_path() / "jyppx_mnist").string();
    const auto epochs =
        static_cast<std::uint64_t>(argc > 2 ? std::stoull(argv[2]) : 3U);
    std::filesystem::create_directories(data_dir);
    const std::string base = "https://ossci-datasets.s3.amazonaws.com/mnist/";
    for (const auto* file :
         {"train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
          "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"}) {
      download_if_missing(data_dir + "/" + file, base + file);
    }

    const auto train_raw =
        mnist_data::load_mnist_idx(data_dir + "/train-images-idx3-ubyte.gz",
                                   data_dir + "/train-labels-idx1-ubyte.gz");
    const auto test_raw =
        mnist_data::load_mnist_idx(data_dir + "/t10k-images-idx3-ubyte.gz",
                                   data_dir + "/t10k-labels-idx1-ubyte.gz");
    const auto device = jyppx::get_default_device();
    std::cout << "training on: " << device << '\n';

    auto holder = std::make_shared<MnistCnn>();
    auto model = std::make_shared<jyppx::nn::ModelAdapter>(
        jyppx::nn::adapt_tensor_module(holder));
    std::vector<jyppx::optim::OptimizerParameterBinding> bindings;
    for (const auto& item : model->state().named_parameters()) {
      bindings.push_back(
          {.parameter_id = item.key(), .parameter = item.value()});
    }
    auto optimizer = std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
        "optimizer.mnist",
        std::vector<jyppx::optim::SgdParameterGroup>{
            {.group_id = "main",
             .parameters = std::move(bindings),
             .options = {.learning_rate = 0.01, .momentum = 0.9}}});
    auto train = make_pipeline(train_raw.images, train_raw.labels, 64,
                               "data.mnist.train", true);
    auto validation = make_pipeline(test_raw.images, test_raw.labels, 256,
                                    "data.mnist.validation", false);
    const auto mapper = [device](const jyppx::data::BatchEnvelope& batch) {
      return jyppx::nn::Invocation{
          .positional = {
              batch.value.toTuple()->elements().at(0).toTensor().to(device)}};
    };
    const auto target = [device](const jyppx::data::BatchEnvelope& batch)
        -> std::optional<c10::IValue> {
      return batch.value.toTuple()->elements().at(1).toTensor().to(device);
    };
    const auto step = [](const jyppx::training::StepInput& input) {
      const auto prediction =
          input.model.invoke(input.invocation).value.toTensor();
      const auto expected =
          input.target->toTensor().to(torch::kLong).view({-1});
      const auto accuracy = prediction.argmax(1)
                                .eq(expected)
                                .to(torch::kFloat32)
                                .mean()
                                .item<double>();
      return jyppx::training::StepOutput{
          .loss = torch::nn::functional::cross_entropy(prediction, expected),
          .metrics = {{input.context.phase ==
                               jyppx::training::StepPhase::training
                           ? "training.accuracy"
                           : "validation.accuracy",
                       accuracy}},
          .sample_count = input.batch.sample_ids.size()};
    };

    jyppx::training::Trainer trainer({.trainer_id = "trainer.mnist",
                                      .run_id = "run.mnist",
                                      .trace_id = "trace.mnist",
                                      .max_epochs = epochs});
    trainer.prepare(
        {.model = model,
         .optimizer = optimizer,
         .training_data = train,
         .training_batch_mapper = mapper,
         .training_target_mapper = target,
         .training_step = step,
         .validation_data = validation,
         .validation_batch_mapper = mapper,
         .validation_target_mapper = target,
         .validation_step = step,
         .strategy = std::make_shared<jyppx::training::SingleDeviceStrategy>(
             jyppx::training::SingleDeviceStrategyOptions{
                 .strategy_id = "strategy.mnist", .device = device}),
         .model_checkpoint =
             std::make_shared<jyppx::nn::ModelCheckpointParticipant>(*model),
         .optimizer_checkpoint =
             std::make_shared<jyppx::optim::SgdOptimizerCheckpointParticipant>(
                 optimizer),
         .rng_checkpoint =
             std::make_shared<jyppx::core::RngCheckpointParticipant>()});
    const auto result = trainer.fit();
    std::cout << "training loss: "
              << result.training_metrics.at("training.loss") << '\n';
    std::cout << "validation accuracy: "
              << result.validation_metrics.at("validation.accuracy") << '\n';
    trainer.close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "mnist example error: " << error.what() << '\n';
    return 1;
  }
}
