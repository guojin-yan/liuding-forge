// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "mnist_data.hpp"

#include <jyppx/liuding/liuding.hpp>

#include <torch/nn/init.h>

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using jyppx::liuding::Tensor;

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

/** @brief Resolves an external MNIST directory without creating a cache. */
std::filesystem::path resolve_data_directory(
    const std::optional<std::filesystem::path>& directory) {
  if (directory) return *directory;
#ifdef _WIN32
  if (const wchar_t* root = _wgetenv(L"JYPPX_DATA_ROOT"); root && *root) {
    return std::filesystem::path(root) / "mnist";
  }
#else
  if (const char* root = std::getenv("JYPPX_DATA_ROOT");
      root != nullptr && *root != '\0') {
    return std::filesystem::path(root) / "mnist";
  }
#endif
  throw std::invalid_argument(
      "set JYPPX_DATA_ROOT or pass a prepared MNIST data directory; "
      "see liuding-forge data prepare --name mnist --source-dir "
      "LOCAL_GZIP_DIRECTORY");
}

/** @brief Parses a complete positive epoch count, rejecting signed suffixes. */
std::uint64_t parse_epochs(std::string_view text) {
  std::uint64_t result = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), result);
  if (error != std::errc{} || end != text.data() + text.size() || result == 0) {
    throw std::invalid_argument("epochs must be a positive integer");
  }
  return result;
}

/** @brief Creates one checkpointable tensor-backed MNIST pipeline. */
std::shared_ptr<jyppx::liuding::data::DataPipeline> make_pipeline(
    const Tensor& images, const Tensor& labels, std::size_t batch_size,
    std::string pipeline_id, bool shuffle) {
  const auto size = static_cast<std::size_t>(images.size(0));
  std::shared_ptr<jyppx::liuding::data::samplers::Sampler> sampler;
  if (shuffle) {
    sampler = std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(
        size, 2026);
  } else {
    sampler =
        std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
            size);
  }
  return std::make_shared<jyppx::liuding::data::DataPipeline>(
      std::move(sampler),
      jyppx::liuding::data::DataPipelineOptions{
          .pipeline_id = std::move(pipeline_id),
          .dataset_version = "mnist.idx/v1",
          .dataset_size = size,
          .schema = jyppx::liuding::data::BatchSchema::any("mnist.batch", 1),
          .batch_factory_id = "mnist.tensor-slice/v1",
          .batch_size = batch_size,
          .num_workers = shuffle ? 2U : 0U,
          .worker_seed = 2026},
      [images, labels](std::span<const std::size_t> indices,
                       const jyppx::liuding::data::DataWorkerContext&,
                       std::stop_token) {
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
    std::uint64_t epochs = 3U;
    std::optional<std::filesystem::path> directory;
    std::optional<std::string> requested_device;
    bool epochs_supplied = false;
    for (int index = 1; index < argc; ++index) {
      const std::string_view argument(argv[index]);
      if (argument == "--help") {
        std::cout << "Usage: liuding-forge-example-mnist [data_dir] [epochs] "
                     "[--epochs N] [--device cpu|cuda[:index]] [--offline]\n"
                     "Default data: $JYPPX_DATA_ROOT/mnist. Reads local IDX "
                     "files only. Prepare data with "
                     "liuding-forge data prepare --name mnist --source-dir "
                     "LOCAL_GZIP_DIRECTORY.\n";
        return 0;
      } else if (argument == "--offline") {
        // All training runs are offline; preparation is a separate command.
        continue;
      } else if (argument == "--epochs" || argument == "--device") {
        if (index + 1 >= argc) {
          throw std::invalid_argument(std::string(argument) +
                                      " requires a value");
        }
        if (argument == "--device") {
          requested_device = argv[++index];
        } else {
          if (epochs_supplied)
            throw std::invalid_argument("epochs specified more than once");
          epochs = parse_epochs(argv[++index]);
          epochs_supplied = true;
        }
      } else if (argument.starts_with('-')) {
        throw std::invalid_argument("unknown option: " + std::string(argument));
      } else if (!directory) {
        directory = std::filesystem::path(argv[index]);
      } else if (!epochs_supplied) {
        epochs = parse_epochs(argument);
        epochs_supplied = true;
      } else {
        throw std::invalid_argument("unexpected argument: " +
                                    std::string(argument));
      }
    }
    const auto data_dir = resolve_data_directory(directory);
    std::cout << "MNIST data directory: " << data_dir << " (offline)\n";
    for (const auto* file :
         {"train-images-idx3-ubyte", "train-labels-idx1-ubyte",
          "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"}) {
      if (!std::filesystem::is_regular_file(data_dir / file)) {
        throw std::runtime_error(
            "missing local MNIST IDX file: " + (data_dir / file).string() +
            "; prepare the external cache with "
            "liuding-forge data prepare --name mnist --source-dir "
            "LOCAL_GZIP_DIRECTORY");
      }
    }

    const auto train_raw =
        mnist_data::load_mnist_idx(data_dir / "train-images-idx3-ubyte",
                                   data_dir / "train-labels-idx1-ubyte");
    const auto test_raw =
        mnist_data::load_mnist_idx(data_dir / "t10k-images-idx3-ubyte",
                                   data_dir / "t10k-labels-idx1-ubyte");
    auto device = requested_device
                      ? jyppx::liuding::parse_device(*requested_device)
                      : jyppx::liuding::get_default_device();
    // Checkpoint topology requires a concrete device; the shorthand selects 0.
    if (device.is_cuda() && !device.has_index()) device.set_index(0);
    torch::manual_seed(2026);
    torch::set_num_threads(2);
    std::cout << "training on: " << device << '\n';

    auto holder = std::make_shared<MnistCnn>();
    auto model = std::make_shared<jyppx::liuding::nn::ModelAdapter>(
        jyppx::liuding::nn::adapt_tensor_module(holder));
    std::vector<jyppx::liuding::optim::OptimizerParameterBinding> bindings;
    for (const auto& item : model->state().named_parameters()) {
      bindings.push_back(
          {.parameter_id = item.key(), .parameter = item.value()});
    }
    auto optimizer =
        std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
            "optimizer.mnist",
            std::vector<jyppx::liuding::optim::SgdParameterGroup>{
                {.group_id = "main",
                 .parameters = std::move(bindings),
                 .options = {.learning_rate = 0.01, .momentum = 0.9}}});
    auto train = make_pipeline(train_raw.images, train_raw.labels, 64,
                               "data.mnist.train", true);
    auto validation = make_pipeline(test_raw.images, test_raw.labels, 256,
                                    "data.mnist.validation", false);
    const auto mapper = [device](
                            const jyppx::liuding::data::BatchEnvelope& batch) {
      return jyppx::liuding::nn::Invocation{
          .positional = {
              batch.value.toTuple()->elements().at(0).toTensor().to(device)}};
    };
    const auto target =
        [device](const jyppx::liuding::data::BatchEnvelope& batch)
        -> std::optional<c10::IValue> {
      return batch.value.toTuple()->elements().at(1).toTensor().to(device);
    };
    const auto step = [](const jyppx::liuding::training::StepInput& input) {
      const auto prediction =
          input.model.invoke(input.invocation).value.toTensor();
      const auto expected =
          input.target->toTensor().to(torch::kLong).view({-1});
      const auto accuracy = prediction.argmax(1)
                                .eq(expected)
                                .to(torch::kFloat32)
                                .mean()
                                .item<double>();
      return jyppx::liuding::training::StepOutput{
          .loss = torch::nn::functional::cross_entropy(prediction, expected),
          .metrics = {{input.context.phase ==
                               jyppx::liuding::training::StepPhase::training
                           ? "training.accuracy"
                           : "validation.accuracy",
                       accuracy}},
          .sample_count = input.batch.sample_ids.size()};
    };

    jyppx::liuding::training::Trainer trainer({.trainer_id = "trainer.mnist",
                                               .run_id = "run.mnist",
                                               .trace_id = "trace.mnist",
                                               .max_epochs = epochs});
    trainer.prepare(
        {.model = model,
         .optimizers =
             {{.optimizer = optimizer,
               .checkpoint = std::make_shared<
                   jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
                   optimizer)}},
         .training_data = train,
         .training_batch_mapper = mapper,
         .training_target_mapper = target,
         .training_step = step,
         .validation_data = validation,
         .validation_batch_mapper = mapper,
         .validation_target_mapper = target,
         .validation_step = step,
         .strategy =
             std::make_shared<jyppx::liuding::training::SingleDeviceStrategy>(
                 jyppx::liuding::training::SingleDeviceStrategyOptions{
                     .strategy_id = "strategy.mnist", .device = device}),
         .model_checkpoint =
             std::make_shared<jyppx::liuding::nn::ModelCheckpointParticipant>(
                 *model),
         .rng_checkpoint = std::make_shared<
             jyppx::liuding::core::RngCheckpointParticipant>()});
    const auto result = trainer.fit();
    std::cout << "completed epochs: " << result.completed_epochs << '\n';
    std::cout << "training steps: " << result.global_step << '\n';
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
