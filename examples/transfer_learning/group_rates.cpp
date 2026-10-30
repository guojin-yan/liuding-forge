// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/models/heads.hpp>
#include <jyppx/liuding/models/registry.hpp>
#include <jyppx/liuding/models/weights.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>

#include <torch/nn/functional/loss.h>
#include <torch/nn/modules/linear.h>
#include <torch/utils.h>

#include <ATen/Parallel.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace lf = jyppx::liuding;

// Use an explicit policy instead of changing requires_grad during a run.
enum class BackbonePolicy { frozen, finetune };

// Separate registered subtrees make optimizer bindings easy to inspect.
class TransferClassifier final : public torch::nn::Module {
 public:
  TransferClassifier(lf::nn::ModelAdapter backbone, BackbonePolicy policy,
                     std::int64_t classes)
      : backbone_(std::move(backbone)), policy_(policy) {
    const auto features =
        lf::models::classifier_info(backbone_).embedding_features;
    register_module("backbone", backbone_.state().shared_module());
    for (auto parameter : backbone_.state().parameters())
      parameter.set_requires_grad(policy_ == BackbonePolicy::finetune);
    head_ = register_module("head", torch::nn::Linear(features, classes));
    train();
  }

  void train(bool on = true) override {
    torch::nn::Module::train(on);
    // Frozen means fixed buffers and dropout behavior as well as gradients.
    if (policy_ == BackbonePolicy::frozen) backbone_.state().eval();
  }

  lf::Tensor forward(const lf::Tensor& images) {
    const auto embed = [&] {
      return backbone_.invoke({.positional = {images}}).value.toTensor();
    };
    lf::Tensor features;
    if (policy_ == BackbonePolicy::frozen) {
      torch::NoGradGuard guard;
      features = embed();
    } else {
      features = embed();
    }
    return head_->forward(features);
  }

 private:
  lf::nn::ModelAdapter backbone_;
  BackbonePolicy policy_;
  torch::nn::Linear head_{nullptr};
};

// This pins the same converted, unmasked official DINOv2-S input as the CLI
// descriptor. An architecture factory alone never means pretrained weights.
lf::models::WeightRecord pinned_weights() {
  return {.id = "facebook.dinov2-small.unmasked.v1",
          .model_name = "vit_small_patch14_dinov2",
          .source_uri =
              "https://huggingface.co/facebook/dinov2-small/resolve/"
              "ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors",
          .license_spdx = "Apache-2.0",
          .license_uri =
              "https://huggingface.co/facebook/dinov2-small/blob/"
              "ed25f3a31f01632728cabb09d1542f84ab7b0056/README.md",
          .availability = lf::models::WeightAvailability::pinned_external,
          .artifact =
              lf::ArtifactRef{
                  .digest = {.hex = "f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca6"
                                    "8b29db8a68c8c84c9810e5"},
                  .size_bytes = 88240512,
                  .logical_name = "dinov2-small.liuding.safetensors",
                  .media_type = "application/x-safetensors",
                  .producer = "jyppx.liuding.dinov2-hf-small.unmasked.v1"},
          .status_reason =
              "Explicit local pretrained input for an API example; task "
              "quality is not qualified."};
}

std::vector<std::byte> read_weights(const std::filesystem::path& path,
                                    std::uint64_t expected_bytes) {
  if (!std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) != expected_bytes)
    throw std::runtime_error("Need the exact local converted DINOv2-S weights");
  std::vector<std::byte> bytes(static_cast<std::size_t>(expected_bytes));
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  if (!stream || stream.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("Weight file changed while reading");
  return bytes;
}

std::shared_ptr<lf::optim::SgdOptimizerAdapter> make_optimizer(
    const lf::nn::ModelAdapter& model, BackbonePolicy policy) {
  std::vector<lf::optim::SgdParameterGroup> groups;
  if (policy == BackbonePolicy::finetune)
    groups.push_back({.group_id = "backbone",
                      .parameters = {},
                      .options = {.learning_rate = 0.0001,
                                  .momentum = 0.9,
                                  .weight_decay = 0.0001}});
  groups.push_back(
      {.group_id = "head",
       .parameters = {},
       .options = {
           .learning_rate = 0.01, .momentum = 0.9, .weight_decay = 0.0001}});
  for (const auto& parameter : model.state().named_parameters()) {
    if (!parameter.value().requires_grad()) continue;
    if (parameter.key().starts_with("backbone.") &&
        policy == BackbonePolicy::finetune)
      groups.front().parameters.push_back({parameter.key(), parameter.value()});
    else if (parameter.key().starts_with("head."))
      groups.back().parameters.push_back({parameter.key(), parameter.value()});
    else
      throw std::runtime_error("A trainable parameter has no explicit group");
  }
  return std::make_shared<lf::optim::SgdOptimizerAdapter>(
      "example.transfer.sgd", std::move(groups));
}

void print_rates(const lf::optim::SgdOptimizerAdapter& optimizer,
                 const char* phase) {
  const auto rates = optimizer.learning_rates();
  for (std::size_t index = 0; index < rates.size(); ++index)
    std::cout << phase << " group=" << optimizer.parameter_group_ids()[index]
              << " lr=" << rates[index]
              << " tensors=" << optimizer.groups()[index].parameters.size()
              << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 || (std::string(argv[2]) != "frozen" &&
                      std::string(argv[2]) != "finetune")) {
      std::cerr << "usage: liuding-forge-example-transfer-group-rates "
                   "<converted-dinov2-small.safetensors> <frozen|finetune>\n";
      return 2;
    }
    const auto policy = std::string(argv[2]) == "frozen"
                            ? BackbonePolicy::frozen
                            : BackbonePolicy::finetune;
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    torch::manual_seed(20261029);
    lf::models::ModelOptions options;
    options.dinov2.num_classes = 0;
    options.dinov2.position_antialias = false;
    auto backbone =
        lf::models::create_model("vit_small_patch14_dinov2", options);
    const auto record = pinned_weights();
    {
      const auto bytes = read_weights(std::filesystem::u8path(argv[1]),
                                      record.artifact->size_bytes);
      // Release the encoded weights before building the training graph.
      if (!lf::models::load_weights(backbone, record, bytes).applied)
        throw std::runtime_error(
            "Strict pretrained weight import was not applied");
    }
    auto concrete =
        std::make_shared<TransferClassifier>(std::move(backbone), policy, 2);
    auto model = lf::nn::adapt_tensor_module(concrete);
    // Import/replace heads and set policy before capturing optimizer bindings.
    auto optimizer = make_optimizer(model, policy);
    lf::optim::StepLrScheduler scheduler(
        optimizer,
        {.scheduler_id = "example.transfer.lr", .step_size = 1, .gamma = 0.9});
    print_rates(*optimizer, "before_epoch");

    // Two generated images keep this a small, deterministic mechanics lesson.
    // Replace this batch with your normalized image pipeline for real learning.
    const auto pixels = torch::rand({2, 3, 224, 224});
    const auto mean =
        torch::tensor({0.485F, 0.456F, 0.406F}).view({1, 3, 1, 1});
    const auto stddev =
        torch::tensor({0.229F, 0.224F, 0.225F}).view({1, 3, 1, 1});
    const auto images = (pixels - mean) / stddev;
    const auto labels = torch::tensor({0, 1}, torch::kLong);
    const auto parameters = model.state().named_parameters();
    const auto patch = parameters["backbone.patch_embed.proj.weight"];
    const auto head = parameters["head.weight"];
    const auto patch_before = patch.detach().clone();
    const auto head_before = head.detach().clone();
    model.state().train();
    optimizer->zero_grad();
    const auto logits = model.invoke({.positional = {images}}).value.toTensor();
    const auto loss = torch::nn::functional::cross_entropy(logits, labels);
    loss.backward();
    static_cast<void>(optimizer->step());
    // One synthetic batch is one demonstration epoch; decay after its updates.
    scheduler.step();
    std::cout << "scope=synthetic_api_demonstration loss="
              << loss.item<double>()
              << " optimizer_steps=" << optimizer->completed_steps()
              << " patch_delta="
              << (patch.detach() - patch_before).norm().item<double>()
              << " head_delta="
              << (head.detach() - head_before).norm().item<double>() << '\n';
    print_rates(*optimizer, "after_epoch");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
