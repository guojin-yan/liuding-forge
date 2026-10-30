// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/models/heads.hpp>
#include <jyppx/liuding/models/registry.hpp>
#include <jyppx/liuding/models/weights.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>

#include <torch/cuda.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/modules/linear.h>
#include <torch/utils.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {
namespace lf = jyppx::liuding;
constexpr std::int64_t kSamples = 12, kBatch = 3, kClasses = 3, kSteps = 12;
constexpr std::uint64_t kDefaultCacheBytes = 1U << 20U;

struct Options {
  std::string model = "vit_small_patch14_dinov2";
  std::string device = "cpu";
  std::filesystem::path weights;
  std::uint64_t cache_bytes = kDefaultCacheBytes;
};

std::uint64_t positive_bytes(std::string_view text) {
  std::uint64_t value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      value == 0)
    throw std::invalid_argument("--cache-bytes needs a positive byte count");
  return value;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 == argc)
      throw std::invalid_argument("Each option needs a value");
    const std::string_view key(argv[index]), value(argv[index + 1]);
    if (key == "--model")
      options.model = value;
    else if (key == "--device")
      options.device = value;
    else if (key == "--weights")
      options.weights = std::filesystem::u8path(value);
    else if (key == "--cache-bytes")
      options.cache_bytes = positive_bytes(value);
    else
      throw std::invalid_argument("Unknown option: " + std::string(key));
  }
  if (options.model != "vit_small_patch14_dinov2" &&
      options.model != "dinov3_vits16" && options.model != "resnet18")
    throw std::invalid_argument(
        "--model supports vit_small_patch14_dinov2, dinov3_vits16 or resnet18");
  if (options.device != "cpu" && options.device != "cuda")
    throw std::invalid_argument("--device supports cpu or cuda");
  if (!options.weights.empty() && options.model != "vit_small_patch14_dinov2")
    throw std::invalid_argument(
        "--weights accepts only the pinned converted DINOv2-S artifact");
  return options;
}

// Explicit local input only: the public loader verifies size, SHA and topology.
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
              "Optional exact local weights; this synthetic lesson does not "
              "qualify task quality."};
}

void load_local_weights(lf::nn::ModelAdapter& backbone,
                        const std::filesystem::path& path) {
  const auto record = pinned_weights();
  const auto size = record.artifact->size_bytes;
  if (!std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) != size)
    throw std::invalid_argument("Need the exact local converted DINOv2-S file");
  std::vector<std::byte> encoded(static_cast<std::size_t>(size));
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(encoded.data()),
              static_cast<std::streamsize>(encoded.size()));
  if (!stream || stream.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("Weights changed while reading");
  if (!lf::models::load_weights(backbone, record, encoded).applied)
    throw std::runtime_error("Strict local weight import was not applied");
  // encoded and loader staging are released before feature extraction.
}

std::string state_digest(const lf::nn::ModelAdapter& backbone) {
  lf::Sha256 digest;
  for (const auto& entries :
       {backbone.state().named_parameters(), backbone.state().named_buffers()})
    for (const auto& entry : entries) {
      digest.update(std::as_bytes(std::span(entry.key())));
      const auto value = entry.value().detach().cpu().contiguous();
      digest.update({static_cast<const std::byte*>(value.const_data_ptr()),
                     value.nbytes()});
    }
  return lf::sha256_digest_hex(digest.finish());
}

// Generate one small normalized RGB batch; no dataset or network is required.
lf::Tensor teaching_images(std::int64_t begin, std::int64_t end,
                           std::int64_t side) {
  const auto sample =
      torch::arange(begin, end, torch::kFloat32).view({-1, 1, 1, 1});
  const auto row =
      torch::arange(side, torch::kFloat32).view({1, 1, -1, 1}) / side;
  const auto column =
      torch::arange(side, torch::kFloat32).view({1, 1, 1, -1}) / side;
  const auto channel = torch::arange(3, torch::kFloat32).view({1, 3, 1, 1});
  const auto pattern =
      column * (sample.remainder(kClasses) + 1) * 6.0 + row * 4.0 + channel;
  const auto pixels = (pattern.sin() + 1.0) / 2.0;
  const auto mean = torch::tensor({0.485F, 0.456F, 0.406F}).view({1, 3, 1, 1});
  const auto stddev =
      torch::tensor({0.229F, 0.224F, 0.225F}).view({1, 3, 1, 1});
  return (pixels - mean) / stddev;
}

struct FeatureCache {
  lf::Tensor features;
  lf::Tensor labels;
  std::uint64_t bytes = 0;
  std::uint64_t forward_calls = 0;
};

FeatureCache extract_features(const Options& options) {
  lf::models::ModelOptions model_options;
  model_options.dinov2.position_antialias = false;
  auto backbone = lf::models::create_model(options.model, model_options);
  if (!options.weights.empty()) load_local_weights(backbone, options.weights);
  lf::models::reset_classifier(backbone, 0);
  const auto width = lf::models::classifier_info(backbone).embedding_features;
  if (width <= 0 ||
      static_cast<std::uint64_t>(width) >
          (std::numeric_limits<std::uint64_t>::max() - sizeof(std::int64_t)) /
              sizeof(float))
    throw std::runtime_error("Invalid feature cache dimensions");
  const auto per_sample =
      sizeof(float) * static_cast<std::uint64_t>(width) + sizeof(std::int64_t);
  if (per_sample > std::numeric_limits<std::uint64_t>::max() / kSamples)
    throw std::runtime_error("Invalid feature cache dimensions");
  const auto bytes = per_sample * kSamples;
  // Reject before allocating the cache or invoking the backbone.
  if (bytes > options.cache_bytes) {
    std::cout << "{\"reason\":\"cache_budget_exceeded\",\"required_bytes\":"
              << bytes << ",\"limit_bytes\":" << options.cache_bytes
              << ",\"feature_cache_allocated\":false,"
                 "\"backbone_forward_calls\":0}\n";
    throw std::invalid_argument("CPU feature cache exceeds --cache-bytes");
  }
  const auto device = torch::Device(options.device);
  if (device.is_cuda() && !torch::cuda::is_available())
    throw std::invalid_argument("CUDA was requested but is unavailable");
  backbone.state().to(device);
  backbone.state().eval();
  for (auto parameter : backbone.state().parameters())
    parameter.set_requires_grad(false);
  const auto before = state_digest(backbone);
  FeatureCache cache{
      torch::empty(
          {kSamples, width},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)),
      torch::arange(
          kSamples,
          torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
          .remainder(kClasses),
      bytes};
  const auto side = options.model == "vit_small_patch14_dinov2" ? 28 : 32;
  {
    torch::NoGradGuard guard;
    for (std::int64_t begin = 0; begin < kSamples; begin += kBatch) {
      const auto end = std::min(begin + kBatch, kSamples);
      const auto images = teaching_images(begin, end, side).to(device);
      auto features =
          backbone.invoke({.positional = {images}}).value.toTensor();
      ++cache.forward_calls;
      if (features.dim() != 2 || features.size(0) != end - begin ||
          features.size(1) != width || features.requires_grad() ||
          !torch::isfinite(features).all().item<bool>())
        throw std::runtime_error("Unexpected detached finite NC embedding");
      // This lesson explicitly chooses L2-normalized frozen representations.
      features = features / features.norm(2, 1, true).clamp_min(1e-12);
      cache.features.slice(0, begin, end).copy_(features.cpu());
    }
  }
  if (backbone.state().is_training() || state_digest(backbone) != before)
    throw std::runtime_error("Frozen backbone changed while extracting");
  for (const auto& parameter : backbone.state().parameters())
    if (parameter.requires_grad() || parameter.grad().defined())
      throw std::runtime_error("Frozen backbone retained a gradient");
  return cache;  // The backbone and the last image/feature batch are destroyed.
}

struct HeadResult {
  std::uint64_t optimizer_steps;
  double delta, loss_before, loss_after;
};

HeadResult train_head(const FeatureCache& cache) {
  auto head =
      std::make_shared<torch::nn::LinearImpl>(cache.features.size(1), kClasses);
  {
    torch::NoGradGuard guard;
    head->weight.zero_();
    head->bias.zero_();
  }
  std::vector<lf::optim::OptimizerParameterBinding> bindings;
  for (const auto& parameter : head->named_parameters())
    bindings.push_back({parameter.key(), parameter.value()});
  lf::optim::SgdOptimizerAdapter optimizer(
      "example.linear_probe.sgd",
      {{.group_id = "head",
        .parameters = std::move(bindings),
        .options = {.learning_rate = 0.25, .momentum = 0.9}}});
  const auto initial_weight = head->weight.detach().clone();
  const auto loss_before = torch::nn::functional::cross_entropy(
                               head->forward(cache.features), cache.labels)
                               .item<double>();
  head->train();
  for (std::int64_t step = 0; step < kSteps; ++step) {
    optimizer.zero_grad();
    const auto loss = torch::nn::functional::cross_entropy(
        head->forward(cache.features), cache.labels);
    if (!torch::isfinite(loss).item<bool>())
      throw std::runtime_error("Head loss is not finite");
    loss.backward();
    static_cast<void>(optimizer.step());
  }
  torch::NoGradGuard guard;
  const auto delta =
      (head->weight.detach() - initial_weight).norm().item<double>();
  const auto loss_after = torch::nn::functional::cross_entropy(
                              head->forward(cache.features), cache.labels)
                              .item<double>();
  if (optimizer.completed_steps() != kSteps || !std::isfinite(delta) ||
      delta <= 0 || !std::isfinite(loss_after))
    throw std::runtime_error("The classification head did not update");
  return {optimizer.completed_steps(), delta, loss_before, loss_after};
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    torch::manual_seed(20261029);
    const auto cache = extract_features(options);
    const auto result = train_head(cache);
    // One JSON result makes the lesson easy to inspect or use in a script.
    std::cout << std::setprecision(10)
              << "{\"scope\":\"synthetic_api_demonstration\",\"model\":\""
              << options.model << "\",\"backbone_device\":\"" << options.device
              << "\",\"weights\":\""
              << (options.weights.empty() ? "random" : "pinned_local")
              << "\",\"samples\":" << kSamples
              << ",\"cache_bytes\":" << cache.bytes
              << ",\"cache_limit_bytes\":" << options.cache_bytes
              << ",\"cache_device\":\"cpu\",\"features_requires_grad\":false,"
                 "\"backbone_forward_calls\":"
              << cache.forward_calls
              << ",\"backbone_unchanged\":true,\"backbone_eval\":true,"
                 "\"backbone_gradients\":0,"
                 "\"backbone_released_before_training\":true,"
                 "\"head_device\":\"cpu\",\"optimizer_steps\":"
              << result.optimizer_steps << ",\"head_delta\":" << result.delta
              << ",\"loss_before\":" << result.loss_before
              << ",\"loss_after\":" << result.loss_after << "}\n";
    return 0;
  } catch (const std::invalid_argument& error) {
    std::cerr << "error=" << error.what()
              << "\nusage: liuding-forge-example-transfer-linear-probe "
                 "[--model vit_small_patch14_dinov2|dinov3_vits16|resnet18] "
                 "[--device cpu|cuda] [--cache-bytes N] "
                 "[--weights converted-dinov2-small.safetensors]\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
