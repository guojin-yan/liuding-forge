// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/random.hpp>
#include <jyppx/liuding/models/efficientnetv2.hpp>

#include <torch/csrc/autograd/autograd.h>
#include <torch/torch.h>

#include <ATen/autocast_mode.h>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace jyppx::liuding::models {
namespace {
struct Policy {
  bool deterministic = at::globalContext().deterministicAlgorithms();
  bool warn = at::globalContext().deterministicAlgorithmsWarnOnly();
  bool benchmark = at::globalContext().benchmarkCuDNN();
  bool cudnn = at::globalContext().deterministicCuDNN();
  bool tf32_cudnn = at::globalContext().allowTF32CuDNN();
  bool tf32_blas = at::globalContext().allowTF32CuBLAS();
  explicit Policy(bool strict = false) {
    torch::set_num_threads(1);
    at::globalContext().setDeterministicAlgorithms(strict, false);
    // Compare the option under one fixed numerical-test backend policy.
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setAllowTF32CuBLAS(false);
  }
  ~Policy() {
    at::globalContext().setDeterministicAlgorithms(deterministic, warn);
    at::globalContext().setBenchmarkCuDNN(benchmark);
    at::globalContext().setDeterministicCuDNN(cudnn);
    at::globalContext().setAllowTF32CuDNN(tf32_cudnn);
    at::globalContext().setAllowTF32CuBLAS(tf32_blas);
  }
};

struct Autocast {
  bool previous = at::autocast::is_autocast_enabled(torch::kCUDA);
  at::ScalarType dtype = at::autocast::get_autocast_dtype(torch::kCUDA);
  Autocast() {
    at::autocast::set_autocast_dtype(torch::kCUDA, at::kHalf);
    at::autocast::set_autocast_enabled(torch::kCUDA, true);
  }
  ~Autocast() {
    at::autocast::set_autocast_enabled(torch::kCUDA, previous);
    at::autocast::set_autocast_dtype(torch::kCUDA, dtype);
    at::autocast::clear_cache();
  }
};

bool HasRecomputation(const Tensor& output) {
  const auto root = output.grad_fn();
  std::vector<const torch::autograd::Node*> pending{root.get()};
  std::unordered_set<const torch::autograd::Node*> seen;
  while (!pending.empty()) {
    auto node = std::move(pending.back());
    pending.pop_back();
    if (!node || !seen.insert(node).second) continue;
    if (node->name().find("RecomputedBnAct") != std::string::npos) return true;
    for (const auto& edge : node->next_edges())
      pending.push_back(edge.function.get());
  }
  return false;
}

enum class Scenario {
  ordinary,
  features,
  frozen_affine,
  frozen_backbone,
  repeated,
  evaluation,
  cumulative,
  untracked,
  channels_last,
  half,
  bfloat16,
  autocast,
  strict,
  no_grad
};
using Values = std::map<std::string, Tensor>;

void CheckPair(const torch::Device& device, Scenario scenario) {
  SCOPED_TRACE(static_cast<int>(scenario));
  Policy policy(scenario == Scenario::strict);
  const auto make = [&](bool enabled) {
    torch::manual_seed(20261128);
    auto model = tf_efficientnetv2_s(
        {.num_classes = scenario == Scenario::features ? 0 : 5,
         .activation_recomputation = enabled});
    model->to(device);
    model->train(scenario != Scenario::evaluation);
    torch::NoGradGuard no_grad;
    for (const auto& module : model->modules(false)) {
      if (auto norm =
              std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(module)) {
        norm->weight.uniform_(.6, 1.4);
        norm->bias.uniform_(-.15, .2);
        norm->running_mean.uniform_(-.5, .5);
        norm->running_var.uniform_(.5, 1.5);
        norm->num_batches_tracked.fill_(7);
        norm->options.momentum(.03);
        if (scenario == Scenario::frozen_affine) {
          norm->weight.set_requires_grad(false);
          norm->bias.set_requires_grad(false);
        }
        if (scenario == Scenario::cumulative)
          norm->options.momentum(std::nullopt);
        if (scenario == Scenario::untracked)
          norm->options.track_running_stats(false);
      }
    }
    if (scenario == Scenario::frozen_backbone)
      for (const auto& parameter : model->named_parameters())
        parameter.value().set_requires_grad(
            parameter.key().starts_with("classifier."));
    if (scenario == Scenario::half) model->to(torch::kFloat16);
    if (scenario == Scenario::bfloat16) model->to(torch::kBFloat16);
    if (scenario == Scenario::channels_last)
      for (auto parameter : model->parameters())
        if (parameter.dim() == 4)
          parameter.set_data(
              parameter.contiguous(torch::MemoryFormat::ChannelsLast));
    return model;
  };
  auto original = make(false), candidate = make(true);
  auto source =
      torch::randn({2, 3, 65, 97}, torch::TensorOptions().device(device));
  if (scenario == Scenario::half) source = source.to(torch::kFloat16);
  if (scenario == Scenario::bfloat16) source = source.to(torch::kBFloat16);
  if (scenario == Scenario::channels_last)
    source = source.contiguous(torch::MemoryFormat::ChannelsLast);
  const auto run = [&](EfficientNetV2 model, bool enabled) {
    Values values;
    const auto save = [&](const std::string& key, const Tensor& value) {
      values.emplace(key, value.detach().cpu().clone());
    };
    auto input = source.detach().clone();
    input.set_requires_grad(scenario != Scenario::frozen_backbone);
    const auto before = core::get_rng_state(device.is_cuda());
    Tensor output;
    {
      torch::AutoGradMode grad(scenario != Scenario::no_grad);
      if (scenario == Scenario::autocast) {
        Autocast precision;
        output = model->forward(input);
      } else {
        output = model->forward(input);
      }
    }
    bool eligible = false;
#if defined(USE_ROCM)
    eligible =
        enabled && device.is_cuda() &&
        (scenario == Scenario::ordinary || scenario == Scenario::features ||
         scenario == Scenario::frozen_affine || scenario == Scenario::repeated);
#else
    static_cast<void>(enabled);
#endif
    // Convolutions may materialize contiguous regions from channels-last
    // images. Eligibility is checked per BN input, not per model input.
    if (scenario != Scenario::channels_last)
      EXPECT_EQ(HasRecomputation(output), eligible);
    save("output", output);
    auto loss = output.square().mean();
    if (scenario == Scenario::features) {
      auto features = model->forward_features(input);
      for (std::size_t index = 0; index < features.size(); ++index) {
        save("feature." + std::to_string(index), features[index]);
        loss = loss + features[index].square().mean() * (index + 1) / 100;
      }
    }
    if (scenario == Scenario::repeated)
      loss = loss + model->forward(input * .7).square().mean();
    save("loss", loss);
    if (scenario != Scenario::no_grad) {
      if (scenario == Scenario::repeated) loss.backward({}, true);
      loss.backward();
      if (input.requires_grad()) save("input.gradient", input.grad());
      for (const auto& parameter : model->named_parameters()) {
        EXPECT_EQ(parameter.value().requires_grad(),
                  parameter.value().grad().defined())
            << parameter.key();
        if (parameter.value().grad().defined())
          save("gradient." + parameter.key(), parameter.value().grad());
      }
      torch::optim::SGD optimizer(model->parameters(),
                                  torch::optim::SGDOptions(.0001).momentum(.9));
      optimizer.step();
    }
    for (const auto& parameter : model->named_parameters())
      save("parameter." + parameter.key(), parameter.value());
    for (const auto& buffer : model->named_buffers())
      save("buffer." + buffer.key(), buffer.value());
    const auto after = core::get_rng_state(device.is_cuda());
    EXPECT_TRUE(torch::equal(before.cpu, after.cpu));
    EXPECT_EQ(before.cuda.size(), after.cuda.size());
    for (std::size_t index = 0; index < before.cuda.size(); ++index)
      EXPECT_TRUE(torch::equal(before.cuda[index], after.cuda[index]));
    return values;
  };
  const auto expected = run(original, false), actual = run(candidate, true);
  ASSERT_EQ(expected.size(), actual.size());
  for (const auto& [key, value] : expected) {
    SCOPED_TRACE(key);
    const auto found = actual.find(key);
    ASSERT_NE(found, actual.end());
    ASSERT_TRUE(torch::isfinite(value).all().item<bool>());
    ASSERT_TRUE(torch::isfinite(found->second).all().item<bool>());
    if (value.is_floating_point())
      EXPECT_TRUE(torch::allclose(found->second, value, 2e-4, 2e-5))
          << "max_abs=" << (found->second - value).abs().max().item<double>();
    else
      EXPECT_TRUE(torch::equal(found->second, value));
    if (device.is_cpu()) EXPECT_TRUE(torch::equal(found->second, value));
  }
}

TEST(EfficientNetRecomputation, CpuFallbackAndDefaultOff) {
  EXPECT_FALSE(EfficientNetV2Options{}.activation_recomputation);
  CheckPair(torch::Device(torch::kCPU), Scenario::ordinary);
}

TEST(EfficientNetRecomputation, CudaFullGradientsStateAndRng) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  CheckPair(torch::Device(torch::kCUDA, 0), Scenario::ordinary);
}

TEST(EfficientNetRecomputation, CudaFeaturesFrozenAndRepeated) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  for (const auto scenario : {Scenario::features, Scenario::frozen_affine,
                              Scenario::frozen_backbone, Scenario::repeated})
    CheckPair(torch::Device(torch::kCUDA, 0), scenario);
}

TEST(EfficientNetRecomputation, CudaModeAndNormalizationFallbacks) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  for (const auto scenario :
       {Scenario::evaluation, Scenario::cumulative, Scenario::untracked,
        Scenario::no_grad, Scenario::strict})
    CheckPair(torch::Device(torch::kCUDA, 0), scenario);
}

TEST(EfficientNetRecomputation, CudaPrecisionAndLayoutFallbacks) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  for (const auto scenario :
       {Scenario::channels_last, Scenario::half, Scenario::autocast})
    CheckPair(torch::Device(torch::kCUDA, 0), scenario);
#if defined(USE_ROCM)
  CheckPair(torch::Device(torch::kCUDA, 0), Scenario::bfloat16);
#endif
}

#if defined(USE_ROCM)
TEST(EfficientNetRecomputation,
     RejectsHigherOrderAutocastBackwardAndSavedMutation) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  Policy policy;
  auto model =
      tf_efficientnetv2_s({.num_classes = 5, .activation_recomputation = true});
  model->to(torch::Device(torch::kCUDA, 0));
  auto input = torch::randn(
      {2, 3, 32, 32},
      torch::TensorOptions().device(torch::kCUDA).requires_grad(true));
  const auto output = model->forward(input);
  ASSERT_TRUE(HasRecomputation(output));
  EXPECT_THROW(torch::autograd::grad({output.sum()}, {input}, {}, false, true),
               c10::Error);
  const auto loss = model->forward(input).sum();
  {
    Autocast precision;
    EXPECT_THROW(loss.backward(), c10::Error);
    EXPECT_TRUE(at::autocast::is_autocast_enabled(torch::kCUDA));
  }
  for (const auto* key : {"bn1.weight", "bn1.bias"}) {
    const auto mutated_loss = model->forward(input).sum();
    {
      torch::NoGradGuard no_grad;
      model->named_parameters()[key].add_(.001);
    }
    EXPECT_THROW(mutated_loss.backward(), c10::Error);
  }
}
#endif
}  // namespace
}  // namespace jyppx::liuding::models
