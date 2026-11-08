// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/models/vision.hpp>

#include <torch/cuda.h>
#include <torch/optim.h>
#include <torch/torch.h>

#include <array>
#include <gtest/gtest.h>
#include <memory>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::models;

void CheckResNetForward(const Device& device) {
  constexpr std::array variants{
      ResNetVariant::resnet18, ResNetVariant::resnet34,
      ResNetVariant::resnet50, ResNetVariant::resnet101,
      ResNetVariant::resnext50_32x4d};
  torch::set_num_threads(1);
  at::globalContext().setDeterministicCuDNN(true);
  at::globalContext().setBenchmarkCuDNN(false);
  at::globalContext().setAllowTF32CuDNN(false);
  at::globalContext().setAllowTF32CuBLAS(false);
  for (const auto variant : variants) {
    SCOPED_TRACE(static_cast<int>(variant));
    torch::manual_seed(49001);
    ResNet classifier(variant, ResNetOptions{.num_classes = 3});
    ResNet features(variant, ResNetOptions{.num_classes = 3});
    classifier->to(device);
    features->to(device);
    {
      torch::NoGradGuard guard;
      for (const auto& item : classifier->named_parameters())
        features->named_parameters()[item.key()].copy_(item.value());
      for (const auto& item : classifier->named_buffers())
        features->named_buffers()[item.key()].copy_(item.value());
    }
    const auto head = std::dynamic_pointer_cast<torch::nn::LinearImpl>(
        features->named_modules()["fc"]);
    ASSERT_TRUE(head);
    torch::optim::SGD first(classifier->parameters(),
                            torch::optim::SGDOptions(0.01).momentum(0.9));
    torch::optim::SGD second(features->parameters(),
                             torch::optim::SGDOptions(0.01).momentum(0.9));
    for (int step = 0; step < 2; ++step) {
      const auto values =
          (torch::linspace(-1, 1, 2 * 3 * 64 * 64) + step / 32.0)
              .reshape({2, 3, 64, 64})
              .to(device);
      auto a = values.clone().requires_grad_(true);
      auto b = values.clone().requires_grad_(true);
      first.zero_grad();
      second.zero_grad();
      const auto actual = classifier->forward(a);
      const auto expected =
          head->forward(features->forward_features(b).back().mean({2, 3}));
      ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
      ASSERT_TRUE(torch::equal(actual, expected));
      const auto labels = torch::tensor({0, 2}, torch::kInt64).to(device);
      torch::nn::functional::cross_entropy(actual, labels).backward();
      torch::nn::functional::cross_entropy(expected, labels).backward();
      ASSERT_TRUE(torch::equal(a.grad(), b.grad()));
      for (const auto& item : classifier->named_parameters()) {
        const auto other = features->named_parameters()[item.key()];
        ASSERT_EQ(item.value().grad().defined(), other.grad().defined());
        if (other.grad().defined()) {
          ASSERT_TRUE(torch::isfinite(other.grad()).all().item<bool>());
          ASSERT_TRUE(torch::equal(item.value().grad(), other.grad()))
              << item.key();
        }
      }
      first.step();
      second.step();
      for (const auto& item : classifier->named_parameters())
        ASSERT_TRUE(torch::equal(
            item.value(), features->named_parameters()[item.key()]))
            << item.key();
      for (const auto& item : classifier->named_buffers())
        ASSERT_TRUE(torch::equal(
            item.value(), features->named_buffers()[item.key()]))
            << item.key();
    }
    classifier->reset_classifier(0);
    classifier->eval();
    torch::NoGradGuard guard;
    const auto input = torch::zeros({2, 3, 64, 64}).to(device);
    ASSERT_TRUE(torch::equal(
        classifier->forward(input),
        classifier->forward_features(input).back().mean({2, 3})));
  }
}

TEST(ResNetForward, CpuMatchesFeaturePathGradientsUpdatesAndBuffers) {
  CheckResNetForward(Device(c10::kCPU));
}

TEST(ResNetForward, CudaMatchesFeaturePathGradientsUpdatesAndBuffers) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is unavailable";
  CheckResNetForward(Device(c10::kCUDA, 0));
}
}  // namespace
