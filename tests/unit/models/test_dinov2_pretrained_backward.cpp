// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "reference_file.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/models/models.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace {
using namespace jyppx::liuding;

constexpr const char* kReferenceDigest =
    "4ae767b4f071bebb46a4e63ad7b0002dae6f25f1f8f7d0471ded57654ecc77e2";
constexpr std::uint64_t kReferenceBytes = 33568048;

Tensor ReferenceTensor(std::span<const std::byte> encoded,
                       const std::string& name) {
  const interop::SafeTensorsView view(encoded);
  const auto& entries = view.directory().tensors;
  const auto entry =
      std::find_if(entries.begin(), entries.end(),
                   [&](const auto& value) { return value.name == name; });
  if (entry == entries.end() || entry->dtype != interop::SafeTensorDtype::f32)
    throw std::runtime_error("Missing DINO backward F32 reference: " + name);
  const auto bytes = view.tensor_bytes(name);
  const std::vector<std::int64_t> shape(entry->shape.begin(),
                                        entry->shape.end());
  auto result = torch::empty(shape, torch::kFloat32);
  std::memcpy(result.data_ptr(), bytes.data(), bytes.size());
  return result;
}

void Compare(const Tensor& actual, const Tensor& expected,
             const std::string& label, double relative, double absolute) {
  ASSERT_EQ(actual.sizes(), expected.sizes()) << label;
  const auto cpu = actual.detach().to(torch::kCPU);
  ASSERT_TRUE(torch::isfinite(cpu).all().item<bool>()) << label;
  EXPECT_TRUE(torch::allclose(cpu, expected, relative, absolute))
      << label
      << " max_abs_error=" << (cpu - expected).abs().max().item<double>();
}

void CheckBackward(bool cuda) {
  const auto* reference_root =
      std::getenv("JYPPX_LIUDING_DINOV2_BACKWARD_REFERENCE_DIR");
  const auto* weight_root =
      std::getenv("JYPPX_LIUDING_PRETRAINED_REFERENCE_DIR");
  if (!reference_root || !*reference_root || !weight_root || !*weight_root)
    GTEST_SKIP() << "Explicit pinned backward reference and weights required";
  if (cuda && !torch::cuda::is_available())
    GTEST_SKIP() << "CUDA runtime unavailable";
  torch::set_num_threads(1);
  at::globalContext().setDeterministicAlgorithms(true, false);
  at::globalContext().setDeterministicCuDNN(true);
  at::globalContext().setBenchmarkCuDNN(false);
  at::globalContext().setAllowTF32CuDNN(false);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setFloat32MatmulPrecision("highest");
  const models::test::MappedReference reference(
      std::filesystem::path(reference_root) /
          "dinov2-backward-reference.safetensors",
      kReferenceBytes);
  ASSERT_EQ(sha256_hex(reference), kReferenceDigest);
  const models::test::MappedReference weights(
      std::filesystem::path(weight_root) / "dinov2-small.liuding.safetensors",
      88240512U);
  models::WeightRecord record;
  record.id = "facebook.dinov2-small.unmasked.v1";
  record.model_name = "vit_small_patch14_dinov2";
  record.source_uri =
      "https://huggingface.co/facebook/dinov2-small/resolve/"
      "ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors";
  record.license_spdx = "Apache-2.0";
  record.license_uri =
      "https://huggingface.co/facebook/dinov2-small/blob/"
      "ed25f3a31f01632728cabb09d1542f84ab7b0056/README.md";
  record.availability = models::WeightAvailability::pinned_external;
  record.status_reason = "Pinned numerical reference; no task qualification.";
  ArtifactRef artifact;
  artifact.digest.hex =
      "f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca68b29db8a68c8c84c9810e5";
  artifact.size_bytes = weights.size();
  artifact.logical_name = "dinov2-small.liuding.safetensors";
  artifact.media_type = "application/x-safetensors";
  artifact.producer = "jyppx.liuding.dinov2-hf-small.unmasked.v1";
  record.artifact = artifact;
  models::ModelOptions options;
  options.dinov2.position_antialias = false;
  auto backbone = models::create_model(record.model_name, options);
  ASSERT_TRUE(models::load_weights(backbone, record, weights).applied);
  const auto device = torch::Device(cuda ? torch::kCUDA : torch::kCPU);
  backbone.state().to(device);
  backbone.state().train();
  auto head = torch::nn::Linear(384, 10);
  head->to(device);
  {
    torch::NoGradGuard guard;
    head->weight.copy_(ReferenceTensor(reference, "head_weight").to(device));
    head->bias.copy_(ReferenceTensor(reference, "head_bias").to(device));
  }
  std::map<std::string, Tensor> parameters;
  std::vector<optim::SgdParameterGroup> groups(2);
  groups[0].group_id = "backbone";
  groups[1].group_id = "head";
  for (std::size_t index = 0; index < groups.size(); ++index) {
    groups[index].options.learning_rate = index == 0 ? 0.0001 : 0.01;
    groups[index].options.momentum = 0.9;
    groups[index].options.weight_decay = 0.0001;
  }
  for (const auto& entry : backbone.state().named_parameters()) {
    const auto name = "backbone." + entry.key();
    parameters.emplace(name, entry.value());
    groups[0].parameters.push_back({name, entry.value()});
  }
  for (const auto& entry : head->named_parameters()) {
    const auto name = "head." + entry.key();
    parameters.emplace(name, entry.value());
    groups[1].parameters.push_back({name, entry.value()});
  }
  ASSERT_EQ(parameters.size(), 176U);
  optim::SgdOptimizerAdapter optimizer("reference.sgd", std::move(groups));
  const auto labels =
      ReferenceTensor(reference, "labels")
          .to(torch::TensorOptions().dtype(torch::kLong).device(device));
  const std::set<std::string> full{"backbone.cls_token",
                                   "backbone.pos_embed",
                                   "backbone.patch_embed.proj.weight",
                                   "backbone.blocks.0.attn.qkv.weight",
                                   "backbone.blocks.11.mlp.fc2.weight",
                                   "backbone.norm.weight",
                                   "head.weight",
                                   "head.bias"};
  for (const auto step : {1, 2}) {
    const auto suffix = std::to_string(step);
    optimizer.zero_grad();
    auto input = ReferenceTensor(reference, "input" + suffix).to(device);
    input.set_requires_grad(true);
    const auto features =
        backbone.invoke({.positional = {input}, .keyword = {}})
            .value.toTensor();
    const auto logits = head->forward(features);
    const auto loss = torch::nn::functional::cross_entropy(logits, labels);
    Compare(features, ReferenceTensor(reference, "features" + suffix),
            "features" + suffix, 3e-4, 3e-4);
    Compare(logits, ReferenceTensor(reference, "logits" + suffix),
            "logits" + suffix, 3e-4, 3e-4);
    Compare(loss.reshape({1}), ReferenceTensor(reference, "loss" + suffix),
            "loss" + suffix, 3e-4, 3e-4);
    loss.backward();
    Compare(input.grad(), ReferenceTensor(reference, "input_gradient" + suffix),
            "input_gradient" + suffix, 3e-3, 3e-6);
    for (const auto& [name, parameter] : parameters) {
      ASSERT_TRUE(parameter.grad().defined()) << name;
      const auto flat = parameter.grad().detach().to(torch::kCPU).flatten();
      const auto values = flat.to(torch::kFloat64);
      const auto stats =
          torch::stack({values.norm(), values.abs().max(), values.mean()})
              .to(torch::kFloat32);
      Compare(
          stats,
          ReferenceTensor(reference, "gradient_stats" + suffix + "." + name),
          "stats" + suffix + "." + name, 3e-3, 1e-6);
      Compare(
          flat.narrow(0, 0, std::min<std::int64_t>(16, flat.numel())),
          ReferenceTensor(reference, "gradient_sample" + suffix + "." + name),
          "sample" + suffix + "." + name, 3e-3, 3e-5);
      if (full.contains(name))
        Compare(parameter.grad(),
                ReferenceTensor(reference, "gradient" + suffix + "." + name),
                "gradient" + suffix + "." + name, 3e-3, 3e-5);
    }
    optimizer.step();
    for (const auto& name : full)
      Compare(parameters.at(name),
              ReferenceTensor(reference, "updated" + suffix + "." + name),
              "updated" + suffix + "." + name, 3e-4, 3e-5);
    EXPECT_EQ(optimizer.completed_steps(), static_cast<std::uint64_t>(step));
    std::cout << "step=" << step << " loss=" << loss.item<double>()
              << " gradient_parameters=" << parameters.size() << '\n';
  }
}

TEST(ModelsPretrainedBackward, DinoV2SmallCpuGradientsAndGroupedSgd) {
  CheckBackward(false);
}
#if defined(JYPPX_LIUDING_PRETRAINED_CUDA_TESTS)
TEST(ModelsPretrainedBackward, DinoV2SmallCudaGradientsAndGroupedSgd) {
  CheckBackward(true);
}
#endif
}  // namespace
