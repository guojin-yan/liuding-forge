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

Tensor RealBatchTensor(const std::filesystem::path& path,
                       const std::vector<std::int64_t>& shape,
                       const char* digest,
                       torch::ScalarType dtype = torch::kFloat32) {
  auto result = torch::empty(shape, torch::TensorOptions().dtype(dtype));
  const models::test::MappedReference encoded(path, result.nbytes());
  if (sha256_hex(encoded) != digest)
    throw std::runtime_error("Fixed real-batch tensor digest differs");
  const std::span<const std::byte> bytes = encoded;
  std::memcpy(result.data_ptr(), bytes.data(), bytes.size());
  return result;
}

// This is the eight-image regression that fails with the C++ frontend's
// separate matmul/bias path. References remain the original independent HF
// eager results; neither their values nor the numerical tolerances change.
TEST(ModelsPretrainedBackward, DinoV2SmallRealFirstBatchCpuGradientsAndSgd) {
  const auto* fixture = std::getenv("JYPPX_LIUDING_DINOV2_REAL_FIRST_STEP_DIR");
  const auto* weight_root =
      std::getenv("JYPPX_LIUDING_PRETRAINED_REFERENCE_DIR");
  if (!fixture || !*fixture || !weight_root || !*weight_root)
    GTEST_SKIP() << "Explicit fixed real first-step capture/reference required";
  torch::set_num_threads(1);
  at::globalContext().setDeterministicAlgorithms(true, false);
  at::globalContext().setFloat32MatmulPrecision("highest");
  const auto root = std::filesystem::path(fixture);
  const auto capture = root / "capture";
  const auto reference = root / "reference";
  const models::test::MappedReference encoded(
      std::filesystem::path(weight_root) / "dinov2-small.liuding.safetensors",
      88240512U);
  ASSERT_EQ(sha256_hex(encoded),
            "f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca68b29db8a68c8c84c9810e5");
  auto backbone = models::dinov2_vits14({.position_antialias = false});
  backbone->train();
  {
    const interop::SafeTensorsView weights(encoded);
    ASSERT_EQ(weights.directory().tensors.size(), 174U);
    torch::NoGradGuard guard;
    for (const auto& parameter : backbone->named_parameters()) {
      const auto value = ReferenceTensor(encoded, parameter.key());
      ASSERT_EQ(value.sizes(), parameter.value().sizes());
      parameter.value().copy_(value);
    }
  }
  auto head = torch::nn::Linear(384, 10);
  {
    torch::NoGradGuard guard;
    head->weight.copy_(RealBatchTensor(
        capture / "head-weight.f32", {10, 384},
        "d5f72ca217166a0d09336fdd6bd3a2968fbb6888e00b2566a648d00c54c60993"));
    head->bias.copy_(RealBatchTensor(
        capture / "head-bias.f32", {10},
        "f0f311719a1285d5b40205d2fed9e9adb034ef9d8dc8332497ebfe7c659943aa"));
  }
  std::map<std::string, Tensor> parameters;
  std::vector<optim::SgdParameterGroup> groups(2);
  for (std::size_t index = 0; index < groups.size(); ++index) {
    groups[index].group_id = index == 0 ? "backbone" : "head";
    groups[index].options.learning_rate = index == 0 ? 0.0001 : 0.01;
    groups[index].options.momentum = 0.9;
    groups[index].options.weight_decay = 0.0001;
  }
  for (const auto& entry : backbone->named_parameters()) {
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
  optim::SgdOptimizerAdapter optimizer("real-first-step.sgd",
                                       std::move(groups));
  const auto input = RealBatchTensor(
      capture / "input.f32", {8, 3, 224, 224},
      "5b26501aaf619fe2548bfca1a4364ad948769f32b7c488bc90fa85fd738207c0");
  const auto labels = RealBatchTensor(
      capture / "labels.i64", {8},
      "94cf11a79244e63b1de366b6887293ddd6c42b867fbdcb53b164ac3cbc235e45",
      torch::kInt64);
  optimizer.zero_grad();
  const auto logits = head->forward(backbone->forward(input));
  const auto loss = torch::nn::functional::cross_entropy(logits, labels);
  Compare(
      logits,
      RealBatchTensor(
          reference / "logits.f32", {8, 10},
          "5153455bb1840520eb1590b8f0041fd9d6c2bf6c38f7f66f2df4b92c8f1c537e"),
      "real-first-step.logits", 3e-4, 3e-4);
  Compare(
      loss.reshape({1}),
      RealBatchTensor(
          reference / "loss.f32", {1},
          "34ae934d033ab4331240d43eccb8574605dc3a78dfa723f1654d3f2a0b99d2ea"),
      "real-first-step.loss", 3e-4, 3e-4);
  loss.backward();
  struct Check {
    const char* name;
    const char* index;
    std::vector<std::int64_t> shape;
    const char* gradient_digest;
    const char* updated_digest;
  };
  const std::array<Check, 6> checks{{
      {"backbone.blocks.0.attn.qkv.weight",
       "003",
       {1152, 384},
       "11861a38d51f42f500547243028ce87ee87c9148f1f831f8502723e48ac754a0",
       "d37785e98670b62c7bc7a5e60aaa730f87518c2a196243c39377b828bebb3520"},
      {"backbone.blocks.11.mlp.fc2.weight",
       "051",
       {384, 1536},
       "810c360da5787ee5aa23959cf23fc0cf41d741855da89ea76e2468ef43f05e1f",
       "2ee28dd3a2ee55314b95f2394363da32f40d58817a1e5dfd4b9c58588b2941a3"},
      {"backbone.norm.weight",
       "170",
       {384},
       "48f800ad6d7b6ce5b3f750fd4e28bbea942117637b0bf390aa4767a514cfca62",
       "1e188926e85ae4cebf1292e2cd1d9e47db8f5466c034ba472c738ca536f7a3d4"},
      {"backbone.patch_embed.proj.weight",
       "172",
       {384, 3, 14, 14},
       "614ed7933029634c84066b02492e84283be27a1843e93a66d644a7c5bc9cf8c4",
       "450d0105a2561cc43fef3d4d27bfab62b514d53032da8ba9d07eb79a6d0d066a"},
      {"head.bias",
       "174",
       {10},
       "95ddc9a30f5b3bc5883153a8eb94cfce93e75cf0962dc1700b372f199b8f2bd2",
       "752c10c1bda7a4c78c6234ae328e5eeec94daf7be724c47b070e56936f0decbe"},
      {"head.weight",
       "175",
       {10, 384},
       "c54d97b95ce128514d36c7262c4e04f570b6217421659e1a152e5229888f800a",
       "6fcc08e1fa9a0f2efca73fc2ae46a1bf038ac3cd93145b841bed82364dc8b2bf"},
  }};
  for (const auto& check : checks) {
    const auto gradient = parameters.at(check.name).grad();
    ASSERT_TRUE(gradient.defined()) << check.name;
    Compare(gradient,
            RealBatchTensor(
                reference / "gradient" / (std::string(check.index) + ".f32"),
                check.shape, check.gradient_digest),
            std::string("real-first-step.gradient.") + check.name, 3e-3, 3e-5);
  }
  optimizer.step();
  EXPECT_EQ(optimizer.completed_steps(), 1U);
  for (const auto& check : checks)
    Compare(parameters.at(check.name),
            RealBatchTensor(
                reference / "updated" / (std::string(check.index) + ".f32"),
                check.shape, check.updated_digest),
            std::string("real-first-step.updated.") + check.name, 3e-4, 3e-5);
}
#if defined(JYPPX_LIUDING_PRETRAINED_CUDA_TESTS)
TEST(ModelsPretrainedBackward, DinoV2SmallCudaGradientsAndGroupedSgd) {
  CheckBackward(true);
}
#endif
}  // namespace
