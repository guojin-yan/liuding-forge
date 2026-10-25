// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");

#include "reference_file.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/models/models.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::models;

Tensor Images(std::int64_t channels = 3) {
  return torch::zeros({2, channels, 64, 64}, torch::kFloat32);
}

TEST(Models, RegistryListsPreviewArchitecturesAndCreatesAdapters) {
  const auto entries = list_models();
  ASSERT_EQ(entries.size(), 9U);
  EXPECT_EQ(entries.front().name, "mobilenet_v3_large");
  EXPECT_EQ(list_models("resnet").size(), 2U);

  auto residual = create_model("resnet18");
  auto mobile = create_model("mobilenet_v3_small");
  auto transformer = create_model("transformer_encoder",
                                  {.transformer = {.model_dimension = 32,
                                                   .attention_heads = 4,
                                                   .feedforward_dimension = 64,
                                                   .layers = 2,
                                                   .dropout = 0.0,
                                                   .gelu = true,
                                                   .final_norm = true}});
  residual.state().eval();
  mobile.state().eval();
  transformer.state().eval();
  EXPECT_EQ(residual.invoke({.positional = {Images()}, .keyword = {}})
                .value.toTensor()
                .sizes(),
            torch::IntArrayRef({2, 1000}));
  EXPECT_EQ(mobile.invoke({.positional = {Images()}, .keyword = {}})
                .value.toTensor()
                .sizes(),
            torch::IntArrayRef({2, 1000}));
  const auto encoded = transformer.invoke(
      {.positional = {torch::zeros({2, 5, 32})}, .keyword = {}});
  EXPECT_EQ(encoded.value.toTensor().sizes(), torch::IntArrayRef({2, 5, 32}));
}

TEST(Models, DinoV2SmallPreservesClsPatchContractAndSupportsHeadReplacement) {
  auto backbone = dinov2_vits14();
  backbone->eval();
  const auto parameters = backbone->named_parameters(true);
  std::vector<std::string> parameter_names;
  parameter_names.reserve(parameters.size());
  for (const auto& entry : parameters) parameter_names.push_back(entry.key());
  for (const auto* key :
       {"patch_embed.proj.weight", "cls_token", "pos_embed",
        "blocks.0.norm1.weight", "blocks.0.attn.qkv.weight",
        "blocks.0.attn.proj.weight", "blocks.0.ls1.gamma",
        "blocks.0.norm2.weight", "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight", "blocks.0.ls2.gamma", "norm.weight"}) {
    EXPECT_NE(std::find(parameter_names.begin(), parameter_names.end(), key),
              parameter_names.end())
        << key;
  }

  const auto input = torch::randn({1, 3, 224, 224});
  const auto tokens = backbone->forward_features(input);
  EXPECT_EQ(tokens.sizes(), torch::IntArrayRef({1, 257, 384}));
  EXPECT_EQ(backbone->forward(input).sizes(), torch::IntArrayRef({1, 384}));
  EXPECT_EQ(parameter_count(*backbone), 22'056'192U);

  auto rectangular_input = torch::randn({1, 3, 28, 42});
  rectangular_input.set_requires_grad(true);
  backbone->train();
  const auto rectangular_tokens = backbone->forward_features(rectangular_input);
  EXPECT_EQ(rectangular_tokens.sizes(), torch::IntArrayRef({1, 7, 384}));
  rectangular_tokens.square().mean().backward();
  EXPECT_TRUE(torch::isfinite(rectangular_input.grad()).all().item<bool>());
  bool qkv_gradient_defined = false;
  for (const auto& entry : backbone->named_parameters(true)) {
    if (entry.key() == "blocks.0.attn.qkv.weight") {
      qkv_gradient_defined =
          entry.value().grad().defined() &&
          torch::isfinite(entry.value().grad()).all().item<bool>();
      break;
    }
  }
  EXPECT_TRUE(qkv_gradient_defined);

  backbone->eval();
  backbone->reset_classifier(5);
  EXPECT_EQ(backbone->forward(input).sizes(), torch::IntArrayRef({1, 5}));
  const auto parameter_count_with_head = parameter_count(*backbone);
  EXPECT_THROW(backbone->reset_classifier(-1), InvalidArgument);
  EXPECT_EQ(backbone->options().num_classes, 5);
  EXPECT_EQ(parameter_count(*backbone), parameter_count_with_head);
  backbone->reset_classifier(0);
  EXPECT_EQ(backbone->forward(input).sizes(), torch::IntArrayRef({1, 384}));

  auto classifier =
      create_model("vit_small_patch14_dinov2",
                   {.dinov2 = {.num_classes = 7, .input_channels = 3}});
  classifier.state().eval();
  const auto logits = classifier.invoke({.positional = {input}, .keyword = {}})
                          .value.toTensor();
  EXPECT_EQ(logits.sizes(), torch::IntArrayRef({1, 7}));
  EXPECT_THROW(backbone->forward(torch::zeros({1, 3, 225, 224})),
               InvalidArgument);

  backbone->to(torch::kFloat64);
  backbone->reset_classifier(3);
  const auto double_input = torch::zeros({1, 3, 14, 14}, torch::kFloat64);
  EXPECT_EQ(backbone->forward_features(double_input).scalar_type(),
            torch::kFloat64);
  EXPECT_EQ(backbone->forward(double_input).sizes(),
            torch::IntArrayRef({1, 3}));
}

TEST(Models,
     DinoV2CapacityMetadataIsExplicitAndGiantIsNotConstructedImplicitly) {
  const auto small = model_info("vit_small_patch14_dinov2");
  const auto base = model_info("vit_base_patch14_dinov2");
  const auto large = model_info("vit_large_patch14_dinov2");
  const auto giant = model_info("vit_giant_patch14_dinov2");
  EXPECT_EQ(small.family, "dinov2");
  EXPECT_LT(small.default_parameters, base.default_parameters);
  EXPECT_LT(base.default_parameters, large.default_parameters);
  EXPECT_LT(large.default_parameters, giant.default_parameters);
  EXPECT_EQ(list_models("dinov2").size(), 4U);
}

TEST(Models, DinoV2BaseAndLargeMatchPinnedCapacityContracts) {
  const auto check = [](DinoV2 model, std::int64_t dimension,
                        std::uint64_t parameters) {
    ASSERT_TRUE(model);
    model->eval();
    const auto named = model->named_parameters(true);
    EXPECT_TRUE(named.contains("patch_embed.proj.weight"));
    EXPECT_TRUE(named.contains("blocks.0.attn.qkv.weight"));
    EXPECT_TRUE(named.contains("blocks.0.mlp.fc1.weight"));
    EXPECT_TRUE(named.contains("blocks.0.ls1.gamma"));
    EXPECT_TRUE(named.contains("norm.weight"));
    EXPECT_EQ(parameter_count(*model), parameters);
    const auto tokens =
        model->forward_features(torch::zeros({1, 3, 14, 28}, torch::kFloat32));
    EXPECT_EQ(tokens.sizes(), torch::IntArrayRef({1, 3, dimension}));
    EXPECT_EQ(model->forward(torch::zeros({1, 3, 14, 28})).sizes(),
              torch::IntArrayRef({1, dimension}));
  };

  // Keep the allocations serial: the large variant is a 304M-parameter
  // model and must not be resident while the base variant is checked.
  check(dinov2_vitb14(), 768, 86'579'712U);
  check(dinov2_vitl14(), 1024, 304'367'616U);
}

TEST(Models, DinoV2GiantResourceGateIsExplicit) {
  const auto* enabled = std::getenv("JYPPX_LIUDING_DINOV2_GIANT_GATE");
  if (enabled == nullptr || std::string_view(enabled) != "1")
    GTEST_SKIP() << "Set JYPPX_LIUDING_DINOV2_GIANT_GATE=1 for the serial "
                    "1.136B-parameter resource gate";
  auto model = dinov2_vit_giant14();
  ASSERT_TRUE(model);
  EXPECT_EQ(parameter_count(*model), 1'136'479'232ULL);
  model->eval();
  EXPECT_EQ(model->forward_features(torch::zeros({1, 3, 14, 14})).sizes(),
            torch::IntArrayRef({1, 2, 1536}));
}

TEST(ModelsCuda, DinoV2SmallForwardUsesCudaWhenAvailable) {
  if (!torch::cuda::is_available() || torch::cuda::device_count() == 0) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  const torch::Device device(torch::kCUDA, 0);
  auto model = dinov2_vits14();
  ASSERT_TRUE(model);
  model->to(device);
  model->eval();

  torch::NoGradGuard no_grad;
  const auto input = torch::randn(
      {1, 3, 224, 224},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  const auto tokens = model->forward_features(input);
  const auto cls = model->forward(input);

  EXPECT_TRUE(tokens.device().is_cuda());
  EXPECT_TRUE(cls.device().is_cuda());
  EXPECT_EQ(tokens.sizes(), torch::IntArrayRef({1, 257, 384}));
  EXPECT_EQ(cls.sizes(), torch::IntArrayRef({1, 384}));
  EXPECT_TRUE(torch::isfinite(tokens).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(cls).all().item<bool>());
}

TEST(Models, ResNetVariantsExposeExpectedFeatureStridesAndParameterCounts) {
  auto small = resnet18({.num_classes = 7});
  auto large = resnet50({.num_classes = 7});
  small->eval();
  large->eval();
  const auto input = Images();
  const auto features = small->forward_features(input);
  ASSERT_EQ(features.size(), 4U);
  EXPECT_EQ(features[0].sizes(), torch::IntArrayRef({2, 64, 16, 16}));
  EXPECT_EQ(features[1].sizes(), torch::IntArrayRef({2, 128, 8, 8}));
  EXPECT_EQ(features[2].sizes(), torch::IntArrayRef({2, 256, 4, 4}));
  EXPECT_EQ(features[3].sizes(), torch::IntArrayRef({2, 512, 2, 2}));
  EXPECT_EQ(small->forward(input).sizes(), torch::IntArrayRef({2, 7}));
  EXPECT_EQ(large->forward(input).sizes(), torch::IntArrayRef({2, 7}));
  EXPECT_EQ(parameter_count(*small), 11180103U);
  EXPECT_EQ(parameter_count(*large), 23522375U);
}

TEST(Models, MobileNetVariantsRunAndRejectNonImageContract) {
  auto small = mobilenet_v3_small({.num_classes = 9});
  auto large = mobilenet_v3_large({.num_classes = 9});
  small->eval();
  large->eval();
  EXPECT_EQ(small->forward(Images()).sizes(), torch::IntArrayRef({2, 9}));
  EXPECT_EQ(large->forward(Images()).sizes(), torch::IntArrayRef({2, 9}));
  EXPECT_EQ(parameter_count(*mobilenet_v3_small()), 2542856U);
  EXPECT_EQ(parameter_count(*mobilenet_v3_large()), 5483032U);
  EXPECT_THROW(small->forward(torch::zeros({2, 3, 64})), InvalidArgument);
  EXPECT_THROW(mobilenet_v3_large({.width_multiplier = 0.1}), InvalidArgument);
}

TEST(Models, TransformerHonorsBatchFirstMasksAndSchema) {
  auto model = transformer_encoder({.model_dimension = 16,
                                    .attention_heads = 4,
                                    .feedforward_dimension = 32,
                                    .layers = 1,
                                    .dropout = 0.0,
                                    .gelu = true,
                                    .final_norm = true});
  model->eval();
  const auto input = torch::randn({2, 4, 16});
  const auto attention = torch::zeros({4, 4}, torch::kBool);
  const auto padding = torch::zeros({2, 4}, torch::kBool);
  const auto output = model->forward(input, attention, padding);
  EXPECT_EQ(output.sizes(), torch::IntArrayRef({2, 4, 16}));
  auto adapter = adapt(model);
  EXPECT_EQ(adapter
                .invoke({.positional = {input},
                         .keyword = {{"attention_mask", attention},
                                     {"padding_mask", padding}}})
                .value.toTensor()
                .sizes(),
            torch::IntArrayRef({2, 4, 16}));
  EXPECT_THROW(
      model->forward(input, torch::zeros({4, 4}, torch::kFloat32), padding),
      InvalidArgument);
}

TEST(Models, TransformerMasksExcludeFutureAndPaddedKeys) {
  auto model = transformer_encoder({.model_dimension = 16,
                                    .attention_heads = 4,
                                    .feedforward_dimension = 32,
                                    .layers = 2,
                                    .dropout = 0.0});
  model->eval();
  torch::NoGradGuard no_grad;
  const auto input = torch::randn({2, 4, 16});
  const auto causal = torch::ones({4, 4}, torch::kBool).triu(1);
  const auto baseline = model->forward(input, causal);
  auto future_changed = input.clone();
  future_changed.select(1, 3).copy_(torch::randn({2, 16}) * 8);
  const auto changed = model->forward(future_changed, causal);
  // Each earlier query must be independent of every later key at every layer.
  EXPECT_TRUE(torch::allclose(baseline.narrow(1, 0, 3), changed.narrow(1, 0, 3),
                              1e-5, 1e-6));
  EXPECT_FALSE(torch::allclose(baseline.select(1, 3), changed.select(1, 3)));
  const auto padding = torch::zeros({2, 4}, torch::kBool);
  padding.select(1, 3).fill_(true);
  const auto padded = model->forward(input, {}, padding);
  const auto changed_padded = model->forward(future_changed, {}, padding);
  EXPECT_TRUE(torch::allclose(padded.narrow(1, 0, 3),
                              changed_padded.narrow(1, 0, 3), 1e-5, 1e-6));
  EXPECT_TRUE(torch::equal(causal, torch::ones({4, 4}, torch::kBool).triu(1)));
  EXPECT_TRUE(padding.select(1, 3).all().item<bool>());
}

TEST(Models, MissingUpstreamWeightsRemainExplicitlyUnqualified) {
  const auto records = available_weights("resnet18");
  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records.front().availability, WeightAvailability::not_provisioned);
  EXPECT_FALSE(records.front().artifact.has_value());
  EXPECT_EQ(records.front().license_spdx, "NOASSERTION");
  validate_weight_record(records.front());
  EXPECT_NE(models_preview_manifest_json().find("not-provisioned"),
            std::string::npos);
  EXPECT_THROW(verify_weights(records.front(), std::span<const std::byte>{}),
               NotReady);
  const auto dino = available_weights("vit_small_patch14_dinov2");
  ASSERT_EQ(dino.size(), 1U);
  EXPECT_EQ(dino.front().id, "vit_small_patch14_dinov2.lvd142m");
  EXPECT_EQ(dino.front().availability, WeightAvailability::not_provisioned);
  EXPECT_FALSE(dino.front().artifact.has_value());
  EXPECT_EQ(dino.front().license_spdx, "NOASSERTION");
  EXPECT_NE(models_preview_manifest_json().find("vit_giant_patch14_dinov2"),
            std::string::npos);
}

struct TinyModel final : torch::nn::Module {
  torch::Tensor weight =
      register_parameter("weight", torch::zeros({3}, torch::kFloat32));
  Tensor forward(const Tensor& input) { return input + weight; }
};

TEST(Models, PinnedSafetensorsWeightsVerifyImportAndCacheByDigest) {
  const std::array<float, 3> values{1.0F, 2.0F, 3.0F};
  const auto bytes =
      interop::serialize_safetensors(std::vector<interop::SafeTensorSource>{
          {.name = "weight",
           .dtype = interop::SafeTensorDtype::f32,
           .shape = {3U},
           .data = std::as_bytes(std::span(values))}});
  const auto digest = ArtifactDigest{.hex = sha256_hex(bytes)};
  const WeightRecord record{
      .id = "tiny-v1",
      .model_name = "tiny",
      .source_uri = "https://example.invalid/tiny-v1.safetensors",
      .license_spdx = "Apache-2.0",
      .license_uri = "https://example.invalid/license",
      .availability = WeightAvailability::pinned_external,
      .artifact = ArtifactRef{.digest = digest,
                              .size_bytes = bytes.size(),
                              .logical_name = "tiny-v1.safetensors",
                              .media_type = "application/x-safetensors",
                              .producer = "models-test"},
      .status_reason = "test artifact"};
  auto module = std::make_shared<TinyModel>();
  auto adapter = nn::adapt_tensor_module(module);
  const auto imported = load_weights(adapter, record, bytes);
  EXPECT_TRUE(imported.applied);
  EXPECT_TRUE(torch::equal(module->weight, torch::tensor({1.0F, 2.0F, 3.0F})));

  // Exercise the public adapter boundary after the imported state is applied.
  // This keeps the verified artifact path tied to an actual model invocation,
  // rather than only checking the module parameter tensor directly.
  const auto prediction = adapter.invoke(
      {.positional = {torch::zeros({3}, torch::kFloat32)}, .keyword = {}});
  ASSERT_TRUE(prediction.value.isTensor());
  EXPECT_TRUE(torch::equal(prediction.value.toTensor(),
                           torch::tensor({1.0F, 2.0F, 3.0F})));

  const auto cache =
      std::filesystem::temp_directory_path() /
      ("jyppx-model-cache-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::size_t fetches = 0U;
  const auto path =
      cache_weights(record, cache, [&](const WeightRecord&, std::uint64_t) {
        ++fetches;
        return bytes;
      });
  EXPECT_TRUE(std::filesystem::is_regular_file(path));
  EXPECT_EQ(cache_weights(record, cache,
                          [&](const WeightRecord&, std::uint64_t) {
                            ++fetches;
                            return bytes;
                          }),
            path);
  EXPECT_EQ(fetches, 1U);
  std::error_code ignored;
  std::filesystem::remove_all(cache, ignored);
}

#ifdef JYPPX_LIUDING_MODELS_PYTHON_INTEROP
struct ReferenceFileIdentity {
  std::string_view filename;
  std::string_view sha256;
  std::uint64_t size_bytes;
};
// Fixed reviewed reference bytes, independent of runtime evidence JSON.
constexpr std::array<ReferenceFileIdentity, 27> kReferenceFiles{{
    {"mobilenet_v3_large-input.safetensors",
     "c33bce57a419a3b79e14fbae93c9bcba4514e1d2211aeb88c2ca02e1446054b3",
     98384ULL},
    {"mobilenet_v3_large-output.safetensors",
     "b1cfef0ea201bcf7472e4d3054dea3bb4702a572637cc8ab5bf8a4655eeaacfc",
     8080ULL},
    {"mobilenet_v3_large-state.safetensors",
     "c28d8eb4d0437515d69501c03d4e5da19d482ea9ec1efb86a0b0f98d8c0f5404",
     22060512ULL},
    {"mobilenet_v3_small-input.safetensors",
     "56c82c0b3e6009ac07360ad32021b6e828cc02e39ba8e7ab79c47695a2dc60e6",
     98384ULL},
    {"mobilenet_v3_small-output.safetensors",
     "cb8b1630ddc5ac331729390377c54707bc3caa05b562f394a583e73b144249cb",
     8080ULL},
    {"mobilenet_v3_small-state.safetensors",
     "5414156665cfec12dcafe53f60d12359e8b0c0dc15703ee32a7af53424dbdca1",
     10243416ULL},
    {"resnet18-input.safetensors",
     "08b128164c087413e272a25af748608e2162069beae422c491087a88f5c0d00c",
     98384ULL},
    {"resnet18-output.safetensors",
     "4ceeeb2cc2e00405ebdec14ecdab4311635d2aa91c056dc3700d87fb5733d92d",
     8080ULL},
    {"resnet18-state.safetensors",
     "1d4ab66c412050ef8544eab4b3df51c38327008309aca10103f0fc1b1a61cd1a",
     46807448ULL},
    {"resnet50-input.safetensors",
     "1195f7fc72b2f5a4fef921f08ff6c63debf5adfa0100d8c2bfdbeabab50af15f",
     98384ULL},
    {"resnet50-output.safetensors",
     "46e6de1ef6c566128e72884781503295f53edaf6ff014a9b44f4c43a7db2c4de",
     8080ULL},
    {"resnet50-state.safetensors",
     "9db254eff7e63cee123e56c0bc858def81cfc099095fe86d72639c873ee805a4",
     102469840ULL},
    {"transformer_encoder-input.safetensors",
     "48e5989ff42ba42d9ad2bdd996c27710758ca7dd8ba424161c4b0e28ab8f9455",
     20747ULL},
    {"transformer_encoder-output.safetensors",
     "9cb1e2e1c1ac0edc186897da83335415aab3669ede4157c847526f13e36e0223",
     20560ULL},
    {"transformer_encoder-state.safetensors",
     "be4d41ef9fd15d7c3345a32d04c12d524a730a555cb3d329323bad490de461f2",
     75668848ULL},
    {"vit_small_patch14_dinov2-input.safetensors",
     "0ba7d144d17e1d8a0d97494357b80d8d16d7448339e7755b2b21d45d767a4632",
     1505368ULL},
    {"vit_small_patch14_dinov2-output.safetensors",
     "c8347532d3e7ae4b5328a889d4ce18f9a20f86c13197d96beb52e456bf46730a",
     989336ULL},
    {"vit_small_patch14_dinov2-state.safetensors",
     "cd1aa20613acd6d68210f84d170e1c3b5c16d67968c10c8918881a682f2971a9",
     88240512ULL},
    {"vit_base_patch14_dinov2-input.safetensors",
     "48b75b936e2b87cca3cfcf4e76f8e21427b2a7461dfa9c6e3c913286c9949242",
     1505368ULL},
    {"vit_base_patch14_dinov2-output.safetensors",
     "c9d11866fe58e00b42f3541f69f33af5aa22f968820c1879b7fd0b96901c9687",
     1978520ULL},
    {"vit_base_patch14_dinov2-state.safetensors",
     "2941c91046ca4ae4cc3c1d2526b01863e9444383dbea1eb0a7d78b32e7d55e9b",
     346334872ULL},
    {"vit_large_patch14_dinov2-input.safetensors",
     "dc67edfd1e67541e68115fd914b8fba35cdec0375d69b42f9ead25b3e26cb4fc",
     1505368ULL},
    {"vit_large_patch14_dinov2-output.safetensors",
     "1dfd3ae6fe886d27593820ee0fb5d2702ae61527a65b97ca5e55c7a1b6d51b00",
     2637976ULL},
    {"vit_large_patch14_dinov2-state.safetensors",
     "0de163cfe892194462c39ab65c4be7ebdcc85ec7dca9858c51b24a6cfc8557ff",
     1217502752ULL},
    {"vit_giant_patch14_dinov2-input.safetensors",
     "2839e43d3b6f84207e66b08628e65a9b05e5d84c0be27f50a87f20ce691de698",
     1505368ULL},
    {"vit_giant_patch14_dinov2-output.safetensors",
     "7c9b742fe3556340c81a4f43437a2a971ca0fc53bb64d6d8e3e1bbb3d3ef1cd4",
     3956896ULL},
    {"vit_giant_patch14_dinov2-state.safetensors",
     "b40769853a079205711aab6ecb7867ad45d9c3e35d5856e189cb321d21a55027",
     4545971216ULL},
}};

test::MappedReference ReadReference(const std::filesystem::path& path) {
  const auto filename = path.filename().string();
  const auto identity =
      std::find_if(kReferenceFiles.begin(), kReferenceFiles.end(),
                   [&](const auto& item) { return item.filename == filename; });
  if (identity == kReferenceFiles.end() ||
      std::filesystem::file_size(path) != identity->size_bytes)
    throw std::runtime_error("models reference file identity/size mismatch");
  test::MappedReference encoded(path, identity->size_bytes);
  if (sha256_hex(encoded) != identity->sha256)
    throw std::runtime_error("models reference file fixed SHA-256 mismatch");
  return encoded;
}

torch::Tensor ReferenceTensor(std::span<const std::byte> encoded,
                              std::string_view name) {
  const interop::SafeTensorsView view(encoded);
  const auto& info = view.directory().tensors;
  const auto found =
      std::find_if(info.begin(), info.end(),
                   [&](const auto& item) { return item.name == name; });
  if (found == info.end() ||
      (found->dtype != interop::SafeTensorDtype::f32 &&
       found->dtype != interop::SafeTensorDtype::boolean))
    throw std::runtime_error("reference tensor is not F32 or BOOL");
  const auto bytes = view.tensor_bytes(name);
  std::vector<std::int64_t> shape;
  shape.reserve(found->shape.size());
  for (const auto extent : found->shape)
    shape.push_back(static_cast<std::int64_t>(extent));
  if (found->dtype == interop::SafeTensorDtype::boolean) {
    std::vector<std::uint8_t> values(bytes.size());
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return torch::from_blob(values.data(), shape, torch::kBool).clone();
  }
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return torch::from_blob(values.data(), shape, torch::kFloat32).clone();
}

void CheckArchitectureReference(const char* architecture) {
  const auto* directory = std::getenv("JYPPX_LIUDING_MODELS_REFERENCE_DIR");
  if (directory == nullptr || *directory == '\0') {
    GTEST_SKIP()
        << "Set JYPPX_LIUDING_MODELS_REFERENCE_DIR for the independent "
           "official reference gate";
  }
  const auto root = std::filesystem::path(directory);
  const auto prefix = std::string(architecture);
  const auto state = ReadReference(root / (prefix + "-state.safetensors"));
  const auto input = ReadReference(root / (prefix + "-input.safetensors"));
  const auto expected = ReadReference(root / (prefix + "-output.safetensors"));
  const auto digest = ArtifactDigest{.hex = sha256_hex(state)};
  const WeightRecord record{
      .id = "reference-" + prefix + "-random-v1",
      .model_name = prefix,
      .source_uri =
          "fixture://official-random-initialized-architecture/" + prefix,
      .license_spdx = "LicenseRef-Generated-Test-Fixture",
      .license_uri = "fixture://models-reference/local-generated-state",
      .availability = WeightAvailability::pinned_external,
      .artifact = ArtifactRef{.digest = digest,
                              .size_bytes = state.size(),
                              .logical_name = prefix + "-state.safetensors",
                              .media_type = "application/x-safetensors",
                              .producer = "models-python-reference"},
      .status_reason =
          "Random architecture differential; no pretrained weights."};
  ModelOptions options;
  options.transformer.dropout = 0.0;
  auto adapter = create_model(prefix, options);
  const auto imported = load_weights(adapter, record, state);
  ASSERT_TRUE(imported.applied) << prefix;
  adapter.state().eval();
  torch::NoGradGuard no_grad;
  nn::Invocation invocation{.positional = {ReferenceTensor(input, "input")},
                            .keyword = {}};
  if (prefix == "transformer_encoder") {
    invocation.keyword.emplace("attention_mask",
                               ReferenceTensor(input, "attention_mask"));
    invocation.keyword.emplace("padding_mask",
                               ReferenceTensor(input, "padding_mask"));
  }
  const auto actual = adapter.invoke(invocation).value.toTensor();
  const auto reference = ReferenceTensor(expected, "output");
  ASSERT_EQ(actual.sizes(), reference.sizes()) << prefix;
  ASSERT_TRUE(torch::isfinite(actual).all().item<bool>()) << prefix;
  ASSERT_TRUE(torch::isfinite(reference).all().item<bool>()) << prefix;
  const auto scale = reference.abs().max().item<double>();
  ASSERT_GT(scale, 0.0) << prefix;
  // Normalize tiny random MobileNet logits before applying the common
  // tolerance. A raw 1e-5 absolute tolerance could accept an all-zero incorrect
  // classifier.
  EXPECT_TRUE(torch::allclose(actual / scale, reference / scale, 1e-4, 1e-5))
      << prefix << " normalized maximum absolute error: "
      << ((actual - reference) / scale).abs().max().item<double>();
}

TEST(ModelsPythonInterop, OfficialTorchvisionResNet18ReferenceReallyRuns) {
  CheckArchitectureReference("resnet18");
}
TEST(ModelsPythonInterop, OfficialTorchvisionResNet50ReferenceReallyRuns) {
  CheckArchitectureReference("resnet50");
}
TEST(ModelsPythonInterop,
     OfficialTorchvisionMobileNetV3SmallReferenceReallyRuns) {
  CheckArchitectureReference("mobilenet_v3_small");
}
TEST(ModelsPythonInterop,
     OfficialTorchvisionMobileNetV3LargeReferenceReallyRuns) {
  CheckArchitectureReference("mobilenet_v3_large");
}
TEST(ModelsPythonInterop, OfficialTransformerEncoderReferenceReallyRuns) {
  CheckArchitectureReference("transformer_encoder");
}

void CheckDinoV2Reference(const char* prefix, DinoV2 model,
                          std::uint64_t staging_bytes) {
  const auto* directory = std::getenv("JYPPX_LIUDING_MODELS_REFERENCE_DIR");
  if (directory == nullptr || *directory == '\0') {
    GTEST_SKIP()
        << "Set JYPPX_LIUDING_MODELS_REFERENCE_DIR for the independent "
           "timm DINOv2 reference gate";
  }
  const auto root = std::filesystem::path(directory);
  const auto state =
      ReadReference(root / (std::string(prefix) + "-state.safetensors"));
  const auto input =
      ReadReference(root / (std::string(prefix) + "-input.safetensors"));
  const auto expected =
      ReadReference(root / (std::string(prefix) + "-output.safetensors"));
  const WeightRecord record{
      .id = "reference-" + std::string(prefix) + "-random-v1",
      .model_name = prefix,
      .source_uri = "fixture://timm/af97f99efef62bd9bd7eaa908fffe5b7a2d98e2c/" +
                    std::string(prefix),
      .license_spdx = "LicenseRef-Generated-Test-Fixture",
      .license_uri = "fixture://models-reference/local-generated-state",
      .availability = WeightAvailability::pinned_external,
      .artifact =
          ArtifactRef{
              .digest = ArtifactDigest{.hex = sha256_hex(state)},
              .size_bytes = state.size(),
              .logical_name = std::string(prefix) + "-state.safetensors",
              .media_type = "application/x-safetensors",
              .producer = "timm-python-reference"},
      .status_reason =
          "Random timm architecture differential; no pretrained weights."};
  auto adapter = adapt(model);
  interop::ModelWeightImportOptions import_options;
  import_options.max_staging_bytes = staging_bytes;
  const auto imported = load_weights(adapter, record, state, import_options);
  ASSERT_TRUE(imported.applied);
  adapter.state().eval();
  torch::NoGradGuard no_grad;
  const auto image = ReferenceTensor(input, "input");
  const auto actual_tokens = model->forward_features(image);
  const auto actual_cls = model->forward(image);
  const auto reference_tokens = ReferenceTensor(expected, "tokens");
  const auto reference_cls = ReferenceTensor(expected, "output");
  ASSERT_EQ(actual_tokens.sizes(), reference_tokens.sizes());
  ASSERT_EQ(actual_cls.sizes(), reference_cls.sizes());
  const auto scale = reference_tokens.abs().max().item<double>();
  ASSERT_GT(scale, 0.0);
  EXPECT_TRUE(torch::allclose(actual_tokens / scale, reference_tokens / scale,
                              2e-4, 2e-5));
  EXPECT_TRUE(torch::allclose(actual_cls, reference_cls, 2e-4, 2e-5));
}

TEST(ModelsPythonInterop, OfficialTimmDinoV2SmallReferenceReallyRuns) {
  CheckDinoV2Reference("vit_small_patch14_dinov2", dinov2_vits14(),
                       512ULL << 20U);
}

TEST(ModelsPythonInterop, OfficialTimmDinoV2BaseReferenceReallyRuns) {
  CheckDinoV2Reference("vit_base_patch14_dinov2", dinov2_vitb14(), 2ULL << 30U);
}

TEST(ModelsPythonInterop, OfficialTimmDinoV2LargeReferenceReallyRuns) {
  CheckDinoV2Reference("vit_large_patch14_dinov2", dinov2_vitl14(),
                       6ULL << 30U);
}

TEST(ModelsPythonInterop, OfficialTimmDinoV2GiantReferenceReallyRuns) {
  const auto* enabled =
      std::getenv("JYPPX_LIUDING_DINOV2_GIANT_REFERENCE_GATE");
  if (enabled == nullptr || std::string_view(enabled) != "1")
    GTEST_SKIP() << "Set JYPPX_LIUDING_DINOV2_GIANT_REFERENCE_GATE=1 for the "
                    "serial DINOv2 Giant reference gate";
  CheckDinoV2Reference("vit_giant_patch14_dinov2", dinov2_vit_giant14(),
                       16ULL << 30U);
}

#endif

}  // namespace

// Configure the test process before any LibTorch parallel work starts.
int main(int argc, char** argv) {
  torch::set_num_threads(2);
  torch::set_num_interop_threads(1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
