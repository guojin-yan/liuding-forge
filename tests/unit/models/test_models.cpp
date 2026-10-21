// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");

#include <jyppx/foundation/sha256.hpp>
#include <jyppx/interop/safetensors.hpp>
#include <jyppx/models/models.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace jyppx;
using namespace jyppx::models;

Tensor Images(std::int64_t channels = 3) {
  return torch::zeros({2, channels, 64, 64}, torch::kFloat32);
}

TEST(Models, RegistryListsFivePreviewArchitecturesAndCreatesAdapters) {
  const auto entries = list_models();
  ASSERT_EQ(entries.size(), 5U);
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

#ifdef JYPPX_MODELS_PYTHON_INTEROP
struct ReferenceFileIdentity {
  std::string_view filename;
  std::string_view sha256;
  std::uint64_t size_bytes;
};
// Fixed reviewed reference bytes, independent of runtime evidence JSON.
constexpr std::array<ReferenceFileIdentity, 15> kReferenceFiles{{
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
}};

std::vector<std::byte> ReadReference(const std::filesystem::path& path) {
  const auto filename = path.filename().string();
  const auto identity =
      std::find_if(kReferenceFiles.begin(), kReferenceFiles.end(),
                   [&](const auto& item) { return item.filename == filename; });
  if (identity == kReferenceFiles.end() ||
      std::filesystem::file_size(path) != identity->size_bytes)
    throw std::runtime_error("models reference file identity/size mismatch");
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input) throw std::runtime_error("cannot open models reference artifact");
  std::vector<std::byte> encoded(
      static_cast<std::size_t>(identity->size_bytes));
  input.read(reinterpret_cast<char*>(encoded.data()),
             static_cast<std::streamsize>(encoded.size()));
  if (!input || input.peek() != std::char_traits<char>::eof() ||
      sha256_hex(encoded) != identity->sha256)
    throw std::runtime_error("models reference file fixed SHA-256 mismatch");
  return encoded;
}

torch::Tensor ReferenceTensor(const std::vector<std::byte>& encoded,
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
  const auto* directory = std::getenv("JYPPX_MODELS_REFERENCE_DIR");
  if (directory == nullptr || *directory == '\0') {
    GTEST_SKIP() << "Set JYPPX_MODELS_REFERENCE_DIR for the independent "
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

#endif

}  // namespace

// Configure the test process before any LibTorch parallel work starts.
int main(int argc, char** argv) {
  torch::set_num_threads(2);
  torch::set_num_interop_threads(1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
