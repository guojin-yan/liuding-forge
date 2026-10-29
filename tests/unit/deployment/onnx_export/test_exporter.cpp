// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#ifdef JYPPX_LIUDING_TEST_ONNX_MODELS
#include <jyppx/liuding/models/vision.hpp>
#endif

#ifdef JYPPX_LIUDING_TEST_ONNX_RUNTIME
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#endif

#include <torch/csrc/jit/ir/ir.h>
#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace jyppx::liuding;
namespace exporter = jyppx::liuding::onnx_export;
using Json = nlohmann::json;
using Cases = std::vector<std::pair<torch::Tensor, torch::Tensor>>;

exporter::ExportOptions TraceOptions(const std::string& name) {
  exporter::ExportOptions options;
  const auto* directory = std::getenv("JYPPX_LIUDING_ONNX_EXPORT_EVIDENCE_DIR");
  if (directory != nullptr && *directory != '\0') {
    const auto path =
        std::filesystem::u8path(directory) / (name + "-raw-trace.txt");
    std::filesystem::create_directories(path.parent_path());
    options.trace_observer = [path](const torch::jit::Graph& graph) {
      std::ofstream stream(path);
      stream << graph.toString();
      if (!stream) throw std::runtime_error("Cannot preserve raw native trace");
    };
  }
  return options;
}

class NovelNetworkImpl : public torch::nn::Module {
 public:
  NovelNetworkImpl() {
    first = register_module("first", torch::nn::Linear(8, 5));
    second = register_module("second", torch::nn::Linear(5, 3));
  }
  torch::Tensor forward(const torch::Tensor& input) {
    return torch::tanh(second(torch::relu(first(input))));
  }
  torch::nn::Linear first{nullptr}, second{nullptr};
};
TORCH_MODULE(NovelNetwork);

class PairNetworkImpl : public torch::nn::Module {
 public:
  PairNetworkImpl() {
    layer = register_module("layer", torch::nn::Linear(8, 3));
  }
  std::vector<torch::Tensor> forward(std::span<const torch::Tensor> inputs) {
    return {layer(inputs[0] + inputs[1]), layer(inputs[0] - inputs[1])};
  }
  torch::nn::Linear layer{nullptr};
};
TORCH_MODULE(PairNetwork);

void Write(const std::filesystem::path& path, std::span<const std::byte> data) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
  if (!stream)
    throw std::runtime_error("Cannot write native ONNX test artifact");
}

Json EvidenceTensor(const std::filesystem::path& root, const std::string& name,
                    const torch::Tensor& tensor) {
  const auto bytes =
      std::as_bytes(std::span(tensor.const_data_ptr<float>(),
                              static_cast<std::size_t>(tensor.numel())));
  Write(root / name, bytes);
  return {{"path", name},
          {"sha256", sha256_hex(bytes)},
          {"shape", std::vector<std::int64_t>(tensor.sizes().begin(),
                                              tensor.sizes().end())},
          {"dtype", "float32"},
          {"byte_order", "little"}};
}

void Evidence(const char* name, const exporter::ExportResult& result,
              const Cases& cases) {
  const auto* directory = std::getenv("JYPPX_LIUDING_ONNX_EXPORT_EVIDENCE_DIR");
  if (directory == nullptr || *directory == '\0') return;
  const auto root = std::filesystem::u8path(directory) / name;
  std::filesystem::create_directories(root);
  Write(root / "model.onnx", result.onnx);
  Json evidence{{"schema_version", 1},
                {"production_exporter", "native-cpp"},
                {"model", name},
                {"onnx_sha256", result.digest.hex},
                {"opset", result.opset},
                {"external_data", result.external_data},
                {"operators", result.operators},
                {"cases", Json::array()}};
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto prefix = "case-" + std::to_string(index);
    evidence["cases"].push_back(
        {{"input",
          EvidenceTensor(root, prefix + "-input.f32", cases[index].first)},
         {"native_output",
          EvidenceTensor(root, prefix + "-output.f32", cases[index].second)}});
  }
  const auto encoded = evidence.dump(2);
  Write(root / "native-evidence.json",
        std::as_bytes(std::span(encoded.data(), encoded.size())));
}

void Identity(const exporter::ExportResult& result, exporter::ModelKind kind,
              const std::vector<std::int64_t>& input,
              const std::vector<std::int64_t>& output) {
  ASSERT_FALSE(result.onnx.empty());
  EXPECT_EQ(result.digest.hex, sha256_hex(result.onnx));
  EXPECT_EQ(result.model, kind);
  EXPECT_EQ(result.opset, 17U);
  EXPECT_FALSE(result.external_data);
  EXPECT_EQ(result.input_shape, input);
  EXPECT_EQ(result.output_shape, output);
  ASSERT_TRUE(std::is_sorted(result.operators.begin(), result.operators.end()));
  for (const auto& name : result.operators) EXPECT_EQ(name.find("onnx::"), 0U);
}

std::map<std::string, torch::Tensor> State(const torch::nn::Module& model) {
  std::map<std::string, torch::Tensor> values;
  for (const auto& entry : model.named_parameters())
    values.emplace(entry.key(), entry.value().detach().clone());
  for (const auto& entry : model.named_buffers())
    values.emplace(entry.key(), entry.value().detach().clone());
  return values;
}

void SameState(const torch::nn::Module& model,
               const std::map<std::string, torch::Tensor>& before) {
  const auto after = State(model);
  ASSERT_EQ(after.size(), before.size());
  for (const auto& [name, value] : before) {
    ASSERT_TRUE(after.contains(name));
    EXPECT_TRUE(torch::equal(value, after.at(name))) << name;
  }
}

TEST(NativeOnnxExport, LinearUsesExactCurrentStateAndOfficialSerializer) {
  at::set_num_threads(1);
  torch::NoGradGuard guard;
  torch::nn::Linear model(8, 3);
  model->weight.copy_(torch::arange(24, torch::kFloat32).reshape({3, 8}) /
                      32.0);
  model->bias.copy_(torch::tensor({-0.25F, 0.5F, 1.25F}));
  const std::array<std::int64_t, 2> shape{2, 8};
  const auto result =
      exporter::export_linear(model, shape, TraceOptions("linear"));
  Identity(result, exporter::ModelKind::linear, {2, 8}, {2, 3});
  EXPECT_EQ(result.operators, (std::vector<std::string>{"onnx::Gemm"}));
  Cases cases;
  for (const auto batch : {2, 5}) {
    auto input =
        torch::arange(batch * 8, torch::kFloat32).reshape({batch, 8}) / 16.0 -
        1.0;
    cases.emplace_back(input.contiguous(), model->forward(input).contiguous());
  }
  Evidence("linear", result, cases);
  model->bias.add_(1.0);
  EXPECT_NE(exporter::export_linear(model, shape).digest, result.digest);
}

TEST(NativeOnnxExport, LinearSupportsBiaslessAndFixedBatchProfiles) {
  torch::nn::Linear model(torch::nn::LinearOptions(8, 3).bias(false));
  const std::array<std::int64_t, 2> shape{2, 8};
  exporter::ExportOptions options;
  options.dynamic_batch = false;
  const auto result = exporter::export_linear(model, shape, options);
  Identity(result, exporter::ModelKind::linear, {2, 8}, {2, 3});
}

TEST(NativeOnnxExport, PreservesModeParametersGradientsAndRng) {
  torch::nn::Linear model(8, 3);
  model->train();
  model->weight.mutable_grad() = torch::ones_like(model->weight);
  const auto state = State(*model);
  const auto gradient = model->weight.grad().clone();
  torch::manual_seed(422);
  const auto expected = torch::randn({4});
  torch::manual_seed(422);
  const std::array<std::int64_t, 2> shape{2, 8};
  (void)exporter::export_linear(model, shape);
  EXPECT_TRUE(model->is_training());
  EXPECT_TRUE(model->weight.requires_grad());
  EXPECT_TRUE(torch::equal(gradient, model->weight.grad()));
  EXPECT_TRUE(torch::equal(expected, torch::randn({4})));
  SameState(*model, state);
}

TEST(NativeOnnxExport, RejectsUnsupportedProfilesAndByteBounds) {
  torch::nn::Linear model(8, 3);
  const std::array<std::int64_t, 2> shape{2, 8};
  exporter::ExportOptions options;
  options.opset = 18;
  EXPECT_THROW(exporter::export_linear(model, shape, options), InvalidArgument);
  options = {};
  options.maximum_batch = 1;
  EXPECT_THROW(exporter::export_linear(model, shape, options), InvalidArgument);
  options = {};
  options.minimum_batch = 0;
  EXPECT_THROW(exporter::export_linear(model, shape, options), InvalidArgument);
  options = {};
  options.max_output_bytes = 64;
  EXPECT_THROW(exporter::export_linear(model, shape, options),
               ResourceExhausted);
  options = {};
  options.input_names = {"first", "unexpected"};
  EXPECT_THROW(exporter::export_linear(model, shape, options), InvalidArgument);
  const std::array<std::int64_t, 2> wrong_width{2, 7};
  EXPECT_THROW(exporter::export_linear(model, wrong_width), CompatibilityError);
  torch::nn::Linear empty(nullptr);
  EXPECT_THROW(exporter::export_linear(empty, shape), InvalidArgument);
}

TEST(NativeOnnxExport, RejectsUnsupportedOrNonfiniteLinearState) {
  const std::array<std::int64_t, 2> shape{2, 8};
  torch::nn::Linear complex(8, 3);
  complex->to(torch::kComplexFloat);
  EXPECT_THROW(exporter::export_linear(complex, shape), CompatibilityError);
  torch::nn::Linear nonfinite(8, 3);
  {
    torch::NoGradGuard guard;
    nonfinite->weight[0][0] = std::numeric_limits<float>::quiet_NaN();
  }
  EXPECT_THROW(exporter::export_linear(nonfinite, shape), CompatibilityError);
  torch::nn::Linear extra(8, 3);
  extra->register_buffer("unreviewed", torch::zeros({1}));
  EXPECT_NO_THROW(exporter::export_linear(extra, shape));
}

#ifdef JYPPX_LIUDING_TEST_ONNX_MODELS
TEST(NativeOnnxExport, ResNet18UsesTheSameGenericGraphFrontend) {
  at::set_num_threads(1);
  torch::manual_seed(1730);
  auto model = models::resnet18({.num_classes = 7});
  model->eval();
  const auto state = State(*model);
  const std::array<std::int64_t, 4> shape{2, 3, 224, 224};
  const auto result =
      exporter::export_resnet18(model, shape, TraceOptions("resnet18"));
  Identity(result, exporter::ModelKind::resnet18, {2, 3, 224, 224}, {2, 7});
  EXPECT_NE(
      std::find(result.operators.begin(), result.operators.end(), "onnx::Conv"),
      result.operators.end());
  EXPECT_FALSE(model->is_training());
  SameState(*model, state);
  torch::NoGradGuard guard;
  Cases cases;
  for (const auto batch : {2, 5}) {
    auto input = torch::randn({batch, 3, 224, 224});
    cases.emplace_back(input.contiguous(), model->forward(input).contiguous());
  }
  Evidence("resnet18", result, cases);
}
#endif

TEST(NativeOnnxExport, ExportsUnseenUserDefinedModuleWithoutModelRegistration) {
  at::set_num_threads(1);
  torch::NoGradGuard guard;
  NovelNetwork model;
  model->eval();
  const auto input = torch::randn({2, 8});
  const auto result =
      exporter::export_model(model, input, TraceOptions("novel"));
  Identity(result, exporter::ModelKind::generic, {2, 8}, {2, 3});
  Cases cases;
  for (const auto batch : {2, 5}) {
    auto values = torch::randn({batch, 8});
    cases.emplace_back(values, model->forward(values).contiguous());
  }
  Evidence("novel", result, cases);
}

TEST(NativeOnnxExport,
     SupportsMultipleInputsAndOutputsWithExplicitNamesAndAxes) {
  torch::NoGradGuard guard;
  PairNetwork model;
  const std::vector<torch::Tensor> inputs{torch::randn({2, 8}),
                                          torch::randn({2, 8})};
  auto options = TraceOptions("pair");
  options.input_names = {"left", "right"};
  options.output_names = {"sum_logits", "difference_logits"};
  options.dynamic_axes = {{"left", {{0, "batch"}}},
                          {"right", {{0, "batch"}}},
                          {"sum_logits", {{0, "batch"}}},
                          {"difference_logits", {{0, "batch"}}}};
  const auto result = exporter::export_module(
      *model,
      [&](std::span<const torch::Tensor> values) {
        return model->forward(values);
      },
      inputs, options);
  EXPECT_EQ(result.input_names, options.input_names);
  EXPECT_EQ(result.output_names, options.output_names);
  EXPECT_EQ(result.input_shapes.size(), 2U);
  EXPECT_EQ(result.output_shapes.size(), 2U);
  const auto* directory = std::getenv("JYPPX_LIUDING_ONNX_EXPORT_EVIDENCE_DIR");
  if (directory != nullptr && *directory != '\0') {
    const auto root = std::filesystem::u8path(directory) / "pair";
    std::filesystem::create_directories(root);
    Write(root / "model.onnx", result.onnx);
    Json evidence{{"schema_version", 1},
                  {"production_exporter", "native-cpp"},
                  {"model", "pair"},
                  {"onnx_sha256", result.digest.hex},
                  {"operators", result.operators},
                  {"opset", result.opset},
                  {"external_data", false},
                  {"cases", Json::array()}};
    for (const auto batch : {2, 5}) {
      const std::vector<torch::Tensor> values{torch::randn({batch, 8}),
                                              torch::randn({batch, 8})};
      const auto outputs = model->forward(values);
      const auto prefix = "batch-" + std::to_string(batch);
      evidence["cases"].push_back(
          {{"inputs",
            {EvidenceTensor(root, prefix + "-left.f32", values[0]),
             EvidenceTensor(root, prefix + "-right.f32", values[1])}},
           {"native_outputs",
            {EvidenceTensor(root, prefix + "-sum.f32", outputs[0]),
             EvidenceTensor(root, prefix + "-difference.f32", outputs[1])}}});
    }
    const auto text = evidence.dump(2);
    Write(root / "native-evidence.json",
          std::as_bytes(std::span(text.data(), text.size())));
  }
}

TEST(NativeOnnxExport,
     ReportsUnsupportedOperationsAndAcceptsExplicitCustomLowering) {
  torch::nn::Module module;
  const std::vector<torch::Tensor> examples{torch::ones({2, 8})};
  try {
    (void)exporter::export_module(
        module,
        [](std::span<const torch::Tensor> inputs) {
          return std::vector<torch::Tensor>{torch::lgamma(inputs.front())};
        },
        examples);
    FAIL() << "Unsupported operation unexpectedly exported";
  } catch (const CompatibilityError& error) {
    EXPECT_NE(std::string(error.what()).find("lgamma"), std::string::npos);
  }
  auto options = TraceOptions("custom-sin");
  int calls = 0;
  options.custom_lowerings["aten::sin"] = [&](torch::jit::Node& node,
                                              torch::jit::Graph& graph) {
    ++calls;
    auto* replacement =
        graph.create(c10::Symbol::onnx("Sin"), {node.input()}, 1);
    replacement->output()->setType(node.output()->type());
    replacement->insertBefore(&node);
    return std::vector<torch::jit::Value*>{replacement->output()};
  };
  const auto result = exporter::export_module(
      module,
      [](std::span<const torch::Tensor> inputs) {
        return std::vector<torch::Tensor>{torch::sin(inputs.front())};
      },
      examples, options);
  EXPECT_EQ(calls, 1);
  EXPECT_NE(
      std::find(result.operators.begin(), result.operators.end(), "onnx::Sin"),
      result.operators.end());
}

TEST(NativeOnnxExport,
     RejectsBatchDependentNativeControlFlowAndShapeConstants) {
  torch::nn::Module module;
  const std::vector<torch::Tensor> examples{torch::ones({2, 8})};
  EXPECT_THROW(exporter::export_module(
                   module,
                   [](std::span<const torch::Tensor> inputs) {
                     return std::vector<torch::Tensor>{
                         inputs.front().reshape({inputs.front().size(0), 8})};
                   },
                   examples),
               CompatibilityError);
  EXPECT_THROW(exporter::export_module(
                   module,
                   [](std::span<const torch::Tensor> inputs) {
                     return std::vector<torch::Tensor>{
                         inputs.front().size(0) == 2
                             ? torch::relu(inputs.front())
                             : torch::sigmoid(inputs.front())};
                   },
                   examples),
               CompatibilityError);
}

#ifdef JYPPX_LIUDING_TEST_ONNX_RUNTIME
namespace runtime = jyppx::liuding::onnx_runtime;

runtime::TensorOutput RunSession(const runtime::Session& session,
                                 const torch::Tensor& input) {
  return session.run(std::span(input.const_data_ptr<float>(),
                               static_cast<std::size_t>(input.numel())),
                     std::span(input.sizes().data(), input.sizes().size()));
}

void Near(const runtime::TensorOutput& actual, const torch::Tensor& expected) {
  ASSERT_EQ(actual.shape, std::vector<std::int64_t>(expected.sizes().begin(),
                                                    expected.sizes().end()));
  ASSERT_EQ(actual.values.size(), static_cast<std::size_t>(expected.numel()));
  const auto* values = expected.const_data_ptr<float>();
  for (std::size_t index = 0; index < actual.values.size(); ++index) {
    ASSERT_TRUE(std::isfinite(actual.values[index]));
    ASSERT_LE(std::abs(actual.values[index] - values[index]),
              1e-5F + 1e-4F * std::abs(values[index]))
        << index;
  }
}

TEST(NativeOnnxRuntime, LinearExecutesRealCpuAndDynamicBatches) {
  at::set_num_threads(1);
  torch::NoGradGuard guard;
  torch::nn::Linear model(8, 3);
  const std::array<std::int64_t, 2> shape{2, 8};
  const auto exported = exporter::export_linear(model, shape);
  runtime::Session session(exported.onnx, exported.digest);
  EXPECT_EQ(session.runtime_version(), "1.23.2");
  EXPECT_EQ(session.execution_provider().name, "CPUExecutionProvider");
  EXPECT_FALSE(session.cpu_fallback_allowed());
  ASSERT_FALSE(session.input().batch_symbol.empty());
  EXPECT_EQ(session.input().batch_symbol, session.output().batch_symbol);
  for (const auto batch : {1, 2, 5}) {
    const auto input = torch::randn({batch, 8});
    Near(RunSession(session, input), model->forward(input).contiguous());
  }
}

TEST(NativeOnnxRuntime, NativeMultipleIoUsesThePublicNamedRuntimeAdapter) {
  torch::NoGradGuard guard;
  torch::nn::Module owner;
  exporter::ExportOptions options;
  options.input_names = {"left", "right"};
  options.output_names = {"sum", "product"};
  const std::vector<torch::Tensor> examples{torch::ones({2, 3}),
                                            torch::ones({2, 1})};
  const auto artifact = exporter::export_module(
      owner,
      [](std::span<const torch::Tensor> inputs) {
        return std::vector<torch::Tensor>{inputs[0] + inputs[1],
                                          inputs[0] * inputs[1]};
      },
      examples, options);
  runtime::Session session(artifact.onnx, artifact.digest);
  for (const auto batch : {1, 2, 5}) {
    const auto left =
        torch::arange(batch * 3, torch::kFloat32).reshape({batch, 3});
    const auto right =
        torch::arange(batch, torch::kFloat32).reshape({batch, 1}) + 2;
    const std::array<runtime::TensorInput, 2> inputs{
        runtime::TensorInput{
            .name = "right",
            .values = std::span(right.const_data_ptr<float>(),
                                static_cast<std::size_t>(right.numel())),
            .shape = std::span(right.sizes().data(), right.sizes().size())},
        runtime::TensorInput{
            .name = "left",
            .values = std::span(left.const_data_ptr<float>(),
                                static_cast<std::size_t>(left.numel())),
            .shape = std::span(left.sizes().data(), left.sizes().size())}};
    const auto outputs = session.run(inputs);
    ASSERT_EQ(outputs.size(), 2U);
    EXPECT_EQ(outputs[0].name, "sum");
    EXPECT_EQ(outputs[1].name, "product");
    Near(outputs[0], (left + right).contiguous());
    Near(outputs[1], (left * right).contiguous());
  }
}

TEST(NativeOnnxRuntime,
     UnknownUserModelAndCustomOperatorExecuteWithoutModelWhitelist) {
  at::set_num_threads(1);
  torch::NoGradGuard guard;
  NovelNetwork model;
  model->eval();
  const auto example = torch::randn({2, 8});
  const auto result = exporter::export_model(model, example);
  runtime::Session session(result.onnx, result.digest);
  for (const auto batch : {2, 5}) {
    const auto input = torch::randn({batch, 8});
    Near(RunSession(session, input), model->forward(input).contiguous());
  }
  torch::nn::Module owner;
  exporter::ExportOptions options;
  options.custom_lowerings["aten::sin"] = [](torch::jit::Node& node,
                                             torch::jit::Graph& graph) {
    auto* replacement =
        graph.create(c10::Symbol::onnx("Sin"), {node.input()}, 1);
    replacement->output()->setType(node.output()->type());
    replacement->insertBefore(&node);
    return std::vector<torch::jit::Value*>{replacement->output()};
  };
  const std::vector<torch::Tensor> examples{example};
  const auto custom = exporter::export_module(
      owner,
      [](std::span<const torch::Tensor> inputs) {
        return std::vector<torch::Tensor>{torch::sin(inputs.front())};
      },
      examples, options);
  runtime::Session custom_session(custom.onnx, custom.digest);
  Near(RunSession(custom_session, example), torch::sin(example));
}

TEST(NativeOnnxRuntime, BiaslessFixedBatchAndResNet18MatchNativeModules) {
  at::set_num_threads(1);
  torch::NoGradGuard guard;
  torch::nn::Linear linear(torch::nn::LinearOptions(8, 3).bias(false));
  const std::array<std::int64_t, 2> linear_shape{2, 8};
  exporter::ExportOptions fixed;
  fixed.dynamic_batch = false;
  const auto dense = exporter::export_linear(linear, linear_shape, fixed);
  runtime::Session dense_session(dense.onnx, dense.digest);
  const auto input = torch::randn({2, 8});
  Near(RunSession(dense_session, input), linear->forward(input));
  EXPECT_THROW(RunSession(dense_session, torch::zeros({5, 8})),
               InvalidArgument);
#ifdef JYPPX_LIUDING_TEST_ONNX_MODELS
  auto model = models::resnet18({.num_classes = 7});
  model->eval();
  const std::array<std::int64_t, 4> image_shape{2, 3, 224, 224};
  const auto image = exporter::export_resnet18(model, image_shape);
  runtime::Session image_session(image.onnx, image.digest);
  for (const auto batch : {2, 5}) {
    const auto images = torch::randn({batch, 3, 224, 224});
    Near(RunSession(image_session, images),
         model->forward(images).contiguous());
  }
#endif
}

TEST(NativeOnnxRuntime, RejectsProviderSubstitutionAndUnwiredProviders) {
  torch::nn::Linear model(8, 3);
  const std::array<std::int64_t, 2> shape{2, 8};
  const auto exported = exporter::export_linear(model, shape);
  const auto providers = runtime::available_providers();
  EXPECT_NE(
      std::find(providers.begin(), providers.end(), "CPUExecutionProvider"),
      providers.end());
  EXPECT_TRUE(runtime::provider_append_supported("CPUExecutionProvider"));
  EXPECT_FALSE(
      runtime::provider_append_supported("UnreviewedExecutionProvider"));
  runtime::SessionOptions options;
  options.provider.name = "UnreviewedExecutionProvider";
  options.allow_cpu_fallback = true;
  EXPECT_THROW(runtime::Session(exported.onnx, exported.digest, options),
               CompatibilityError);
  for (const auto* name : {"CUDAExecutionProvider", "TensorrtExecutionProvider",
                           "DmlExecutionProvider"}) {
    if (std::find(providers.begin(), providers.end(), name) ==
        providers.end()) {
      options.provider.name = name;
      EXPECT_THROW(runtime::Session(exported.onnx, exported.digest, options),
                   CompatibilityError);
    }
  }
}

TEST(NativeOnnxRuntime,
     EnforcesIdentityInputBoundsAndInterruptionBeforePublication) {
  torch::nn::Linear model(8, 3);
  const std::array<std::int64_t, 2> shape{2, 8};
  const auto exported = exporter::export_linear(model, shape);
  auto wrong = exported.digest;
  wrong.hex.front() = wrong.hex.front() == 'a' ? 'b' : 'a';
  EXPECT_THROW(runtime::Session(exported.onnx, wrong), CompatibilityError);
  runtime::Session session(exported.onnx, exported.digest);
  std::array<float, 16> input{};
  const std::array<std::int64_t, 2> bad_shape{2, 7};
  EXPECT_THROW(session.run(input, bad_shape), InvalidArgument);
  EXPECT_THROW(session.run(std::span(input).first(15), shape), InvalidArgument);
  input.front() = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(session.run(input, shape), InvalidArgument);
  input.front() = 0;
  std::stop_source cancelled;
  cancelled.request_stop();
  EXPECT_THROW(session.run(input, shape, cancelled.get_token()), Cancelled);
  EXPECT_THROW(session.run(input, shape, {}, std::chrono::steady_clock::now()),
               DeadlineExceeded);
  EXPECT_EQ(session.run(input, shape).values.size(), 6U);
  runtime::SessionOptions bounded;
  bounded.max_batch_size = 1;
  runtime::Session limited(exported.onnx, exported.digest, bounded);
  EXPECT_THROW(limited.run(input, shape), ResourceExhausted);
  bounded = {};
  bounded.max_output_bytes = 12;
  runtime::Session small_output(exported.onnx, exported.digest, bounded);
  EXPECT_THROW(small_output.run(input, shape), ResourceExhausted);
}

TEST(NativeOnnxRuntime, CancelsActualExecutionAndKeepsIndependentRunState) {
  at::set_num_threads(1);
  torch::NoGradGuard guard;
  torch::nn::Linear model(2048, 2048);
  const std::array<std::int64_t, 2> shape{512, 2048};
  const auto exported = exporter::export_linear(model, shape);
  runtime::Session session(exported.onnx, exported.digest);
  const auto input = torch::ones({512, 2048});
  const auto values = std::span(input.const_data_ptr<float>(),
                                static_cast<std::size_t>(input.numel()));
  std::stop_source stop;
  std::jthread requester([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    stop.request_stop();
  });
  EXPECT_THROW(session.run(values, shape, stop.get_token()), Cancelled);
  EXPECT_THROW(session.run(values, shape, {},
                           std::chrono::steady_clock::now() +
                               std::chrono::milliseconds(2)),
               DeadlineExceeded);
  const auto short_input = torch::zeros({1, 2048});
  auto first = std::async(std::launch::async,
                          [&] { return RunSession(session, short_input); });
  auto second = std::async(std::launch::async,
                           [&] { return RunSession(session, short_input); });
  auto a = first.get();
  auto b = second.get();
  EXPECT_EQ(a.values, b.values);
  a.values.front() += 1;
  EXPECT_NE(a.values.front(), b.values.front());
}

TEST(NativeOnnxRuntime, RejectsNonfiniteOutputsFromRealGraph) {
  torch::NoGradGuard guard;
  torch::nn::Linear model(1, 1);
  model->weight.fill_(std::numeric_limits<float>::max());
  model->bias.zero_();
  const std::array<std::int64_t, 2> shape{1, 1};
  const auto exported = exporter::export_linear(model, shape);
  runtime::Session session(exported.onnx, exported.digest);
  const std::array<float, 1> input{2.0F};
  EXPECT_THROW(session.run(input, shape), NumericalError);
}

TEST(NativeOnnxRuntime, ProfilingObservesActualCpuKernelAndClosesIdempotently) {
  torch::nn::Linear model(8, 3);
  const std::array<std::int64_t, 2> shape{2, 8};
  const auto exported = exporter::export_linear(model, shape);
  const auto* evidence = std::getenv("JYPPX_LIUDING_ONNX_EXPORT_EVIDENCE_DIR");
  const bool preserve = evidence != nullptr && *evidence != '\0';
  const auto root = preserve
                        ? std::filesystem::u8path(evidence) / "profiles"
                        : std::filesystem::temp_directory_path() /
                              ("liuding-native-onnx-profile-" +
                               std::to_string(std::chrono::steady_clock::now()
                                                  .time_since_epoch()
                                                  .count()));
  std::filesystem::create_directories(root);
  runtime::SessionOptions options;
  options.profile_prefix = root / "linear";
  runtime::Session session(exported.onnx, exported.digest, options);
  EXPECT_EQ(RunSession(session, torch::zeros({2, 8})).values.size(), 6U);
  const auto path = session.end_profiling();
  EXPECT_EQ(session.end_profiling(), path);
  std::ifstream stream(path);
  ASSERT_TRUE(stream);
  Json events;
  stream >> events;
  bool cpu_gemm = false;
  for (const auto& event : events) {
    if (event.value("cat", "") == "Node" && event.contains("args")) {
      const auto& args = event.at("args");
      if (args.value("provider", "") == "CPUExecutionProvider" &&
          args.value("op_name", "") == "Gemm")
        cpu_gemm = true;
    }
  }
  EXPECT_TRUE(cpu_gemm);
  EXPECT_EQ(RunSession(session, torch::zeros({2, 8})).values.size(), 6U);
  stream.close();
  if (!preserve) {
    std::filesystem::remove(path);
    std::filesystem::remove(root);
  }
}
#endif
}  // namespace
