// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/torch.h>

#ifdef JYPPX_LIUDING_TEST_GENERIC_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace jyppx::liuding;
namespace exporter = jyppx::liuding::onnx_export;
using Tensors = std::vector<at::Tensor>;

// These module types are local to this consumer. No production model registry
// or architecture-specific exporter can identify them.
class UserPairModel final : public torch::nn::Module {
 public:
  UserPairModel()
      : gain_(register_parameter("gain", torch::tensor({0.5F, -1.0F, 2.0F}))),
        offset_(
            register_buffer("offset", torch::tensor({-0.25F, 0.5F, 1.0F}))) {}

  Tensors forward(const at::Tensor& left, const at::Tensor& right) const {
    return {at::relu(left * gain_ + right + offset_), left - right};
  }

 private:
  at::Tensor gain_;
  at::Tensor offset_;
};

class UserUnaryModel final : public torch::nn::Module {
 public:
  UserUnaryModel()
      : shift_(register_buffer("shift", torch::tensor({-0.5F, 0.25F, 1.5F}))) {}

  at::Tensor forward(const at::Tensor& input) const {
    return at::relu(input + shift_);
  }

 private:
  at::Tensor shift_;
};

Tensors Examples(std::int64_t batch = 2) {
  auto sequence = torch::arange(batch * 3, torch::kFloat32).reshape({batch, 3});
  return {(sequence / 4.0F - 1.0F).contiguous(),
          (sequence / 7.0F + 0.125F).contiguous()};
}

exporter::ExportOptions NamedOptions() {
  exporter::ExportOptions options;
  options.input_names = {"left", "right"};
  options.output_names = {"activated", "difference"};
  options.maximum_batch = 8;
  options.dynamic_axes = {{"left", {{0, "batch"}}},
                          {"right", {{0, "batch"}}},
                          {"activated", {{0, "batch"}}},
                          {"difference", {{0, "batch"}}}};
  return options;
}

exporter::ExportResult ExportPair(UserPairModel& model,
                                  std::span<const at::Tensor> examples,
                                  const exporter::ExportOptions& options = {}) {
  return exporter::export_module(
      model,
      [&model](std::span<const at::Tensor> values) {
        return model.forward(values[0], values[1]);
      },
      examples, options);
}

exporter::ForwardFunction ErfcForward() {
  return [](std::span<const at::Tensor> inputs) {
    return Tensors{at::erfc(inputs.front())};
  };
}

// erfc(x) = 1 - erf(x). The consumer supplies this lowering; it is not a
// production built-in and must never be accepted as a silent ATen fallback.
exporter::OperatorLowering ErfcLowering(std::size_t& calls) {
  return [&calls](torch::jit::Node& source, torch::jit::Graph& graph) {
    ++calls;
    if (source.inputs().size() != 1U || source.outputs().size() != 1U) {
      throw std::runtime_error("Unexpected erfc schema in consumer lowering");
    }
    auto* erf = graph.create(c10::Symbol::onnx("Erf"), {source.input()}, 1U);
    erf->output()->setType(source.output()->type());
    erf->insertBefore(&source);

    const auto one = at::ones({}, at::TensorOptions().dtype(at::kFloat));
    auto* constant = graph.create(c10::Symbol::onnx("Constant"), 1U);
    constant->t_(c10::Symbol::attr("value"), one);
    constant->output()->setType(c10::TensorType::create(one));
    constant->insertBefore(&source);

    auto* difference = graph.create(c10::Symbol::onnx("Sub"),
                                    {constant->output(), erf->output()}, 1U);
    difference->output()->setType(source.output()->type());
    difference->insertBefore(&source);
    return std::vector<torch::jit::Value*>{difference->output()};
  };
}

TEST(NativeOnnxGenericContracts, ExportsUnregisteredModuleWithNamedMultipleIo) {
  UserPairModel model;
  model.eval();
  const auto examples = Examples();
  const auto options = NamedOptions();
  const auto original_left = examples[0].clone();
  const auto original_right = examples[1].clone();
  const auto gain = model.named_parameters()["gain"].detach().clone();
  const auto offset = model.named_buffers()["offset"].clone();

  const auto artifact = ExportPair(model, examples, options);
  ASSERT_FALSE(artifact.onnx.empty());
  EXPECT_EQ(artifact.digest.hex, sha256_hex(artifact.onnx));
  EXPECT_EQ(artifact.model, exporter::ModelKind::generic);
  EXPECT_EQ(artifact.opset, 17U);
  EXPECT_FALSE(artifact.external_data);
  EXPECT_EQ(artifact.input_names, options.input_names);
  EXPECT_EQ(artifact.output_names, options.output_names);
  EXPECT_EQ(artifact.input_shapes,
            (std::vector<std::vector<std::int64_t>>{{2, 3}, {2, 3}}));
  EXPECT_EQ(artifact.output_shapes,
            (std::vector<std::vector<std::int64_t>>{{2, 3}, {2, 3}}));
  EXPECT_EQ(artifact.input_shape, artifact.input_shapes.front());
  EXPECT_EQ(artifact.output_shape, artifact.output_shapes.front());
  EXPECT_TRUE(
      std::is_sorted(artifact.operators.begin(), artifact.operators.end()));
  for (const auto& operation : artifact.operators) {
    EXPECT_EQ(operation.find("onnx::"), 0U) << operation;
  }
  EXPECT_TRUE(torch::equal(examples[0], original_left));
  EXPECT_TRUE(torch::equal(examples[1], original_right));
  EXPECT_TRUE(torch::equal(model.named_parameters()["gain"], gain));
  EXPECT_TRUE(torch::equal(model.named_buffers()["offset"], offset));
  EXPECT_FALSE(model.is_training());
}

TEST(NativeOnnxGenericContracts, HolderConvenienceAcceptsUserDefinedForward) {
  auto model = std::make_shared<UserUnaryModel>();
  model->eval();
  const auto input = Examples()[0];
  const auto artifact = exporter::export_model(model, input);
  EXPECT_EQ(artifact.model, exporter::ModelKind::generic);
  EXPECT_EQ(artifact.input_shapes,
            (std::vector<std::vector<std::int64_t>>{{2, 3}}));
  EXPECT_EQ(artifact.output_shapes, artifact.input_shapes);
  ASSERT_EQ(artifact.input_names.size(), 1U);
  ASSERT_EQ(artifact.output_names.size(), 1U);
  EXPECT_FALSE(artifact.input_names.front().empty());
  EXPECT_FALSE(artifact.output_names.front().empty());
}

TEST(NativeOnnxGenericContracts, UnsupportedOperatorReportsItsExactIdentity) {
  torch::nn::Module model;
  const Tensors examples{Examples()[0]};
  try {
    (void)exporter::export_module(model, ErfcForward(), examples);
    FAIL() << "aten::erfc requires an explicit consumer lowering";
  } catch (const CompatibilityError& error) {
    EXPECT_NE(std::string(error.what()).find("aten::erfc"), std::string::npos);
  }
}

TEST(NativeOnnxGenericContracts, ConsumerCanRegisterOperatorLevelLowering) {
  torch::nn::Module model;
  const Tensors examples{Examples()[0]};
  std::size_t calls = 0;
  exporter::ExportOptions options;
  options.custom_lowerings.emplace("aten::erfc", ErfcLowering(calls));
  const auto artifact =
      exporter::export_module(model, ErfcForward(), examples, options);
  EXPECT_GT(calls, 0U);
  EXPECT_NE(std::find(artifact.operators.begin(), artifact.operators.end(),
                      "onnx::Erf"),
            artifact.operators.end());
  EXPECT_NE(std::find(artifact.operators.begin(), artifact.operators.end(),
                      "onnx::Sub"),
            artifact.operators.end());
  for (const auto& operation : artifact.operators) {
    EXPECT_EQ(operation.find("onnx::"), 0U) << operation;
  }
  EXPECT_EQ(artifact.digest.hex, sha256_hex(artifact.onnx));
}

TEST(NativeOnnxGenericContracts, RejectsMissingEmptyAndUndefinedExamples) {
  UserPairModel model;
  const auto examples = Examples();
  const exporter::ForwardFunction missing;
  EXPECT_THROW(exporter::export_module(model, missing, examples),
               InvalidArgument);

  std::size_t calls = 0;
  const exporter::ForwardFunction observed =
      [&calls](std::span<const at::Tensor> values) {
        ++calls;
        return Tensors{values.front() + values.front()};
      };
  const Tensors empty;
  EXPECT_THROW(exporter::export_module(model, observed, empty),
               InvalidArgument);
  const Tensors undefined{at::Tensor{}};
  EXPECT_THROW(exporter::export_module(model, observed, undefined),
               InvalidArgument);
  const Tensors zero_batch{torch::empty({0, 3}, torch::kFloat32)};
  EXPECT_THROW(exporter::export_module(model, observed, zero_batch),
               InvalidArgument);
  EXPECT_EQ(calls, 0U);
}

TEST(NativeOnnxGenericContracts,
     RejectsEmptyUndefinedAndUnsupportedDtypeOutputs) {
  torch::nn::Module model;
  const Tensors examples{Examples()[0]};
  const exporter::ForwardFunction empty = [](std::span<const at::Tensor>) {
    return Tensors{};
  };
  EXPECT_THROW(exporter::export_module(model, empty, examples),
               InvalidArgument);
  const exporter::ForwardFunction undefined = [](std::span<const at::Tensor>) {
    return Tensors{at::Tensor{}};
  };
  EXPECT_THROW(exporter::export_module(model, undefined, examples),
               InvalidArgument);
  const exporter::ForwardFunction unsupported =
      [](std::span<const at::Tensor> values) {
        return Tensors{values.front().to(at::kComplexFloat)};
      };
  EXPECT_THROW(exporter::export_module(model, unsupported, examples),
               CompatibilityError);
}

TEST(NativeOnnxGenericContracts, RejectsInvalidInputAndOutputNameContracts) {
  UserPairModel model;
  const auto examples = Examples();
  const std::vector<std::vector<std::string>> bad_names{
      {"only_one"},
      {"one", "two", "three"},
      {"same", "same"},
      {"", "second"},
      {std::string("hidden\0name", 11U), "second"}};
  for (const auto& names : bad_names) {
    auto options = NamedOptions();
    options.dynamic_axes.clear();
    options.input_names = names;
    EXPECT_THROW(ExportPair(model, examples, options), InvalidArgument);
    options = NamedOptions();
    options.dynamic_axes.clear();
    options.output_names = names;
    EXPECT_THROW(ExportPair(model, examples, options), InvalidArgument);
  }
  auto options = NamedOptions();
  options.dynamic_axes.clear();
  options.output_names = {"left", "difference"};
  EXPECT_THROW(ExportPair(model, examples, options), InvalidArgument);
}

TEST(NativeOnnxGenericContracts, RejectsInvalidAxesBatchBoundsAndByteLimits) {
  UserPairModel model;
  const auto examples = Examples();
  for (const auto axis : {-1, 2}) {
    auto options = NamedOptions();
    options.dynamic_axes = {{"left", {{axis, "batch"}}}};
    EXPECT_THROW(ExportPair(model, examples, options), InvalidArgument) << axis;
  }
  auto options = NamedOptions();
  options.dynamic_axes = {{"absent_input", {{0, "batch"}}}};
  EXPECT_THROW(ExportPair(model, examples, options), InvalidArgument);
  options = NamedOptions();
  options.dynamic_axes = {{"left", {{0, ""}}}};
  EXPECT_THROW(ExportPair(model, examples, options), InvalidArgument);
  options = NamedOptions();
  options.minimum_batch = 0;
  EXPECT_THROW(ExportPair(model, examples, options), InvalidArgument);
  options = NamedOptions();
  options.maximum_batch = 1;
  EXPECT_THROW(ExportPair(model, examples, options), InvalidArgument);
  options = NamedOptions();
  options.opset = 18;
  EXPECT_THROW(ExportPair(model, examples, options), InvalidArgument);
  options = NamedOptions();
  options.max_output_bytes = 0;
  EXPECT_THROW(ExportPair(model, examples, options), InvalidArgument);
  options.max_output_bytes = 4;
  EXPECT_THROW(ExportPair(model, examples, options), ResourceExhausted);
}

TEST(NativeOnnxGenericContracts, RejectsUnsupportedInputDtypeBeforeForward) {
  torch::nn::Module model;
  const Tensors examples{Examples()[0].to(at::kComplexFloat)};
  std::size_t calls = 0;
  const exporter::ForwardFunction forward =
      [&calls](std::span<const at::Tensor> values) {
        ++calls;
        return Tensors{values.front() + values.front()};
      };
  EXPECT_THROW(exporter::export_module(model, forward, examples),
               CompatibilityError);
  EXPECT_EQ(calls, 0U);
}

TEST(NativeOnnxGenericContracts, RejectsExamplesThatAliasRegisteredState) {
  UserUnaryModel model;
  const Tensors examples{model.named_buffers()["shift"]};
  std::size_t calls = 0;
  const exporter::ForwardFunction forward =
      [&calls](std::span<const at::Tensor> values) {
        ++calls;
        return Tensors{values.front() + values.front()};
      };
  EXPECT_THROW(exporter::export_module(model, forward, examples),
               InvalidArgument);
  EXPECT_EQ(calls, 0U);
}

TEST(NativeOnnxGenericContracts, RejectsUnqualifiedNonBatchDynamicAxes) {
  UserPairModel model;
  auto options = NamedOptions();
  options.dynamic_axes = {{"left", {{1, "features"}}}};
  EXPECT_THROW(ExportPair(model, Examples(), options), CompatibilityError);
}

TEST(NativeOnnxGenericContracts, RejectsCustomLoweringArityAndNullValues) {
  torch::nn::Module model;
  const Tensors examples{Examples()[0]};
  exporter::ExportOptions options;
  options.custom_lowerings["aten::erfc"] = [](torch::jit::Node&,
                                              torch::jit::Graph&) {
    return std::vector<torch::jit::Value*>{};
  };
  EXPECT_THROW(exporter::export_module(model, ErfcForward(), examples, options),
               CompatibilityError);
  options.custom_lowerings["aten::erfc"] = [](torch::jit::Node&,
                                              torch::jit::Graph&) {
    return std::vector<torch::jit::Value*>{nullptr};
  };
  EXPECT_THROW(exporter::export_module(model, ErfcForward(), examples, options),
               CompatibilityError);
}

#ifdef JYPPX_LIUDING_TEST_GENERIC_ONNX_RUNTIME

// Use the independent official C++ runtime directly. This helper deliberately
// does not rely on the product adapter's initial single-input/output profile.
void CheckOfficialCpu(const exporter::ExportResult& artifact,
                      std::span<at::Tensor> inputs,
                      std::span<const at::Tensor> expected) {
  Ort::Env environment(ORT_LOGGING_LEVEL_ERROR, "liuding-generic-contracts");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  Ort::Session session(environment, artifact.onnx.data(), artifact.onnx.size(),
                       options);
  ASSERT_EQ(session.GetInputCount(), inputs.size());
  ASSERT_EQ(session.GetOutputCount(), expected.size());
  ASSERT_EQ(artifact.input_names.size(), inputs.size());
  ASSERT_EQ(artifact.output_names.size(), expected.size());
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  std::vector<Ort::Value> values;
  const auto memory =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto name = session.GetInputNameAllocated(index, allocator);
    EXPECT_EQ(artifact.input_names[index], name.get());
    input_names.push_back(artifact.input_names[index].c_str());
    auto& input = inputs[index];
    ASSERT_TRUE(input.is_contiguous());
    ASSERT_EQ(input.scalar_type(), at::kFloat);
    values.push_back(Ort::Value::CreateTensor<float>(
        memory, input.data_ptr<float>(),
        static_cast<std::size_t>(input.numel()), input.sizes().data(),
        static_cast<std::size_t>(input.dim())));
    const auto type = session.GetInputTypeInfo(index);
    const auto shape = type.GetTensorTypeAndShapeInfo();
    EXPECT_EQ(shape.GetShape().front(), -1);
    const auto symbols = shape.GetSymbolicDimensions();
    ASSERT_FALSE(symbols.empty());
    EXPECT_STREQ(symbols.front(), "batch");
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto name = session.GetOutputNameAllocated(index, allocator);
    EXPECT_EQ(artifact.output_names[index], name.get());
    output_names.push_back(artifact.output_names[index].c_str());
    const auto type = session.GetOutputTypeInfo(index);
    const auto shape = type.GetTensorTypeAndShapeInfo();
    EXPECT_EQ(shape.GetShape().front(), -1);
    const auto symbols = shape.GetSymbolicDimensions();
    ASSERT_FALSE(symbols.empty());
    EXPECT_STREQ(symbols.front(), "batch");
  }
  const auto actual =
      session.Run(Ort::RunOptions{nullptr}, input_names.data(), values.data(),
                  values.size(), output_names.data(), output_names.size());
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t output = 0; output < expected.size(); ++output) {
    const auto target = expected[output].contiguous();
    const auto signature = actual[output].GetTensorTypeAndShapeInfo();
    EXPECT_EQ(signature.GetShape(),
              std::vector<std::int64_t>(target.sizes().begin(),
                                        target.sizes().end()));
    ASSERT_EQ(signature.GetElementCount(),
              static_cast<std::size_t>(target.numel()));
    const auto* result = actual[output].GetTensorData<float>();
    const auto* reference = target.const_data_ptr<float>();
    for (std::size_t index = 0; index < signature.GetElementCount(); ++index) {
      ASSERT_TRUE(std::isfinite(result[index]));
      EXPECT_NEAR(result[index], reference[index],
                  1e-5F + 1e-4F * std::abs(reference[index]))
          << "output=" << output << " element=" << index;
    }
  }
}

TEST(NativeOnnxGenericRuntime, NamedMultipleIoRunsForIndependentBatchSizes) {
  torch::NoGradGuard no_grad;
  UserPairModel model;
  model.eval();
  const auto artifact = ExportPair(model, Examples(), NamedOptions());
  for (const auto batch : {1, 2, 5}) {
    auto inputs = Examples(batch);
    const auto expected = model.forward(inputs[0], inputs[1]);
    CheckOfficialCpu(artifact, inputs, expected);
  }
}

TEST(NativeOnnxGenericRuntime, AlternateBatchPreservesPositiveInputDomain) {
  torch::nn::Module module;
  const exporter::ForwardFunction forward =
      [](std::span<const at::Tensor> inputs) {
        return Tensors{at::log(inputs.front())};
      };
  const Tensors examples{at::full({2, 4}, 2.5F)};
  const auto artifact = exporter::export_module(module, forward, examples);
  for (const auto batch : {1, 2, 5}) {
    Tensors inputs{at::linspace(0.25F, 2.75F, batch * 4).reshape({batch, 4})};
    CheckOfficialCpu(artifact, inputs, forward(inputs));
  }
}

TEST(NativeOnnxGenericRuntime,
     ScaledAddmmPreservesAlphaBetaAndTransposedState) {
  torch::nn::Module module;
  const auto weight = module.register_parameter(
      "weight", at::linspace(-0.75F, 1.25F, 12).reshape({3, 4}));
  const auto bias =
      module.register_parameter("bias", at::tensor({0.2F, -0.4F, 0.6F}));
  const exporter::ForwardFunction forward =
      [weight, bias](std::span<const at::Tensor> inputs) {
        return Tensors{at::addmm(bias, inputs.front(), weight.t(), -0.5, 1.75)};
      };
  const Tensors examples{at::ones({2, 4})};
  const auto artifact = exporter::export_module(module, forward, examples);
  for (const auto batch : {1, 2, 5}) {
    Tensors inputs{at::linspace(-1.25F, 0.75F, batch * 4).reshape({batch, 4})};
    CheckOfficialCpu(artifact, inputs, forward(inputs));
  }
}

TEST(NativeOnnxGenericRuntime,
     ConsumerLoweringHasIndependentNumericalEvidence) {
  torch::NoGradGuard no_grad;
  torch::nn::Module model;
  const Tensors examples{Examples()[0]};
  std::size_t calls = 0;
  exporter::ExportOptions options;
  options.input_names = {"features"};
  options.output_names = {"complement"};
  options.dynamic_axes = {{"features", {{0, "batch"}}},
                          {"complement", {{0, "batch"}}}};
  options.custom_lowerings.emplace("aten::erfc", ErfcLowering(calls));
  const auto artifact =
      exporter::export_module(model, ErfcForward(), examples, options);
  ASSERT_GT(calls, 0U);
  for (const auto batch : {1, 2, 5}) {
    Tensors inputs{Examples(batch)[0]};
    const Tensors expected{at::erfc(inputs[0])};
    CheckOfficialCpu(artifact, inputs, expected);
  }
}

#endif
}  // namespace
