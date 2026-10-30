// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/onnx_export/static_int8_linear.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <torch/csrc/jit/serialization/export.h>

#include <ATen/Functions.h>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

namespace jyppx::liuding::onnx_export {
namespace {

ErrorDetails Details(std::string operation) {
  return {.component = "onnx_export.static_int8_linear",
          .operation = std::move(operation)};
}

at::Tensor Scalar(float value) {
  auto result = at::empty({}, at::TensorOptions().dtype(at::kFloat));
  *result.data_ptr<float>() = value;
  return result;
}

at::Tensor ZeroPoint(std::uint8_t value) {
  auto result = at::empty({}, at::TensorOptions().dtype(at::kByte));
  *result.data_ptr<std::uint8_t>() = value;
  return result;
}

StaticInt8LinearExportResult ExportPreparedLinear(
    const quantization::StaticInt8LinearPreparation& prepared,
    std::span<const std::int64_t> shape, const ExportOptions& options,
    std::span<const float> product_scales = {}) {
  if (options.opset != 17U || options.minimum_batch <= 0 ||
      options.maximum_batch < options.minimum_batch ||
      options.maximum_batch > 65536 || options.max_output_bytes == 0U ||
      options.max_output_bytes > (1ULL << 32U))
    throw InvalidArgument("Invalid static INT8 ONNX options",
                          Details("options"));
  if (shape.size() != 2U || shape[0] <= 0 || shape[0] > options.maximum_batch ||
      (options.dynamic_batch && shape[0] < options.minimum_batch) ||
      shape[1] != static_cast<std::int64_t>(prepared.in_features))
    throw InvalidArgument("Invalid static INT8 Linear shape", Details("shape"));
  auto remaining_payload_bytes = options.max_output_bytes;
  if (prepared.weights.size() > remaining_payload_bytes)
    throw ResourceExhausted("INT8 graph payload exceeds its byte bound",
                            Details("graph"));
  remaining_payload_bytes -= prepared.weights.size();
  if (prepared.bias.size() > remaining_payload_bytes / sizeof(float))
    throw ResourceExhausted("INT8 graph payload exceeds its byte bound",
                            Details("graph"));
  remaining_payload_bytes -= prepared.bias.size() * sizeof(float);
  if (product_scales.size() > remaining_payload_bytes / sizeof(float))
    throw ResourceExhausted("INT8 graph payload exceeds its byte bound",
                            Details("graph"));

  auto graph = std::make_shared<torch::jit::Graph>();
  std::map<std::string, at::Tensor> initializers;
  const auto tensor_type = [](at::ScalarType type,
                              const std::vector<std::int64_t>& sizes) {
    return c10::TensorType::createContiguous(type, c10::Device(c10::kCPU),
                                             sizes);
  };
  const std::vector<std::int64_t> input_shape(shape.begin(), shape.end());
  const std::vector<std::int64_t> output_shape{
      shape[0], static_cast<std::int64_t>(prepared.out_features)};
  auto* input = graph->addInput("input");
  input->setType(tensor_type(at::kFloat, input_shape));
  const auto initializer = [&](std::string name, at::Tensor tensor) {
    auto* value = graph->addInput(name);
    value->setType(c10::TensorType::create(tensor));
    initializers.emplace(std::move(name), std::move(tensor));
    return value;
  };
  const auto node =
      [&](const char* operation, std::vector<torch::jit::Value*> inputs,
          at::ScalarType type, const std::vector<std::int64_t>& sizes) {
        auto* result = graph->create(c10::Symbol::onnx(operation), inputs, 1U);
        graph->appendNode(result);
        result->output()->setType(tensor_type(type, sizes));
        return result;
      };
  auto* activation_scale =
      initializer("activation_scale", Scalar(prepared.activation_scale));
  auto* activation_zero = initializer(
      "activation_zero_point", ZeroPoint(prepared.activation_zero_point));
  auto weights = at::empty({static_cast<std::int64_t>(prepared.in_features),
                            static_cast<std::int64_t>(prepared.out_features)},
                           at::TensorOptions().dtype(at::kByte));
  std::memcpy(weights.data_ptr<std::uint8_t>(), prepared.weights.data(),
              prepared.weights.size());
  auto* weight = initializer("weight_u8", std::move(weights));
  auto* weight_zero =
      initializer("weight_zero_point", ZeroPoint(prepared.weight_zero_point));
  auto products =
      product_scales.empty()
          ? Scalar(prepared.product_scale)
          : at::empty({static_cast<std::int64_t>(product_scales.size())},
                      at::TensorOptions().dtype(at::kFloat));
  if (!product_scales.empty())
    std::memcpy(products.data_ptr<float>(), product_scales.data(),
                product_scales.size_bytes());
  auto* scale_product = initializer("scale_product", std::move(products));
  auto* quantized =
      node("QuantizeLinear", {input, activation_scale, activation_zero},
           at::kByte, input_shape);
  auto* accumulated =
      node("MatMulInteger",
           {quantized->output(), weight, activation_zero, weight_zero},
           at::kInt, output_shape);
  auto* cast = node("Cast", {accumulated->output()}, at::kFloat, output_shape);
  cast->i_(c10::Symbol::attr("to"), 1);  // Official ONNX TensorProto::FLOAT.
  auto* scaled =
      node("Mul", {cast->output(), scale_product}, at::kFloat, output_shape);
  auto* output = scaled->output();
  if (!prepared.bias.empty()) {
    auto bias = at::empty({static_cast<std::int64_t>(prepared.out_features)},
                          at::TensorOptions().dtype(at::kFloat));
    std::memcpy(bias.data_ptr<float>(), prepared.bias.data(),
                prepared.bias.size() * sizeof(float));
    auto* bias_value = initializer("bias", std::move(bias));
    output =
        node("Add", {output, bias_value}, at::kFloat, output_shape)->output();
  }
  output->setDebugName("output");
  graph->registerOutput(output);
  std::unordered_map<std::string, std::unordered_map<std::int64_t, std::string>>
      dynamic_axes;
  if (options.dynamic_batch)
    dynamic_axes = {{"input", {{0, "batch"}}}, {"output", {{0, "batch"}}}};
  try {
    const auto exported = torch::jit::export_onnx(
        graph, initializers, 17, dynamic_axes, false,
        torch::onnx::OperatorExportTypes::ONNX, true, false, {}, true, false);
    const auto bytes =
        torch::jit::serialize_model_proto_to_string(std::get<0>(exported));
    if (bytes.empty() || bytes.size() > options.max_output_bytes)
      throw ResourceExhausted("INT8 ONNX output exceeds its byte bound",
                              Details("serialize"));
    torch::jit::check_onnx_proto(bytes);
    StaticInt8LinearExportResult result;
    result.artifact.onnx.resize(bytes.size());
    std::memcpy(result.artifact.onnx.data(), bytes.data(), bytes.size());
    result.artifact.digest = {.hex = sha256_hex(result.artifact.onnx)};
    result.artifact.model = ModelKind::static_int8_linear;
    result.artifact.input_shape = input_shape;
    result.artifact.output_shape = output_shape;
    result.artifact.opset = 17U;
    std::set<std::string> operators;
    for (const auto* operation : graph->nodes())
      operators.insert(operation->kind().toQualString());
    result.artifact.operators.assign(operators.begin(), operators.end());
    result.source_artifact_digest = prepared.source_artifact_digest;
    result.profile_id = prepared.profile_id;
    return result;
  } catch (const Error&) {
    throw;
  } catch (const std::exception&) {
    std::throw_with_nested(CompatibilityError(
        "Native INT8 ONNX serialization failed", Details("serialize")));
  }
}

}  // namespace

StaticInt8LinearExportResult export_static_int8_linear(
    std::span<const std::byte> source_artifact,
    const ArtifactDigest& expected_source_digest,
    std::span<const std::int64_t> shape, const ExportOptions& options,
    quantization::QuantizationLimits limits) {
  validate_artifact_digest(expected_source_digest);
  if (source_artifact.size() > limits.max_artifact_bytes)
    throw ResourceExhausted("Q0 source exceeds its byte bound",
                            Details("source"));
  if (sha256_hex(source_artifact) != expected_source_digest.hex)
    throw CompatibilityError("Q0 source digest mismatch", Details("identity"));
  const auto source =
      quantization::deserialize_linear_artifact(source_artifact, limits);
  const auto prepared =
      quantization::prepare_static_int8_linear(source, limits);
  if (prepared.source_artifact_digest != expected_source_digest)
    throw CompatibilityError("Canonical Q0 identity mismatch",
                             Details("identity"));
  return ExportPreparedLinear(prepared, shape, options);
}

StaticInt8LinearExportResult export_static_int8_linear_per_output_channel(
    std::span<const std::byte> source_artifact,
    const ArtifactDigest& expected_source_digest,
    std::span<const std::int64_t> shape, const ExportOptions& options,
    quantization::QuantizationLimits limits) {
  validate_artifact_digest(expected_source_digest);
  if (source_artifact.size() > limits.max_artifact_bytes)
    throw ResourceExhausted("Per-output-channel source exceeds its byte bound",
                            Details("source"));
  if (sha256_hex(source_artifact) != expected_source_digest.hex)
    throw CompatibilityError("Per-output-channel source digest mismatch",
                             Details("identity"));
  const auto source =
      quantization::deserialize_per_output_channel_linear_artifact(
          source_artifact, limits);
  if (source.in_features > 65536U || source.out_features > 65536U ||
      source.in_features > limits.max_artifact_bytes / source.out_features)
    throw ResourceExhausted(
        "Per-output-channel INT8 dimensions exceed profile bounds",
        Details("shape"));
  quantization::StaticInt8LinearPreparation merged;
  merged.profile_id = "jyppx.linear.ptq.ort.u8u8.per_output_channel/v1";
  merged.in_features = source.in_features;
  merged.out_features = source.out_features;
  merged.source_artifact_digest = expected_source_digest;
  merged.weights.resize(source.in_features * source.out_features);
  std::vector<float> products;
  products.reserve(source.out_features);
  if (!source.channels.front().bias.empty())
    merged.bias.reserve(source.out_features);
  for (std::size_t out = 0U; out < source.out_features; ++out) {
    const auto channel =
        quantization::prepare_static_int8_linear(source.channels[out], limits);
    if (out == 0U) {
      merged.activation_scale = channel.activation_scale;
      merged.activation_zero_point = channel.activation_zero_point;
    }
    products.push_back(channel.product_scale);
    if (!channel.bias.empty()) merged.bias.push_back(channel.bias.front());
    for (std::size_t in = 0U; in < source.in_features; ++in)
      merged.weights[in * source.out_features + out] = channel.weights[in];
  }
  return ExportPreparedLinear(merged, shape, options, products);
}

}  // namespace jyppx::liuding::onnx_export
