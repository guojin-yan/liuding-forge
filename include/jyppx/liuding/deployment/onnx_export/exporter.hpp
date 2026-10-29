// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_EXPORTER_HPP
#define JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_EXPORTER_HPP
/** @file exporter.hpp
 * @brief Native C++ tracing, extensible ONNX lowering and serialization.
 */

#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <torch/nn/module.h>

#include <ATen/ATen.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace torch::jit {
struct Graph;
struct Node;
struct Value;
}  // namespace torch::jit

namespace jyppx::liuding::onnx_export {

/** @brief Descriptive artifact labels; export does not use an architecture
 * list. */
enum class ModelKind : std::uint8_t {
  generic = 0,   ///< Arbitrary native module and caller-owned forward function.
  linear = 1,    ///< One dense FP32 Linear layer.
  resnet18 = 2,  ///< The native Liuding Forge basic-block ResNet18.
  static_int8_linear = 3,  ///< Bounded static Linear integer-kernel profile.
};

/** @brief Inserts ONNX nodes and returns one replacement per source output.
 * @note Insert nodes before the supplied source node. The lowerer replaces uses
 * and destroys it; callbacks must not destroy that node. Callbacks are trusted
 * native code and must return typed values in the supplied graph.
 */
using OperatorLowering = std::function<std::vector<torch::jit::Value*>(
    torch::jit::Node&, torch::jit::Graph&)>;

/** @brief Symbolic names for validated leading batch axes of public tensors. */
using DynamicAxes = std::map<std::string, std::map<std::int64_t, std::string>>;

/** @brief Explicit bounded export profile; values are never inferred. */
struct ExportOptions {
  std::uint32_t opset = 17U;  ///< Only opset 17 is qualified by this adapter.
  std::int64_t minimum_batch = 1;     ///< Positive dynamic batch lower bound.
  std::int64_t maximum_batch = 1024;  ///< Finite dynamic batch upper bound.
  bool dynamic_batch = true;  ///< Give only dimension zero a symbolic name.
  std::uint64_t max_output_bytes = 256U << 20U;  ///< Serialized model bound.
  std::vector<std::string> input_names;   ///< Empty generates input/input_N.
  std::vector<std::string> output_names;  ///< Empty generates output/output_N.
  DynamicAxes dynamic_axes;  ///< Explicit leading axes replace batch defaults.
  std::map<std::string, OperatorLowering> custom_lowerings;  ///< ATen handlers.
  std::function<void(const torch::jit::Graph&)>
      trace_observer;  ///< Optional raw graph diagnostics.
};

/** @brief Native export result with exact bytes and graph identity. */
struct ExportResult {
  std::vector<std::byte> onnx;  ///< Official protobuf bytes, self-contained.
  ArtifactDigest digest;        ///< SHA-256 of the complete `onnx` byte range.
  ModelKind model = ModelKind::generic;  ///< Descriptive artifact label only.
  std::vector<std::string> operators;  ///< Sorted domain-qualified node types.
  std::vector<std::int64_t> input_shape;   ///< Concrete input profile shape.
  std::vector<std::int64_t> output_shape;  ///< Concrete output shape.
  std::vector<std::vector<std::int64_t>>
      input_shapes;  ///< Every example input.
  std::vector<std::vector<std::int64_t>>
      output_shapes;                      ///< Every traced output.
  std::vector<std::string> input_names;   ///< Public graph input names.
  std::vector<std::string> output_names;  ///< Public graph output names.
  std::uint32_t opset = 0U;    ///< Actual ONNX opset encoded in the bytes.
  bool external_data = false;  ///< Always false for this bounded first slice.
};

/** @brief Calls trusted native forward code with positional tensor inputs. */
using ForwardFunction =
    std::function<std::vector<at::Tensor>(std::span<const at::Tensor>)>;

/** @brief Traces a native module, lowers ATen operations and serializes ONNX.
 * @param module Registered parameters and buffers are lifted as graph inputs.
 * @param forward Trusted caller-owned forward function; no model list is used.
 * @param examples Defined nonempty positional dense CPU tensors, with no tensor
 * identity shared with registered module state.
 * @param options Names, symbolic axes, opset, bounds and operator extensions.
 * @return Self-contained official ONNX bytes and exact example signatures.
 * @throws InvalidArgument For invalid names, axes, examples or options.
 * @throws CompatibilityError For unsupported operations, dtype or device.
 * @throws ResourceExhausted For configured input, state or output limits.
 * @note Forward executes under NoGradGuard in the supplied mode; use eval() for
 * inference. It must be deterministic and must not mutate inputs, module state
 * or external resources. Native side effects cannot be sandboxed or rolled
 * back. Tracing captures data-dependent native control flow only for examples.
 * Dynamic export executes an alternate batch trace and rejects changed graph
 * topology, constants, attributes or output references. Only leading batch
 * axes are currently supported; this does not prove all batches or values.
 * Dynamic labels require independent representative-shape validation. No
 * Python or subprocess is used; unsupported nodes fail with operation details.
 */
ExportResult export_module(torch::nn::Module& module,
                           const ForwardFunction& forward,
                           std::span<const at::Tensor> examples,
                           const ExportOptions& options = {});

/** @brief Exports any owning holder whose forward accepts and returns a Tensor.
 * @param model Nonempty holder; dynamic export invokes two traced forwards.
 * @param example Real input example for the native graph frontend.
 * @param options Generic export settings.
 * @return The same generic artifact as export_module.
 */
template <typename Holder>
ExportResult export_model(Holder model, const at::Tensor& example,
                          const ExportOptions& options = {}) {
  if (!model)
    throw InvalidArgument("Native ONNX export requires a nonempty holder");
  std::vector<at::Tensor> inputs{example};
  auto& module = *model;
  return export_module(
      module,
      [model = std::move(model)](std::span<const at::Tensor> values) mutable {
        return std::vector<at::Tensor>{model->forward(values.front())};
      },
      inputs, options);
}

/** @brief Generic graph export using a zero FP32 CPU example.
 * @param model Native Linear holder; dispatch uses traced operations.
 * @param example_shape Positive input example dimensions.
 * @param options Generic graph export settings.
 * @return Generic artifact with a descriptive Linear label.
 */
template <typename Holder>
ExportResult export_linear(Holder model,
                           std::span<const std::int64_t> example_shape,
                           const ExportOptions& options = {}) {
  std::uint64_t count = 1;
  for (const auto extent : example_shape) {
    if (extent <= 0 || static_cast<std::uint64_t>(extent) >
                           options.max_output_bytes / sizeof(float) / count)
      throw InvalidArgument("Native ONNX example shape exceeds its bound");
    count *= static_cast<std::uint64_t>(extent);
  }
  auto result = export_model(
      std::move(model),
      at::zeros(
          std::vector<std::int64_t>(example_shape.begin(), example_shape.end()),
          at::TensorOptions().dtype(at::kFloat).device(at::kCPU)),
      options);
  result.model = ModelKind::linear;
  return result;
}

/** @brief ResNet convenience label using the same generic graph frontend.
 * @param model Native holder with a tensor forward method.
 * @param example_shape Positive FP32 CPU input example shape.
 * @param options Generic graph export settings.
 * @return Generic artifact with a descriptive ResNet18 label.
 * @note No architecture whitelist or separate model graph implementation
 * exists.
 */
template <typename Holder>
ExportResult export_resnet18(Holder model,
                             std::span<const std::int64_t> example_shape,
                             const ExportOptions& options = {}) {
  auto result = export_linear(std::move(model), example_shape, options);
  result.model = ModelKind::resnet18;
  return result;
}

}  // namespace jyppx::liuding::onnx_export

#endif  // JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_EXPORTER_HPP
