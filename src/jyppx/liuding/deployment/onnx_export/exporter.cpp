// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "generic_lowering.hpp"

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/export.h>

#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace jyppx::liuding::onnx_export {
namespace {
ErrorDetails Details(std::string operation) {
  return {.component = "onnx_export.native", .operation = std::move(operation)};
}

ExportTensorRecord TensorRecord(const at::Tensor& tensor, std::string name,
                                std::string role) {
  static_assert(std::endian::native == std::endian::little);
  std::string dtype;
  switch (tensor.scalar_type()) {
    case at::kBool:
      dtype = "bool";
      break;
    case at::kByte:
      dtype = "uint8";
      break;
    case at::kChar:
      dtype = "int8";
      break;
    case at::kShort:
      dtype = "int16";
      break;
    case at::kInt:
      dtype = "int32";
      break;
    case at::kLong:
      dtype = "int64";
      break;
    case at::kHalf:
      dtype = "float16";
      break;
    case at::kBFloat16:
      dtype = "bfloat16";
      break;
    case at::kFloat:
      dtype = "float32";
      break;
    case at::kDouble:
      dtype = "float64";
      break;
    default:
      throw CompatibilityError("Unsupported export report tensor dtype");
  }
  const auto size = static_cast<std::size_t>(tensor.nbytes());
  return {.name = std::move(name),
          .dtype = std::move(dtype),
          .shape = {tensor.sizes().begin(), tensor.sizes().end()},
          .digest = {.hex = sha256_hex(std::span(
                         static_cast<const std::byte*>(tensor.const_data_ptr()),
                         size))},
          .size_bytes = size,
          .role = std::move(role)};
}

void ValidateOptions(const ExportOptions& options) {
  if (options.opset != 17U || options.minimum_batch <= 0 ||
      options.maximum_batch < options.minimum_batch ||
      options.maximum_batch > 65536 || options.max_output_bytes == 0U ||
      options.max_output_bytes > (1ULL << 32U))
    throw InvalidArgument("Invalid native ONNX export options",
                          Details("options"));
  for (const auto& [name, function] : options.custom_lowerings) {
    if (name.empty() || name.size() > 256U || !function)
      throw InvalidArgument("Invalid custom ONNX lowering", Details("options"));
  }
}

std::vector<std::int64_t> ValidateTensor(const at::Tensor& value,
                                         const ExportOptions& options,
                                         const std::string& name,
                                         bool check_values = true) {
  if (!value.defined() || value.numel() == 0)
    throw InvalidArgument(
        "Native ONNX requires a defined nonempty tensor: " + name,
        Details("tensor"));
  if (!value.device().is_cpu() || value.layout() != at::kStrided ||
      value.dim() > 8 || !value.is_contiguous())
    throw CompatibilityError("Native ONNX requires a dense CPU tensor: " + name,
                             Details("tensor"));
  switch (value.scalar_type()) {
    case at::kBool:
    case at::kByte:
    case at::kChar:
    case at::kShort:
    case at::kInt:
    case at::kLong:
    case at::kHalf:
    case at::kBFloat16:
    case at::kFloat:
    case at::kDouble:
      break;
    default:
      throw CompatibilityError("Unsupported native ONNX tensor dtype: " + name,
                               Details("tensor"));
  }
  if (value.nbytes() > options.max_output_bytes)
    throw ResourceExhausted("Native ONNX tensor exceeds its bound: " + name,
                            Details("tensor"));
  if (check_values && value.is_floating_point() &&
      !at::isfinite(value).all().item<bool>())
    throw CompatibilityError(
        "Native ONNX tensor contains nonfinite state: " + name,
        Details("tensor"));
  return {value.sizes().begin(), value.sizes().end()};
}

void ValidateOutputs(const torch::jit::Stack& outputs,
                     const ExportOptions& options) {
  for (const auto& output : outputs) {
    if (!output.isTensor())
      throw CompatibilityError("Native ONNX trace returned a nontensor output",
                               Details("trace"));
    (void)ValidateTensor(output.toTensor(), options, "output");
  }
}

std::vector<std::string> Names(const std::vector<std::string>& requested,
                               std::size_t count, const char* prefix) {
  if (!requested.empty() && requested.size() != count)
    throw InvalidArgument("Native ONNX tensor name count mismatch",
                          Details("names"));
  auto names = requested;
  if (names.empty()) {
    for (std::size_t index = 0; index < count; ++index)
      names.push_back(count == 1U
                          ? prefix
                          : std::string(prefix) + "_" + std::to_string(index));
  }
  std::set<std::string> unique;
  for (const auto& name : names) {
    if (name.empty() || name.size() > 256U || name.starts_with("state.") ||
        name.find('\0') != std::string::npos || !unique.insert(name).second)
      throw InvalidArgument("Invalid or duplicate native ONNX tensor name",
                            Details("names"));
  }
  return names;
}

DynamicAxes Axes(const ExportOptions& options, const ExportResult& result) {
  DynamicAxes axes = options.dynamic_axes;
  if (axes.empty() && options.dynamic_batch) {
    const auto batch = result.input_shapes.front().empty()
                           ? 0
                           : result.input_shapes.front().front();
    for (std::size_t index = 0; index < result.input_names.size(); ++index) {
      if (result.input_shapes[index].empty() ||
          result.input_shapes[index].front() != batch)
        throw InvalidArgument(
            "Automatic dynamic batch needs matching leading dimensions",
            Details("dynamic_axes"));
      axes[result.input_names[index]][0] = "batch";
    }
    for (std::size_t index = 0; index < result.output_names.size(); ++index) {
      if (result.output_shapes[index].empty() ||
          result.output_shapes[index].front() != batch)
        throw InvalidArgument(
            "Automatic dynamic batch needs matching output dimensions",
            Details("dynamic_axes"));
      axes[result.output_names[index]][0] = "batch";
    }
  }
  std::map<std::string, const std::vector<std::int64_t>*> shapes;
  for (std::size_t index = 0; index < result.input_names.size(); ++index)
    shapes.emplace(result.input_names[index], &result.input_shapes[index]);
  for (std::size_t index = 0; index < result.output_names.size(); ++index)
    shapes.emplace(result.output_names[index], &result.output_shapes[index]);
  std::map<std::string, std::int64_t> symbol_sizes;
  for (const auto& [name, dimensions] : axes) {
    const auto shape = shapes.find(name);
    if (shape == shapes.end() || dimensions.empty())
      throw InvalidArgument("Dynamic axes reference an unknown tensor",
                            Details("dynamic_axes"));
    for (const auto& [axis, symbol] : dimensions) {
      if (axis < 0 || static_cast<std::size_t>(axis) >= shape->second->size() ||
          symbol.empty() || symbol.size() > 128U ||
          symbol.find('\0') != std::string::npos)
        throw InvalidArgument("Invalid native ONNX symbolic axis",
                              Details("dynamic_axes"));
      if (axis != 0)
        throw CompatibilityError(
            "Native ONNX currently validates only a dynamic leading batch axis",
            Details("dynamic_axes"));
      const auto extent = (*shape->second)[static_cast<std::size_t>(axis)];
      const auto [found, inserted] = symbol_sizes.emplace(symbol, extent);
      if (!inserted && found->second != extent)
        throw InvalidArgument(
            "One symbolic axis binds unequal example dimensions",
            Details("dynamic_axes"));
    }
  }
  return axes;
}

void SymbolicTypes(const std::shared_ptr<torch::jit::Graph>& graph,
                   const DynamicAxes& axes) {
  std::map<std::string, c10::ShapeSymbol> symbols;
  const auto apply = [&](torch::jit::Value* value) {
    const auto found = axes.find(value->debugName());
    if (found == axes.end()) return;
    const auto type = value->type()->cast<c10::TensorType>();
    if (!type || !type->sizes().concrete_sizes())
      throw CompatibilityError(
          "Traced ONNX public tensor has no concrete shape", Details("trace"));
    const auto shape = *type->sizes().concrete_sizes();
    std::vector<c10::ShapeSymbol> dimensions;
    for (std::size_t index = 0; index < shape.size(); ++index) {
      const auto symbolic =
          found->second.find(static_cast<std::int64_t>(index));
      if (symbolic == found->second.end())
        dimensions.push_back(c10::ShapeSymbol::fromStaticSize(shape[index]));
      else {
        const auto [entry, inserted] = symbols.try_emplace(
            symbolic->second, c10::ShapeSymbol::newSymbol());
        (void)inserted;
        dimensions.push_back(entry->second);
      }
    }
    value->setType(
        type->withSymbolicShapes(c10::SymbolicShape(std::move(dimensions))));
  };
  for (auto* input : graph->inputs()) apply(input);
  for (auto* output : graph->outputs()) apply(output);
}

bool EqualTensor(const at::Tensor& left, const at::Tensor& right) {
  if (!left.defined() || !right.defined())
    return left.defined() == right.defined();
  return left.scalar_type() == right.scalar_type() &&
         left.device() == right.device() && left.layout() == right.layout() &&
         left.sizes().equals(right.sizes()) && at::equal(left, right);
}

bool EqualAttribute(const torch::jit::Node& left, const torch::jit::Node& right,
                    c10::Symbol name) {
  if (left.kindOf(name) != right.kindOf(name)) return false;
  using Kind = torch::jit::AttributeKind;
  switch (left.kindOf(name)) {
    case Kind::f:
      return left.f(name) == right.f(name);
    case Kind::fs:
      return left.fs(name) == right.fs(name);
    case Kind::c:
      return left.c(name) == right.c(name);
    case Kind::cs:
      return left.cs(name) == right.cs(name);
    case Kind::i:
      return left.i(name) == right.i(name);
    case Kind::is:
      return left.is(name) == right.is(name);
    case Kind::s:
      return left.s(name) == right.s(name);
    case Kind::ss:
      return left.ss(name) == right.ss(name);
    case Kind::t:
      return EqualTensor(left.t(name), right.t(name));
    case Kind::ts: {
      const auto& one = left.ts(name);
      const auto& two = right.ts(name);
      if (one.size() != two.size()) return false;
      for (std::size_t index = 0; index < one.size(); ++index)
        if (!EqualTensor(one[index], two[index])) return false;
      return true;
    }
    case Kind::ty:
      return *left.ty(name) == *right.ty(name);
    case Kind::tys: {
      const auto& one = left.tys(name);
      const auto& two = right.tys(name);
      if (one.size() != two.size()) return false;
      for (std::size_t index = 0; index < one.size(); ++index)
        if (*one[index] != *two[index]) return false;
      return true;
    }
    case Kind::ival: {
      const auto& one = left.ival(name);
      const auto& two = right.ival(name);
      if (one.isTensor() && two.isTensor())
        return EqualTensor(one.toTensor(), two.toTensor());
      return one == two;
    }
    case Kind::g:
    case Kind::gs:
      throw CompatibilityError(
          "Dynamic native ONNX cannot validate graph-valued operator "
          "attributes",
          Details("dynamic_trace"));
  }
  return false;
}

void CheckTraceStructure(const torch::jit::Graph& first,
                         const torch::jit::Graph& second) {
  std::map<const torch::jit::Value*, const torch::jit::Value*> values;
  if (first.inputs().size() != second.inputs().size() ||
      first.outputs().size() != second.outputs().size())
    throw CompatibilityError("Dynamic native ONNX trace changed its signature",
                             Details("dynamic_trace"));
  for (std::size_t index = 0; index < first.inputs().size(); ++index)
    values.emplace(first.inputs()[index], second.inputs()[index]);
  auto left = first.nodes().begin();
  auto right = second.nodes().begin();
  while (left != first.nodes().end() && right != second.nodes().end()) {
    const auto* a = *left;
    const auto* b = *right;
    if (a->kind() != b->kind() || a->inputs().size() != b->inputs().size() ||
        a->outputs().size() != b->outputs().size() || !a->blocks().empty() ||
        !b->blocks().empty())
      throw CompatibilityError(
          "Dynamic native ONNX trace changed operation at " +
              std::string(a->kind().toQualString()),
          Details("dynamic_trace"));
    for (std::size_t index = 0; index < a->inputs().size(); ++index) {
      const auto expected = values.find(a->inputs()[index]);
      if (expected == values.end() || expected->second != b->inputs()[index])
        throw CompatibilityError(
            "Dynamic native ONNX trace changed value dependencies",
            Details("dynamic_trace"));
    }
    if (a->kind() == torch::jit::prim::Constant) {
      const auto one = torch::jit::toIValue(a->output());
      const auto two = torch::jit::toIValue(b->output());
      bool equal = one.has_value() && two.has_value();
      if (equal && one->isTensor() && two->isTensor())
        equal = EqualTensor(one->toTensor(), two->toTensor());
      else if (equal)
        equal = *one == *two;
      if (!equal)
        throw CompatibilityError(
            "Dynamic native ONNX trace captured a shape-dependent constant",
            Details("dynamic_trace"));
    } else if (a->hasAttributes() || b->hasAttributes()) {
      if (a->attributeNames() != b->attributeNames())
        throw CompatibilityError(
            "Dynamic native ONNX trace changed operator attributes",
            Details("dynamic_trace"));
      for (const auto name : a->attributeNames()) {
        if (!EqualAttribute(*a, *b, name))
          throw CompatibilityError(
              "Dynamic native ONNX trace changed attribute " +
                  std::string(name.toUnqualString()) + " on " +
                  a->kind().toQualString(),
              Details("dynamic_trace"));
      }
    }
    for (std::size_t index = 0; index < a->outputs().size(); ++index)
      values.emplace(a->outputs()[index], b->outputs()[index]);
    ++left;
    ++right;
  }
  if (left != first.nodes().end() || right != second.nodes().end())
    throw CompatibilityError("Dynamic native ONNX trace changed its node count",
                             Details("dynamic_trace"));
  for (std::size_t index = 0; index < first.outputs().size(); ++index) {
    const auto expected = values.find(first.outputs()[index]);
    if (expected == values.end() || expected->second != second.outputs()[index])
      throw CompatibilityError(
          "Dynamic native ONNX trace changed a public output reference",
          Details("dynamic_trace"));
  }
}

ExportResult Export(torch::nn::Module& module, const ForwardFunction& forward,
                    std::span<const at::Tensor> examples,
                    const ExportOptions& options) {
  ValidateOptions(options);
  if (!forward || examples.empty() || examples.size() > 64U ||
      torch::jit::tracer::getTracingState())
    throw InvalidArgument(
        "Native ONNX needs a nonnested tensor forward function",
        Details("trace"));
  ExportResult result;
  result.report.emplace();
  result.input_names = Names(options.input_names, examples.size(), "input");
  torch::jit::Stack arguments;
  std::set<const c10::TensorImpl*> identities;
  std::uint64_t input_bytes = 0;
  for (std::size_t index = 0; index < examples.size(); ++index) {
    result.input_shapes.push_back(
        ValidateTensor(examples[index], options, result.input_names[index]));
    if (!result.input_shapes.back().empty() &&
        (result.input_shapes.back().front() < options.minimum_batch ||
         result.input_shapes.back().front() > options.maximum_batch))
      throw InvalidArgument("Native ONNX example batch exceeds its profile",
                            Details("shape"));
    const auto bytes = static_cast<std::uint64_t>(examples[index].nbytes());
    if (input_bytes > options.max_output_bytes - bytes ||
        !identities.insert(examples[index].unsafeGetTensorImpl()).second)
      throw InvalidArgument(
          "Native ONNX inputs exceed bounds or alias one tensor",
          Details("inputs"));
    input_bytes += bytes;
    result.report->inputs.push_back(
        TensorRecord(examples[index], result.input_names[index], "input"));
    arguments.emplace_back(examples[index]);
  }
  const auto example_identities = identities;
  std::map<std::string, at::Tensor> initializers;
  std::vector<std::string> state_names;
  std::vector<at::Tensor> state_tensors;
  std::uint64_t state_bytes = 0;
  const auto lift = [&](const auto& entries, const char* role) {
    for (const auto& entry : entries) {
      const auto& tensor = entry.value();
      const auto name = "state." + entry.key();
      ValidateTensor(tensor, options, name);
      result.report->state.push_back(TensorRecord(tensor, name, role));
      if (example_identities.contains(tensor.unsafeGetTensorImpl()))
        throw InvalidArgument(
            "Native ONNX example aliases registered module state: " + name,
            Details("inputs"));
      if (!identities.insert(tensor.unsafeGetTensorImpl()).second) continue;
      const auto bytes = static_cast<std::uint64_t>(tensor.nbytes());
      if (state_bytes > options.max_output_bytes - bytes)
        throw ResourceExhausted(
            "Native ONNX registered state exceeds its bound", Details("state"));
      state_bytes += bytes;
      arguments.emplace_back(tensor);
      state_names.push_back(name);
      state_tensors.push_back(tensor);
      initializers.emplace(name, tensor.detach());
    }
  };
  lift(module.named_parameters(), "parameter");
  lift(module.named_buffers(), "buffer");
  at::NoGradGuard no_grad;
  const auto traced = torch::jit::tracer::trace(
      std::move(arguments),
      [&](torch::jit::Stack values) {
        std::vector<at::Tensor> inputs;
        for (std::size_t index = 0; index < examples.size(); ++index)
          inputs.push_back(values[index].toTensor());
        auto outputs = forward(inputs);
        if (outputs.empty() || outputs.size() > 64U)
          throw InvalidArgument("Native ONNX needs one or more tensor outputs",
                                Details("trace"));
        torch::jit::Stack stack;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
          result.output_shapes.push_back(
              ValidateTensor(outputs[index], options, "output", false));
          stack.emplace_back(outputs[index]);
        }
        return stack;
      },
      [](const auto&) { return std::string(); }, true, false);
  ValidateOutputs(traced.second, options);
  const auto graph = traced.first->graph;
  if (!graph ||
      graph->inputs().size() != examples.size() + state_names.size() ||
      graph->outputs().size() != result.output_shapes.size())
    throw CompatibilityError(
        "Native ONNX tracer produced an unexpected signature",
        Details("trace"));
  result.output_names =
      Names(options.output_names, result.output_shapes.size(), "output");
  for (std::size_t index = 0; index < traced.second.size(); ++index)
    result.report->outputs.push_back(TensorRecord(
        traced.second[index].toTensor(), result.output_names[index], "output"));
  std::set<std::string> names(result.input_names.begin(),
                              result.input_names.end());
  for (const auto& name : result.output_names)
    if (!names.insert(name).second)
      throw InvalidArgument("Native ONNX input/output names overlap",
                            Details("names"));
  const auto axes = Axes(options, result);
  if (options.dynamic_batch || !options.dynamic_axes.empty()) {
    const auto batch = result.input_shapes.front().empty()
                           ? 0
                           : result.input_shapes.front().front();
    const auto alternate_batch =
        batch < options.maximum_batch
            ? batch + 1
            : (batch > options.minimum_batch ? batch - 1 : 0);
    if (alternate_batch <= 0)
      throw InvalidArgument(
          "Dynamic native ONNX export needs two batch profiles",
          Details("dynamic_trace"));
    torch::jit::Stack alternate_arguments;
    for (std::size_t index = 0; index < examples.size(); ++index) {
      auto shape = result.input_shapes[index];
      const auto named_axes = axes.find(result.input_names[index]);
      const bool changes =
          named_axes != axes.end() && named_axes->second.contains(0);
      if (changes && shape.front() != batch)
        throw CompatibilityError(
            "Dynamic native ONNX currently requires one shared input batch "
            "profile",
            Details("dynamic_axes"));
      auto alternate_input = examples[index];
      if (changes) {
        const auto indices = at::remainder(
            at::arange(alternate_batch,
                       at::TensorOptions().dtype(at::kLong).device(at::kCPU)),
            batch);
        alternate_input = at::index_select(examples[index], 0, indices);
        ValidateTensor(alternate_input, options, "alternate_input");
      }
      alternate_arguments.emplace_back(std::move(alternate_input));
    }
    // The forward uses the module's original TensorImpl identities. Detached
    // initializer copies are only for serialization, not trace arguments.
    for (const auto& tensor : state_tensors)
      alternate_arguments.emplace_back(tensor);
    std::vector<std::vector<std::int64_t>> alternate_outputs;
    const auto alternate = torch::jit::tracer::trace(
        std::move(alternate_arguments),
        [&](torch::jit::Stack values) {
          std::vector<at::Tensor> inputs;
          for (std::size_t index = 0; index < examples.size(); ++index)
            inputs.push_back(values[index].toTensor());
          const auto outputs = forward(inputs);
          torch::jit::Stack stack;
          for (const auto& output : outputs) {
            alternate_outputs.push_back(
                ValidateTensor(output, options, "alternate_output", false));
            stack.emplace_back(output);
          }
          return stack;
        },
        [](const auto&) { return std::string(); }, true, false);
    ValidateOutputs(alternate.second, options);
    CheckTraceStructure(*graph, *alternate.first->graph);
    if (alternate_outputs.size() != result.output_shapes.size())
      throw CompatibilityError("Dynamic native ONNX output count changed",
                               Details("dynamic_trace"));
    for (std::size_t index = 0; index < alternate_outputs.size(); ++index) {
      auto expected = result.output_shapes[index];
      if (axes.contains(result.output_names[index]))
        expected.front() = alternate_batch;
      if (expected != alternate_outputs[index])
        throw CompatibilityError(
            "Dynamic native ONNX output shape did not follow its declared "
            "batch profile",
            Details("dynamic_trace"));
    }
  }
  for (std::size_t index = 0; index < examples.size(); ++index)
    graph->inputs()[index]->setDebugName(result.input_names[index]);
  for (std::size_t index = 0; index < state_names.size(); ++index)
    graph->inputs()[examples.size() + index]->setDebugName(state_names[index]);
  std::set<torch::jit::Value*> public_outputs;
  for (std::size_t index = 0; index < result.output_names.size(); ++index) {
    auto* old = graph->outputs()[index];
    const bool graph_input =
        std::find(graph->inputs().begin(), graph->inputs().end(), old) !=
        graph->inputs().end();
    if (!graph_input && public_outputs.insert(old).second) {
      old->setDebugName(result.output_names[index]);
      continue;
    }
    auto* identity = graph->create(c10::Symbol::onnx("Identity"), {old}, 1);
    identity->output()->setType(old->type());
    identity->output()->setDebugName(result.output_names[index]);
    graph->appendNode(identity);
    graph->return_node()->replaceInput(index, identity->output());
  }
  if (options.trace_observer) options.trace_observer(*graph);
  if (const auto* directory =
          std::getenv("JYPPX_LIUDING_ONNX_EXPORT_TRACE_DIR");
      directory != nullptr && *directory != '\0') {
    const auto root = std::filesystem::u8path(directory);
    std::filesystem::create_directories(root);
    const auto text = graph->toString();
    const auto digest =
        sha256_hex(std::as_bytes(std::span(text.data(), text.size())));
    std::ofstream stream(root / (digest + ".trace.txt"), std::ios::binary);
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!stream)
      throw IoError("Cannot preserve native ONNX trace diagnostics",
                    Details("trace"));
  }
  detail::lower_to_onnx(graph, options, initializers);
  for (std::size_t index = 0; index < result.output_names.size(); ++index)
    graph->outputs()[index]->setDebugName(result.output_names[index]);
  for (std::size_t index = graph->inputs().size(); index-- > examples.size();) {
    if (graph->inputs()[index]->uses().empty()) {
      initializers.erase(graph->inputs()[index]->debugName());
      graph->eraseInput(index);
    }
  }
  SymbolicTypes(graph, axes);
  std::set<std::string> operators;
  for (const auto* node : graph->nodes()) {
    if (!node->kind().is_onnx())
      throw CompatibilityError("Unconverted native ONNX operator " +
                                   std::string(node->kind().toQualString()) +
                                   " at " + node->scopeName(),
                               Details("lower"));
    operators.insert(node->kind().toQualString());
  }
  graph->lint();
  std::unordered_map<std::string, std::unordered_map<std::int64_t, std::string>>
      dynamic_axes;
  for (const auto& [name, dimensions] : axes)
    dynamic_axes[name].insert(dimensions.begin(), dimensions.end());
  const auto exported = torch::jit::export_onnx(
      graph, initializers, options.opset, dynamic_axes, false,
      torch::onnx::OperatorExportTypes::ONNX, true, false, {}, true, false);
  if (!std::get<1>(exported).empty() || std::get<3>(exported))
    throw CompatibilityError("Native ONNX unexpectedly produced external data",
                             Details("serialize"));
  const auto bytes =
      torch::jit::serialize_model_proto_to_string(std::get<0>(exported));
  if (bytes.empty() || bytes.size() > options.max_output_bytes)
    throw ResourceExhausted(
        "Native ONNX artifact exceeds serialized byte bound",
        Details("serialize"));
  torch::jit::check_onnx_proto(bytes);
  result.onnx.resize(bytes.size());
  std::memcpy(result.onnx.data(), bytes.data(), bytes.size());
  result.digest = {.hex = sha256_hex(result.onnx)};
  result.opset = options.opset;
  result.operators.assign(operators.begin(), operators.end());
  result.input_shape = result.input_shapes.front();
  result.output_shape = result.output_shapes.front();
  result.report->model_digest = result.digest;
  result.report->model_size_bytes = result.onnx.size();
  result.report->build = current_build_fingerprint();
  result.report->opset = result.opset;
  result.report->minimum_batch = options.minimum_batch;
  result.report->maximum_batch = options.maximum_batch;
  result.report->dynamic_axes = axes;
  result.report->operators = result.operators;
  std::sort(result.report->state.begin(), result.report->state.end(),
            [](const auto& left, const auto& right) {
              return left.name < right.name;
            });
  return result;
}
}  // namespace

ExportResult export_module(torch::nn::Module& module,
                           const ForwardFunction& forward,
                           std::span<const at::Tensor> examples,
                           const ExportOptions& options) {
  try {
    return Export(module, forward, examples, options);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    std::throw_with_nested(ResourceExhausted(
        "Native ONNX exporter allocation failed", Details("export")));
  } catch (const std::exception& exception) {
    auto details = Details("export");
    if (const auto* diagnostic = exception.what(); diagnostic != nullptr) {
      constexpr std::size_t limit = 4096U;
      std::size_t length = 0;
      while (length < limit && diagnostic[length] != '\0') ++length;
      details.actual.assign(diagnostic, length);
      if (length == limit) details.actual.append(" [truncated]");
    }
    std::throw_with_nested(CompatibilityError(
        "Native ONNX trace or serialization failed", std::move(details)));
  }
}
}  // namespace jyppx::liuding::onnx_export
