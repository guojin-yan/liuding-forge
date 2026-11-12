// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "generic_lowering.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <torch/csrc/jit/ir/constants.h>

#include <ATen/Functions.h>
#include <ATen/native/TypeProperties.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>

namespace jyppx::liuding::onnx_export::detail {
namespace {
using Node = torch::jit::Node;
using Value = torch::jit::Value;

[[noreturn]] void Unsupported(const Node& node, std::string reason) {
  const auto symbol = std::string(node.kind().toQualString());
  throw CompatibilityError(
      "ONNX conversion unsupported at " + symbol + ": " + reason,
      {.component = "onnx_export.lowering",
       .operation = symbol,
       .expected = "a supported ATen schema or an explicit custom lowering",
       .actual = reason.substr(0, 1024),
       .hint = "Register an operator lowering or adjust the model operation."});
}

std::optional<c10::IValue> Literal(Value* value) {
  if (const auto result = torch::jit::toIValue(value)) return result;
  if (value->node()->kind() == torch::jit::prim::ListConstruct) {
    std::vector<std::int64_t> integers;
    for (auto* entry : value->node()->inputs()) {
      const auto literal = Literal(entry);
      if (!literal || !literal->isInt()) return std::nullopt;
      integers.push_back(literal->toInt());
    }
    return c10::IValue(integers);
  }
  return std::nullopt;
}

bool IsNone(Value* value) {
  const auto literal = Literal(value);
  return literal && literal->isNone();
}

std::int64_t Int(Value* value, const Node& owner) {
  const auto literal = Literal(value);
  if (!literal || !literal->isInt())
    Unsupported(owner, "requires a constant integer option");
  return literal->toInt();
}

double Number(Value* value, const Node& owner) {
  const auto literal = Literal(value);
  if (!literal || !(literal->isDouble() || literal->isInt()))
    Unsupported(owner, "requires a constant numeric option");
  return literal->isDouble() ? literal->toDouble()
                             : static_cast<double>(literal->toInt());
}

bool Boolean(Value* value, const Node& owner) {
  const auto literal = Literal(value);
  if (!literal || !literal->isBool())
    Unsupported(owner, "requires a constant boolean option");
  return literal->toBool();
}

std::vector<std::int64_t> Ints(Value* value, const Node& owner) {
  const auto literal = Literal(value);
  if (!literal || !literal->isIntList())
    Unsupported(owner, "requires a constant integer list");
  return literal->toIntVector();
}

at::ScalarType ScalarType(Value* value) {
  const auto type = value->type()->cast<c10::TensorType>();
  return type && type->scalarType() ? *type->scalarType() : at::kFloat;
}

std::int64_t OnnxType(at::ScalarType type, const Node& node) {
  switch (type) {
    case at::kFloat:
      return 1;
    case at::kByte:
      return 2;
    case at::kChar:
      return 3;
    case at::kShort:
      return 5;
    case at::kInt:
      return 6;
    case at::kLong:
      return 7;
    case at::kBool:
      return 9;
    case at::kHalf:
      return 10;
    case at::kDouble:
      return 11;
    case at::kBFloat16:
      return 16;
    default:
      Unsupported(node, "tensor dtype has no supported ONNX representation");
  }
}

class Lowerer final {
 public:
  Lowerer(std::shared_ptr<torch::jit::Graph> graph,
          const ExportOptions& options,
          std::map<std::string, at::Tensor>& initializers)
      : graph_(std::move(graph)),
        options_(options),
        initializers_(initializers) {}

  void Run() {
    std::vector<Node*> nodes;
    for (auto* node : graph_->nodes()) nodes.push_back(node);
    for (auto* node : nodes) {
      if (!node->blocks().empty())
        Unsupported(
            *node,
            "data-dependent control flow requires an explicit graph lowering");
      if (node->kind().is_onnx()) continue;
      const std::string symbol(node->kind().toQualString());
      if (symbol == "prim::Constant" || symbol == "prim::ListConstruct" ||
          symbol == "prim::TupleConstruct")
        continue;
      std::vector<Value*> outputs;
      if (const auto custom = options_.custom_lowerings.find(symbol);
          custom != options_.custom_lowerings.end()) {
        outputs = custom->second(*node, *graph_);
      } else {
        outputs = Convert(*node, symbol);
      }
      if (outputs.size() != node->outputs().size())
        Unsupported(*node,
                    "lowering output arity differs from the source schema");
      for (std::size_t index = 0; index < outputs.size(); ++index) {
        if (!outputs[index] || outputs[index]->owningGraph() != graph_.get())
          Unsupported(*node, "custom lowering returned an invalid graph value");
        outputs[index]->copyMetadata(node->output(index));
        node->output(index)->replaceAllUsesWith(outputs[index]);
      }
      node->destroy();
    }
    // All executable nodes have been converted or rejected above. Remove dead
    // pure ONNX operations and metadata, including transposes folded into Gemm.
    bool changed = true;
    while (changed) {
      changed = false;
      std::vector<Node*> dead;
      for (auto* node : graph_->nodes()) {
        if ((node->kind().is_onnx() ||
             node->kind() == torch::jit::prim::Constant ||
             node->kind() == torch::jit::prim::ListConstruct ||
             node->kind() == torch::jit::prim::TupleConstruct) &&
            std::all_of(node->outputs().begin(), node->outputs().end(),
                        [](Value* value) { return value->uses().empty(); }))
          dead.push_back(node);
      }
      for (auto* node : dead) {
        node->destroy();
        changed = true;
      }
    }
    for (auto* node : graph_->nodes())
      if (!node->kind().is_onnx())
        Unsupported(*node,
                    "unconverted primitive remains in the exported graph");
    for (std::size_t index = graph_->inputs().size(); index > 0; --index) {
      auto* value = graph_->inputs()[index - 1];
      const auto found = initializers_.find(value->debugName());
      if (found != initializers_.end() && value->uses().empty()) {
        initializers_.erase(found);
        graph_->eraseInput(index - 1);
      }
    }
    graph_->lint();
  }

 private:
  Value* Constant(at::Tensor tensor) {
    std::string name;
    do {
      name = "constant_" + std::to_string(counter_++);
    } while (initializers_.contains(name));
    auto* value = graph_->addInput(name);
    value->inferTypeFrom(tensor);
    initializers_.emplace(name, std::move(tensor));
    return value;
  }

  Value* Scalar(const at::Scalar& number, at::ScalarType dtype) {
    return Constant(at::scalar_tensor(
        number, at::TensorOptions().dtype(dtype).device(at::kCPU)));
  }

  Value* IntegerVector(const std::vector<std::int64_t>& values) {
    auto tensor = at::empty({static_cast<std::int64_t>(values.size())},
                            at::TensorOptions().dtype(at::kLong));
    if (!values.empty())
      std::memcpy(tensor.data_ptr<std::int64_t>(), values.data(),
                  values.size() * sizeof(std::int64_t));
    return Constant(std::move(tensor));
  }

  Value* Tensor(Value* value, Node& owner,
                std::optional<at::ScalarType> hint = {}) {
    const auto literal = Literal(value);
    if (!literal && value->type()->cast<c10::TensorType>()) return value;
    if (!literal) Unsupported(owner, "expected a tensor or a literal scalar");
    if (literal->isTensor())
      return Constant(literal->toTensor().detach().contiguous());
    if (literal->isBool())
      return Scalar(literal->toBool() ? 1.0 : 0.0, hint.value_or(at::kBool));
    if (literal->isInt())
      return Scalar(literal->toInt(), hint.value_or(at::kLong));
    if (literal->isDouble())
      return Scalar(literal->toDouble(), hint.value_or(at::kFloat));
    if (literal->isIntList()) return IntegerVector(literal->toIntVector());
    Unsupported(owner, "unsupported constant type");
  }

  Node* Op(Node& before, const char* operation,
           const std::vector<Value*>& inputs, std::size_t outputs = 1) {
    auto* result =
        graph_->create(c10::Symbol::onnx(operation), inputs, outputs);
    result->insertBefore(&before);
    if (outputs == 1 && before.outputs().size() == 1)
      result->output()->copyMetadata(before.output());
    return result;
  }

  Value* Unary(Node& node, const char* operation) {
    auto* input =
        Cast(node, Tensor(node.input(0), node), ScalarType(node.output()));
    return Op(node, operation, {input})->output();
  }

  Value* Cast(Node& node, Value* input, at::ScalarType dtype) {
    if (ScalarType(input) == dtype) return input;
    const auto type = input->type()->cast<c10::TensorType>();
    if (!type) Unsupported(node, "type conversion requires a typed tensor");
    auto* result = Op(node, "Cast", {input});
    result->i_(c10::Symbol::attr("to"), OnnxType(dtype, node));
    result->output()->setType(type->withScalarType(dtype));
    return result->output();
  }

  at::ScalarType ComparisonType(Node& node) {
    at::native::ResultTypeState state;
    for (const auto index : {0U, 1U}) {
      auto* input = node.input(index);
      const auto literal = Literal(input);
      if (literal && literal->isScalar()) {
        state =
            at::native::update_result_type_state(literal->toScalar(), state);
        continue;
      }
      const auto type = input->type()->cast<c10::TensorType>();
      if (!type || !type->dim() || !type->scalarType())
        Unsupported(node, "comparison promotion requires known dtype and rank");
      auto& category = *type->dim() == 0 ? state.zeroResult : state.dimResult;
      category = category == at::ScalarType::Undefined
                     ? *type->scalarType()
                     : c10::promoteTypes(category, *type->scalarType());
    }
    return at::native::result_type(state);
  }

  Value* Binary(Node& node, const char* operation,
                std::optional<at::ScalarType> computation_type = {}) {
    const auto dtype = computation_type.value_or(ScalarType(node.output()));
    auto* left = Cast(node, Tensor(node.input(0), node, dtype), dtype);
    auto* right = Cast(node, Tensor(node.input(1), node, dtype), dtype);
    // ONNX numeric arithmetic/ordering operators do not accept bool inputs.
    if (dtype == at::kBool) {
      const std::string name(operation);
      if (name == "Mul" || name == "Min")
        operation = "And";
      else if (name == "Add" || name == "Max")
        operation = "Or";
      else if (name == "Greater" || name == "Less") {
        left = Cast(node, left, at::kLong);
        right = Cast(node, right, at::kLong);
      }
    }
    return Op(node, operation, {left, right})->output();
  }

  std::vector<Value*> Convert(Node& node, const std::string& symbol) {
    static const std::unordered_map<std::string, const char*> unary{
        {"aten::relu", "Relu"},
        {"aten::sigmoid", "Sigmoid"},
        {"aten::tanh", "Tanh"},
        {"aten::exp", "Exp"},
        {"aten::log", "Log"},
        {"aten::sqrt", "Sqrt"},
        {"aten::abs", "Abs"},
        {"aten::neg", "Neg"},
        {"aten::floor", "Floor"},
        {"aten::ceil", "Ceil"},
        {"aten::round", "Round"},
        {"aten::erf", "Erf"},
        {"aten::hardswish", "HardSwish"}};
    if ((symbol == "aten::floor" || symbol == "aten::ceil" ||
         symbol == "aten::round") &&
        c10::isIntegralType(ScalarType(node.input(0)), false)) {
      // ATen keeps integer values and dtype unchanged. ONNX rounding operators
      // accept floating point only; a float round-trip would lose integer bits.
      return {Unary(node, "Identity")};
    }
    if (const auto found = unary.find(symbol); found != unary.end())
      return {Unary(node, found->second)};
    if (symbol == "aten::hardsigmoid") {
      auto* result = Op(node, "HardSigmoid", {Tensor(node.input(0), node)});
      result->f_(c10::Symbol::attr("alpha"), 1.0 / 6.0);
      result->f_(c10::Symbol::attr("beta"), 0.5);
      return {result->output()};
    }
    static const std::unordered_map<std::string, const char*> binary{
        {"aten::mul", "Mul"},       {"aten::pow", "Pow"},
        {"aten::matmul", "MatMul"}, {"aten::mm", "MatMul"},
        {"aten::bmm", "MatMul"},    {"aten::maximum", "Max"},
        {"aten::minimum", "Min"}};
    if (const auto found = binary.find(symbol); found != binary.end())
      return {Binary(node, found->second)};
    if (symbol == "aten::eq" || symbol == "aten::gt" || symbol == "aten::lt") {
      const auto* operation = symbol == "aten::eq"   ? "Equal"
                              : symbol == "aten::gt" ? "Greater"
                                                     : "Less";
      return {Binary(node, operation, ComparisonType(node))};
    }
    if (symbol == "aten::add" || symbol == "aten::sub") {
      const auto dtype = ScalarType(node.output());
      auto* left = Cast(node, Tensor(node.input(0), node, dtype), dtype);
      auto* right = Cast(node, Tensor(node.input(1), node, dtype), dtype);
      if (node.inputs().size() > 2 && Number(node.input(2), node) != 1.0) {
        auto* alpha = Tensor(node.input(2), node, dtype);
        const auto right_type = right->type();
        right = Op(node, dtype == at::kBool ? "And" : "Mul", {right, alpha})
                    ->output();
        right->setType(right_type);
      }
      if (dtype == at::kBool && symbol == "aten::add")
        return {Op(node, "Or", {left, right})->output()};
      return {Op(node, symbol == "aten::add" ? "Add" : "Sub", {left, right})
                  ->output()};
    }
    if (symbol == "aten::div") {
      if (node.inputs().size() > 2 && !IsNone(node.input(2)))
        Unsupported(node, "division rounding mode is not supported");
      return {Binary(node, "Div")};
    }
    if (symbol == "aten::linear") {
      const auto type = node.input(0)->type()->cast<c10::TensorType>();
      if (type && type->dim() == 2) {
        std::vector<Value*> inputs{Tensor(node.input(0), node),
                                   Tensor(node.input(1), node)};
        if (!IsNone(node.input(2)))
          inputs.push_back(Tensor(node.input(2), node));
        auto* result = Op(node, "Gemm", inputs);
        result->i_(c10::Symbol::attr("transB"), 1);
        return {result->output()};
      }
      auto* transposed = Op(node, "Transpose", {Tensor(node.input(1), node)});
      transposed->is_(c10::Symbol::attr("perm"), {1, 0});
      auto* result = Op(node, "MatMul",
                        {Tensor(node.input(0), node), transposed->output()})
                         ->output();
      if (!IsNone(node.input(2)))
        result =
            Op(node, "Add", {result, Tensor(node.input(2), node)})->output();
      return {result};
    }
    if (symbol == "aten::addmm") {
      auto* matrix = Tensor(node.input(2), node);
      auto* transpose = matrix->node();
      const bool transpose_b =
          transpose->kind() == c10::Symbol::onnx("Transpose") &&
          transpose->hasAttribute(c10::Symbol::attr("perm")) &&
          transpose->is(c10::Symbol::attr("perm")) ==
              std::vector<std::int64_t>{1, 0};
      if (transpose_b) matrix = transpose->input();
      auto* result =
          Op(node, "Gemm",
             {Tensor(node.input(1), node), matrix,
              Tensor(node.input(0), node, ScalarType(node.input(1)))});
      result->f_(c10::Symbol::attr("alpha"), Number(node.input(4), node));
      result->f_(c10::Symbol::attr("beta"), Number(node.input(3), node));
      result->i_(c10::Symbol::attr("transB"), transpose_b ? 1 : 0);
      return {result->output()};
    }
    if (symbol == "aten::_convolution" || symbol == "aten::convolution" ||
        symbol == "aten::conv1d" || symbol == "aten::conv2d" ||
        symbol == "aten::conv3d") {
      const bool internal =
          symbol == "aten::_convolution" || symbol == "aten::convolution";
      const bool transposed = internal && Boolean(node.input(6), node);
      std::vector<Value*> inputs{Tensor(node.input(0), node),
                                 Tensor(node.input(1), node)};
      if (!IsNone(node.input(2))) inputs.push_back(Tensor(node.input(2), node));
      auto* result = Op(node, transposed ? "ConvTranspose" : "Conv", inputs);
      auto padding = Ints(node.input(4), node);
      auto pads = padding;
      pads.insert(pads.end(), padding.begin(), padding.end());
      result->is_(c10::Symbol::attr("strides"), Ints(node.input(3), node));
      result->is_(c10::Symbol::attr("pads"), pads);
      result->is_(c10::Symbol::attr("dilations"), Ints(node.input(5), node));
      result->i_(c10::Symbol::attr("group"),
                 Int(node.input(internal ? 8 : 6), node));
      if (transposed)
        result->is_(c10::Symbol::attr("output_padding"),
                    Ints(node.input(7), node));
      return {result->output()};
    }
    if (symbol == "aten::batch_norm" ||
        symbol == "aten::_native_batch_norm_legit_no_training") {
      const bool no_training = symbol != "aten::batch_norm";
      if (!no_training && Boolean(node.input(5), node))
        Unsupported(node, "training BatchNorm is not an inference export");
      std::vector<Value*> inputs;
      for (std::size_t index = 0; index < 5; ++index) {
        if (IsNone(node.input(index)))
          Unsupported(node,
                      "BatchNorm requires affine state and running statistics");
        inputs.push_back(Tensor(node.input(index), node));
      }
      auto* result = Op(node, "BatchNormalization", inputs);
      result->f_(c10::Symbol::attr("epsilon"),
                 Number(node.input(no_training ? 6 : 7), node));
      result->i_(c10::Symbol::attr("training_mode"), 0);
      std::vector<Value*> outputs{result->output()};
      for (std::size_t index = 1; index < node.outputs().size(); ++index) {
        if (!node.output(index)->uses().empty())
          Unsupported(node, "training statistics outputs are used");
        outputs.push_back(result->output());
      }
      return outputs;
    }
    if (symbol == "aten::max_pool1d" || symbol == "aten::max_pool2d" ||
        symbol == "aten::max_pool3d") {
      auto* result = Op(node, "MaxPool", {Tensor(node.input(0), node)});
      const std::size_t spatial_rank = symbol == "aten::max_pool1d"   ? 1U
                                       : symbol == "aten::max_pool2d" ? 2U
                                                                      : 3U;
      const auto spatial = [&](std::vector<std::int64_t> values) {
        if (values.size() == 1U) values.resize(spatial_rank, values.front());
        if (values.size() != spatial_rank)
          Unsupported(node, "pooling option has an unexpected spatial rank");
        return values;
      };
      const auto kernel = spatial(Ints(node.input(1), node));
      auto strides = Ints(node.input(2), node);
      if (strides.empty()) strides = kernel;
      strides = spatial(std::move(strides));
      const auto padding = spatial(Ints(node.input(3), node));
      auto pads = padding;
      pads.insert(pads.end(), padding.begin(), padding.end());
      result->is_(c10::Symbol::attr("kernel_shape"), kernel);
      result->is_(c10::Symbol::attr("strides"), strides);
      result->is_(c10::Symbol::attr("pads"), pads);
      result->is_(c10::Symbol::attr("dilations"),
                  spatial(Ints(node.input(4), node)));
      result->i_(c10::Symbol::attr("ceil_mode"),
                 Boolean(node.input(5), node) ? 1 : 0);
      return {result->output()};
    }
    if (symbol == "aten::adaptive_avg_pool1d" ||
        symbol == "aten::adaptive_avg_pool2d" ||
        symbol == "aten::adaptive_avg_pool3d") {
      const auto sizes = Ints(node.input(1), node);
      if (std::any_of(sizes.begin(), sizes.end(),
                      [](auto value) { return value != 1; }))
        Unsupported(node,
                    "adaptive pooling with non-unit output requires an "
                    "explicit lowering");
      return {Unary(node, "GlobalAveragePool")};
    }
    if (symbol == "aten::mean" || symbol == "aten::sum") {
      const bool mean = symbol == "aten::mean";
      auto* input = Tensor(node.input(0), node);
      const auto dtype = ScalarType(node.output());
      // ATen applies an explicit dtype (or sum's integral promotion) before
      // accumulation. Casting the reduced result cannot recover lost values.
      if (ScalarType(input) != dtype) {
        const auto type = input->type()->cast<c10::TensorType>();
        if (!type) Unsupported(node, "reduction requires a typed tensor");
        auto* cast = Op(node, "Cast", {input});
        cast->i_(c10::Symbol::attr("to"), OnnxType(dtype, node));
        cast->output()->setType(type->withScalarType(dtype));
        input = cast->output();
      }
      const bool dimensioned = node.inputs().size() >= 3;
      const auto axes = dimensioned && !IsNone(node.input(1))
                            ? Ints(node.input(1), node)
                            : std::vector<std::int64_t>{};
      std::vector<Value*> inputs{input};
      if (!mean && !axes.empty()) inputs.push_back(IntegerVector(axes));
      auto* result = Op(node, mean ? "ReduceMean" : "ReduceSum", inputs);
      if (mean && !axes.empty()) result->is_(c10::Symbol::attr("axes"), axes);
      result->i_(c10::Symbol::attr("keepdims"),
                 dimensioned && Boolean(node.input(2), node) ? 1 : 0);
      return {result->output()};
    }
    if (symbol == "aten::flatten") {
      const auto input_type = node.input(0)->type()->cast<c10::TensorType>();
      const auto rank = input_type ? input_type->dim() : std::nullopt;
      if (!rank) Unsupported(node, "flatten requires a known tensor rank");
      const auto dimensions =
          std::max<std::int64_t>(1, static_cast<std::int64_t>(*rank));
      auto start = Int(node.input(1), node);
      auto end = Int(node.input(2), node);
      if (start < -dimensions || start >= dimensions || end < -dimensions ||
          end >= dimensions)
        Unsupported(node, "invalid flatten dimensions");
      if (start < 0) start += dimensions;
      if (end < 0) end += dimensions;
      if (end != dimensions - 1)
        Unsupported(node,
                    "flatten currently requires the last dimension as end_dim");
      if (start == 1) {
        auto* result = Op(node, "Flatten", {Tensor(node.input(0), node)});
        result->i_(c10::Symbol::attr("axis"), 1);
        return {result->output()};
      }
      // ONNX Flatten always returns rank two. ATen keeps every prefix axis
      // and may return rank one. Reshape zeros copy the runtime prefix sizes,
      // including a dynamic batch; -1 combines only the remaining suffix.
      std::vector<std::int64_t> target(static_cast<std::size_t>(start), 0);
      target.push_back(-1);
      return {Op(node, "Reshape",
                 {Tensor(node.input(0), node), IntegerVector(target)})
                  ->output()};
    }
    if (symbol == "aten::reshape" || symbol == "aten::view") {
      const auto target = Ints(node.input(1), node);
      return {Op(node, "Reshape",
                 {Tensor(node.input(0), node), IntegerVector(target)})
                  ->output()};
    }
    if (symbol == "aten::permute" || symbol == "aten::transpose" ||
        symbol == "aten::t") {
      const auto type = node.input(0)->type()->cast<c10::TensorType>();
      if (!type || !type->dim())
        Unsupported(node, "transpose requires a known tensor rank");
      const auto rank = static_cast<std::int64_t>(*type->dim());
      const auto dimension = [&](std::int64_t axis) {
        const auto dimensions = std::max<std::int64_t>(rank, 1);
        if (axis < -dimensions || axis >= dimensions)
          Unsupported(node, "invalid transpose dimension");
        return axis < 0 ? axis + dimensions : axis;
      };
      std::vector<std::int64_t> permutation;
      if (symbol == "aten::permute") {
        permutation = Ints(node.input(1), node);
        if (permutation.size() != static_cast<std::size_t>(rank))
          Unsupported(node, "permutation length differs from tensor rank");
        std::set<std::int64_t> unique;
        for (auto& axis : permutation) {
          axis = dimension(axis);
          if (!unique.insert(axis).second)
            Unsupported(node, "permutation contains duplicate dimensions");
        }
      } else {
        for (std::int64_t index = 0; index < rank; ++index)
          permutation.push_back(index);
        const auto first =
            symbol == "aten::t" ? 0 : dimension(Int(node.input(1), node));
        const auto second = symbol == "aten::t"
                                ? std::max<std::int64_t>(rank - 1, 0)
                                : dimension(Int(node.input(2), node));
        if (rank == 0) return {Unary(node, "Identity")};
        std::swap(permutation[static_cast<std::size_t>(first)],
                  permutation[static_cast<std::size_t>(second)]);
      }
      if (rank == 0) return {Unary(node, "Identity")};
      auto* result = Op(node, "Transpose", {Tensor(node.input(0), node)});
      result->is_(c10::Symbol::attr("perm"), permutation);
      return {result->output()};
    }
    if (symbol == "aten::cat" || symbol == "aten::stack") {
      auto* list = node.input(0)->node();
      if (list->kind() != torch::jit::prim::ListConstruct)
        Unsupported(node, "concat requires an explicit tensor list");
      const auto axis = Int(node.input(1), node);
      std::vector<Value*> inputs;
      for (auto* entry : list->inputs()) {
        auto* value = Tensor(entry, node);
        if (symbol == "aten::stack")
          value =
              Op(node, "Unsqueeze", {value, IntegerVector({axis})})->output();
        inputs.push_back(value);
      }
      auto* result = Op(node, "Concat", inputs);
      result->i_(c10::Symbol::attr("axis"), axis);
      return {result->output()};
    }
    if (symbol == "aten::squeeze") {
      const auto type = node.input(0)->type()->cast<c10::TensorType>();
      if (!type || !type->dim())
        Unsupported(node, "squeeze requires a known tensor rank");
      const auto rank = static_cast<std::int64_t>(*type->dim());
      if (rank == 0) return {Unary(node, "Identity")};
      const bool dynamic =
          options_.dynamic_batch || !options_.dynamic_axes.empty();
      std::vector<Value*> inputs{Tensor(node.input(0), node)};
      if (node.inputs().size() > 1) {
        auto axis = Int(node.input(1), node);
        if (axis < -rank || axis >= rank)
          Unsupported(node, "invalid squeeze dimension");
        if (axis < 0) axis += rank;
        if (dynamic && axis == 0)
          Unsupported(node,
                      "squeezing a dynamic batch may change output rank; use "
                      "an explicit non-batch dimension or a static export");
        const auto sizes = type->sizes().concrete_sizes();
        if (!sizes) Unsupported(node, "squeeze requires known dimension sizes");
        // ATen leaves a selected non-singleton dimension unchanged. ONNX
        // Squeeze with that axis instead rejects the tensor at execution.
        if ((*sizes)[static_cast<std::size_t>(axis)] != 1)
          return {Unary(node, "Identity")};
        inputs.push_back(IntegerVector({axis}));
      } else if (dynamic) {
        Unsupported(node,
                    "dynamic squeeze needs an explicit non-batch dimension; "
                    "removing all singleton axes may remove batch one");
      }
      return {Op(node, "Squeeze", inputs)->output()};
    }
    if (symbol == "aten::unsqueeze") {
      return {Op(node, "Unsqueeze",
                 {Tensor(node.input(0), node),
                  IntegerVector({Int(node.input(1), node)})})
                  ->output()};
    }
    if (symbol == "aten::softmax" || symbol == "aten::_softmax" ||
        symbol == "aten::log_softmax" || symbol == "aten::_log_softmax") {
      auto* input = Tensor(node.input(0), node);
      auto type = input->type()->cast<c10::TensorType>();
      if (!type || !type->dim())
        Unsupported(node, "softmax requires a known tensor rank");
      const auto dtype = ScalarType(node.output());
      // Both explicit dtype and the internal half_to_float option select the
      // computation type, so conversion must precede normalization.
      if (ScalarType(input) != dtype) {
        auto* cast = Op(node, "Cast", {input});
        cast->i_(c10::Symbol::attr("to"), OnnxType(dtype, node));
        type = type->withScalarType(dtype);
        cast->output()->setType(type);
        input = cast->output();
      }
      const bool scalar = *type->dim() == 0;
      if (scalar) {
        auto* reshape = Op(node, "Reshape", {input, IntegerVector({1})});
        type = type->withSizes(std::vector<std::int64_t>{1});
        reshape->output()->setType(type);
        input = reshape->output();
      }
      const bool logarithmic =
          symbol == "aten::log_softmax" || symbol == "aten::_log_softmax";
      auto* result = Op(node, logarithmic ? "LogSoftmax" : "Softmax", {input});
      result->i_(c10::Symbol::attr("axis"),
                 scalar ? 0 : Int(node.input(1), node));
      if (!scalar) return {result->output()};
      result->output()->setType(type);
      return {Op(node, "Squeeze", {result->output(), IntegerVector({0})})
                  ->output()};
    }
    if (symbol == "aten::dropout") {
      if (Boolean(node.input(2), node))
        Unsupported(node, "training dropout is not an inference export");
      return {Unary(node, "Identity")};
    }
    if (symbol == "aten::detach" || symbol == "aten::contiguous" ||
        symbol == "aten::clone" || symbol == "aten::alias" ||
        symbol == "aten::resolve_conj" || symbol == "aten::resolve_neg")
      return {Unary(node, "Identity")};
    if (symbol == "aten::gelu") {
      const auto approximation = Literal(node.input(1));
      if (!approximation || !approximation->isString() ||
          approximation->toStringRef() != "none")
        Unsupported(node, "GELU currently requires the exact erf formulation");
      auto* input = Tensor(node.input(0), node);
      const auto dtype = ScalarType(input);
      auto* normalized =
          Op(node, "Div", {input, Scalar(std::sqrt(2.0), dtype)})->output();
      auto* erf = Op(node, "Erf", {normalized})->output();
      auto* offset = Op(node, "Add", {erf, Scalar(1.0, dtype)})->output();
      auto* product = Op(node, "Mul", {input, offset})->output();
      return {Op(node, "Mul", {product, Scalar(0.5, dtype)})->output()};
    }
    if (symbol == "aten::silu") {
      auto* input = Tensor(node.input(0), node);
      auto* sigmoid = Op(node, "Sigmoid", {input})->output();
      return {Op(node, "Mul", {input, sigmoid})->output()};
    }
    if (symbol == "aten::layer_norm") {
      const auto normalized = Ints(node.input(1), node);
      auto* weight =
          IsNone(node.input(2))
              ? Constant(at::ones(normalized, at::TensorOptions().dtype(
                                                  ScalarType(node.input(0)))))
              : Tensor(node.input(2), node);
      std::vector<Value*> inputs{Tensor(node.input(0), node), weight};
      if (!IsNone(node.input(3))) inputs.push_back(Tensor(node.input(3), node));
      auto* result = Op(node, "LayerNormalization", inputs);
      result->i_(c10::Symbol::attr("axis"),
                 -static_cast<std::int64_t>(normalized.size()));
      result->f_(c10::Symbol::attr("epsilon"), Number(node.input(4), node));
      return {result->output()};
    }
    if (symbol == "aten::leaky_relu") {
      auto* result = Op(node, "LeakyRelu", {Tensor(node.input(0), node)});
      result->f_(c10::Symbol::attr("alpha"), Number(node.input(1), node));
      return {result->output()};
    }
    if (symbol == "aten::embedding") {
      auto* result =
          Op(node, "Gather",
             {Tensor(node.input(0), node), Tensor(node.input(1), node)});
      result->i_(c10::Symbol::attr("axis"), 0);
      return {result->output()};
    }
    if (symbol == "aten::to" || symbol == "aten::type_as") {
      auto* result = Op(node, "Cast", {Tensor(node.input(0), node)});
      result->i_(c10::Symbol::attr("to"),
                 OnnxType(ScalarType(node.output()), node));
      return {result->output()};
    }
    Unsupported(node, "no registered lowering for this operator/schema");
  }

  std::shared_ptr<torch::jit::Graph> graph_;
  const ExportOptions& options_;
  std::map<std::string, at::Tensor>& initializers_;
  std::size_t counter_ = 0;
};
}  // namespace

void lower_to_onnx(const std::shared_ptr<torch::jit::Graph>& graph,
                   const ExportOptions& options,
                   std::map<std::string, at::Tensor>& initializers) {
  Lowerer(graph, options, initializers).Run();
}

}  // namespace jyppx::liuding::onnx_export::detail
