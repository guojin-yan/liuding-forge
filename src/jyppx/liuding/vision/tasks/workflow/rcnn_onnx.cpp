// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/vision/models/rcnn/faster_rcnn.hpp>
#include <jyppx/liuding/vision/tasks/workflow/export.hpp>

#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <cmath>
namespace jyppx::liuding::vision::tasks::detail {
namespace {
constexpr auto symbolic_name = "liuding_vision::onnx_symbolic";
c10::Symbol Symbolic() {
  const auto symbol = c10::Symbol::fromQualString(symbolic_name);
  static const bool registered = [symbol] {
    torch::jit::registerOperator(torch::jit::Operator(
        symbol,
        [](const torch::jit::Node*) -> torch::jit::Operation {
          return [](torch::jit::Stack&) {
            throw InternalError(
                "Vision ONNX symbolic nodes must be lowered before execution");
          };
        },
        c10::AliasAnalysisKind::PURE_FUNCTION));
    return true;
  }();
  static_cast<void>(registered);
  return symbol;
}
}  // namespace
// All data-dependent proposal selection is authored as ONNX tensor operations.
// Dense convolution/FC stages are traced from the original registered module.
// A registered pure symbolic node keeps the tracer's alias analysis valid;
// the exporter replaces it with the standard ONNX operation after tracing.
class RcnnGraph {
 public:
  using V = torch::jit::Value;
  using N = torch::jit::Node;
  using I = std::int64_t;
  torch::jit::Graph& graph;
  explicit RcnnGraph(torch::jit::Graph& g) : graph(g) {}
  V* constant(const Tensor& t) {
    auto* n = node("Constant", {});
    n->t_(c10::Symbol::attr("value"), t);
    n->output()->setType(c10::TensorType::create(t));
    return n->output();
  }
  V* ints(std::vector<I> v) {
    return constant(torch::tensor(v, torch::kInt64));
  }
  V* scalar(double v) {
    return constant(torch::scalar_tensor(v, torch::kFloat32));
  }
  N* node(const char* kind, std::vector<V*> input, std::size_t outputs = 1) {
    auto* n = graph.create(Symbolic(), input, outputs);
    n->s_(c10::Symbol::attr("onnx_kind"), kind);
    graph.appendNode(n);
    for (auto* v : n->outputs()) v->setType(c10::TensorType::get());
    return n;
  }
  V* op(const char* kind, std::vector<V*> input) {
    return node(kind, std::move(input))->output();
  }
  V* reshape(V* input, std::vector<I> shape) {
    return op("Reshape", {input, ints(std::move(shape))});
  }
  V* unsqueeze(V* input, std::vector<I> axes) {
    return op("Unsqueeze", {input, ints(std::move(axes))});
  }
  V* gather(V* input, V* ids, I axis = 0) {
    auto* n = node("Gather", {input, ids});
    n->i_(c10::Symbol::attr("axis"), axis);
    return n->output();
  }
  V* column(V* input, I column) {
    return gather(input, constant(torch::scalar_tensor(column, torch::kInt64)),
                  1);
  }
  V* slice(V* input, I begin, I end, I axis) {
    return op("Slice", {input, ints({begin}), ints({end}), ints({axis})});
  }
  V* cat(std::vector<V*> input, I axis = 0) {
    auto* n = node("Concat", std::move(input));
    n->i_(c10::Symbol::attr("axis"), axis);
    return n->output();
  }
  V* clip(V* input, double lo, double hi) {
    return op("Clip", {input, scalar(lo), scalar(hi)});
  }
  std::pair<V*, V*> topk(V* input, I count) {
    auto* n = node("TopK", {input, ints({count})}, 2);
    n->i_(c10::Symbol::attr("axis"), 0);
    n->i_(c10::Symbol::attr("largest"), 1);
    n->i_(c10::Symbol::attr("sorted"), 1);
    return {n->output(0), n->output(1)};
  }
  V* cast(V* input, I type) {
    auto* n = node("Cast", {input});
    n->i_(c10::Symbol::attr("to"), type);
    return n->output();
  }
};
inline std::vector<Tensor> trace_rcnn(models::rcnn::FasterRcnnImpl& model,
                                      const Tensor& images,
                                      const VisionExportOptions& profile) {
  using I = std::int64_t;
  using V = torch::jit::Value;
  auto raw = model.forward_rpn(images);
  const auto state = torch::jit::tracer::getTracingState();
  if (!state)
    throw InternalError("RCNN ONNX graph requires the native tracing scope");
  std::vector<V*> values;
  for (const auto& t : raw)
    values.push_back(torch::jit::tracer::getValueTrace(t));
  auto resume = torch::jit::tracer::pauseTracing();
  struct Resume {
    std::function<void()> callback;
    ~Resume() {
      if (callback) callback();
    }
  } resume_guard{resume};
  RcnnGraph g(*state->graph);
  const auto& o = model.options();
  const I batch = images.size(0), height = images.size(2),
          width = images.size(3), limit = profile.rcnn_proposals;
  std::vector<V*> batch_rois, batch_valid;
  for (I b = 0; b < batch; ++b) {
    std::vector<V*> all_boxes, all_scores;
    for (std::size_t level = 0; level < 5; ++level) {
      const I stride = 4LL << level, h = (height + stride - 1) / stride,
              w = (width + stride - 1) / stride;
      std::vector<float> anchors;
      anchors.reserve(
          static_cast<std::size_t>(h * w * o.aspect_ratios.size() * 4));
      for (I y = 0; y < h; ++y)
        for (I x = 0; x < w; ++x)
          for (auto ratio : o.aspect_ratios) {
            const auto aw = o.anchor_sizes[level] / std::sqrt(ratio),
                       ah = o.anchor_sizes[level] * std::sqrt(ratio);
            anchors.insert(
                anchors.end(),
                {static_cast<float>(x * stride + std::nearbyint(-aw / 2)),
                 static_cast<float>(y * stride + std::nearbyint(-ah / 2)),
                 static_cast<float>(x * stride + std::nearbyint(aw / 2)),
                 static_cast<float>(y * stride + std::nearbyint(ah / 2))});
          }
      auto* batch_index = g.constant(torch::scalar_tensor(b, torch::kInt64));
      const I k = (std::min)(raw[4 + level].size(1), o.rpn_pre_nms_eval);
      auto [scores_logits, ids] =
          g.topk(g.gather(values[4 + level], batch_index), k);
      auto* scores = g.op("Sigmoid", {scores_logits});
      auto* anchor =
          g.gather(g.constant(torch::tensor(anchors).view({-1, 4})), ids);
      auto* delta = g.gather(g.gather(values[9 + level], batch_index), ids);
      auto* wh = g.op(
          "Max",
          {g.op("Sub", {g.slice(anchor, 2, 4, 1), g.slice(anchor, 0, 2, 1)}),
           g.scalar(1e-6)});
      auto* center = g.op(
          "Add", {g.slice(anchor, 0, 2, 1), g.op("Mul", {wh, g.scalar(.5)})});
      auto* pc =
          g.op("Add", {center, g.op("Mul", {g.slice(delta, 0, 2, 1), wh})});
      auto* scale = g.op(
          "Min", {g.slice(delta, 2, 4, 1), g.scalar(std::log(1000. / 16))});
      auto* half =
          g.op("Mul", {g.op("Mul", {g.op("Exp", {scale}), wh}), g.scalar(.5)});
      auto* boxes =
          g.cat({g.op("Sub", {pc, half}), g.op("Add", {pc, half})}, 1);
      boxes = g.cat({g.unsqueeze(g.clip(g.column(boxes, 0), 0, width), {1}),
                     g.unsqueeze(g.clip(g.column(boxes, 1), 0, height), {1}),
                     g.unsqueeze(g.clip(g.column(boxes, 2), 0, width), {1}),
                     g.unsqueeze(g.clip(g.column(boxes, 3), 0, height), {1})},
                    1);
      auto* valid = g.op(
          "And", {g.op("GreaterOrEqual",
                       {g.op("Sub", {g.column(boxes, 2), g.column(boxes, 0)}),
                        g.scalar(.001)}),
                  g.op("GreaterOrEqual",
                       {g.op("Sub", {g.column(boxes, 3), g.column(boxes, 1)}),
                        g.scalar(.001)})});
      scores = g.op("Where", {valid, scores, g.scalar(-1)});
      auto* selected = g.op(
          "NonMaxSuppression",
          {g.unsqueeze(boxes, {0}), g.reshape(scores, {1, 1, k}),
           g.constant(torch::scalar_tensor(o.rpn_post_nms_eval, torch::kInt64)),
           g.scalar(o.rpn_nms_threshold), g.scalar(0)});
      auto* kept = g.column(selected, 2);
      all_boxes.push_back(g.gather(boxes, kept));
      all_scores.push_back(g.gather(scores, kept));
    }
    // Padding precedes TopK, so the public row count stays fixed even for empty
    // images.
    all_boxes.push_back(g.constant(torch::zeros({limit, 4})));
    all_scores.push_back(g.constant(torch::full({limit}, -1.f)));
    auto* candidates = g.cat(all_boxes);
    auto [scores, ids] = g.topk(g.cat(all_scores), limit);
    auto* boxes = g.gather(candidates, ids);
    batch_rois.push_back(g.cat(
        {g.constant(torch::full({limit, 1}, static_cast<float>(b))), boxes},
        1));
    batch_valid.push_back(g.cast(g.op("Greater", {scores, g.scalar(0)}), 7));
  }
  auto* rois = g.cat(batch_rois);
  auto* valid = g.cat(batch_valid);
  auto* boxes = g.slice(rois, 1, 5, 1);
  auto* wh = g.op(
      "Max", {g.op("Sub", {g.slice(boxes, 2, 4, 1), g.slice(boxes, 0, 2, 1)}),
              g.scalar(0)});
  auto* area = g.op("Mul", {g.column(wh, 0), g.column(wh, 1)});
  auto* level = g.clip(
      g.op("Floor",
           {g.op("Add",
                 {g.scalar(4),
                  g.op("Div",
                       {g.op("Log",
                             {g.op("Add", {g.op("Div", {g.op("Sqrt", {area}),
                                                        g.scalar(224)}),
                                           g.scalar(1e-6)})}),
                        g.scalar(std::log(2.))})})}),
      2, 5);
  auto* batch_indices = g.cast(g.column(rois, 0), 7);
  V* pooled = nullptr;
  for (I i = 0; i < 4; ++i) {
    auto* roi = g.node("RoiAlign", {values[i], boxes, batch_indices});
    roi->i_(c10::Symbol::attr("output_height"), 7);
    roi->i_(c10::Symbol::attr("output_width"), 7);
    roi->i_(c10::Symbol::attr("sampling_ratio"), 2);
    roi->f_(c10::Symbol::attr("spatial_scale"), 1. / (4LL << i));
    roi->s_(c10::Symbol::attr("mode"), "avg");
    roi->s_(c10::Symbol::attr("coordinate_transformation_mode"),
            "output_half_pixel");
    auto* mask = g.reshape(g.cast(g.op("Equal", {level, g.scalar(i + 2)}), 1),
                           {batch * limit, 1, 1, 1});
    auto* contribution = g.op("Mul", {roi->output(), mask});
    pooled = pooled ? g.op("Add", {pooled, contribution}) : contribution;
  }
  auto fake_pooled = torch::zeros({batch * limit, o.fpn_channels, 7, 7}),
       fake_rois = torch::zeros({batch * limit, 5}),
       fake_valid = torch::zeros({batch * limit}, torch::kInt64);
  pooled->setType(c10::TensorType::create(fake_pooled));
  rois->setType(c10::TensorType::create(fake_rois));
  valid->setType(c10::TensorType::create(fake_valid));
  resume_guard.callback();
  resume_guard.callback = {};
  torch::jit::tracer::setValueTrace(fake_pooled, pooled);
  torch::jit::tracer::setValueTrace(fake_rois, rois);
  torch::jit::tracer::setValueTrace(fake_valid, valid);
  auto head = model.forward_box_head(fake_pooled);
  return {fake_rois, head[0], head[1], fake_valid};
}
onnx_export::ForwardFunction make_rcnn_onnx_forward(
    const std::shared_ptr<nn::ModelAdapter>& model,
    const VisionExportOptions& profile) {
  auto detector = std::dynamic_pointer_cast<models::rcnn::FasterRcnnImpl>(
      model->state().shared_module());
  if (!detector)
    throw CompatibilityError("Export requires the native RCNN model type");
  return [detector, profile](std::span<const Tensor> inputs) {
    return trace_rcnn(*detector, inputs[0], profile);
  };
}
void add_rcnn_onnx_lowerings(
    std::map<std::string, onnx_export::OperatorLowering>& lowerings) {
  lowerings[symbolic_name] = [](torch::jit::Node& source,
                                torch::jit::Graph& graph) {
    const auto tag = c10::Symbol::attr("onnx_kind");
    auto* lowered = graph.create(c10::Symbol::onnx(source.s(tag)),
                                 source.inputs(), source.outputs().size());
    lowered->copyAttributes(source);
    lowered->removeAttribute(tag);
    lowered->insertBefore(&source);
    std::vector<torch::jit::Value*> outputs;
    for (std::size_t i = 0; i < source.outputs().size(); ++i) {
      lowered->output(i)->copyMetadata(source.output(i));
      outputs.push_back(lowered->output(i));
    }
    return outputs;
  };
}
}  // namespace jyppx::liuding::vision::tasks::detail
