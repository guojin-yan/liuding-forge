// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/models/registry.hpp>
#include <jyppx/liuding/vision/models/rcnn/faster_rcnn.hpp>
#include <jyppx/liuding/vision/ops/boxes.hpp>
#include <jyppx/liuding/vision/ops/roi.hpp>

#include <torch/nn.h>
#include <torch/nn/functional/upsampling.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
namespace jyppx::liuding::models::rcnn {
namespace {
using I = std::int64_t;
Tensor Clip(const Tensor& boxes, I height, I width) {
  return torch::stack(
      {boxes.select(1, 0).clamp(0, width), boxes.select(1, 1).clamp(0, height),
       boxes.select(1, 2).clamp(0, width), boxes.select(1, 3).clamp(0, height)},
      1);
}
Tensor Sample(const Tensor& indices, I maximum) {
  if (indices.size(0) <= maximum) return indices;
  return indices.index_select(
      0,
      torch::randperm(indices.size(0), indices.options()).slice(0, 0, maximum));
}
void Validate(const FasterRcnnOptions& o, const FeatureSpec& spec) {
  if (o.num_classes < 1 || o.num_classes > 10000 || o.fpn_channels < 8 ||
      o.fpn_channels > 2048 || o.representation_size < 8 ||
      o.representation_size > 4096 || spec.outputs.size() != 4 ||
      o.anchor_sizes.size() != 5 || o.aspect_ratios.empty() ||
      o.aspect_ratios.size() > 16 || o.rpn_pre_nms_train < 1 ||
      o.rpn_pre_nms_train > 16384 || o.rpn_pre_nms_eval < 1 ||
      o.rpn_pre_nms_eval > 16384 || o.rpn_post_nms_train < 1 ||
      o.rpn_post_nms_train > 4096 || o.rpn_post_nms_eval < 1 ||
      o.rpn_post_nms_eval > 4096 || o.roi_batch_size < 1 ||
      o.roi_batch_size > 4096 || o.max_targets < 1 ||
      o.max_anchor_elements < 1 || !std::isfinite(o.roi_positive_fraction) ||
      o.roi_positive_fraction < 0 || o.roi_positive_fraction > 1 ||
      !std::isfinite(o.rpn_nms_threshold) || o.rpn_nms_threshold < 0 ||
      o.rpn_nms_threshold > 1)
    throw InvalidArgument(
        "Invalid Faster R-CNN options or four-feature backbone contract");
  std::set<std::string> names;
  for (std::size_t i = 0; i < 4; ++i) {
    const auto& feature = spec.outputs[i];
    if (feature.layout != FeatureLayout::nchw || feature.channels < 1 ||
        feature.reduction != (4LL << i) || feature.name.empty() ||
        !names.insert(feature.name).second)
      throw InvalidArgument(
          "Faster R-CNN requires named NCHW C2-C5 features at strides "
          "4/8/16/32");
  }
  for (auto size : o.anchor_sizes)
    if (size < 1 || size > 65536)
      throw InvalidArgument("Invalid RPN anchor size");
  for (auto ratio : o.aspect_ratios)
    if (!std::isfinite(ratio) || ratio <= 0 || ratio > 100)
      throw InvalidArgument("Invalid RPN aspect ratio");
}
}  // namespace
std::vector<Tensor> FasterRcnnOutput::tensors() const {
  return {rpn_logits, rpn_deltas, anchors,       proposals,  class_logits,
          box_deltas, labels,     matched_boxes, image_sizes};
}
struct FasterRcnnImpl::Impl {
  nn::ModelAdapter backbone;
  FeatureSpec spec;
  FasterRcnnOptions options;
  torch::nn::ModuleList lateral{nullptr}, output{nullptr};
  torch::nn::Conv2d rpn_conv{nullptr}, rpn_objectness{nullptr},
      rpn_box{nullptr};
  torch::nn::Linear fc6{nullptr}, fc7{nullptr}, classifier{nullptr},
      regressor{nullptr};
  Impl(FasterRcnnImpl& owner, nn::ModelAdapter base, FeatureSpec contract,
       const FasterRcnnOptions& o)
      : backbone(std::move(base)), spec(std::move(contract)), options(o) {
    Validate(o, spec);
    owner.register_module("backbone", backbone.state().shared_module());
    lateral = owner.register_module("fpn_inner", torch::nn::ModuleList());
    output = owner.register_module("fpn_layer", torch::nn::ModuleList());
    for (const auto& feature : spec.outputs) {
      lateral->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(feature.channels, o.fpn_channels, 1)));
      output->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(o.fpn_channels, o.fpn_channels, 3)
              .padding(1)));
    }
    const auto anchors = static_cast<I>(o.aspect_ratios.size());
    rpn_conv = owner.register_module(
        "rpn_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                          o.fpn_channels, o.fpn_channels, 3)
                                          .padding(1)));
    rpn_objectness = owner.register_module(
        "rpn_objectness", torch::nn::Conv2d(torch::nn::Conv2dOptions(
                              o.fpn_channels, anchors, 1)));
    rpn_box = owner.register_module("rpn_bbox_pred",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                        o.fpn_channels, anchors * 4, 1)));
    fc6 = owner.register_module(
        "roi_fc6",
        torch::nn::Linear(o.fpn_channels * 7 * 7, o.representation_size));
    fc7 = owner.register_module(
        "roi_fc7",
        torch::nn::Linear(o.representation_size, o.representation_size));
    classifier = owner.register_module(
        "roi_cls_score",
        torch::nn::Linear(o.representation_size, o.num_classes + 1));
    regressor = owner.register_module(
        "roi_bbox_pred",
        torch::nn::Linear(o.representation_size, (o.num_classes + 1) * 4));
    torch::NoGradGuard guard;
    for (auto* layer : {&rpn_conv, &rpn_objectness, &rpn_box}) {
      torch::nn::init::normal_((*layer)->weight, 0, .01);
      (*layer)->bias.zero_();
    }
    torch::nn::init::normal_(classifier->weight, 0, .01);
    classifier->bias.zero_();
    torch::nn::init::normal_(regressor->weight, 0, .001);
    regressor->bias.zero_();
  }
  std::vector<Tensor> pyramid(const Tensor& images) {
    auto result = backbone.invoke(nn::Invocation{{images}, {}}).value;
    if (!result.isTuple() || result.toTupleRef().elements().size() != 4)
      throw InvalidArgument(
          "Backbone feature invoker must return the declared four-tensor "
          "tuple");
    std::vector<Tensor> features;
    for (std::size_t i = 0; i < 4; ++i) {
      auto x = result.toTupleRef().elements()[i].toTensor();
      if (x.dim() != 4 || x.size(0) != images.size(0) ||
          x.size(1) != spec.outputs[i].channels ||
          x.size(2) != (images.size(2) + spec.outputs[i].reduction - 1) /
                           spec.outputs[i].reduction ||
          x.size(3) != (images.size(3) + spec.outputs[i].reduction - 1) /
                           spec.outputs[i].reduction)
        throw InvalidArgument(
            "Backbone feature tensor violates its named channel/spatial "
            "contract");
      features.push_back(lateral->at<torch::nn::Conv2dImpl>(i).forward(x));
    }
    auto inner = features.back();
    features[3] = output->at<torch::nn::Conv2dImpl>(3).forward(inner);
    for (int i = 2; i >= 0; --i) {
      inner = features[i] +
              torch::nn::functional::interpolate(
                  inner, torch::nn::functional::InterpolateFuncOptions()
                             .size(std::vector<I>{features[i].size(2),
                                                  features[i].size(3)})
                             .mode(torch::kNearest));
      features[i] = output->at<torch::nn::Conv2dImpl>(i).forward(inner);
    }
    features.push_back(torch::max_pool2d(features.back(), {1, 1}, {2, 2}));
    return features;
  }
  Tensor anchors(const Tensor& feature, std::size_t level) const {
    const auto stride = 4LL << level;
    auto xs =
        torch::arange(feature.size(3), feature.options().requires_grad(false)) *
        stride;
    auto ys =
        torch::arange(feature.size(2), feature.options().requires_grad(false)) *
        stride;
    auto xx =
        xs.view({1, -1}).expand({feature.size(2), feature.size(3)}).flatten();
    auto yy =
        ys.view({-1, 1}).expand({feature.size(2), feature.size(3)}).flatten();
    auto shift = torch::stack({xx, yy, xx, yy}, 1);
    std::vector<float> bases;
    for (auto ratio : options.aspect_ratios) {
      const double width = options.anchor_sizes[level] / std::sqrt(ratio),
                   height = options.anchor_sizes[level] * std::sqrt(ratio);
      for (double v : {-width / 2, -height / 2, width / 2, height / 2})
        bases.push_back(static_cast<float>(std::nearbyint(v)));
    }
    auto base = torch::tensor(bases, feature.options()).view({-1, 4});
    return (shift.unsqueeze(1) + base.unsqueeze(0)).reshape({-1, 4});
  }
};
FasterRcnnImpl::FasterRcnnImpl(nn::ModelAdapter backbone, FeatureSpec features,
                               const FasterRcnnOptions& options)
    : impl_(std::make_unique<Impl>(*this, std::move(backbone),
                                   std::move(features), options)) {}
FasterRcnnImpl::~FasterRcnnImpl() = default;
const FasterRcnnOptions& FasterRcnnImpl::options() const noexcept {
  return impl_->options;
}

std::vector<Tensor> FasterRcnnImpl::forward_rpn(const Tensor& images) {
  if (is_training())
    throw InvalidArgument("Export RPN stages require evaluation mode");
  auto& s = *impl_;
  auto features = s.pyramid(images);
  std::vector<Tensor> result(features.begin(), features.begin() + 4), logits,
      deltas;
  for (const auto& feature : features) {
    auto shared = torch::relu(s.rpn_conv(feature));
    logits.push_back(s.rpn_objectness(shared)
                         .permute({0, 2, 3, 1})
                         .reshape({images.size(0), -1}));
    deltas.push_back(s.rpn_box(shared)
                         .permute({0, 2, 3, 1})
                         .reshape({images.size(0), -1, 4}));
  }
  result.insert(result.end(), logits.begin(), logits.end());
  result.insert(result.end(), deltas.begin(), deltas.end());
  return result;
}
std::vector<Tensor> FasterRcnnImpl::forward_box_head(const Tensor& pooled) {
  auto& s = *impl_;
  if (is_training() || pooled.dim() != 4 ||
      pooled.size(1) != s.options.fpn_channels || pooled.size(2) != 7 ||
      pooled.size(3) != 7)
    throw InvalidArgument("Export ROI head requires eval [R,C,7,7] features");
  auto representation =
      torch::relu(s.fc7(torch::relu(s.fc6(pooled.flatten(1)))));
  return {s.classifier(representation), s.regressor(representation)};
}

FasterRcnnOutput FasterRcnnImpl::forward(const Tensor& images,
                                         const Tensor& targets,
                                         const Tensor& input_sizes) {
  auto& s = *impl_;
  const auto& o = s.options;
  if (!images.defined() || images.dim() != 4 || !images.is_floating_point() ||
      images.size(0) < 1 || images.size(1) != 3 || images.size(2) < 32 ||
      images.size(3) < 32 || images.size(2) % 32 || images.size(3) % 32)
    throw InvalidArgument(
        "Faster R-CNN requires floating RGB NCHW padded to a multiple of 32");
  auto sizes =
      input_sizes.defined()
          ? input_sizes
          : torch::tensor(std::vector<I>{images.size(2), images.size(3)},
                          images.options())
                .view({1, 2})
                .expand({images.size(0), 2});
  if (sizes.dim() != 2 || sizes.size(0) != images.size(0) ||
      sizes.size(1) != 2 || !torch::isfinite(sizes).all().item<bool>())
    throw InvalidArgument("Faster R-CNN image_sizes must be finite [B,2]");
  auto host_sizes = sizes.to(torch::kCPU, torch::kInt64).contiguous();
  for (I b = 0; b < images.size(0); ++b)
    if (host_sizes[b][0].item<I>() < 1 || host_sizes[b][1].item<I>() < 1 ||
        host_sizes[b][0].item<I>() > images.size(2) ||
        host_sizes[b][1].item<I>() > images.size(3))
      throw InvalidArgument(
          "Faster R-CNN actual image size exceeds padded canvas");
  if (targets.defined() &&
      (targets.dim() != 2 || targets.size(1) != 6 ||
       targets.device() != images.device() || !targets.is_floating_point() ||
       targets.size(0) > o.max_targets ||
       !torch::isfinite(targets).all().item<bool>()))
    throw InvalidArgument(
        "Faster R-CNN targets require bounded floating [T,6]");
  if (targets.defined() && targets.numel() &&
      (!(targets.slice(1, 0, 2) == targets.slice(1, 0, 2).round())
            .all()
            .item<bool>() ||
       targets.select(1, 0).min().item<double>() < 0 ||
       targets.select(1, 0).max().item<double>() >= images.size(0) ||
       targets.select(1, 1).min().item<double>() < 0 ||
       targets.select(1, 1).max().item<double>() >= o.num_classes ||
       !(targets.slice(1, 4, 6) > targets.slice(1, 2, 4)).all().item<bool>()))
    throw InvalidArgument(
        "Invalid Faster R-CNN target indices or positive box extents");
  auto features = s.pyramid(images);
  std::vector<Tensor> logits, deltas, anchors, proposals;
  std::vector<I> counts;
  I total_anchors = 0;
  for (std::size_t i = 0; i < features.size(); ++i) {
    const auto count = features[i].size(2) * features[i].size(3) *
                       static_cast<I>(o.aspect_ratios.size());
    if (count > o.max_anchor_elements / images.size(0) / 4 - total_anchors)
      throw ResourceExhausted("RPN raw anchor element bound exceeded");
    total_anchors += count;
    counts.push_back(count);
    auto shared = torch::relu(s.rpn_conv(features[i]));
    logits.push_back(s.rpn_objectness(shared)
                         .permute({0, 2, 3, 1})
                         .reshape({images.size(0), -1}));
    deltas.push_back(s.rpn_box(shared)
                         .permute({0, 2, 3, 1})
                         .reshape({images.size(0), -1, 4}));
    anchors.push_back(s.anchors(features[i], i));
  }
  {
    torch::NoGradGuard guard;
    for (I b = 0; b < images.size(0); ++b) {
      const auto height = host_sizes[b][0].item<I>(),
                 width = host_sizes[b][1].item<I>();
      std::vector<Tensor> level_boxes, level_scores;
      for (std::size_t level = 0; level < features.size(); ++level) {
        const auto k =
            (std::min)(counts[level], is_training() ? o.rpn_pre_nms_train
                                                    : o.rpn_pre_nms_eval);
        auto top = logits[level][b].topk(k);
        auto indices = std::get<1>(top);
        auto boxes = Clip(vision_ops::decode_box_deltas(
                              anchors[level].index_select(0, indices),
                              deltas[level][b].index_select(0, indices)),
                          height, width);
        auto scores = std::get<0>(top).sigmoid();
        auto valid = ((boxes.select(1, 2) - boxes.select(1, 0) >= .001) &
                      (boxes.select(1, 3) - boxes.select(1, 1) >= .001) &
                      torch::isfinite(boxes).all(1) & torch::isfinite(scores))
                         .nonzero()
                         .flatten();
        boxes = boxes.index_select(0, valid);
        scores = scores.index_select(0, valid);
        auto keep =
            vision_ops::nms(
                boxes.to(torch::kCPU, torch::kFloat32),
                scores.to(torch::kCPU, torch::kFloat32),
                {.iou_threshold = o.rpn_nms_threshold,
                 .max_candidates = o.rpn_pre_nms_train > o.rpn_pre_nms_eval
                                       ? o.rpn_pre_nms_train
                                       : o.rpn_pre_nms_eval,
                 .max_output = is_training() ? o.rpn_post_nms_train
                                             : o.rpn_post_nms_eval})
                .to(images.device());
        level_boxes.push_back(boxes.index_select(0, keep));
        level_scores.push_back(scores.index_select(0, keep));
      }
      auto boxes = torch::cat(level_boxes), scores = torch::cat(level_scores);
      auto selected = std::get<1>(scores.topk(
          (std::min)(scores.size(0), is_training() ? o.rpn_post_nms_train
                                                   : o.rpn_post_nms_eval)));
      boxes = boxes.index_select(0, selected);
      proposals.push_back(torch::cat(
          {torch::full({boxes.size(0), 1}, b, boxes.options()), boxes}, 1));
    }
  }
  auto rois = torch::cat(proposals);
  auto labels =
      torch::full({rois.size(0)}, -1, images.options().dtype(torch::kInt64));
  auto matched = torch::zeros({rois.size(0), 4}, images.options());
  if (is_training() && targets.defined()) {
    torch::NoGradGuard guard;
    std::vector<Tensor> sampled_rois, sampled_labels, sampled_boxes;
    for (I b = 0; b < images.size(0); ++b) {
      auto gt = targets.index({targets.select(1, 0) == b});
      auto boxes = rois.index({rois.select(1, 0) == b}).slice(1, 1, 5);
      if (gt.size(0)) boxes = torch::cat({boxes, gt.slice(1, 2, 6)});
      auto assigned_labels = torch::zeros({boxes.size(0)}, labels.options());
      auto assigned_boxes = torch::zeros_like(boxes);
      if (gt.size(0)) {
        auto iou = vision_ops::pairwise_box_iou(boxes, gt.slice(1, 2, 6),
                                                {.max_pairs = 16777216});
        auto best = iou.max(1);
        auto ids = std::get<1>(best);
        assigned_labels =
            gt.select(1, 1).to(torch::kInt64).index_select(0, ids) + 1;
        assigned_labels = torch::where(std::get<0>(best) >= .5, assigned_labels,
                                       torch::zeros_like(assigned_labels));
        assigned_boxes = gt.slice(1, 2, 6).index_select(0, ids);
      }
      auto pos =
          Sample((assigned_labels > 0).nonzero().flatten(),
                 static_cast<I>(o.roi_batch_size * o.roi_positive_fraction));
      auto neg = Sample((assigned_labels == 0).nonzero().flatten(),
                        o.roi_batch_size - pos.size(0));
      auto keep = torch::cat({pos, neg});
      boxes = boxes.index_select(0, keep);
      sampled_rois.push_back(torch::cat(
          {torch::full({boxes.size(0), 1}, b, boxes.options()), boxes}, 1));
      sampled_labels.push_back(assigned_labels.index_select(0, keep));
      sampled_boxes.push_back(assigned_boxes.index_select(0, keep));
    }
    rois = torch::cat(sampled_rois);
    labels = torch::cat(sampled_labels);
    matched = torch::cat(sampled_boxes);
  }
  features.pop_back();
  auto pooled = vision_ops::multiscale_roi_align(features, rois, {4, 8, 16, 32},
                                                 {.max_rois = 32768});
  auto representation =
      torch::relu(s.fc7(torch::relu(s.fc6(pooled.flatten(1)))));
  return {torch::cat(logits, 1),
          torch::cat(deltas, 1),
          torch::cat(anchors),
          rois,
          s.classifier(representation),
          s.regressor(representation),
          labels,
          matched,
          sizes};
}
nn::ModelAdapter create_faster_rcnn_resnet50_fpn(const FasterRcnnOptions& o) {
  models::ModelOptions base_options;
  base_options.resnet.num_classes = 0;
  auto backbone = models::create_feature_extractor("resnet50", base_options);
  auto spec = models::model_feature_spec("resnet50", base_options);
  auto model =
      std::make_shared<FasterRcnnImpl>(std::move(backbone), std::move(spec), o);
  nn::InvocationSchema schema;
  schema.positional = {
      ValueSchema::tensor(TensorSchema{.rank = 4}, "images"),
      ValueSchema::tensor(TensorSchema{.rank = 2}, "targets"),
      ValueSchema::tensor(TensorSchema{.rank = 2}, "image_sizes")};
  std::vector<ValueSchema> outputs;
  for (int i = 0; i < 9; ++i) outputs.push_back(ValueSchema::tensor());
  return nn::adapt_module(
      model,
      [](FasterRcnnImpl& concrete, const nn::Invocation& input) {
        auto output = concrete.forward(input.positional[0].toTensor(),
                                       input.positional[1].toTensor(),
                                       input.positional[2].toTensor());
        std::vector<c10::IValue> values;
        for (auto& tensor : output.tensors()) values.emplace_back(tensor);
        return c10::IValue(c10::ivalue::Tuple::create(std::move(values)));
      },
      std::move(schema),
      ValueSchema::tuple(std::move(outputs), "faster_rcnn_raw"));
}
}  // namespace jyppx::liuding::models::rcnn
