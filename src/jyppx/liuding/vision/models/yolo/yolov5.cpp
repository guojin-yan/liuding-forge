// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// Architecture/anchor constants: original YOLOv5 v7.0 at
// 915bbf294bb74c859f0b41f1c23bc395014ea679, models/yolov5*.yaml and
// models/hub/yolov5*6.yaml. GPL-3.0 configuration attribution is retained in
// third_party/licenses/MODEL_PROVENANCE.md and yolov5-GPL-3.0.txt.
#include "blocks.hpp"

#include <jyppx/liuding/vision/models/yolo/yolov5.hpp>
namespace jyppx::liuding::models::yolo {
namespace {
using namespace detail;
struct Scale {
  double depth, width;
};
Scale ResolveV5(Capacity capacity) {
  switch (capacity) {
    case Capacity::nano_capacity:
      return {.33, .25};
    case Capacity::small_capacity:
      return {.33, .50};
    case Capacity::medium_capacity:
      return {.67, .75};
    case Capacity::large_capacity:
      return {1., 1.};
    case Capacity::extra_large_capacity:
      return {1.33, 1.25};
  }
  throw InvalidArgument("Unknown original YOLOv5 capacity");
}
struct Node {
  std::vector<int> from;
  U unit;
  bool concat = false, upsample = false;
};
}  // namespace
std::vector<std::array<double, 6>> original_v5_anchors(bool p6) {
  if (p6)
    return {{19, 27, 44, 40, 38, 94},
            {96, 68, 86, 152, 180, 137},
            {140, 301, 303, 264, 238, 542},
            {436, 615, 739, 380, 925, 792}};
  return {{10, 13, 16, 30, 33, 23},
          {30, 61, 62, 45, 59, 119},
          {116, 90, 156, 198, 373, 326}};
}
struct OriginalV5Impl::Impl {
  OriginalV5Options options;
  torch::nn::ModuleList layers{nullptr}, head{nullptr};
  std::vector<Node> graph;
  std::vector<int> head_from;
  Impl(OriginalV5Impl& owner, const OriginalV5Options& requested)
      : options(requested) {
    const auto scale = ResolveV5(options.capacity);
    if (options.num_classes < 1 || options.num_classes > 10000 ||
        options.input_channels < 1 || options.input_channels > 16)
      throw InvalidArgument("Invalid original YOLOv5 class/channel count");
    if (options.anchors.empty())
      options.anchors = original_v5_anchors(options.p6);
    if (options.anchors.size() != (options.p6 ? 4U : 3U))
      throw InvalidArgument("Original YOLOv5 anchor level count mismatch");
    for (const auto& level : options.anchors)
      for (auto v : level)
        if (!std::isfinite(v) || v <= 0 || v > 65536)
          throw InvalidArgument("Invalid original YOLOv5 pixel anchor");
    const auto ch = [&](I base) {
      return static_cast<I>(std::ceil(base * scale.width / 8)) * 8;
    };
    const auto rep = [&](I n) {
      return (std::max)(I{1}, static_cast<I>(std::nearbyint(n * scale.depth)));
    };
    layers = owner.register_module("model", torch::nn::ModuleList());
    const auto add = [&](U unit, std::vector<int> from = {-1},
                         bool concat = false, bool upsample = false) {
      graph.push_back({std::move(from), unit, concat, upsample});
      layers->push_back(unit);
      return static_cast<int>(graph.size() - 1);
    };
    add(C(options.input_channels, ch(64), 6, 2, 1, true, 2));
    const std::vector<I> widths = options.p6
                                      ? std::vector<I>{128, 256, 512, 768, 1024}
                                      : std::vector<I>{128, 256, 512, 1024};
    const std::vector<I> depths{3, 6, 9, 3, 3};
    std::vector<int> stages;
    I previous = ch(64);
    for (std::size_t i = 0; i < widths.size(); ++i) {
      add(C(previous, ch(widths[i]), 3, 2));
      stages.push_back(add(
          std::make_shared<C3>(ch(widths[i]), ch(widths[i]), rep(depths[i]))));
      previous = ch(widths[i]);
    }
    add(std::make_shared<Sppf>(previous, previous, 5, 3, false, true));
    std::vector<int> reduced;
    const int last = static_cast<int>(widths.size()) - 1;
    for (int i = last - 1; i >= 1; --i) {
      reduced.push_back(add(C(previous, ch(widths[i]))));
      add(std::make_shared<Identity>(), {-1}, false, true);
      add(std::make_shared<Identity>(), {-1, stages[i]}, true);
      add(std::make_shared<C3>(2 * ch(widths[i]), ch(widths[i]), rep(3),
                               false));
      previous = ch(widths[i]);
    }
    head_from.push_back(static_cast<int>(graph.size() - 1));
    for (int i = 1; i < last; ++i) {
      add(C(previous, previous, 3, 2));
      add(std::make_shared<Identity>(), {-1, reduced[reduced.size() - i]},
          true);
      add(std::make_shared<C3>(2 * previous, ch(widths[i + 1]), rep(3), false));
      previous = ch(widths[i + 1]);
      head_from.push_back(static_cast<int>(graph.size() - 1));
    }
    head = torch::nn::ModuleList();
    layers->push_back(head);
    torch::NoGradGuard guard;
    for (std::size_t i = 0; i < head_from.size(); ++i) {
      auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(
          ch(widths[i + 1]), 3 * (5 + options.num_classes), 1));
      auto bias = conv->bias.view({3, -1});
      bias.select(1, 4).add_(std::log(8. / std::pow(640. / (8LL << i), 2)));
      bias.slice(1, 5).add_(std::log(.6 / (options.num_classes - .999999)));
      head->push_back(conv);
    }
    std::vector<float> anchor_values;
    for (const auto& level : options.anchors)
      for (auto v : level) anchor_values.push_back(static_cast<float>(v));
    owner.register_buffer(
        "anchors_pixels",
        torch::tensor(anchor_values)
            .view({static_cast<I>(options.anchors.size()), 3, 2}));
  }
};
OriginalV5Impl::OriginalV5Impl(const OriginalV5Options& options)
    : impl_(std::make_unique<Impl>(*this, options)) {}
OriginalV5Impl::~OriginalV5Impl() = default;
const OriginalV5Options& OriginalV5Impl::options() const noexcept {
  return impl_->options;
}
Tensor OriginalV5Impl::forward(const Tensor& images) {
  const auto divisor = options().p6 ? 64 : 32;
  if (!images.defined() || images.dim() != 4 || !images.is_floating_point() ||
      images.size(0) < 1 || images.size(1) != options().input_channels ||
      images.size(2) < divisor || images.size(3) < divisor ||
      images.size(2) % divisor || images.size(3) % divisor)
    throw InvalidArgument(
        "Original YOLOv5 expects floating stride-aligned NCHW images");
  std::vector<Tensor> cache;
  for (const auto& node : impl_->graph) {
    const auto get = [&](int from) {
      return from == -1 ? (cache.empty() ? images : cache.back())
                        : cache.at(from);
    };
    auto x = get(node.from[0]);
    if (node.concat)
      x = torch::cat({x, get(node.from[1])}, 1);
    else if (node.upsample)
      x = Resize(x, 2 * x.size(2), 2 * x.size(3));
    else
      x = node.unit->run(x);
    cache.push_back(x);
  }
  std::vector<Tensor> outputs;
  for (std::size_t i = 0; i < impl_->head_from.size(); ++i) {
    auto value = impl_->head->at<torch::nn::Conv2dImpl>(i).forward(
        cache[impl_->head_from[i]]);
    outputs.push_back(
        value
            .view({images.size(0), 3, 5 + options().num_classes, value.size(2),
                   value.size(3)})
            .permute({0, 1, 3, 4, 2})
            .reshape({images.size(0), -1, 5 + options().num_classes}));
  }
  return torch::cat(outputs, 1);
}
nn::ModelAdapter create_original_v5(const OriginalV5Options& options) {
  return nn::adapt_tensor_module(std::make_shared<OriginalV5Impl>(options),
                                 TensorSchema{.rank = 4},
                                 TensorSchema{.rank = 3});
}
std::string original_v5_model_id(const OriginalV5Options& options) {
  ResolveV5(options.capacity);
  return "yolo.original.v5." +
         std::string(1, "nsmlx"[static_cast<std::size_t>(options.capacity)]) +
         (options.p6 ? ".p6" : ".p5");
}
}  // namespace jyppx::liuding::models::yolo
