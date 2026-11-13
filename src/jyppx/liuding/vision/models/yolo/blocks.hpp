// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_MODELS_YOLO_PRIVATE_BLOCKS_HPP
#define JYPPX_LIUDING_VISION_MODELS_YOLO_PRIVATE_BLOCKS_HPP

#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <torch/nn.h>
#include <torch/nn/functional/upsampling.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace jyppx::liuding::models::yolo::detail {

using I = std::int64_t;
struct Unit : torch::nn::Module {
  virtual Tensor run(const Tensor& x) = 0;
};
using U = std::shared_ptr<Unit>;

struct Conv final : Unit {
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  bool activation;
  bool fused = false;
  Conv(I input, I output, I kernel = 1, I stride = 1, I groups = 1,
       bool active = true, I padding = -1)
      : activation(active) {
    conv = register_module(
        "conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input, output, kernel)
                              .stride(stride)
                              .groups(groups)
                              .padding(padding < 0 ? kernel / 2 : padding)
                              .bias(false)));
    bn = register_module(
        "bn",
        torch::nn::BatchNorm2d(
            torch::nn::BatchNorm2dOptions(output).eps(1e-3).momentum(0.03)));
  }
  Tensor run(const Tensor& x) override {
    auto y = fused ? conv(x) : bn(conv(x));
    return activation ? torch::silu(y) : y;
  }
  bool fuse_inference() {
    if (fused) return false;
    if (is_training() || bn->is_training())
      throw InvalidArgument("Conv/BN fusion requires an evaluation clone");
    torch::NoGradGuard guard;
    const auto& source = conv->options;
    auto options =
        torch::nn::Conv2dOptions(source.in_channels(), source.out_channels(),
                                 source.kernel_size())
            .stride(source.stride())
            .padding(source.padding())
            .dilation(source.dilation())
            .groups(source.groups())
            .padding_mode(source.padding_mode())
            .bias(true);
    auto replacement = torch::nn::Conv2d(options);
    replacement->to(conv->weight.device(), conv->weight.scalar_type());
    replacement->eval();
    auto scale = bn->weight / torch::sqrt(bn->running_var + bn->options.eps());
    replacement->weight.copy_(conv->weight * scale.view({-1, 1, 1, 1}));
    auto bias =
        conv->bias.defined() ? conv->bias : torch::zeros_like(bn->running_mean);
    replacement->bias.copy_((bias - bn->running_mean) * scale + bn->bias);
    conv = replace_module("conv", replacement);
    unregister_module("bn");
    bn = nullptr;
    fused = true;
    return true;
  }
};
inline U C(I in, I out, I kernel = 1, I stride = 1, I groups = 1,
           bool act = true, I padding = -1) {
  return std::make_shared<Conv>(in, out, kernel, stride, groups, act, padding);
}

struct Chain final : Unit {
  std::vector<U> children;
  explicit Chain(std::vector<U> units) : children(std::move(units)) {
    for (std::size_t n = 0; n < children.size(); ++n)
      register_module(std::to_string(n), children[n]);
  }
  Tensor run(const Tensor& x) override {
    auto y = x;
    for (auto& child : children) y = child->run(y);
    return y;
  }
};
struct Identity final : Unit {
  Tensor run(const Tensor& x) override { return x; }
};
struct Bottleneck final : Unit {
  U a, b;
  bool shortcut;
  Bottleneck(I in, I out, bool residual = true, double expansion = .5,
             I first_kernel = 3, I second_kernel = 3)
      : shortcut(residual && in == out) {
    const auto hidden = static_cast<I>(out * expansion);
    a = register_module("cv1", C(in, hidden, first_kernel));
    b = register_module("cv2", C(hidden, out, second_kernel));
  }
  Tensor run(const Tensor& x) override {
    auto y = b->run(a->run(x));
    return shortcut ? x + y : y;
  }
};

struct GhostConv final : Unit {
  U a, b;
  GhostConv(I in, I out, I kernel = 1, I stride = 1, bool act = true) {
    if (out % 2) throw InvalidArgument("Ghost output channels must be even");
    a = register_module("cv1", C(in, out / 2, kernel, stride, 1, act));
    b = register_module("cv2", C(out / 2, out / 2, 5, 1, out / 2, act));
  }
  Tensor run(const Tensor& x) override {
    auto y = a->run(x);
    return torch::cat({y, b->run(y)}, 1);
  }
};
struct GhostBottleneck final : Unit {
  U body, skip;
  GhostBottleneck(I in, I out, I kernel = 3, I stride = 1) {
    std::vector<U> parts{std::make_shared<GhostConv>(in, out / 2)};
    parts.push_back(stride == 2
                        ? C(out / 2, out / 2, kernel, stride, out / 2, false)
                        : std::make_shared<Identity>());
    parts.push_back(std::make_shared<GhostConv>(out / 2, out, 1, 1, false));
    body = register_module("conv", std::make_shared<Chain>(parts));
    skip = register_module(
        "shortcut", stride == 2 ? std::make_shared<Chain>(std::vector<U>{
                                      C(in, in, kernel, stride, in, false),
                                      C(in, out, 1, 1, 1, false)})
                                : U(std::make_shared<Identity>()));
  }
  Tensor run(const Tensor& x) override { return body->run(x) + skip->run(x); }
};

// q/k/v are explicit tensor algebra, shared by spatial and area attention.
struct Attention final : Unit {
  U qkv, projection, positional;
  I channels, heads, key_width, value_width, areas;
  Attention(I dim, I num_heads, bool area = false, I area_count = 1)
      : channels(dim),
        heads(num_heads),
        key_width(dim / num_heads / (area ? 1 : 2)),
        value_width(dim / num_heads),
        areas(area ? area_count : 1) {
    if (heads < 1 || channels % heads || key_width < 1 || areas < 1)
      throw InvalidArgument("Invalid attention channel/head/area contract");
    qkv = register_module(
        "qkv", C(dim, heads * (2 * key_width + value_width), 1, 1, 1, false));
    projection = register_module("proj", C(dim, dim, 1, 1, 1, false));
    positional =
        register_module("pe", C(dim, dim, area ? 7 : 3, 1, dim, false));
  }
  Tensor run(const Tensor& x) override {
    const I batch = x.size(0), h = x.size(2), w = x.size(3), n = h * w;
    if (n % areas || n / areas > 16384 ||
        batch * heads * (n / areas) > 268435456 / (n / areas))
      throw ResourceExhausted("Attention area or score element bound exceeded");
    auto packed = qkv->run(x)
                      .flatten(2)
                      .transpose(1, 2)
                      .reshape({batch * areas, n / areas, heads,
                                2 * key_width + value_width})
                      .permute({0, 2, 1, 3});
    auto q = packed.slice(-1, 0, key_width);
    auto k = packed.slice(-1, key_width, 2 * key_width);
    auto v = packed.slice(-1, 2 * key_width);
    auto scores = torch::matmul(q, k.transpose(-1, -2)) /
                  std::sqrt(static_cast<double>(key_width));
    auto weights =
        torch::softmax(scores.to(torch::kFloat32), -1).to(v.scalar_type());
    auto attended = torch::matmul(weights, v)
                        .permute({0, 2, 1, 3})
                        .reshape({batch, n, channels})
                        .transpose(1, 2)
                        .reshape({batch, channels, h, w});
    auto values = v.permute({0, 2, 1, 3})
                      .reshape({batch, n, channels})
                      .transpose(1, 2)
                      .reshape({batch, channels, h, w});
    return projection->run(attended + positional->run(values));
  }
};
struct AttentionBlock final : Unit {
  U attention, ffn;
  AttentionBlock(I dim, bool area = false, I count = 1, double ratio = 2.) {
    attention = register_module(
        "attn",
        std::make_shared<Attention>(
            dim, (std::max)(I{1}, dim / (area ? 32 : 64)), area, count));
    ffn = register_module(
        area ? "mlp" : "ffn",
        std::make_shared<Chain>(std::vector<U>{
            C(dim, static_cast<I>(dim * ratio)),
            C(static_cast<I>(dim * ratio), dim, 1, 1, 1, false)}));
  }
  Tensor run(const Tensor& x) override {
    auto y = x + attention->run(x);
    return y + ffn->run(y);
  }
};

struct C3 final : Unit {
  U a, b, tail, blocks;
  C3(I in, I out, I repeats, bool shortcut = true, double expansion = .5,
     bool ghost = false, I kernel = 1) {
    const auto hidden = static_cast<I>(out * expansion);
    a = register_module("cv1", C(in, hidden));
    b = register_module("cv2", C(in, hidden));
    tail = register_module("cv3", C(2 * hidden, out));
    std::vector<U> units;
    for (I i = 0; i < repeats; ++i)
      units.push_back(ghost
                          ? U(std::make_shared<GhostBottleneck>(hidden, hidden))
                          : U(std::make_shared<Bottleneck>(
                                hidden, hidden, shortcut, 1., kernel, 3)));
    blocks = register_module("m", std::make_shared<Chain>(units));
  }
  Tensor run(const Tensor& x) override {
    return tail->run(torch::cat({blocks->run(a->run(x)), b->run(x)}, 1));
  }
};
struct Csp final : Unit {
  U a, tail;
  torch::nn::ModuleList blocks{nullptr};
  I hidden;
  bool chain_only;
  Csp(I in, I out, I repeats, bool shortcut, const std::string& kind,
      double expansion = .5, bool c3k = false, bool attn = false)
      : hidden(static_cast<I>(out * expansion)), chain_only(kind == "C2") {
    a = register_module("cv1", C(in, 2 * hidden));
    tail =
        register_module("cv2", C((chain_only ? 2 : 2 + repeats) * hidden, out));
    blocks = register_module("m", torch::nn::ModuleList());
    for (I i = 0; i < repeats; ++i) {
      U unit;
      if (attn)
        unit = std::make_shared<Chain>(std::vector<U>{
            std::make_shared<Bottleneck>(hidden, hidden, shortcut),
            std::make_shared<AttentionBlock>(hidden)});
      else if (kind == "C3k2" && c3k)
        unit = std::make_shared<C3>(hidden, hidden, 2, shortcut, .5, false, 3);
      else
        unit = std::make_shared<Bottleneck>(hidden, hidden, shortcut,
                                            kind == "C3k2" ? .5 : 1.);
      blocks->push_back(unit);
    }
  }
  Tensor run(const Tensor& x) override {
    auto halves = a->run(x).split(hidden, 1);
    if (chain_only) {
      auto y = halves[0];
      for (auto& module : *blocks) y = module->as<Unit>()->run(y);
      return tail->run(torch::cat({y, halves[1]}, 1));
    }
    for (auto& module : *blocks)
      halves.push_back(module->as<Unit>()->run(halves.back()));
    return tail->run(torch::cat(halves, 1));
  }
};
struct Sppf final : Unit {
  U a, b;
  I kernel, repeat;
  bool residual;
  Sppf(I in, I out, I k = 5, I n = 3, bool shortcut = false,
       bool first_active = false)
      : kernel(k), repeat(n), residual(shortcut && in == out) {
    a = register_module("cv1", C(in, in / 2, 1, 1, 1, first_active));
    b = register_module("cv2", C((n + 1) * (in / 2), out));
  }
  Tensor run(const Tensor& x) override {
    std::vector<Tensor> ys{a->run(x)};
    for (I i = 0; i < repeat; ++i)
      ys.push_back(torch::max_pool2d(ys.back(), {kernel, kernel}, {1, 1},
                                     {kernel / 2, kernel / 2}));
    auto y = b->run(torch::cat(ys, 1));
    return residual ? x + y : y;
  }
};
struct C2Psa final : Unit {
  U a, b, blocks;
  I hidden;
  C2Psa(I in, I out, I repeats) : hidden(out / 2) {
    if (in != out)
      throw InvalidArgument("C2PSA requires equal input/output channels");
    a = register_module("cv1", C(in, 2 * hidden));
    b = register_module("cv2", C(2 * hidden, out));
    std::vector<U> units;
    for (I i = 0; i < repeats; ++i)
      units.push_back(std::make_shared<AttentionBlock>(hidden));
    blocks = register_module("m", std::make_shared<Chain>(units));
  }
  Tensor run(const Tensor& x) override {
    auto ys = a->run(x).split(hidden, 1);
    return b->run(torch::cat({ys[0], blocks->run(ys[1])}, 1));
  }
};
struct AreaCsp final : Unit {
  U a, b;
  torch::nn::ModuleList blocks{nullptr};
  Tensor gamma;
  AreaCsp(I in, I out, I repeats, bool area, I count, bool residual) {
    const auto hidden = out / 2;
    if (area && hidden % 32)
      throw InvalidArgument(
          "Area attention hidden channels must be divisible by 32");
    a = register_module("cv1", C(in, hidden));
    b = register_module("cv2", C((repeats + 1) * hidden, out));
    blocks = register_module("m", torch::nn::ModuleList());
    for (I i = 0; i < repeats; ++i) {
      blocks->push_back(
          area
              ? U(std::make_shared<Chain>(std::vector<U>{
                    std::make_shared<AttentionBlock>(hidden, true, count,
                                                     residual ? 1.2 : 2.),
                    std::make_shared<AttentionBlock>(hidden, true, count,
                                                     residual ? 1.2 : 2.)}))
              : U(std::make_shared<C3>(hidden, hidden, 2, true, .5, false, 3)));
    }
    if (area && residual && in == out)
      gamma = register_parameter("gamma", torch::full({out}, .01));
  }
  Tensor run(const Tensor& x) override {
    std::vector<Tensor> ys{a->run(x)};
    for (auto& m : *blocks) ys.push_back(m->as<Unit>()->run(ys.back()));
    auto y = b->run(torch::cat(ys, 1));
    return gamma.defined() ? x + gamma.view({1, -1, 1, 1}) * y : y;
  }
};

inline Tensor Resize(const Tensor& x, I h, I w) {
  return torch::nn::functional::interpolate(
      x, torch::nn::functional::InterpolateFuncOptions()
             .size(std::vector<I>{h, w})
             .mode(torch::kNearest));
}
}  // namespace jyppx::liuding::models::yolo::detail
#endif  // JYPPX_LIUDING_VISION_MODELS_YOLO_PRIVATE_BLOCKS_HPP
