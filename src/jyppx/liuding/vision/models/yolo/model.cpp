// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "blocks.hpp"

#include <jyppx/liuding/vision/models/yolo/model.hpp>

#include <array>
#include <limits>

namespace jyppx::liuding::models::yolo {
namespace {
using namespace detail;
struct Scale {
  double depth, width;
  I cap;
};
struct LayerSpec {
  std::vector<int> from;
  int repeats;
  std::string kind;
  std::vector<double> args;
};
struct GraphSpec {
  std::string key, path, sha;
  std::array<Scale, 5> scales;
  std::vector<LayerSpec> layers;
};
#include "definitions.inc"

std::string FamilyName(Family family) {
  switch (family) {
    case Family::v8:
      return "v8";
    case Family::yolo11:
      return "11";
    case Family::yolo12:
      return "12";
    case Family::yolo26:
      return "26";
  }
  throw InvalidArgument("Unknown YOLO family");
}
std::size_t CapacityIndex(Capacity capacity) {
  const auto index = static_cast<std::size_t>(capacity);
  if (index > 4) throw InvalidArgument("Unknown YOLO capacity");
  return index;
}
std::string TaskSuffix(Task task) {
  switch (task) {
    case Task::detect:
      return "";
    case Task::classify:
      return "-cls";
    case Task::instance_segment:
      return "-seg";
    case Task::pose:
      return "-pose";
    case Task::obb:
      return "-obb";
  }
  throw InvalidArgument("Unknown YOLO task");
}
std::string VariantSuffix(Variant variant) {
  switch (variant) {
    case Variant::p5:
      return "";
    case Variant::p2:
      return "-p2";
    case Variant::p6:
      return "-p6";
    case Variant::ghost_p5:
      return "-ghost";
    case Variant::ghost_p2:
      return "-ghost-p2";
    case Variant::ghost_p6:
      return "-ghost-p6";
  }
  throw InvalidArgument("Unknown YOLO variant");
}
const GraphSpec& Resolve(const ModelOptions& o) {
  CapacityIndex(o.capacity);
  if (o.num_classes < 1 || o.num_classes > 10000 || o.input_channels < 1 ||
      o.input_channels > 16 || o.mask_channels < 1 || o.mask_channels > 256 ||
      o.prototype_channels < 8 || o.prototype_channels > 2048 ||
      o.keypoints < 1 || o.keypoints > 256 ||
      (o.keypoint_dimensions != 2 && o.keypoint_dimensions != 3) ||
      !std::isfinite(o.dropout) || o.dropout < 0 || o.dropout >= 1 ||
      o.reg_max < 0 || o.reg_max > 32 ||
      (o.family == Family::yolo26 ? (o.reg_max != 0 && o.reg_max != 1)
                                  : o.reg_max == 1))
    throw InvalidArgument("Invalid YOLO classes/channels/head/dropout options");
  const auto key = "yolo" + FamilyName(o.family) + TaskSuffix(o.task) +
                   VariantSuffix(o.variant);
  for (const auto& graph : Graphs())
    if (graph.key == key) return graph;
  throw InvalidArgument(
      "YOLO task/variant is absent from the frozen catalog: " + key);
}
I Channel(double base, Scale scale) {
  return static_cast<I>(
             std::ceil((std::min)(base, static_cast<double>(scale.cap)) *
                       scale.width / 8.)) *
         8;
}
I Repeats(int count, Scale scale) {
  return count > 1
             ? (std::max)(I{1},
                          static_cast<I>(std::nearbyint(count * scale.depth)))
             : count;
}

struct OutputConv final : Unit {
  torch::nn::Conv2d layer{nullptr};
  OutputConv(I in, I out) {
    layer = register_module(
        "output", torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 1)));
  }
  Tensor run(const Tensor& x) override { return layer(x); }
};
struct Classification final : Unit {
  U conv;
  torch::nn::Dropout drop{nullptr};
  torch::nn::Linear linear{nullptr};
  Classification(I in, const ModelOptions& o) {
    conv = register_module("conv", C(in, 1280));
    drop = register_module("drop", torch::nn::Dropout(o.dropout));
    linear = register_module("linear", torch::nn::Linear(1280, o.num_classes));
  }
  Tensor run(const Tensor& x) override {
    return linear(drop(conv->run(x).mean({2, 3})));
  }
};

// Each head branch has separate state. Detached features isolate one-to-one
// optimization while preserving its own parameter gradients.
struct DenseHead final : torch::nn::Module {
  torch::nn::ModuleList boxes{nullptr}, classes{nullptr}, extra{nullptr},
      kpts{nullptr}, sigma{nullptr};
  ModelOptions options;
  DenseHead(const std::vector<I>& ch, const std::vector<I>& strides,
            const ModelOptions& o, I bins, bool legacy)
      : options(o) {
    boxes = register_module("cv2", torch::nn::ModuleList());
    classes = register_module("cv3", torch::nn::ModuleList());
    extra = register_module("cv4", torch::nn::ModuleList());
    kpts = register_module("cv4_kpts", torch::nn::ModuleList());
    sigma = register_module("cv4_sigma", torch::nn::ModuleList());
    const I box_width = (std::max)({I{16}, ch[0] / 4, 4 * bins});
    const I cls_width = (std::max)(ch[0], (std::min)(o.num_classes, I{100}));
    const I e = o.task == Task::instance_segment ? o.mask_channels
                : o.task == Task::pose ? o.keypoints * o.keypoint_dimensions
                : o.task == Task::obb  ? 1
                                       : 0;
    const bool pose26 = o.family == Family::yolo26 && o.task == Task::pose;
    const I ew = (std::max)(
        ch[0] / 4, pose26 ? o.keypoints * (o.keypoint_dimensions + 2) : e);
    for (std::size_t i = 0; i < ch.size(); ++i) {
      auto bout = std::make_shared<OutputConv>(box_width, 4 * bins);
      auto cout = std::make_shared<OutputConv>(cls_width, o.num_classes);
      {
        torch::NoGradGuard guard;
        bout->layer->bias.fill_(1.);
        cout->layer->bias.fill_(
            std::log(5. / o.num_classes / std::pow(640. / strides[i], 2.)));
      }
      boxes->push_back(std::make_shared<Chain>(std::vector<U>{
          C(ch[i], box_width, 3), C(box_width, box_width, 3), bout}));
      std::vector<U> classifier;
      if (legacy)
        classifier = {C(ch[i], cls_width, 3), C(cls_width, cls_width, 3), cout};
      else
        classifier = {std::make_shared<Chain>(std::vector<U>{
                          C(ch[i], ch[i], 3, 1, ch[i]), C(ch[i], cls_width)}),
                      std::make_shared<Chain>(std::vector<U>{
                          C(cls_width, cls_width, 3, 1, cls_width),
                          C(cls_width, cls_width)}),
                      cout};
      classes->push_back(std::make_shared<Chain>(classifier));
      if (e) {
        std::vector<U> layers{C(ch[i], ew, 3), C(ew, ew, 3)};
        if (!pose26) layers.push_back(std::make_shared<OutputConv>(ew, e));
        extra->push_back(std::make_shared<Chain>(layers));
        if (pose26) {
          kpts->push_back(std::make_shared<OutputConv>(ew, e));
          sigma->push_back(std::make_shared<OutputConv>(ew, 2 * o.keypoints));
        }
      }
    }
  }
  Tensor run(const std::vector<Tensor>& x) {
    std::vector<Tensor> packed;
    for (std::size_t i = 0; i < x.size(); ++i) {
      std::vector<Tensor> parts{boxes[i]->as<Unit>()->run(x[i]),
                                classes[i]->as<Unit>()->run(x[i])};
      if (!extra->is_empty()) {
        auto feature = extra[i]->as<Unit>()->run(x[i]);
        if (!kpts->is_empty()) {
          parts.push_back(kpts[i]->as<Unit>()->run(feature));
          parts.push_back(sigma[i]->as<Unit>()->run(feature));
        } else
          parts.push_back(feature);
      }
      packed.push_back(torch::cat(parts, 1).flatten(2));
    }
    return torch::cat(packed, 2);
  }
};

struct Prototypes final : torch::nn::Module {
  U cv1, cv2, cv3, fuse, semantic;
  torch::nn::ConvTranspose2d upsample{nullptr};
  torch::nn::ModuleList refine{nullptr};
  bool modern;
  Prototypes(const std::vector<I>& ch, const ModelOptions& o, Scale scale)
      : modern(o.family == Family::yolo26) {
    const auto mid = Channel(o.prototype_channels, scale);
    refine = register_module("feat_refine", torch::nn::ModuleList());
    if (modern) {
      for (std::size_t i = 1; i < ch.size(); ++i)
        refine->push_back(C(ch[i], ch[0]));
      fuse = register_module("feat_fuse", C(ch[0], mid, 3));
      semantic = register_module(
          "semseg", std::make_shared<Chain>(std::vector<U>{
                        C(ch[0], mid, 3), C(mid, mid, 3),
                        std::make_shared<OutputConv>(mid, o.num_classes)}));
    }
    cv1 = register_module("cv1", C(modern ? mid : ch[0], mid, 3));
    upsample = register_module(
        "upsample",
        torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(mid, mid, 2).stride(2)));
    cv2 = register_module("cv2", C(mid, mid, 3));
    cv3 = register_module("cv3", C(mid, o.mask_channels));
  }
  std::vector<Tensor> run(const std::vector<Tensor>& x) {
    auto feature = x[0];
    if (modern)
      for (std::size_t i = 1; i < x.size(); ++i)
        feature = feature + Resize(refine[i - 1]->as<Unit>()->run(x[i]),
                                   x[0].size(2), x[0].size(3));
    auto output = cv3->run(
        cv2->run(upsample(cv1->run(modern ? fuse->run(feature) : feature))));
    if (modern) return {output, semantic->run(feature)};
    return {output};
  }
};
struct PoseFlow final : torch::nn::Module {
  torch::nn::ModuleList scales{nullptr}, translations{nullptr};
  Tensor mask;
  PoseFlow() {
    scales = register_module("s", torch::nn::ModuleList());
    translations = register_module("t", torch::nn::ModuleList());
    mask = register_buffer("mask", torch::tensor({{0.f, 1.f},
                                                  {1.f, 0.f},
                                                  {0.f, 1.f},
                                                  {1.f, 0.f},
                                                  {0.f, 1.f},
                                                  {1.f, 0.f}}));
    register_buffer("loc", torch::zeros({2}));
    register_buffer("cov", torch::eye(2));
    for (int i = 0; i < 6; ++i) {
      scales->push_back(
          torch::nn::Sequential(torch::nn::Linear(2, 64), torch::nn::SiLU(),
                                torch::nn::Linear(64, 64), torch::nn::SiLU(),
                                torch::nn::Linear(64, 2), torch::nn::Tanh()));
      translations->push_back(
          torch::nn::Sequential(torch::nn::Linear(2, 64), torch::nn::SiLU(),
                                torch::nn::Linear(64, 64), torch::nn::SiLU(),
                                torch::nn::Linear(64, 2)));
    }
    torch::NoGradGuard guard;
    for (const auto& p : named_parameters())
      if (p.value().dim() == 2)
        torch::nn::init::xavier_uniform_(p.value(), .01);
  }
  Tensor log_prob(const Tensor& input) {
    auto z = input.to(torch::kFloat32);
    auto log_det = torch::zeros({z.size(0)}, z.options());
    for (int i = 5; i >= 0; --i) {
      auto m = mask[i].to(z.options());
      auto fixed = z * m;
      auto s =
          scales->at<torch::nn::SequentialImpl>(i).forward(fixed) * (1 - m);
      auto t = translations->at<torch::nn::SequentialImpl>(i).forward(fixed) *
               (1 - m);
      z = fixed + (1 - m) * (z - t) * torch::exp(-s);
      log_det = log_det - s.sum(1);
    }
    return -.5 * z.square().sum(1) - std::log(2 * std::acos(-1.)) + log_det;
  }
};
struct Head final : torch::nn::Module {
  std::shared_ptr<DenseHead> many, one;
  std::shared_ptr<Prototypes> proto;
  std::shared_ptr<PoseFlow> flow;
  Head(const std::vector<I>& ch, const ModelOptions& o,
       const ModelDefinition& d, Scale scale) {
    many = register_module(
        "one_to_many", std::make_shared<DenseHead>(ch, d.strides, o, d.reg_max,
                                                   o.family == Family::v8));
    if (d.end_to_end) {
      one = register_module(
          "one_to_one",
          std::make_shared<DenseHead>(ch, d.strides, o, d.reg_max, false));
      torch::NoGradGuard guard;
      auto params = many->named_parameters();
      for (auto& p : one->named_parameters()) p.value().copy_(params[p.key()]);
      auto buffers = many->named_buffers();
      for (auto& b : one->named_buffers()) b.value().copy_(buffers[b.key()]);
    }
    if (d.reg_max > 1) {
      auto dfl = register_module(
          "dfl", torch::nn::Conv2d(
                     torch::nn::Conv2dOptions(d.reg_max, 1, 1).bias(false)));
      torch::NoGradGuard guard;
      dfl->weight.copy_(torch::arange(d.reg_max).view({1, d.reg_max, 1, 1}));
      dfl->weight.requires_grad_(false);
    }
    if (o.task == Task::instance_segment)
      proto =
          register_module("proto", std::make_shared<Prototypes>(ch, o, scale));
    if (o.task == Task::pose && o.family == Family::yolo26)
      flow = register_module("flow_model", std::make_shared<PoseFlow>());
  }
  std::vector<Tensor> run(const std::vector<Tensor>& features) {
    std::vector<Tensor> result{many->run(features)};
    if (one) {
      std::vector<Tensor> detached;
      for (const auto& feature : features) detached.push_back(feature.detach());
      result.push_back(one->run(detached));
    }
    if (proto) {
      auto extra = proto->run(features);
      result.insert(result.end(), extra.begin(), extra.end());
    }
    return result;
  }
};

std::vector<I> Reductions(const GraphSpec& graph) {
  std::vector<I> reductions;
  for (const auto& layer : graph.layers) {
    const auto source = layer.from[0];
    const I r = source == -1 ? (reductions.empty() ? 1 : reductions.back())
                             : reductions.at(source);
    if (layer.kind == "Conv" || layer.kind == "GhostConv")
      reductions.push_back(r * static_cast<I>(layer.args.at(2)));
    else if (layer.kind == "nn.Upsample")
      reductions.push_back(r / 2);
    else
      reductions.push_back(r);
  }
  std::vector<I> out;
  for (auto from : graph.layers.back().from)
    out.push_back(from == -1 ? reductions[reductions.size() - 2]
                             : reductions.at(from));
  return out;
}
}  // namespace

ModelDefinition describe_model(const ModelOptions& o) {
  const auto& graph = Resolve(o);
  ModelDefinition d;
  d.id = "yolo.ultralytics." + FamilyName(o.family) + "." +
         std::string(1, "nsmlx"[CapacityIndex(o.capacity)]);
  if (o.task != Task::detect) d.id += "." + TaskSuffix(o.task).substr(1);
  if (o.variant != Variant::p5)
    d.id += "." + VariantSuffix(o.variant).substr(1);
  d.reference_path = graph.path;
  d.reference_sha256 = graph.sha;
  d.revision = "native-yolo-graph/v1/" + graph.sha;
  d.reg_max = o.reg_max ? o.reg_max : o.family == Family::yolo26 ? 1 : 16;
  d.end_to_end = o.family == Family::yolo26 && o.task != Task::classify;
  if (o.task == Task::classify)
    d.output_names = {"logits"};
  else {
    d.strides = Reductions(graph);
    d.input_divisor = (std::max)(I{32}, d.strides.back());
    d.output_names = {"raw_one_to_many"};
    if (d.end_to_end) d.output_names.push_back("raw_one_to_one");
    if (o.task == Task::instance_segment) {
      d.output_names.push_back("mask_prototypes");
      if (o.family == Family::yolo26)
        d.output_names.push_back("semantic_logits");
    }
  }
  return d;
}

std::vector<ModelOptions> model_catalog() {
  std::vector<ModelOptions> out;
  for (auto family :
       {Family::v8, Family::yolo11, Family::yolo12, Family::yolo26})
    for (auto task : {Task::detect, Task::classify, Task::instance_segment,
                      Task::pose, Task::obb})
      for (auto variant :
           {Variant::p5, Variant::p2, Variant::p6, Variant::ghost_p5,
            Variant::ghost_p2, Variant::ghost_p6}) {
        ModelOptions o;
        o.family = family;
        o.task = task;
        o.variant = variant;
        try {
          Resolve(o);
        } catch (const InvalidArgument&) {
          continue;
        }
        for (std::size_t c = 0; c < 5; ++c) {
          o.capacity = static_cast<Capacity>(c);
          out.push_back(o);
        }
      }
  return out;
}

struct ModelImpl::Impl {
  ModelOptions options;
  ModelDefinition definition;
  const GraphSpec& graph;
  torch::nn::ModuleList layers{nullptr};
  std::vector<U> units;
  std::shared_ptr<Head> head;
  Impl(ModelImpl& owner, const ModelOptions& o)
      : options(o), definition(describe_model(o)), graph(Resolve(o)) {
    const auto scale = graph.scales[CapacityIndex(o.capacity)];
    layers = owner.register_module("model", torch::nn::ModuleList());
    std::vector<I> channels;
    for (std::size_t index = 0; index < graph.layers.size(); ++index) {
      const auto& l = graph.layers[index];
      const auto get = [&](int from) {
        return from == -1
                   ? (channels.empty() ? o.input_channels : channels.back())
                   : channels.at(from);
      };
      I in = get(l.from[0]), out = in;
      const auto a = [&](std::size_t i, double fallback) {
        return i < l.args.size() ? l.args[i] : fallback;
      };
      const auto n = Repeats(l.repeats, scale);
      U unit;
      if (index + 1 == graph.layers.size() && o.task != Task::classify) {
        std::vector<I> ch;
        for (auto f : l.from) ch.push_back(get(f));
        head = std::make_shared<Head>(ch, o, definition, scale);
        layers->push_back(head);
        units.push_back({});
        channels.push_back(0);
        continue;
      }
      if (l.kind == "Concat") {
        out = 0;
        for (auto f : l.from) out += get(f);
        unit = std::make_shared<Identity>();
      } else if (l.kind == "nn.Upsample")
        unit = std::make_shared<Identity>();
      else if (l.kind == "Classify") {
        out = o.num_classes;
        unit = std::make_shared<Classification>(in, o);
      } else {
        out = Channel(a(0, in), scale);
        if (l.kind == "Conv")
          unit = C(in, out, static_cast<I>(a(1, 1)), static_cast<I>(a(2, 1)));
        else if (l.kind == "GhostConv")
          unit = std::make_shared<GhostConv>(in, out, static_cast<I>(a(1, 1)),
                                             static_cast<I>(a(2, 1)));
        else if (l.kind == "C3Ghost")
          unit = std::make_shared<C3>(in, out, n, a(1, 1) != 0, .5, true);
        else if (l.kind == "C2f" || l.kind == "C2")
          unit = std::make_shared<Csp>(in, out, n, a(1, 0) != 0, l.kind);
        else if (l.kind == "C3k2")
          unit = std::make_shared<Csp>(
              in, out, n, true, l.kind, a(2, .5),
              a(1, 0) != 0 || CapacityIndex(o.capacity) >= 2, a(3, 0) != 0);
        else if (l.kind == "SPPF")
          unit = std::make_shared<Sppf>(in, out, static_cast<I>(a(1, 5)),
                                        static_cast<I>(a(2, 3)), a(3, 0) != 0);
        else if (l.kind == "C2PSA")
          unit = std::make_shared<C2Psa>(in, out, n);
        else if (l.kind == "A2C2f")
          unit = std::make_shared<AreaCsp>(in, out, n, a(1, 1) != 0,
                                           static_cast<I>(a(2, 1)),
                                           CapacityIndex(o.capacity) >= 3);
        else
          throw InvalidArgument("Unimplemented native graph operator: " +
                                l.kind);
      }
      channels.push_back(out);
      units.push_back(unit);
      layers->push_back(unit);
    }
  }
};
ModelImpl::ModelImpl(const ModelOptions& options)
    : impl_(std::make_unique<Impl>(*this, options)) {}
ModelImpl::~ModelImpl() = default;
const ModelOptions& ModelImpl::options() const noexcept {
  return impl_->options;
}
const ModelDefinition& ModelImpl::definition() const noexcept {
  return impl_->definition;
}

std::vector<Tensor> ModelImpl::forward(const Tensor& images) {
  const auto& d = definition();
  if (!images.defined() || images.layout() != torch::kStrided ||
      !images.is_floating_point() || images.dim() != 4 || images.size(0) <= 0 ||
      images.size(1) != options().input_channels || images.size(2) <= 0 ||
      images.size(3) <= 0 || images.size(2) % d.input_divisor ||
      images.size(3) % d.input_divisor)
    throw InvalidArgument(
        "YOLO expects floating NCHW with declared channels and stride-aligned "
        "H/W");
  std::vector<Tensor> cache;
  cache.reserve(impl_->graph.layers.size());
  for (std::size_t i = 0; i < impl_->graph.layers.size(); ++i) {
    const auto& l = impl_->graph.layers[i];
    const auto get = [&](int from) {
      return from == -1 ? (cache.empty() ? images : cache.back())
                        : cache.at(from);
    };
    if (i + 1 == impl_->graph.layers.size() && impl_->head) {
      std::vector<Tensor> features;
      for (auto f : l.from) features.push_back(get(f));
      return impl_->head->run(features);
    }
    auto x = get(l.from[0]);
    if (l.kind == "Concat") {
      std::vector<Tensor> parts;
      for (auto f : l.from) parts.push_back(get(f));
      cache.push_back(torch::cat(parts, 1));
    } else if (l.kind == "nn.Upsample")
      cache.push_back(Resize(x, x.size(2) * 2, x.size(3) * 2));
    else
      cache.push_back(impl_->units[i]->run(x));
  }
  return {cache.back()};
}
Tensor ModelImpl::forward_tensor(const Tensor& images) {
  if (definition().output_names.size() != 1)
    throw InvalidArgument("This YOLO definition requires its tuple adapter");
  return forward(images).front();
}
Tensor ModelImpl::pose_log_probability(const Tensor& residuals) {
  if (!impl_->head || !impl_->head->flow || !residuals.defined() ||
      residuals.dim() != 2 || residuals.size(1) != 2 ||
      !residuals.is_floating_point())
    throw InvalidArgument("YOLO26 pose flow requires floating [N,2] residuals");
  return impl_->head->flow->log_prob(residuals);
}
nn::ModelAdapter create_model(const ModelOptions& options) {
  auto model = std::make_shared<ModelImpl>(options);
  nn::InvocationSchema input;
  input.positional.push_back(
      ValueSchema::tensor(TensorSchema{.rank = 4}, "images"));
  std::vector<ValueSchema> outputs;
  for (const auto& name : model->definition().output_names)
    outputs.push_back(ValueSchema::tensor({}, name));
  return nn::adapt_module(
      model,
      [](ModelImpl& concrete, const nn::Invocation& call) {
        std::vector<c10::IValue> values;
        for (auto& output :
             concrete.forward(call.positional.front().toTensor()))
          values.emplace_back(output);
        return c10::IValue(c10::ivalue::Tuple::create(std::move(values)));
      },
      std::move(input), ValueSchema::tuple(std::move(outputs), "raw_outputs"));
}
nn::ModelAdapter create_tensor_model(const ModelOptions& options) {
  auto model = std::make_shared<ModelImpl>(options);
  if (model->definition().output_names.size() != 1)
    throw InvalidArgument("Multiple YOLO outputs require create_model()");
  nn::InvocationSchema input;
  input.positional.push_back(
      ValueSchema::tensor(TensorSchema{.rank = 4}, "images"));
  return nn::adapt_module(
      model,
      [](ModelImpl& concrete, const nn::Invocation& call) {
        return concrete.forward_tensor(call.positional.front().toTensor());
      },
      std::move(input),
      ValueSchema::tensor(
          TensorSchema{.rank = options.task == Task::classify ? 2 : 3}));
}
}  // namespace jyppx::liuding::models::yolo
