// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "detail.hpp"

#include <jyppx/liuding/models/registry.hpp>

#include <limits>
#include <utility>
namespace jyppx::liuding::models {
namespace {
bool Name(std::string_view name) {
  if (name.empty() || name.size() > 128U) return false;
  for (const auto ch : name)
    if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_'))
      return false;
  return true;
}
[[noreturn]] void Unknown(std::string_view name) {
  std::string hint = "Use list_models(); closest families: ";
  if (name.starts_with("res"))
    hint += "resnet18, resnet34, resnet50, resnet101, resnext50_32x4d.";
  else if (name.starts_with("dense"))
    hint += "densenet121.";
  else if (name.starts_with("efficient"))
    hint += "efficientnet_b0, efficientnet_b2, tf_efficientnetv2_s.";
  else if (name.starts_with("mobile"))
    hint += "mobilenetv2_100, mobilenet_v3_small, mobilenet_v3_large.";
  else if (name.starts_with("dinov3"))
    hint += "dinov3_vits16, dinov3_vitb16, dinov3_convnext_tiny.";
  else
    hint += "transformer_encoder, resnet18, dinov3_vits16.";
  throw InvalidArgument("Unknown model architecture",
                        {.component = "models.registry",
                         .operation = "lookup",
                         .expected = "a registered architecture",
                         .actual = std::string(name.substr(0, 128)),
                         .hint = std::move(hint)});
}
}  // namespace
void ModelRegistry::add(ModelInfo info, ModelFactory factory) {
  if (!Name(info.name) || info.family.empty() || info.task.empty() ||
      info.reference.empty() || !factory)
    detail::Invalid("registry.add", "valid metadata and a nonempty factory");
  if (entries_.contains(info.name))
    detail::Invalid("registry.add", "a unique explicit model name");
  auto key = info.name;
  entries_.emplace(std::move(key), Entry{std::move(info), std::move(factory)});
}
ModelInfo ModelRegistry::info(std::string_view name) const {
  const auto found = entries_.find(name);
  if (found == entries_.end()) Unknown(name);
  return found->second.info;
}
std::vector<ModelInfo> ModelRegistry::list(std::string_view filter) const {
  std::vector<ModelInfo> result;
  for (const auto& [name, entry] : entries_)
    if (name.find(filter) != std::string::npos) result.push_back(entry.info);
  return result;
}
nn::ModelAdapter ModelRegistry::create(std::string_view name,
                                       const ModelOptions& options) const {
  const auto found = entries_.find(name);
  if (found == entries_.end()) Unknown(name);
  return found->second.factory(options);
}
ModelRegistry default_model_registry() {
  ModelRegistry registry;
  registry.add({.name = "densenet121",
                .family = "densenet",
                .task = "image_classification",
                .reference = "torchvision.models.densenet121",
                .default_parameters = 7978856U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(densenet121(o.densenet));
               });
  registry.add({.name = "efficientnet_b0",
                .family = "efficientnet",
                .task = "image_classification",
                .reference = "timm efficientnet_b0",
                .default_parameters = 5288548U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(efficientnet_b0(o.efficientnet));
               });
  registry.add({.name = "efficientnet_b2",
                .family = "efficientnet",
                .task = "image_classification",
                .reference = "timm efficientnet_b2",
                .default_parameters = 9109994U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(efficientnet_b2(o.efficientnet));
               });
  registry.add({.name = "tf_efficientnetv2_s",
                .family = "efficientnetv2",
                .task = "image_classification",
                .reference = "timm tf_efficientnetv2_s",
                .default_parameters = 21458488U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(tf_efficientnetv2_s(o.efficientnet_v2));
               });
  registry.add({.name = "mobilenetv2_100",
                .family = "mobilenetv2",
                .task = "image_classification",
                .reference = "timm mobilenetv2_100",
                .default_parameters = 3504872U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(mobilenetv2_100(o.mobile_v2));
               });
  registry.add({.name = "resnet18",
                .family = "resnet",
                .task = "image_classification",
                .reference = "torchvision.models.resnet18",
                .default_parameters = 11689512U,
                .preview = true},
               [](const ModelOptions& o) { return adapt(resnet18(o.resnet)); });
  registry.add({.name = "resnet34",
                .family = "resnet",
                .task = "image_classification",
                .reference = "torchvision.models.resnet34",
                .default_parameters = 21797672U,
                .preview = true},
               [](const ModelOptions& o) { return adapt(resnet34(o.resnet)); });
  registry.add({.name = "resnet50",
                .family = "resnet",
                .task = "image_classification",
                .reference = "torchvision.models.resnet50 (v1.5 stride)",
                .default_parameters = 25557032U,
                .preview = true},
               [](const ModelOptions& o) { return adapt(resnet50(o.resnet)); });
  registry.add(
      {.name = "resnet101",
       .family = "resnet",
       .task = "image_classification",
       .reference = "torchvision.models.resnet101 (v1.5 stride)",
       .default_parameters = 44549160U,
       .preview = true},
      [](const ModelOptions& o) { return adapt(resnet101(o.resnet)); });
  registry.add(
      {.name = "resnext50_32x4d",
       .family = "resnext",
       .task = "image_classification",
       .reference = "torchvision.models.resnext50_32x4d (v1.5 stride)",
       .default_parameters = 25028904U,
       .preview = true},
      [](const ModelOptions& o) { return adapt(resnext50_32x4d(o.resnet)); });
  registry.add({.name = "mobilenet_v3_small",
                .family = "mobilenet",
                .task = "image_classification",
                .reference = "torchvision.models.mobilenet_v3_small",
                .default_parameters = 2542856U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(mobilenet_v3_small(o.mobile));
               });
  registry.add({.name = "mobilenet_v3_large",
                .family = "mobilenet",
                .task = "image_classification",
                .reference = "torchvision.models.mobilenet_v3_large",
                .default_parameters = 5483032U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(mobilenet_v3_large(o.mobile));
               });
  registry.add({.name = "transformer_encoder",
                .family = "transformer",
                .task = "feature_encoding",
                .reference = "torch.nn.TransformerEncoder post-norm",
                .default_parameters = 18915328U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(transformer_encoder(o.transformer));
               });
  registry.add(
      {.name = "vit_small_patch14_dinov2",
       .family = "dinov2",
       .task = "image_classification_or_embedding",
       .reference = "timm vit_small_patch14_dinov2",
       .default_parameters = 22056192U,
       .preview = true},
      [](const ModelOptions& o) { return adapt(dinov2_vits14(o.dinov2)); });
  registry.add(
      {.name = "vit_base_patch14_dinov2",
       .family = "dinov2",
       .task = "image_classification_or_embedding",
       .reference = "timm vit_base_patch14_dinov2",
       .default_parameters = 86579712U,
       .preview = true},
      [](const ModelOptions& o) { return adapt(dinov2_vitb14(o.dinov2)); });
  registry.add(
      {.name = "vit_large_patch14_dinov2",
       .family = "dinov2",
       .task = "image_classification_or_embedding",
       .reference = "timm vit_large_patch14_dinov2",
       .default_parameters = 304367616U,
       .preview = true},
      [](const ModelOptions& o) { return adapt(dinov2_vitl14(o.dinov2)); });
  registry.add({.name = "vit_giant_patch14_dinov2",
                .family = "dinov2",
                .task = "image_classification_or_embedding",
                .reference = "timm vit_giant_patch14_dinov2",
                .default_parameters = 1136479232ULL,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(dinov2_vit_giant14(o.dinov2));
               });
  registry.add({.name = "dinov3_vits16",
                .family = "dinov3_vit",
                .task = "image_classification_or_embedding",
                .reference =
                    "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
                .default_parameters = 21601152U,
                .preview = true},
               [](const ModelOptions& o) { return adapt(dinov3_vits16(o.dinov3)); });
  registry.add({.name = "dinov3_vits16plus",
                .family = "dinov3_vit",
                .task = "image_classification_or_embedding",
                .reference =
                    "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
                .default_parameters = 28697472U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(dinov3_vits16plus(o.dinov3));
               });
  registry.add({.name = "dinov3_vitb16",
                .family = "dinov3_vit",
                .task = "image_classification_or_embedding",
                .reference =
                    "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
                .default_parameters = 85669632U,
                .preview = true},
               [](const ModelOptions& o) { return adapt(dinov3_vitb16(o.dinov3)); });
  registry.add({.name = "dinov3_vitl16",
                .family = "dinov3_vit",
                .task = "image_classification_or_embedding",
                .reference =
                    "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
                .default_parameters = 303154176ULL,
                .preview = true},
               [](const ModelOptions& o) { return adapt(dinov3_vitl16(o.dinov3)); });
  registry.add({.name = "dinov3_vitl16plus",
                .family = "dinov3_vit",
                .task = "image_classification_or_embedding",
                .reference =
                    "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
                .default_parameters = 403915776ULL,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(dinov3_vitl16plus(o.dinov3));
               });
  registry.add({.name = "dinov3_vith16plus",
                .family = "dinov3_vit",
                .task = "image_classification_or_embedding",
                .reference =
                    "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
                .default_parameters = 840000000ULL,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(dinov3_vith16plus(o.dinov3));
               });
  registry.add({.name = "dinov3_vit7b16",
                .family = "dinov3_vit",
                .task = "image_classification_or_embedding",
                .reference =
                    "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
                .default_parameters = 6700000000ULL,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(dinov3_vit7b16(o.dinov3));
               });
  registry.add({.name = "dinov3_convnext_tiny",
                .family = "dinov3_convnext",
                .task = "image_classification_or_embedding",
                .reference =
                    "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
                .default_parameters = 27820128U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(dinov3_convnext_tiny(o.dinov3_convnext));
               });
  registry.add({.name = "dinov3_convnext_small",
                .family = "dinov3_convnext",
                .task = "image_classification_or_embedding",
                .reference =
                    "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
                .default_parameters = 49454688U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(dinov3_convnext_small(o.dinov3_convnext));
               });
  registry.add({.name = "dinov3_convnext_base",
                .family = "dinov3_convnext",
                .task = "image_classification_or_embedding",
                .reference =
                    "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
                .default_parameters = 87566464U,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(dinov3_convnext_base(o.dinov3_convnext));
               });
  registry.add({.name = "dinov3_convnext_large",
                .family = "dinov3_convnext",
                .task = "image_classification_or_embedding",
                .reference =
                    "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
                .default_parameters = 196230336ULL,
                .preview = true},
               [](const ModelOptions& o) {
                 return adapt(dinov3_convnext_large(o.dinov3_convnext));
               });
  return registry;
}
std::vector<ModelInfo> list_models(std::string_view filter) {
  return default_model_registry().list(filter);
}
ModelInfo model_info(std::string_view name) {
  return default_model_registry().info(name);
}
nn::ModelAdapter create_model(std::string_view name,
                              const ModelOptions& options) {
  return default_model_registry().create(name, options);
}
std::uint64_t parameter_count(const torch::nn::Module& module) {
  return detail::Backend("parameter_count", [&] {
    std::uint64_t count = 0;
    for (const auto& parameter : module.parameters()) {
      const auto size = parameter.numel();
      if (size < 0 || static_cast<std::uint64_t>(size) >
                          std::numeric_limits<std::uint64_t>::max() - count)
        throw ResourceExhausted(
            "Model parameter count overflow",
            detail::Details("parameter_count", "uint64 size"));
      count += static_cast<std::uint64_t>(size);
    }
    return count;
  });
}
}  // namespace jyppx::liuding::models
