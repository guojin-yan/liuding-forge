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
    hint += "resnet18, resnet50.";
  else if (name.starts_with("mobile"))
    hint += "mobilenet_v3_small, mobilenet_v3_large.";
  else
    hint += "transformer_encoder, resnet18, mobilenet_v3_small.";
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
  registry.add({.name = "resnet18",
                .family = "resnet",
                .task = "image_classification",
                .reference = "torchvision.models.resnet18",
                .default_parameters = 11689512U,
                .preview = true},
               [](const ModelOptions& o) { return adapt(resnet18(o.resnet)); });
  registry.add({.name = "resnet50",
                .family = "resnet",
                .task = "image_classification",
                .reference = "torchvision.models.resnet50 (v1.5 stride)",
                .default_parameters = 25557032U,
                .preview = true},
               [](const ModelOptions& o) { return adapt(resnet50(o.resnet)); });
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
