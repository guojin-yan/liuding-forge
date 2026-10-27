// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "detail.hpp"
#include "feature_adapters.hpp"

#include <jyppx/liuding/models/registry.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
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
  else if (name.starts_with("convnext"))
    hint += "convnext_tiny, convnext_small.";
  else if (name.starts_with("efficient"))
    hint += "efficientnet_b0, efficientnet_b2, tf_efficientnetv2_s.";
  else if (name.starts_with("mobile"))
    hint += "mobilenetv2_100, mobilenet_v3_small, mobilenet_v3_large.";
  else if (name.starts_with("dinov3"))
    hint += "dinov3_vits16, dinov3_vitb16, dinov3_convnext_tiny.";
  else if (name.starts_with("vit"))
    hint +=
        "vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224.";
  else if (name.starts_with("swin"))
    hint += "swin_tiny_patch4_window7_224.";
  else
    hint += "transformer_encoder, resnet18, dinov3_vits16.";
  throw InvalidArgument("Unknown model architecture",
                        {.component = "models.registry",
                         .operation = "lookup",
                         .expected = "a registered architecture",
                         .actual = std::string(name.substr(0, 128)),
                         .hint = std::move(hint)});
}

void ValidateFeatures(const ModelInfo& info, const ModelFactory& factory) {
  if (info.features.empty() != !factory ||
      info.features.empty() != !info.feature_input ||
      info.features.size() > 64U)
    detail::Invalid("registry.features",
                    "matching bounded feature metadata and factory");
  if (info.feature_input) {
    const auto& input = *info.feature_input;
    if (input.spatial_multiple <= 0 || input.minimum_spatial <= 0)
      detail::Invalid("registry.features",
                      "positive spatial alignment and minimum");
    const auto& fixed = input.fixed_size;
    if (fixed != std::array<std::int64_t, 2>{0, 0} &&
        (input.crop_to_multiple || fixed[0] < input.minimum_spatial ||
         fixed[1] < input.minimum_spatial ||
         fixed[0] % input.spatial_multiple ||
         fixed[1] % input.spatial_multiple))
      detail::Invalid("registry.features", "valid fixed dimensions or {0,0}");
  }
  std::vector<std::string> names;
  for (const auto& feature : info.features) {
    if (!Name(feature.name) || feature.channels <= 0 ||
        feature.prefix_tokens < 0 ||
        std::find(names.begin(), names.end(), feature.name) != names.end())
      detail::Invalid("registry.features",
                      "unique named features with positive channels");
    names.push_back(feature.name);
    switch (feature.layout) {
      case FeatureLayout::nc:
        if (feature.reduction != 0 || feature.prefix_tokens != 0)
          detail::Invalid("registry.features",
                          "global features without spatial stride or tokens");
        break;
      case FeatureLayout::ntc:
        if (feature.reduction <= 0)
          detail::Invalid("registry.features",
                          "token features with positive patch stride");
        break;
      case FeatureLayout::nchw:
        if (feature.reduction <= 0 || feature.prefix_tokens != 0)
          detail::Invalid(
              "registry.features",
              "spatial features with positive stride and no prefix tokens");
        break;
      default:
        detail::Invalid("registry.features", "a declared feature layout");
    }
  }
}

ValueSchema FeatureSchema(const FeatureInfo& feature) {
  TensorSchema tensor;
  switch (feature.layout) {
    case FeatureLayout::nc:
      tensor.rank = 2;
      tensor.shape = {{std::nullopt, feature.channels}};
      break;
    case FeatureLayout::ntc:
      tensor.rank = 3;
      tensor.shape = {{std::nullopt, std::nullopt, feature.channels}};
      break;
    case FeatureLayout::nchw:
      tensor.rank = 4;
      tensor.shape = {
          {std::nullopt, feature.channels, std::nullopt, std::nullopt}};
      break;
  }
  return ValueSchema::tensor(std::move(tensor), feature.name, "image_feature");
}
}  // namespace
void ModelRegistry::add(ModelInfo info, ModelFactory factory,
                        ModelFactory feature_factory,
                        FeatureSpecFactory feature_spec_factory) {
  if (!Name(info.name) || info.family.empty() || info.task.empty() ||
      info.reference.empty() || !factory)
    detail::Invalid("registry.add", "valid metadata and a nonempty factory");
  if (entries_.contains(info.name))
    detail::Invalid("registry.add", "a unique explicit model name");
  ValidateFeatures(info, feature_factory);
  if (feature_spec_factory && !feature_factory)
    detail::Invalid("registry.features",
                    "a feature factory with the options resolver");
  auto key = info.name;
  entries_.emplace(std::move(key), Entry{std::move(info), std::move(factory),
                                         std::move(feature_factory),
                                         std::move(feature_spec_factory)});
}
ModelInfo ModelRegistry::info(std::string_view name) const {
  const auto found = entries_.find(name);
  if (found == entries_.end()) Unknown(name);
  return found->second.info;
}
FeatureSpec ModelRegistry::feature_spec(std::string_view name,
                                        const ModelOptions& options) const {
  const auto found = entries_.find(name);
  if (found == entries_.end()) Unknown(name);
  const auto& entry = found->second;
  if (!entry.feature_factory)
    throw NotImplemented("Model has no registered feature extractor",
                         detail::Details("registry.feature_spec",
                                         "an entry with feature metadata"));
  if (!entry.feature_spec_factory)
    return {.outputs = entry.info.features, .input = *entry.info.feature_input};
  auto spec = entry.feature_spec_factory(options);
  auto resolved = entry.info;
  resolved.features = spec.outputs;
  resolved.feature_input = spec.input;
  ValidateFeatures(resolved, entry.feature_factory);
  if (spec.outputs.size() != entry.info.features.size())
    detail::Invalid("registry.feature_spec",
                    "stable feature count across options");
  for (std::size_t index = 0; index < spec.outputs.size(); ++index) {
    const auto& output = spec.outputs[index];
    const auto& original = entry.info.features[index];
    if (output.name != original.name || output.layout != original.layout ||
        output.reduction != original.reduction ||
        output.prefix_tokens != original.prefix_tokens)
      detail::Invalid("registry.feature_spec",
                      "stable feature identity across options");
  }
  return spec;
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

nn::ModelAdapter ModelRegistry::create_features(
    std::string_view name, const ModelOptions& options,
    const std::vector<std::size_t>& out_indices) const {
  const auto found = entries_.find(name);
  if (found == entries_.end()) Unknown(name);
  const auto& entry = found->second;
  if (!entry.feature_factory)
    throw NotImplemented("Model has no registered feature extractor",
                         detail::Details("registry.create_features",
                                         "an entry with feature metadata"));
  if (out_indices.size() > entry.info.features.size())
    detail::Invalid("registry.feature_indices", "unique valid feature indices");
  auto selected = out_indices;
  if (selected.empty()) {
    selected.resize(entry.info.features.size());
    std::iota(selected.begin(), selected.end(), 0U);
  }
  std::vector<bool> seen(entry.info.features.size(), false);
  const auto spec = feature_spec(name, options);
  std::vector<ValueSchema> output;
  for (const auto index : selected) {
    if (index >= seen.size() || seen[index])
      detail::Invalid("registry.feature_indices",
                      "unique in-range feature indices");
    seen[index] = true;
    output.push_back(FeatureSchema(spec.outputs[index]));
  }
  auto full = entry.feature_factory(options);
  const auto& schema = full.output_schema();
  if (schema.kind != ValueKind::tuple ||
      schema.elements.size() != entry.info.features.size() ||
      !std::all_of(schema.elements.begin(), schema.elements.end(),
                   [](const ValueSchema& value) {
                     return value.kind == ValueKind::tensor;
                   }))
    detail::Invalid("registry.feature_factory",
                    "an ordered tuple of all declared feature tensors");
  if (full.input_schema().positional.empty() ||
      full.input_schema().positional.front().kind != ValueKind::tensor ||
      !full.input_schema().positional.front().tensor_constraints ||
      full.input_schema().positional.front().tensor_constraints->rank != 4)
    detail::Invalid("registry.feature_factory",
                    "an NCHW tensor as the first positional input");
  auto input = full.input_schema();
  auto module = full.state().shared_module();
  const auto spatial = spec.input;
  return nn::adapt_module(
      std::move(module),
      [full = std::move(full), selected = std::move(selected),
       spatial](const nn::Invocation& invocation) {
        const auto images = invocation.positional.front().toTensor();
        const auto multiple = spatial.spatial_multiple;
        const auto minimum = std::max(multiple, spatial.minimum_spatial);
        if (images.size(2) < minimum || images.size(3) < minimum ||
            (spatial.fixed_size[0] &&
             (images.size(2) != spatial.fixed_size[0] ||
              images.size(3) != spatial.fixed_size[1])) ||
            (!spatial.crop_to_multiple &&
             (images.size(2) % multiple || images.size(3) % multiple)))
          detail::Invalid("registry.feature_input",
                          "the resolved minimum, fixed size and alignment for "
                          "this architecture");
        auto value = [&] {
          if (spatial.crop_to_multiple &&
              (images.size(2) % multiple || images.size(3) % multiple)) {
            auto cropped = invocation;
            cropped.positional.front() =
                images.narrow(2, 0, (images.size(2) / multiple) * multiple)
                    .narrow(3, 0, (images.size(3) / multiple) * multiple);
            return full.invoke(cropped).value;
          }
          return full.invoke(invocation).value;
        }();
        const auto features = value.toTuple();
        std::vector<c10::IValue> result;
        result.reserve(selected.size());
        for (const auto index : selected)
          result.push_back(features->elements()[index]);
        return c10::ivalue::Tuple::create(std::move(result));
      },
      std::move(input),
      ValueSchema::tuple(std::move(output), "features", "image_features"));
}
ModelRegistry default_model_registry() {
  ModelRegistry registry;
  registry.add(
      {.name = "convnext_tiny",
       .family = "convnext",
       .task = "image_classification",
       .reference = "timm convnext_tiny",
       .default_parameters = 28589128U,
       .preview = true,
       .features = detail::StageFeatureInfo({96, 192, 384, 768}),
       .feature_input = FeatureInputSpec{.minimum_spatial = 32}},
      [](const ModelOptions& o) { return adapt(convnext_tiny(o.convnext)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(convnext_tiny(o.convnext));
      });
  registry.add(
      {.name = "convnext_small",
       .family = "convnext",
       .task = "image_classification",
       .reference = "timm convnext_small",
       .default_parameters = 50223688U,
       .preview = true,
       .features = detail::StageFeatureInfo({96, 192, 384, 768}),
       .feature_input = FeatureInputSpec{.minimum_spatial = 32}},
      [](const ModelOptions& o) { return adapt(convnext_small(o.convnext)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(convnext_small(o.convnext));
      });
  registry.add(
      {.name = "densenet121",
       .family = "densenet",
       .task = "image_classification",
       .reference = "torchvision.models.densenet121",
       .default_parameters = 7978856U,
       .preview = true,
       .features = detail::StageFeatureInfo({256, 512, 1024, 1024}),
       .feature_input = FeatureInputSpec{.minimum_spatial = 29}},
      [](const ModelOptions& o) { return adapt(densenet121(o.densenet)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(densenet121(o.densenet));
      });
  registry.add(
      {.name = "efficientnet_b0",
       .family = "efficientnet",
       .task = "image_classification",
       .reference = "timm efficientnet_b0",
       .default_parameters = 5288548U,
       .preview = true,
       .features = detail::StageFeatureInfo({16, 24, 40, 112, 320}, 2),
       .feature_input = FeatureInputSpec{}},
      [](const ModelOptions& o) {
        return adapt(efficientnet_b0(o.efficientnet));
      },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(efficientnet_b0(o.efficientnet));
      });
  registry.add(
      {.name = "efficientnet_b2",
       .family = "efficientnet",
       .task = "image_classification",
       .reference = "timm efficientnet_b2",
       .default_parameters = 9109994U,
       .preview = true,
       .features = detail::StageFeatureInfo({16, 24, 48, 120, 352}, 2),
       .feature_input = FeatureInputSpec{}},
      [](const ModelOptions& o) {
        return adapt(efficientnet_b2(o.efficientnet));
      },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(efficientnet_b2(o.efficientnet));
      });
  registry.add(
      {.name = "tf_efficientnetv2_s",
       .family = "efficientnetv2",
       .task = "image_classification",
       .reference = "timm tf_efficientnetv2_s",
       .default_parameters = 21458488U,
       .preview = true,
       .features = detail::StageFeatureInfo({24, 48, 64, 160, 256}, 2),
       .feature_input = FeatureInputSpec{}},
      [](const ModelOptions& o) {
        return adapt(tf_efficientnetv2_s(o.efficientnet_v2));
      },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(tf_efficientnetv2_s(o.efficientnet_v2));
      });
  registry.add(
      {.name = "mobilenetv2_100",
       .family = "mobilenetv2",
       .task = "image_classification",
       .reference = "timm mobilenetv2_100",
       .default_parameters = 3504872U,
       .preview = true,
       .features = detail::StageFeatureInfo({16, 24, 32, 96, 320}, 2),
       .feature_input = FeatureInputSpec{}},
      [](const ModelOptions& o) { return adapt(mobilenetv2_100(o.mobile_v2)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(mobilenetv2_100(o.mobile_v2));
      });
  registry.add(
      {.name = "resnet18",
       .family = "resnet",
       .task = "image_classification",
       .reference = "torchvision.models.resnet18",
       .default_parameters = 11689512U,
       .preview = true,
       .features = detail::StageFeatureInfo({64, 128, 256, 512}),
       .feature_input = FeatureInputSpec{}},
      [](const ModelOptions& o) { return adapt(resnet18(o.resnet)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(resnet18(o.resnet));
      });
  registry.add(
      {.name = "resnet34",
       .family = "resnet",
       .task = "image_classification",
       .reference = "torchvision.models.resnet34",
       .default_parameters = 21797672U,
       .preview = true,
       .features = detail::StageFeatureInfo({64, 128, 256, 512}),
       .feature_input = FeatureInputSpec{}},
      [](const ModelOptions& o) { return adapt(resnet34(o.resnet)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(resnet34(o.resnet));
      });
  registry.add(
      {.name = "resnet50",
       .family = "resnet",
       .task = "image_classification",
       .reference = "torchvision.models.resnet50 (v1.5 stride)",
       .default_parameters = 25557032U,
       .preview = true,
       .features = detail::StageFeatureInfo({256, 512, 1024, 2048}),
       .feature_input = FeatureInputSpec{}},
      [](const ModelOptions& o) { return adapt(resnet50(o.resnet)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(resnet50(o.resnet));
      });
  registry.add(
      {.name = "resnet101",
       .family = "resnet",
       .task = "image_classification",
       .reference = "torchvision.models.resnet101 (v1.5 stride)",
       .default_parameters = 44549160U,
       .preview = true,
       .features = detail::StageFeatureInfo({256, 512, 1024, 2048}),
       .feature_input = FeatureInputSpec{}},
      [](const ModelOptions& o) { return adapt(resnet101(o.resnet)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(resnet101(o.resnet));
      });
  registry.add(
      {.name = "resnext50_32x4d",
       .family = "resnext",
       .task = "image_classification",
       .reference = "torchvision.models.resnext50_32x4d (v1.5 stride)",
       .default_parameters = 25028904U,
       .preview = true,
       .features = detail::StageFeatureInfo({256, 512, 1024, 2048}),
       .feature_input = FeatureInputSpec{}},
      [](const ModelOptions& o) { return adapt(resnext50_32x4d(o.resnet)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(resnext50_32x4d(o.resnet));
      });
  registry.add(
      {.name = "mobilenet_v3_small",
       .family = "mobilenet",
       .task = "image_classification",
       .reference = "torchvision.models.mobilenet_v3_small",
       .default_parameters = 2542856U,
       .preview = true,
       .features = detail::MobileFeatureSpec(1.0, false).outputs,
       .feature_input = detail::MobileFeatureSpec(1.0, false).input},
      [](const ModelOptions& o) { return adapt(mobilenet_v3_small(o.mobile)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(mobilenet_v3_small(o.mobile));
      },
      [](const ModelOptions& o) {
        return detail::MobileFeatureSpec(o.mobile.width_multiplier, false);
      });
  registry.add(
      {.name = "mobilenet_v3_large",
       .family = "mobilenet",
       .task = "image_classification",
       .reference = "torchvision.models.mobilenet_v3_large",
       .default_parameters = 5483032U,
       .preview = true,
       .features = detail::MobileFeatureSpec(1.0, true).outputs,
       .feature_input = detail::MobileFeatureSpec(1.0, true).input},
      [](const ModelOptions& o) { return adapt(mobilenet_v3_large(o.mobile)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(mobilenet_v3_large(o.mobile));
      },
      [](const ModelOptions& o) {
        return detail::MobileFeatureSpec(o.mobile.width_multiplier, true);
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
      {.name = "vit_tiny_patch16_224",
       .family = "vit",
       .task = "image_classification",
       .reference = "timm vit_tiny_patch16_224",
       .default_parameters = 5717416U,
       .preview = true,
       .features = detail::ViTFeatureSpec(192, false).outputs,
       .feature_input = detail::ViTFeatureSpec(192, false).input},
      [](const ModelOptions& o) { return adapt(vit_tiny_patch16_224(o.vit)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(vit_tiny_patch16_224(o.vit));
      },
      [](const ModelOptions& o) {
        return detail::ViTFeatureSpec(192, o.vit.dynamic_image_size);
      });
  registry.add(
      {.name = "vit_small_patch16_224",
       .family = "vit",
       .task = "image_classification",
       .reference = "timm vit_small_patch16_224",
       .default_parameters = 22050664U,
       .preview = true,
       .features = detail::ViTFeatureSpec(384, false).outputs,
       .feature_input = detail::ViTFeatureSpec(384, false).input},
      [](const ModelOptions& o) { return adapt(vit_small_patch16_224(o.vit)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(vit_small_patch16_224(o.vit));
      },
      [](const ModelOptions& o) {
        return detail::ViTFeatureSpec(384, o.vit.dynamic_image_size);
      });
  registry.add(
      {.name = "vit_base_patch16_224",
       .family = "vit",
       .task = "image_classification",
       .reference = "timm vit_base_patch16_224",
       .default_parameters = 86567656U,
       .preview = true,
       .features = detail::ViTFeatureSpec(768, false).outputs,
       .feature_input = detail::ViTFeatureSpec(768, false).input},
      [](const ModelOptions& o) { return adapt(vit_base_patch16_224(o.vit)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(vit_base_patch16_224(o.vit));
      },
      [](const ModelOptions& o) {
        return detail::ViTFeatureSpec(768, o.vit.dynamic_image_size);
      });
  registry.add(
      {.name = "swin_tiny_patch4_window7_224",
       .family = "swin",
       .task = "image_classification",
       .reference = "timm swin_tiny_patch4_window7_224",
       .default_parameters = 28288354U,
       .preview = true,
       .features = detail::SwinFeatureSpec(false).outputs,
       .feature_input = detail::SwinFeatureSpec(false).input},
      [](const ModelOptions& o) {
        return adapt(swin_tiny_patch4_window7_224(o.swin));
      },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(swin_tiny_patch4_window7_224(o.swin));
      },
      [](const ModelOptions& o) {
        return detail::SwinFeatureSpec(o.swin.dynamic_image_size);
      });
  registry.add(
      {.name = "vit_small_patch14_dinov2",
       .family = "dinov2",
       .task = "image_classification_or_embedding",
       .reference = "timm vit_small_patch14_dinov2",
       .default_parameters = 22056192U,
       .preview = true,
       .features = detail::TokenFeatureInfo(384, 14, 1),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 14, .crop_to_multiple = false}},
      [](const ModelOptions& o) { return adapt(dinov2_vits14(o.dinov2)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov2_vits14(o.dinov2));
      });
  registry.add(
      {.name = "vit_base_patch14_dinov2",
       .family = "dinov2",
       .task = "image_classification_or_embedding",
       .reference = "timm vit_base_patch14_dinov2",
       .default_parameters = 86579712U,
       .preview = true,
       .features = detail::TokenFeatureInfo(768, 14, 1),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 14, .crop_to_multiple = false}},
      [](const ModelOptions& o) { return adapt(dinov2_vitb14(o.dinov2)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov2_vitb14(o.dinov2));
      });
  registry.add(
      {.name = "vit_large_patch14_dinov2",
       .family = "dinov2",
       .task = "image_classification_or_embedding",
       .reference = "timm vit_large_patch14_dinov2",
       .default_parameters = 304367616U,
       .preview = true,
       .features = detail::TokenFeatureInfo(1024, 14, 1),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 14, .crop_to_multiple = false}},
      [](const ModelOptions& o) { return adapt(dinov2_vitl14(o.dinov2)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov2_vitl14(o.dinov2));
      });
  registry.add(
      {.name = "vit_giant_patch14_dinov2",
       .family = "dinov2",
       .task = "image_classification_or_embedding",
       .reference = "timm vit_giant_patch14_dinov2",
       .default_parameters = 1136479232ULL,
       .preview = true,
       .features = detail::TokenFeatureInfo(1536, 14, 1),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 14, .crop_to_multiple = false}},
      [](const ModelOptions& o) { return adapt(dinov2_vit_giant14(o.dinov2)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov2_vit_giant14(o.dinov2));
      });
  registry.add(
      {.name = "dinov3_vits16",
       .family = "dinov3_vit",
       .task = "image_classification_or_embedding",
       .reference =
           "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
       .default_parameters = 21601152U,
       .preview = true,
       .features = detail::TokenFeatureInfo(384, 16, 5),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 16, .crop_to_multiple = true}},
      [](const ModelOptions& o) { return adapt(dinov3_vits16(o.dinov3)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov3_vits16(o.dinov3));
      });
  registry.add(
      {.name = "dinov3_vits16plus",
       .family = "dinov3_vit",
       .task = "image_classification_or_embedding",
       .reference =
           "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
       .default_parameters = 28697472U,
       .preview = true,
       .features = detail::TokenFeatureInfo(384, 16, 5),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 16, .crop_to_multiple = true}},
      [](const ModelOptions& o) { return adapt(dinov3_vits16plus(o.dinov3)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov3_vits16plus(o.dinov3));
      });
  registry.add(
      {.name = "dinov3_vitb16",
       .family = "dinov3_vit",
       .task = "image_classification_or_embedding",
       .reference =
           "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
       .default_parameters = 85669632U,
       .preview = true,
       .features = detail::TokenFeatureInfo(768, 16, 5),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 16, .crop_to_multiple = true}},
      [](const ModelOptions& o) { return adapt(dinov3_vitb16(o.dinov3)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov3_vitb16(o.dinov3));
      });
  registry.add(
      {.name = "dinov3_vitl16",
       .family = "dinov3_vit",
       .task = "image_classification_or_embedding",
       .reference =
           "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
       .default_parameters = 303154176ULL,
       .preview = true,
       .features = detail::TokenFeatureInfo(1024, 16, 5),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 16, .crop_to_multiple = true}},
      [](const ModelOptions& o) { return adapt(dinov3_vitl16(o.dinov3)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov3_vitl16(o.dinov3));
      });
  registry.add(
      {.name = "dinov3_vitl16plus",
       .family = "dinov3_vit",
       .task = "image_classification_or_embedding",
       .reference =
           "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
       .default_parameters = 403915776ULL,
       .preview = true,
       .features = detail::TokenFeatureInfo(1024, 16, 5),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 16, .crop_to_multiple = true}},
      [](const ModelOptions& o) { return adapt(dinov3_vitl16plus(o.dinov3)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov3_vitl16plus(o.dinov3));
      });
  registry.add(
      {.name = "dinov3_vith16plus",
       .family = "dinov3_vit",
       .task = "image_classification_or_embedding",
       .reference =
           "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
       .default_parameters = 840000000ULL,
       .preview = true,
       .features = detail::TokenFeatureInfo(1280, 16, 5),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 16, .crop_to_multiple = true}},
      [](const ModelOptions& o) { return adapt(dinov3_vith16plus(o.dinov3)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov3_vith16plus(o.dinov3));
      });
  registry.add(
      {.name = "dinov3_vit7b16",
       .family = "dinov3_vit",
       .task = "image_classification_or_embedding",
       .reference =
           "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
       .default_parameters = 6700000000ULL,
       .preview = true,
       .features = detail::TokenFeatureInfo(4096, 16, 5),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 16, .crop_to_multiple = true}},
      [](const ModelOptions& o) { return adapt(dinov3_vit7b16(o.dinov3)); },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov3_vit7b16(o.dinov3));
      });
  registry.add(
      {.name = "dinov3_convnext_tiny",
       .family = "dinov3_convnext",
       .task = "image_classification_or_embedding",
       .reference =
           "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
       .default_parameters = 27820128U,
       .preview = true,
       .features = detail::StageFeatureInfo({96, 192, 384, 768}),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 32, .crop_to_multiple = true}},
      [](const ModelOptions& o) {
        return adapt(dinov3_convnext_tiny(o.dinov3_convnext));
      },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov3_convnext_tiny(o.dinov3_convnext));
      });
  registry.add(
      {.name = "dinov3_convnext_small",
       .family = "dinov3_convnext",
       .task = "image_classification_or_embedding",
       .reference =
           "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
       .default_parameters = 49454688U,
       .preview = true,
       .features = detail::StageFeatureInfo({96, 192, 384, 768}),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 32, .crop_to_multiple = true}},
      [](const ModelOptions& o) {
        return adapt(dinov3_convnext_small(o.dinov3_convnext));
      },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov3_convnext_small(o.dinov3_convnext));
      });
  registry.add(
      {.name = "dinov3_convnext_base",
       .family = "dinov3_convnext",
       .task = "image_classification_or_embedding",
       .reference =
           "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
       .default_parameters = 87566464U,
       .preview = true,
       .features = detail::StageFeatureInfo({128, 256, 512, 1024}),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 32, .crop_to_multiple = true}},
      [](const ModelOptions& o) {
        return adapt(dinov3_convnext_base(o.dinov3_convnext));
      },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov3_convnext_base(o.dinov3_convnext));
      });
  registry.add(
      {.name = "dinov3_convnext_large",
       .family = "dinov3_convnext",
       .task = "image_classification_or_embedding",
       .reference =
           "facebookresearch/dinov3@6876159a11b4df116f30f667f8c9888617df0751",
       .default_parameters = 196230336ULL,
       .preview = true,
       .features = detail::StageFeatureInfo({192, 384, 768, 1536}),
       .feature_input =
           FeatureInputSpec{.spatial_multiple = 32, .crop_to_multiple = true}},
      [](const ModelOptions& o) {
        return adapt(dinov3_convnext_large(o.dinov3_convnext));
      },
      [](const ModelOptions& o) {
        return detail::AdaptFeatures(dinov3_convnext_large(o.dinov3_convnext));
      });
  return registry;
}
std::vector<ModelInfo> list_models(std::string_view filter) {
  return default_model_registry().list(filter);
}
ModelInfo model_info(std::string_view name) {
  return default_model_registry().info(name);
}
FeatureSpec model_feature_spec(std::string_view name,
                               const ModelOptions& options) {
  return default_model_registry().feature_spec(name, options);
}
nn::ModelAdapter create_model(std::string_view name,
                              const ModelOptions& options) {
  return default_model_registry().create(name, options);
}
nn::ModelAdapter create_feature_extractor(
    std::string_view name, const ModelOptions& options,
    const std::vector<std::size_t>& out_indices) {
  return default_model_registry().create_features(name, options, out_indices);
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
