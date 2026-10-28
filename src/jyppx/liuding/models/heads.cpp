// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "head_utils.hpp"

#include <jyppx/liuding/models/heads.hpp>
#include <jyppx/liuding/models/registry.hpp>

#include <array>
#include <memory>
#include <type_traits>
#include <utility>

namespace jyppx::liuding::models {
namespace {
template <class First, class... Rest, class Function>
decltype(auto) Visit(const std::shared_ptr<torch::nn::Module>& module,
                     Function&& function) {
  if (auto concrete = std::dynamic_pointer_cast<First>(module))
    return function(std::move(concrete));
  if constexpr (sizeof...(Rest) > 0)
    return Visit<Rest...>(module, std::forward<Function>(function));
  else
    throw NotImplemented("Module has no built-in classifier contract",
                         detail::Details("classifier.module",
                                         "a concrete built-in vision module"));
}

template <class Function>
decltype(auto) VisitVision(const nn::ModelAdapter& model, Function&& function) {
  return Visit<ResNetImpl, DenseNetImpl, MobileNetV2Impl, MobileNetV3Impl,
               EfficientNetImpl, EfficientNetV2Impl, ConvNeXtImpl, ViTImpl,
               SwinTinyImpl, DinoV2Impl, DinoV3Impl, DinoV3ConvNeXtImpl>(
      model.state().shared_module(), std::forward<Function>(function));
}

template <class Module>
ClassifierInfo Info(const Module& module) {
  ClassifierInfo result;
  result.num_classes = module.options().num_classes;
  result.parameter_prefix = "classifier";
  if constexpr (std::is_same_v<Module, ResNetImpl>) {
    result.embedding_features = module.embedding_dimension();
    result.parameter_prefix = "fc";
  } else if constexpr (std::is_same_v<Module, MobileNetV3Impl>) {
    result.kind = ClassifierKind::mobile_mlp;
    result.embedding_features = module.embedding_dimension();
  } else if constexpr (std::is_same_v<Module, DenseNetImpl>) {
    result.embedding_features = 1024;
  } else if constexpr (std::is_same_v<Module, EfficientNetImpl>) {
    result.embedding_features =
        module.variant() == EfficientNetVariant::b0 ? 1280 : 1408;
  } else if constexpr (std::is_same_v<Module, MobileNetV2Impl> ||
                       std::is_same_v<Module, EfficientNetV2Impl>) {
    result.embedding_features = 1280;
  } else if constexpr (std::is_same_v<Module, ConvNeXtImpl> ||
                       std::is_same_v<Module, SwinTinyImpl>) {
    result.embedding_features = 768;
    result.parameter_prefix = "head.fc";
  } else if constexpr (std::is_same_v<Module, DinoV3ConvNeXtImpl>) {
    constexpr std::array<std::int64_t, 4> widths{768, 768, 1024, 1536};
    result.embedding_features =
        widths.at(static_cast<std::size_t>(module.variant()) - 1);
    result.parameter_prefix = "head";
  } else {
    result.embedding = EmbeddingKind::cls_token;
    result.embedding_features =
        module.named_parameters(false)["cls_token"].size(2);
    result.parameter_prefix = "head";
  }
  return result;
}
}  // namespace

ClassifierInfo classifier_info(const nn::ModelAdapter& model) {
  return VisitVision(model, [](const auto& module) { return Info(*module); });
}

void reset_classifier(nn::ModelAdapter& model, std::int64_t num_classes) {
  detail::ValidateClasses(num_classes);
  VisitVision(model, [&](const auto& module) {
    const auto& input = model.input_schema();
    const auto& output = model.output_schema();
    if (input.positional.size() != 1 || !input.keyword.empty() ||
        input.allow_extra_keywords ||
        input.positional[0].kind != ValueKind::tensor ||
        !input.positional[0].tensor_constraints ||
        input.positional[0].tensor_constraints->rank != 4 ||
        output.kind != ValueKind::tensor || !output.tensor_constraints ||
        output.tensor_constraints->rank != 2)
      detail::Invalid(
          "classifier.adapter",
          "one NCHW input and one NC output; feature tuples cannot be reset");
    const auto info = Info(*module);
    TensorSchema tensor;
    tensor.rank = 2;
    tensor.shape = {{std::nullopt,
                     num_classes > 0 ? num_classes : info.embedding_features}};
    auto replacement = nn::adapt_module(
        module,
        [module](const nn::Invocation& invocation) -> c10::IValue {
          return module->forward(invocation.positional.front().toTensor());
        },
        input,
        ValueSchema::tensor(
            std::move(tensor), "output",
            num_classes > 0 ? "classification_logits" : "image_embedding"));
    module->reset_classifier(num_classes);
    static_assert(std::is_nothrow_move_assignable_v<nn::ModelAdapter>);
    model = std::move(replacement);
  });
}
}  // namespace jyppx::liuding::models
