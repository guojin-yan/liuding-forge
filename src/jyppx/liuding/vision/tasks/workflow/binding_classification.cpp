// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "binding_backends.hpp"

#include <jyppx/liuding/models/registry.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
namespace jyppx::liuding::vision::tasks {
namespace detail {
void catalog_classification(std::vector<CapabilityRecord>& entries) {
  for (const auto& info : models::list_models())
    if (info.task == "image_classification") {
      entries.push_back({.architecture = info.name,
                         .revision = "models-registry/native-v1",
                         .task = TaskKind::classify,
                         .variant = "p5",
                         .component = "models"});
      if (info.feature_input) {
        entries.back().fixed_input_size = info.feature_input->fixed_size;
        entries.back().minimum_input_size = info.feature_input->minimum_spatial;
      }
      auto multi=entries.back();multi.task=TaskKind::classify_multilabel;
      multi.revision="models-registry/native-multilabel-v1";multi.output_contract="independent_logits/v1";entries.push_back(std::move(multi));
    }
}
void bind_classification(const RecipeSpec& r, TaskBinding& b,
                         const YoloDataOptions& data) {
  if (b.capability.component == "models") {
    models::ModelOptions o;
    o.resnet.num_classes = o.convnext.num_classes = o.densenet.num_classes =
        o.efficientnet.num_classes = o.efficientnet_v2.num_classes =
            o.mobile.num_classes = o.mobile_v2.num_classes =
                o.swin.num_classes = o.vit.num_classes = o.dinov2.num_classes =
                    o.dinov3.num_classes = o.dinov3_convnext.num_classes =
                        r.model.num_classes;
    o.swin.activation_recomputation=o.vit.activation_recomputation=r.extension.activation_recomputation;
    b.model = std::make_shared<nn::ModelAdapter>(
        models::create_model(r.model.architecture, o));
    if(r.extension.feature_outputs) {
      auto residual=std::dynamic_pointer_cast<models::ResNetImpl>(b.model->state().shared_module());
      if(!residual)throw InvalidArgument("Feature distillation currently declares ResNet global features only");
      auto head=std::dynamic_pointer_cast<torch::nn::LinearImpl>(residual->named_modules()["fc"]);
      std::shared_ptr<torch::nn::LinearImpl> projection;
      if(r.extension.feature_projection)projection=residual->register_module("distillation_projection",std::make_shared<torch::nn::LinearImpl>(torch::nn::LinearOptions(residual->embedding_dimension(),r.extension.feature_projection).bias(false)));
      b.model=std::make_shared<nn::ModelAdapter>(nn::adapt_module(residual,[head,projection](models::ResNetImpl& m,const nn::Invocation& x){auto features=m.forward_features(x.positional[0].toTensor()).back().mean({2,3});return c10::ivalue::Tuple::create({head->forward(features),projection?projection->forward(features):features});},
        {.positional={ValueSchema::tensor({})},.keyword={},.allow_extra_keywords=false},ValueSchema::tuple({ValueSchema::tensor({}),ValueSchema::tensor({})})));
    }
  }
}
}  // namespace detail
}  // namespace jyppx::liuding::vision::tasks
