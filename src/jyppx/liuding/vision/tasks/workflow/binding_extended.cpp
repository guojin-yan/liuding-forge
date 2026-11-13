// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "binding_backends.hpp"
#include <jyppx/liuding/vision/models/extended/detectors.hpp>
#include <jyppx/liuding/vision/models/extended/dense.hpp>
#include <jyppx/liuding/vision/models/extended/region.hpp>
#include <jyppx/liuding/vision/tasks/dense_detection.hpp>
#include <jyppx/liuding/vision/tasks/rtdetr.hpp>
#include <jyppx/liuding/vision/tasks/open_vocabulary.hpp>
#include <jyppx/liuding/vision/models/extended/open_vocabulary.hpp>
#include <jyppx/liuding/vision/models/extended/text.hpp>
#include <jyppx/liuding/vision/models/extended/rtdetr.hpp>
#include <jyppx/liuding/vision/tasks/extended.hpp>
#include <jyppx/liuding/vision/tasks/training/advanced.hpp>
#include <jyppx/liuding/vision/tasks/data/extended.hpp>
namespace jyppx::liuding::vision::tasks::detail {
namespace ex=models::extended;
void catalog_extended(std::vector<CapabilityRecord>& entries){
  auto add=[&](std::string name,TaskKind task,std::string revision,std::string contract="objects/v1"){
    CapabilityRecord c;c.architecture=std::move(name);c.revision=std::move(revision);c.task=task;c.variant="p5";c.component="models_vision";c.output_contract=std::move(contract);entries.push_back(std::move(c));};
  for(auto name:{"fcos_resnet50_fpn","retinanet_resnet50_fpn"})add(name,TaskKind::detect,"torchvision-v0.24.0/native-unqualified-v1");
  for(auto name:{"fcos_swin_tiny_fpn","fcos_swin_tiny_bifpn","fcos_resnet50_bifpn","fcos_resnet50_bifpn_softmax","fcos_convnext_tiny_bifpn","fcos_vit_tiny_token_pyramid","fcos_vit_adapter_tiny_fpn"})add(name,TaskKind::detect,"native-feature-composition/v1");
  for(auto name:{"rtdetr_resnet18_vd","rtdetr_resnet50_vd"}){add(name,TaskKind::detect,"rtdetr-v1/native-unqualified-v1","rtdetr_queries/v1");entries.back().end_to_end=true;entries.back().minimum_input_size=256;}
  for(auto name:{"yoloworld_v8s_text","yoloe_v8s_text","yoloe_v8s_visual","yoloe_v8s_prompt_free"}){add(name,TaskKind::detect,"native-open-vocabulary-p5/v1","prompt_yolo/v1");if(std::string_view(name).starts_with("yoloe_"))add(std::string(name)+"_segment",TaskKind::instance_segment,"native-open-vocabulary-segment/v1","prompt_yolo_masks/v1");}
  add("ssd300_vgg16",TaskKind::detect,"torchvision-v0.24.0/native-ssd300-v1");entries.back().input_divisor=1;entries.back().fixed_input_size={300,300};
  for(auto name:{"fast_rcnn_resnet50_fpn","fast_rcnn_vgg16"}){add(name,TaskKind::detect,"native-external-proposals/v1");entries.back().requires_external_proposals=true;}
  add("vgg16",TaskKind::classify,"torchvision-v0.24.0/native-vgg16-v1");add("vgg16",TaskKind::classify_multilabel,"native-vgg16-multilabel/v1","independent_logits/v1");
  for(auto name:{"fcn_resnet50","deeplabv3_resnet50"})add(name,TaskKind::semantic_segment,"native-dilated-resnet50-os8/v1","semantic_logits/v1");
  add("depth_resnet50_fpn",TaskKind::depth,"native-supervised-depth/v1","numeric_depth/v1");
  for(auto name:{"simmim_swin_tiny","simmim_vit_tiny","simmim_vit_base"})add(name,TaskKind::pretrain_simmim,"native-masked-patch-pixel-l1/v1","reconstruction/v1");
  add("reid_resnet50",TaskKind::reid,"native-r50-projection-bnneck/v1","reid_embedding/v1");
}
void bind_extended(const RecipeSpec& r,TaskBinding& b,const YoloDataOptions& data){
  if(b.capability.component!="models_vision")return;const auto& name=r.model.architecture;
  if(name.starts_with("yoloworld_")||name.starts_with("yoloe_")) {
    ex::OpenVocabularyOptions o;o.num_classes=r.model.num_classes;o.segmentation=r.model.task==TaskKind::instance_segment;o.mask_channels=r.model.mask_channels;
    if(name.starts_with("yoloe_"))o.mode=name.find("visual")!=name.npos?ex::PromptMode::yoloe_visual:name.find("prompt_free")!=name.npos?ex::PromptMode::yoloe_prompt_free:ex::PromptMode::yoloe_text;
    b.model=std::make_shared<nn::ModelAdapter>(ex::adapt(ex::OpenVocabulary(o)));
    b.yolo.task=r.model.task;b.yolo.mask_channels=r.model.mask_channels;b.yolo.detection.num_classes=r.model.num_classes;
    PromptInputOptions input;input.classes=r.model.num_classes;input.visual=o.mode==ex::PromptMode::yoloe_visual;input.prompt_free=o.mode==ex::PromptMode::yoloe_prompt_free;
    if(!r.extension.tokenizer_merges.empty()) {
      auto tokenizer=std::make_shared<ex::ClipTokenizer>(r.extension.tokenizer_merges,r.extension.tokenizer_sha256);
      auto encoder=std::make_shared<nn::ModelAdapter>(ex::adapt(ex::ClipText()));InitializationOptions weights;weights.weights=r.extension.text_encoder_weights;weights.sha256=r.extension.text_encoder_sha256;
      initialize_vision(*encoder,weights,"clip_text_vit_b32");encoder->state().eval();for(auto& p:encoder->state().parameters())p.set_requires_grad(false);
      input.text_encoder_identity=tokenizer->identity()+":clip-text-b32:"+r.extension.text_encoder_sha256;
      input.encode_text=[tokenizer,encoder](const std::vector<std::string>& prompts){torch::NoGradGuard guard;auto ids=tokenizer->encode(prompts);return encoder->invoke(nn::Invocation{{ids},{}}).value.toTensor();};
    }
    b.image_mapper=make_prompt_image_mapper(std::move(input));b.target_mapper=make_task_target_mapper(b.yolo,data);b.step=make_prompt_step(b.yolo,o.mode==ex::PromptMode::yoloe_prompt_free);
    b.decode=[task=b.yolo](const c10::IValue& v,auto size,const ResultOptions& options){auto raw=raw_tensors(v);raw.resize(task.task==TaskKind::instance_segment?2:1);return decode_yolo_results(raw,size,task,options);};
  }else if(name.starts_with("rtdetr_")){
    ex::RtDetrOptions o;o.num_classes=r.model.num_classes;if(name=="rtdetr_resnet18_vd")o.backbone=ex::RtDetrBackbone::resnet18_vd;
    b.model=std::make_shared<nn::ModelAdapter>(ex::adapt(ex::RtDetr(o)));b.step=make_rtdetr_step(r.model.num_classes);b.image_mapper=make_rtdetr_image_mapper(data);
    b.decode=[](const c10::IValue& v,auto size,const ResultOptions& options){return decode_rtdetr(raw_tensors(v),size,options);};
  }else if(name.starts_with("fcos_")||name=="retinanet_resnet50_fpn"){
    ex::DenseDetectorOptions o;o.num_classes=r.model.num_classes;
    if(name=="retinanet_resnet50_fpn"){o.kind=ex::DenseDetectorKind::retinanet_v1;o.pyramid.extra_from_p5=false;}
    if(name=="fcos_vit_adapter_tiny_fpn")o.pyramid.backbone="vit_adapter_tiny";
    if(name=="fcos_swin_tiny_fpn"||name=="fcos_swin_tiny_bifpn")o.pyramid.backbone="swin_tiny_patch4_window7_224";
    if(name=="fcos_resnet50_bifpn"||name=="fcos_resnet50_bifpn_softmax"||name=="fcos_swin_tiny_bifpn")o.pyramid.kind=ex::PyramidKind::bifpn;
    if(name=="fcos_resnet50_bifpn_softmax")o.pyramid.softmax_fusion=true;
    if(name=="fcos_convnext_tiny_bifpn"){o.pyramid.backbone="convnext_tiny";o.pyramid.kind=ex::PyramidKind::bifpn;}
    if(name=="fcos_vit_tiny_token_pyramid"){o.pyramid.backbone="vit_tiny_patch16_224";o.pyramid.kind=ex::PyramidKind::token_pyramid;}
    b.model=std::make_shared<nn::ModelAdapter>(ex::adapt(ex::DenseDetector(o)));
    auto kind=o.kind==ex::DenseDetectorKind::fcos?DenseDetectionCriterion::fcos:DenseDetectionCriterion::retinanet_v1;
    b.step=make_dense_detection_step({.kind=kind,.num_classes=r.model.num_classes});
    b.decode=[kind](const c10::IValue& value,auto size,const ResultOptions& options){return decode_dense_detection(raw_tensors(value),size,kind,options);};
  }else if(name=="ssd300_vgg16"){
    b.model=std::make_shared<nn::ModelAdapter>(ex::create_ssd300_vgg16(r.model.num_classes));b.step=make_dense_detection_step({.kind=DenseDetectionCriterion::ssd,.num_classes=r.model.num_classes});
    b.decode=[](const c10::IValue& v,auto size,const ResultOptions& o){return decode_dense_detection(raw_tensors(v),size,DenseDetectionCriterion::ssd,o);};
  }else if(name.starts_with("fast_rcnn_")){
    ex::FastRcnnOptions o;o.num_classes=r.model.num_classes;if(name=="fast_rcnn_vgg16")o.profile=ex::FastRcnnProfile::vgg16_roi_pool;
    b.model=std::make_shared<nn::ModelAdapter>(ex::create_fast_rcnn(o));b.step=make_fast_rcnn_step(r.model.num_classes);
    b.image_mapper=make_extended_image_mapper(r.model.task,data,true);b.decode=[](const c10::IValue& v,auto,const ResultOptions& o){return decode_faster_rcnn_results(raw_tensors(v),o);};
  }else if(name=="vgg16")b.model=std::make_shared<nn::ModelAdapter>(ex::create_vgg16(r.model.num_classes));
  else if(r.model.task==TaskKind::semantic_segment||r.model.task==TaskKind::depth){
    ex::DenseFieldOptions o;o.num_classes=r.model.num_classes;o.kind=r.model.task==TaskKind::depth?ex::DenseFieldKind::depth_fpn:name=="fcn_resnet50"?ex::DenseFieldKind::fcn:ex::DenseFieldKind::deeplabv3;
    b.model=std::make_shared<nn::ModelAdapter>(ex::create_dense_field(o));
  }else if(r.model.task==TaskKind::pretrain_simmim)b.model=std::make_shared<nn::ModelAdapter>(ex::create_simmim(name=="simmim_swin_tiny"?ex::SimMimEncoder::swin_tiny:name=="simmim_vit_tiny"?ex::SimMimEncoder::vit_tiny:ex::SimMimEncoder::vit_base,r.extension.activation_recomputation));
  else if(r.model.task==TaskKind::reid)b.model=std::make_shared<nn::ModelAdapter>(ex::create_reid_resnet50(r.model.num_classes));
}
}
