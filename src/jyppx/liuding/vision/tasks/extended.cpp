// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/extended.hpp>
#include <jyppx/liuding/vision/tasks/training/advanced.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
#include <jyppx/liuding/vision/ops/roi.hpp>
#include <torch/torch.h>
namespace jyppx::liuding::vision::tasks {
using I=std::int64_t;
std::vector<ImageResult> decode_extended_results(std::span<const Tensor> raw,std::array<I,2> size,TaskKind task,const ResultOptions& o,std::span<const double> thresholds){
  if(raw.empty()||size[0]<1||size[1]<1||!torch::isfinite(raw[0]).all().item<bool>())throw InvalidArgument("Invalid supplementary prediction");
  torch::NoGradGuard guard;std::vector<ImageResult> result;
  for(I b=0;b<raw[0].size(0);++b){ImageResult r;r.task=task;r.extent={static_cast<std::uint32_t>(size[1]),static_cast<std::uint32_t>(size[0])};
    r.boxes=torch::empty({0,4});r.classes=r.anchor_indices=torch::empty({0},torch::kInt64);r.scores=torch::empty({0});r.decode_revision="native-task-field/v1";
    if(task==TaskKind::classify_multilabel){auto scores=raw[0][b].detach().to(torch::kCPU,torch::kFloat32).sigmoid();auto ids=torch::argsort(scores,true,0,true);
      if(!thresholds.empty()&&thresholds.size()!=scores.numel())throw CompatibilityError("Multilabel threshold vocabulary mismatch");auto boundary=thresholds.empty()?torch::full_like(scores,o.confidence):torch::tensor(std::vector<double>(thresholds.begin(),thresholds.end())).to(scores.options());auto keep=scores.index_select(0,ids)>=boundary.index_select(0,ids);if(!o.classes.empty()){auto selected=torch::zeros_like(scores,torch::kBool);for(auto c:o.classes)selected[c]=true;keep=keep&selected.index_select(0,ids);}
      ids=ids.index({keep});ids=ids.slice(0,0,(std::min)(ids.numel(),o.max_detections));r.classes=ids;r.scores=scores.index_select(0,ids);}
    else if(task==TaskKind::reid){if(raw.size()!=2)throw InvalidArgument("ReID output needs logits and embedding");auto v=raw[1][b].detach().to(torch::kCPU,torch::kFloat32);r.embedding=v/v.square().sum().sqrt().clamp_min(1e-12);}
    else if(task==TaskKind::semantic_segment){r.field=raw[0][b].argmax(0).to(torch::kCPU);r.field_valid=torch::ones({size[0],size[1]},torch::kBool);}
    else if(task==TaskKind::depth||task==TaskKind::pretrain_simmim){r.field=raw[0][b].detach().to(torch::kCPU,torch::kFloat32);r.field_valid=torch::ones({size[0],size[1]},torch::kBool);}
    else throw InvalidArgument("Unknown supplementary prediction task");result.push_back(std::move(r));}
  return result;
}
training::StepFunction make_fast_rcnn_step(I classes){return[classes](const training::StepInput& in){auto prediction=in.model.invoke(in.invocation);auto raw=raw_tensors(prediction.value);
  if(raw.size()!=9||raw[4].size(1)!=classes+1||raw[6].dim()!=1)throw InvalidArgument("Invalid Fast R-CNN raw schema");
  auto loss=raw[4].sum()*0+raw[5].sum()*0;auto labels=raw[6];
  if(labels.numel()){loss=loss+torch::nn::functional::cross_entropy(raw[4].to(torch::kFloat32),labels);auto pos=(labels>0).nonzero().flatten();if(pos.numel()){
    auto target=vision_ops::encode_box_deltas(raw[3].index_select(0,pos).slice(1,1,5),raw[7].index_select(0,pos),{10,10,5,5});
    auto predicted=raw[5].view({-1,classes+1,4}).index({pos,labels.index_select(0,pos)}).to(torch::kFloat32);auto a=(predicted-target).abs();
    loss=loss+torch::where(a<1./9,4.5*a.square(),a-1./18).sum()/labels.numel();}}
  training::StepOutput out;out.loss=loss;out.prediction=std::move(prediction);out.sample_count=in.invocation.positional[0].toTensor().size(0);return out;};}
training::StepFunction make_reid_step(double margin,double weight){if(!std::isfinite(weight)||weight<0)throw InvalidArgument("Invalid triplet weight");return[margin,weight](const training::StepInput& in){
  if(!in.target||!in.target->isTensor())throw InvalidArgument("ReID requires identity labels");auto prediction=in.model.invoke(in.invocation);auto raw=raw_tensors(prediction.value);
  if(raw.size()!=2)throw InvalidArgument("ReID requires two outputs");auto labels=in.target->toTensor();auto loss=in.context.phase==training::StepPhase::validation?raw[0].sum()*0:classification_loss(raw[0],labels,.1);
  if(weight&&in.context.phase==training::StepPhase::training)loss=loss+weight*batch_hard_triplet_loss(raw[1],labels,margin);
  training::StepOutput out;out.loss=loss;out.prediction=std::move(prediction);out.sample_count=labels.numel();return out;};}
}
