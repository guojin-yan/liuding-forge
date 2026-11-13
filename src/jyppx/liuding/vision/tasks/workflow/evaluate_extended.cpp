// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/extended.hpp>
#include <jyppx/liuding/vision/tasks/data/extended.hpp>
#include <jyppx/liuding/vision/tasks/advanced_metrics.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
namespace jyppx::liuding::vision::tasks {
std::string evaluate_extended_task(TaskBinding& binding,const data::image::ImageDataset& dataset,const RecipeSpec& recipe,std::string dataset_id,std::stop_token stop){
  using I=std::int64_t;namespace im=data::image;torch::NoGradGuard guard;
  auto options=yolo_evaluation_options(recipe.data);options.allow_unannotated_negatives=true;options.num_classes=recipe.model.num_classes;options.augmentation.dataset_id=dataset_id;
  options.letterbox.geometry.mask_padding_id=recipe.extension.semantic_ignore;
  options.classification.enabled=recipe.model.task==TaskKind::classify_multilabel||recipe.model.task==TaskKind::reid;
  std::vector<Tensor> scores,labels,known,embeddings;std::vector<std::string> cameras;
  Tensor confusion;double loss_sum=0,error=0,squared=0,relative=0,delta=0;std::uint64_t pixels=0;
  if(recipe.model.task==TaskKind::semantic_segment){if(recipe.model.num_classes>4096)throw ResourceExhausted("Semantic confusion matrix bound exceeded");confusion=torch::zeros({recipe.model.num_classes,recipe.model.num_classes},torch::kInt64);}
  for(std::size_t i=0;i<dataset.size();++i){if(stop.stop_requested())throw Cancelled("Supplementary task evaluation cancelled");auto sample=dataset.get(i,stop);
    auto b=prepare_yolo_batch(std::array{sample},std::array<im::ImageBatchKey,1>{{{.id=sample.image_path}}},options,stop);b.images=b.images.to(recipe.device);b.labels=b.labels.to(recipe.device);
    data::BatchEnvelope envelope;envelope.value=im::image_batch_value(b);auto invocation=binding.image_mapper(envelope);auto target=binding.target_mapper(envelope);
    training::StepContext context;context.phase=training::StepPhase::validation;
    auto out=binding.step({*binding.model,envelope,invocation,target,context});loss_sum+=out.loss.item<double>();auto raw=raw_tensors(out.prediction->value);
    if(recipe.model.task==TaskKind::classify_multilabel){scores.push_back(raw[0].sigmoid().to(torch::kCPU));auto t=target->toTupleRef().elements();labels.push_back(t[0].toTensor().to(torch::kCPU));known.push_back(t[1].toTensor().to(torch::kCPU));}
    else if(recipe.model.task==TaskKind::semantic_segment){auto y=target->toTensor().to(torch::kCPU).flatten(),p=raw[0].argmax(1).to(torch::kCPU).flatten();auto valid=y!=recipe.extension.semantic_ignore;
      if(valid.any().item<bool>()){auto a=y.index({valid});if(a.min().item<I>()<0||a.max().item<I>()>=recipe.model.num_classes)throw InvalidArgument("Semantic labels outside vocabulary");confusion+=torch::bincount(a*recipe.model.num_classes+p.index({valid}),{},recipe.model.num_classes*recipe.model.num_classes).view_as(confusion);}}
    else if(recipe.model.task==TaskKind::depth){auto t=target->toTupleRef().elements();auto y=t[0].toTensor().to(torch::kCPU,torch::kFloat64),valid=t[1].toTensor().to(torch::kCPU),p=raw[0].to(torch::kCPU,torch::kFloat64);
      if(recipe.extension.relative_inverse_depth){auto pv=p.index({valid}),yv=y.index({valid});auto centered=pv-pv.mean();auto a=(centered*(yv-yv.mean())).sum()/centered.square().sum().clamp_min(1e-12);p=a*p+(yv.mean()-a*pv.mean());}
      auto v=y.index({valid}),pred=p.index({valid}),res=pred-v;pixels+=v.numel();error+=res.abs().sum().item<double>();squared+=res.square().sum().item<double>();
      if(!recipe.extension.relative_inverse_depth){relative+=(res.abs()/v.clamp_min(1e-6)).sum().item<double>();delta+=(torch::maximum(v/pred.clamp_min(1e-6),pred/v.clamp_min(1e-6))<1.25).sum().item<double>();}}
    else if(recipe.model.task==TaskKind::reid){embeddings.push_back(raw[1].to(torch::kCPU));labels.push_back(torch::tensor({sample.label_categories->at(sample.labels.at(0)).source_id},torch::kInt64));cameras.push_back(sample.task_payload?sample.task_payload->camera_id:std::string{});}
  }
  nlohmann::json report={{"schema","vision.task_metrics/v1"},{"dataset_id",dataset_id},{"task",task_name(recipe.model.task)},{"samples",dataset.size()},{"loss",loss_sum/dataset.size()}};
  if(recipe.model.task==TaskKind::classify_multilabel)for(const auto& [k,v]:multilabel_metrics(torch::cat(scores),torch::cat(labels),torch::cat(known),recipe.extension.multilabel_thresholds.empty()?Tensor{}:torch::tensor(recipe.extension.multilabel_thresholds)))report[k]=v;
  if(recipe.model.task==TaskKind::semantic_segment){auto m=confusion.to(torch::kFloat64),tp=m.diag(),un=m.sum(0)+m.sum(1)-tp;auto present=un>0;report["miou"]=present.any().item<bool>()?(tp/un.clamp_min(1)).index({present}).mean().item<double>():0.;report["pixel_accuracy"]=m.sum().item<double>()?tp.sum().item<double>()/m.sum().item<double>():0.;}
  if(recipe.model.task==TaskKind::depth){report["valid_pixels"]=pixels;report["mae"]=pixels?error/pixels:0.;report["rmse"]=pixels?std::sqrt(squared/pixels):0.;report["alignment"]=recipe.extension.relative_inverse_depth?"per-image-scale-shift":"none";if(!recipe.extension.relative_inverse_depth){report["absrel"]=pixels?relative/pixels:0.;report["delta1"]=pixels?delta/pixels:0.;}}
  if(recipe.model.task==TaskKind::reid)for(const auto& [k,v]:reid_retrieval_metrics(torch::cat(embeddings),torch::cat(labels),cameras))report[k]=v;
  return report.dump(2);
}
std::string calibrate_vision_multilabel(TaskBinding& binding,const data::image::ImageDataset& dataset,const RecipeSpec& r,std::string identity,std::stop_token stop){
  if(r.model.task!=TaskKind::classify_multilabel||identity.empty()||dataset.empty()||dataset.size()>67108864/static_cast<std::size_t>(r.model.num_classes))throw InvalidArgument("Calibration requires a bounded identified multilabel validation split");
  struct Modes{std::vector<std::pair<std::shared_ptr<torch::nn::Module>,bool>> state;~Modes(){for(auto& [m,mode]:state)m->train(mode);}} modes;for(auto& m:binding.model->state().shared_module()->modules())modes.state.emplace_back(m,m->is_training());binding.model->state().eval();torch::NoGradGuard guard;
  auto options=yolo_evaluation_options(r.data);options.classification.enabled=true;options.allow_unannotated_negatives=true;options.num_classes=r.model.num_classes;options.augmentation.dataset_id=identity;std::vector<Tensor> scores,labels,known;
  for(std::size_t i=0;i<dataset.size();++i){if(stop.stop_requested())throw Cancelled("Multilabel calibration cancelled");auto sample=dataset.get(i,stop);std::array<data::image::ImageBatchKey,1> keys{{{.id=sample.image_path}}};auto batch=prepare_yolo_batch(std::span(&sample,1),keys,options,stop);batch.images=batch.images.to(r.device);data::BatchEnvelope envelope;envelope.value=data::image::image_batch_value(batch);auto owned_target=binding.target_mapper(envelope);auto target=owned_target->toTupleRef().elements();auto raw=raw_tensors(binding.model->invoke(binding.image_mapper(envelope)).value);scores.push_back(raw[0].sigmoid().cpu());labels.push_back(target[0].toTensor().cpu());known.push_back(target[1].toTensor().cpu());}
  auto s=torch::cat(scores),y=torch::cat(labels),k=torch::cat(known);auto thresholds=calibrate_multilabel_thresholds(s,y,k,identity).contiguous();std::vector<double> values(thresholds.const_data_ptr<double>(),thresholds.const_data_ptr<double>()+thresholds.numel());
  return nlohmann::json{{"schema","vision.multilabel_calibration/v1"},{"validation_identity",identity},{"recipe_identity",recipe_identity(r)},{"extension_patch",{{"multilabel_thresholds",values},{"threshold_validation_identity",identity}}},{"metrics",multilabel_metrics(s,y,k,thresholds)}}.dump(2);
}

}
