// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/training/advanced.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
#include <torch/nn/functional/loss.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
namespace jyppx::liuding::vision::tasks {
namespace {
using I = std::int64_t;
namespace F = torch::nn::functional;
void Require(bool b, const char* message) { if (!b) throw InvalidArgument(message); }
void Matching(const Tensor& a, const Tensor& b) {
  Require(a.defined() && b.defined() && a.is_floating_point() && b.is_floating_point() &&
      a.sizes() == b.sizes() && a.device() == b.device(), "Floating target/prediction shape/device mismatch");
}
void Finite(const Tensor& t) { Require(torch::isfinite(t).all().item<bool>(), "Nonfinite supervised tensor"); }
void Validity(const Tensor& x, const Tensor& known) {
  Require(known.defined() && known.scalar_type() == torch::kBool && known.sizes() == x.sizes() &&
      known.device() == x.device(), "Invalid supervision mask");
  Require(known.any().item<bool>(), "Batch contains no supervised values");
}
const auto& Targets(const training::StepInput& in) {
  Require(in.target && in.target->isTuple(), "Task needs a fixed target tuple");
  return in.target->toTupleRef().elements();
}
training::StepOutput Output(nn::InvocationResult result, Tensor loss, I samples) {
  training::StepOutput output;
  output.loss=std::move(loss); output.sample_count=samples;
  output.prediction=std::move(result); return output;
}
}
Tensor multilabel_loss(const Tensor& logits, const Tensor& labels, const Tensor& known,
                        const MultilabelLossOptions& o) {
  Matching(logits,labels); Require(logits.dim()==2, "Multilabel logits must be [B,C]");
  Validity(logits,known); Finite(logits); Finite(labels);
  Require(labels.ge(0).all().item<bool>() && labels.le(1).all().item<bool>(), "Multilabel labels outside [0,1]");
  Require(std::isfinite(o.gamma_positive) && std::isfinite(o.gamma_negative) &&
      o.gamma_positive>=0 && o.gamma_negative>=0 && std::isfinite(o.negative_clip) &&
      o.negative_clip>=0 && o.negative_clip<1, "Invalid asymmetric loss parameters");
  auto x=logits.to(torch::kFloat32), y=labels.to(torch::kFloat32);
  Tensor positive_weight=torch::ones({x.size(1)},x.options());
  if(o.positive_weights.defined()) {
    Require(o.positive_weights.dim()==1 && o.positive_weights.size(0)==x.size(1), "Positive weight count differs");
    positive_weight=o.positive_weights.to(x.options()); Finite(positive_weight);
    Require(positive_weight.ge(0).all().item<bool>(), "Negative positive class weight");
  }
  Tensor terms;
  if (!o.asymmetric) {
    terms=F::binary_cross_entropy_with_logits(x,y,
        F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(positive_weight).reduction(torch::kNone));
  } else {
    auto probability=x.sigmoid();
    auto negative=(1-probability+o.negative_clip).clamp_max(1.);
    auto positive=probability.clamp_min(1e-8);
    auto focal=(1-(probability*y+negative*(1-y))).clamp_min(0).pow(
        o.gamma_positive*y+o.gamma_negative*(1-y)).detach();
    terms=-(y*positive.log()*positive_weight+(1-y)*negative.clamp_min(1e-8).log())*focal;
  }
  return terms.masked_select(known).mean();
}
Tensor logits_distillation_loss(const Tensor& student, const Tensor& teacher,double temperature) {
  Matching(student,teacher); Require(student.dim()==2 && student.size(0)>0 &&
      std::isfinite(temperature) && temperature>0, "Invalid logits distillation inputs");
  Finite(student); Finite(teacher);
  auto s=(student.to(torch::kFloat32)/temperature).log_softmax(1);
  auto t=(teacher.detach().to(torch::kFloat32)/temperature).log_softmax(1);
  return (t.exp()*(t-s)).sum(1).mean()*(temperature*temperature);
}
Tensor bernoulli_distillation_loss(const Tensor& student,const Tensor& teacher,
                                   const Tensor& known,double temperature) {
  Matching(student,teacher); Validity(student,known);
  Require(std::isfinite(temperature)&&temperature>0,"Invalid Bernoulli temperature");
  Finite(student); Finite(teacher);
  auto s=student.to(torch::kFloat32)/temperature;
  auto t=(teacher.detach().to(torch::kFloat32)/temperature).sigmoid().clamp(1e-7,1-1e-7);
  auto cross=F::binary_cross_entropy_with_logits(s,t,F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone));
  auto entropy=-(t*t.log()+(1-t)*(1-t).log());
  return (cross-entropy).masked_select(known).mean()*(temperature*temperature);
}
Tensor feature_distillation_loss(const Tensor& student,const Tensor& teacher,bool normalize) {
  Matching(student,teacher); Require(student.dim()>=2,"Feature needs a channel axis");
  Finite(student); Finite(teacher);
  auto s=student.to(torch::kFloat32), t=teacher.detach().to(torch::kFloat32);
  if(normalize) { s=s/(s.square().sum(1,true)+1e-6).sqrt(); t=t/(t.square().sum(1,true)+1e-6).sqrt(); }
  return (s-t).square().mean();
}
Tensor semantic_segmentation_loss(const Tensor& logits,const Tensor& labels,I ignore,const Tensor& weights) {
  Require(logits.defined()&&logits.dim()==4&&logits.is_floating_point()&&labels.defined()&&
      labels.scalar_type()==torch::kInt64&&labels.dim()==3&&labels.size(0)==logits.size(0)&&
      labels.size(1)==logits.size(2)&&labels.size(2)==logits.size(3)&&labels.device()==logits.device(),"Invalid semantic targets");
  auto valid=labels.ne(ignore); Require(valid.any().item<bool>(),"Semantic batch contains no valid pixels");
  auto ids=labels.masked_select(valid);
  Require(ids.min().item<I>()>=0&&ids.max().item<I>()<logits.size(1),"Semantic class out of range");
  auto options=F::CrossEntropyFuncOptions().ignore_index(ignore);
  if(weights.defined()) {
    Require(weights.dim()==1&&weights.size(0)==logits.size(1),"Semantic class weight count differs");
    auto w=weights.to(logits.device(),torch::kFloat32); Finite(w);
    Require(w.ge(0).all().item<bool>(),"Negative semantic weights"); options.weight(w);
  }
  return F::cross_entropy(logits.to(torch::kFloat32),labels,options);
}
Tensor depth_loss(const Tensor& prediction,const Tensor& target,const Tensor& valid,const DepthLossOptions& o) {
  Matching(prediction,target); Validity(prediction,valid);
  Require(prediction.dim()==4&&prediction.size(1)==1&&std::isfinite(o.epsilon)&&o.epsilon>0&&
      std::isfinite(o.gradient_weight)&&o.gradient_weight>=0,"Invalid depth loss options/shape");
  auto p=prediction.to(torch::kFloat32),t=target.to(torch::kFloat32);
  Finite(p.masked_select(valid)); Finite(t.masked_select(valid));
  auto safe_t=torch::where(valid,t,torch::zeros_like(t));
  auto safe_p=torch::where(valid,p,torch::zeros_like(p));
  Tensor residual;
  if(o.scale_shift_invariant) {
    std::vector<Tensor> adjusted;
    for(I i=0;i<p.size(0);++i) {
      auto m=valid[i].to(torch::kFloat32),x=safe_p[i],y=safe_t[i];
      auto a=(m*x*x).sum(),b=(m*x).sum(),n=m.sum(),c=(m*x*y).sum(),d=(m*y).sum();
      auto det=a*n-b*b, denominator=det.clamp_min(o.epsilon);
      auto scale=torch::where(det.gt(o.epsilon),(c*n-b*d)/denominator,torch::ones_like(det));
      auto shift=torch::where(n.gt(0),(d-scale*b)/n.clamp_min(1),torch::zeros_like(n));
      adjusted.push_back(x*scale+shift-y);
    }
    residual=torch::stack(adjusted);
  } else {
    Require(t.masked_select(valid).gt(0).all().item<bool>(),"Metric depth labels must be positive");
    residual=safe_p.clamp_min(o.epsilon).log()-safe_t.clamp_min(o.epsilon).log();
  }
  auto loss=residual.masked_select(valid).abs().mean();
  if(o.gradient_weight) {
    for(I axis:{I{2},I{3}}) {
      auto length=residual.size(axis);
      if(length<2) continue;
      auto mask=valid.slice(axis,1,length)&valid.slice(axis,0,length-1);
      if(mask.any().item<bool>()) loss=loss+o.gradient_weight*(residual.slice(axis,1,length)-residual.slice(axis,0,length-1)).masked_select(mask).abs().mean();
    }
  }
  return loss;
}
Tensor simmim_mask(std::array<I,2> size,I patch,I mask_patch,double ratio,std::uint64_t seed) {
  Require(patch>0&&mask_patch>=patch&&mask_patch%patch==0&&size[0]>0&&size[1]>0&&
      size[0]%mask_patch==0&&size[1]%mask_patch==0&&std::isfinite(ratio)&&ratio>0&&ratio<=1,"Invalid SimMIM mask grid");
  I h=size[0]/mask_patch,w=size[1]/mask_patch;
  if(h>1048576/w) throw ResourceExhausted("SimMIM mask budget exceeded");
  std::vector<I> order(h*w); std::iota(order.begin(),order.end(),0);
  std::mt19937_64 generator(seed);
  for(I i=static_cast<I>(order.size())-1;i>0;--i) {
    std::uniform_int_distribution<I> index(0,i); std::swap(order[i],order[index(generator)]);
  }
  auto result=torch::zeros({h*w},torch::kBool);
  auto count=static_cast<I>(std::ceil(ratio*order.size()));
  for(I i=0;i<count;++i) result[order[i]]=true;
  return result.view({h,w}).repeat_interleave(mask_patch/patch,0).repeat_interleave(mask_patch/patch,1);
}
Tensor simmim_reconstruction_loss(const Tensor& reconstructed,const Tensor& target,const Tensor& mask,I patch) {
  Matching(reconstructed,target);
  Require(reconstructed.dim()==4&&mask.defined()&&mask.dim()==3&&mask.scalar_type()==torch::kBool&&
      mask.device()==target.device()&&patch>0&&mask.size(0)==target.size(0)&&
      mask.size(1)*patch==target.size(2)&&mask.size(2)*patch==target.size(3),"SimMIM mask/target dimensions differ");
  auto pixels=mask.repeat_interleave(patch,1).repeat_interleave(patch,2).unsqueeze(1).expand_as(target);
  Require(pixels.any().item<bool>(),"SimMIM requires masked pixels");
  Finite(reconstructed); Finite(target);
  return (reconstructed.to(torch::kFloat32)-target.detach().to(torch::kFloat32)).abs().masked_select(pixels).mean();
}
Tensor batch_hard_triplet_loss(const Tensor& embeddings,const Tensor& ids,double margin,bool normalize) {
  Require(embeddings.defined()&&embeddings.dim()==2&&embeddings.is_floating_point()&&
      ids.defined()&&ids.dim()==1&&ids.scalar_type()==torch::kInt64&&ids.size(0)==embeddings.size(0)&&
      ids.device()==embeddings.device()&&std::isfinite(margin)&&margin>=0,"Invalid ReID triplet inputs");
  if(embeddings.size(0)>1024) throw ResourceExhausted("Triplet batch exceeds pair budget");
  auto x=embeddings.to(torch::kFloat32); Finite(x);
  if(normalize)x=x/(x.square().sum(1,true)+1e-12).sqrt();
  auto distance=(x.unsqueeze(1)-x.unsqueeze(0)).square().sum(2).clamp_min(1e-12).sqrt();
  auto positive=ids.unsqueeze(1).eq(ids.unsqueeze(0))&~torch::eye(ids.size(0),torch::TensorOptions().dtype(torch::kBool).device(ids.device()));
  auto negative=ids.unsqueeze(1).ne(ids.unsqueeze(0));
  auto anchors=positive.any(1)&negative.any(1);
  Require(anchors.any().item<bool>(),"Triplet batch needs repeated and distinct identities");
  auto hard_p=std::get<0>(distance.masked_fill(~positive,-1e9).max(1));
  auto hard_n=std::get<0>(distance.masked_fill(~negative,1e9).min(1));
  return (hard_p-hard_n+margin).clamp_min(0).masked_select(anchors).mean();
}
training::StepFunction make_multilabel_step(MultilabelLossOptions options) {
  return [options](const training::StepInput& input) {
    const auto& t=Targets(input); Require(t.size()==2,"Multilabel targets=(values,known)");
    auto prediction=input.model.invoke(input.invocation); auto raw=raw_tensors(prediction.value);
    Require(raw.size()==1,"Multilabel model needs one logits output");
    auto policy=options;
    if(input.context.phase==training::StepPhase::validation)policy.positive_weights=Tensor{};
    return Output(std::move(prediction),multilabel_loss(raw[0],t[0].toTensor(),t[1].toTensor(),policy),raw[0].size(0));
  };
}
training::StepFunction make_semantic_step(I ignore,double auxiliary) {
  Require(std::isfinite(auxiliary)&&auxiliary>=0,"Invalid semantic auxiliary weight");
  return [ignore,auxiliary](const training::StepInput& input) {
    Require(input.target&&input.target->isTensor(),"Semantic task needs tensor labels");
    auto prediction=input.model.invoke(input.invocation); auto raw=raw_tensors(prediction.value);
    Require(!raw.empty()&&raw.size()<=2,"Semantic outputs=(logits[,aux])");
    auto loss=semantic_segmentation_loss(raw[0],input.target->toTensor(),ignore);
    if(raw.size()==2&&auxiliary)loss=loss+auxiliary*semantic_segmentation_loss(raw[1],input.target->toTensor(),ignore);
    return Output(std::move(prediction),loss,raw[0].size(0));
  };
}
training::StepFunction make_depth_step(DepthLossOptions options) {
  return [options](const training::StepInput& input) {
    const auto& t=Targets(input); Require(t.size()==2,"Depth targets=(values,valid)");
    auto prediction=input.model.invoke(input.invocation); auto raw=raw_tensors(prediction.value);
    Require(raw.size()==1,"Depth needs one prediction field");
    return Output(std::move(prediction),depth_loss(raw[0],t[0].toTensor(),t[1].toTensor(),options),raw[0].size(0));
  };
}
training::StepFunction make_simmim_step(I patch) {
  return [patch](const training::StepInput& input) {
    const auto& t=Targets(input); Require(t.size()==2,"SimMIM targets=(image,mask)");
    auto prediction=input.model.invoke(input.invocation); auto raw=raw_tensors(prediction.value);
    Require(raw.size()==1,"SimMIM needs one reconstruction output");
    return Output(std::move(prediction),simmim_reconstruction_loss(raw[0],t[0].toTensor(),t[1].toTensor(),patch),raw[0].size(0));
  };
}
}  // namespace jyppx::liuding::vision::tasks
