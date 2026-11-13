// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/dense_detection.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
#include <jyppx/liuding/vision/ops/boxes.hpp>
#include <jyppx/liuding/vision/ops/deformable.hpp>
#include <jyppx/liuding/vision/ops/roi.hpp>
#include <torch/torch.h>
#include <cmath>
namespace jyppx::liuding::vision::tasks {
namespace {
using I=std::int64_t;
Tensor Smooth(const Tensor& x,double beta){auto a=x.abs();return torch::where(a<beta,.5*x.square()/beta,a-.5*beta);}
Tensor Focal(const Tensor& logits,const Tensor& target){auto p=logits.sigmoid();auto pt=p*target+(1-p)*(1-target);
  return torch::nn::functional::binary_cross_entropy_with_logits(logits,target,
    torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))*(1-pt).square()*(.25*target+.75*(1-target));}
Tensor Boxes(const Tensor& anchors,const Tensor& deltas,bool fcos){
  if(!fcos)return vision_ops::decode_box_deltas(anchors,deltas);
  auto center=(anchors.slice(1,0,2)+anchors.slice(1,2,4))*.5;
  auto stride=(anchors.select(1,2)-anchors.select(1,0)).unsqueeze(1);
  return torch::cat({center-deltas.slice(1,0,2)*stride,center+deltas.slice(1,2,4)*stride},1);
}
void Check(std::span<const Tensor> x,const Tensor& targets,const DenseDetectionLossOptions& o){
  if(x.size()!=5||o.num_classes<1||o.num_classes>10000||o.max_assignment_pairs<1||!std::isfinite(o.center_sampling_radius)||o.center_sampling_radius<=0||
     !std::isfinite(o.negative_positive_ratio)||o.negative_positive_ratio<1||!targets.defined()||targets.dim()!=2||targets.size(1)!=6||targets.size(0)>4096||
     !targets.is_floating_point()||x[0].dim()!=3||x[1].dim()!=3||x[1].size(2)!=4||x[0].size(0)<1||x[0].size(0)!=x[1].size(0)||x[0].size(1)!=x[1].size(1)||
     x[3].dim()!=2||x[3].size(0)!=x[0].size(1)||x[3].size(1)!=4||x[0].size(2)!=o.num_classes+(o.kind==DenseDetectionCriterion::ssd?1:0))
    throw InvalidArgument("Invalid dense detection loss contract");
  for(const auto& t:x)if(t.device()!=x[0].device()||(!torch::isfinite(t).all().item<bool>()))throw InvalidArgument("Nonfinite/mixed device dense output");
  if(targets.device()!=x[0].device()||!torch::isfinite(targets).all().item<bool>())throw InvalidArgument("Invalid detection targets");
  if(targets.numel()&&(!(targets.slice(1,0,2)==targets.slice(1,0,2).round()).all().item<bool>()||targets.select(1,0).min().item<double>()<0||
     targets.select(1,0).max().item<double>()>=x[0].size(0)||targets.select(1,1).min().item<double>()<0||targets.select(1,1).max().item<double>()>=o.num_classes||
     !(targets.slice(1,4,6)>targets.slice(1,2,4)).all().item<bool>()))throw InvalidArgument("Dense detection target identities/extents invalid");
}
}
TaskLoss dense_detection_loss(std::span<const Tensor> x,const Tensor& targets,const DenseDetectionLossOptions& o){
  Check(x,targets,o);bool fcos=o.kind==DenseDetectionCriterion::fcos,ssd=o.kind==DenseDetectionCriterion::ssd;
  auto logits=x[0].to(torch::kFloat32),reg=x[1].to(torch::kFloat32),anchors=x[3].to(torch::kFloat32);
  if(fcos&&(x[2].sizes()!=torch::IntArrayRef({logits.size(0),logits.size(1),1})))throw InvalidArgument("FCOS centerness shape mismatch");
  auto cls_loss=logits.sum()*0,box_loss=reg.sum()*0,ctr_loss=fcos?x[2].sum()*0:reg.sum()*0;I positives=0;
  auto center=(anchors.slice(1,0,2)+anchors.slice(1,2,4))*.5;auto stride=anchors.select(1,2)-anchors.select(1,0);
  for(I b=0;b<logits.size(0);++b){auto gt=targets.index({targets.select(1,0)==b}).to(torch::kFloat32);
    auto labels=torch::full({anchors.size(0)},-1,anchors.options().dtype(torch::kInt64));auto valid=torch::ones_like(labels,torch::kBool);Tensor matched;
    {torch::NoGradGuard guard;
      if(gt.size(0)){
        if(anchors.size(0)>o.max_assignment_pairs/gt.size(0))throw ResourceExhausted("Dense assignment pair budget exceeded");
        if(fcos){auto lo=center.unsqueeze(1)-gt.slice(1,2,4).unsqueeze(0),hi=gt.slice(1,4,6).unsqueeze(0)-center.unsqueeze(1);
          auto distance=torch::cat({lo,hi},2);auto maxd=std::get<0>(distance.max(2));auto mind=std::get<0>(distance.min(2));
          auto centers=(gt.slice(1,2,4)+gt.slice(1,4,6))*.5;
          auto in_center=std::get<0>((center.unsqueeze(1)-centers.unsqueeze(0)).abs().max(2))<stride.unsqueeze(1)*o.center_sampling_radius;
          auto lower=stride*4,upper=stride*8;lower=torch::where(stride==stride.min(),torch::zeros_like(lower),lower);upper=torch::where(stride==stride.max(),torch::full_like(upper,1e9),upper);
          auto eligible=(mind>0)&in_center&(maxd>lower.unsqueeze(1))&(maxd<upper.unsqueeze(1));
          auto area=(gt.select(1,4)-gt.select(1,2))*(gt.select(1,5)-gt.select(1,3));
          auto choice=area.unsqueeze(0).expand({anchors.size(0),-1}).masked_fill(~eligible,1e10).min(1);
          auto ids=std::get<1>(choice);auto pos=std::get<0>(choice)<1e10;
          labels=torch::where(pos,gt.select(1,1).to(torch::kInt64).index_select(0,ids),labels);matched=gt.slice(1,2,6).index_select(0,ids);
        }else{
          auto iou=vision_ops::pairwise_box_iou(anchors,gt.slice(1,2,6),{.max_pairs=o.max_assignment_pairs});auto best=iou.max(1);auto ids=std::get<1>(best);
          auto max_iou=std::get<0>(best);auto positive=max_iou>=.5;
          if(ssd){auto best_anchor=std::get<1>(iou.max(0));ids.index_copy_(0,best_anchor,torch::arange(gt.size(0),ids.options()));positive.index_fill_(0,best_anchor,true);}
          else positive=positive|(iou==std::get<0>(iou.max(0,true))).any(1);
          if(!ssd)valid=(max_iou<.4)|positive;
          labels=torch::where(positive,gt.select(1,1).to(torch::kInt64).index_select(0,ids),labels);matched=gt.slice(1,2,6).index_select(0,ids);
        }
      }
    }
    auto ids=(labels>=0).nonzero().flatten();I n=ids.numel();positives+=n;double denominator=fcos||ssd?1.:static_cast<double>((std::max)(I{1},n))*logits.size(0);
    if(ssd){auto expected=labels+1;auto ce=torch::nn::functional::cross_entropy(logits[b],expected,torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
      auto negative=(labels<0).nonzero().flatten();I nneg=(std::min)(negative.numel(),static_cast<I>(o.negative_positive_ratio*n));
      cls_loss=cls_loss+ce.index_select(0,ids).sum();if(nneg)cls_loss=cls_loss+std::get<0>(ce.index_select(0,negative).topk(nneg)).sum();
    }else{auto expected=torch::zeros_like(logits[b]);if(n)expected.index_put_({ids,labels.index_select(0,ids)},1);
      cls_loss=cls_loss+Focal(logits[b],expected).index({valid}).sum()/denominator;}
    if(n){auto a=anchors.index_select(0,ids),r=reg[b].index_select(0,ids),m=matched.index_select(0,ids);
      if(fcos){box_loss=box_loss+(1-vision_ops::aligned_generalized_iou(Boxes(a,r,true),m)).sum();
        auto c=center.index_select(0,ids);auto d=torch::cat({c-m.slice(1,0,2),m.slice(1,2,4)-c},1);
        auto lr=torch::stack({d.select(1,0),d.select(1,2)},1),tb=torch::stack({d.select(1,1),d.select(1,3)},1);
        auto target=((std::get<0>(lr.min(1))/std::get<0>(lr.max(1)))*(std::get<0>(tb.min(1))/std::get<0>(tb.max(1)))).sqrt();
        ctr_loss=ctr_loss+torch::nn::functional::binary_cross_entropy_with_logits(x[2][b].squeeze(1).index_select(0,ids).to(torch::kFloat32),target,torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kSum));
      }else{auto expected=vision_ops::encode_box_deltas(a,m,ssd?std::array<double,4>{10,10,5,5}:std::array<double,4>{1,1,1,1});
        box_loss=box_loss+Smooth(r-expected,ssd?1.:1./9).sum()/denominator;}
    }
  }
  if(fcos||ssd){double n=static_cast<double>((std::max)(1LL,positives));cls_loss=cls_loss/n;box_loss=box_loss/n;ctr_loss=ctr_loss/n;}
  return{cls_loss+box_loss+ctr_loss,{{"class",cls_loss},{"box",box_loss},{"centerness",ctr_loss}}};
}
training::StepFunction make_dense_detection_step(DenseDetectionLossOptions o){return[o](const training::StepInput& in){
  if(!in.target||!in.target->isTensor())throw InvalidArgument("Dense detector needs tensor box targets");auto prediction=in.model.invoke(in.invocation);
  auto loss=dense_detection_loss(raw_tensors(prediction.value),in.target->toTensor(),o);training::StepOutput out;
  out.loss=loss.total;out.prediction=std::move(prediction);out.sample_count=in.invocation.positional[0].toTensor().size(0);return out;};}
std::vector<ImageResult> decode_dense_detection(std::span<const Tensor> x,std::array<I,2> size,DenseDetectionCriterion kind,const ResultOptions& o){
  if(x.size()!=5)throw InvalidArgument("Dense detector decoder requires five raw tensors");torch::NoGradGuard guard;std::vector<Tensor> boxes;
  bool fcos=kind==DenseDetectionCriterion::fcos,ssd=kind==DenseDetectionCriterion::ssd;
  for(I b=0;b<x[0].size(0);++b)boxes.push_back(ssd?vision_ops::decode_box_deltas(x[3],x[1][b],{10,10,5,5}):Boxes(x[3],x[1][b],fcos));
  auto scores=ssd?x[0].softmax(2).slice(2,1):x[0].sigmoid();if(fcos)scores=(scores*x[2].sigmoid()).sqrt();
  return decode_detection_results({.boxes=torch::stack(boxes),.class_scores=scores},size,o);
}
}
