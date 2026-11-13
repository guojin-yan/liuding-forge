// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/models/extended/detectors.hpp>
#include "tensor_literals.hpp"
#include <torch/torch.h>
#include <cmath>
namespace jyppx::liuding::models::extended {
namespace {
using I=std::int64_t;
Tensor Flatten(const Tensor& x,I channels){return x.view({x.size(0),-1,channels,x.size(2),x.size(3)}).permute({0,3,4,1,2}).reshape({x.size(0),-1,channels});}
Tensor Anchors(I h,I w,I stride,bool fcos,const Tensor& ref){
  auto xs=torch::arange(w,ref.options())*stride,ys=torch::arange(h,ref.options())*stride;
  if(fcos){xs=xs+stride*.5;ys=ys+stride*.5;}
  auto center=torch::stack({xs.view({1,w}).expand({h,w}),ys.view({h,1}).expand({h,w})},2).reshape({-1,2});
  std::vector<double> base;
  if(fcos)base={-stride*.5,-stride*.5,stride*.5,stride*.5};
  else for(double ratio:{.5,1.,2.})for(I scale=0;scale<3;++scale){auto size=stride*4*std::pow(2.,scale/3.);auto bw=std::round(size/std::sqrt(ratio)),bh=std::round(size*std::sqrt(ratio));base.insert(base.end(),{-bw/2,-bh/2,bw/2,bh/2});}
  auto offsets=detail::tensor_literals(base,ref.options()).view({-1,4});auto centers=torch::cat({center,center},1);
  return (centers.unsqueeze(1)+offsets.unsqueeze(0)).reshape({-1,4});
}
}
DenseDetectorImpl::DenseDetectorImpl(DenseDetectorOptions o):options_(std::move(o)){
  auto c=options_.pyramid.channels;bool fcos=options_.kind==DenseDetectorKind::fcos;
  if(options_.num_classes<1||options_.num_classes>10000||options_.max_anchors<1||options_.max_anchors>1048576)throw InvalidArgument("Invalid dense detector options");
  pyramid_=register_module("backbone",FeaturePyramid(options_.pyramid));
  cls_tower_=register_module("classification_tower",torch::nn::Sequential());box_tower_=register_module("regression_tower",torch::nn::Sequential());
  for(I i=0;i<4;++i)for(auto* tower:{&cls_tower_,&box_tower_}){
    (*tower)->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c,c,3).padding(1)));
    if(fcos)(*tower)->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(32,c)));
    (*tower)->push_back(torch::nn::ReLU());}
  I a=fcos?1:9;
  classes_=register_module("class_logits",torch::nn::Conv2d(torch::nn::Conv2dOptions(c,a*options_.num_classes,3).padding(1)));
  boxes_=register_module("bbox_regression",torch::nn::Conv2d(torch::nn::Conv2dOptions(c,a*4,3).padding(1)));
  if(fcos)center_=register_module("bbox_ctrness",torch::nn::Conv2d(torch::nn::Conv2dOptions(c,1,3).padding(1)));
  torch::NoGradGuard guard;
  for(const auto& module:{cls_tower_.ptr(),box_tower_.ptr()})for(const auto& child:module->modules(false))
    if(auto conv=std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(child)){torch::nn::init::normal_(conv->weight,0,.01);torch::nn::init::zeros_(conv->bias);}
  for(auto* conv:{&classes_,&boxes_,&center_})if(*conv){torch::nn::init::normal_((*conv)->weight,0,.01);torch::nn::init::zeros_((*conv)->bias);}
  classes_->bias.fill_(-std::log(99.));
}
std::vector<Tensor> DenseDetectorImpl::forward(const Tensor& images){
  auto features=pyramid_->forward(images);std::vector<Tensor> logits,reg,ctr,anchors;std::vector<I> counts;
  bool fcos=options_.kind==DenseDetectorKind::fcos;I total=0;
  for(I i=0;i<5;++i){auto f=features[i];I n=f.size(2)*f.size(3)*(fcos?1:9);total+=n;
    if(total>options_.max_anchors||total*options_.num_classes>67108864)throw ResourceExhausted("Dense detector output budget exceeded");
    auto cls=cls_tower_->forward(f),box=box_tower_->forward(f);logits.push_back(Flatten(classes_(cls),options_.num_classes));
    auto r=Flatten(boxes_(box),4);reg.push_back(fcos?torch::relu(r):r);
    if(fcos)ctr.push_back(Flatten(center_(box),1));anchors.push_back(Anchors(f.size(2),f.size(3),8LL<<i,fcos,images));counts.push_back(n);}
  return {torch::cat(logits,1),torch::cat(reg,1),fcos?torch::cat(ctr,1):torch::empty({images.size(0),total,0},images.options()),
          torch::cat(anchors,0),detail::tensor_literals(counts,torch::TensorOptions().dtype(torch::kInt64).device(images.device()))};
}
nn::ModelAdapter adapt(DenseDetector model){
  if(!model)throw InvalidArgument("Dense detector holder is empty");
  std::vector<ValueSchema> outputs;for(int i=0;i<5;++i)outputs.push_back(ValueSchema::tensor({}));
  return nn::adapt_holder(model,[](DenseDetectorImpl& m,const nn::Invocation& x){auto raw=m.forward(x.positional[0].toTensor());std::vector<c10::IValue> v;for(auto& t:raw)v.emplace_back(t);return c10::ivalue::Tuple::create(std::move(v));},
     {.positional={ValueSchema::tensor({})},.keyword={},.allow_extra_keywords=false},ValueSchema::tuple(std::move(outputs)));
}
}
