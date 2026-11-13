// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/models/extended/region.hpp>
#include "tensor_literals.hpp"
#include <jyppx/liuding/models/registry.hpp>
#include <jyppx/liuding/vision/ops/roi.hpp>
#include <jyppx/liuding/vision/ops/deformable.hpp>
#include <jyppx/liuding/vision/ops/boxes.hpp>
#include <torch/torch.h>
#include <cmath>
namespace jyppx::liuding::models::extended {
namespace {
using I=std::int64_t;
class VggFeatures:public torch::nn::Module{
 public:
  explicit VggFeatures(bool ssd=false){stages=register_module("stages",torch::nn::ModuleList());I in=3;
    std::array<I,5> widths{64,128,256,512,512},depths{2,2,3,3,3};
    for(I i=0;i<5;++i){torch::nn::Sequential stage;
      if(i)stage->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2).ceil_mode(ssd&&i==3)));
      for(I j=0;j<depths[i];++j){stage->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in,widths[i],3).padding(1)));stage->push_back(torch::nn::ReLU());in=widths[i];}stages->push_back(stage);}}
  std::vector<Tensor> forward(const Tensor& input){if(input.dim()!=4||input.size(1)!=3||input.size(2)<32||input.size(3)<32)throw InvalidArgument("VGG expects RGB NCHW images at least 32px");
    auto x=input;std::vector<Tensor> f;for(auto& stage:*stages){x=stage->as<torch::nn::Sequential>()->forward(x);f.push_back(x);}return f;}
  torch::nn::ModuleList stages{nullptr};
};
class VggClassifier:public torch::nn::Module{
 public:
  VggClassifier(I classes){if(classes<1||classes>10000)throw InvalidArgument("Invalid VGG classes");encoder=register_module("features",std::make_shared<VggFeatures>());
    classifier=register_module("classifier",torch::nn::Sequential(torch::nn::Linear(512*7*7,4096),torch::nn::ReLU(),torch::nn::Dropout(.5),torch::nn::Linear(4096,4096),torch::nn::ReLU(),torch::nn::Dropout(.5),torch::nn::Linear(4096,classes)));}
  Tensor forward(const Tensor& x){auto f=encoder->forward(x).back();f=torch::max_pool2d(f,{2,2},{2,2});return classifier->forward(torch::adaptive_avg_pool2d(f,{7,7}).flatten(1));}
  std::shared_ptr<VggFeatures> encoder;torch::nn::Sequential classifier{nullptr};
};
class Ssd:public torch::nn::Module{
 public:
  explicit Ssd(I classes):classes_(classes){if(classes<1||classes>10000)throw InvalidArgument("Invalid SSD classes");encoder=register_module("backbone",std::make_shared<VggFeatures>(true));
    l2=register_parameter("scale_weight",torch::full({512},20.));extra=register_module("extra",torch::nn::ModuleList());
    extra->push_back(torch::nn::Sequential(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(1).padding(1)),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(512,1024,3).padding(6).dilation(6)),torch::nn::ReLU(),torch::nn::Conv2d(torch::nn::Conv2dOptions(1024,1024,1)),torch::nn::ReLU()));
    std::array<I,4> ins{1024,512,256,256},mids{256,128,128,128},outs{512,256,256,256};
    for(I i=0;i<4;++i)extra->push_back(torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(ins[i],mids[i],1)),torch::nn::ReLU(),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(mids[i],outs[i],3).stride(i<2?2:1).padding(i<2?1:0)),torch::nn::ReLU()));
    cls=register_module("classification_head",torch::nn::ModuleList());box=register_module("regression_head",torch::nn::ModuleList());
    std::array<I,6> channels{512,1024,512,256,256,256},a{4,6,6,6,4,4};
    for(I i=0;i<6;++i){cls->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(channels[i],a[i]*(classes+1),3).padding(1)));box->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(channels[i],a[i]*4,3).padding(1)));}
    std::vector<float> values;std::array<I,6> grids{38,19,10,5,3,1},steps{8,16,32,64,100,300};std::array<double,7> scales{.07,.15,.33,.51,.69,.87,1.05};
    for(I i=0;i<6;++i){std::vector<std::array<double,2>> wh{{scales[i],scales[i]},{std::sqrt(scales[i]*scales[i+1]),std::sqrt(scales[i]*scales[i+1])}};
      for(I ratio=2;ratio<=(i>0&&i<4?3:2);++ratio){double r=std::sqrt(ratio);wh.push_back({scales[i]*r,scales[i]/r});wh.push_back({scales[i]/r,scales[i]*r});}
      for(I y=0;y<grids[i];++y)for(I x=0;x<grids[i];++x)for(auto size:wh){double cx=(x+.5)*steps[i],cy=(y+.5)*steps[i],w=(std::min)(1.,size[0])*300,h=(std::min)(1.,size[1])*300;
        for(double v:{cx-w/2,cy-h/2,cx+w/2,cy+h/2})values.push_back(static_cast<float>(v));}}
    defaults=register_buffer("default_boxes",torch::tensor(values).view({-1,4}));
    torch::NoGradGuard guard;for(const auto& branch:{extra.ptr(),cls.ptr(),box.ptr()})for(auto& m:branch->modules(false))if(auto c=std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(m)){torch::nn::init::xavier_uniform_(c->weight);torch::nn::init::zeros_(c->bias);}
  }
  c10::IValue forward(const Tensor& x){if(x.dim()!=4||x.size(2)!=300||x.size(3)!=300||x.size(0)>64)throw InvalidArgument("SSD300 requires fixed RGB 300x300 images");
    auto features=encoder->forward(x);auto a=features[3];a=a/(a.square().sum(1,true).sqrt().clamp_min(1e-12))*l2.view({1,-1,1,1});std::vector<Tensor> f{a};
    auto z=features[4];for(auto& branch:*extra){z=branch->as<torch::nn::Sequential>()->forward(z);f.push_back(z);}
    std::vector<Tensor> scores,deltas;std::vector<I> counts;
    for(I i=0;i<6;++i){auto s=cls[i]->as<torch::nn::Conv2d>()->forward(f[i]),b=box[i]->as<torch::nn::Conv2d>()->forward(f[i]);
      scores.push_back(s.view({x.size(0),-1,classes_+1,s.size(2),s.size(3)}).permute({0,3,4,1,2}).reshape({x.size(0),-1,classes_+1}));
      deltas.push_back(b.view({x.size(0),-1,4,b.size(2),b.size(3)}).permute({0,3,4,1,2}).reshape({x.size(0),-1,4}));counts.push_back(deltas.back().size(1));}
    return c10::ivalue::Tuple::create({torch::cat(scores,1),torch::cat(deltas,1),torch::empty({x.size(0),8732,0},x.options()),defaults,detail::tensor_literals(counts,defaults.options().dtype(torch::kInt64))});
  }
  I classes_;std::shared_ptr<VggFeatures> encoder;torch::nn::ModuleList extra{nullptr},cls{nullptr},box{nullptr};Tensor l2,defaults;
};
Tensor Sample(const Tensor& mask,I n){auto rows=mask.nonzero().flatten();if(rows.numel()<=n)return rows;return rows.index_select(0,torch::randperm(rows.numel(),rows.options()).slice(0,0,n));}
class Fast:public torch::nn::Module{
 public:
  explicit Fast(FastRcnnOptions options):o(options){if(o.num_classes<1||o.num_classes>10000||o.max_proposals<1||o.max_proposals>4096||o.roi_batch_size<1||o.roi_batch_size>4096||!std::isfinite(o.positive_fraction)||o.positive_fraction<=0||o.positive_fraction>1)throw InvalidArgument("Invalid Fast R-CNN options");
    I channels=256,representation=1024;
    if(o.profile==FastRcnnProfile::vgg16_roi_pool){vgg=register_module("backbone",std::make_shared<VggFeatures>());channels=512;representation=4096;}
    else{backbone=std::make_unique<nn::ModelAdapter>(create_feature_extractor("resnet50"));register_module("backbone",backbone->state().shared_module());
      lateral=register_module("lateral",torch::nn::ModuleList());smooth=register_module("smooth",torch::nn::ModuleList());for(I in:{256,512,1024,2048}){
        lateral->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in,256,1)));smooth->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(256,256,3).padding(1)));}}
    head=register_module("box_head",torch::nn::Sequential(torch::nn::Linear(channels*7*7,representation),torch::nn::ReLU()));
    if(vgg)head->push_back(torch::nn::Dropout(.5));head->push_back(torch::nn::Linear(representation,representation));head->push_back(torch::nn::ReLU());if(vgg)head->push_back(torch::nn::Dropout(.5));
    classifier=register_module("cls_score",torch::nn::Linear(representation,o.num_classes+1));regressor=register_module("bbox_pred",torch::nn::Linear(representation,4*(o.num_classes+1)));
  }
  c10::IValue forward(const Tensor& images,Tensor proposals,const Tensor& targets,const Tensor& sizes){
    if(images.dim()!=4||proposals.dim()!=2||proposals.size(1)!=5||proposals.size(0)>o.max_proposals||targets.dim()!=2||targets.size(1)!=6||targets.size(0)>4096||sizes.sizes()!=torch::IntArrayRef({images.size(0),2})||proposals.device()!=images.device()||targets.device()!=images.device()||sizes.device()!=images.device()||!torch::isfinite(proposals).all().item<bool>()||!torch::isfinite(targets).all().item<bool>())throw InvalidArgument("Invalid Fast R-CNN external proposal contract");
    if(proposals.numel()&&(!(proposals.select(1,0)==proposals.select(1,0).round()).all().item<bool>()||proposals.select(1,0).min().item<double>()<0||proposals.select(1,0).max().item<double>()>=images.size(0)||!(proposals.slice(1,3,5)>proposals.slice(1,1,3)).all().item<bool>()))throw InvalidArgument("Invalid external proposal geometry");
    Tensor labels,matched;std::vector<Tensor> rois,ys,gs;
    {torch::NoGradGuard guard;for(I b=0;b<images.size(0);++b){auto p=proposals.index({proposals.select(1,0)==b});auto gt=targets.index({targets.select(1,0)==b});
      if(is_training()&&gt.size(0))p=torch::cat({p,torch::cat({torch::full({gt.size(0),1},static_cast<double>(b),gt.options()),gt.slice(1,2,6)},1)},0);
      auto y=torch::zeros({p.size(0)},images.options().dtype(torch::kInt64)),m=torch::zeros({p.size(0),4},images.options());
      if(gt.size(0)&&p.size(0)){auto best=vision_ops::pairwise_box_iou(p.slice(1,1,5),gt.slice(1,2,6),{.max_pairs=16777216}).max(1);
        auto ids=std::get<1>(best);m=gt.slice(1,2,6).index_select(0,ids);y=torch::where(std::get<0>(best)>=.5,gt.select(1,1).to(torch::kInt64).index_select(0,ids)+1,y);}
      if(is_training()){auto pos=Sample(y>0,static_cast<I>(o.roi_batch_size*o.positive_fraction)),neg=Sample(y==0,o.roi_batch_size-pos.numel());auto ids=torch::cat({pos,neg});p=p.index_select(0,ids);y=y.index_select(0,ids);m=m.index_select(0,ids);}
      rois.push_back(p);ys.push_back(y);gs.push_back(m);}
      proposals=torch::cat(rois);labels=torch::cat(ys);matched=torch::cat(gs);}
    auto f=features(images);
    auto pooled=vgg?vision_ops::roi_pool(f[0],proposals,{7,7},1./16):vision_ops::multiscale_roi_align(f,proposals,{4,8,16,32});
    auto representation=head->forward(pooled.flatten(1));return c10::ivalue::Tuple::create({torch::empty({images.size(0),0},images.options()),torch::empty({images.size(0),0,4},images.options()),torch::empty({0,4},images.options()),proposals,classifier(representation),regressor(representation),labels,matched,sizes});
  }
  std::vector<Tensor> features(const Tensor& images){if(vgg)return{vgg->forward(images).back()};
    auto value=backbone->invoke({.positional={images}}).value;auto raw=value.toTupleRef().elements();std::vector<Tensor> f(4);
      for(I i=0;i<4;++i)f[i]=lateral[i]->as<torch::nn::Conv2d>()->forward(raw[i].toTensor());
      for(I i=2;i>=0;--i)f[i]=f[i]+torch::nn::functional::interpolate(f[i+1],torch::nn::functional::InterpolateFuncOptions().size(std::vector<I>{f[i].size(2),f[i].size(3)}).mode(torch::kNearest));
      for(I i=0;i<4;++i)f[i]=smooth[i]->as<torch::nn::Conv2d>()->forward(f[i]);return f;}
  std::vector<Tensor> box_head(const Tensor& pooled){auto z=head->forward(pooled.flatten(1));return{classifier(z),regressor(z)};}
  FastRcnnOptions o;std::unique_ptr<nn::ModelAdapter> backbone;std::shared_ptr<VggFeatures> vgg;torch::nn::ModuleList lateral{nullptr},smooth{nullptr};torch::nn::Sequential head{nullptr};torch::nn::Linear classifier{nullptr},regressor{nullptr};
};
ValueSchema RawSchema(int n){std::vector<ValueSchema> v;for(int i=0;i<n;++i)v.push_back(ValueSchema::tensor({}));return ValueSchema::tuple(std::move(v));}
}
nn::ModelAdapter create_vgg16(I c){return nn::adapt_module(std::make_shared<VggClassifier>(c),[](VggClassifier& m,const nn::Invocation& x){return m.forward(x.positional[0].toTensor());},
  {.positional={ValueSchema::tensor({})},.keyword={},.allow_extra_keywords=false},ValueSchema::tensor({}));}
nn::ModelAdapter create_ssd300_vgg16(I c){return nn::adapt_module(std::make_shared<Ssd>(c),[](Ssd& m,const nn::Invocation& x){return m.forward(x.positional[0].toTensor());},
  {.positional={ValueSchema::tensor({})},.keyword={},.allow_extra_keywords=false},RawSchema(5));}
nn::ModelAdapter create_fast_rcnn(FastRcnnOptions o){return nn::adapt_module(std::make_shared<Fast>(o),[](Fast& m,const nn::Invocation& x){return m.forward(x.positional[0].toTensor(),x.positional[1].toTensor(),x.positional[2].toTensor(),x.positional[3].toTensor());},
  {.positional={ValueSchema::tensor({}),ValueSchema::tensor({}),ValueSchema::tensor({}),ValueSchema::tensor({})},.keyword={},.allow_extra_keywords=false},RawSchema(9));}
std::vector<Tensor> fast_rcnn_features(nn::ModelAdapter& m,const Tensor& x){auto f=std::dynamic_pointer_cast<Fast>(m.state().shared_module());if(!f)throw CompatibilityError("Not a native Fast R-CNN");return f->features(x);}
std::vector<Tensor> fast_rcnn_box_head(nn::ModelAdapter& m,const Tensor& x){auto f=std::dynamic_pointer_cast<Fast>(m.state().shared_module());if(!f)throw CompatibilityError("Not a native Fast R-CNN");return f->box_head(x);}
FastRcnnProfile fast_rcnn_profile(nn::ModelAdapter& m){auto f=std::dynamic_pointer_cast<Fast>(m.state().shared_module());if(!f)throw CompatibilityError("Not a native Fast R-CNN");return f->o.profile;}

}
