// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/models/extended/dense.hpp>
#include <torch/torch.h>
namespace jyppx::liuding::models::extended {
namespace {
using I=std::int64_t;
Tensor Up(const Tensor& x,I h,I w){return torch::nn::functional::interpolate(x,
  torch::nn::functional::InterpolateFuncOptions().size(std::vector<I>{h,w}).mode(torch::kBilinear).align_corners(false));}
torch::nn::Sequential ConvBn(I in,I out,I kernel,I dilation=1){return torch::nn::Sequential(
  torch::nn::Conv2d(torch::nn::Conv2dOptions(in,out,kernel).padding(kernel==1?0:dilation).dilation(dilation).bias(false)),
  torch::nn::BatchNorm2d(out),torch::nn::ReLU());}
torch::nn::Sequential FcnHead(I in,I out){auto c=in/4;return torch::nn::Sequential(
  torch::nn::Conv2d(torch::nn::Conv2dOptions(in,c,3).padding(1).bias(false)),torch::nn::BatchNorm2d(c),torch::nn::ReLU(),
  torch::nn::Dropout(.1),torch::nn::Conv2d(torch::nn::Conv2dOptions(c,out,1)));}
class AspPool:public torch::nn::Module{
 public:
  AspPool(){conv=register_module("conv",ConvBn(2048,256,1));}
  Tensor forward(const Tensor& x){return Up(conv->forward(x.mean({2,3},true)),x.size(2),x.size(3));}
  torch::nn::Sequential conv{nullptr};
};
class Asp:public torch::nn::Module{
 public:
  Asp(){branches=register_module("convs",torch::nn::ModuleList());branches->push_back(ConvBn(2048,256,1));
    for(I rate:{12,24,36})branches->push_back(ConvBn(2048,256,3,rate));pool=register_module("pool",std::make_shared<AspPool>());
    project=register_module("project",torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(1280,256,1).bias(false)),torch::nn::BatchNorm2d(256),torch::nn::ReLU(),torch::nn::Dropout(.5)));}
  Tensor forward(const Tensor& x){std::vector<Tensor> out;for(auto& m:*branches)out.push_back(m->as<torch::nn::Sequential>()->forward(x));out.push_back(pool->forward(x));return project->forward(torch::cat(out,1));}
  torch::nn::ModuleList branches{nullptr};std::shared_ptr<AspPool> pool;torch::nn::Sequential project{nullptr};
};
class Field:public torch::nn::Module{
 public:
  explicit Field(DenseFieldOptions options):o(options){
    if(o.num_classes<1||o.num_classes>10000)throw InvalidArgument("Invalid dense field classes");
    if(o.kind==DenseFieldKind::depth_fpn){pyramid=register_module("backbone",FeaturePyramid(PyramidOptions{}));head=register_module("classifier",torch::nn::Sequential(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(256,128,3).padding(1)),torch::nn::ReLU(),torch::nn::Conv2d(torch::nn::Conv2dOptions(128,1,1))));}
    else{ResNetOptions r;r.num_classes=0;r.replace_stride_with_dilation={false,true,true};backbone=register_module("backbone",resnet50(r));
      if(o.kind==DenseFieldKind::fcn)head=register_module("classifier",FcnHead(2048,o.num_classes));
      else{aspp=register_module("aspp",std::make_shared<Asp>());head=register_module("classifier",torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256,256,3).padding(1).bias(false)),torch::nn::BatchNorm2d(256),torch::nn::ReLU(),torch::nn::Conv2d(torch::nn::Conv2dOptions(256,o.num_classes,1))));}
      if(o.auxiliary)aux=register_module("aux_classifier",FcnHead(1024,o.num_classes));}
  }
  c10::IValue forward(const Tensor& x){
    if(x.dim()!=4||x.size(1)!=3||x.size(2)<32||x.size(3)<32||x.size(0)*x.size(2)*x.size(3)>16777216)throw InvalidArgument("Invalid dense field image shape");
    if(o.kind==DenseFieldKind::depth_fpn)return Up(torch::softplus(head->forward(pyramid->forward(x)[0]))+1e-6,x.size(2),x.size(3));
    auto f=backbone->forward_features(x);auto main=Up(head->forward(aspp?aspp->forward(f[3]):f[3]),x.size(2),x.size(3));
    if(!aux)return main;return c10::ivalue::Tuple::create({main,Up(aux->forward(f[2]),x.size(2),x.size(3))});
  }
  DenseFieldOptions o;ResNet backbone{nullptr};FeaturePyramid pyramid{nullptr};torch::nn::Sequential head{nullptr},aux{nullptr};std::shared_ptr<Asp> aspp;
};
class SimMim:public torch::nn::Module{
 public:
  explicit SimMim(SimMimEncoder kind,bool recomputation):swin(kind==SimMimEncoder::swin_tiny){
    I width=0,dimension=0;
    if(swin){SwinOptions o;o.activation_recomputation=recomputation;o.num_classes=0;o.dynamic_image_size=true;sw=register_module("encoder",swin_tiny_patch4_window7_224(o));width=768;dimension=96;stride=32;
      token=register_parameter("mask_token",torch::empty({1,1,1,dimension}));}
    else{ViTOptions o;o.activation_recomputation=recomputation;o.num_classes=0;o.dynamic_image_size=true;vit=register_module("encoder",ViT(kind==SimMimEncoder::vit_base?ViTVariant::base:ViTVariant::tiny,o));
      width=dimension=kind==SimMimEncoder::vit_base?768:192;stride=16;token=register_parameter("mask_token",torch::empty({1,1,dimension}));}
    decoder=register_module("decoder",torch::nn::Conv2d(torch::nn::Conv2dOptions(width,3*stride*stride,1)));
    torch::NoGradGuard guard;torch::nn::init::normal_(token,0,.02);
  }
  Tensor forward(const Tensor& x,const Tensor& mask){
    if(x.dim()!=4||x.size(2)%stride||x.size(3)%stride||x.size(0)*x.size(2)*x.size(3)>16777216)throw InvalidArgument("SimMIM image violates encoder grid/budget");
    Tensor f;if(swin)f=sw->forward_masked_grid(x,mask,token);
    else f=token_grid(vit->forward_masked_features(x,mask,token),{x.size(2)/16,x.size(3)/16},1);
    return torch::pixel_shuffle(decoder(f),stride);
  }
  bool swin;I stride=0;SwinTiny sw{nullptr};ViT vit{nullptr};torch::nn::Conv2d decoder{nullptr};Tensor token;
};
class ReId:public torch::nn::Module{
 public:
  ReId(I classes,I d){if(classes<2||classes>10000||d<32||d>4096)throw InvalidArgument("Invalid ReID head dimensions");
    ResNetOptions o;o.num_classes=0;encoder=register_module("encoder",resnet50(o));projection=register_module("projection",torch::nn::Linear(2048,d));
    neck=register_module("neck",torch::nn::BatchNorm1d(d));neck->bias.set_requires_grad(false);classifier=register_module("classifier",torch::nn::Linear(torch::nn::LinearOptions(d,classes).bias(false)));}
  c10::IValue forward(const Tensor& x){auto embedding=projection(encoder->forward(x));return c10::ivalue::Tuple::create({classifier(neck(embedding)),embedding});}
  ResNet encoder{nullptr};torch::nn::Linear projection{nullptr},classifier{nullptr};torch::nn::BatchNorm1d neck{nullptr};
};
}
nn::ModelAdapter create_dense_field(DenseFieldOptions o){auto model=std::make_shared<Field>(o);auto output=ValueSchema::tensor({});
  if(o.kind!=DenseFieldKind::depth_fpn&&o.auxiliary)output=ValueSchema::tuple({output,output});
  return nn::adapt_module(model,[](Field& m,const nn::Invocation& x){return m.forward(x.positional[0].toTensor());},
      {.positional={ValueSchema::tensor({})},.keyword={},.allow_extra_keywords=false},output);}
nn::ModelAdapter create_simmim(SimMimEncoder kind,bool recomputation){auto model=std::make_shared<SimMim>(kind,recomputation);
  return nn::adapt_module(model,[](SimMim& m,const nn::Invocation& x){return m.forward(x.positional[0].toTensor(),x.positional[1].toTensor());},
      {.positional={ValueSchema::tensor({}),ValueSchema::tensor({})},.keyword={},.allow_extra_keywords=false},ValueSchema::tensor({}));}
nn::ModelAdapter create_reid_resnet50(I identities,I d){return nn::adapt_module(std::make_shared<ReId>(identities,d),
    [](ReId& m,const nn::Invocation& x){return m.forward(x.positional[0].toTensor());},
    {.positional={ValueSchema::tensor({})},.keyword={},.allow_extra_keywords=false},ValueSchema::tuple({ValueSchema::tensor({}),ValueSchema::tensor({})}));}
}
