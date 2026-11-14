// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/models/extended/pyramid.hpp>
#include <jyppx/liuding/vision/models/extended/vit_adapter.hpp>

#include <torch/torch.h>

#include <algorithm>
namespace jyppx::liuding::models::extended {
namespace {
using I=std::int64_t;
Tensor Resize(const Tensor& x,const Tensor& target){return torch::nn::functional::interpolate(x,
  torch::nn::functional::InterpolateFuncOptions().size(std::vector<I>{target.size(2),target.size(3)}).mode(torch::kNearest));}
std::vector<Tensor> Raw(const c10::IValue& x){std::vector<Tensor> y;if(x.isTensor())return{x.toTensor()};
  for(const auto& t:x.toTupleRef().elements())y.push_back(t.toTensor());return y;}
class Fusion:public torch::nn::Module{
 public:
  bool softmax_;
  Fusion(I inputs,I channels,bool softmax):softmax_(softmax){w=register_parameter("weights",torch::ones({inputs}));
    depth=register_module("depthwise",torch::nn::Conv2d(torch::nn::Conv2dOptions(channels,channels,3).padding(1).groups(channels).bias(false)));
    point=register_module("pointwise",torch::nn::Conv2d(torch::nn::Conv2dOptions(channels,channels,1).bias(true)));
    bn=register_module("bn",torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels).eps(.001).momentum(.01)));}
  Tensor forward(const std::vector<Tensor>& x){auto weight=softmax_?w.softmax(0):torch::relu(w);if(!softmax_)weight=weight/(weight.sum()+1e-4);
    auto y=x[0]*weight[0];for(I i=1;i<static_cast<I>(x.size());++i)y=y+Resize(x[i],x[0])*weight[i];
    return bn(point(depth(torch::silu(y))));}
  Tensor w;torch::nn::Conv2d depth{nullptr},point{nullptr};torch::nn::BatchNorm2d bn{nullptr};
};
}
Tensor token_grid(const Tensor& tokens,std::array<I,2> grid,I prefix){
  if(tokens.dim()!=3||grid[0]<1||grid[1]<1||prefix<0||grid[0]>1048576/grid[1]||tokens.size(1)!=prefix+grid[0]*grid[1])
    throw InvalidArgument("Token grid requires explicit matching H/W and prefix count");
  return tokens.slice(1,prefix).transpose(1,2).reshape({tokens.size(0),tokens.size(2),grid[0],grid[1]});
}
struct FeaturePyramidImpl::Impl{
  PyramidOptions o;nn::ModelAdapter backbone;FeatureSpec spec;
  std::size_t token_index = 0;
  torch::nn::ModuleList lateral{nullptr},smooth{nullptr},fusion{nullptr};
  torch::nn::Conv2d p6{nullptr},p7{nullptr};
  Impl(FeaturePyramidImpl& owner,PyramidOptions options):o(std::move(options)),backbone([&]{
    if(o.backbone=="vit_adapter_tiny")return adapt(ViTAdapter());
    ModelOptions m;m.swin.dynamic_image_size=m.vit.dynamic_image_size=true;
    return create_feature_extractor(o.backbone,m);}()){
    if(o.channels<32||o.channels>1024||o.channels%32||o.repeats<1||o.repeats>8)throw InvalidArgument("Invalid pyramid width/repeats");
    ModelOptions m;m.swin.dynamic_image_size=m.vit.dynamic_image_size=true;if(o.backbone=="vit_adapter_tiny"){for(I i=0;i<4;++i)spec.outputs.push_back({"c"+std::to_string(i+1),FeatureLayout::nchw,192,4LL<<i,0});}
    else spec=model_feature_spec(o.backbone,m);
    owner.register_module("backbone",backbone.state().shared_module());
    lateral=owner.register_module("lateral",torch::nn::ModuleList());smooth=owner.register_module("smooth",torch::nn::ModuleList());
    // Registered token encoders expose global, patch-token and grid views.
    // Select the declared token view instead of assuming a one-output factory.
    auto token_view = std::find_if(
        spec.outputs.begin(), spec.outputs.end(),
        [](const auto& f) { return f.layout == FeatureLayout::ntc; });
    const bool token = token_view != spec.outputs.end();
    if (token)
      token_index = static_cast<std::size_t>(token_view - spec.outputs.begin());
    if(token!=(o.kind==PyramidKind::token_pyramid))throw InvalidArgument("Token backbones require the explicit token_pyramid profile");
    if(!token&&spec.outputs.size()!=4)throw InvalidArgument("Pyramid requires four declared spatial stages");
    for (I i = 0; i < 3; ++i) {
      auto c = token ? spec.outputs[token_index].channels
                     : spec.outputs[i + 1].channels;
      if(!token&&(spec.outputs[i+1].layout!=FeatureLayout::nchw||spec.outputs[i+1].reduction!=(8LL<<i)))throw InvalidArgument("Pyramid stage layout/stride mismatch");
      lateral->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c,o.channels,1)));
      smooth->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(o.channels, o.channels, 3).padding(1)));
    }
    auto last=(token||o.extra_from_p5)?o.channels:spec.outputs.back().channels;
    p6=owner.register_module("p6",torch::nn::Conv2d(torch::nn::Conv2dOptions(last,o.channels,3).stride(2).padding(1)));
    p7=owner.register_module("p7",torch::nn::Conv2d(torch::nn::Conv2dOptions(o.channels,o.channels,3).stride(2).padding(1)));
    fusion=owner.register_module("fusion",torch::nn::ModuleList());
    if(o.kind==PyramidKind::bifpn)for(I r=0;r<o.repeats;++r)for(I i=0;i<8;++i)
      fusion->push_back(std::make_shared<Fusion>(i>=4&&i<7?3:2,o.channels,o.softmax_fusion));
  }
};
FeaturePyramidImpl::FeaturePyramidImpl(PyramidOptions o):impl_(std::make_unique<Impl>(*this,std::move(o))){}
FeaturePyramidImpl::~FeaturePyramidImpl()=default;
const PyramidOptions& FeaturePyramidImpl::options()const noexcept{return impl_->o;}
std::vector<Tensor> FeaturePyramidImpl::forward(const Tensor& images){
  auto& p=*impl_;if(images.dim()!=4||images.size(2)<32||images.size(3)<32)throw InvalidArgument("Pyramid expects NCHW images at least 32 pixels");
  auto raw=Raw(p.backbone.invoke({.positional={images}}).value);std::vector<Tensor> c;
  if (p.o.kind == PyramidKind::token_pyramid) {
    auto f = p.spec.outputs[p.token_index];
    auto x =
        token_grid(raw[p.token_index],
                   {images.size(2) / f.reduction, images.size(3) / f.reduction},
                   f.prefix_tokens);
    for(I i=0;i<3;++i)c.push_back(torch::nn::functional::interpolate(x,torch::nn::functional::InterpolateFuncOptions().size(std::vector<I>{(images.size(2)+(8LL<<i)-1)/(8LL<<i),(images.size(3)+(8LL<<i)-1)/(8LL<<i)}).mode(torch::kBilinear).align_corners(false)));
  } else
    c = {raw[1], raw[2], raw[3]};
  std::vector<Tensor> y(5);for(I i=0;i<3;++i)y[i]=p.lateral[i]->as<torch::nn::Conv2d>()->forward(c[i]);
  if(p.o.kind!=PyramidKind::bifpn){for(I i=1;i>=0;--i)y[i]=y[i]+Resize(y[i+1],y[i]);for(I i=0;i<3;++i)y[i]=p.smooth[i]->as<torch::nn::Conv2d>()->forward(y[i]);}
  y[3]=p.p6(p.o.kind==PyramidKind::token_pyramid||p.o.extra_from_p5?y[2]:c[2]);y[4]=p.p7(torch::relu(y[3]));
  if(p.o.kind==PyramidKind::bifpn)for(I r=0;r<p.o.repeats;++r){auto original=y;auto top=y;
    for(I i=3;i>=0;--i)top[i]=p.fusion[r*8+3-i]->as<Fusion>()->forward({y[i],top[i+1]});y[0]=top[0];
    for(I i=1;i<5;++i){auto down=torch::max_pool2d(y[i-1],{3,3},{2,2},{1,1});
      y[i]=p.fusion[r*8+3+i]->as<Fusion>()->forward(i==4?std::vector<Tensor>{original[i],down}:std::vector<Tensor>{original[i],top[i],down});}}
  return y;
}
}
