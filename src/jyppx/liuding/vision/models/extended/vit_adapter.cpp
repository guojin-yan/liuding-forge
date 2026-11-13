// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/models/extended/vit_adapter.hpp>
#include "tensor_literals.hpp"
#include <jyppx/liuding/vision/ops/deformable.hpp>
#include <torch/torch.h>
#include <numbers>
namespace jyppx::liuding::models::extended {
namespace {
using I=std::int64_t;
struct ConvBlock:torch::nn::Module{torch::nn::Sequential layers{nullptr};ConvBlock(I in,I out,I stride){layers=register_module("layers",torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(in,out,3).padding(1).stride(stride).bias(false)),torch::nn::BatchNorm2d(out),torch::nn::ReLU()));}Tensor forward(const Tensor& x){return layers->forward(x);}};
std::shared_ptr<ConvBlock> Cbr(I in,I out,I stride){return std::make_shared<ConvBlock>(in,out,stride);}
Tensor Tokens(const Tensor& x){return x.flatten(2).transpose(1,2);}
Tensor Grid(const Tensor& x,I h,I w){return x.transpose(1,2).reshape({x.size(0),x.size(2),h,w});}
Tensor References(std::span<const std::array<I,2>> shapes,const Tensor& ref){std::vector<Tensor> all;for(const auto& hw:shapes){auto h=hw[0],w=hw[1];auto x=(torch::arange(w,ref.options())+.5)/w,y=(torch::arange(h,ref.options())+.5)/h;all.push_back(torch::stack({x.view({1,w}).expand({h,w}),y.view({h,1}).expand({h,w})},2).reshape({-1,2}));}return torch::cat(all).view({1,-1,1,1,1,2});}
struct Deform:torch::nn::Module {
  I d,levels;torch::nn::LayerNorm qn{nullptr},vn{nullptr};torch::nn::Linear offset{nullptr},attention{nullptr},value{nullptr},output{nullptr};
  Deform(I dimension,I count):d(dimension),levels(count){qn=register_module("query_norm",torch::nn::LayerNorm(torch::nn::LayerNormOptions({d}).eps(1e-6)));vn=register_module("feat_norm",torch::nn::LayerNorm(torch::nn::LayerNormOptions({d}).eps(1e-6)));offset=register_module("sampling_offsets",torch::nn::Linear(d,6*levels*4*2));attention=register_module("attention_weights",torch::nn::Linear(d,6*levels*4));value=register_module("value_proj",torch::nn::Linear(d,d));output=register_module("output_proj",torch::nn::Linear(d,d));
    torch::NoGradGuard guard;offset->weight.zero_();attention->weight.zero_();attention->bias.zero_();std::vector<float> radial;for(I h=0;h<6;++h){double angle=2*std::numbers::pi*h/6,x=std::cos(angle),y=std::sin(angle),scale=(std::max)(std::abs(x),std::abs(y));for(I l=0;l<levels;++l)for(I p=0;p<4;++p){radial.push_back(x/scale*(p+1));radial.push_back(y/scale*(p+1));}}offset->bias.copy_(torch::tensor(radial));}
  Tensor forward(const Tensor& query,const Tensor& memory,std::span<const std::array<I,2>> shapes,const Tensor& reference){auto q=qn(query);I b=q.size(0),n=q.size(1);auto offsets=offset(q).view({b,n,6,levels,4,2});std::vector<double> sizes;for(auto s:shapes){sizes.push_back(s[1]);sizes.push_back(s[0]);}auto divisor=detail::tensor_literals(sizes,q.options()).view({1,1,1,levels,1,2});auto locations=reference+offsets/divisor;auto weights=attention(q).view({b,n,6,levels*4}).softmax(-1).view({b,n,6,levels,4});return output(vision_ops::multiscale_deformable_attention(value(vn(memory)).view({b,memory.size(1),6,d/6}),shapes,locations,weights));}
};
struct Extractor:torch::nn::Module {
  std::shared_ptr<Deform> deform;torch::nn::LayerNorm norm{nullptr};torch::nn::Linear fc1{nullptr},fc2{nullptr};torch::nn::Conv2d depthwise{nullptr};
  explicit Extractor(I d){deform=register_module("attention",std::make_shared<Deform>(d,1));norm=register_module("ffn_norm",torch::nn::LayerNorm(torch::nn::LayerNormOptions({d}).eps(1e-6)));I hidden=d/4;fc1=register_module("fc1",torch::nn::Linear(d,hidden));fc2=register_module("fc2",torch::nn::Linear(hidden,d));depthwise=register_module("dwconv",torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden,hidden,3).padding(1).groups(hidden)));}
  Tensor forward(const Tensor& c,const Tensor& x,std::span<const std::array<I,2>> grid,std::span<const std::array<I,2>> spatial,const Tensor& reference){auto y=c+deform->forward(c,x,grid,reference);auto ff=fc1(norm(y));std::vector<Tensor> levels;I begin=0;for(auto s:spatial){I n=s[0]*s[1];levels.push_back(Tokens(depthwise(Grid(ff.slice(1,begin,begin+n),s[0],s[1]))));begin+=n;}return y+fc2(torch::gelu(torch::cat(levels,1)));}
};
}
struct ViTAdapterImpl::Impl:torch::nn::Module {
  ViTAdapterOptions o;I dimension;ViT vit{nullptr};torch::nn::Sequential stem{nullptr};std::array<std::shared_ptr<ConvBlock>,3> spatial;std::array<torch::nn::Conv2d,4> projections{nullptr,nullptr,nullptr,nullptr};std::array<torch::nn::BatchNorm2d,4> norms{nullptr,nullptr,nullptr,nullptr};torch::nn::ConvTranspose2d up{nullptr};Tensor level;
  std::array<std::shared_ptr<Deform>,4> inject;std::array<Tensor,4> gamma;std::array<std::shared_ptr<Extractor>,4> extract;std::array<std::shared_ptr<Extractor>,2> extra;
  explicit Impl(ViTAdapterOptions options):o(options){if(o.max_patch_tokens<1||o.max_patch_tokens>4096)throw InvalidArgument("Invalid ViT-Adapter token limit");dimension=o.variant==ViTVariant::tiny?192:o.variant==ViTVariant::small_capacity?384:768;ViTOptions v;v.num_classes=0;v.dynamic_image_size=true;v.activation_recomputation=o.activation_recomputation;vit=register_module("vit",ViT(o.variant,v));
    stem=register_module("spatial_stem",torch::nn::Sequential(Cbr(3,64,2),Cbr(64,64,1),Cbr(64,64,1),torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1))));std::array<I,4> channels{64,128,256,256};
    for(I i=0;i<3;++i)spatial[i]=register_module("spatial"+std::to_string(i),Cbr(channels[i],channels[i+1],2));for(I i=0;i<4;++i){projections[i]=register_module("projection"+std::to_string(i),torch::nn::Conv2d(torch::nn::Conv2dOptions(channels[i],dimension,1)));norms[i]=register_module("norm"+std::to_string(i),torch::nn::BatchNorm2d(dimension));inject[i]=register_module("injector"+std::to_string(i),std::make_shared<Deform>(dimension,3));extract[i]=register_module("extractor"+std::to_string(i),std::make_shared<Extractor>(dimension));gamma[i]=register_parameter("gamma"+std::to_string(i),torch::zeros({dimension}));}
    for(I i=0;i<2;++i)extra[i]=register_module("extra_extractor"+std::to_string(i),std::make_shared<Extractor>(dimension));up=register_module("up",torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(dimension,dimension,2).stride(2)));level=register_parameter("level_embed",torch::randn({3,dimension}));}
  std::vector<Tensor> forward(const Tensor& images){if(images.dim()!=4||images.size(0)<1||images.size(0)>32||images.size(1)!=3||images.size(2)<32||images.size(3)<32||images.size(2)%32||images.size(3)%32)throw InvalidArgument("ViT-Adapter requires NCHW images divisible by 32");I h=images.size(2)/16,w=images.size(3)/16;if(h*w>o.max_patch_tokens||images.size(0)*h*w*h*w*(dimension/64)>67108864)throw ResourceExhausted("ViT-Adapter attention token budget exceeded");
    std::vector<Tensor> maps;auto c=stem->forward(images);maps.push_back(projections[0](c));for(I i=0;i<3;++i){c=spatial[i]->forward(c);maps.push_back(projections[i+1](c));}std::vector<std::array<I,2>> shapes{{h*2,w*2},{h,w},{h/2,w/2}},grid{{h,w}};auto spatial_tokens=torch::cat({Tokens(maps[1])+level[0],Tokens(maps[2])+level[1],Tokens(maps[3])+level[2]},1);auto x=vit->patch_tokens(images);auto patch_ref=References(grid,x),spatial_ref=References(shapes,x);
    for(I i=0;i<4;++i){x=x+gamma[i]*inject[i]->forward(x,spatial_tokens,shapes,patch_ref);x=vit->forward_block_range(x,i*3,(i+1)*3);spatial_tokens=extract[i]->forward(spatial_tokens,x,grid,shapes,spatial_ref);}for(auto& e:extra)spatial_tokens=e->forward(spatial_tokens,x,grid,shapes,spatial_ref);
    I begin=0;for(I i=0;i<3;++i){auto s=shapes[i];maps[i+1]=Grid(spatial_tokens.slice(1,begin,begin+s[0]*s[1]),s[0],s[1]);begin+=s[0]*s[1];}maps[0]=maps[0]+up(maps[1]);
    auto vit_grid=Grid(x,h,w);for(I i=0;i<4;++i){if(o.add_vit_features)maps[i]=maps[i]+torch::nn::functional::interpolate(vit_grid,torch::nn::functional::InterpolateFuncOptions().size(std::vector<I>{maps[i].size(2),maps[i].size(3)}).mode(torch::kBilinear).align_corners(false));maps[i]=norms[i](maps[i]);}return maps;}
};
ViTAdapterImpl::ViTAdapterImpl(ViTAdapterOptions o){impl_=register_module("adapter",std::make_shared<Impl>(o));}
ViTAdapterImpl::~ViTAdapterImpl()=default;
std::vector<Tensor> ViTAdapterImpl::forward(const Tensor& images){return impl_->forward(images);}
nn::ModelAdapter adapt(ViTAdapter model){return nn::adapt_holder(model,[](ViTAdapterImpl& m,const nn::Invocation& x){auto out=m.forward(x.positional[0].toTensor());return c10::ivalue::Tuple::create({out[0],out[1],out[2],out[3]});},{.positional={ValueSchema::tensor({})},.keyword={},.allow_extra_keywords=false},ValueSchema::tuple({ValueSchema::tensor({}),ValueSchema::tensor({}),ValueSchema::tensor({}),ValueSchema::tensor({})}));}
}
