// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/models/extended/rtdetr.hpp>
#include <jyppx/liuding/vision/ops/deformable.hpp>
#include <torch/torch.h>
#include <cmath>
#include <numbers>
namespace jyppx::liuding::models::extended {
namespace {
using I=std::int64_t;
Tensor Inverse(const Tensor& x){auto a=x.clamp(1e-5,1-1e-5);return torch::log(a/(1-a));}
struct FrozenNorm:torch::nn::Module {
  Tensor weight,bias,mean,var;
  explicit FrozenNorm(I c){weight=register_buffer("weight",torch::ones({c}));bias=register_buffer("bias",torch::zeros({c}));mean=register_buffer("running_mean",torch::zeros({c}));var=register_buffer("running_var",torch::ones({c}));}
  Tensor forward(const Tensor& x){auto scale=weight*(var+1e-5).rsqrt();return x*scale.view({1,-1,1,1})+(bias-mean*scale).view({1,-1,1,1});}
};
struct Conv:torch::nn::Module {
  torch::nn::Conv2d conv{nullptr};torch::nn::BatchNorm2d bn{nullptr};std::shared_ptr<FrozenNorm> frozen;int activation;
  Conv(I in,I out,I kernel=1,I stride=1,int act=0,bool freeze=false):activation(act){conv=register_module("conv",torch::nn::Conv2d(torch::nn::Conv2dOptions(in,out,kernel).stride(stride).padding(kernel/2).bias(false)));
    if(freeze)frozen=register_module("norm",std::make_shared<FrozenNorm>(out));else bn=register_module("norm",torch::nn::BatchNorm2d(out));}
  Tensor forward(const Tensor& x){auto y=conv(x);y=frozen?frozen->forward(y):bn(y);return activation==1?torch::relu(y):activation==2?torch::silu(y):y;}
};
struct VdBlock:torch::nn::Module {
  std::shared_ptr<Conv> a,b,c,shortcut;bool pool=false;
  VdBlock(I in,I width,I stride,bool bottleneck,bool first){I out=width*(bottleneck?4:1);
    a=register_module("branch2a",std::make_shared<Conv>(in,width,bottleneck?1:3,bottleneck?1:stride,1,true));
    b=register_module("branch2b",std::make_shared<Conv>(width,width,3,bottleneck?stride:1,bottleneck?1:0,true));
    if(bottleneck)c=register_module("branch2c",std::make_shared<Conv>(width,out,1,1,0,true));
    if(first){pool=stride==2;shortcut=register_module("short",std::make_shared<Conv>(in,out,1,1,0,true));}}
  Tensor forward(const Tensor& x){auto y=b->forward(a->forward(x));if(c)y=c->forward(y);auto skip=x;if(pool)skip=torch::nn::functional::avg_pool2d(skip,torch::nn::functional::AvgPool2dFuncOptions(2).stride(2).ceil_mode(true));if(shortcut)skip=shortcut->forward(skip);return torch::relu(y+skip);}
};
struct VdBackbone:torch::nn::Module {
  torch::nn::Sequential stem{nullptr};std::array<torch::nn::Sequential,4> stages;
  explicit VdBackbone(bool large){stem=register_module("stem",torch::nn::Sequential(std::make_shared<Conv>(3,32,3,2,1,true),std::make_shared<Conv>(32,32,3,1,1,true),std::make_shared<Conv>(32,64,3,1,1,true)));
    std::array<I,4> counts=large?std::array<I,4>{3,4,6,3}:std::array<I,4>{2,2,2,2};I in=64;
    for(I l=0;l<4;++l){stages[l]=register_module("stage"+std::to_string(l),torch::nn::Sequential());I width=64LL<<l;for(I n=0;n<counts[l];++n){stages[l]->push_back(std::make_shared<VdBlock>(in,width,n==0&&l>0?2:1,large,n==0));in=width*(large?4:1);}}}
  std::vector<Tensor> forward(const Tensor& images){auto x=stem->forward(images);x=torch::max_pool2d(x,3,2,1);std::vector<Tensor> out;for(I i=0;i<4;++i){x=stages[i]->forward(x);if(i)out.push_back(x);}return out;}
};
struct Attention:torch::nn::Module {
  torch::nn::Linear q{nullptr},k{nullptr},v{nullptr},out{nullptr};I dimension,heads;
  Attention(I d,I h):dimension(d),heads(h){q=register_module("q",torch::nn::Linear(d,d));k=register_module("k",torch::nn::Linear(d,d));v=register_module("v",torch::nn::Linear(d,d));out=register_module("out",torch::nn::Linear(d,d));}
  Tensor forward(const Tensor& query,const Tensor& key,const Tensor& value,const Tensor& mask={}){auto split=[&](const Tensor& t){return t.view({t.size(0),t.size(1),heads,dimension/heads}).transpose(1,2);};
    auto score=split(q(query)).matmul(split(k(key)).transpose(-1,-2))/std::sqrt(static_cast<double>(dimension/heads));if(mask.defined())score=score.masked_fill(mask.unsqueeze(0).unsqueeze(0),-1e4);
    return out(score.softmax(-1).matmul(split(v(value))).transpose(1,2).contiguous().view({query.size(0),query.size(1),dimension}));}
};
struct Encoder:torch::nn::Module {
  std::shared_ptr<Attention> attention;torch::nn::Linear f1{nullptr},f2{nullptr};torch::nn::LayerNorm n1{nullptr},n2{nullptr};
  Encoder(){attention=register_module("attention",std::make_shared<Attention>(256,8));f1=register_module("linear1",torch::nn::Linear(256,1024));f2=register_module("linear2",torch::nn::Linear(1024,256));n1=register_module("norm1",torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));n2=register_module("norm2",torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));}
  Tensor forward(const Tensor& x,const Tensor& position){auto y=n1(x+attention->forward(x+position,x+position,x));return n2(y+f2(torch::gelu(f1(y))));}
};
struct Rep:torch::nn::Module {std::shared_ptr<Conv> a,b;Rep(){a=register_module("conv3",std::make_shared<Conv>(256,256,3));b=register_module("conv1",std::make_shared<Conv>(256,256,1));}Tensor forward(const Tensor& x){return torch::silu(a->forward(x)+b->forward(x));}};
struct Csp:torch::nn::Module {std::shared_ptr<Conv> a,b;torch::nn::Sequential blocks{nullptr};Csp(I count){a=register_module("conv1",std::make_shared<Conv>(512,256,1,1,2));b=register_module("conv2",std::make_shared<Conv>(512,256,1,1,2));blocks=register_module("blocks",torch::nn::Sequential());for(I i=0;i<count;++i)blocks->push_back(std::make_shared<Rep>());}Tensor forward(const Tensor& x){return blocks->forward(a->forward(x))+b->forward(x);}};
Tensor Position(I h,I w,const Tensor& reference){auto opts=reference.options();auto omega=1./torch::pow(10000.,torch::arange(64,opts)/64.);auto x=torch::arange(w,opts).view({1,w}).expand({h,w}).reshape({-1,1})*omega;
  auto y=torch::arange(h,opts).view({h,1}).expand({h,w}).reshape({-1,1})*omega;return torch::cat({x.sin(),x.cos(),y.sin(),y.cos()},1).unsqueeze(0);}
struct Hybrid:torch::nn::Module {
  std::array<std::shared_ptr<Conv>,3> project;std::array<std::shared_ptr<Conv>,2> lateral,down;std::array<std::shared_ptr<Csp>,2> fpn,pan;std::shared_ptr<Encoder> encoder;
  Hybrid(bool large){for(I l=0;l<3;++l)project[l]=register_module("input"+std::to_string(l),std::make_shared<Conv>((128LL<<l)*(large?4:1),256));encoder=register_module("encoder",std::make_shared<Encoder>());
    for(I l=0;l<2;++l){lateral[l]=register_module("lateral"+std::to_string(l),std::make_shared<Conv>(256,256,1,1,2));down[l]=register_module("down"+std::to_string(l),std::make_shared<Conv>(256,256,3,2,2));fpn[l]=register_module("fpn"+std::to_string(l),std::make_shared<Csp>(large?3:1));pan[l]=register_module("pan"+std::to_string(l),std::make_shared<Csp>(large?3:1));}}
  std::vector<Tensor> forward(const std::vector<Tensor>& x){std::vector<Tensor> y;for(I i=0;i<3;++i)y.push_back(project[i]->forward(x[i]));auto h=y[2].size(2),w=y[2].size(3);auto tokens=y[2].flatten(2).transpose(1,2);y[2]=encoder->forward(tokens,Position(h,w,tokens)).transpose(1,2).reshape_as(y[2]);
    auto resize=[](const Tensor& a,const Tensor& b){return torch::nn::functional::interpolate(a,torch::nn::functional::InterpolateFuncOptions().size(std::vector<I>{b.size(2),b.size(3)}).mode(torch::kNearest));};
    for(I k=0;k<2;++k){I l=2-k;y[l]=lateral[k]->forward(y[l]);y[l-1]=fpn[k]->forward(torch::cat({resize(y[l],y[l-1]),y[l-1]},1));}
    for(I l=0;l<2;++l)y[l+1]=pan[l]->forward(torch::cat({down[l]->forward(y[l]),y[l+1]},1));return y;}
};
struct Mlp:torch::nn::Module {torch::nn::Sequential layers{nullptr};torch::nn::Linear last{nullptr};Mlp(I in,I hidden,I out,I count){layers=register_module("layers",torch::nn::Sequential());for(I i=0;i<count-1;++i){layers->push_back(torch::nn::Linear(i?hidden:in,hidden));layers->push_back(torch::nn::ReLU());}last=register_module("last",torch::nn::Linear(count>1?hidden:in,out));}Tensor forward(const Tensor& x){return last(layers->forward(x));}};
struct Deform:torch::nn::Module {
  torch::nn::Linear offsets{nullptr},weights{nullptr},value{nullptr},output{nullptr};
  Deform(){offsets=register_module("sampling_offsets",torch::nn::Linear(256,8*3*4*2));weights=register_module("attention_weights",torch::nn::Linear(256,8*3*4));value=register_module("value_proj",torch::nn::Linear(256,256));output=register_module("output_proj",torch::nn::Linear(256,256));
    torch::NoGradGuard guard;offsets->weight.zero_();weights->weight.zero_();weights->bias.zero_();std::vector<float> radial;for(I h=0;h<8;++h){double theta=2*std::numbers::pi*h/8,dx=std::cos(theta),dy=std::sin(theta),scale=(std::max)(std::abs(dx),std::abs(dy));for(I l=0;l<3;++l)for(I p=0;p<4;++p){radial.push_back(static_cast<float>(dx/scale*(p+1)));radial.push_back(static_cast<float>(dy/scale*(p+1)));}}offsets->bias.copy_(torch::tensor(radial));}
  Tensor forward(const Tensor& q,const Tensor& references,const Tensor& memory,std::span<const std::array<I,2>> shapes){auto b=q.size(0),n=q.size(1);auto off=offsets(q).view({b,n,8,3,4,2});auto weight=weights(q).view({b,n,8,12}).softmax(-1).view({b,n,8,3,4});
    auto xy=references.slice(-1,0,2).view({b,n,1,1,1,2}),wh=references.slice(-1,2,4).view({b,n,1,1,1,2});auto locations=xy+off/4*wh*.5;
    return output(vision_ops::multiscale_deformable_attention(value(memory).view({b,memory.size(1),8,32}),shapes,locations,weight));}
};
struct Decoder:torch::nn::Module {
  std::shared_ptr<Attention> self;std::shared_ptr<Deform> cross;std::array<torch::nn::LayerNorm,3> norms{nullptr,nullptr,nullptr};torch::nn::Linear f1{nullptr},f2{nullptr};
  Decoder(){self=register_module("self_attention",std::make_shared<Attention>(256,8));cross=register_module("cross_attention",std::make_shared<Deform>());for(I i=0;i<3;++i)norms[i]=register_module("norm"+std::to_string(i),torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));f1=register_module("linear1",torch::nn::Linear(256,1024));f2=register_module("linear2",torch::nn::Linear(1024,256));}
  Tensor forward(const Tensor& x,const Tensor& position,const Tensor& ref,const Tensor& memory,std::span<const std::array<I,2>> shapes,const Tensor& mask){auto q=x+position;auto y=norms[0](x+self->forward(q,q,x,mask));y=norms[1](y+cross->forward(y+position,ref,memory,shapes));return norms[2]((y+f2(torch::relu(f1(y)))).clamp(-65504,65504));}
};
}
struct RtDetrImpl::Impl:torch::nn::Module {
  RtDetrOptions o;std::shared_ptr<VdBackbone> backbone;std::shared_ptr<Hybrid> hybrid;std::array<std::shared_ptr<Conv>,3> project;
  torch::nn::Linear enc_output{nullptr},enc_class{nullptr};torch::nn::LayerNorm enc_norm{nullptr};std::shared_ptr<Mlp> enc_box,query_position;
  torch::nn::Embedding class_embed{nullptr};std::vector<std::shared_ptr<Decoder>> decoder;std::vector<torch::nn::Linear> classes;std::vector<std::shared_ptr<Mlp>> boxes;
  explicit Impl(RtDetrOptions options):o(options){if(o.num_classes<1||o.num_classes>10000||o.queries<1||o.queries>1024||o.decoder_layers<1||o.decoder_layers>6||o.denoising_queries<0||o.denoising_queries>1000||o.max_memory_tokens<1||o.max_memory_tokens>32768||!std::isfinite(o.label_noise)||o.label_noise<0||o.label_noise>1||!std::isfinite(o.box_noise)||o.box_noise<0||o.box_noise>2)throw InvalidArgument("Invalid RT-DETR options");
    bool large=o.backbone==RtDetrBackbone::resnet50_vd;backbone=register_module("backbone",std::make_shared<VdBackbone>(large));hybrid=register_module("encoder",std::make_shared<Hybrid>(large));for(I i=0;i<3;++i)project[i]=register_module("decoder_input"+std::to_string(i),std::make_shared<Conv>(256,256));
    enc_output=register_module("enc_output",torch::nn::Linear(256,256));enc_norm=register_module("enc_norm",torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));enc_class=register_module("enc_class",torch::nn::Linear(256,o.num_classes));enc_box=register_module("enc_box",std::make_shared<Mlp>(256,256,4,3));query_position=register_module("query_position",std::make_shared<Mlp>(4,512,256,2));class_embed=register_module("denoising_class_embed",torch::nn::Embedding(torch::nn::EmbeddingOptions(o.num_classes+1,256).padding_idx(o.num_classes)));
    for(I l=0;l<o.decoder_layers;++l){decoder.push_back(register_module("decoder"+std::to_string(l),std::make_shared<Decoder>()));classes.push_back(register_module("score"+std::to_string(l),torch::nn::Linear(256,o.num_classes)));boxes.push_back(register_module("box"+std::to_string(l),std::make_shared<Mlp>(256,256,4,3)));}
    torch::NoGradGuard guard;enc_class->bias.fill_(-std::log(99.));enc_box->last->weight.zero_();enc_box->last->bias.zero_();for(I l=0;l<o.decoder_layers;++l){classes[l]->bias.fill_(-std::log(99.));boxes[l]->last->weight.zero_();boxes[l]->last->bias.zero_();}}
  std::vector<Tensor> forward(const Tensor& images,const Tensor& targets){if(images.dim()!=4||images.size(0)<1||images.size(0)>32||images.size(1)!=3||images.size(2)%32||images.size(3)%32||images.size(2)<32||images.size(3)<32||!images.is_floating_point()||targets.dim()!=2||targets.size(1)!=6||targets.size(0)>4096||targets.device()!=images.device()||!torch::isfinite(targets).all().item<bool>())throw InvalidArgument("Invalid RT-DETR images/targets");
    if(targets.numel()&&(!(targets.slice(1,0,2)==targets.slice(1,0,2).round()).all().item<bool>()||targets.select(1,0).min().item<double>()<0||targets.select(1,0).max().item<double>()>=images.size(0)||targets.select(1,1).min().item<double>()<0||targets.select(1,1).max().item<double>()>=o.num_classes||!(targets.slice(1,4,6)>targets.slice(1,2,4)).all().item<bool>()))throw InvalidArgument("RT-DETR target IDs or boxes invalid");
    I b=images.size(0);auto features=hybrid->forward(backbone->forward(images));std::vector<Tensor> flatten,anchors;std::vector<std::array<I,2>> shapes;I tokens=0;
    for(I l=0;l<3;++l){auto x=project[l]->forward(features[l]);I h=x.size(2),w=x.size(3);tokens+=h*w;if(tokens>o.max_memory_tokens)throw ResourceExhausted("RT-DETR memory token budget exceeded");shapes.push_back({h,w});flatten.push_back(x.flatten(2).transpose(1,2));auto xs=(torch::arange(w,images.options())+.5)/w,ys=(torch::arange(h,images.options())+.5)/h;auto xy=torch::stack({xs.view({1,w}).expand({h,w}),ys.view({h,1}).expand({h,w})},2).reshape({1,-1,2});anchors.push_back(torch::cat({xy,torch::full_like(xy,.05*(1LL<<l))},2));}
    if(tokens<o.queries)throw InvalidArgument("RT-DETR query count exceeds feature locations");if(b*tokens*256>16777216||b*(shapes.back()[0]*shapes.back()[1])*(shapes.back()[0]*shapes.back()[1])*8>16777216)throw ResourceExhausted("RT-DETR attention allocation budget exceeded");
    auto memory=torch::cat(flatten,1),anchor=torch::cat(anchors,1);auto valid=((anchor>.01)&(anchor<.99)).all(-1,true);auto output=enc_norm(enc_output(memory*valid.to(memory.scalar_type())));auto enc_logits=enc_class(output);auto unact=enc_box->forward(output)+Inverse(anchor).masked_fill(~valid,std::numeric_limits<float>::infinity());
    auto top=std::get<1>(std::get<0>(enc_logits.max(-1)).topk(o.queries,1));auto gather=[&](const Tensor& x){return x.gather(1,top.unsqueeze(-1).expand({b,o.queries,x.size(2)}));};auto enc_boxes=gather(unact).sigmoid(),enc_scores=gather(enc_logits),reference=Inverse(enc_boxes).detach(),query=gather(output).detach();
    I dn=0,groups=0,max_gt=0;std::vector<Tensor> ground;for(I i=0;i<b;++i){ground.push_back(targets.index({targets.select(1,0)==i}));max_gt=(std::max)(max_gt,ground.back().size(0));}
    auto mapping=torch::empty({b,0},images.options().dtype(torch::kInt64)),positive=torch::empty({b,0},images.options().dtype(torch::kBool));Tensor attention_mask;
    if(is_training()&&o.denoising_queries>0&&max_gt){groups=(std::max)(I{1},o.denoising_queries/max_gt);dn=max_gt*2*groups;if(dn>2048||b*dn*o.num_classes>16777216)throw ResourceExhausted("RT-DETR denoising budget exceeded");
      mapping=torch::full({b,dn},-1,images.options().dtype(torch::kInt64));positive=torch::zeros({b,dn},images.options().dtype(torch::kBool));auto labels=torch::full({b,dn},o.num_classes,mapping.options());auto dn_boxes=torch::zeros({b,dn,4},images.options());
      for(I i=0;i<b;++i){auto gt=ground[i];for(I g=0;g<groups;++g)for(I sign=0;sign<2;++sign){I begin=(2*g+sign)*max_gt;auto range=torch::arange(gt.size(0),mapping.options())+begin;mapping[i].index_copy_(0,range,torch::arange(gt.size(0),mapping.options()));positive[i].index_fill_(0,range,sign==0);labels[i].index_copy_(0,range,gt.select(1,1).to(torch::kInt64));auto xyxy=gt.slice(1,2,6)/torch::tensor({static_cast<double>(images.size(3)),static_cast<double>(images.size(2)),static_cast<double>(images.size(3)),static_cast<double>(images.size(2))},images.options());auto wh=xyxy.slice(1,2,4)-xyxy.slice(1,0,2);auto perturb=(torch::rand_like(xyxy)+(sign?1.:0.))*torch::where(torch::rand_like(xyxy)>.5,1.,-1.)*torch::cat({wh,wh},1)*.5*o.box_noise;auto noisy=(xyxy+perturb).clamp(0,1);auto lo=torch::minimum(noisy.slice(1,0,2),noisy.slice(1,2,4)),hi=torch::maximum(noisy.slice(1,0,2),noisy.slice(1,2,4));dn_boxes[i].index_copy_(0,range,torch::cat({(lo+hi)*.5,(hi-lo).clamp_min(1e-5)},1));}}
      auto noisy_labels=torch::where((torch::rand({b,dn},images.options())<o.label_noise*.5)&(mapping>=0),torch::randint(o.num_classes,{b,dn},mapping.options()),labels);query=torch::cat({class_embed(noisy_labels),query},1);reference=torch::cat({Inverse(dn_boxes),reference},1);
      attention_mask=torch::zeros({dn+o.queries,dn+o.queries},positive.options());attention_mask.slice(0,dn).slice(1,0,dn).fill_(true);for(I g=0;g<groups;++g){I lo=g*max_gt*2,hi=(g+1)*max_gt*2;attention_mask.slice(0,lo,hi).slice(1,0,lo).fill_(true);attention_mask.slice(0,lo,hi).slice(1,hi,dn).fill_(true);}}
    if(b*(dn+o.queries)*(dn+o.queries)*8>67108864)throw ResourceExhausted("RT-DETR decoder attention budget exceeded");
    std::vector<Tensor> all_boxes,all_scores,dn_boxes,dn_scores;auto ref=reference.sigmoid();Tensor previous;
    for(I l=0;l<o.decoder_layers;++l){query=decoder[l]->forward(query,query_position->forward(ref),ref,memory,shapes,attention_mask);auto delta=boxes[l]->forward(query);auto current=(delta+Inverse(ref)).sigmoid();auto result=l&&is_training()?(delta+Inverse(previous)).sigmoid():current;previous=current;ref=is_training()?current.detach():current;
      all_boxes.push_back(result.slice(1,dn));all_scores.push_back(classes[l](query).slice(1,dn));dn_boxes.push_back(result.slice(1,0,dn));dn_scores.push_back(classes[l](query).slice(1,0,dn));}
    return{torch::stack(all_scores),torch::stack(all_boxes),enc_scores,enc_boxes,torch::stack(dn_scores),torch::stack(dn_boxes),mapping,positive};}
};
RtDetrImpl::RtDetrImpl(RtDetrOptions o){impl_=register_module("model",std::make_shared<Impl>(o));}
RtDetrImpl::~RtDetrImpl()=default;
std::vector<Tensor> RtDetrImpl::forward(const Tensor& x,const Tensor& target){return impl_->forward(x,target);}
nn::ModelAdapter adapt(RtDetr model){std::vector<ValueSchema> outputs(8,ValueSchema::tensor({}));return nn::adapt_holder(model,[](RtDetrImpl& m,const nn::Invocation& x){auto raw=m.forward(x.positional[0].toTensor(),x.positional[1].toTensor());std::vector<c10::IValue> v;for(auto& t:raw)v.emplace_back(t);return c10::ivalue::Tuple::create(std::move(v));},{.positional={ValueSchema::tensor({}),ValueSchema::tensor({})},.keyword={},.allow_extra_keywords=false},ValueSchema::tuple(std::move(outputs)));}
}
