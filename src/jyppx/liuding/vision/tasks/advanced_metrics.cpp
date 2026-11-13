// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/advanced_metrics.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <torch/torch.h>
#include <algorithm>
namespace jyppx::liuding::vision::tasks {
namespace {
using I=std::int64_t;
void Check(const Tensor& s,const Tensor& y,const Tensor& k){if(!s.defined()||s.dim()!=2||s.sizes()!=y.sizes()||s.sizes()!=k.sizes()||k.scalar_type()!=torch::kBool||s.numel()>67108864||!torch::isfinite(s).all().item<bool>()||!torch::isfinite(y).all().item<bool>()||s.min().item<double>()<0||s.max().item<double>()>1||!(y.eq(0)|y.eq(1)).all().item<bool>())throw InvalidArgument("Metrics require finite probability scores, binary values and a known mask");}
double F1(double tp,double fp,double fn){return 2*tp+fp+fn?2*tp/(2*tp+fp+fn):0.;}
}
std::map<std::string,double> multilabel_metrics(const Tensor& scores,const Tensor& values,const Tensor& known,const Tensor& thresholds){
  Check(scores,values,known);auto s=scores.to(torch::kCPU,torch::kFloat64),y=values.to(torch::kCPU,torch::kFloat64),k=known.to(torch::kCPU);
  auto t=thresholds.defined()?thresholds.to(torch::kCPU,torch::kFloat64):torch::full({s.size(1)},.5,torch::kFloat64);
  if(t.dim()!=1||t.size(0)!=s.size(1)||!torch::isfinite(t).all().item<bool>()||t.min().item<double>()<0||t.max().item<double>()>1)throw InvalidArgument("Invalid multilabel threshold vector");
  double aps=0,macro=0,tp=0,fp=0,fn=0;I classes=0,supervised=0;
  for(I c=0;c<s.size(1);++c){auto mask=k.select(1,c);auto sc=s.select(1,c).index({mask}),target=y.select(1,c).index({mask});if(!target.numel())continue;++supervised;
    auto pred=sc>=t[c];double a=(pred&target.eq(1)).sum().item<double>(),b=(pred&target.eq(0)).sum().item<double>(),d=((~pred)&target.eq(1)).sum().item<double>();
    tp+=a;fp+=b;fn+=d;macro+=F1(a,b,d);double positives=target.sum().item<double>();if(!positives)continue;
    auto order=torch::argsort(sc,true,0,true);auto sorted=target.index_select(0,order),prob=sc.index_select(0,order);double cumulative=0,previous=0,ap=0;
    for(I i=0;i<sorted.numel();++i){cumulative+=sorted[i].item<double>();if(i+1<sorted.numel()&&prob[i].item<double>()==prob[i+1].item<double>())continue;ap+=(cumulative-previous)/positives*cumulative/(i+1);previous=cumulative;}
    aps+=ap;++classes;}
  return{{"map",classes?aps/classes:0.},{"macro_f1",supervised?macro/supervised:0.},{"micro_f1",F1(tp,fp,fn)},{"ap_classes",static_cast<double>(classes)}};
}
Tensor calibrate_multilabel_thresholds(const Tensor& s,const Tensor& y,const Tensor& k,std::string_view id){Check(s,y,k);if(id.empty()||id.size()>4096)throw InvalidArgument("Threshold calibration requires an explicit validation split identity");
  auto thresholds=torch::full({s.size(1)},.5,torch::kFloat64);
  for(I c=0;c<s.size(1);++c){auto sc=s.select(1,c).index({k.select(1,c)}).to(torch::kCPU,torch::kFloat64),target=y.select(1,c).index({k.select(1,c)}).to(torch::kCPU,torch::kFloat64);
    if(target.sum().item<double>()<=0)continue;auto sorted=torch::argsort(sc,true,0,true);auto truth=target.index_select(0,sorted),prob=sc.index_select(0,sorted);auto tp=truth.cumsum(0);
    auto score=2*tp/(torch::arange(tp.numel(),tp.options())+1+truth.sum());double best=-1;I index=0;
    for(I i=0;i<score.numel();++i){if(i+1<prob.numel()&&prob[i].item<double>()==prob[i+1].item<double>())continue;auto value=score[i].item<double>();if(value>best){best=value;index=i;}}thresholds[c]=prob[index];}
  return thresholds;
}
std::map<std::string,double> reid_retrieval_metrics(const Tensor& embeddings,const Tensor& identities,const std::vector<std::string>& cameras){
  if(embeddings.dim()!=2||embeddings.size(0)>8192||embeddings.size(0)!=identities.numel()||embeddings.size(0)!=cameras.size()||!torch::isfinite(embeddings).all().item<bool>())throw InvalidArgument("Invalid ReID retrieval inventory");
  auto x=embeddings.to(torch::kCPU,torch::kFloat64),ids=identities.to(torch::kCPU,torch::kInt64);x=x/x.square().sum(1,true).sqrt().clamp_min(1e-12);double aps=0,rank1=0;I queries=0;
  for(I i=0;i<x.size(0);++i){auto order=torch::argsort(x.matmul(x[i]),true,0,true);I positives=0;for(I j=0;j<x.size(0);++j)if(i!=j&&ids[i].item<I>()==ids[j].item<I>()&&cameras[i]!=cameras[j])++positives;if(!positives)continue;
    ++queries;I rank=0,hits=0;double ap=0;for(I k=0;k<order.numel();++k){auto j=order[k].item<I>();bool same=ids[i].item<I>()==ids[j].item<I>();if(i==j||(same&&cameras[i]==cameras[j]))continue;++rank;if(same){++hits;ap+=static_cast<double>(hits)/rank;if(rank==1)rank1+=1;}}
    aps+=ap/positives;}
  if(!queries)throw InvalidArgument("ReID evaluation has no cross-camera positive queries");return{{"map",aps/queries},{"rank1",rank1/queries},{"queries",static_cast<double>(queries)}};
}
}
