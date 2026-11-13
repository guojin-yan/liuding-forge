// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/data/sampling.hpp>
#include <jyppx/liuding/foundation/random.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <map>
#include <numeric>
#include <cmath>
namespace jyppx::liuding::vision::tasks {
namespace {
class IdentitySampler:public data::samplers::Sampler{
 public:
  IdentitySampler(const std::vector<std::int64_t>& labels,std::size_t batch,std::size_t k,std::uint64_t seed):batch_(batch),k_(k),seed_(seed){
    if(labels.empty()||labels.size()>1000000||k<2||batch<2*k||batch>4096||batch%k)throw InvalidArgument("Identity sampler needs P>=2,K>=2 and batch=P*K");
    for(std::size_t i=0;i<labels.size();++i){if(labels[i]<0)throw InvalidArgument("Negative training identity");groups_[labels[i]].push_back(i);}
    if(groups_.size()<batch/k)throw InvalidArgument("Insufficient distinct identities for P*K batch");
    for(const auto& [label,rows]:groups_){if(rows.size()<2)throw InvalidArgument("Each training identity needs two distinct images");identities_.push_back(label);}
    size_=((labels.size()+batch-1)/batch)*batch;auto bytes=nlohmann::json{{"labels",labels},{"batch",batch},{"k",k},{"seed",seed}}.dump();identity_="identity-pk/v1:"+sha256_hex(std::as_bytes(std::span(bytes)));}
  std::vector<std::size_t> indices()const override{
    auto state=core::derive_seed(seed_,generation_++,epoch_);auto next=[&]{state+=0x9e3779b97f4a7c15ULL;auto z=state;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);};
    auto bounded=[&](std::size_t bound){auto threshold=-static_cast<std::uint64_t>(bound)%bound;std::uint64_t x;do{x=next();}while(x<threshold);return static_cast<std::size_t>(x%bound);};
    auto shuffle=[&](auto& v){for(std::size_t i=v.size();i>1;--i)std::swap(v[i-1],v[bounded(i)]);};
    std::vector<std::size_t> result;result.reserve(size_);
    for(std::size_t batch=0;batch<size_/batch_;++batch){auto identities=identities_;shuffle(identities);
      for(std::size_t p=0;p<batch_/k_;++p){auto rows=groups_.at(identities[p]);shuffle(rows);for(std::size_t k=0;k<k_;++k)result.push_back(rows[k%rows.size()]);}}
    return result;
  }
  std::size_t size()const override{return size_;}
  void set_epoch(std::uint64_t epoch)override{epoch_=epoch;generation_=0;}
  bool supports_epoch_control()const override{return true;}bool supports_state()const override{return true;}
  std::string configuration_identity()const override{return identity_;}
  std::string save_state()const override{return nlohmann::json{{"schema","identity-pk/v1"},{"identity",identity_},{"epoch",epoch_},{"generation",generation_}}.dump();}
  void load_state(const std::string& state)override{if(state.size()>16384)throw SerializationError("Identity sampler state too large");auto s=nlohmann::json::parse(state);
    if(s.at("schema")!="identity-pk/v1"||s.at("identity")!=identity_)throw CompatibilityError("Identity sampler source/configuration differs");auto epoch=s.at("epoch").get<std::uint64_t>(),generation=s.at("generation").get<std::uint64_t>();epoch_=epoch;generation_=generation;}
  std::map<std::int64_t,std::vector<std::size_t>> groups_;std::vector<std::int64_t> identities_;std::size_t batch_,k_,size_;std::uint64_t seed_,epoch_=0;mutable std::uint64_t generation_=0;std::string identity_;
};
}
std::shared_ptr<data::samplers::Sampler> make_identity_sampler(const std::vector<std::int64_t>& ids,std::size_t b,std::size_t k,std::uint64_t seed){return std::make_shared<IdentitySampler>(ids,b,k,seed);}
std::vector<double> long_tail_weights(const std::vector<std::int64_t>& labels,double power){if(labels.empty()||labels.size()>1000000||!std::isfinite(power)||power<0||power>1)throw InvalidArgument("Invalid training label frequency inventory");
  std::map<std::int64_t,std::size_t> counts;for(auto label:labels){if(label<0)throw InvalidArgument("Negative training label");++counts[label];}
  std::vector<double> out;for(auto label:labels)out.push_back(std::pow(static_cast<double>(counts[label]),-power));return out;}
}
