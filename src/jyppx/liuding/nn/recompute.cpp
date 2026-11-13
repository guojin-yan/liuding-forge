// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/nn/recompute.hpp>
#include <jyppx/liuding/foundation/random.hpp>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/instancenorm.h>
#include <ATen/autocast_mode.h>
namespace jyppx::liuding::nn {
namespace {
using I=std::int64_t;
struct AutocastState {
  bool cpu=at::autocast::is_autocast_enabled(torch::kCPU),cuda=at::autocast::is_autocast_enabled(torch::kCUDA);
  at::ScalarType cpu_dtype=at::autocast::get_autocast_dtype(torch::kCPU),cuda_dtype=at::autocast::get_autocast_dtype(torch::kCUDA);
  bool cache=at::autocast::is_autocast_cache_enabled();
  void restore()const{at::autocast::set_autocast_enabled(torch::kCPU,cpu);at::autocast::set_autocast_enabled(torch::kCUDA,cuda);
    at::autocast::set_autocast_dtype(torch::kCPU,cpu_dtype);at::autocast::set_autocast_dtype(torch::kCUDA,cuda_dtype);at::autocast::set_autocast_cache_enabled(cache);}
};
struct State:torch::CustomClassHolder {
  std::shared_ptr<torch::nn::Module> module;
  std::function<Tensor(torch::nn::Module&,const Tensor&)> callback;
  core::RngState rng;AutocastState autocast;bool cuda=false,input_grad=false;
  std::vector<std::pair<std::shared_ptr<torch::nn::Module>,bool>> modes;
  std::vector<std::pair<Tensor,I>> versions;
  void check()const{for(const auto& [m,mode]:modes)if(m->is_training()!=mode)throw InvalidArgument("Recompute region mode changed");
    for(const auto& [t,v]:versions)if(t._version()!=v)throw InvalidArgument("Recompute region mutated a saved parameter/buffer");}
};
struct Recompute:torch::autograd::Function<Recompute>{
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor input,at::TensorList parameters,c10::intrusive_ptr<State> state){
    state->rng=core::get_rng_state(state->cuda);state->autocast=AutocastState{};state->input_grad=input.requires_grad();
    std::vector<Tensor> saved{input};saved.insert(saved.end(),parameters.begin(),parameters.end());ctx->save_for_backward(saved);
    ctx->saved_data["state"]=c10::IValue::make_capsule(state);auto result=state->callback(*state->module,input);state->check();
    if(!result.defined()||!result.is_floating_point())throw InvalidArgument("Recompute region must return a floating tensor");return result;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list gradient){
    auto capsule=ctx->saved_data.at("state").toCapsule();auto* state=static_cast<State*>(capsule.get());state->check();auto saved=ctx->get_saved_variables();
    const auto caller_rng=core::get_rng_state(state->cuda);const auto caller_autocast=AutocastState{};
    try {
      core::set_rng_state(state->rng,state->cuda);state->autocast.restore();at::autocast::set_autocast_cache_enabled(false);
      torch::AutoGradMode enable(true);auto input=saved[0].detach();input.set_requires_grad(state->input_grad);
      auto output=state->callback(*state->module,input);state->check();std::vector<Tensor> differentiable;std::vector<std::size_t> slots;
      if(input.requires_grad()){differentiable.push_back(input);slots.push_back(0);}
      for(std::size_t i=1;i<saved.size();++i)if(saved[i].requires_grad()){differentiable.push_back(saved[i]);slots.push_back(i);}
      torch::autograd::variable_list result(saved.size()+1);
      if(!differentiable.empty()&&output.requires_grad()){
        auto grads=torch::autograd::grad({output},differentiable,gradient,false,false,true);
        for(std::size_t i=0;i<grads.size();++i)result[slots[i]]=grads[i];}
      caller_autocast.restore();core::set_rng_state(caller_rng,state->cuda);return result;
    }catch(...){caller_autocast.restore();core::set_rng_state(caller_rng,state->cuda);throw;}
  }
};
}
Tensor recompute(const Tensor& input,std::shared_ptr<torch::nn::Module> module,
    std::function<Tensor(torch::nn::Module&,const Tensor&)> forward,bool enabled){
  if(!module||!forward||!input.defined())throw InvalidArgument("Recompute needs input, owning module and callback");
  if(!enabled||!torch::GradMode::is_enabled())return forward(*module,input);
  auto state=c10::make_intrusive<State>();state->module=std::move(module);state->callback=std::move(forward);state->cuda=input.is_cuda();
  for(const auto& m:state->module->modules()){
    state->modes.emplace_back(m,m->is_training());bool statistics=false;
    if(auto bn=std::dynamic_pointer_cast<torch::nn::BatchNorm1dImpl>(m))statistics=bn->options.track_running_stats();
    if(auto bn=std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(m))statistics=bn->options.track_running_stats();
    if(auto bn=std::dynamic_pointer_cast<torch::nn::BatchNorm3dImpl>(m))statistics=bn->options.track_running_stats();
    if(auto bn=std::dynamic_pointer_cast<torch::nn::InstanceNorm1dImpl>(m))statistics=bn->options.track_running_stats();
    if(auto bn=std::dynamic_pointer_cast<torch::nn::InstanceNorm2dImpl>(m))statistics=bn->options.track_running_stats();
    if(auto bn=std::dynamic_pointer_cast<torch::nn::InstanceNorm3dImpl>(m))statistics=bn->options.track_running_stats();
    if(statistics&&m->is_training())throw InvalidArgument("Activation recomputation rejects training normalization running statistics");}
  auto parameters=state->module->parameters();for(const auto& t:parameters)state->versions.emplace_back(t,t._version());
  for(const auto& t:state->module->buffers())state->versions.emplace_back(t,t._version());
  return Recompute::apply(input,at::TensorList(parameters),std::move(state));
}
}
