// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/training/distillation.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
#include <cmath>
namespace jyppx::liuding::vision::tasks {
training::StepFunction make_distillation_step(training::StepFunction supervised,std::shared_ptr<nn::ModelAdapter> teacher,DistillationOptions o){
  if(!supervised||!teacher||!std::isfinite(o.temperature)||o.temperature<=0||!std::isfinite(o.logits_weight)||o.logits_weight<0||!std::isfinite(o.feature_weight)||o.feature_weight<0||o.logits_weight+o.feature_weight<=0)throw InvalidArgument("Invalid distillation strategy");
  teacher->state().eval();for(auto& t:teacher->state().parameters())t.set_requires_grad(false);
  auto versions=std::make_shared<std::vector<std::pair<Tensor,std::int64_t>>>();
  return[supervised=std::move(supervised),teacher=std::move(teacher),versions,o](const training::StepInput& in){auto out=supervised(in);if(in.context.phase!=training::StepPhase::training)return out;
    if(versions->empty()){for(auto& t:teacher->state().parameters())versions->emplace_back(t,t._version());for(auto& t:teacher->state().buffers())versions->emplace_back(t,t._version());}
    if(!out.prediction)throw InvalidArgument("Distillation needs student predictions");for(const auto& [t,v]:*versions)if(t._version()!=v)throw InvalidArgument("Frozen teacher state mutated");teacher->state().eval();
    std::vector<Tensor> target;{torch::NoGradGuard guard;target=raw_tensors(teacher->invoke(in.invocation).value);}auto student=raw_tensors(out.prediction->value);
    auto kd=out.loss*0;if(o.logits_weight){Tensor loss;if(o.multilabel){if(!in.target||!in.target->isTuple())throw InvalidArgument("Bernoulli distillation needs known-label mask");loss=bernoulli_distillation_loss(student[0],target[0],in.target->toTupleRef().elements()[1].toTensor(),o.temperature);}else loss=logits_distillation_loss(student[0],target[0],o.temperature);out.metrics["kd.logits"]=loss.detach().item<double>();kd=kd+loss*o.logits_weight;}
    if(o.feature_weight){if(student.size()<2||target.size()<2)throw InvalidArgument("Feature distillation needs explicit registered feature projections");auto loss=feature_distillation_loss(student[1],target[1]);out.metrics["kd.feature"]=loss.detach().item<double>();kd=kd+loss*o.feature_weight;}
    for(const auto& [t,v]:*versions)if(t._version()!=v)throw InvalidArgument("Teacher forward mutated frozen state");out.metrics["supervised.loss"]=out.loss.detach().item<double>();out.loss=out.loss+kd;return out;};
}
}
