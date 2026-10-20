// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#if defined(JYPPX_CONSUMER_FOUNDATION)
#include <jyppx/foundation/version.hpp>
#elif defined(JYPPX_CONSUMER_CORE)
#include <jyppx/core/device.hpp>
#elif defined(JYPPX_CONSUMER_INTEROP)
#include <jyppx/interop/checkpoint_manifest.hpp>
#elif defined(JYPPX_CONSUMER_DATA)
#include <jyppx/data/samplers/sequential_sampler.hpp>
#elif defined(JYPPX_CONSUMER_NN)
#include <jyppx/nn/loss/label_smoothing_cross_entropy.hpp>
#elif defined(JYPPX_CONSUMER_OPTIM)
#include <jyppx/optim/sgd_adapter.hpp>
#elif defined(JYPPX_CONSUMER_TRAINING)
#include <jyppx/training/trainer.hpp>
#elif defined(JYPPX_CONSUMER_EXPERIMENT)
#include <jyppx/experiment/config.hpp>
#elif defined(JYPPX_CONSUMER_ALL)
#include <jyppx/foundation/version.hpp>
#elif defined(JYPPX_CONSUMER_SDK)
#include <jyppx/jyppx.hpp>
#else
#error "Exactly one installed component consumer must be selected"
#endif

#include <cmath>
#include <string>
#include <vector>

int main() {
#if defined(JYPPX_CONSUMER_FOUNDATION)
  const auto fingerprint = jyppx::current_build_fingerprint();
  return !std::string(jyppx::version()).empty() &&
                 !fingerprint.git_commit.empty()
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_CORE)
  return jyppx::parse_device("cpu").is_cpu() ? 0 : 1;
#elif defined(JYPPX_CONSUMER_INTEROP)
  const auto capability = jyppx::interop::make_checkpoint_manifest_capability();
  return capability.feature_id == "interop.checkpoint_manifest" ? 0 : 1;
#elif defined(JYPPX_CONSUMER_DATA)
  const auto indices = jyppx::data::samplers::SequentialSampler(3).indices();
  return indices == std::vector<std::size_t>({0, 1, 2}) ? 0 : 1;
#elif defined(JYPPX_CONSUMER_NN)
  const auto loss = jyppx::nn::functional::label_smoothing_cross_entropy(
      torch::tensor({{2.0, -1.0}}, torch::kFloat64),
      torch::tensor({0}, torch::kLong), 0.1);
  return loss.numel() == 1 && std::isfinite(loss.item<double>()) ? 0 : 1;
#elif defined(JYPPX_CONSUMER_OPTIM)
  auto parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  jyppx::optim::SgdOptimizerAdapter optimizer(
      "optimizer.consumer",
      {{.group_id = "main",
        .parameters = {{.parameter_id = "weight", .parameter = parameter}},
        .options = {.learning_rate = 0.01}}});
  return optimizer.optimizer_id() == "optimizer.consumer" &&
                 optimizer.parameter_count() == 1
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_TRAINING)
  jyppx::training::Trainer trainer({.trainer_id = "trainer.consumer"});
  return trainer.phase() == jyppx::training::TrainerPhase::created &&
                 trainer.options().trainer_id == "trainer.consumer"
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_EXPERIMENT)
  return jyppx::experiment::config::is_secret_reference(
             "secret://env/consumer/token")
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_ALL)
  return std::string(jyppx::version()).empty() ? 1 : 0;
#elif defined(JYPPX_CONSUMER_SDK)
  return !std::string(jyppx::version()).empty() &&
                 jyppx::parse_device("cpu").is_cpu()
             ? 0
             : 1;
#endif
}
