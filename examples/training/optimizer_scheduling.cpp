// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/** @file optimizer_scheduling.cpp
 * @brief Demonstrates LR events and coordinated model/SGD/scheduler replay.
 */

#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/event_lr_scheduler.hpp>
#include <jyppx/liuding/optim/linear_lr_scheduler.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace liuding = jyppx::liuding;
namespace optim = liuding::optim;
namespace training = liuding::training;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

/** @brief Fixed CPU quadratic: loss = weight^2 + bias^2, momentum = 1/2. */
struct Domain {
  torch::nn::Linear holder{torch::nn::LinearOptions(1, 1)};
  std::shared_ptr<liuding::nn::ModelAdapter> model;
  std::shared_ptr<optim::SgdOptimizerAdapter> optimizer;
  std::shared_ptr<optim::LrScheduler> scheduler;

  explicit Domain(std::string_view mode) {
    holder->to(torch::kFloat64);
    {
      torch::NoGradGuard guard;
      holder->weight.fill_(1.0);
      holder->bias.fill_(2.0);
    }
    model = std::make_shared<liuding::nn::ModelAdapter>(
        liuding::nn::adapt_tensor_holder(holder));
    optimizer = std::make_shared<optim::SgdOptimizerAdapter>(
        "optimizer.manual",
        std::vector<optim::SgdParameterGroup>{
            {.group_id = "weight",
             .parameters = {{.parameter_id = "weight",
                             .parameter = holder->weight}},
             .options = {.learning_rate = 0.125, .momentum = 0.5}},
            {.group_id = "bias",
             .parameters = {{.parameter_id = "bias",
                             .parameter = holder->bias}},
             .options = {.learning_rate = 0.0625, .momentum = 0.5}}});
    if (mode == "epoch") {
      scheduler = std::make_shared<optim::StepLrScheduler>(
          optimizer, optim::StepLrSchedulerOptions{
                         .scheduler_id = "scheduler.manual",
                         .step_size = 2, .gamma = 0.5});
    } else if (mode == "warmup") {
      scheduler = std::make_shared<optim::WarmupLrScheduler>(
          optimizer, optim::WarmupLrSchedulerOptions{
                         .scheduler_id = "scheduler.manual",
                         .warmup_iters = 2, .start_factor = 0.5,
                         .step_mode = optim::SchedulerStepMode::batch});
    } else {
      require(mode == "plateau", "Unknown scheduler mode");
      scheduler = std::make_shared<optim::ReduceLrOnPlateauScheduler>(
          optimizer, optim::ReduceLrOnPlateauSchedulerOptions{
                         .scheduler_id = "scheduler.manual",
                         .factor = 0.5, .patience = 0});
    }
  }
};

struct Expected {
  std::array<double, 4> used_rates;
  std::array<double, 4> next_rates;
  std::array<double, 4> weight;
  std::array<double, 4> bias;
};

Expected reference(std::string_view mode) {
  // Independently derived rational values. All constants are exactly binary.
  if (mode == "epoch") {
    return {{0.125, 0.125, 0.0625, 0.0625},
            {0.125, 0.0625, 0.0625, 0.03125},
            {0.75, 0.4375, 0.3046875, 0.2001953125},
            {1.75, 1.40625, 1.232421875, 1.0684814453125}};
  }
  if (mode == "warmup") {
    return {{0.0625, 0.09375, 0.125, 0.125},
            {0.09375, 0.125, 0.125, 0.125},
            {0.875, 0.6171875, 0.291015625, 0.05517578125},
            {1.875, 1.60546875, 1.22509765625, 0.88177490234375}};
  }
  return {{0.125, 0.125, 0.0625, 0.03125},
          {0.125, 0.0625, 0.03125, 0.03125},
          {0.75, 0.4375, 0.3046875, 0.25244140625},
          {1.75, 1.40625, 1.232421875, 1.15045166015625}};
}

void advance(Domain& domain, std::string_view mode, std::uint64_t end) {
  const auto expected = reference(mode);
  const std::array<double, 4> validation_losses{1.0, 1.0, 2.0, 0.5};
  for (auto i = domain.optimizer->completed_steps(); i < end; ++i) {
    require(domain.scheduler->completed_steps() == i &&
                domain.optimizer->learning_rates() ==
                    std::vector<double>{expected.used_rates.at(i),
                                         expected.used_rates.at(i) / 2},
            "Unexpected rate before optimizer update");
    domain.optimizer->zero_grad();
    const auto loss = domain.holder->weight.square().sum() +
                      domain.holder->bias.square().sum();
    loss.backward();
    domain.optimizer->step();
    if (mode == "plateau") {
      domain.scheduler->step(validation_losses.at(i));
    } else {
      // This example has one successful update per epoch in epoch mode.
      domain.scheduler->step();
    }
    require(domain.optimizer->completed_steps() == i + 1 &&
                domain.scheduler->completed_steps() == i + 1 &&
                domain.holder->weight.item<double>() == expected.weight.at(i) &&
                domain.holder->bias.item<double>() == expected.bias.at(i) &&
                domain.optimizer->learning_rates() ==
                    std::vector<double>{expected.next_rates.at(i),
                                         expected.next_rates.at(i) / 2},
            "Optimizer or scheduler differs from scalar reference");
    std::cout << "step=" << i + 1 << " used_rate=" << expected.used_rates.at(i)
              << " weight=" << domain.holder->weight.item<double>()
              << " bias=" << domain.holder->bias.item<double>()
              << " next_rate=" << expected.next_rates.at(i) << '\n';
  }
}

void persistent_state(Domain& domain, bool saving,
                      const std::filesystem::path& path) {
  auto model_state = std::make_shared<liuding::nn::ModelCheckpointParticipant>(
      *domain.model, liuding::nn::ModelCheckpointParticipantOptions{
                         .max_topology_bytes = 64U << 10U,
                         .max_tensor_bytes = 1024,
                         .max_total_bytes = 256U << 10U});
  auto optimizer_state =
      std::make_shared<optim::SgdOptimizerCheckpointParticipant>(
          domain.optimizer, optim::SgdOptimizerCheckpointOptions{
                                .max_topology_bytes = 64U << 10U,
                                .max_control_bytes = 64U << 10U,
                                .max_tensor_bytes = 1024,
                                .max_total_bytes = 256U << 10U});
  auto scheduler_state =
      std::make_shared<optim::LrSchedulerCheckpointParticipant>(
          domain.scheduler, optim::LrSchedulerCheckpointOptions{
                                .max_topology_bytes = 64U << 10U,
                                .max_control_bytes = 64U << 10U,
                                .max_total_bytes = 256U << 10U});
  training::CheckpointParticipantRegistry registry;
  registry.register_participants({model_state, optimizer_state, scheduler_state});
  training::CheckpointCoordinator coordinator(
      registry, {.max_participants = 3, .max_rollback_bytes = 1U << 20U});
  training::FileCheckpointStore file(
      path, {.max_artifact_bytes = 1U << 20U, .max_field_bytes = 64U << 10U});
  if (saving) {
    advance(domain, "epoch", 2);
    const auto descriptor = coordinator.save(file);
    require(descriptor.participants.size() == 3, "Incomplete checkpoint");
  } else {
    coordinator.restore(file, liuding::RestorePolicy::exact_only);
    const auto expected = reference("epoch");
    require(domain.optimizer->completed_steps() == 2 &&
                domain.scheduler->completed_steps() == 2 &&
                domain.holder->weight.item<double>() == expected.weight[1] &&
                domain.holder->bias.item<double>() == expected.bias[1],
            "Incomplete restored optimizer domain");
    advance(domain, "epoch", 4);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    torch::set_num_threads(1);
    torch::manual_seed(7);
    require(argc >= 2, "Usage: program epoch|warmup|plateau OR save|restore PATH");
    const std::string_view mode = argv[1];
    const bool persistent = mode == "save" || mode == "restore";
    require(argc == (persistent ? 3 : 2), "Unexpected example arguments");
    Domain domain(persistent ? "epoch" : mode);
    std::cout << std::setprecision(17);
    if (persistent) {
      persistent_state(domain, mode == "save", argv[2]);
    } else {
      advance(domain, mode, 4);
    }
    std::cout << "mode=" << mode
              << " steps=" << domain.optimizer->completed_steps()
              << " scheduler_steps=" << domain.scheduler->completed_steps()
              << " checked=true\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
