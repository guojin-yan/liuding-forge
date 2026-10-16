#include <jyppx/core/device.hpp>
#include <jyppx/data/data_pipeline.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/foundation/checkpoint.hpp>
#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/random_checkpoint_participant.hpp>
#include <jyppx/foundation/version.hpp>
#include <jyppx/nn/loss/contrastive_loss.hpp>
#include <jyppx/nn/loss/dice_loss.hpp>
#include <jyppx/nn/loss/focal_loss.hpp>
#include <jyppx/nn/loss/label_smoothing_cross_entropy.hpp>
#include <jyppx/nn/model_adapter.hpp>
#include <jyppx/nn/model_checkpoint_participant.hpp>
#include <jyppx/optim/composite_lr_scheduler.hpp>
#include <jyppx/optim/epoch_lr_scheduler.hpp>
#include <jyppx/optim/event_lr_scheduler.hpp>
#include <jyppx/optim/linear_lr_scheduler.hpp>
#include <jyppx/optim/sgd_adapter.hpp>
#include <jyppx/optim/step_lr_scheduler.hpp>
#include <jyppx/training/callback_dispatcher.hpp>
#include <jyppx/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/training/checkpoint_coordinator.hpp>
#include <jyppx/training/metrics/classification.hpp>
#include <jyppx/training/scheduler_adapter.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

class ConsumerParticipant final : public jyppx::CheckpointParticipant {
 public:
  std::string participant_id() const override { return "consumer.state"; }

  jyppx::StateManifest describe_state() const override {
    return {.participant_id = participant_id(),
            .schema_version = 1,
            .restore_level = jyppx::RestoreLevel::exact,
            .fields = {{.name = "value", .max_bytes = 8}}};
  }

  void save_state(jyppx::StateWriter& writer) const override {
    const std::array<std::byte, 8> value{};
    writer.write_blob("value", value);
  }

  void load_state(const jyppx::StateReader& reader,
                  jyppx::RestorePolicy policy) override {
    if (!jyppx::restore_policy_accepts(policy,
                                       reader.manifest().restore_level)) {
      throw jyppx::CompatibilityError("consumer policy rejected");
    }
    (void)reader.read_blob("value", 8);
  }
};

}  // namespace

int main() {
  const jyppx::InvalidArgument error(
      "consumer probe", jyppx::ErrorDetails{.component = "InstallConsumer"});
  const bool valid_version = std::string(jyppx::version()) == "0.3.0";
  const bool valid_error = error.code() == jyppx::ErrorCode::invalid_argument &&
                           error.details().component == "InstallConsumer";
  const bool valid_device = jyppx::parse_device("cpu").is_cpu();
  const jyppx::StateManifest state_manifest{
      .participant_id = "consumer.probe",
      .schema_version = 1,
      .restore_level = jyppx::RestoreLevel::exact,
      .fields = {{.name = "state", .max_bytes = 8}},
  };
  jyppx::validate_state_manifest(state_manifest);
  const bool valid_checkpoint_contract = jyppx::restore_policy_accepts(
      jyppx::RestorePolicy::exact_only, state_manifest.restore_level);
  jyppx::training::CheckpointParticipantRegistry registry;
  registry.register_participant(std::make_shared<ConsumerParticipant>());
  jyppx::training::CheckpointCoordinator coordinator(registry);
  const auto checkpoint_descriptor = coordinator.describe();
  const bool valid_checkpoint_coordinator =
      checkpoint_descriptor.participants.size() == 1 &&
      checkpoint_descriptor.participants.front().participant_id ==
          "consumer.state";
  auto installed_early_stopping =
      std::make_shared<jyppx::training::callbacks::EarlyStoppingCallback>(
          jyppx::training::callbacks::EarlyStoppingOptions{
              .callback_id = "callback.consumer"});
  auto installed_callback_dispatcher =
      std::make_shared<jyppx::training::CallbackDispatcher>(
          std::vector<std::shared_ptr<jyppx::training::CallbackHandler>>{
              installed_early_stopping},
          jyppx::training::CallbackDispatcherOptions{.dispatcher_id =
                                                         "callbacks.consumer"});
  jyppx::training::CheckpointParticipantRegistry callback_registry;
  jyppx::training::register_callback_checkpoint_participants(
      callback_registry, installed_callback_dispatcher);
  const bool valid_callback =
      callback_registry.participant_ids() ==
      std::vector<std::string>({"callback.consumer", "callbacks.consumer"});
  jyppx::core::RngCheckpointParticipant installed_rng_participant(
      {.participant_id = "rng.consumer", .include_cuda = false});
  const bool valid_rng_checkpoint =
      installed_rng_participant.describe_state().fields.size() == 2;
  const auto indices = jyppx::data::samplers::SequentialSampler(3).indices();
  const bool valid_data = indices == std::vector<std::size_t>({0, 1, 2});
  auto installed_pipeline = jyppx::data::DataPipeline(
      std::make_shared<jyppx::data::samplers::SequentialSampler>(3),
      {.pipeline_id = "data.consumer",
       .dataset_version = "dataset.consumer/v1",
       .dataset_size = 3,
       .batch_size = 2},
      [](std::span<const std::size_t> batch_indices) {
        return c10::IValue(static_cast<std::int64_t>(batch_indices.size()));
      });
  installed_pipeline.set_epoch(0);
  const auto installed_batch = installed_pipeline.next();
  const bool valid_data_pipeline =
      installed_batch.has_value() && installed_batch->sample_ids.size() == 2 &&
      installed_pipeline.describe_state().restore_level ==
          jyppx::RestoreLevel::exact;
  const auto logits = torch::tensor({{2.0, -1.0}}, torch::kFloat64);
  const auto target = torch::tensor({0}, torch::kLong);
  const auto loss =
      jyppx::nn::functional::label_smoothing_cross_entropy(logits, target, 0.1);
  const bool valid_nn = loss.numel() == 1 && std::isfinite(loss.item<double>());
  auto installed_metric = std::make_shared<jyppx::training::metrics::Precision>(
      jyppx::training::metrics::ClassificationMetricOptions{
          .metric_id = "metric.consumer", .num_classes = 2});
  installed_metric->update(logits, target);
  jyppx::validate_checkpoint_participant(*installed_metric);
  const bool valid_metric =
      installed_metric->descriptor().type_id == "jyppx.metric.precision/v1" &&
      installed_metric->total_observations() == 1 &&
      installed_metric->compute() == 0.5;
  const auto pair_a = torch::tensor({{0.0, 0.0}}, torch::kFloat64);
  const auto pair_b = torch::tensor({{0.5, 0.0}}, torch::kFloat64);
  const auto pair_target = torch::tensor({1}, torch::kLong);
  const auto contrastive =
      jyppx::nn::functional::contrastive_loss(pair_a, pair_b, pair_target, 1.0);
  const bool valid_contrastive =
      contrastive.numel() == 1 && std::isfinite(contrastive.item<double>());
  const auto focal = jyppx::nn::functional::focal_loss(
      torch::tensor({{0.0, 0.0}}, torch::kFloat64),
      torch::tensor({{1.0, 0.0}}, torch::kFloat64), 0.25, 2.0);
  const bool valid_focal =
      focal.numel() == 1 && std::isfinite(focal.item<double>());
  jyppx::nn::DiceLossOptions dice_options;
  dice_options.mode = jyppx::nn::DiceMode::multiclass;
  const auto dice = jyppx::nn::functional::dice_loss(
      torch::tensor({{2.0, -1.0}}, torch::kFloat64),
      torch::tensor({0}, torch::kLong), dice_options);
  const bool valid_dice =
      dice.numel() == 1 && std::isfinite(dice.item<double>());

  torch::nn::Linear linear(2, 1);
  auto installed_model_state =
      std::make_shared<jyppx::nn::ModelStateView>(linear.ptr());
  jyppx::nn::ModelCheckpointParticipant installed_model_participant(
      installed_model_state);
  const bool valid_model_checkpoint =
      installed_model_participant.describe_state().fields.size() == 3;
  jyppx::nn::TensorSchema adapter_input;
  adapter_input.dtype = torch::kFloat32;
  adapter_input.rank = 2;
  adapter_input.shape =
      std::vector<std::optional<std::int64_t>>{std::nullopt, 2};
  jyppx::nn::TensorSchema adapter_output;
  adapter_output.dtype = torch::kFloat32;
  adapter_output.rank = 2;
  adapter_output.shape =
      std::vector<std::optional<std::int64_t>>{std::nullopt, 1};
  auto adapter = jyppx::nn::adapt_tensor_holder(
      linear, std::move(adapter_input), std::move(adapter_output));
  const auto adapted =
      adapter.invoke({.positional = {torch::ones({2, 2})}}).value.toTensor();
  const bool valid_adapter =
      adapted.sizes().vec() == std::vector<std::int64_t>({2, 1}) &&
      adapter.state().parameters().size() == 2;
  auto installed_parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto installed_optimizer =
      std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
          "optimizer.main",
          std::vector<jyppx::optim::SgdParameterGroup>{
              {.group_id = "main",
               .parameters = {{.parameter_id = "weight",
                               .parameter = installed_parameter}},
               .options = {.learning_rate = 0.01, .momentum = 0.9}}});
  jyppx::optim::SgdOptimizerCheckpointParticipant
      installed_optimizer_participant(installed_optimizer);
  const bool valid_optimizer =
      installed_optimizer_participant.describe_state().fields.size() == 3 &&
      installed_optimizer->parameter_count() == 1;
  auto installed_scheduler = std::make_shared<jyppx::optim::StepLrScheduler>(
      installed_optimizer,
      jyppx::optim::StepLrSchedulerOptions{.step_size = 2, .gamma = 0.5});
  installed_scheduler->step();
  jyppx::optim::LrSchedulerCheckpointParticipant
      installed_scheduler_participant(installed_scheduler);
  auto installed_epoch_scheduler =
      std::make_shared<jyppx::optim::ExponentialLrScheduler>(
          installed_optimizer,
          jyppx::optim::ExponentialLrSchedulerOptions{
              .scheduler_id = "scheduler.exponential", .gamma = 0.5});
  installed_epoch_scheduler->step();
  jyppx::optim::LrSchedulerCheckpointParticipant
      installed_epoch_scheduler_participant(installed_epoch_scheduler);
  auto installed_restart_scheduler =
      std::make_shared<jyppx::optim::CosineAnnealingWarmRestartsLrScheduler>(
          installed_optimizer,
          jyppx::optim::CosineAnnealingWarmRestartsLrSchedulerOptions{
              .scheduler_id = "scheduler.restart", .t_0 = 2, .t_mult = 2});
  jyppx::training::SchedulerAdapter installed_scheduler_adapter(
      installed_restart_scheduler);
  installed_scheduler_adapter.step(
      {.trigger = jyppx::optim::SchedulerStepMode::epoch, .event_index = 1});
  auto installed_linear_scheduler =
      std::make_shared<jyppx::optim::WarmupLrScheduler>(
          installed_optimizer, jyppx::optim::WarmupLrSchedulerOptions{
                                   .scheduler_id = "scheduler.warmup",
                                   .warmup_iters = 2,
                                   .start_factor = 0.5});
  installed_linear_scheduler->step();
  jyppx::optim::LrSchedulerCheckpointParticipant
      installed_linear_scheduler_participant(installed_linear_scheduler);
  auto installed_event_scheduler =
      std::make_shared<jyppx::optim::ReduceLrOnPlateauScheduler>(
          installed_optimizer,
          jyppx::optim::ReduceLrOnPlateauSchedulerOptions{
              .scheduler_id = "scheduler.plateau", .patience = 0});
  installed_event_scheduler->step(1.0);
  jyppx::optim::LrSchedulerCheckpointParticipant
      installed_event_scheduler_participant(installed_event_scheduler);
  std::vector<std::unique_ptr<jyppx::optim::LrScheduler>> scheduler_children;
  scheduler_children.push_back(std::make_unique<jyppx::optim::StepLrScheduler>(
      installed_optimizer, jyppx::optim::StepLrSchedulerOptions{
                               .scheduler_id = "scheduler.child.first",
                               .step_size = 1,
                               .gamma = 0.5}));
  scheduler_children.push_back(std::make_unique<jyppx::optim::StepLrScheduler>(
      installed_optimizer, jyppx::optim::StepLrSchedulerOptions{
                               .scheduler_id = "scheduler.child.second",
                               .step_size = 1,
                               .gamma = 0.5}));
  auto installed_composite =
      std::make_shared<jyppx::optim::SequentialLrScheduler>(
          installed_optimizer, std::move(scheduler_children),
          jyppx::optim::SequentialLrSchedulerOptions{
              .scheduler_id = "scheduler.composite", .milestones = {1}});
  installed_composite->step();
  jyppx::optim::LrSchedulerCheckpointParticipant
      installed_composite_participant(installed_composite);
  const bool valid_scheduler =
      installed_scheduler_participant.describe_state().fields.size() == 2 &&
      installed_scheduler->completed_steps() == 1 &&
      installed_scheduler->descriptor().optimizer_id == "optimizer.main" &&
      installed_epoch_scheduler_participant.describe_state().fields.size() ==
          2 &&
      installed_epoch_scheduler->completed_steps() == 1 &&
      installed_scheduler_adapter.participant_id() == "scheduler.restart" &&
      installed_restart_scheduler->completed_steps() == 1 &&
      installed_linear_scheduler_participant.describe_state().fields.size() ==
          2 &&
      installed_linear_scheduler->completed_steps() == 1 &&
      installed_event_scheduler_participant.describe_state().schema_version ==
          3 &&
      installed_event_scheduler->completed_steps() == 1 &&
      installed_composite_participant.describe_state().fields.size() == 2 &&
      installed_composite->descriptor().children.size() == 2;
  return valid_version && valid_error && valid_device &&
                 valid_checkpoint_contract && valid_checkpoint_coordinator &&
                 valid_rng_checkpoint && valid_model_checkpoint && valid_data &&
                 valid_data_pipeline && valid_nn && valid_contrastive &&
                 valid_focal && valid_dice && valid_adapter && valid_metric &&
                 valid_optimizer && valid_scheduler && valid_callback
             ? 0
             : 1;
}
