#include <jyppx/core/device.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/foundation/checkpoint.hpp>
#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/version.hpp>
#include <jyppx/nn/loss/contrastive_loss.hpp>
#include <jyppx/nn/loss/dice_loss.hpp>
#include <jyppx/nn/loss/focal_loss.hpp>
#include <jyppx/nn/loss/label_smoothing_cross_entropy.hpp>
#include <jyppx/nn/model_adapter.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
  const bool valid_checkpoint_contract =
      jyppx::restore_policy_accepts(jyppx::RestorePolicy::exact_only,
                                    state_manifest.restore_level);
  const auto indices = jyppx::data::samplers::SequentialSampler(3).indices();
  const bool valid_data = indices == std::vector<std::size_t>({0, 1, 2});
  const auto logits = torch::tensor({{2.0, -1.0}}, torch::kFloat64);
  const auto target = torch::tensor({0}, torch::kLong);
  const auto loss =
      jyppx::nn::functional::label_smoothing_cross_entropy(logits, target, 0.1);
  const bool valid_nn = loss.numel() == 1 && std::isfinite(loss.item<double>());
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
  return valid_version && valid_error && valid_device &&
                 valid_checkpoint_contract && valid_data && valid_nn &&
                 valid_contrastive && valid_focal && valid_dice && valid_adapter
             ? 0
             : 1;
}
