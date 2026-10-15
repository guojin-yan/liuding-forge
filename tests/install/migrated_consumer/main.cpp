#include <jyppx/core/device.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/version.hpp>
#include <jyppx/nn/loss/contrastive_loss.hpp>
#include <jyppx/nn/loss/focal_loss.hpp>
#include <jyppx/nn/loss/label_smoothing_cross_entropy.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

int main() {
  const jyppx::InvalidArgument error(
      "consumer probe", jyppx::ErrorDetails{.component = "InstallConsumer"});
  const bool valid_version = std::string(jyppx::version()) == "0.3.0";
  const bool valid_error = error.code() == jyppx::ErrorCode::invalid_argument &&
                           error.details().component == "InstallConsumer";
  const bool valid_device = jyppx::parse_device("cpu").is_cpu();
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
  return valid_version && valid_error && valid_device && valid_data &&
                 valid_nn && valid_contrastive && valid_focal
             ? 0
             : 1;
}
