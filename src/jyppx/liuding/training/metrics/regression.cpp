// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/metrics/regression.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace jyppx::liuding {
namespace training {
namespace metrics {

namespace {

constexpr char kComponent[] = "training.metrics.regression";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid regression metric contract",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

// Flattens and validates a prediction/target pair for elementwise metrics.
std::pair<Tensor, Tensor> flatten_pair(const Tensor& preds,
                                       const Tensor& targets) {
  if (!preds.defined() || !targets.defined()) {
    throw_invalid("update.inputs", "defined prediction and target tensors",
                  "undefined", "Provide both tensors.");
  }
  if ((preds.numel() == 0) != (targets.numel() == 0)) {
    throw_invalid("update.inputs.empty", "both tensors empty or both non-empty",
                  "one empty tensor",
                  "Provide a complete prediction/target pair.");
  }
  if (preds.numel() == 0) {
    return {};
  }
  if (preds.device() != targets.device()) {
    throw_invalid("update.inputs.device", "matching tensor devices",
                  preds.device().str() + " vs " + targets.device().str(),
                  "Move predictions and targets to the same device.");
  }
  if (preds.is_complex() || targets.is_complex()) {
    throw_invalid("update.inputs.dtype", "real prediction and target tensors",
                  "complex tensor", "Convert inputs to a real dtype.");
  }
  if ((preds.is_floating_point() &&
       !torch::isfinite(preds).all().item<bool>()) ||
      (targets.is_floating_point() &&
       !torch::isfinite(targets).all().item<bool>())) {
    throw_invalid("update.inputs.finite", "finite prediction and target values",
                  "NaN or Inf", "Remove non-finite values before updating.");
  }
  if (preds.numel() != targets.numel()) {
    throw_invalid("update.targets.shape",
                  std::to_string(preds.numel()) + " target elements",
                  std::to_string(targets.numel()),
                  "Provide matching prediction and target element counts.");
  }
  return {preds.reshape({-1}).to(torch::kFloat64),
          targets.reshape({-1}).to(torch::kFloat64)};
}

void validate_count(std::int64_t current, std::int64_t incoming) {
  if (incoming > std::numeric_limits<std::int64_t>::max() - current) {
    throw ResourceExhausted(
        "Regression metric element count exceeds its supported range",
        ErrorDetails{.component = kComponent,
                     .operation = "update.count",
                     .expected = "an accumulated count within int64",
                     .actual = std::to_string(current) + " + " +
                               std::to_string(incoming),
                     .hint = "Reset or publish the metric before continuing."});
  }
}

void validate_finite_candidate(double value, const char* operation) {
  if (!std::isfinite(value)) {
    throw NumericalError(
        "Regression metric accumulator became non-finite",
        ErrorDetails{.component = kComponent,
                     .operation = operation,
                     .expected = "a finite accumulator",
                     .actual = std::to_string(value),
                     .hint = "Use finite inputs or reset the metric."});
  }
}

}  // namespace

void MSE::update(const Tensor& preds, const Tensor& targets) {
  const auto [p, t] = flatten_pair(preds, targets);
  if (!p.defined()) {
    return;
  }
  const auto batch_sum = (p - t).pow(2).sum().item<double>();
  validate_finite_candidate(batch_sum, "update.squared_error");
  validate_count(count_, p.numel());
  const auto next_sum = squared_error_sum_ + batch_sum;
  validate_finite_candidate(next_sum, "update.squared_error.accumulator");
  squared_error_sum_ = next_sum;
  count_ += p.numel();
}

double MSE::compute() {
  return count_ == 0 ? 0.0 : squared_error_sum_ / static_cast<double>(count_);
}

void MSE::reset() {
  squared_error_sum_ = 0.0;
  count_ = 0;
}

void MAE::update(const Tensor& preds, const Tensor& targets) {
  const auto [p, t] = flatten_pair(preds, targets);
  if (!p.defined()) {
    return;
  }
  const auto batch_sum = (p - t).abs().sum().item<double>();
  validate_finite_candidate(batch_sum, "update.absolute_error");
  validate_count(count_, p.numel());
  const auto next_sum = absolute_error_sum_ + batch_sum;
  validate_finite_candidate(next_sum, "update.absolute_error.accumulator");
  absolute_error_sum_ = next_sum;
  count_ += p.numel();
}

double MAE::compute() {
  return count_ == 0 ? 0.0 : absolute_error_sum_ / static_cast<double>(count_);
}

void MAE::reset() {
  absolute_error_sum_ = 0.0;
  count_ = 0;
}

void R2Score::update(const Tensor& preds, const Tensor& targets) {
  const auto [p, t] = flatten_pair(preds, targets);
  if (!p.defined()) {
    return;
  }
  const auto batch_residual = (p - t).pow(2).sum().item<double>();
  const auto batch_count = p.numel();
  validate_finite_candidate(batch_residual, "update.residual");
  validate_count(count_, batch_count);
  const auto batch_mean = t.mean().item<double>();
  const auto batch_centered = (t - batch_mean).pow(2).sum().item<double>();
  validate_finite_candidate(batch_mean, "update.target.mean");
  validate_finite_candidate(batch_centered, "update.target.m2");

  // Merge batch moments with the existing population using the stable
  // parallel form of Welford's algorithm.
  const auto current_count = static_cast<double>(count_);
  const auto incoming_count = static_cast<double>(batch_count);
  const auto merged_count = current_count + incoming_count;
  double merged_mean = batch_mean;
  double merged_m2 = batch_centered;
  if (count_ != 0) {
    const auto delta = batch_mean - target_mean_;
    merged_mean = target_mean_ + delta * incoming_count / merged_count;
    merged_m2 = target_m2_ + batch_centered +
                delta * delta * current_count * incoming_count / merged_count;
  }
  const auto next_residual = residual_sum_ + batch_residual;
  validate_finite_candidate(merged_mean, "update.target.mean.accumulator");
  validate_finite_candidate(merged_m2, "update.target.m2.accumulator");
  validate_finite_candidate(next_residual, "update.residual.accumulator");
  residual_sum_ = next_residual;
  target_mean_ = merged_mean;
  target_m2_ = merged_m2;
  count_ += batch_count;
}

double R2Score::compute() {
  if (count_ == 0) {
    return 0.0;
  }
  const double total_variance = target_m2_;
  if (total_variance <= 0.0) {
    // Constant targets have no variance; R^2 is conventionally reported
    // as 1 when the fit is also exact and 0 otherwise.
    return residual_sum_ == 0.0 ? 1.0 : 0.0;
  }
  return 1.0 - residual_sum_ / total_variance;
}

void R2Score::reset() {
  residual_sum_ = 0.0;
  target_mean_ = 0.0;
  target_m2_ = 0.0;
  count_ = 0;
}

}  // namespace metrics
}  // namespace training
}  // namespace jyppx::liuding
