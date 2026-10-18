#include <jyppx/foundation/error.hpp>
#include <jyppx/training/metrics/regression.hpp>

#include <string>
#include <utility>

namespace jyppx {
namespace training {
namespace metrics {

namespace {

constexpr char kComponent[] = "training.metrics.regression";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
    throw InvalidArgument(
        "Invalid regression metric contract",
        ErrorDetails{.component = kComponent,
                     .operation = std::move(operation),
                     .expected = std::move(expected),
                     .actual = std::move(actual),
                     .hint = std::move(hint)});
}

// Flattens and validates a prediction/target pair for elementwise metrics.
std::pair<Tensor, Tensor> flatten_pair(const Tensor& preds, const Tensor& targets) {
    if (!preds.defined() || !targets.defined()) {
        throw_invalid("update.inputs", "defined prediction and target tensors",
                      "undefined", "Provide both tensors.");
    }
    if (preds.numel() == 0 || targets.numel() == 0) {
        return {};
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

} // namespace

void MSE::update(const Tensor& preds, const Tensor& targets) {
    const auto [p, t] = flatten_pair(preds, targets);
    if (!p.defined()) {
        return;
    }
    squared_error_sum_ += (p - t).pow(2).sum().item<double>();
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
    absolute_error_sum_ += (p - t).abs().sum().item<double>();
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
    const auto errors = p - t;
    residual_sum_ += errors.pow(2).sum().item<double>();
    target_sum_ += t.sum().item<double>();
    target_sq_sum_ += t.pow(2).sum().item<double>();
    count_ += p.numel();
}

double R2Score::compute() {
    if (count_ == 0) {
        return 0.0;
    }
    const double n = static_cast<double>(count_);
    const double mean = target_sum_ / n;
    const double total_variance = target_sq_sum_ - n * mean * mean;
    if (total_variance <= 0.0) {
        // Constant targets have no variance; R^2 is conventionally reported
        // as 1 when the fit is also exact and 0 otherwise.
        return residual_sum_ == 0.0 ? 1.0 : 0.0;
    }
    return 1.0 - residual_sum_ / total_variance;
}

void R2Score::reset() {
    residual_sum_ = 0.0;
    target_sum_ = 0.0;
    target_sq_sum_ = 0.0;
    count_ = 0;
}

} // namespace metrics
} // namespace training
} // namespace jyppx
