// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/metrics/average_meter.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace jyppx::liuding {
namespace training {
namespace metrics {

namespace {

constexpr char kComponent[] = "training.metrics.average_meter";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid AverageMeter contract",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

}  // namespace

void AverageMeter::update(double value, int64_t count) {
  if (count <= 0) {
    throw_invalid("update.count", "a positive observation count",
                  std::to_string(count), "Choose count greater than zero.");
  }
  if (!std::isfinite(value)) {
    throw_invalid("update.value", "a finite scalar", std::to_string(value),
                  "Remove NaN or Inf before updating.");
  }
  if (count > std::numeric_limits<std::int64_t>::max() - count_) {
    throw ResourceExhausted(
        "AverageMeter observation count exceeds its supported range",
        ErrorDetails{
            .component = kComponent,
            .operation = "update.count",
            .expected = "an accumulated count within int64",
            .actual = std::to_string(count_) + " + " + std::to_string(count),
            .hint = "Reset or publish the meter before continuing."});
  }
  const auto next_sum = sum_ + value * static_cast<double>(count);
  if (!std::isfinite(next_sum)) {
    throw NumericalError(
        "AverageMeter weighted sum became non-finite",
        ErrorDetails{.component = kComponent,
                     .operation = "update.sum",
                     .expected = "a finite weighted sum",
                     .actual = std::to_string(next_sum),
                     .hint = "Use finite bounded values or reset the meter."});
  }
  sum_ = next_sum;
  count_ += count;
  last_value_ = value;
}

void AverageMeter::update(const Tensor& value, int64_t count) {
  if (!value.defined() || value.numel() == 0) {
    throw_invalid("update.value", "a defined non-empty tensor", "empty",
                  "Provide at least one scalar observation.");
  }
  if (value.is_complex()) {
    throw_invalid("update.value", "a real tensor", "complex tensor",
                  "Convert the tensor to a real dtype.");
  }
  update(value.to(torch::kFloat64).mean().item<double>(), count);
}

double AverageMeter::average() const {
  return count_ == 0 ? 0.0 : sum_ / static_cast<double>(count_);
}

void AverageMeter::reset() {
  sum_ = 0.0;
  count_ = 0;
  last_value_ = 0.0;
}

}  // namespace metrics
}  // namespace training
}  // namespace jyppx::liuding
