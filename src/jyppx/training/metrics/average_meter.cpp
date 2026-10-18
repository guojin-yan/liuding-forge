#include <jyppx/foundation/error.hpp>
#include <jyppx/training/metrics/average_meter.hpp>

#include <string>
#include <utility>

namespace jyppx {
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
  sum_ += value * static_cast<double>(count);
  count_ += count;
  last_value_ = value;
}

void AverageMeter::update(const Tensor& value, int64_t count) {
  if (!value.defined() || value.numel() == 0) {
    throw_invalid("update.value", "a defined non-empty tensor", "empty",
                  "Provide at least one scalar observation.");
  }
  update(value.mean().item<double>(), count);
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
}  // namespace jyppx
