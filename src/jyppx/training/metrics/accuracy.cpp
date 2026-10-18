#include <jyppx/foundation/error.hpp>
#include <jyppx/training/metrics/accuracy.hpp>

#include <string>
#include <utility>

namespace jyppx {
namespace training {
namespace metrics {

namespace {

constexpr char kComponent[] = "training.metrics.accuracy";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid accuracy metric contract",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

}  // namespace

void Accuracy::update(const Tensor& preds, const Tensor& targets) {
  if (!preds.defined() || !targets.defined()) {
    throw_invalid("update.inputs", "defined prediction and target tensors",
                  "undefined", "Provide both tensors.");
  }
  if (preds.numel() == 0 || targets.numel() == 0) {
    return;
  }

  Tensor predicted;
  if (preds.dim() >= 2) {
    predicted = preds.argmax(1);
  } else {
    predicted = (preds > 0).to(torch::kInt64);
  }
  auto expected = targets.to(torch::kInt64).reshape({-1});
  predicted = predicted.reshape({-1});
  if (predicted.numel() != expected.numel()) {
    throw_invalid("update.targets.shape",
                  std::to_string(predicted.numel()) + " labels",
                  std::to_string(expected.numel()),
                  "Provide one target for every prediction.");
  }
  correct_ += predicted.eq(expected).sum().item<int64_t>();
  total_ += expected.numel();
}

double Accuracy::compute() {
  return total_ == 0
             ? 0.0
             : static_cast<double>(correct_) / static_cast<double>(total_);
}

void Accuracy::reset() {
  correct_ = 0;
  total_ = 0;
}

}  // namespace metrics
}  // namespace training
}  // namespace jyppx
