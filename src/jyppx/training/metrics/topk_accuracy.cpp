#include <jyppx/foundation/error.hpp>
#include <jyppx/training/metrics/topk_accuracy.hpp>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace jyppx {
namespace training {
namespace metrics {

namespace {

constexpr char kComponent[] = "training.metrics.topk_accuracy";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid TopKAccuracy metric contract",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_resource(std::int64_t current, std::int64_t incoming) {
  throw ResourceExhausted(
      "TopKAccuracy label count exceeds its supported range",
      ErrorDetails{
          .component = kComponent,
          .operation = "update.total",
          .expected = "an accumulated count within int64",
          .actual = std::to_string(current) + " + " + std::to_string(incoming),
          .hint = "Reset or publish the metric before continuing."});
}

void validate_finite_real(const Tensor& value, const char* operation,
                          const char* description) {
  if (value.is_complex()) {
    throw_invalid(operation, std::string("a real ") + description,
                  "complex tensor", "Convert the tensor to a real dtype.");
  }
  if (value.is_floating_point() && !torch::isfinite(value).all().item<bool>()) {
    throw_invalid(operation, std::string("a finite ") + description,
                  "NaN or Inf", "Remove non-finite values before updating.");
  }
}

}  // namespace

TopKAccuracy::TopKAccuracy(int k)
    : Metric("top" + std::to_string(k) + "_accuracy"),
      k_(k),
      correct_(0),
      total_(0) {
  if (k_ <= 0) {
    throw_invalid("options.k", "a positive integer", std::to_string(k_),
                  "Choose k greater than zero.");
  }
}

void TopKAccuracy::update(const Tensor& preds, const Tensor& targets) {
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
    return;
  }
  if (preds.dim() != 2) {
    throw_invalid("update.predictions.shape", "exactly 2 dimensions",
                  std::to_string(preds.dim()) + " dimensions",
                  "Provide a {batch, classes} score matrix.");
  }
  if (preds.device() != targets.device()) {
    throw_invalid("update.inputs.device", "matching tensor devices",
                  preds.device().str() + " vs " + targets.device().str(),
                  "Move predictions and targets to the same device.");
  }
  validate_finite_real(preds, "update.predictions.dtype", "prediction tensor");
  validate_finite_real(targets, "update.targets.dtype", "class-index tensor");
  // Indices of the k largest scores per row.
  const int64_t classes = preds.size(1);
  if (classes <= 0) {
    throw_invalid("update.predictions.shape", "a non-empty class dimension",
                  std::to_string(classes) + " classes",
                  "Provide at least one class score.");
  }
  if (targets.is_floating_point()) {
    if (!torch::equal(targets, torch::trunc(targets))) {
      throw_invalid("update.targets.dtype", "integral class indices",
                    "fractional values",
                    "Round or encode labels as integer class indices.");
    }
    const auto minimum = targets.min().item<double>();
    const auto maximum = targets.max().item<double>();
    if (minimum < 0.0 || maximum >= static_cast<double>(classes)) {
      throw_invalid("update.targets.range",
                    "class indices in [0, " + std::to_string(classes) + ")",
                    "floating-point value outside the class range",
                    "Encode every target within the prediction class range.");
    }
  }
  const int64_t k = std::min<int64_t>(k_, classes);
  const auto topk = std::get<1>(preds.topk(k, /*dim=*/1));

  auto expected = targets.to(torch::kInt64).reshape({-1});
  if (expected.numel() != preds.size(0)) {
    throw_invalid("update.targets.shape",
                  std::to_string(preds.size(0)) + " labels",
                  std::to_string(expected.numel()),
                  "Provide one target for every prediction row.");
  }
  const auto minimum = expected.min().item<std::int64_t>();
  const auto maximum = expected.max().item<std::int64_t>();
  if (minimum < 0 || maximum >= classes) {
    throw_invalid(
        "update.targets.range",
        "class indices in [0, " + std::to_string(classes) + ")",
        "min=" + std::to_string(minimum) + ", max=" + std::to_string(maximum),
        "Encode every target within the prediction class range.");
  }
  const auto incoming = expected.numel();
  if (incoming > std::numeric_limits<std::int64_t>::max() - total_) {
    throw_resource(total_, incoming);
  }

  auto hits = topk.eq(expected.unsqueeze(1)).any(/*dim=*/1).to(torch::kInt64);
  const auto batch_correct = hits.sum().item<std::int64_t>();
  correct_ += batch_correct;
  total_ += incoming;
}

double TopKAccuracy::compute() {
  return total_ == 0
             ? 0.0
             : static_cast<double>(correct_) / static_cast<double>(total_);
}

void TopKAccuracy::reset() {
  correct_ = 0;
  total_ = 0;
}

}  // namespace metrics
}  // namespace training
}  // namespace jyppx
