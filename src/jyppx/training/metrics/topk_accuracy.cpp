#include <jyppx/foundation/error.hpp>
#include <jyppx/training/metrics/topk_accuracy.hpp>

#include <algorithm>
#include <string>
#include <utility>

namespace jyppx {
namespace training {
namespace metrics {

namespace {

constexpr char kComponent[] = "training.metrics.topk_accuracy";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
    throw InvalidArgument(
        "Invalid TopKAccuracy metric contract",
        ErrorDetails{.component = kComponent,
                     .operation = std::move(operation),
                     .expected = std::move(expected),
                     .actual = std::move(actual),
                     .hint = std::move(hint)});
}

} // namespace

TopKAccuracy::TopKAccuracy(int k)
    : Metric("top" + std::to_string(k) + "_accuracy"), k_(k), correct_(0), total_(0) {
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
    if (preds.numel() == 0 || targets.numel() == 0) {
        return;
    }
    if (preds.dim() < 2) {
        throw_invalid("update.predictions.shape", "at least 2 dimensions",
                      std::to_string(preds.dim()) + " dimensions",
                      "Provide scores with a batch and class dimension.");
    }

    // Indices of the k largest scores per row.
    const int64_t classes = preds.size(1);
    const int64_t k = std::min<int64_t>(k_, classes);
    const auto topk = std::get<1>(preds.topk(k, /*dim=*/1));

    auto expected = targets.to(torch::kInt64).reshape({-1});
    if (expected.numel() != preds.size(0)) {
        throw_invalid("update.targets.shape",
                      std::to_string(preds.size(0)) + " labels",
                      std::to_string(expected.numel()),
                      "Provide one target for every prediction row.");
    }

    auto hits = topk.eq(expected.unsqueeze(1)).any(/*dim=*/1).to(torch::kInt64);
    correct_ += hits.sum().item<int64_t>();
    total_ += expected.numel();
}

double TopKAccuracy::compute() {
    return total_ == 0 ? 0.0
                       : static_cast<double>(correct_) / static_cast<double>(total_);
}

void TopKAccuracy::reset() {
    correct_ = 0;
    total_ = 0;
}

} // namespace metrics
} // namespace training
} // namespace jyppx
