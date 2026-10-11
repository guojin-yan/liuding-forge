#include <jyppx/training/metrics/topk_accuracy.hpp>

#include <stdexcept>

namespace jyppx {
namespace training {
namespace metrics {

TopKAccuracy::TopKAccuracy(int k)
    : Metric("top" + std::to_string(k) + "_accuracy"), k_(k), correct_(0), total_(0) {
    if (k_ <= 0) {
        throw std::invalid_argument("TopKAccuracy k must be positive");
    }
}

void TopKAccuracy::update(const Tensor& preds, const Tensor& targets) {
    if (!preds.defined() || !targets.defined()) {
        throw std::invalid_argument("TopKAccuracy inputs must be defined");
    }
    if (preds.numel() == 0 || targets.numel() == 0) {
        return;
    }
    if (preds.dim() < 2) {
        throw std::invalid_argument(
            "TopKAccuracy predictions must be {batch, classes} scores");
    }

    // Indices of the k largest scores per row.
    const int64_t classes = preds.size(1);
    const int64_t k = std::min<int64_t>(k_, classes);
    const auto topk = std::get<1>(preds.topk(k, /*dim=*/1));

    auto expected = targets.to(torch::kInt64).reshape({-1});
    if (expected.numel() != preds.size(0)) {
        throw std::invalid_argument("TopKAccuracy prediction and target sizes differ");
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
