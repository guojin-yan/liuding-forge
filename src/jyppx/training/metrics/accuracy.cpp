#include <jyppx/training/metrics/accuracy.hpp>

#include <stdexcept>

namespace jyppx {
namespace training {
namespace metrics {

void Accuracy::update(const Tensor& preds, const Tensor& targets) {
    if (!preds.defined() || !targets.defined()) {
        throw std::invalid_argument("Accuracy inputs must be defined");
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
        throw std::invalid_argument("Accuracy prediction and target sizes differ");
    }
    correct_ += predicted.eq(expected).sum().item<int64_t>();
    total_ += expected.numel();
}

double Accuracy::compute() {
    return total_ == 0 ? 0.0 : static_cast<double>(correct_) / static_cast<double>(total_);
}

void Accuracy::reset() {
    correct_ = 0;
    total_ = 0;
}

} // namespace metrics
} // namespace training
} // namespace jyppx
