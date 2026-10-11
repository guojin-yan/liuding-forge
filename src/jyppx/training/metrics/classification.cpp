#include <jyppx/training/metrics/classification.hpp>

#include <stdexcept>

namespace jyppx {
namespace training {
namespace metrics {

namespace {

// Extracts predicted and true class indices using the same conventions as
// Accuracy (1D scores binarized with > 0, 2D scores via argmax).
std::pair<Tensor, Tensor> class_pair(const Tensor& preds, const Tensor& targets) {
    if (!preds.defined() || !targets.defined()) {
        throw std::invalid_argument("Metric inputs must be defined");
    }
    if (preds.numel() == 0 || targets.numel() == 0) {
        return {};
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
        throw std::invalid_argument("Metric prediction and target sizes differ");
    }
    return {predicted.contiguous(), expected.contiguous()};
}

int64_t max_class(const Tensor& t) {
    return t.max().item<int64_t>();
}

} // namespace

void ConfusionMatrixMetric::update(const Tensor& preds, const Tensor& targets) {
    const auto [predicted, expected] = class_pair(preds, targets);
    if (!predicted.defined()) {
        return;
    }

    const int64_t highest = std::max(max_class(predicted), max_class(expected));
    const size_t needed = static_cast<size_t>(highest) + 1;
    if (matrix_.size() < needed) {
        matrix_.resize(needed, std::vector<int64_t>(needed, 0));
        for (auto& row : matrix_) {
            row.resize(needed, 0);
        }
    }

    const auto p = predicted.accessor<int64_t, 1>();
    const auto t = expected.accessor<int64_t, 1>();
    for (int64_t i = 0; i < predicted.numel(); ++i) {
        ++matrix_[static_cast<size_t>(t[i])][static_cast<size_t>(p[i])];
    }
}

void ConfusionMatrixMetric::reset() {
    matrix_.clear();
}

void ConfusionMatrixMetric::compute_prf(std::vector<double>& precision,
                                        std::vector<double>& recall) const {
    precision.clear();
    recall.clear();
    for (size_t c = 0; c < matrix_.size(); ++c) {
        int64_t predicted_as_c = 0;
        int64_t actual_c = 0;
        for (size_t r = 0; r < matrix_.size(); ++r) {
            predicted_as_c += matrix_[r][c];
            actual_c += matrix_[c][r];
        }
        precision.push_back(predicted_as_c == 0
                                ? 0.0
                                : static_cast<double>(matrix_[c][c]) /
                                      static_cast<double>(predicted_as_c));
        recall.push_back(actual_c == 0
                             ? 0.0
                             : static_cast<double>(matrix_[c][c]) /
                                   static_cast<double>(actual_c));
    }
}

double Precision::compute() {
    std::vector<double> precision, recall;
    compute_prf(precision, recall);
    if (precision.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const double p : precision) {
        sum += p;
    }
    return sum / static_cast<double>(precision.size());
}

double Recall::compute() {
    std::vector<double> precision, recall;
    compute_prf(precision, recall);
    if (recall.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const double r : recall) {
        sum += r;
    }
    return sum / static_cast<double>(recall.size());
}

double F1Score::compute() {
    std::vector<double> precision, recall;
    compute_prf(precision, recall);
    if (precision.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (size_t c = 0; c < precision.size(); ++c) {
        const double p = precision[c];
        const double r = recall[c];
        sum += (p + r) == 0.0 ? 0.0 : 2.0 * p * r / (p + r);
    }
    return sum / static_cast<double>(precision.size());
}

} // namespace metrics
} // namespace training
} // namespace jyppx
