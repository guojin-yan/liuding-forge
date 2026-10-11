#ifndef JYPPX_TRAINING_METRICS_CLASSIFICATION_HPP
#define JYPPX_TRAINING_METRICS_CLASSIFICATION_HPP

#include <jyppx/training/metric.hpp>

#include <string>
#include <vector>

namespace jyppx {
namespace training {
namespace metrics {

/**
 * @brief Base for confusion-matrix classification metrics
 *
 * Predictions are 1D scores (binarized with > 0) or 2D {batch, classes}
 * logits (argmax). The class count grows lazily to the largest class seen.
 * Exposes the accumulated confusion matrix; subclasses define the score.
 */
class ConfusionMatrixMetric : public Metric {
public:
    explicit ConfusionMatrixMetric(const std::string& name)
        : Metric(name) {}

    void update(const Tensor& preds, const Tensor& targets) override;
    void reset() override;

    /** @brief matrix_[true][predicted] counts; index = class id. */
    const std::vector<std::vector<int64_t>>& confusion_matrix() const {
        return matrix_;
    }

    /** @brief Number of classes currently tracked. */
    size_t classes() const { return matrix_.size(); }

protected:
    /// Per-class precision/recall; classes with no support are skipped.
    void compute_prf(std::vector<double>& precision,
                     std::vector<double>& recall) const;

    std::vector<std::vector<int64_t>> matrix_;
};

/**
 * @brief Precision, averaged over classes (macro)
 */
class Precision : public ConfusionMatrixMetric {
public:
    Precision() : ConfusionMatrixMetric("precision") {}
    double compute() override;
};

/**
 * @brief Recall, averaged over classes (macro)
 */
class Recall : public ConfusionMatrixMetric {
public:
    Recall() : ConfusionMatrixMetric("recall") {}
    double compute() override;
};

/**
 * @brief F1 score (harmonic mean of precision and recall), macro-averaged
 */
class F1Score : public ConfusionMatrixMetric {
public:
    F1Score() : ConfusionMatrixMetric("f1") {}
    double compute() override;
};

} // namespace metrics
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_METRICS_CLASSIFICATION_HPP
