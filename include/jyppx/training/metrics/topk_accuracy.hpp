#ifndef JYPPX_TRAINING_METRICS_TOPK_ACCURACY_HPP
#define JYPPX_TRAINING_METRICS_TOPK_ACCURACY_HPP

#include <jyppx/training/metric.hpp>

namespace jyppx {
namespace training {
namespace metrics {

/**
 * @brief Top-K accuracy: the true class appears among the K highest logits
 *
 * Predictions must be 2D logits/scores of shape {batch, classes}; targets are
 * class indices (any shape that flattens to {batch}).
 */
class TopKAccuracy : public Metric {
public:
    explicit TopKAccuracy(int k);

    void update(const Tensor& preds, const Tensor& targets) override;
    double compute() override;
    void reset() override;

    int k() const { return k_; }

private:
    int k_;
    int64_t correct_;
    int64_t total_;
};

} // namespace metrics
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_METRICS_TOPK_ACCURACY_HPP
