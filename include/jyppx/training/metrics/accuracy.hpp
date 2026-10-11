#ifndef JYPPX_TRAINING_METRICS_ACCURACY_HPP
#define JYPPX_TRAINING_METRICS_ACCURACY_HPP

#include <jyppx/training/metric.hpp>

namespace jyppx {
namespace training {
namespace metrics {

/**
 * @brief Accuracy metric
 */
class Accuracy : public Metric {
public:
    Accuracy() : Metric("accuracy"), correct_(0), total_(0) {}

    void update(const Tensor& preds, const Tensor& targets) override;
    double compute() override;
    void reset() override;

private:
    int64_t correct_;
    int64_t total_;
};

} // namespace metrics
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_METRICS_ACCURACY_HPP
