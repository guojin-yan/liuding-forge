#ifndef JYPPX_TRAINING_METRICS_REGRESSION_HPP
#define JYPPX_TRAINING_METRICS_REGRESSION_HPP

#include <jyppx/training/metric.hpp>

namespace jyppx {
namespace training {
namespace metrics {

/**
 * @brief Mean squared error accumulated across batches
 */
class MSE : public Metric {
public:
    MSE() : Metric("mse"), squared_error_sum_(0.0), count_(0) {}

    void update(const Tensor& preds, const Tensor& targets) override;
    double compute() override;
    void reset() override;

private:
    double squared_error_sum_;
    int64_t count_;
};

/**
 * @brief Mean absolute error accumulated across batches
 */
class MAE : public Metric {
public:
    MAE() : Metric("mae"), absolute_error_sum_(0.0), count_(0) {}

    void update(const Tensor& preds, const Tensor& targets) override;
    double compute() override;
    void reset() override;

private:
    double absolute_error_sum_;
    int64_t count_;
};

/**
 * @brief Coefficient of determination (R^2) over all accumulated samples
 *
 * The total variance is computed from the accumulated targets, so the score
 * is the exact R^2 of the whole evaluation set rather than a mean of
 * per-batch scores.
 */
class R2Score : public Metric {
public:
    R2Score() : Metric("r2_score"), residual_sum_(0.0), target_sum_(0.0), target_sq_sum_(0.0), count_(0) {}

    void update(const Tensor& preds, const Tensor& targets) override;
    double compute() override;
    void reset() override;

private:
    double residual_sum_;
    double target_sum_;
    double target_sq_sum_;
    int64_t count_;
};

} // namespace metrics
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_METRICS_REGRESSION_HPP
