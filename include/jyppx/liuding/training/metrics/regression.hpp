// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_TRAINING_METRICS_REGRESSION_HPP
#define JYPPX_LIUDING_TRAINING_METRICS_REGRESSION_HPP
#include <jyppx/liuding/training/metric.hpp>

#include <cstdint>

namespace jyppx::liuding {
namespace training {
namespace metrics {

/**
 * @brief Accumulates mean squared error over every observed tensor element.
 */
class MSE : public Metric {
 public:
  /** @brief Creates an empty metric named `mse`. */
  MSE() : Metric("mse"), squared_error_sum_(0.0), count_(0) {}

  /**
   * @brief Accumulates squared elementwise errors from one batch.
   * @param preds Defined finite real predictions; an empty pair is ignored.
   * @param targets Defined finite real targets on the same device with the
   *     same element count as `preds`.
   * @throws InvalidArgument If tensors have incompatible definition, device,
   *     dtype, emptiness, or element count.
   * @throws NumericalError If the accumulated squared error is not finite.
   * @throws ResourceExhausted If the element count would overflow.
   */
  void update(const Tensor& preds, const Tensor& targets) override;

  /**
   * @brief Computes the mean squared error across all observed elements.
   * @return Mean squared error, or `0` when no elements were observed.
   */
  double compute() override;

  /** @brief Clears the accumulated error and element count. */
  void reset() override;

 private:
  /** Sum of squared elementwise errors since the last reset. */
  double squared_error_sum_;

  /** Number of prediction elements included in the accumulated sum. */
  int64_t count_;
};

/**
 * @brief Accumulates mean absolute error over every observed tensor element.
 */
class MAE : public Metric {
 public:
  /** @brief Creates an empty metric named `mae`. */
  MAE() : Metric("mae"), absolute_error_sum_(0.0), count_(0) {}

  /**
   * @brief Accumulates absolute elementwise errors from one batch.
   * @param preds Defined finite real predictions; an empty pair is ignored.
   * @param targets Defined finite real targets on the same device with the
   *     same element count as `preds`.
   * @throws InvalidArgument If tensors have incompatible definition, device,
   *     dtype, emptiness, or element count.
   * @throws NumericalError If the accumulated absolute error is not finite.
   * @throws ResourceExhausted If the element count would overflow.
   */
  void update(const Tensor& preds, const Tensor& targets) override;

  /**
   * @brief Computes the mean absolute error across all observed elements.
   * @return Mean absolute error, or `0` when no elements were observed.
   */
  double compute() override;

  /** @brief Clears the accumulated error and element count. */
  void reset() override;

 private:
  /** Sum of absolute elementwise errors since the last reset. */
  double absolute_error_sum_;

  /** Number of prediction elements included in the accumulated sum. */
  int64_t count_;
};

/**
 * @brief Computes the coefficient of determination over accumulated samples.
 *
 * Target moments are accumulated across batches, so compute() returns the exact
 * data-set-level R-squared value rather than an average of per-batch scores.
 * Constant targets produce `1` for an exact fit and `0` otherwise.
 */
class R2Score : public Metric {
 public:
  /** @brief Creates an empty metric named `r2_score`. */
  R2Score()
      : Metric("r2_score"),
        residual_sum_(0.0),
        target_mean_(0.0),
        target_m2_(0.0),
        count_(0) {}

  /**
   * @brief Accumulates residuals and target moments from one batch.
   * @param preds Defined finite real predictions; an empty pair is ignored.
   * @param targets Defined finite real targets on the same device with the
   *     same element count as `preds`.
   * @throws InvalidArgument If tensors have incompatible definition, device,
   *     dtype, emptiness, or element count.
   * @throws NumericalError If a candidate accumulated moment is not finite.
   * @throws ResourceExhausted If the element count would overflow.
   */
  void update(const Tensor& preds, const Tensor& targets) override;

  /**
   * @brief Computes R-squared across all observed elements.
   * @return Coefficient of determination, or `0` for an empty metric.
   */
  double compute() override;

  /** @brief Clears all residual and target-moment accumulators. */
  void reset() override;

 private:
  /** Sum of squared prediction residuals. */
  double residual_sum_;

  /** Online mean of all observed target values. */
  double target_mean_;

  /** Sum of squared deviations from `target_mean_` using Welford merging. */
  double target_m2_;

  /** Number of target elements included in the moments. */
  int64_t count_;
};

}  // namespace metrics
}  // namespace training
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_TRAINING_METRICS_REGRESSION_HPP