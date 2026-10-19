#ifndef JYPPX_TRAINING_METRICS_AVERAGE_METER_HPP
#define JYPPX_TRAINING_METRICS_AVERAGE_METER_HPP

#include <jyppx/core/types.hpp>

#include <cstdint>

namespace jyppx {
namespace training {
namespace metrics {

/**
 * @brief Maintains a sample-count-weighted running average of scalar values.
 *
 * This utility is intended for training-loop statistics such as epoch loss.
 * Instances are stateful and are not thread-safe.
 */
class AverageMeter {
 public:
  /** @brief Creates an empty meter whose average and last value are zero. */
  AverageMeter() = default;

  /**
   * @brief Records one scalar value with an observation count.
   * @param value Scalar mean for the represented observations.
   * @param count Number of observations represented by `value`.
   * @throws InvalidArgument If `count` is not positive.
   */
  void update(double value, int64_t count = 1);

  /**
   * @brief Records a tensor's elementwise mean with an observation count.
   * @param value Defined, non-empty tensor whose mean is recorded.
   * @param count Number of observations represented by the tensor mean.
   * @throws InvalidArgument If the tensor is empty or `count` is not positive.
   */
  void update(const Tensor& value, int64_t count = 1);

  /**
   * @brief Computes the weighted mean of all recorded values.
   * @return Weighted mean, or `0` when no values have been recorded.
   */
  double average() const;

  /**
   * @brief Returns the accumulated observation count.
   * @return Sum of all positive `count` arguments since reset().
   */
  int64_t count() const { return count_; }

  /**
   * @brief Returns the most recently recorded scalar value.
   * @return Last scalar or tensor mean, or `0` in the empty state.
   */
  double last_value() const { return last_value_; }

  /** @brief Restores the empty state. */
  void reset();

 private:
  /** Sum of each recorded value multiplied by its observation count. */
  double sum_ = 0.0;

  /** Sum of observation counts contributing to `sum_`. */
  int64_t count_ = 0;

  /** Most recently recorded scalar value or tensor mean. */
  double last_value_ = 0.0;
};

}  // namespace metrics
}  // namespace training
}  // namespace jyppx

#endif  // JYPPX_TRAINING_METRICS_AVERAGE_METER_HPP
