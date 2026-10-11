#ifndef JYPPX_TRAINING_METRICS_AVERAGE_METER_HPP
#define JYPPX_TRAINING_METRICS_AVERAGE_METER_HPP

#include <jyppx/core/types.hpp>

namespace jyppx {
namespace training {
namespace metrics {

/**
 * @brief Running average of a scalar value, weighted per sample count
 *
 * Used for training-loop statistics such as the running epoch loss.
 */
class AverageMeter {
public:
    AverageMeter() = default;

    /** @brief Record one or more observations of `value`. */
    void update(double value, int64_t count = 1);

    /** @brief Record the mean of a tensor over all elements as one entry. */
    void update(const Tensor& value, int64_t count = 1);

    /** @brief Weighted mean over all recorded observations. */
    double average() const;

    /** @brief Sum of the recorded counts. */
    int64_t count() const { return count_; }

    /** @brief Most recent raw value. */
    double last_value() const { return last_value_; }

    void reset();

private:
    double sum_ = 0.0;
    int64_t count_ = 0;
    double last_value_ = 0.0;
};

} // namespace metrics
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_METRICS_AVERAGE_METER_HPP
