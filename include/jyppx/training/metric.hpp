#ifndef JYPPX_TRAINING_METRIC_HPP
#define JYPPX_TRAINING_METRIC_HPP

#include <jyppx/core/types.hpp>
#include <string>

namespace jyppx {
namespace training {

/**
 * @brief Base class for all metrics
 */
class Metric {
public:
    explicit Metric(const std::string& name) : name_(name) {}
    virtual ~Metric() = default;

    /**
     * @brief Update metric with predictions and targets
     */
    virtual void update(const Tensor& preds, const Tensor& targets) = 0;

    /**
     * @brief Compute final metric value
     */
    virtual double compute() = 0;

    /**
     * @brief Reset metric state
     */
    virtual void reset() = 0;

    /**
     * @brief Get metric name
     */
    const std::string& name() const { return name_; }

protected:
    std::string name_;
};

} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_METRIC_HPP
