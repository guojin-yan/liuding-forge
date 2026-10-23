// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_TRAINING_METRIC_HPP
#define JYPPX_LIUDING_TRAINING_METRIC_HPP
#include <jyppx/liuding/core/types.hpp>

#include <string>

namespace jyppx::liuding {
namespace training {

/**
 * @brief Stateful interface for metrics accumulated over one or more batches.
 *
 * A metric has a stable, non-localized name used by collections, logs, and
 * checkpoint metadata. Implementations accumulate observations in update(),
 * expose the aggregate through compute(), and return to their initial state in
 * reset(). Metric instances are not required to be thread-safe.
 */
class Metric {
 public:
  /**
   * @brief Creates a metric with the supplied stable name.
   * @param name Non-localized identifier used for lookup and reporting.
   */
  explicit Metric(const std::string& name) : name_(name) {}

  /** @brief Destroys the metric through its polymorphic interface. */
  virtual ~Metric() = default;

  /**
   * @brief Accumulates one prediction/target batch.
   * @param preds Prediction values interpreted by the concrete metric.
   * @param targets Reference values interpreted by the concrete metric.
   */
  virtual void update(const Tensor& preds, const Tensor& targets) = 0;

  /**
   * @brief Computes the aggregate value for all observations since reset().
   * @return Metric-specific scalar value, or the implementation's documented
   *     empty-state value when no observations have been accumulated.
   */
  virtual double compute() = 0;

  /** @brief Removes every accumulated observation. */
  virtual void reset() = 0;

  /**
   * @brief Returns the metric's stable lookup name.
   * @return Reference valid for the lifetime of this metric.
   */
  const std::string& name() const { return name_; }

 protected:
  /** Stable non-localized identifier exposed by name(). */
  std::string name_;
};

}  // namespace training
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_TRAINING_METRIC_HPP