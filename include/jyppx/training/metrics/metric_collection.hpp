#ifndef JYPPX_TRAINING_METRICS_METRIC_COLLECTION_HPP
#define JYPPX_TRAINING_METRICS_METRIC_COLLECTION_HPP

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/metric.hpp>

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace jyppx {
namespace training {
namespace metrics {

/**
 * @brief Owns named metrics that receive the same prediction/target batches.
 *
 * Metric names are unique stable keys. compute() preserves insertion order
 * while constructing its ordered map result. The collection and retained
 * metrics are stateful and are not thread-safe.
 */
class MetricCollection {
 public:
  /** @brief Creates an empty collection. */
  MetricCollection() = default;

  /**
   * @brief Adds and retains a metric under its stable name.
   * @param metric Metric instance to retain.
   * @throws InvalidArgument If the metric is null, unnamed, or duplicated.
   */
  void add(std::shared_ptr<Metric> metric) {
    if (!metric) {
      throw InvalidArgument(
          "MetricCollection metric cannot be null",
          ErrorDetails{.component = "training.metrics.collection",
                       .operation = "add.metric",
                       .expected = "a non-null metric",
                       .actual = "null",
                       .hint = "Construct the metric before adding it."});
    }
    if (metric->name().empty()) {
      throw InvalidArgument(
          "MetricCollection metric name cannot be empty",
          ErrorDetails{.component = "training.metrics.collection",
                       .operation = "add.metric.name",
                       .expected = "a non-empty stable name",
                       .actual = "empty",
                       .hint = "Give each metric a stable lookup name."});
    }
    if (!metrics_.emplace(metric->name(), metric).second) {
      throw InvalidArgument(
          "MetricCollection already contains '" + metric->name() + "'",
          ErrorDetails{
              .component = "training.metrics.collection",
              .operation = "add.metric.name",
              .expected = "a unique metric name",
              .actual = metric->name(),
              .hint = "Use a distinct name or replace the collection entry."});
    }
    order_.push_back(metric->name());
  }

  /**
   * @brief Accesses a retained metric by stable name.
   * @param name Name registered with add().
   * @return Shared ownership of the retained metric instance.
   * @throws InvalidArgument If `name` is not registered.
   */
  std::shared_ptr<Metric> at(const std::string& name) const {
    const auto it = metrics_.find(name);
    if (it == metrics_.end()) {
      throw InvalidArgument(
          "MetricCollection has no metric named '" + name + "'",
          ErrorDetails{.component = "training.metrics.collection",
                       .operation = "at.name",
                       .expected = "a registered metric name",
                       .actual = name,
                       .hint = "Register the metric before accessing it."});
    }
    return it->second;
  }

  /**
   * @brief Sends one prediction/target batch to every retained metric.
   * @param preds Prediction tensor forwarded without conversion.
   * @param targets Target tensor forwarded without conversion.
   */
  void update(const Tensor& preds, const Tensor& targets) {
    for (const auto& entry : metrics_) {
      entry.second->update(preds, targets);
    }
  }

  /**
   * @brief Computes every retained metric.
   * @return Stable-name-to-value map containing one entry per metric.
   */
  std::map<std::string, double> compute() const {
    std::map<std::string, double> values;
    for (const auto& name : order_) {
      values.emplace(name, metrics_.at(name)->compute());
    }
    return values;
  }

  /** @brief Resets every retained metric without removing it. */
  void reset() {
    for (const auto& entry : metrics_) {
      entry.second->reset();
    }
  }

  /**
   * @brief Returns the number of retained metrics.
   * @return Collection size.
   */
  std::size_t size() const { return metrics_.size(); }

 private:
  /** Metrics indexed by unique stable name. */
  std::map<std::string, std::shared_ptr<Metric>> metrics_;

  /** Stable names in insertion order. */
  std::vector<std::string> order_;
};

}  // namespace metrics
}  // namespace training
}  // namespace jyppx

#endif  // JYPPX_TRAINING_METRICS_METRIC_COLLECTION_HPP
