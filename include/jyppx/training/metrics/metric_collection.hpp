#ifndef JYPPX_TRAINING_METRICS_METRIC_COLLECTION_HPP
#define JYPPX_TRAINING_METRICS_METRIC_COLLECTION_HPP

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/metric.hpp>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace jyppx {
namespace training {
namespace metrics {

/**
 * @brief Bundle of named metrics updated and computed together
 *
 * Every contained metric receives the same predictions/targets; compute()
 * returns a metric-name -> value map. Duplicated metric names are rejected.
 */
class MetricCollection {
 public:
  MetricCollection() = default;

  /**
   * @brief Adds a metric and retains it under its stable name.
   * @param metric Metric instance to retain; must not be null.
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
   * @brief Accesses a metric by its stable name.
   * @param name Name registered with add().
   * @return The retained metric instance.
   * @throws InvalidArgument If name is not registered.
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

  void update(const Tensor& preds, const Tensor& targets) {
    for (const auto& [name, metric] : metrics_) {
      metric->update(preds, targets);
    }
  }

  /** @brief Computes every metric, in insertion order. */
  std::map<std::string, double> compute() const {
    std::map<std::string, double> values;
    for (const auto& name : order_) {
      values.emplace(name, metrics_.at(name)->compute());
    }
    return values;
  }

  void reset() {
    for (const auto& [name, metric] : metrics_) {
      metric->reset();
    }
  }

  /** @brief Number of metrics in the collection. */
  size_t size() const { return metrics_.size(); }

 private:
  std::map<std::string, std::shared_ptr<Metric>> metrics_;
  std::vector<std::string> order_;
};

}  // namespace metrics
}  // namespace training
}  // namespace jyppx

#endif  // JYPPX_TRAINING_METRICS_METRIC_COLLECTION_HPP
