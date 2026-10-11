#ifndef JYPPX_TRAINING_METRICS_METRIC_COLLECTION_HPP
#define JYPPX_TRAINING_METRICS_METRIC_COLLECTION_HPP

#include <jyppx/training/metric.hpp>

#include <map>
#include <memory>
#include <stdexcept>
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

    /** @brief Adds a metric; takes ownership of its name for lookups. */
    void add(std::shared_ptr<Metric> metric) {
        if (!metric) {
            throw std::invalid_argument("MetricCollection metric cannot be null");
        }
        if (metric->name().empty()) {
            throw std::invalid_argument("MetricCollection metric name cannot be empty");
        }
        if (!metrics_.emplace(metric->name(), metric).second) {
            throw std::invalid_argument("MetricCollection already contains '" + metric->name() + "'");
        }
        order_.push_back(metric->name());
    }

    /** @brief Access a metric by name. */
    std::shared_ptr<Metric> at(const std::string& name) const {
        const auto it = metrics_.find(name);
        if (it == metrics_.end()) {
            throw std::out_of_range("MetricCollection has no metric named '" + name + "'");
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

} // namespace metrics
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_METRICS_METRIC_COLLECTION_HPP
