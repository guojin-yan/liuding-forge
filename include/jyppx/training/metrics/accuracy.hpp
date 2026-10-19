#ifndef JYPPX_TRAINING_METRICS_ACCURACY_HPP
#define JYPPX_TRAINING_METRICS_ACCURACY_HPP

#include <jyppx/training/metric.hpp>

#include <cstdint>

namespace jyppx {
namespace training {
namespace metrics {

/**
 * @brief Accumulates exact classification accuracy across batches.
 *
 * One-dimensional predictions are converted to labels with a zero threshold.
 * Predictions with two or more dimensions use argmax over dimension 1. Targets
 * are flattened and converted to 64-bit class indices before comparison.
 */
class Accuracy : public Metric {
 public:
  /** @brief Creates an empty metric named `accuracy`. */
  Accuracy() : Metric("accuracy"), correct_(0), total_(0) {}

  /**
   * @brief Accumulates correct predictions from one batch.
   * @param preds Defined prediction tensor; empty tensors are ignored.
   * @param targets Defined target tensor with one label per prediction.
   * @throws InvalidArgument If an input is undefined or label counts differ.
   */
  void update(const Tensor& preds, const Tensor& targets) override;

  /**
   * @brief Computes accumulated exact accuracy.
   * @return Ratio in `[0, 1]`, or `0` when no labels have been observed.
   */
  double compute() override;

  /** @brief Clears the accumulated correct and total label counts. */
  void reset() override;

 private:
  /** Number of labels predicted correctly since the last reset. */
  int64_t correct_;

  /** Total number of labels observed since the last reset. */
  int64_t total_;
};

}  // namespace metrics
}  // namespace training
}  // namespace jyppx

#endif  // JYPPX_TRAINING_METRICS_ACCURACY_HPP
