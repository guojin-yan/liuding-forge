#ifndef JYPPX_TRAINING_METRICS_TOPK_ACCURACY_HPP
#define JYPPX_TRAINING_METRICS_TOPK_ACCURACY_HPP

#include <jyppx/training/metric.hpp>

#include <cstdint>

namespace jyppx {
namespace training {
namespace metrics {

/**
 * @brief Accumulates whether each target appears among the highest K scores.
 *
 * Predictions must contain a batch dimension followed by a class dimension.
 * Targets are flattened and converted to 64-bit class indices. If K exceeds
 * the number of classes, all classes participate in the comparison.
 */
class TopKAccuracy : public Metric {
 public:
  /**
   * @brief Creates an empty top-K accuracy metric.
   * @param k Positive maximum number of ranked class candidates.
   * @throws InvalidArgument If `k` is not positive.
   */
  explicit TopKAccuracy(int k);

  /**
   * @brief Accumulates top-K hits from one batch.
   * @param preds Defined scores with shape `{batch, classes, ...}`.
   * @param targets Defined tensor containing one class index per batch row.
   * @throws InvalidArgument If the inputs violate the documented shape
   * contract.
   */
  void update(const Tensor& preds, const Tensor& targets) override;

  /**
   * @brief Computes accumulated top-K accuracy.
   * @return Ratio in `[0, 1]`, or `0` when no labels have been observed.
   */
  double compute() override;

  /** @brief Clears the accumulated hit and total label counts. */
  void reset() override;

  /**
   * @brief Returns the configured rank cutoff.
   * @return Positive K value supplied to the constructor.
   */
  int k() const { return k_; }

 private:
  /** Maximum number of highest-scoring classes considered per prediction. */
  int k_;

  /** Number of targets found within their prediction's top-K classes. */
  int64_t correct_;

  /** Total number of target labels observed. */
  int64_t total_;
};

}  // namespace metrics
}  // namespace training
}  // namespace jyppx

#endif  // JYPPX_TRAINING_METRICS_TOPK_ACCURACY_HPP
