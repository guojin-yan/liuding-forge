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
 * Predictions must be a finite rank-2 `{batch, classes}` score tensor.
 * Targets must be finite integral class indices on the same device. If K
 * exceeds the number of classes, all classes participate in the comparison.
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
   * @param preds Defined finite scores with shape `{batch, classes}`.
   * @param targets Defined tensor containing one class index per batch row.
   * @throws InvalidArgument If inputs violate the documented shape, device,
   *     dtype, emptiness, label-count, or class-range contract.
   * @throws ResourceExhausted If the accumulated label count would overflow.
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
