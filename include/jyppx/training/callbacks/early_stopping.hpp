#ifndef JYPPX_TRAINING_CALLBACKS_EARLY_STOPPING_HPP
#define JYPPX_TRAINING_CALLBACKS_EARLY_STOPPING_HPP

#include <jyppx/training/callback.hpp>

#include <string>

namespace jyppx {
namespace training {
namespace callbacks {

/**
 * @brief Stops training after a monitored loss stops improving.
 *
 * The callback is stateful and not thread-safe. Trainer checkpoints preserve
 * its best score, patience counter, and stop decision.
 */
class EarlyStopping : public Callback {
 public:
  /**
   * @brief Creates an early-stopping callback.
   * @param monitor `train_loss` or `val_loss`.
   * @param patience Number of non-improving epochs allowed; zero stops on
   *     the first non-improving observation.
   * @param min_delta Minimum absolute improvement, greater than or equal to
   * zero.
   * @param mode `min` for loss minimization or `max` for score maximization.
   * @throws std::invalid_argument An option is unsupported or out of range.
   */
  EarlyStopping(const std::string& monitor = "val_loss", int patience = 10,
                double min_delta = 0.0, const std::string& mode = "min");

  /** @brief Updates patience from the completed epoch. @param trainer Active
   * trainer. */
  void on_epoch_end(Trainer* trainer) override;

  /** @brief Returns whether this callback has requested training to stop. */
  bool should_stop() const { return should_stop_; }

  /** @brief Clears the best score, patience counter, and stop decision. */
  void reset();

  /** @brief Returns the stable checkpoint schema key. */
  std::string state_key() const override { return "early_stopping"; }

  /** @brief Serializes configuration identity and mutable callback state. */
  CallbackState state_dict() const override;

  /**
   * @brief Validates and restores a state produced by state_dict().
   * @param state Candidate callback state.
   * @throws std::invalid_argument State is incomplete or configuration differs.
   */
  void load_state_dict(const CallbackState& state) override;

 private:
  std::string monitor_;  ///< Trainer loss name observed at each epoch boundary.
  int patience_;  ///< Allowed consecutive non-improving epochs; non-negative.
  double min_delta_;   ///< Required absolute improvement; non-negative.
  std::string mode_;   ///< Comparison direction, either `min` or `max`.
  int wait_count_;     ///< Consecutive non-improving epoch count; non-negative.
  double best_score_;  ///< Best observed score, or mode-specific infinity
                       ///< initially.
  bool should_stop_;   ///< Whether the patience limit has been reached.
};

}  // namespace callbacks
}  // namespace training
}  // namespace jyppx

#endif  // JYPPX_TRAINING_CALLBACKS_EARLY_STOPPING_HPP
