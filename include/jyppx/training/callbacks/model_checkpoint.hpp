#ifndef JYPPX_TRAINING_CALLBACKS_MODEL_CHECKPOINT_HPP
#define JYPPX_TRAINING_CALLBACKS_MODEL_CHECKPOINT_HPP

#include <jyppx/training/callback.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace jyppx {
namespace training {
namespace callbacks {

/**
 * @brief Saves the best K complete Trainer checkpoints at epoch boundaries.
 *
 * File publication is deferred until every epoch-end callback has run, so the
 * archive contains a consistent callback state. The callback is stateful and
 * not thread-safe.
 */
class ModelCheckpoint : public Callback {
 public:
  /**
   * @brief Creates a checkpoint-retention callback.
   * @param filepath Destination path; K greater than one adds an epoch suffix.
   * @param monitor `train_loss` or `val_loss`.
   * @param mode `min` keeps smaller values and `max` keeps larger values.
   * @param save_top_k Number of retained checkpoints; zero disables saving
   *     and -1 saves every epoch.
   * @throws std::invalid_argument An option is empty, unsupported, or below -1.
   */
  ModelCheckpoint(const std::string& filepath,
                  const std::string& monitor = "val_loss",
                  const std::string& mode = "min", int save_top_k = 1);

  /** @brief Evaluates retention after an epoch. @param trainer Active trainer.
   */
  void on_epoch_end(Trainer* trainer) override;

  /** @brief Clears retained paths, score history, and the epoch counter. */
  void reset();

  /** @brief Returns the stable checkpoint schema key. */
  std::string state_key() const override { return "model_checkpoint"; }

  /** @brief Serializes retention configuration and mutable history. */
  CallbackState state_dict() const override;

  /**
   * @brief Validates and restores state produced by state_dict().
   * @param state Candidate callback state.
   * @throws std::invalid_argument State is incomplete or configuration differs.
   */
  void load_state_dict(const CallbackState& state) override;

 private:
  std::string filepath_;  ///< Base destination path for published checkpoints.
  std::string monitor_;   ///< Trainer loss name used for ranking.
  std::string mode_;      ///< Ranking direction, either `min` or `max`.
  int save_top_k_;  ///< Retention count; -1 means all and zero means disabled.
  double best_score_;  ///< Best retained metric, or mode-specific infinity.
  std::vector<std::pair<double, std::string>>
      saved_;                ///< Ranked score/path pairs.
  int64_t epoch_index_ = 0;  ///< One-based suffix source for candidate paths.
};

}  // namespace callbacks
}  // namespace training
}  // namespace jyppx

#endif  // JYPPX_TRAINING_CALLBACKS_MODEL_CHECKPOINT_HPP
