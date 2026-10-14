#ifndef JYPPX_TRAINING_CALLBACK_HPP
#define JYPPX_TRAINING_CALLBACK_HPP

#include <jyppx/core/types.hpp>

#include <map>
#include <string>

namespace jyppx {
namespace training {

/** @brief Forward declaration for lifecycle callback parameters. */
class Trainer;

/**
 * @brief Portable scalar and text state stored in a Trainer checkpoint.
 *
 * Callback implementations own the schema associated with their state key.
 * Tensor payloads are intentionally excluded from this lightweight legacy
 * checkpoint contract.
 */
struct CallbackState {
  std::map<std::string, double> scalars;       ///< Named numeric state fields.
  std::map<std::string, std::string> strings;  ///< Named UTF-8 text fields.
};

/**
 * @brief Observes Trainer lifecycle events and optionally persists state.
 *
 * Callback objects are shared with the Trainer but remain owned by their
 * `std::shared_ptr` holders. Callbacks are invoked serially by one training
 * thread and are not required to be thread-safe.
 */
class Callback {
 public:
  /** @brief Destroys a callback through the base interface. */
  virtual ~Callback() = default;

  /** @brief Runs once before the first training epoch. @param trainer Active
   * trainer. */
  virtual void on_fit_start(Trainer* trainer) {}

  /** @brief Runs once after the fit loop exits. @param trainer Active trainer.
   */
  virtual void on_fit_end(Trainer* trainer) {}

  /** @brief Runs before an epoch starts. @param trainer Active trainer. */
  virtual void on_epoch_start(Trainer* trainer) {}

  /** @brief Runs after training and validation for an epoch. @param trainer
   * Active trainer. */
  virtual void on_epoch_end(Trainer* trainer) {}

  /** @brief Runs before the training batches of an epoch. @param trainer Active
   * trainer. */
  virtual void on_train_epoch_start(Trainer* trainer) {}

  /** @brief Runs after the training batches of an epoch. @param trainer Active
   * trainer. */
  virtual void on_train_epoch_end(Trainer* trainer) {}

  /** @brief Runs before validation batches. @param trainer Active trainer. */
  virtual void on_validation_epoch_start(Trainer* trainer) {}

  /** @brief Runs after validation batches. @param trainer Active trainer. */
  virtual void on_validation_epoch_end(Trainer* trainer) {}

  /**
   * @brief Runs before one training batch.
   * @param trainer Active trainer.
   * @param batch_idx Zero-based batch index in the current epoch.
   */
  virtual void on_train_batch_start(Trainer* trainer, int batch_idx) {}

  /**
   * @brief Runs after one training batch.
   * @param trainer Active trainer.
   * @param batch_idx Zero-based batch index in the current epoch.
   * @param loss Detached scalar loss for this batch.
   */
  virtual void on_train_batch_end(Trainer* trainer, int batch_idx,
                                  double loss) {}

  /**
   * @brief Runs before one validation batch.
   * @param trainer Active trainer.
   * @param batch_idx Zero-based batch index in the validation pass.
   */
  virtual void on_validation_batch_start(Trainer* trainer, int batch_idx) {}

  /**
   * @brief Runs after one validation batch.
   * @param trainer Active trainer.
   * @param batch_idx Zero-based batch index in the validation pass.
   */
  virtual void on_validation_batch_end(Trainer* trainer, int batch_idx) {}

  /**
   * @brief Stable identifier for checkpointable callback state.
   *
   * An empty identifier marks a stateless callback. Stateful callbacks with
   * the same identifier are matched by their order in the callback list.
   */
  virtual std::string state_key() const { return {}; }

  /**
   * @brief Returns state associated with state_key().
   * @return Portable state; stateless callbacks return an empty value.
   */
  virtual CallbackState state_dict() const { return {}; }

  /**
   * @brief Restores a state produced by state_dict().
   * @param state Candidate state to validate and apply transactionally.
   * @throws std::invalid_argument The state schema or configuration differs.
   */
  virtual void load_state_dict(const CallbackState& state) { (void)state; }
};

}  // namespace training
}  // namespace jyppx

#endif  // JYPPX_TRAINING_CALLBACK_HPP
