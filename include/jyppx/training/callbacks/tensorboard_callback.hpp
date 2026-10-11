#ifndef JYPPX_TRAINING_CALLBACKS_TENSORBOARD_CALLBACK_HPP
#define JYPPX_TRAINING_CALLBACKS_TENSORBOARD_CALLBACK_HPP

#include <jyppx/training/callback.hpp>
#include <jyppx/utils/tensorboard.hpp>

#include <memory>
#include <string>

namespace jyppx {
namespace training {
namespace callbacks {

/**
 * @brief Writes training curves to a TensorBoard event file
 *
 * Logs the batch loss every log_every_n_steps steps, the per-epoch mean
 * train/validation losses, and the optimizer learning rates at epoch
 * boundaries.
 */
class TensorBoardCallback : public Callback {
public:
    TensorBoardCallback(const std::string& log_dir, int log_every_n_steps = 50);

    void on_fit_start(Trainer* trainer) override;
    void on_train_batch_end(Trainer* trainer, int batch_idx, double loss) override;
    void on_epoch_end(Trainer* trainer) override;
    void on_fit_end(Trainer* trainer) override;

    /** @brief The underlying writer (for custom scalars). */
    utils::TensorBoardWriter& writer() { return *writer_; }

private:
    std::unique_ptr<utils::TensorBoardWriter> writer_;
    int log_every_n_steps_;
};

} // namespace callbacks
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_CALLBACKS_TENSORBOARD_CALLBACK_HPP
