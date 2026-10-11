#ifndef JYPPX_TRAINING_CALLBACKS_LOGGING_CALLBACK_HPP
#define JYPPX_TRAINING_CALLBACKS_LOGGING_CALLBACK_HPP

#include <jyppx/training/callback.hpp>

#include <string>

namespace jyppx {
namespace training {
namespace callbacks {

/**
 * @brief Logs per-batch and per-epoch training progress
 *
 * Every log_every_n_steps-th batch (and every epoch boundary) writes a line
 * through the framework logger, optionally to a log file as well.
 */
class LoggingCallback : public Callback {
public:
    LoggingCallback(int log_every_n_steps = 50, std::string log_file = "");

    void on_fit_start(Trainer* trainer) override;
    void on_train_batch_end(Trainer* trainer, int batch_idx, double loss) override;
    void on_epoch_end(Trainer* trainer) override;
    void on_fit_end(Trainer* trainer) override;

    int log_every_n_steps() const { return log_every_n_steps_; }
    const std::string& log_file() const { return log_file_; }

private:
    int log_every_n_steps_;
    std::string log_file_;
    int64_t global_step_ = 0;
};

/**
 * @brief Logs the learning rate at epoch boundaries
 *
 * Reads the current learning rate(s) from the trainer's optimizer and emits
 * one line per epoch.
 */
class LRMonitor : public Callback {
public:
    void on_epoch_start(Trainer* trainer) override;
};

} // namespace callbacks
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_CALLBACKS_LOGGING_CALLBACK_HPP
