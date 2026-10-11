#ifndef JYPPX_TRAINING_CALLBACKS_PROGRESS_CALLBACK_HPP
#define JYPPX_TRAINING_CALLBACKS_PROGRESS_CALLBACK_HPP

#include <jyppx/training/callback.hpp>

namespace jyppx {
namespace training {
namespace callbacks {

/**
 * @brief Progress bar callback
 */
class ProgressCallback : public Callback {
public:
    ProgressCallback() = default;

    void on_epoch_start(Trainer* trainer) override;
    void on_epoch_end(Trainer* trainer) override;
    void on_train_batch_end(Trainer* trainer, int batch_idx) override;
};

} // namespace callbacks
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_CALLBACKS_PROGRESS_CALLBACK_HPP
