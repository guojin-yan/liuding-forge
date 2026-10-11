#include <jyppx/training/callbacks/progress_callback.hpp>
#include <jyppx/training/trainer.hpp>

#include <iostream>

namespace jyppx {
namespace training {
namespace callbacks {

void ProgressCallback::on_epoch_start(Trainer* trainer) {
    if (trainer) {
        std::cout << "Epoch " << (trainer->current_epoch() + 1) << " started" << std::endl;
    }
}

void ProgressCallback::on_train_batch_end(Trainer*, int) {}

void ProgressCallback::on_epoch_end(Trainer* trainer) {
    if (!trainer) {
        return;
    }
    std::cout << "Epoch " << (trainer->current_epoch() + 1)
              << ": train_loss=" << trainer->last_train_loss();
    if (trainer->has_validation_loss()) {
        std::cout << ", val_loss=" << trainer->last_val_loss();
    }
    std::cout << std::endl;
}

} // namespace callbacks
} // namespace training
} // namespace jyppx
