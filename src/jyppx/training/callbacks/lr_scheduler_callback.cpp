#include <jyppx/training/callbacks/lr_scheduler_callback.hpp>
#include <jyppx/training/trainer.hpp>

#include <stdexcept>

namespace jyppx {
namespace training {
namespace callbacks {

LRSchedulerCallback::LRSchedulerCallback(
    std::shared_ptr<optim::schedulers::LRScheduler> scheduler)
    : scheduler_(std::move(scheduler)) {
    if (!scheduler_) {
        throw std::invalid_argument("LRSchedulerCallback scheduler cannot be null");
    }
}

void LRSchedulerCallback::on_epoch_end(Trainer* trainer) {
    if (trainer && trainer->scheduler() == scheduler_) {
        return;
    }
    if (scheduler_->requires_metric()) {
        if (!trainer || !trainer->has_validation_loss()) {
            throw std::invalid_argument(
                "Metric-driven scheduler callback requires a validation loss");
        }
        scheduler_->step(trainer->last_val_loss());
    } else {
        scheduler_->step();
    }
}

} // namespace callbacks
} // namespace training
} // namespace jyppx
