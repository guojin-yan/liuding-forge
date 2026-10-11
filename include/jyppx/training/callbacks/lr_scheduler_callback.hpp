#ifndef JYPPX_TRAINING_CALLBACKS_LR_SCHEDULER_CALLBACK_HPP
#define JYPPX_TRAINING_CALLBACKS_LR_SCHEDULER_CALLBACK_HPP

#include <jyppx/training/callback.hpp>
#include <jyppx/optim/schedulers/lr_scheduler.hpp>
#include <memory>

namespace jyppx {
namespace training {
namespace callbacks {

/**
 * @brief Learning rate scheduler callback
 */
class LRSchedulerCallback : public Callback {
public:
    explicit LRSchedulerCallback(std::shared_ptr<optim::schedulers::LRScheduler> scheduler);

    void on_epoch_end(Trainer* trainer) override;

private:
    std::shared_ptr<optim::schedulers::LRScheduler> scheduler_;
};

} // namespace callbacks
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_CALLBACKS_LR_SCHEDULER_CALLBACK_HPP
