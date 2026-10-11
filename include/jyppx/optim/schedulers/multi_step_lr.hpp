#ifndef JYPPX_OPTIM_SCHEDULERS_MULTI_STEP_LR_HPP
#define JYPPX_OPTIM_SCHEDULERS_MULTI_STEP_LR_HPP

#include <jyppx/optim/schedulers/lr_scheduler.hpp>

#include <vector>

namespace jyppx {
namespace optim {
namespace schedulers {

/**
 * @brief Decays the learning rate by gamma whenever a milestone is passed
 *
 * The learning rate becomes base_lr * gamma^k where k is the number of
 * milestones that are <= the current step count.
 */
class MultiStepLR : public LRScheduler {
public:
    MultiStepLR(
        std::shared_ptr<Optimizer> optimizer,
        std::vector<int64_t> milestones,
        double gamma = 0.1);

    using LRScheduler::step;  // keep the metric-less overload set visible

    void step() override;

private:
    std::vector<int64_t> milestones_;
    double gamma_;
};

} // namespace schedulers
} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_SCHEDULERS_MULTI_STEP_LR_HPP
