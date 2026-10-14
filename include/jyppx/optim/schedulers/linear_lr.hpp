#ifndef JYPPX_OPTIM_SCHEDULERS_LINEAR_LR_HPP
#define JYPPX_OPTIM_SCHEDULERS_LINEAR_LR_HPP

#include <jyppx/optim/schedulers/lr_scheduler.hpp>

namespace jyppx::optim::schedulers {
/** @brief Linearly interpolates learning rates between two factors. */
class LinearLR : public LRScheduler {
public:
    LinearLR(std::shared_ptr<Optimizer> optimizer,
             double start_factor = 1.0 / 3.0,
             double end_factor = 1.0,
             int64_t total_iters = 5);
    using LRScheduler::step;
    void step() override;
private:
    double start_factor_;
    double end_factor_;
    int64_t total_iters_;
};
} // namespace jyppx::optim::schedulers
namespace jyppx::optim { using LinearLR = schedulers::LinearLR; }
#endif
