#ifndef JYPPX_OPTIM_SCHEDULERS_REDUCE_LR_ON_PLATEAU_HPP
#define JYPPX_OPTIM_SCHEDULERS_REDUCE_LR_ON_PLATEAU_HPP

#include <jyppx/optim/schedulers/lr_scheduler.hpp>

#include <limits>
#include <string>

namespace jyppx {
namespace optim {
namespace schedulers {

/**
 * @brief ReduceLROnPlateau configuration
 */
struct ReduceLROnPlateauOptions {
    std::string mode = "min";            ///< "min" when lower is better, "max" otherwise
    double factor = 0.1;                 ///< Multiplicative decay applied on plateau
    int patience = 10;                   ///< Non-improving steps tolerated before decay
    double threshold = 1e-4;             ///< Minimum change that counts as an improvement
    std::string threshold_mode = "rel";  ///< "rel" (relative to best) or "abs"
    double min_lr = 0.0;                 ///< Lower bound for the scheduled learning rate
};

/**
 * @brief Reduce the learning rate when a monitored metric stops improving
 *
 * Mirrors torch.optim.lr_scheduler.ReduceLROnPlateau. Unlike the epoch-based
 * schedulers this one is driven by a metric: call step(metric) after each
 * validation. Calling the metric-less step() is rejected.
 */
class ReduceLROnPlateau : public LRScheduler {
public:
    explicit ReduceLROnPlateau(
        std::shared_ptr<Optimizer> optimizer,
        ReduceLROnPlateauOptions options = ReduceLROnPlateauOptions{});

    void step() override;

    void step(double metric) override;

    /** @brief Best metric value observed so far. */
    double best() const { return best_; }

    /** @brief Number of consecutive non-improving steps. */
    int num_bad_epochs() const { return num_bad_epochs_; }

private:
    bool is_better(double current) const;

    ReduceLROnPlateauOptions options_;
    double best_;
    int num_bad_epochs_;
};

} // namespace schedulers
} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_SCHEDULERS_REDUCE_LR_ON_PLATEAU_HPP
