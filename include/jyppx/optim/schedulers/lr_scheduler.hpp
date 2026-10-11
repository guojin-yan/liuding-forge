#ifndef JYPPX_OPTIM_SCHEDULERS_LR_SCHEDULER_HPP
#define JYPPX_OPTIM_SCHEDULERS_LR_SCHEDULER_HPP

#include <jyppx/optim/optimizer.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

namespace jyppx {
namespace optim {
namespace schedulers {

/**
 * @brief Base class for learning rate schedulers
 */
class LRScheduler {
public:
    /**
     * @brief Constructor
     * @param optimizer Optimizer to schedule
     */
    explicit LRScheduler(std::shared_ptr<Optimizer> optimizer);

    virtual ~LRScheduler() = default;

    /**
     * @brief Perform a single step (update learning rate)
     */
    virtual void step() = 0;

    /**
     * @brief Step with a monitored metric (ReduceLROnPlateau style)
     *
     * Schedulers that only step on epochs reject this call; metric-driven
     * schedulers override it and reject step() instead.
     */
    virtual void step(double metric) {
        (void)metric;
        throw std::logic_error(
            "This scheduler does not consume a metric; call step()");
    }

    /**
     * @brief Get current learning rate
     */
    double get_last_lr() const;

    /** @brief Get the current learning rate of every parameter group. */
    const std::vector<double>& get_last_lrs() const { return last_lrs_; }

    /**
     * @brief Get current epoch/step count
     */
    int64_t get_step() const { return step_count_; }

protected:
    std::shared_ptr<Optimizer> optimizer_;
    int64_t step_count_;
    double base_lr_;
    double last_lr_;
    std::vector<double> base_lrs_;
    std::vector<double> last_lrs_;

    /** @brief Apply one learning rate per optimizer parameter group. */
    void set_last_lrs(const std::vector<double>& lrs);
};

} // namespace schedulers
} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_SCHEDULERS_LR_SCHEDULER_HPP
