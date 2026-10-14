#ifndef JYPPX_OPTIM_SCHEDULERS_LR_SCHEDULER_HPP
#define JYPPX_OPTIM_SCHEDULERS_LR_SCHEDULER_HPP

#include <jyppx/optim/optimizer.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace jyppx {
namespace optim {
namespace schedulers {

enum class SchedulerStepMode {
    batch,
    epoch,
    metric,
    manual,
};

struct SchedulerDescriptor {
    uint32_t schema_version = 1;
    std::string type_id;
    SchedulerStepMode step_mode = SchedulerStepMode::manual;
    std::map<std::string, std::string> config;
    std::vector<SchedulerDescriptor> children;
    bool checkpoint_safe = true;
};

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

    /** @brief Whether this scheduler must be stepped with a monitored metric. */
    virtual bool requires_metric() const { return false; }

    /** Stable scheduler identity and future-trajectory configuration. */
    const SchedulerDescriptor& descriptor() const { return descriptor_; }

    /** Optimizer whose parameter groups are controlled by this scheduler. */
    const std::shared_ptr<Optimizer>& optimizer() const { return optimizer_; }

    /** Canonical FNV-1a fingerprint of descriptor(), represented as hex. */
    std::string descriptor_fingerprint() const;

    /** Validate only identity/schema fields without changing scheduler state. */
    void validate_descriptor(const std::map<std::string, double>& state) const;

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

    /** @brief Serializable scheduler state (step count and current LRs). */
    virtual std::map<std::string, double> state_dict() const;

    /** @brief Restore scheduler state into an already configured scheduler. */
    virtual void load_state_dict(const std::map<std::string, double>& state);

protected:
    std::shared_ptr<Optimizer> optimizer_;
    int64_t step_count_;
    double base_lr_;
    double last_lr_;
    std::vector<double> base_lrs_;
    std::vector<double> last_lrs_;
    SchedulerDescriptor descriptor_;

    /** @brief Apply one learning rate per optimizer parameter group. */
    void set_last_lrs(const std::vector<double>& lrs);

    /** Validate and decode common state without mutating this scheduler. */
    void decode_state_dict(
        const std::map<std::string, double>& state,
        bool allow_negative_lrs,
        int64_t& restored_step,
        std::vector<double>& restored_lrs) const;

    void configure_descriptor(
        std::string type_id,
        SchedulerStepMode step_mode,
        std::map<std::string, std::string> config = {},
        std::vector<SchedulerDescriptor> children = {},
        bool checkpoint_safe = true);

    static std::string descriptor_number(double value);
    static std::string descriptor_integer(int64_t value);
    static std::string descriptor_boolean(bool value);
};

} // namespace schedulers
} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_SCHEDULERS_LR_SCHEDULER_HPP
