#ifndef JYPPX_OPTIM_OPTIMIZER_HPP
#define JYPPX_OPTIM_OPTIMIZER_HPP

#include <jyppx/core/types.hpp>
#include <map>
#include <string>

namespace jyppx {
namespace optim {

/**
 * @brief Parameter group for optimizer
 */
struct ParamGroup {
    TensorList params;                          ///< Parameters in this group
    std::map<std::string, double> options;      ///< Options for this group

    ParamGroup() = default;
    ParamGroup(const TensorList& params_) : params(params_) {}
};

/**
 * @brief Base class for all optimizers
 *
 * Follows PyTorch Optimizer API
 */
class Optimizer {
public:
    /**
     * @brief Constructor
     * @param params Parameters to optimize
     */
    explicit Optimizer(const TensorList& params);

    /**
     * @brief Virtual destructor
     */
    virtual ~Optimizer() = default;

    /**
     * @brief Perform a single optimization step
     */
    virtual void step() = 0;

    /**
     * @brief Zero out gradients
     * @param set_to_none If true, set gradients to undefined instead of zero
     */
    void zero_grad(bool set_to_none = false);

    /**
     * @brief Get parameter groups
     */
    std::vector<ParamGroup>& param_groups() { return param_groups_; }
    const std::vector<ParamGroup>& param_groups() const { return param_groups_; }

    /** @brief Get the learning rate of every parameter group. */
    std::vector<double> get_lrs() const;

    /** @brief Set one learning rate for every parameter group. */
    void set_lr(double lr);

    /** @brief Set one learning rate per parameter group. */
    void set_lrs(const std::vector<double>& lrs);

    /** @brief Number of completed optimization steps. */
    int64_t step_count() const { return step_count_; }

    /**
     * @brief Get state dict for serialization
     */
    virtual std::map<std::string, torch::Tensor> state_dict() const;

    /**
     * @brief Load state dict
     */
    virtual void load_state_dict(const std::map<std::string, torch::Tensor>& state);

protected:
    std::vector<ParamGroup> param_groups_;      ///< Parameter groups
    std::map<const c10::TensorImpl*, std::map<std::string, Tensor>> state_;  ///< Optimizer state
    int64_t step_count_ = 0;
};

} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_OPTIMIZER_HPP
