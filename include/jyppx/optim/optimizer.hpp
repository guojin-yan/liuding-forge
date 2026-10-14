#ifndef JYPPX_OPTIM_OPTIMIZER_HPP
#define JYPPX_OPTIM_OPTIMIZER_HPP

#include <jyppx/core/types.hpp>

#include <map>
#include <string>

namespace jyppx {
namespace optim {

/**
 * @brief Groups parameters that share optimizer hyperparameters.
 *
 * Tensor handles refer to parameters owned by the model. A parameter may
 * belong to only one group in an Optimizer.
 */
struct ParamGroup {
  TensorList params;  ///< Defined model parameters updated by this group.
  std::map<std::string, double> options;  ///< Finite named hyperparameters.

  /** @brief Creates an empty group for staged option construction. */
  ParamGroup() = default;

  /** @brief Creates a group referencing @p params_. @param params_ Model
   * parameters. */
  ParamGroup(const TensorList& params_) : params(params_) {}
};

/**
 * @brief Base class for stateful parameter optimizers.
 *
 * The optimizer references model tensors and owns its update buffers.
 * Instances are not thread-safe. The API follows PyTorch optimizer state and
 * parameter-group concepts.
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
   * @brief Performs one optimization step and increments step_count().
   */
  virtual void step() = 0;

  /**
   * @brief Zero out gradients
   * @param set_to_none If true, set gradients to undefined instead of zero
   */
  void zero_grad(bool set_to_none = false);

  /**
   * @brief Returns mutable parameter groups for algorithm implementations.
   * @return Groups in stable serialization order.
   */
  std::vector<ParamGroup>& param_groups() { return param_groups_; }

  /** @brief Returns parameter groups in stable serialization order. */
  const std::vector<ParamGroup>& param_groups() const { return param_groups_; }

  /**
   * @brief Add a parameter group after construction.
   *
   * Options omitted by @p group inherit the defaults stored on the first
   * parameter group. Parameters must be defined and may occur in only one
   * group owned by this optimizer.
   *
   * @param group Group to append; tensor handles remain model-owned.
   * @throws std::invalid_argument The group or one of its options is invalid.
   */
  virtual void add_param_group(ParamGroup group);

  /** @brief Returns one non-negative learning rate per parameter group. */
  std::vector<double> get_lrs() const;

  /**
   * @brief Sets one learning rate for every parameter group.
   * @param lr Finite, non-negative learning rate.
   * @throws std::invalid_argument @p lr is negative or non-finite.
   */
  void set_lr(double lr);

  /**
   * @brief Sets one learning rate per parameter group atomically.
   * @param lrs Finite, non-negative values matching the group count.
   * @throws std::invalid_argument The count or a value is invalid.
   */
  void set_lrs(const std::vector<double>& lrs);

  /** @brief Number of completed optimization steps. */
  int64_t step_count() const { return step_count_; }

  /**
   * @brief Serializes topology, options, step count, and update buffers.
   * @return Detached tensor values keyed by stable checkpoint names.
   */
  virtual std::map<std::string, torch::Tensor> state_dict() const;

  /**
   * @brief Validates and replaces optimizer state atomically.
   * @param state State produced by state_dict().
   * @throws std::invalid_argument Topology, metadata, or tensor state is
   * invalid.
   */
  virtual void load_state_dict(
      const std::map<std::string, torch::Tensor>& state);

 protected:
  std::vector<ParamGroup>
      param_groups_;  ///< Groups in stable checkpoint order.
  std::map<const c10::TensorImpl*, std::map<std::string, Tensor>>
      state_;               ///< Per-parameter algorithm buffers.
  int64_t step_count_ = 0;  ///< Completed optimization steps; non-negative.
};

}  // namespace optim
}  // namespace jyppx

#endif  // JYPPX_OPTIM_OPTIMIZER_HPP
