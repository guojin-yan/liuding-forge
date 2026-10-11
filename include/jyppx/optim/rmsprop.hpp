#ifndef JYPPX_OPTIM_RMSPROP_HPP
#define JYPPX_OPTIM_RMSPROP_HPP

#include <jyppx/optim/optimizer.hpp>

namespace jyppx {
namespace optim {

/**
 * @brief RMSprop optimizer options
 */
struct RMSpropOptions {
    double lr = 0.01;                   ///< Learning rate
    double alpha = 0.99;                ///< Smoothing constant for the moving average
    double eps = 1e-8;                  ///< Epsilon for numerical stability
    double weight_decay = 0.0;          ///< Weight decay (L2 penalty)
    double momentum = 0.0;              ///< Momentum factor
    bool centered = false;              ///< Center the moving average on the gradient
};

/**
 * @brief RMSprop optimizer
 *
 * Matches torch.optim.RMSprop: a moving average of squared gradients (and
 * optionally of the gradients themselves when centered) normalizes the update,
 * with optional momentum applied to the raw gradient.
 */
class RMSprop : public Optimizer {
public:
    explicit RMSprop(const TensorList& params, const RMSpropOptions& options = RMSpropOptions{});

    void step() override;

    /** @brief Mutable defaults synchronized into the first group on step(). */
    RMSpropOptions& options() { return options_; }
    const RMSpropOptions& options() const { return options_; }

private:
    void sync_options_to_default_group();

    RMSpropOptions options_;
    RMSpropOptions last_synced_options_;
};

} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_RMSPROP_HPP
