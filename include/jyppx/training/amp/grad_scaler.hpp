#ifndef JYPPX_TRAINING_AMP_GRAD_SCALER_HPP
#define JYPPX_TRAINING_AMP_GRAD_SCALER_HPP

#include <jyppx/core/types.hpp>
#include <jyppx/optim/optimizer.hpp>

#include <map>
#include <memory>
#include <string>

namespace jyppx {
namespace training {
namespace amp {

/**
 * @brief GradScaler configuration options
 */
struct GradScalerOptions {
    double init_scale = 65536.0;    ///< Initial scale factor (2^16)
    double growth_factor = 2.0;     ///< Multiplier applied when gradients stay finite
    double backoff_factor = 0.5;    ///< Multiplier applied when inf/NaN gradients appear
    int growth_interval = 2000;     ///< Successful steps between growth factor applications
    bool enabled = true;            ///< Disable to pass through scale/step unchanged
};

/**
 * @brief Gradient scaler for mixed-precision training
 *
 * Multiplies the loss so small FP16 gradients do not underflow, then unscales
 * gradients before the optimizer step and skips the step entirely when the
 * unscaled gradients contain inf/NaN (shrinking the scale factor). Mirrors
 * the semantics of torch.cuda.amp.GradScaler.
 *
 * @example
 * @code
 * amp::GradScaler scaler;
 * optimizer->zero_grad();
 * Tensor loss = loss_fn(model->forward(input), target);
 * scaler.scale(loss).backward();
 * scaler.step(optimizer);
 * scaler.update();
 * @endcode
 */
class GradScaler {
public:
    explicit GradScaler(GradScalerOptions options = GradScalerOptions{});

    /**
     * @brief Scale a loss tensor (no-op when disabled).
     */
    Tensor scale(const Tensor& loss) const;

    /**
     * @brief Divide every gradient by the current scale factor and remember
     *        whether any gradient became inf/NaN.
     *
     * Calling this before gradient clipping is the recommended order; step()
     * unscales automatically when it has not been done yet.
     */
    void unscale(const std::shared_ptr<optim::Optimizer>& optimizer);

    /**
     * @brief Unscale (if needed) and step the optimizer, skipping the step
     *        when inf/NaN gradients were found.
     */
    void step(const std::shared_ptr<optim::Optimizer>& optimizer);

    /**
     * @brief Grow the scale factor after growth_interval successful steps.
     *        Call once after every step() (PyTorch semantics: only steps
     *        without inf/NaN count toward the interval).
     */
    void update();

    /// Current scale factor (1.0 when disabled).
    double get_scale() const;

    /// Whether the most recent unscale/step detected inf or NaN gradients.
    bool found_inf() const;

    /// Serializable state (scale factor and growth tracker).
    std::map<std::string, double> state_dict() const;
    void load_state_dict(const std::map<std::string, double>& state);

private:
    GradScalerOptions options_;
    double scale_;
    int growth_tracker_;
    bool found_inf_;
    bool unscaled_;
};

} // namespace amp
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_AMP_GRAD_SCALER_HPP
