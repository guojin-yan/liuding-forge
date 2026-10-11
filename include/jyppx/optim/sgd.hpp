#ifndef JYPPX_OPTIM_SGD_HPP
#define JYPPX_OPTIM_SGD_HPP

#include <jyppx/optim/optimizer.hpp>

namespace jyppx {
namespace optim {

/**
 * @brief SGD optimizer options
 */
struct SGDOptions {
    double lr = 0.01;                   ///< Learning rate
    double momentum = 0.0;              ///< Momentum factor
    double dampening = 0.0;             ///< Dampening for momentum
    double weight_decay = 0.0;          ///< Weight decay (L2 penalty)
    bool nesterov = false;              ///< Enable Nesterov momentum
};

/**
 * @brief Stochastic Gradient Descent optimizer
 */
class SGD : public Optimizer {
public:
    explicit SGD(const TensorList& params, const SGDOptions& options = SGDOptions{});

    void step() override;

    /**
     * @brief Mutable defaults for the first parameter group.
     *
     * Changes are synchronized into the first group on the next step while
     * scheduler updates made through set_lr/set_lrs remain effective.
     */
    SGDOptions& options() { return options_; }
    const SGDOptions& options() const { return options_; }

private:
    void sync_options_to_default_group();

    SGDOptions options_;
    SGDOptions last_synced_options_;
};

} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_SGD_HPP
