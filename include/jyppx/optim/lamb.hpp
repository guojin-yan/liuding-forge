#ifndef JYPPX_OPTIM_LAMB_HPP
#define JYPPX_OPTIM_LAMB_HPP

#include <jyppx/optim/optimizer.hpp>

namespace jyppx {
namespace optim {

/**
 * @brief LAMB optimizer options
 */
struct LambOptions {
    double lr = 1e-3;                   ///< Learning rate
    double beta1 = 0.9;                 ///< Beta1
    double beta2 = 0.999;               ///< Beta2
    double eps = 1e-6;                  ///< Epsilon inside the update denominator
    double weight_decay = 0.0;          ///< Weight decay folded into the update
};

/**
 * @brief Layer-wise Adaptive Moments optimizer (You et al., 2020)
 *
 * Matches torch.optim.Lamb: an Adam-style update with decoupled weight decay
 * added to the update direction, then scaled by the per-parameter trust ratio
 * ||param|| / ||update|| (both norms clamped like the reference
 * implementation: param norm clamped to [0, 10]; a zero norm on either side
 * falls back to a trust ratio of 1).
 */
class Lamb : public Optimizer {
public:
    explicit Lamb(const TensorList& params, const LambOptions& options = LambOptions{});

    void step() override;

    /** @brief Mutable defaults synchronized into the first group on step(). */
    LambOptions& options() { return options_; }
    const LambOptions& options() const { return options_; }

private:
    void sync_options_to_default_group();

    LambOptions options_;
    LambOptions last_synced_options_;
};

} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_LAMB_HPP
