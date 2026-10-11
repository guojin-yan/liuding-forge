#ifndef JYPPX_OPTIM_NADAM_HPP
#define JYPPX_OPTIM_NADAM_HPP

#include <jyppx/optim/optimizer.hpp>

namespace jyppx {
namespace optim {

/**
 * @brief NAdam optimizer options
 */
struct NAdamOptions {
    double lr = 0.002;                  ///< Learning rate
    double beta1 = 0.9;                 ///< Beta1 (momentum schedule base)
    double beta2 = 0.999;               ///< Beta2
    double eps = 1e-8;                  ///< Epsilon for numerical stability
    double weight_decay = 0.0;          ///< Weight decay (L2 penalty)
    double momentum_decay = 0.0;        ///< Extra per-step momentum decay (0 only)
};

/**
 * @brief Adam with Nesterov momentum (Dozat, 2016)
 *
 * Matches torch.optim.NAdam with momentum_decay = 0 (the default schedule
 * mu_t = beta1 * (1 - 0.5 * 0.96^step), cumulative mu-product bias
 * correction, and the two-term Nesterov update). momentum_decay > 0 is
 * rejected for now.
 */
class NAdam : public Optimizer {
public:
    explicit NAdam(const TensorList& params, const NAdamOptions& options = NAdamOptions{});

    void step() override;

    /** @brief Mutable defaults synchronized into the first group on step(). */
    NAdamOptions& options() { return options_; }
    const NAdamOptions& options() const { return options_; }

private:
    void sync_options_to_default_group();

    NAdamOptions options_;
    NAdamOptions last_synced_options_;
    double mu_product_ = 1.0;
};

} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_NADAM_HPP
