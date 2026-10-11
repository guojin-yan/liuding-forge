#ifndef JYPPX_OPTIM_ADAGRAD_HPP
#define JYPPX_OPTIM_ADAGRAD_HPP

#include <jyppx/optim/optimizer.hpp>

namespace jyppx {
namespace optim {

/**
 * @brief Adagrad optimizer options
 */
struct AdagradOptions {
    double lr = 0.01;                   ///< Learning rate
    double lr_decay = 0.0;              ///< Learning rate decay per step
    double weight_decay = 0.0;          ///< Weight decay (L2 penalty)
    double eps = 1e-10;                 ///< Epsilon for numerical stability
};

/**
 * @brief Adagrad optimizer
 *
 * Matches torch.optim.Adagrad: the learning rate is decayed over time
 * (lr / (1 + step * lr_decay)) and normalized by the accumulated sum of
 * squared gradients.
 */
class Adagrad : public Optimizer {
public:
    explicit Adagrad(const TensorList& params, const AdagradOptions& options = AdagradOptions{});

    void step() override;

    /** @brief Mutable defaults synchronized into the first group on step(). */
    AdagradOptions& options() { return options_; }
    const AdagradOptions& options() const { return options_; }

private:
    void sync_options_to_default_group();

    AdagradOptions options_;
    AdagradOptions last_synced_options_;
};

} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_ADAGRAD_HPP
