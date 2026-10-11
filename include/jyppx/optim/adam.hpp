#ifndef JYPPX_OPTIM_ADAM_HPP
#define JYPPX_OPTIM_ADAM_HPP

#include <jyppx/optim/optimizer.hpp>

namespace jyppx {
namespace optim {

/**
 * @brief Adam optimizer options
 */
struct AdamOptions {
    double lr = 0.001;                  ///< Learning rate
    double beta1 = 0.9;                 ///< Beta1 (exponential decay rate for first moment)
    double beta2 = 0.999;               ///< Beta2 (exponential decay rate for second moment)
    double eps = 1e-8;                  ///< Epsilon for numerical stability
    double weight_decay = 0.0;          ///< Weight decay (L2 penalty)
    bool amsgrad = false;               ///< Use AMSGrad variant
};

/**
 * @brief Adam optimizer
 */
class Adam : public Optimizer {
public:
    explicit Adam(const TensorList& params, const AdamOptions& options = AdamOptions{});

    void step() override;

    /** @brief Mutable defaults synchronized into the first group on step(). */
    AdamOptions& options() { return options_; }
    const AdamOptions& options() const { return options_; }

private:
    void sync_options_to_default_group();

    AdamOptions options_;
    AdamOptions last_synced_options_;
};

} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_ADAM_HPP
