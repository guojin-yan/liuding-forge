#ifndef JYPPX_OPTIM_ADAMW_HPP
#define JYPPX_OPTIM_ADAMW_HPP

#include <jyppx/optim/optimizer.hpp>

namespace jyppx {
namespace optim {

/**
 * @brief AdamW optimizer options
 */
struct AdamWOptions {
    double lr = 0.001;                  ///< Learning rate
    double beta1 = 0.9;                 ///< Beta1 (exponential decay rate for first moment)
    double beta2 = 0.999;               ///< Beta2 (exponential decay rate for second moment)
    double eps = 1e-8;                  ///< Epsilon for numerical stability
    double weight_decay = 0.01;         ///< Weight decay (decoupled)
    bool amsgrad = false;               ///< Use AMSGrad variant
};

/**
 * @brief AdamW optimizer with decoupled weight decay
 */
class AdamW : public Optimizer {
public:
    explicit AdamW(const TensorList& params, const AdamWOptions& options = AdamWOptions{});

    void step() override;

    /** @brief Mutable defaults synchronized into the first group on step(). */
    AdamWOptions& options() { return options_; }
    const AdamWOptions& options() const { return options_; }

private:
    void sync_options_to_default_group();

    AdamWOptions options_;
    AdamWOptions last_synced_options_;
};

} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_ADAMW_HPP
