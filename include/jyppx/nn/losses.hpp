#ifndef JYPPX_NN_LOSSES_HPP
#define JYPPX_NN_LOSSES_HPP

#include <jyppx/core/types.hpp>
#include <torch/nn/module.h>

#include <string>

namespace jyppx {
namespace nn {

/**
 * @brief Loss functions beyond the ATen built-ins
 *
 * Functional implementations live in jyppx::nn::functional; thin
 * torch::nn::Module wrappers for criterion-style use live in jyppx::nn.
 */
namespace functional {

/**
 * @brief Applies a reduction ("none", "mean" or "sum") to a per-element loss
 */
Tensor apply_reduction(Tensor loss, const std::string& reduction);

/**
 * @brief Soft Dice loss (1 - Dice), for segmentation
 *
 * Multiclass: input are logits {N, C, ...} and target class indices
 * {N, ...}; the class probabilities come from a softmax and the Dice score
 * is computed per class, then averaged. Binary: input logits and a {0, 1}
 * float target of the same shape.
 *
 * @param smooth smoothing constant added to both sums
 */
Tensor dice_loss(const Tensor& input, const Tensor& target,
                 double smooth = 1.0, const std::string& reduction = "mean");

/**
 * @brief Knowledge-distillation loss (Hinton et al., 2015)
 *
 * KL divergence between the teacher's and student's softened class
 * distributions, scaled by T^2 so gradient magnitudes match the hard loss:
 *   T^2 * KL(softmax(teacher / T) || log_softmax(student / T))
 *
 * @param student_logits {N, C} student logits (gradients flow here)
 * @param teacher_logits {N, C} teacher logits (detached internally)
 * @param temperature softening temperature (> 0)
 */
Tensor distillation_loss(const Tensor& student_logits,
                         const Tensor& teacher_logits, double temperature = 2.0,
                         const std::string& reduction = "mean");

} // namespace functional

/**
 * @brief Dice loss criterion module
 */
class DiceLoss : public torch::nn::Module {
public:
    explicit DiceLoss(double smooth = 1.0, std::string reduction = "mean");

    Tensor forward(const Tensor& input, const Tensor& target) const;

    double smooth() const { return smooth_; }
    const std::string& reduction() const { return reduction_; }

private:
    double smooth_;
    std::string reduction_;
};

} // namespace nn
} // namespace jyppx

#endif // JYPPX_NN_LOSSES_HPP
