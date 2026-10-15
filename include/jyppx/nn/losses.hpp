#ifndef JYPPX_NN_LOSSES_HPP
#define JYPPX_NN_LOSSES_HPP

#include <jyppx/core/types.hpp>

#include <string>

namespace jyppx {
namespace nn {

/**
 * @brief Loss functions beyond the ATen built-ins
 *
 * Remaining legacy functional implementations live in
 * jyppx::nn::functional. Migrated losses use focused headers under
 * `jyppx/nn/loss/`.
 */
namespace functional {

/**
 * @brief Applies a reduction ("none", "mean" or "sum") to a per-element loss
 */
Tensor apply_reduction(Tensor loss, const std::string& reduction);

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

}  // namespace functional

}  // namespace nn
}  // namespace jyppx

#endif  // JYPPX_NN_LOSSES_HPP
