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
 * @brief Focal loss (Lin et al., 2017)
 *
 * Multiclass: input are logits {N, C} and target class indices {N};
 *   FL = -alpha * (1 - pt)^gamma * log(pt),  pt = exp(-CE)
 * Binary: input logits and target {0, 1} float of the same shape; the
 * positive class is weighted by alpha, the negative by (1 - alpha).
 *
 * @param alpha class-balancing weight (> 0; 1.0 disables balancing)
 * @param gamma focusing parameter (>= 0; 0 recovers weighted cross entropy)
 */
Tensor focal_loss(const Tensor& input, const Tensor& target,
                  double alpha = 1.0, double gamma = 2.0,
                  const std::string& reduction = "mean");

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
 * @brief Contrastive loss over pairs
 *
 * loss = y * d^2 + (1 - y) * max(0, margin - d)^2 with the Euclidean
 * distance d between each feature pair; y = 1 marks a matching pair.
 *
 * @param x1, x2 feature tensors {N, D}
 * @param target {N} float/int values in {0, 1}
 */
Tensor contrastive_loss(const Tensor& x1, const Tensor& x2,
                        const Tensor& target, double margin = 1.0,
                        const std::string& reduction = "mean");

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
 * @brief Focal loss criterion module
 */
class FocalLoss : public torch::nn::Module {
public:
    FocalLoss(double alpha = 1.0, double gamma = 2.0,
              std::string reduction = "mean");

    Tensor forward(const Tensor& input, const Tensor& target) const;

    double alpha() const { return alpha_; }
    double gamma() const { return gamma_; }
    const std::string& reduction() const { return reduction_; }

private:
    double alpha_;
    double gamma_;
    std::string reduction_;
};

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

/**
 * @brief Contrastive loss criterion module
 */
class ContrastiveLoss : public torch::nn::Module {
public:
    explicit ContrastiveLoss(double margin = 1.0, std::string reduction = "mean");

    Tensor forward(const Tensor& x1, const Tensor& x2, const Tensor& target) const;

    double margin() const { return margin_; }
    const std::string& reduction() const { return reduction_; }

private:
    double margin_;
    std::string reduction_;
};

} // namespace nn
} // namespace jyppx

#endif // JYPPX_NN_LOSSES_HPP
