#include <jyppx/nn/losses.hpp>

#include <torch/nn/functional.h>

#include <stdexcept>

namespace jyppx {
namespace nn {

namespace functional {

namespace {

void validate_target_values(const Tensor& target, const Tensor& allowed) {
    const auto flat = target.reshape({-1}).to(allowed.scalar_type());
    if (!torch::isin(flat, allowed).all().item<bool>()) {
        throw std::invalid_argument("target contains values outside the allowed set");
    }
}

} // namespace

Tensor apply_reduction(Tensor loss, const std::string& reduction) {
    if (reduction == "none") {
        return loss;
    }
    if (reduction == "mean") {
        return loss.mean();
    }
    if (reduction == "sum") {
        return loss.sum();
    }
    throw std::invalid_argument("reduction must be 'none', 'mean' or 'sum'");
}

Tensor dice_loss(const Tensor& input, const Tensor& target,
                 double smooth, const std::string& reduction) {
    if (smooth < 0.0) {
        throw std::invalid_argument("dice_loss smooth must be non-negative");
    }
    if (!input.defined() || !target.defined()) {
        throw std::invalid_argument("dice_loss inputs must be defined");
    }
    if (reduction != "mean") {
        throw std::invalid_argument("dice_loss supports reduction='mean' only");
    }

    // All reductions stay inside tensor ops so gradients flow to the logits.
    Tensor loss;
    if (input.dim() == target.dim()) {
        // Binary path: logits vs {0, 1} float targets of the same shape.
        validate_target_values(target, torch::tensor({0.0, 1.0}, torch::kFloat));
        const auto probs = torch::sigmoid(input);
        const auto truth = target.to(input.scalar_type());
        const auto dice =
            (2.0 * (probs * truth).sum() + smooth) /
            (probs.sum() + truth.sum() + smooth);
        loss = 1.0 - dice;
    } else {
        // Multiclass path: logits {N, C, ...} with class-index targets
        // {N, ...}; one-hot the target and average the per-class Dice loss.
        if (target.dim() != input.dim() - 1) {
            throw std::invalid_argument(
                "dice_loss multiclass target must have one fewer dimension than input");
        }
        const int64_t classes = input.size(1);
        validate_target_values(target, torch::arange(classes, torch::kLong));

        const auto probs = input.softmax(1);
        auto truth = torch::nn::functional::one_hot(
            target.to(torch::kLong), classes);
        // One-hot appends the class axis last; move it next to the batch axis.
        std::vector<int64_t> permutation(truth.dim(), 0);
        permutation[1] = truth.dim() - 1;
        for (int64_t d = 1; d + 1 < truth.dim(); ++d) {
            permutation[static_cast<size_t>(d + 1)] = d;
        }
        truth = truth.permute(permutation).to(input.scalar_type());

        std::vector<Tensor> per_class;
        per_class.reserve(static_cast<size_t>(classes));
        for (int64_t c = 0; c < classes; ++c) {
            const auto p = probs.select(1, c);
            const auto y = truth.select(1, c);
            const auto dice =
                (2.0 * (p * y).sum() + smooth) / (p.sum() + y.sum() + smooth);
            per_class.push_back(1.0 - dice);
        }
        loss = torch::stack(per_class).mean();
    }
    return apply_reduction(std::move(loss), reduction);
}

Tensor distillation_loss(const Tensor& student_logits,
                         const Tensor& teacher_logits, double temperature,
                         const std::string& reduction) {
    if (temperature <= 0.0) {
        throw std::invalid_argument("distillation_loss temperature must be positive");
    }
    if (!student_logits.defined() || !teacher_logits.defined()) {
        throw std::invalid_argument("distillation_loss inputs must be defined");
    }
    if (student_logits.sizes() != teacher_logits.sizes()) {
        throw std::invalid_argument(
            "distillation_loss student and teacher logits must have the same shape");
    }

    const auto softened =
        teacher_logits.detach().to(torch::kFloat32) / temperature;
    const auto teacher_probs = torch::softmax(softened, /*dim=*/1);
    const auto student_logp = torch::log_softmax(
        student_logits.to(torch::kFloat32) / temperature, /*dim=*/1);
    // batchmean divides by the batch size only, matching the KL definition
    // (and PyTorch's recommendation); the T^2 factor keeps gradient
    // magnitudes comparable to the hard loss.
    auto loss = torch::nn::functional::kl_div(
        student_logp, teacher_probs,
        torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean)) *
        (temperature * temperature);
    return apply_reduction(std::move(loss), reduction);
}

} // namespace functional

DiceLoss::DiceLoss(double smooth, std::string reduction)
    : smooth_(smooth), reduction_(std::move(reduction)) {}

Tensor DiceLoss::forward(const Tensor& input, const Tensor& target) const {
    return functional::dice_loss(input, target, smooth_, reduction_);
}

} // namespace nn
} // namespace jyppx
