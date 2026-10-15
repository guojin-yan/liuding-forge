#include <jyppx/nn/losses.hpp>

#include <torch/nn/functional.h>

#include <stdexcept>

namespace jyppx {
namespace nn {

namespace functional {

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

Tensor distillation_loss(const Tensor& student_logits,
                         const Tensor& teacher_logits, double temperature,
                         const std::string& reduction) {
  if (temperature <= 0.0) {
    throw std::invalid_argument(
        "distillation_loss temperature must be positive");
  }
  if (!student_logits.defined() || !teacher_logits.defined()) {
    throw std::invalid_argument("distillation_loss inputs must be defined");
  }
  if (student_logits.sizes() != teacher_logits.sizes()) {
    throw std::invalid_argument(
        "distillation_loss student and teacher logits must have the same "
        "shape");
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
                  torch::nn::functional::KLDivFuncOptions().reduction(
                      torch::kBatchMean)) *
              (temperature * temperature);
  return apply_reduction(std::move(loss), reduction);
}

}  // namespace functional

}  // namespace nn
}  // namespace jyppx
