#ifndef JYPPX_LIUDING_TEST_HELPERS_HPP
#define JYPPX_LIUDING_TEST_HELPERS_HPP

// Shared helpers for the JYPPX unit tests.

#include <jyppx/liuding/liuding.hpp>

#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace jyppx::liuding::test {

using jyppx::liuding::Tensor;
using jyppx::liuding::TensorList;

// A linear model y = Wx + b with a fixed, deterministic initialization so
// tests that compare two independently constructed models get identical
// starting weights (combine with torch::manual_seed).
class LinearModel : public torch::nn::Module {
 public:
  LinearModel(int in_features = 2, int out_features = 1)
      : linear(register_module("linear",
                               torch::nn::Linear(in_features, out_features))) {}

  Tensor forward(const Tensor& input) { return linear->forward(input); }

  torch::nn::Linear linear{nullptr};
};

// Copies the parameter values of one module into another module with the
// same architecture (requires_grad is preserved on the destination).
inline void copy_module_weights(const torch::nn::Module& source,
                                torch::nn::Module& destination) {
  const auto source_params = source.parameters();
  auto destination_params = destination.parameters();
  ASSERT_EQ(source_params.size(), destination_params.size());
  for (size_t i = 0; i < source_params.size(); ++i) {
    torch::NoGradGuard no_grad;
    destination_params[i].copy_(source_params[i].detach());
  }
}

// Asserts two tensor lists have matching shapes and values.
inline void expect_tensor_lists_close(const TensorList& expected,
                                      const TensorList& actual,
                                      double tolerance = 1e-10) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected[i].sizes(), actual[i].sizes());
    const double max_diff =
        (expected[i] - actual[i]).abs().max().item<double>();
    EXPECT_TRUE(torch::allclose(expected[i], actual[i], tolerance, tolerance))
        << "tensor " << i << " differs (max abs diff " << max_diff
        << ", tolerance " << tolerance << ")\nexpected:\n"
        << expected[i] << "\nactual:\n"
        << actual[i];
  }
}

// Builds an (inputs, targets) pair for y = 2*x0 - x1 + 0.5 with deterministic
// inputs. Both tensors are float32 with `samples` rows.
inline std::pair<Tensor, Tensor> make_linear_data(int samples,
                                                  int64_t seed = 1234) {
  torch::manual_seed(seed);
  auto inputs =
      torch::rand({samples, 2}, torch::TensorOptions().dtype(torch::kFloat32));
  auto targets = 2.0 * inputs.select(1, 0) - inputs.select(1, 1) + 0.5;
  return {inputs, targets.view({samples, 1})};
}

// Clones a parameter list into fresh leaf tensors that require grad, so the
// clones can receive gradients from backward() like original parameters do.
inline TensorList clone_leaf_params(const TensorList& params) {
  TensorList clones;
  clones.reserve(params.size());
  for (const auto& parameter : params) {
    clones.push_back(parameter.detach().clone().set_requires_grad(true));
  }
  return clones;
}

// Deterministic quadratic loss over a parameter list (float64), with a
// distinct coefficient for every scalar so each parameter receives a unique
// gradient. Used for optimizer parity comparisons.
inline Tensor quadratic_loss(const TensorList& params) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat64);
  Tensor loss = torch::zeros({}, options);
  int64_t offset = 0;
  for (const auto& parameter : params) {
    auto coefficients =
        torch::arange(offset + 1.0,
                      offset + 1.0 + static_cast<double>(parameter.numel()),
                      options)
            .view(parameter.sizes());
    loss = loss + (parameter * coefficients).pow(2).sum();
    offset += parameter.numel();
  }
  return loss;
}

// Creates a float64 parameter list with the given shapes and requires_grad.
// Call torch::manual_seed before to obtain reproducible values.
inline TensorList make_params(const std::vector<std::vector<int64_t>>& shapes) {
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true);
  TensorList params;
  for (const auto& shape : shapes) {
    params.push_back(torch::randn(shape, options));
  }
  return params;
}

// Drives `steps` identical training steps through a jyppx optimizer and a
// torch::optim optimizer over value-identical parameter lists, then asserts
// the parameters stayed equal. Both optimizers must expose zero_grad/step.
template <typename JyppxOpt, typename TorchOpt>
void expect_optimizer_parity(JyppxOpt& jyppx_liuding_optimizer,
                             const TensorList& jyppx_params,
                             TorchOpt& torch_optimizer,
                             const std::vector<torch::Tensor>& torch_params,
                             int steps, double tolerance = 1e-9) {
  for (int step = 0; step < steps; ++step) {
    jyppx_liuding_optimizer.zero_grad();
    torch_optimizer.zero_grad();
    quadratic_loss(jyppx_params).backward();
    quadratic_loss(torch_params).backward();
    jyppx_liuding_optimizer.step();
    torch_optimizer.step();
  }
  expect_tensor_lists_close(
      jyppx_params, TensorList(torch_params.begin(), torch_params.end()),
      tolerance);
}

}  // namespace jyppx::liuding::test

#endif  // JYPPX_LIUDING_TEST_HELPERS_HPP
