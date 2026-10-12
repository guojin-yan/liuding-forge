// Optimizer convergence benchmark.  Output is CSV for easy comparison with
// an equivalent PyTorch script using the same seed and generated tensors.

#include <jyppx/optim/adam.hpp>
#include <jyppx/optim/adamw.hpp>
#include <jyppx/optim/sgd.hpp>

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace {

template <typename OptimizerT, typename OptionsT>
void benchmark(const std::string& name, const OptionsT& options) {
    torch::manual_seed(20261012);
    auto first = torch::randn({32, 16}, torch::TensorOptions().requires_grad(true));
    auto second = torch::randn({16, 1}, torch::TensorOptions().requires_grad(true));
    const auto inputs = torch::randn({128, 32});
    const auto targets = torch::randn({128, 1});
    auto optimizer = std::make_shared<OptimizerT>(jyppx::TensorList{first, second}, options);

    std::cout << name << ",step,loss\n";
    for (int step = 0; step < 100; ++step) {
        optimizer->zero_grad();
        const auto prediction = inputs.matmul(first).matmul(second);
        const auto loss = torch::mse_loss(prediction, targets);
        loss.backward();
        optimizer->step();
        std::cout << name << ',' << step << ',' << std::setprecision(10)
                  << loss.item<double>() << "\n";
    }
}

} // namespace

int main() {
    benchmark<jyppx::optim::SGD>("SGD", jyppx::optim::SGDOptions{.lr = 1e-2});
    benchmark<jyppx::optim::Adam>("Adam", jyppx::optim::AdamOptions{.lr = 1e-3});
    benchmark<jyppx::optim::AdamW>("AdamW", jyppx::optim::AdamWOptions{.lr = 1e-3});
}
