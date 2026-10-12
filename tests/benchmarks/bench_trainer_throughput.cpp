// Reproducible CPU Trainer throughput benchmark.

#include <jyppx/jyppx.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace {

class ThroughputModel final : public torch::nn::Module {
public:
    ThroughputModel() : network(register_module(
        "network", torch::nn::Sequential(
            torch::nn::Linear(3 * 32 * 32, 512), torch::nn::ReLU(),
            torch::nn::Linear(512, 256), torch::nn::ReLU(),
            torch::nn::Linear(256, 10)))) {}

    jyppx::Tensor forward(const jyppx::Tensor& input) {
        return network->forward(input.view({input.size(0), -1}));
    }

    torch::nn::Sequential network{nullptr};
};

double run_configuration(std::size_t batch_size, int epochs, int samples) {
    auto data = torch::randn({samples, 3, 32, 32});
    auto targets = torch::randn({samples, 10});
    auto dataset = std::make_shared<jyppx::data::TensorDataset>(data, targets);
    jyppx::data::DataLoader loader(dataset, {.batch_size = batch_size, .num_workers = 0});
    auto model = std::make_shared<ThroughputModel>();
    auto optimizer = std::make_shared<jyppx::optim::Adam>(
        model->parameters(), jyppx::optim::AdamOptions{.lr = 1e-3});
    jyppx::training::Trainer trainer(
        model, optimizer, jyppx::training::TrainerOptions{.max_epochs = epochs});

    const auto start = std::chrono::steady_clock::now();
    trainer.fit(loader);
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return static_cast<double>(samples * epochs) / elapsed;
}

} // namespace

int main() {
    constexpr int kSamples = 50'000;
    constexpr int kEpochs = 1;
    std::cout << "batch_size,samples,epochs,samples_per_second\n";
    std::cout << std::fixed << std::setprecision(2);
    for (const std::size_t batch_size : {16U, 32U, 64U, 128U}) {
        const double throughput = run_configuration(batch_size, kEpochs, kSamples);
        std::cout << batch_size << ',' << kSamples << ',' << kEpochs << ','
                  << throughput << "\n";
    }
}
