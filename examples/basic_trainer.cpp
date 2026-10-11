#include <jyppx/jyppx.hpp>

#include <iostream>

class LinearModelImpl : public torch::nn::Module {
public:
    LinearModelImpl() : linear(register_module("linear", torch::nn::Linear(1, 1))) {}

    jyppx::Tensor forward(const jyppx::Tensor& input) {
        return linear->forward(input);
    }

    torch::nn::Linear linear{nullptr};
};

int main() {
    try {
        torch::manual_seed(7);
        auto data =
            torch::arange(0, 8, torch::TensorOptions().dtype(torch::kFloat32)).view({8, 1});
        auto targets = data * 2.0 + 1.0;
        auto dataset = std::make_shared<jyppx::data::TensorDataset>(data, targets);
        jyppx::data::DataLoader loader(dataset, {.batch_size = 4});

        auto model = std::make_shared<LinearModelImpl>();
        auto optimizer = std::make_shared<jyppx::optim::SGD>(
            model->parameters(), jyppx::optim::SGDOptions{.lr = 0.01});
        jyppx::training::Trainer trainer(
            model,
            optimizer,
            jyppx::training::TrainerOptions{.max_epochs = 2});
        trainer.fit(loader);
        std::cout << "final loss=" << trainer.last_train_loss() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
