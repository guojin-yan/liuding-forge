#include <jyppx/jyppx.hpp>

#include <iostream>

int main() {
    auto data = torch::arange(0, 10, torch::TensorOptions().dtype(torch::kFloat32)).view({10, 1});
    auto targets = torch::arange(0, 10, torch::TensorOptions().dtype(torch::kFloat32));
    auto dataset = std::make_shared<jyppx::data::TensorDataset>(data, targets);

    jyppx::data::DataLoader loader(
        dataset,
        {.batch_size = 4, .shuffle = true, .drop_last = false, .seed = 7});

    for (const auto batch : loader) {
        std::cout << "batch: " << batch.data.size(0)
                  << " samples, target mean=" << batch.targets.mean().item<double>() << '\n';
    }
    return 0;
}
