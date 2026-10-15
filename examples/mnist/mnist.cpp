// MNIST training example: downloads the dataset (if missing), loads it with
// a self-contained gzip/IDX reader, and trains a small CNN through the
// jyppx Trainer with a multi-threaded DataLoader, LR scheduling, metric
// tracking and a TensorBoard callback.
//
// usage: jyppx_mnist [data_dir] [epochs]

#include <jyppx/jyppx.hpp>
#include <torch/nn/init.h>

#include "mnist_data.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

using namespace jyppx;

class MnistCnn : public torch::nn::Module {
public:
    MnistCnn()
        : conv1(register_module("conv1", torch::nn::Conv2d(
                                       torch::nn::Conv2dOptions(1, 8, 3).padding(1)))),
          conv2(register_module("conv2", torch::nn::Conv2d(
                                       torch::nn::Conv2dOptions(8, 16, 3).padding(1)))),
          fc1(register_module("fc1", torch::nn::Linear(16 * 7 * 7, 64))),
          fc2(register_module("fc2", torch::nn::Linear(64, 10))) {
        torch::nn::init::kaiming_uniform_(conv1->weight, 0.0, torch::kFanIn,
                                          torch::kLeakyReLU);
        torch::nn::init::kaiming_uniform_(conv2->weight, 0.0, torch::kFanIn,
                                          torch::kLeakyReLU);
    }

    Tensor forward(const Tensor& input) {
        // The loader feeds flattened {N, 784} rows; reshape into images.
        auto x = input.view({-1, 1, 28, 28});
        x = torch::relu(conv1->forward(x));            // {N, 8, 28, 28}
        x = torch::max_pool2d(x, 2);                   // {N, 8, 14, 14}
        x = torch::relu(conv2->forward(x));            // {N, 16, 14, 14}
        x = torch::max_pool2d(x, 2);                   // {N, 16, 7, 7}
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        return fc2->forward(x);                        // {N, 10} logits
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};

void download_if_missing(const std::string& path, const std::string& url) {
    if (std::filesystem::exists(path)) {
        return;
    }
    std::cout << "downloading " << url << "\n";
    const std::string command =
        "curl -L --retry 3 --fail -o \"" + path + "\" \"" + url + "\"";
    if (std::system(command.c_str()) != 0 || !std::filesystem::exists(path)) {
        throw std::runtime_error("failed to download " + url +
                                 " — download the four MNIST .gz files into " +
                                 std::filesystem::path(path).parent_path().string() +
                                 " and rerun");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string data_dir =
            argc > 1 ? argv[1] : (std::filesystem::temp_directory_path() / "jyppx_mnist").string();
        const int epochs = argc > 2 ? std::atoi(argv[2]) : 3;

        std::filesystem::create_directories(data_dir);
        const std::string base = "https://ossci-datasets.s3.amazonaws.com/mnist/";
        download_if_missing(data_dir + "/train-images-idx3-ubyte.gz",
                            base + "train-images-idx3-ubyte.gz");
        download_if_missing(data_dir + "/train-labels-idx1-ubyte.gz",
                            base + "train-labels-idx1-ubyte.gz");
        download_if_missing(data_dir + "/t10k-images-idx3-ubyte.gz",
                            base + "t10k-images-idx3-ubyte.gz");
        download_if_missing(data_dir + "/t10k-labels-idx1-ubyte.gz",
                            base + "t10k-labels-idx1-ubyte.gz");

        std::cout << "loading MNIST...\n";
        const auto train_raw = mnist_data::load_mnist_idx(
            data_dir + "/train-images-idx3-ubyte.gz",
            data_dir + "/train-labels-idx1-ubyte.gz");
        const auto test_raw = mnist_data::load_mnist_idx(
            data_dir + "/t10k-images-idx3-ubyte.gz",
            data_dir + "/t10k-labels-idx1-ubyte.gz");
        std::cout << "train samples: " << train_raw.images.size(0)
                  << ", test samples: " << test_raw.images.size(0) << "\n";

        auto model = std::make_shared<MnistCnn>();
        auto optimizer = std::make_shared<jyppx::optim::Adam>(
            model->parameters(), jyppx::optim::AdamOptions{.lr = 1e-3});

        // Train on the GPU when one is available.
        const Device device = get_default_device();
        std::cout << "training on: " << device << "\n";

        jyppx::training::TrainerOptions options;
        options.max_epochs = epochs;
        options.device = device;
        options.loss_fn = [](const Tensor& prediction, const Tensor& target) {
            return torch::nn::functional::cross_entropy(
                prediction.view({-1, 10}),
                target.to(torch::kLong).view({-1}));
        };
        options.callbacks.push_back(
            std::make_shared<jyppx::training::callbacks::ProgressCallback>());
        const auto tb_dir = data_dir + "/tb";
        options.callbacks.push_back(
            std::make_shared<jyppx::training::callbacks::TensorBoardCallback>(
                tb_dir, /*log_every_n_steps=*/50));

        jyppx::training::Trainer trainer(model, optimizer, options);

        // The images are flattened to rows for the default collate; the model
        // reshapes them back inside forward().
        auto flat_dataset = std::make_shared<jyppx::data::TensorDataset>(
            train_raw.images.view({-1, 784}),
            train_raw.labels.to(torch::kFloat32));

        jyppx::data::DataLoader train_loader(
            flat_dataset, {.batch_size = 64, .shuffle = true, .num_workers = 2});
        trainer.fit(train_loader);
        std::cout << "\nfinal train loss: " << trainer.last_train_loss() << "\n";

        // Evaluation pass with accuracy metrics.
        jyppx::training::metrics::MetricCollection metrics;
        metrics.add(std::make_shared<jyppx::training::metrics::Accuracy>());
        metrics.add(std::make_shared<jyppx::training::metrics::TopKAccuracy>(3));

        model->eval();
        jyppx::data::DataLoader test_loader(
            std::make_shared<jyppx::data::TensorDataset>(
                test_raw.images.view({-1, 784}), test_raw.labels.to(torch::kFloat32)),
            {.batch_size = 256});
        torch::NoGradGuard no_grad;
        for (const auto& batch : test_loader) {
            // The model lives on the training device; move the batch there.
            auto logits = model->forward(batch.data.to(device).view({-1, 1, 28, 28}));
            metrics.update(logits.cpu(), batch.targets);
        }
        for (const auto& [name, value] : metrics.compute()) {
            std::cout << "test " << name << ": " << value << "\n";
        }
        std::cout << "tensorboard logs: " << tb_dir << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "mnist example error: " << e.what() << '\n';
        return 1;
    }
}
