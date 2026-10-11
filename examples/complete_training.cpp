// Complete training example: exercises the framework end to end — custom
// dataset, dataset split, multi-worker DataLoader, AdamW + cosine schedule,
// gradient clipping, callbacks (progress, logging, TensorBoard, checkpoint,
// early stopping, LR monitor), metrics and checkpoint resume.

#include <jyppx/jyppx.hpp>

#include <filesystem>
#include <iostream>
#include <memory>

namespace {

using namespace jyppx;

// A 3-class spiral-ish classification problem.
class ClassifierMlp : public torch::nn::Module {
public:
    ClassifierMlp()
        : fc1(register_module("fc1", torch::nn::Linear(2, 64))),
          fc2(register_module("fc2", torch::nn::Linear(64, 64))),
          fc3(register_module("fc3", torch::nn::Linear(64, 3))) {}

    Tensor forward(const Tensor& input) {
        auto x = torch::relu(fc1->forward(input));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);   // {N, 3} logits
    }

    torch::nn::Linear fc1, fc2, fc3;
};

Tensor make_targets(const Tensor& x) {
    // Three clusters at (0,0), (2,0), (0,2).
    auto d0 = torch::norm(x - torch::tensor({0.0f, 0.0f}), 2, 1, true);
    auto d1 = torch::norm(x - torch::tensor({2.0f, 0.0f}), 2, 1, true);
    auto d2 = torch::norm(x - torch::tensor({0.0f, 2.0f}), 2, 1, true);
    auto stacked = torch::cat({d0, d1, d2}, /*dim=*/1);
    return stacked.argmin(1).to(torch::kFloat32).view({-1});
}

} // namespace

int main() {
    try {
        torch::manual_seed(2026);

        // ---- Data -----------------------------------------------------------
        auto inputs = torch::rand({1200, 2}) * 3.0f - 0.5f;
        auto targets = make_targets(inputs);
        auto dataset = std::make_shared<jyppx::data::TensorDataset>(inputs, targets);
        auto subsets = jyppx::data::random_split(dataset, std::vector<size_t>{1000, 200},
                                                 /*seed=*/42);
        jyppx::data::DataLoader train_loader(subsets[0],
                                             {.batch_size = 64, .shuffle = true,
                                              .num_workers = 2, .seed = 7});
        jyppx::data::DataLoader val_loader(subsets[1], {.batch_size = 200});

        // ---- Model / optimizer ----------------------------------------------
        auto model = std::make_shared<ClassifierMlp>();
        std::cout << jyppx::utils::model_summary(*model, "ClassifierMlp") << "\n";

        auto optimizer = std::make_shared<jyppx::optim::AdamW>(
            model->parameters(),
            jyppx::optim::AdamWOptions{.lr = 3e-3, .weight_decay = 1e-4});
        auto scheduler = std::make_shared<jyppx::optim::schedulers::CosineAnnealingLR>(
            optimizer, /*T_max=*/30, /*eta_min=*/1e-5);

        // ---- Callbacks -------------------------------------------------------
        const auto run_dir = std::filesystem::temp_directory_path() / "jyppx_complete";
        std::filesystem::create_directories(run_dir);
        jyppx::training::TrainerOptions options;
        options.max_epochs = 30;
        options.gradient_clip_val = 1.0;
        options.loss_fn = [](const Tensor& prediction, const Tensor& target) {
            return torch::nn::functional::cross_entropy(
                prediction, target.to(torch::kLong).view({-1}));
        };
        options.callbacks.push_back(
            std::make_shared<jyppx::training::callbacks::ProgressCallback>());
        options.callbacks.push_back(
            std::make_shared<jyppx::training::callbacks::LoggingCallback>(
                /*log_every_n_steps=*/10));
        options.callbacks.push_back(
            std::make_shared<jyppx::training::callbacks::LRMonitor>());
        options.callbacks.push_back(
            std::make_shared<jyppx::training::callbacks::TensorBoardCallback>(
                (run_dir / "tb").string(), /*log_every_n_steps=*/20));
        options.callbacks.push_back(
            std::make_shared<jyppx::training::callbacks::ModelCheckpoint>(
                (run_dir / "best.pt").string(), "val_loss", "min", 1));
        options.callbacks.push_back(
            std::make_shared<jyppx::training::callbacks::EarlyStopping>(
                "val_loss", /*patience=*/8, /*min_delta=*/1e-4));

        jyppx::training::Trainer trainer(model, optimizer, options);
        trainer.set_scheduler(scheduler);
        trainer.fit(train_loader, &val_loader);

        std::cout << "training stopped at epoch " << trainer.current_epoch()
                  << " (early_stopping=" << trainer.should_stop() << "), best "
                  << "val_loss=" << trainer.last_val_loss() << "\n";

        // ---- Evaluation with classification metrics ---------------------------
        jyppx::training::metrics::MetricCollection metrics;
        metrics.add(std::make_shared<jyppx::training::metrics::Accuracy>());
        metrics.add(std::make_shared<jyppx::training::metrics::F1Score>());
        metrics.add(std::make_shared<jyppx::training::metrics::Precision>());
        metrics.add(std::make_shared<jyppx::training::metrics::Recall>());

        trainer.load_checkpoint((run_dir / "best.pt").string());
        model->eval();
        torch::NoGradGuard no_grad;
        for (const auto& batch : val_loader) {
            auto logits = model->forward(batch.data);
            metrics.update(logits, batch.targets);
        }
        for (const auto& [name, value] : metrics.compute()) {
            std::cout << "val " << name << ": " << value << "\n";
        }

        std::error_code ec;
        std::filesystem::remove_all(run_dir, ec);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "complete training error: " << e.what() << '\n';
        return 1;
    }
}
