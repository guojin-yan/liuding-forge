// Knowledge distillation example (Phase 5): trains a wide teacher through the
// Trainer, then distills it into a narrow student with a combined objective
//   L = hard_weight * CE(student, targets)
//     + (1 - hard_weight) * T^2 * KL(teacher_T || student_T)
// and compares the student's accuracy before (plain hard-label baseline) and
// after distillation.

#include <jyppx/jyppx.hpp>

#include <iostream>
#include <memory>
#include <numeric>

namespace {

using namespace jyppx;

Tensor cluster_targets(const Tensor& x) {
    auto d0 = torch::norm(x - torch::tensor({0.0f, 0.0f}), 2, 1, true);
    auto d1 = torch::norm(x - torch::tensor({2.0f, 0.0f}), 2, 1, true);
    auto d2 = torch::norm(x - torch::tensor({0.0f, 2.0f}), 2, 1, true);
    return torch::cat({d0, d1, d2}, 1).argmin(1).to(torch::kFloat32).view({-1});
}

class Mlp : public torch::nn::Module {
public:
    Mlp(int hidden_layers, int width) {
        for (int i = 0; i < hidden_layers; ++i) {
            layers_.push_back(register_module(
                "layer" + std::to_string(i),
                torch::nn::Linear(i == 0 ? 2 : width, width)));
        }
        head_ = register_module("head", torch::nn::Linear(width, 3));
    }

    Tensor forward(const Tensor& input) {
        auto x = input;
        for (auto& layer : layers_) {
            x = torch::relu(layer->forward(x));
        }
        return head_->forward(x);
    }

private:
    std::vector<torch::nn::Linear> layers_;
    torch::nn::Linear head_{nullptr};
};

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    return torch::nn::functional::cross_entropy(
        logits, targets.to(torch::kLong).view({-1}));
}

template <typename ModuleT>
double evaluate(const std::shared_ptr<ModuleT>& model,
                const Tensor& inputs, const Tensor& targets) {
    torch::NoGradGuard no_grad;
    training::metrics::Accuracy accuracy;
    accuracy.update(model->forward(inputs), targets);
    return accuracy.compute();
}

// Unified student training: identical initialization, data order and epochs
// for both runs — only the loss composition differs (that is what makes the
// comparison fair).
double train_student(const std::shared_ptr<Mlp>& teacher, bool distill,
                     const Tensor& train_x, const Tensor& train_y,
                     const Tensor& val_x, const Tensor& val_y) {
    constexpr double kTemperature = 1.0;
    constexpr double kHardWeight = 0.3;
    constexpr int kEpochs = 80;
    constexpr int64_t kBatchSize = 64;

    torch::manual_seed(2026);
    auto student = std::make_shared<Mlp>(1, 8);
    auto optimizer = std::make_shared<jyppx::optim::Adam>(
        student->parameters(), jyppx::optim::AdamOptions{.lr = 3e-3});

    std::vector<int64_t> order(train_x.size(0));
    std::iota(order.begin(), order.end(), 0);
    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        std::shuffle(order.begin(), order.end(),
                     std::mt19937{static_cast<unsigned>(epoch + 1)});
        for (size_t start = 0; start < order.size(); start += kBatchSize) {
            const auto end = std::min<size_t>(start + kBatchSize, order.size());
            const auto index_tensor = torch::tensor(
                std::vector<int64_t>(order.begin() + static_cast<long>(start),
                                     order.begin() + static_cast<long>(end)),
                torch::TensorOptions().dtype(torch::kLong));
            const auto batch_x = train_x.index_select(0, index_tensor);
            const auto batch_y = train_y.index_select(0, index_tensor);

            optimizer->zero_grad();
            const auto logits = student->forward(batch_x);
            const auto hard = cross_entropy(logits, batch_y);
            Tensor loss = hard;
            if (distill) {
                const auto teacher_logits = teacher->forward(batch_x).detach();
                const auto soft = nn::functional::distillation_loss(
                    logits, teacher_logits, kTemperature);
                loss = kHardWeight * hard + (1.0 - kHardWeight) * soft;
            }
            loss.backward();
            optimizer->step();
        }
    }
    const double val_acc = evaluate(student, val_x, val_y);
    return val_acc;
}

} // namespace

int main() {
    try {
        torch::manual_seed(2026);
        auto train_x = torch::rand({800, 2}) * 3.0f - 0.5f;
        auto train_y = cluster_targets(train_x);
        auto val_x = torch::rand({200, 2}) * 3.0f - 0.5f;
        auto val_y = cluster_targets(val_x);

        jyppx::data::DataLoader train_loader(
            std::make_shared<jyppx::data::TensorDataset>(train_x, train_y),
            {.batch_size = 64, .shuffle = true, .seed = 3});

        // ---- Teacher: wide model, all 800 labeled samples -------------------
        auto teacher = std::make_shared<Mlp>(3, 96);
        auto teacher_optimizer = std::make_shared<jyppx::optim::Adam>(
            teacher->parameters(), jyppx::optim::AdamOptions{.lr = 3e-3});
        jyppx::training::Trainer teacher_trainer(teacher, teacher_optimizer,
                                                 {.max_epochs = 40});
        teacher_trainer.set_loss_fn(cross_entropy);
        teacher_trainer.fit(train_loader);
        std::cout << "teacher accuracy:      "
                  << evaluate(teacher, val_x, val_y) << "\n";

        // ---- Students: only 120 labeled samples each, identical setup -------
        const auto student_x = train_x.narrow(0, 0, 120);
        const auto student_y = train_y.narrow(0, 0, 120);

        const double baseline = train_student(teacher, false, student_x,
                                              student_y, val_x, val_y);
        std::cout << "student baseline (hard labels only): " << baseline << "\n";

        const double distilled = train_student(teacher, true, student_x,
                                               student_y, val_x, val_y);
        std::cout << "distilled student:                   " << distilled << "\n";

        if (distilled > baseline) {
            std::cout << "distillation transferred teacher knowledge: +"
                      << (distilled - baseline) * 100 << " points\n";
        } else if (distilled < baseline) {
            std::cout << "distillation underperformed on this split by "
                      << (baseline - distilled) * 100
                      << " points — the task is too easy for soft targets to add "
                         "signal; tune temperature/hard weight or use a harder task\n";
        } else {
            std::cout << "distillation matched the baseline\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "distillation example error: " << e.what() << '\n';
        return 1;
    }
}
