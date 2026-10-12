// DataLoader throughput benchmark.
//
// An in-memory TensorDataset loads in microseconds, which would hide any
// multi-worker gain. Real data pipelines pay for decode/preprocessing per
// sample, so the benchmark dataset simulates that cost with a fixed amount
// of matmul work per sample before returning a row. The per-configuration
// timings therefore measure the loader's parallelism, not tensor indexing.
//
// Run with the benchmark build (JYPPX_BUILD_BENCHMARKS=ON):
//   build-framework/tests/benchmarks/Release/jyppx_bench_dataloader.exe
//
// Optional arguments (all times in milliseconds):
//   --work-ms <ms>       simulated per-sample preprocessing cost (default 2)
//   --batches <n>        batches per epoch (default 64)
//   --batch-size <n>     samples per batch (default 16)
//   --repeats <n>        timed epochs per configuration (default 3)

#include <jyppx/jyppx.hpp>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using jyppx::Tensor;
using Clock = std::chrono::steady_clock;

// Dataset whose get() burns ~work_ms of CPU per sample on matmuls, then
// returns a row of an underlying payload tensor.
class SlowDataset : public jyppx::data::Dataset<jyppx::data::Example> {
public:
    SlowDataset(int samples, int features, double work_ms, uint64_t seed = 123)
        : features_(features),
          targets_(torch::randn({samples, 1},
                                torch::TensorOptions().dtype(torch::kFloat32))),
          workload_(torch::randn({64, 64}, torch::TensorOptions().dtype(torch::kFloat32)
                                                        .requires_grad(false))),
          payload_(torch::randn({samples, features},
                                torch::TensorOptions().dtype(torch::kFloat32))) {
        torch::manual_seed(seed);
        work_per_sample_ = std::chrono::microseconds(static_cast<int64_t>(work_ms * 1000));
        // Calibrate the iteration count once so each sample burns ~work_ms
        // regardless of the machine.
        const auto start = Clock::now();
        double sink = 0.0;
        int iterations = 0;
        do {
            sink += workload_.sum().item<float>();
            ++iterations;
        } while (Clock::now() - start < work_per_sample_);
        sink_iterations_ = iterations;
        (void)sink;
    }

    jyppx::data::Example get(size_t index) const override {
        const auto start = Clock::now();
        double sink = 0.0;
        for (int i = 0; i < sink_iterations_; ++i) {
            sink += workload_.sum().item<float>();
        }
        // Keep the sink observable so the loop is not optimized away.
        if (sink == 12345.678f) {
            std::cerr << sink;
        }
        while (Clock::now() - start < work_per_sample_) {
            std::this_thread::yield();
        }
        // Return independent samples and a single target column matching the
        // benchmark model output, so the loss does not rely on broadcasting.
        return {payload_[static_cast<int64_t>(index)].clone(),
                targets_[static_cast<int64_t>(index)].clone()};
    }

    size_t size() const override { return payload_.size(0); }

    int features() const { return features_; }

private:
    int features_;
    torch::Tensor targets_;
    torch::Tensor workload_;
    torch::Tensor payload_;
    std::chrono::microseconds work_per_sample_{};
    int sink_iterations_ = 1;
};

// Minimal model for the training-throughput reference line.
class BenchLinearModel : public torch::nn::Module {
public:
    BenchLinearModel()
        : linear(register_module("linear", torch::nn::Linear(16, 1))) {}
    Tensor forward(const Tensor& input) { return linear->forward(input); }
    torch::nn::Linear linear{nullptr};
};

double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct BenchResult {
    std::size_t workers;
    double best_ms = 0.0; // best of repeats: least scheduler noise
    bool valid = false;
};

BenchResult bench_workers(std::shared_ptr<jyppx::data::Dataset<jyppx::data::Example>> dataset,
                          int batches, int batch_size, std::size_t workers,
                          int repeats, std::size_t prefetch_factor) {
    jyppx::data::DataLoaderOptions options;
    options.batch_size = static_cast<std::size_t>(batch_size);
    options.num_workers = workers;
    options.prefetch_factor = prefetch_factor;

    BenchResult result;
    result.workers = workers;

    for (int repeat = 0; repeat < repeats; ++repeat) {
        jyppx::data::DataLoader<jyppx::data::Example> loader(dataset, options);

        const auto start = Clock::now();
        std::size_t seen = 0;
        for (const auto& batch : loader) {
            seen += batch.data.size(0);
        }
        const double elapsed = milliseconds_since(start);
        const std::size_t expected = static_cast<std::size_t>(batches) *
                                     static_cast<std::size_t>(batch_size);
        if (seen != expected) {
            std::cerr << "configuration with " << workers
                      << " workers produced " << seen << " samples, expected "
                      << expected << "\n";
            return result;
        }
        if (repeat == 0 || elapsed < result.best_ms) {
            result.best_ms = elapsed;
        }
        result.valid = true;
    }
    return result;
}

void print_help() {
    std::cout
        << "usage: jyppx_bench_dataloader [--work-ms ms] [--batches n] "
           "[--batch-size n] [--repeats n]\n";
}

} // namespace

int main(int argc, char** argv) {
    double work_ms = 2.0;
    int batches = 64;
    int batch_size = 16;
    int repeats = 3;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value_of = [&]() -> double {
            if (i + 1 >= argc) {
                print_help();
                std::exit(2);
            }
            return std::atof(argv[++i]);
        };
        if (arg == "--work-ms") {
            work_ms = value_of();
        } else if (arg == "--batches") {
            batches = static_cast<int>(value_of());
        } else if (arg == "--batch-size") {
            batch_size = static_cast<int>(value_of());
        } else if (arg == "--repeats") {
            repeats = static_cast<int>(value_of());
        } else {
            print_help();
            return 2;
        }
    }
    if (batches <= 0 || batch_size <= 0 || repeats <= 0 || work_ms < 0.0) {
        print_help();
        return 2;
    }

    const int hardware_threads = static_cast<int>(std::thread::hardware_concurrency());
    std::cout << "JYPPX DataLoader benchmark\n"
              << "  batches=" << batches << "  batch_size=" << batch_size
              << "  work_ms=" << work_ms << "  repeats=" << repeats
              << "  hardware_threads=" << hardware_threads << "\n\n";

    const int samples = batches * batch_size;
    auto dataset = std::make_shared<SlowDataset>(samples, /*features=*/16, work_ms);
    std::cout << "dataset: " << samples << " samples x " << dataset->features()
              << " features\n\n";

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "workers | ms/epoch (best of " << repeats << ") | speedup\n"
              << "--------+----------------------+--------\n";

    double baseline_ms = 0.0;
    std::vector<BenchResult> results;
    for (const std::size_t workers : {std::size_t{0}, std::size_t{1}, std::size_t{2},
                                      std::size_t{4}, std::size_t{8}}) {
        auto result =
            bench_workers(dataset, batches, batch_size, workers, repeats, 2);
        if (!result.valid) {
            std::cerr << "worker configuration " << workers << " failed\n";
            return 1;
        }
        if (workers == 0) {
            baseline_ms = result.best_ms;
        }
        results.push_back(result);
    }

    for (const auto& result : results) {
        const double speedup = baseline_ms / result.best_ms;
        std::cout << std::setw(7) << result.workers << " | " << std::setw(20)
                  << result.best_ms << " | " << std::setw(5) << speedup << "x\n";
    }

    // Training-throughput reference: one epoch of real optimizer steps over
    // the slow loader (single thread), for context.
    {
        auto dataset_ptr = std::make_shared<SlowDataset>(samples, 16, work_ms);
        auto model = std::make_shared<BenchLinearModel>();
        auto optimizer = std::make_shared<jyppx::optim::SGD>(
            model->parameters(), jyppx::optim::SGDOptions{.lr = 0.01});
        jyppx::training::Trainer trainer(
            model, optimizer, jyppx::training::TrainerOptions{.max_epochs = 1});

        jyppx::data::DataLoaderOptions options;
        options.batch_size = static_cast<std::size_t>(batch_size);
        jyppx::data::DataLoader<jyppx::data::Example> loader(dataset_ptr, options);

        const auto start = Clock::now();
        trainer.fit(loader);
        std::cout << "\ntraining reference (1 worker): "
                  << milliseconds_since(start) << " ms, final loss="
                  << trainer.last_train_loss() << "\n";
    }

    return 0;
}
