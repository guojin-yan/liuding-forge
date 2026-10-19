// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Reproducible DataPipeline throughput benchmark with structured CSV output.
//
// An in-memory TensorDataset loads in microseconds, which would hide worker
// scaling. SlowDataset therefore applies a calibrated per-sample CPU delay.
// Every timed repetition is emitted as an independent CSV row; aggregation,
// stability checks, and regression decisions belong to baseline_stats.py.

#include <jyppx/jyppx.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <stop_token>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

/// Dataset that simulates a bounded preprocessing cost for every sample.
class SlowDataset final : public jyppx::data::Dataset<jyppx::data::Example> {
 public:
  /// Constructs a deterministic payload and calibrates the requested delay.
  ///
  /// @param samples Number of samples exposed by the dataset.
  /// @param features Number of floating-point features per sample.
  /// @param work_ms Minimum CPU work duration applied by `get()`.
  /// @param seed Seed used to construct deterministic payload tensors.
  SlowDataset(int samples, int features, double work_ms,
              std::uint64_t seed = 123)
      : features_(features) {
    torch::manual_seed(seed);
    targets_ = torch::randn({samples, 1},
                            torch::TensorOptions().dtype(torch::kFloat32));
    workload_ = torch::randn(
        {64, 64},
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false));
    payload_ = torch::randn({samples, features},
                            torch::TensorOptions().dtype(torch::kFloat32));
    work_per_sample_ =
        std::chrono::microseconds(static_cast<std::int64_t>(work_ms * 1000));
    const auto start = Clock::now();
    double sink = 0.0;
    int iterations = 0;
    do {
      sink += workload_.sum().item<float>();
      ++iterations;
    } while (Clock::now() - start < work_per_sample_);
    sink_iterations_ = iterations;
    static_cast<void>(sink);
  }

  /// Returns one independently owned example after the calibrated work.
  jyppx::data::Example get(std::size_t index) const override {
    const auto start = Clock::now();
    double sink = 0.0;
    for (int i = 0; i < sink_iterations_; ++i) {
      sink += workload_.sum().item<float>();
    }
    if (sink == 12345.678F) {
      std::cerr << sink;
    }
    while (Clock::now() - start < work_per_sample_) {
      std::this_thread::yield();
    }
    return {payload_[static_cast<std::int64_t>(index)].clone(),
            targets_[static_cast<std::int64_t>(index)].clone()};
  }

  /// Returns the number of samples exposed by the dataset.
  std::size_t size() const override {
    return static_cast<std::size_t>(payload_.size(0));
  }

  /// Returns the number of features contained in each sample.
  int features() const { return features_; }

 private:
  /// Number of features in each payload row.
  int features_;

  /// Target tensor paired with the payload rows.
  torch::Tensor targets_;

  /// Small tensor repeatedly reduced to simulate preprocessing work.
  torch::Tensor workload_;

  /// In-memory feature rows returned by `get()`.
  torch::Tensor payload_;

  /// Minimum simulated preprocessing time per sample.
  std::chrono::microseconds work_per_sample_{};

  /// Calibrated number of reductions executed before the duration guard.
  int sink_iterations_ = 1;
};

/// One raw timing sample for a worker configuration.
struct BenchResult {
  /// Configured DataPipeline worker count.
  std::size_t workers;

  /// Zero-based repetition index within the process.
  int repeat;

  /// Number of samples observed during the epoch.
  std::size_t samples;

  /// Wall-clock duration of the epoch in milliseconds.
  double elapsed_ms;
};

/// Returns milliseconds elapsed since `start`.
double milliseconds_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

/// Measures one or more epochs for a fixed DataPipeline worker count.
std::vector<BenchResult> bench_workers(
    const std::shared_ptr<jyppx::data::Dataset<jyppx::data::Example>>& dataset,
    int batches, int batch_size, std::size_t workers, int repeats,
    std::size_t prefetch_factor) {
  std::vector<BenchResult> results;
  results.reserve(static_cast<std::size_t>(repeats));
  const auto expected =
      static_cast<std::size_t>(batches) * static_cast<std::size_t>(batch_size);
  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto sampler = std::make_shared<jyppx::data::samplers::SequentialSampler>(
        dataset->size());
    jyppx::data::DataPipeline pipeline(
        std::move(sampler),
        jyppx::data::DataPipelineOptions{
            .pipeline_id = "benchmark.data_pipeline",
            .dataset_version = "benchmark.slow_dataset/v1",
            .dataset_size = dataset->size(),
            .schema = jyppx::data::BatchSchema::any("benchmark.batch", 1),
            .batch_factory_id = "benchmark.slow_dataset.collate/v1",
            .batch_size = static_cast<std::size_t>(batch_size),
            .num_workers = workers,
            .prefetch_batches = prefetch_factor},
        [dataset](std::span<const std::size_t> indices,
                  const jyppx::data::DataWorkerContext&, std::stop_token) {
          std::vector<jyppx::Tensor> data;
          std::vector<jyppx::Tensor> targets;
          data.reserve(indices.size());
          targets.reserve(indices.size());
          for (const auto index : indices) {
            const auto sample = dataset->get(index);
            data.push_back(sample.data);
            targets.push_back(sample.target);
          }
          return c10::ivalue::Tuple::create(
              {torch::stack(data), torch::stack(targets)});
        });
    pipeline.set_epoch(0);
    const auto start = Clock::now();
    std::size_t seen = 0;
    while (const auto batch = pipeline.next()) {
      seen += batch->sample_ids.size();
    }
    pipeline.close();
    const double elapsed = milliseconds_since(start);
    if (seen != expected) {
      std::cerr << "configuration with " << workers << " workers produced "
                << seen << " samples, expected " << expected << '\n';
      return {};
    }
    results.push_back({workers, repeat, seen, elapsed});
  }
  return results;
}

/// Prints accepted command-line options.
void print_help() {
  std::cerr << "usage: jyppx_bench_dataloader [--work-ms ms] [--batches n] "
               "[--batch-size n] [--repeats n]\n";
}

}  // namespace

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
  if (batches <= 0 || batch_size <= 0 || repeats <= 0 || work_ms <= 0.0) {
    print_help();
    return 2;
  }

  const int samples = batches * batch_size;
  const auto dataset =
      std::make_shared<SlowDataset>(samples, /*features=*/16, work_ms);
  std::cout << "workers,repeat,samples,batch_size,features,work_ms,"
               "elapsed_ms,samples_per_second\n";
  std::cout << std::fixed << std::setprecision(3);
  for (const std::size_t workers :
       {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{4},
        std::size_t{8}}) {
    const auto results =
        bench_workers(dataset, batches, batch_size, workers, repeats, 2);
    if (results.size() != static_cast<std::size_t>(repeats)) {
      return 1;
    }
    for (const auto& result : results) {
      const double throughput =
          static_cast<double>(result.samples) / (result.elapsed_ms / 1000.0);
      std::cout << result.workers << ',' << result.repeat << ','
                << result.samples << ',' << batch_size << ','
                << dataset->features() << ',' << work_ms << ','
                << result.elapsed_ms << ',' << throughput << '\n';
    }
  }
  return 0;
}
