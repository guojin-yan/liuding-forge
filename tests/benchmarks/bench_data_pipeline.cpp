// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Reproducible DataPipeline resource benchmark with structured CSV output.
//
// An in-memory TensorDataset loads in microseconds, which would hide worker
// scaling. SlowDataset therefore applies a calibrated per-sample CPU delay.
// Every timed repetition is emitted as an independent CSV row; aggregation,
// stability checks, and regression decisions belong to baseline_stats.py.

#include <jyppx/liuding/liuding.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// Psapi declarations depend on Windows SDK types declared by windows.h.
#include <psapi.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

/// Returns the current process resident set in bytes when supported.
std::uint64_t resident_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) ==
      0) {
    return 0;
  }
  return static_cast<std::uint64_t>(counters.WorkingSetSize);
#elif defined(__linux__)
  std::ifstream status("/proc/self/status");
  std::string key;
  std::uint64_t kb = 0;
  while (status >> key) {
    if (key == "VmRSS:") {
      status >> kb;
      break;
    }
    status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return kb * 1024;
#else
  return 0;
#endif
}

/// Accumulates factory concurrency and aggregate busy time.
class WorkerMetrics final {
 public:
  /// Records one active factory invocation for this scope's lifetime.
  class Scope final {
   public:
    /// Starts one timed factory interval.
    explicit Scope(WorkerMetrics& metrics)
        : metrics_(metrics), start_(Clock::now()) {
      const auto active = metrics_.active_.fetch_add(1) + 1;
      auto peak = metrics_.peak_active_.load();
      while (active > peak &&
             !metrics_.peak_active_.compare_exchange_weak(peak, active)) {
      }
    }

    /// Finishes the interval and contributes its duration.
    ~Scope() {
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               Clock::now() - start_)
                               .count();
      metrics_.busy_ns_.fetch_add(static_cast<std::uint64_t>(elapsed));
      metrics_.active_.fetch_sub(1);
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

   private:
    WorkerMetrics& metrics_;   ///< Aggregate receiving this interval.
    Clock::time_point start_;  ///< Factory invocation start time.
  };

  /// Returns aggregate factory busy time across execution lanes.
  std::uint64_t busy_ns() const noexcept { return busy_ns_.load(); }

  /// Returns the maximum simultaneous factory invocation count.
  std::size_t peak_active() const noexcept { return peak_active_.load(); }

 private:
  friend class Scope;

  std::atomic<std::size_t> active_{0};  ///< Currently active invocations.
  std::atomic<std::size_t> peak_active_{
      0};                                  ///< Maximum concurrent invocations.
  std::atomic<std::uint64_t> busy_ns_{0};  ///< Sum of invocation durations.
};

/// Dataset that simulates a bounded preprocessing cost for every sample.
class SlowDataset final : public jyppx::liuding::data::Dataset<jyppx::liuding::data::Example> {
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
  jyppx::liuding::data::Example get(std::size_t index) const override {
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

  /// Median successful `next()` latency in milliseconds.
  double p50_batch_latency_ms;

  /// Nearest-rank p95 successful `next()` latency in milliseconds.
  double p95_batch_latency_ms;

  /// Mean scheduled-but-uncommitted batch count sampled after commits.
  double mean_outstanding_batches;

  /// Maximum scheduled-but-uncommitted batch count sampled after commits.
  std::size_t peak_outstanding_batches;

  /// Aggregate factory busy time divided by elapsed execution-lane capacity.
  double worker_utilization;

  /// Maximum simultaneous factory invocation count.
  std::size_t peak_active_workers;

  /// Resident bytes immediately before the measured epoch.
  std::uint64_t resident_before_bytes;

  /// Maximum resident bytes observed at batch boundaries.
  std::uint64_t peak_resident_bytes;
};

/// Returns milliseconds elapsed since `start`.
double milliseconds_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

/// Returns the nearest-rank percentile from millisecond samples.
double percentile(std::vector<double> values, double probability) {
  std::sort(values.begin(), values.end());
  const auto rank = static_cast<std::size_t>(
      std::ceil(probability * static_cast<double>(values.size())));
  return values[std::max<std::size_t>(1, rank) - 1];
}

/// Parses a finite positive floating-point command-line value.
std::optional<double> parse_positive_double(std::string_view text) {
  std::string owned(text);
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(owned.c_str(), &end);
  if (errno == ERANGE || end != owned.c_str() + owned.size() ||
      !std::isfinite(value) || value <= 0.0) {
    return std::nullopt;
  }
  return value;
}

/// Parses a strictly positive decimal integer command-line value.
std::optional<int> parse_positive_int(std::string_view text) {
  int value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value <= 0) {
    return std::nullopt;
  }
  return value;
}

/// Parses a bounded non-negative worker count command-line value.
std::optional<std::size_t> parse_worker_count(std::string_view text) {
  std::size_t value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() ||
      value > 1024) {
    return std::nullopt;
  }
  return value;
}

/// Measures one or more epochs for a fixed DataPipeline worker count.
std::vector<BenchResult> bench_workers(
    const std::shared_ptr<jyppx::liuding::data::Dataset<jyppx::liuding::data::Example>>& dataset,
    int batches, int batch_size, std::size_t workers, int repeats,
    std::size_t prefetch_batches) {
  std::vector<BenchResult> results;
  results.reserve(static_cast<std::size_t>(repeats));
  const auto expected =
      static_cast<std::size_t>(batches) * static_cast<std::size_t>(batch_size);
  for (int repeat = 0; repeat < repeats; ++repeat) {
    auto worker_metrics = std::make_shared<WorkerMetrics>();
    auto sampler = std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
        dataset->size());
    jyppx::liuding::data::DataPipeline pipeline(
        std::move(sampler),
        jyppx::liuding::data::DataPipelineOptions{
            .pipeline_id = "benchmark.data_pipeline",
            .dataset_version = "benchmark.slow_dataset/v1",
            .dataset_size = dataset->size(),
            .schema = jyppx::liuding::data::BatchSchema::any("benchmark.batch", 1),
            .batch_factory_id = "benchmark.slow_dataset.collate/v1",
            .batch_size = static_cast<std::size_t>(batch_size),
            .num_workers = workers,
            .prefetch_batches = prefetch_batches},
        [dataset, worker_metrics](std::span<const std::size_t> indices,
                                  const jyppx::liuding::data::DataWorkerContext&,
                                  std::stop_token) {
          WorkerMetrics::Scope measurement(*worker_metrics);
          std::vector<jyppx::liuding::Tensor> data;
          std::vector<jyppx::liuding::Tensor> targets;
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
    const auto resident_before = resident_bytes();
    auto peak_resident = resident_before;
    const auto start = Clock::now();
    pipeline.set_epoch(0);
    std::size_t seen = 0;
    std::vector<double> batch_latencies_ms;
    batch_latencies_ms.reserve(static_cast<std::size_t>(batches));
    std::uint64_t outstanding_sum = 0;
    std::size_t outstanding_samples = 0;
    std::size_t peak_outstanding = 0;
    while (true) {
      const auto batch_start = Clock::now();
      const auto batch = pipeline.next();
      if (!batch) {
        break;
      }
      batch_latencies_ms.push_back(milliseconds_since(batch_start));
      seen += batch->sample_ids.size();
      const auto outstanding = pipeline.outstanding_batches();
      outstanding_sum += outstanding;
      ++outstanding_samples;
      peak_outstanding = std::max(peak_outstanding, outstanding);
      peak_resident = std::max(peak_resident, resident_bytes());
    }
    pipeline.close();
    const double elapsed = milliseconds_since(start);
    peak_resident = std::max(peak_resident, resident_bytes());
    if (seen != expected) {
      std::cerr << "configuration with " << workers << " workers produced "
                << seen << " samples, expected " << expected << '\n';
      return {};
    }
    if (batch_latencies_ms.empty() || outstanding_samples == 0) {
      std::cerr << "configuration with " << workers
                << " workers produced no measurable batches\n";
      return {};
    }
    const auto execution_lanes = std::max<std::size_t>(1, workers);
    const auto elapsed_ns = elapsed * 1'000'000.0;
    results.push_back(
        {.workers = workers,
         .repeat = repeat,
         .samples = seen,
         .elapsed_ms = elapsed,
         .p50_batch_latency_ms = percentile(batch_latencies_ms, 0.50),
         .p95_batch_latency_ms = percentile(batch_latencies_ms, 0.95),
         .mean_outstanding_batches = static_cast<double>(outstanding_sum) /
                                     static_cast<double>(outstanding_samples),
         .peak_outstanding_batches = peak_outstanding,
         .worker_utilization =
             static_cast<double>(worker_metrics->busy_ns()) /
             (elapsed_ns * static_cast<double>(execution_lanes)),
         .peak_active_workers = worker_metrics->peak_active(),
         .resident_before_bytes = resident_before,
         .peak_resident_bytes = peak_resident});
  }
  return results;
}

/// Prints accepted command-line options.
void print_help() {
  std::cerr << "usage: liuding-forge-bench-data-pipeline [--work-ms ms] "
               "[--batches n] "
               "[--batch-size n] [--repeats n] [--workers n]\n";
}

}  // namespace

int main(int argc, char** argv) {
  double work_ms = 2.0;
  int batches = 64;
  int batch_size = 16;
  int repeats = 3;
  std::optional<std::size_t> selected_workers;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value_of = [&]() -> std::optional<std::string_view> {
      if (i + 1 >= argc) {
        return std::nullopt;
      }
      return argv[++i];
    };
    if (arg == "--work-ms") {
      const auto text = value_of();
      const auto value = text ? parse_positive_double(*text) : std::nullopt;
      if (!value) {
        print_help();
        return 2;
      }
      work_ms = *value;
    } else if (arg == "--batches") {
      const auto text = value_of();
      const auto value = text ? parse_positive_int(*text) : std::nullopt;
      if (!value) {
        print_help();
        return 2;
      }
      batches = *value;
    } else if (arg == "--batch-size") {
      const auto text = value_of();
      const auto value = text ? parse_positive_int(*text) : std::nullopt;
      if (!value) {
        print_help();
        return 2;
      }
      batch_size = *value;
    } else if (arg == "--repeats") {
      const auto text = value_of();
      const auto value = text ? parse_positive_int(*text) : std::nullopt;
      if (!value) {
        print_help();
        return 2;
      }
      repeats = *value;
    } else if (arg == "--workers") {
      const auto text = value_of();
      const auto workers = text ? parse_worker_count(*text) : std::nullopt;
      if (!workers) {
        print_help();
        return 2;
      }
      selected_workers = *workers;
    } else {
      print_help();
      return 2;
    }
  }
  if (static_cast<std::uint64_t>(batches) *
          static_cast<std::uint64_t>(batch_size) >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    print_help();
    return 2;
  }

  const int samples = batches * batch_size;
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  const auto dataset =
      std::make_shared<SlowDataset>(samples, /*features=*/16, work_ms);
  std::cout << "workers,repeat,samples,batch_size,features,work_ms,"
               "elapsed_ms,samples_per_second,p50_batch_latency_ms,"
               "p95_batch_latency_ms,mean_outstanding_batches,"
               "peak_outstanding_batches,worker_utilization,"
               "peak_active_workers,resident_before_bytes,"
               "peak_resident_bytes,resident_delta_bytes\n";
  std::cout << std::fixed << std::setprecision(3);
  const std::vector<std::size_t> worker_counts =
      selected_workers ? std::vector{*selected_workers}
                       : std::vector<std::size_t>{0, 1, 2, 4, 8};
  for (const std::size_t workers : worker_counts) {
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
                << result.elapsed_ms << ',' << throughput << ','
                << result.p50_batch_latency_ms << ','
                << result.p95_batch_latency_ms << ','
                << result.mean_outstanding_batches << ','
                << result.peak_outstanding_batches << ','
                << result.worker_utilization << ','
                << result.peak_active_workers << ','
                << result.resident_before_bytes << ','
                << result.peak_resident_bytes << ','
                << (result.peak_resident_bytes >= result.resident_before_bytes
                        ? result.peak_resident_bytes -
                              result.resident_before_bytes
                        : 0)
                << '\n';
    }
  }
  return 0;
}
