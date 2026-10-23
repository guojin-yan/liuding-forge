// Memory footprint benchmark for data pipeline and optimizer configurations.

#include <jyppx/liuding/liuding.hpp>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>

// Psapi declarations depend on Windows SDK types declared by windows.h.
#include <psapi.h>
#elif defined(__linux__)
#include <fstream>
#endif

namespace {

std::uint64_t resident_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
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

std::uint64_t peak_resident_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters));
  return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
#elif defined(__linux__)
  std::ifstream status("/proc/self/status");
  std::string key;
  std::uint64_t kb = 0;
  while (status >> key) {
    if (key == "VmHWM:") {
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

class MemoryDataset final : public jyppx::liuding::data::Dataset<jyppx::liuding::Tensor> {
 public:
  explicit MemoryDataset(int count) : data_(torch::randn({count, 256})) {}
  jyppx::liuding::Tensor get(std::size_t index) const override { return data_[index]; }
  std::size_t size() const override { return data_.size(0); }

 private:
  jyppx::liuding::Tensor data_;
};

}  // namespace

int main(int argc, char** argv) {
  constexpr int samples = 4096;
  std::vector<std::size_t> worker_counts{0U, 1U, 4U, 8U};
  if (argc == 3 && std::string(argv[1]) == "--workers") {
    char* end = nullptr;
    const auto workers = std::strtoull(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || workers > 1024) {
      std::cerr << "worker count must be an integer from 0 to 1024\n";
      return 2;
    }
    worker_counts = {static_cast<std::size_t>(workers)};
  } else if (argc != 1) {
    std::cerr << "usage: liuding-forge-bench-memory-usage [--workers count]\n";
    return 2;
  }

  std::cout
      << "workers,batch_size,bytes_before,bytes_after,peak_bytes,delta_bytes\n";
  for (const std::size_t workers : worker_counts) {
    const auto dataset = std::make_shared<MemoryDataset>(samples);
    const auto before = resident_bytes();
    auto sampler =
        std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(samples);
    jyppx::liuding::data::DataPipeline pipeline(
        std::move(sampler),
        jyppx::liuding::data::DataPipelineOptions{
            .pipeline_id = "benchmark.memory_pipeline",
            .dataset_version = "benchmark.memory_dataset/v1",
            .dataset_size = samples,
            .schema =
                jyppx::liuding::data::BatchSchema::any("benchmark.memory.batch", 1),
            .batch_factory_id = "benchmark.memory_dataset.collate/v1",
            .batch_size = 64,
            .num_workers = workers,
            .prefetch_batches = 2},
        [dataset](std::span<const std::size_t> indices,
                  const jyppx::liuding::data::DataWorkerContext&, std::stop_token) {
          std::vector<jyppx::liuding::Tensor> samples;
          samples.reserve(indices.size());
          for (const auto index : indices) {
            samples.push_back(dataset->get(index));
          }
          return torch::stack(samples);
        });
    pipeline.set_epoch(0);
    std::size_t seen = 0;
    while (const auto batch = pipeline.next()) {
      seen += batch->sample_ids.size();
    }
    pipeline.close();
    const auto after = resident_bytes();
    const auto peak = peak_resident_bytes();
    std::cout << workers << ",64," << before << ',' << after << ',' << peak
              << ',' << (peak >= before ? peak - before : 0) << "\n";
    if (seen != samples) {
      return 1;
    }
  }
}
