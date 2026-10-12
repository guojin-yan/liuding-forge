// Memory footprint benchmark for loader and optimizer configurations.

#include <jyppx/jyppx.hpp>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#if defined(_WIN32)
#include <windows.h>
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

class MemoryDataset final : public jyppx::data::Dataset<jyppx::Tensor> {
public:
    explicit MemoryDataset(int count) : data_(torch::randn({count, 256})) {}
    jyppx::Tensor get(std::size_t index) const override { return data_[index]; }
    std::size_t size() const override { return data_.size(0); }
private:
    jyppx::Tensor data_;
};

} // namespace

int main() {
    constexpr int samples = 4096;
    std::cout << "workers,batch_size,bytes_before,bytes_after,delta_bytes\n";
    for (const std::size_t workers : {0U, 1U, 4U, 8U}) {
        const auto dataset = std::make_shared<MemoryDataset>(samples);
        const auto before = resident_bytes();
        jyppx::data::DataLoader loader(
            dataset, {.batch_size = 64, .num_workers = workers, .prefetch_factor = 2});
        std::size_t seen = 0;
        for (const auto& batch : loader) {
            seen += batch.data.size(0);
        }
        const auto after = resident_bytes();
        std::cout << workers << ",64," << before << ',' << after << ','
                  << (after >= before ? after - before : 0) << "\n";
        if (seen != samples) {
            return 1;
        }
    }
}
