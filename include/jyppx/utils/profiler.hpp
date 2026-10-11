#ifndef JYPPX_UTILS_PROFILER_HPP
#define JYPPX_UTILS_PROFILER_HPP

#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace jyppx {
namespace utils {

/**
 * @brief Lightweight named-scope wall-clock profiler
 *
 * ScopedTimer instances record their duration into a process-global report;
 * ProfilerReport::summary() renders the aggregated statistics (total, mean,
 * max, count per scope). Not thread-safe — use one report per thread or
 * guard externally.
 *
 * @example
 * @code
 * ProfilerReport report;
 * {
 *     ScopedTimer timer(report, "forward");
 *     // ...
 * }
 * std::cout << report.summary();
 * @endcode
 */
class ProfilerReport;

class ScopedTimer {
public:
    ScopedTimer(ProfilerReport& report, std::string name);
    ~ScopedTimer();

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    ProfilerReport& report_;
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

class ProfilerReport {
public:
    struct Entry {
        double total_ms = 0.0;
        double max_ms = 0.0;
        int64_t count = 0;
    };

    /** @brief Records one duration in milliseconds. */
    void record(const std::string& name, double milliseconds) {
        auto& entry = entries_[name];
        entry.total_ms += milliseconds;
        entry.max_ms = std::max(entry.max_ms, milliseconds);
        ++entry.count;
    }

    /** @brief Aggregated per-scope statistics, sorted by total time. */
    std::vector<std::pair<std::string, Entry>> sorted() const {
        std::vector<std::pair<std::string, Entry>> items(entries_.begin(),
                                                         entries_.end());
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) {
                      return a.second.total_ms > b.second.total_ms;
                  });
        return items;
    }

    /** @brief Multi-line human-readable report. */
    std::string summary() const {
        std::string text = "Profiler report (scope, total ms, mean ms, max ms, count)\n";
        for (const auto& [name, entry] : sorted()) {
            const double mean =
                entry.count == 0 ? 0.0 : entry.total_ms / entry.count;
            text += "  " + name + ": " + std::to_string(entry.total_ms) + " ms total, " +
                    std::to_string(mean) + " ms mean, " +
                    std::to_string(entry.max_ms) + " ms max, " +
                    std::to_string(entry.count) + " calls\n";
        }
        return text;
    }

    void clear() { entries_.clear(); }

private:
    friend class ScopedTimer;
    std::map<std::string, Entry> entries_;
};

inline ScopedTimer::ScopedTimer(ProfilerReport& report, std::string name)
    : report_(report), name_(std::move(name)),
      start_(std::chrono::steady_clock::now()) {}

inline ScopedTimer::~ScopedTimer() {
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start_)
                          .count();
    report_.record(name_, ms);
}

} // namespace utils
} // namespace jyppx

#endif // JYPPX_UTILS_PROFILER_HPP
