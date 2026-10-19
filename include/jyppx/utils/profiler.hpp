#ifndef JYPPX_UTILS_PROFILER_HPP
#define JYPPX_UTILS_PROFILER_HPP

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace jyppx {
namespace utils {

class ProfilerReport;

/**
 * @brief RAII timer that records one named wall-clock duration on destruction.
 *
 * The referenced report must outlive the timer. Instances cannot be copied and
 * are not thread-safe.
 */
class ScopedTimer {
 public:
  /**
   * @brief Starts timing a named scope immediately.
   * @param report Report that receives the elapsed duration.
   * @param name Stable display name of the measured scope.
   */
  ScopedTimer(ProfilerReport& report, std::string name);

  /** @brief Stops timing and records the elapsed duration. */
  ~ScopedTimer();

  /** @brief Copy construction is disabled to prevent duplicate recording. */
  ScopedTimer(const ScopedTimer&) = delete;

  /** @brief Copy assignment is disabled because the report cannot be rebound.
   */
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  /** Report that receives this timer's elapsed duration. */
  ProfilerReport& report_;

  /** Stable name under which this duration is accumulated. */
  std::string name_;

  /** Monotonic clock reading captured by the constructor. */
  std::chrono::steady_clock::time_point start_;
};

/**
 * @brief Aggregates named wall-clock durations for lightweight local profiling.
 *
 * Each scope stores total time, maximum time, and invocation count. The report
 * is intentionally unsynchronized; callers must guard shared access or use one
 * instance per thread.
 */
class ProfilerReport {
 public:
  /** @brief Accumulated statistics for one named scope. */
  struct Entry {
    /** Sum of recorded durations in milliseconds. */
    double total_ms = 0.0;

    /** Largest recorded duration in milliseconds. */
    double max_ms = 0.0;

    /** Number of durations included in the aggregate. */
    int64_t count = 0;
  };

  /**
   * @brief Adds one duration to a named scope.
   * @param name Stable scope name used as the aggregation key.
   * @param milliseconds Non-negative wall-clock duration in milliseconds.
   */
  void record(const std::string& name, double milliseconds) {
    auto& entry = entries_[name];
    entry.total_ms += milliseconds;
    entry.max_ms = std::max(entry.max_ms, milliseconds);
    ++entry.count;
  }

  /**
   * @brief Returns all scope statistics sorted by descending total duration.
   * @return Independent vector of name/statistics pairs.
   */
  std::vector<std::pair<std::string, Entry>> sorted() const {
    std::vector<std::pair<std::string, Entry>> items(entries_.begin(),
                                                     entries_.end());
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
      return a.second.total_ms > b.second.total_ms;
    });
    return items;
  }

  /**
   * @brief Renders a multi-line report with total, mean, maximum, and count.
   * @return Plain-text report sorted by descending total duration.
   */
  std::string summary() const {
    std::string text =
        "Profiler report (scope, total ms, mean ms, max ms, count)\n";
    for (const auto& [name, entry] : sorted()) {
      const double mean = entry.count == 0 ? 0.0 : entry.total_ms / entry.count;
      text += "  " + name + ": " + std::to_string(entry.total_ms) +
              " ms total, " + std::to_string(mean) + " ms mean, " +
              std::to_string(entry.max_ms) + " ms max, " +
              std::to_string(entry.count) + " calls\n";
    }
    return text;
  }

  /** @brief Removes every named aggregate from the report. */
  void clear() { entries_.clear(); }

 private:
  friend class ScopedTimer;

  /** Aggregates indexed by stable scope name. */
  std::map<std::string, Entry> entries_;
};

/**
 * @brief Starts timing a named scope.
 * @param report Report that receives the elapsed duration.
 * @param name Stable display name of the measured scope.
 */
inline ScopedTimer::ScopedTimer(ProfilerReport& report, std::string name)
    : report_(report),
      name_(std::move(name)),
      start_(std::chrono::steady_clock::now()) {}

/** @brief Stops timing and records the elapsed duration. */
inline ScopedTimer::~ScopedTimer() {
  const double milliseconds = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start_)
                                  .count();
  report_.record(name_, milliseconds);
}

}  // namespace utils
}  // namespace jyppx

#endif  // JYPPX_UTILS_PROFILER_HPP
