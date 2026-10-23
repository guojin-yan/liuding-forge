// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_UTILS_PROGRESS_BAR_HPP
#define JYPPX_LIUDING_UTILS_PROGRESS_BAR_HPP
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <utility>

namespace jyppx::liuding {
namespace utils {

/**
 * @brief Renders mutable progress as one line on the standard output stream.
 *
 * Output has the form `description [####----] current/total (percent) postfix`.
 * Updates use carriage returns and close() terminates the line. Invalid totals
 * are normalized to one and widths below five are normalized to five. Instances
 * are not thread-safe.
 */
class ProgressBar {
 public:
  /**
   * @brief Creates an empty progress bar.
   * @param total Completion position; non-positive values are normalized to
   * one.
   * @param width Character width; values below five are normalized to five.
   */
  explicit ProgressBar(int64_t total, int width = 30)
      : total_(total), width_(width) {
    if (total_ <= 0) {
      total_ = 1;
    }
    if (width_ < 5) {
      width_ = 5;
    }
  }

  /**
   * @brief Moves the current position and redraws the line.
   * @param n Signed increment; the resulting position is saturated to
   *     `[0, total]`.
   */
  void update(int64_t n = 1) {
    if (n > 0) {
      current_ = n >= total_ - current_ ? total_ : current_ + n;
    } else if (n < 0) {
      current_ = n <= -current_ ? 0 : current_ + n;
    }
    draw();
  }

  /**
   * @brief Replaces the text displayed before the bar and redraws it.
   * @param description New display text.
   */
  void set_description(const std::string& description) {
    description_ = description;
    draw();
  }

  /**
   * @brief Replaces the ordered key/value text displayed after the percentage.
   * @param postfix Values joined as `key=value` pairs in key order.
   */
  void set_postfix(const std::map<std::string, std::string>& postfix) {
    postfix_.clear();
    std::string joined;
    for (const auto& [key, value] : postfix) {
      if (!joined.empty()) {
        joined += ", ";
      }
      joined += key + "=" + value;
    }
    postfix_ = std::move(joined);
    draw();
  }

  /**
   * @brief Returns the current saturated position.
   * @return Value in `[0, total]`.
   */
  int64_t current() const { return current_; }

  /**
   * @brief Returns the completed fraction.
   * @return Ratio in `[0, 1]`.
   */
  double progress() const {
    return static_cast<double>(current_) / static_cast<double>(total_);
  }

  /**
   * @brief Terminates the output line exactly once.
   *
   * Once closed, later state changes do not emit more output.
   */
  void close() {
    if (!closed_) {
      std::cout << std::endl;
      closed_ = true;
    }
  }

  /** @brief Closes the output line if it is still open. */
  ~ProgressBar() { close(); }

  /** @brief Copy construction is disabled for stream-owning state. */
  ProgressBar(const ProgressBar&) = delete;

  /** @brief Copy assignment is disabled for stream-owning state. */
  ProgressBar& operator=(const ProgressBar&) = delete;

 private:
  /** @brief Renders the current state unless the line is already closed. */
  void draw() {
    if (closed_) {
      return;
    }
    const double fraction = progress();
    const int filled = static_cast<int>(std::round(fraction * width_));
    std::string bar;
    bar.reserve(static_cast<std::size_t>(width_));
    for (int i = 0; i < width_; ++i) {
      bar += i < filled ? '#' : '-';
    }
    std::cout << '\r' << description_ << " [" << bar << "] " << current_ << '/'
              << total_ << " (" << static_cast<int>(fraction * 100.0) << "%)";
    if (!postfix_.empty()) {
      std::cout << ' ' << postfix_;
    }
    std::cout << std::flush;
    if (current_ >= total_) {
      close();
    }
  }

  /** Positive position at which the bar is complete. */
  int64_t total_;

  /** Character width of the rendered bar body. */
  int width_;

  /** Current position saturated to `[0, total_]`. */
  int64_t current_ = 0;

  /** Text rendered before the bar body. */
  std::string description_;

  /** Prejoined ordered key/value text rendered after the percentage. */
  std::string postfix_;

  /** Whether close() has already terminated the output line. */
  bool closed_ = false;
};

}  // namespace utils
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_UTILS_PROGRESS_BAR_HPP