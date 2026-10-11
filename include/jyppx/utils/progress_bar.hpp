#ifndef JYPPX_UTILS_PROGRESS_BAR_HPP
#define JYPPX_UTILS_PROGRESS_BAR_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <utility>

namespace jyppx {
namespace utils {

/**
 * @brief Single-line console progress bar
 *
 * Renders a `[desc ####---- 12/50 (24%) postfix]` line on one terminal line
 * (carriage returns) and closes with a newline. Not thread-safe; use one
 * instance per stream.
 */
class ProgressBar {
public:
    explicit ProgressBar(int64_t total, int width = 30)
        : total_(total), width_(width) {
        if (total_ <= 0) {
            total_ = 1;
        }
        if (width_ < 5) {
            width_ = 5;
        }
    }

    /** @brief Advance the progress by n units and redraw. */
    void update(int64_t n = 1) {
        current_ = std::min(current_ + n, total_);
        draw();
    }

    void set_description(const std::string& description) {
        description_ = description;
        draw();
    }

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

    /** @brief Current position. */
    int64_t current() const { return current_; }

    /** @brief Fraction complete in [0, 1]. */
    double progress() const {
        return static_cast<double>(current_) / static_cast<double>(total_);
    }

    /** @brief Finish the line (newline) so following output starts clean. */
    void close() {
        if (!closed_) {
            std::cout << std::endl;
            closed_ = true;
        }
    }

    ~ProgressBar() { close(); }

    ProgressBar(const ProgressBar&) = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;

private:
    void draw() {
        const double fraction = progress();
        const int filled = static_cast<int>(std::round(fraction * width_));
        std::string bar;
        bar.reserve(static_cast<size_t>(width_));
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

    int64_t total_;
    int width_;
    int64_t current_ = 0;
    std::string description_;
    std::string postfix_;
    bool closed_ = false;
};

} // namespace utils
} // namespace jyppx

#endif // JYPPX_UTILS_PROGRESS_BAR_HPP
