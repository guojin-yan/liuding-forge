#include <jyppx/utils/logger.hpp>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace jyppx {
namespace utils {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

Logger::Logger() : level_(LogLevel::INFO) {}

Logger::~Logger() = default;

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

void Logger::set_file(const std::string& filepath) {
    auto file = std::make_unique<std::ofstream>(filepath, std::ios::out | std::ios::app);
    if (!file->is_open()) {
        throw std::runtime_error("Failed to open log file: " + filepath);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    file_ = std::move(file);
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

void Logger::log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (level < level_) {
        return;
    }

    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &now_time);
#else
    localtime_r(&now_time, &local_time);
#endif

    std::ostringstream formatted;
    formatted << '[' << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << '.'
              << std::setfill('0') << std::setw(3) << milliseconds.count() << "] ["
              << level_to_string(level) << "] " << message << '\n';
    const auto line = formatted.str();

    std::cout << line;
    std::cout.flush();
    if (file_ && file_->is_open()) {
        *file_ << line;
        file_->flush();
    }
}

std::string Logger::level_to_string(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::WARNING:
            return "WARNING";
        case LogLevel::ERROR:
            return "ERROR";
    }
    return "UNKNOWN";
}

} // namespace utils
} // namespace jyppx
