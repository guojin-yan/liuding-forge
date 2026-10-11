#ifndef JYPPX_UTILS_LOGGER_HPP
#define JYPPX_UTILS_LOGGER_HPP

#include <string>
#include <fstream>
#include <memory>
#include <mutex>

namespace jyppx {
namespace utils {

/**
 * @brief Log level enum
 */
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

/**
 * @brief Simple logger singleton
 */
class Logger {
public:
    /**
     * @brief Get logger instance
     */
    static Logger& instance();

    /**
     * @brief Set log level
     */
    void set_level(LogLevel level);

    /**
     * @brief Set log file
     */
    void set_file(const std::string& filepath);

    /**
     * @brief Log debug message
     */
    void debug(const std::string& message);

    /**
     * @brief Log info message
     */
    void info(const std::string& message);

    /**
     * @brief Log warning message
     */
    void warning(const std::string& message);

    /**
     * @brief Log error message
     */
    void error(const std::string& message);

private:
    Logger();
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void log(LogLevel level, const std::string& message);
    std::string level_to_string(LogLevel level) const;

    LogLevel level_;
    std::unique_ptr<std::ofstream> file_;
    std::mutex mutex_;
};

// Convenience macros
#define LOG_DEBUG(msg) jyppx::utils::Logger::instance().debug(msg)
#define LOG_INFO(msg) jyppx::utils::Logger::instance().info(msg)
#define LOG_WARNING(msg) jyppx::utils::Logger::instance().warning(msg)
#define LOG_ERROR(msg) jyppx::utils::Logger::instance().error(msg)

} // namespace utils
} // namespace jyppx

#endif // JYPPX_UTILS_LOGGER_HPP
