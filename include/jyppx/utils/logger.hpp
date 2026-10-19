#ifndef JYPPX_UTILS_LOGGER_HPP
#define JYPPX_UTILS_LOGGER_HPP

#include <jyppx/foundation/error.hpp>

#include <fstream>
#include <memory>
#include <mutex>
#include <string>

namespace jyppx {
namespace utils {

/**
 * @brief Severity threshold used by `Logger`.
 */
enum class LogLevel {
  DEBUG,    ///< Diagnostic messages useful during development.
  INFO,     ///< Normal progress and lifecycle messages.
  WARNING,  ///< Recoverable condition that needs attention.
  ERROR     ///< Operation failure that was handled by the caller.
};

/**
 * @brief Thread-safe process logger with optional append-only file output.
 *
 * Messages at or above the configured threshold are written to standard
 * output and, when configured, to the selected file. File configuration
 * failures use the stable `jyppx::IoError` contract.
 */
class Logger {
 public:
  /**
   * @brief Returns the process-wide logger instance.
   * @return Singleton logger shared by the current process.
   */
  static Logger& instance();

  /**
   * @brief Sets the minimum severity emitted by this logger.
   * @param level Lowest severity that may be written to configured sinks.
   */
  void set_level(LogLevel level);

  /**
   * @brief Opens an append-only log file.
   * @param filepath Path to a writable file.
   * @throws jyppx::IoError If the path cannot be opened for appending.
   */
  void set_file(const std::string& filepath);

  /**
   * @brief Emits a debug message when the threshold permits it.
   * @param message Plain-text message to write.
   */
  void debug(const std::string& message);

  /**
   * @brief Emits an informational message when permitted.
   * @param message Plain-text message to write.
   */
  void info(const std::string& message);

  /**
   * @brief Emits a warning message when permitted.
   * @param message Plain-text message to write.
   */
  void warning(const std::string& message);

  /**
   * @brief Emits an error message when permitted.
   * @param message Plain-text message to write.
   */
  void error(const std::string& message);

 private:
  /** @brief Initializes the logger with an informational threshold. */
  Logger();
  /** @brief Flushes and releases the optional file stream. */
  ~Logger();
  Logger(const Logger&) = delete;             ///< Logger is process-global.
  Logger& operator=(const Logger&) = delete;  ///< Logger is process-global.

  /**
   * @brief Emits one message while holding the logger mutex.
   * @param level Severity associated with the message.
   * @param message Plain-text message to write.
   */
  void log(LogLevel level, const std::string& message);

  /**
   * @brief Converts a severity to its stable text representation.
   * @param level Severity to convert.
   * @return Uppercase non-localized severity name.
   */
  std::string level_to_string(LogLevel level) const;

  LogLevel level_;                       ///< Current minimum severity.
  std::unique_ptr<std::ofstream> file_;  ///< Optional append-only sink.
  std::mutex mutex_;                     ///< Protects configuration and writes.
};

/** @brief Convenience macro for emitting a debug message. */
#define LOG_DEBUG(msg) jyppx::utils::Logger::instance().debug(msg)
/** @brief Convenience macro for emitting an informational message. */
#define LOG_INFO(msg) jyppx::utils::Logger::instance().info(msg)
/** @brief Convenience macro for emitting a warning message. */
#define LOG_WARNING(msg) jyppx::utils::Logger::instance().warning(msg)
/** @brief Convenience macro for emitting an error message. */
#define LOG_ERROR(msg) jyppx::utils::Logger::instance().error(msg)

}  // namespace utils
}  // namespace jyppx

#endif  // JYPPX_UTILS_LOGGER_HPP
