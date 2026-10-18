// Unit tests for jyppx::utils::Logger.

#include <jyppx/foundation/error.hpp>
#include <jyppx/utils/logger.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

namespace {

namespace fs = std::filesystem;

// Unique temp file per test so the append-mode logger never sees stale data.
// gtest_discover_tests runs every test in its own process, so a plain counter
// would collide across processes; the monotonic clock keeps names unique.
class LoggerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto ticks =
        std::chrono::steady_clock::now().time_since_epoch().count();
    log_path_ = (fs::temp_directory_path() /
                 ("jyppx_logger_test_" + std::to_string(ticks) + ".log"))
                    .string();
    std::error_code ec;
    fs::remove(log_path_, ec);
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove(log_path_, ec);
  }

  std::string read_log() const {
    std::ifstream file(log_path_);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }

  std::string log_path_;
};

TEST_F(LoggerTest, WritesRequestedLevelToFile) {
  auto& logger = jyppx::utils::Logger::instance();
  logger.set_level(jyppx::utils::LogLevel::INFO);
  logger.set_file(log_path_);
  logger.info("info-message");
  logger.error("error-message");

  const auto contents = read_log();
  EXPECT_NE(contents.find("info-message"), std::string::npos);
  EXPECT_NE(contents.find("error-message"), std::string::npos);
  EXPECT_NE(contents.find("[INFO]"), std::string::npos);
  EXPECT_NE(contents.find("[ERROR]"), std::string::npos);
}

TEST_F(LoggerTest, LevelFiltersQuieterMessages) {
  auto& logger = jyppx::utils::Logger::instance();
  logger.set_level(jyppx::utils::LogLevel::WARNING);
  logger.set_file(log_path_);
  logger.debug("debug-message");
  logger.info("info-message");
  logger.warning("warning-message");
  logger.error("error-message");

  const auto contents = read_log();
  EXPECT_EQ(contents.find("debug-message"), std::string::npos);
  EXPECT_EQ(contents.find("info-message"), std::string::npos);
  EXPECT_NE(contents.find("warning-message"), std::string::npos);
  EXPECT_NE(contents.find("error-message"), std::string::npos);
}

TEST_F(LoggerTest, SetFileRejectsUnwritablePath) {
  auto& logger = jyppx::utils::Logger::instance();
#ifdef _WIN32
  const std::string bad_path = "Z:\\definitely\\missing\\dir\\jyppx.log";
#else
  const std::string bad_path = "/definitely/missing/dir/jyppx.log";
#endif
  EXPECT_THROW(logger.set_file(bad_path), jyppx::IoError);
}

}  // namespace
