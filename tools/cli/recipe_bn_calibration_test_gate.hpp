// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_CLI_RECIPE_BN_CALIBRATION_TEST_GATE_HPP
#define JYPPX_LIUDING_CLI_RECIPE_BN_CALIBRATION_TEST_GATE_HPP
/** @file recipe_bn_calibration_test_gate.hpp
 * @brief Private, explicit and finite native-process acceptance handshake.
 */
#if !defined(JYPPX_LIUDING_CLI_CALIBRATION_TEST_HOOKS)
#error "Calibration acceptance handshake is available only to test builds"
#endif

#include "recipe_bn_calibration.hpp"
#include "recipe_support.hpp"

#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/nn/modules/batchnorm.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <stop_token>
#include <string>
#include <thread>

namespace jyppx::liuding::cli {

// No public CLI/config field is added. The installed CLI from a test-enabled
// build must also receive this private environment request to activate it.
class RecipeCalibrationTestGate final {
 public:
  static std::unique_ptr<RecipeCalibrationTestGate> from_environment() {
    const auto* value = std::getenv("JYPPX_LIUDING_TEST_BN_CALIBRATION_GATE");
    if (!value) return {};
    std::size_t size = 0;
    while (size <= 2048 && value[size]) ++size;
    recipe_support::require_recipe(size > 0 && size <= 2048,
                                   "Invalid calibration test request size");
    auto request = parse_json(std::string_view(value, size));
    fields(request,
           {"schema_version", "nonce", "boundary", "after_batches",
            "expected_samples", "expected_updates", "expected_bn_modules",
            "source_model_sha256", "timeout_ms"});
    recipe_support::require_recipe(
        request.at("schema_version") == 1 && request.at("nonce").is_string() &&
            request.at("boundary").is_string() &&
            request.at("source_model_sha256").is_string(),
        "Malformed calibration test request");
    validate_artifact_digest({.hex = request.at("nonce").get<std::string>()});
    validate_artifact_digest(
        {.hex = request.at("source_model_sha256").get<std::string>()});
    const auto boundary = request.at("boundary").get<std::string>();
    recipe_support::require_recipe(
        boundary == "batch" || boundary == "completed",
        "Unknown calibration test boundary");
    const auto batches = unsigned_number(request.at("after_batches"), 1000000);
    const auto samples =
        unsigned_number(request.at("expected_samples"), 1000000000);
    const auto updates =
        unsigned_number(request.at("expected_updates"), 1000000000);
    const auto modules =
        unsigned_number(request.at("expected_bn_modules"), 100000);
    const auto timeout = unsigned_number(request.at("timeout_ms"), 30000);
    recipe_support::require_recipe(samples >= batches && updates > 0 &&
                                       modules > 0 && timeout >= 1000 &&
                                       (boundary != "completed" || batches > 0),
                                   "Invalid finite calibration test counts");
    return std::unique_ptr<RecipeCalibrationTestGate>(
        new RecipeCalibrationTestGate(std::move(request)));
  }

  void observe(std::string_view boundary, const std::filesystem::path& root,
               const RecipeBatchNormCalibration& stage,
               const nn::ModelAdapter& model, std::uint64_t batches,
               std::uint64_t samples, std::uint64_t updates,
               std::stop_token stop) {
    if (fired_ || boundary != request_.at("boundary").get<std::string>() ||
        batches != request_.at("after_batches").get<std::uint64_t>())
      return;
    using namespace recipe_support;
    const auto expected_phase =
        boundary == "batch" ? RecipeBatchNormCalibration::Phase::ready
                            : RecipeBatchNormCalibration::Phase::completed;
    require_recipe(
        stage.phase() == expected_phase &&
            stage.source_model_sha256() ==
                request_.at("source_model_sha256").get<std::string>() &&
            updates == request_.at("expected_updates").get<std::uint64_t>() &&
            stage.optimizer_updates() == updates &&
            samples == request_.at("expected_samples").get<std::uint64_t>(),
        "Calibration test boundary has a different stage or count");
    Json counters = Json::array();
    for (const auto& entry : model.state().module().named_modules()) {
      const auto norm =
          std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(entry.value());
      if (!norm) continue;
      const auto count = norm->num_batches_tracked.item<std::int64_t>();
      require_recipe(count >= 0 && static_cast<std::uint64_t>(count) == batches,
                     "Calibration test boundary has incomplete BN forwards");
      counters.push_back({{"module", entry.key()}, {"batches", count}});
    }
    require_recipe(counters.size() ==
                       request_.at("expected_bn_modules").get<std::uint64_t>(),
                   "Calibration test boundary has a different BN inventory");
    const auto checkpoint_sha = file_digest(root / "latest.jyck").hex;
    const auto nonce = request_.at("nonce").get<std::string>();
    const auto directory = root / (".bn-calibration-test-" + nonce);
    require_recipe(root.is_absolute() && directory.parent_path() == root &&
                       std::filesystem::create_directory(directory),
                   "Calibration test directory already exists or is invalid");
    const Json marker{
        {"schema_version", 1},
        {"nonce", nonce},
#if defined(_WIN32)
        {"pid", static_cast<std::uint64_t>(GetCurrentProcessId())},
#else
        {"pid", static_cast<std::uint64_t>(::getpid())},
#endif
        {"boundary", boundary},
        {"completed_batches", batches},
        {"completed_samples", samples},
        {"optimizer_updates", updates},
        {"checkpoint_sha256", checkpoint_sha},
        {"source_model_sha256", stage.source_model_sha256()},
        {"calibration", stage.report()},
        {"bn_counters", std::move(counters)}};
    const auto encoded = marker.dump();
    require_recipe(encoded.size() <= 16384,
                   "Calibration test marker exceeds 16 KiB");
    const auto temporary = directory / "waiting.tmp";
    write_exclusive(temporary, encoded);
    // Link publication is atomic and fails if the destination already exists.
    const auto waiting = directory / "waiting.json";
    std::filesystem::create_hard_link(temporary, waiting);
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    const auto marker_sha = sha256_hex(bytes(encoded));
    fired_ = true;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(
                              request_.at("timeout_ms").get<std::uint64_t>());
    const auto release = directory / "release.json";
    while (std::chrono::steady_clock::now() < deadline) {
      if (stop.stop_requested())
        throw Cancelled("Calibration test handshake cancelled");
      if (std::filesystem::exists(release)) {
        static_cast<void>(file_digest(release, 1024));
        const auto response = parse_json(read_text(release, 1024));
        fields(response, {"schema_version", "nonce", "marker_sha256"});
        require_recipe(
            response.at("schema_version") == 1 &&
                response.at("nonce") == nonce &&
                response.at("marker_sha256") == marker_sha &&
                file_digest(waiting, 16384).hex == marker_sha &&
                file_digest(root / "latest.jyck").hex == checkpoint_sha,
            "Calibration test release does not match its marker");
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw DeadlineExceeded("Calibration test handshake timed out");
  }

 private:
  explicit RecipeCalibrationTestGate(Json request)
      : request_(std::move(request)) {}

  static void write_exclusive(const std::filesystem::path& path,
                              const std::string& content) {
#if defined(_WIN32)
    const auto file = CreateFileW(
        path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (file == INVALID_HANDLE_VALUE)
      throw IoError("Cannot exclusively create calibration test marker");
    DWORD written = 0;
    const auto success =
        WriteFile(file, content.data(), static_cast<DWORD>(content.size()),
                  &written, nullptr) &&
        written == content.size() && FlushFileBuffers(file);
    const auto closed = CloseHandle(file);
    if (!success || !closed)
      throw IoError("Cannot publish complete calibration test marker");
#else
    const auto file =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (file < 0)
      throw IoError("Cannot exclusively create calibration test marker");
    std::size_t written = 0;
    while (written < content.size()) {
      const auto count =
          ::write(file, content.data() + written, content.size() - written);
      if (count <= 0) {
        ::close(file);
        throw IoError("Cannot publish complete calibration test marker");
      }
      written += static_cast<std::size_t>(count);
    }
    const auto flushed = ::fsync(file);
    const auto closed = ::close(file);
    if (flushed != 0 || closed != 0)
      throw IoError("Cannot publish complete calibration test marker");
#endif
  }
  Json request_;
  bool fired_ = false;
};
}  // namespace jyppx::liuding::cli
#endif
