// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_CLI_RECIPES_DIAGNOSTICS_TEST_FAULT_HPP
#define JYPPX_CLI_RECIPES_DIAGNOSTICS_TEST_FAULT_HPP
#if !defined(JYPPX_LIUDING_CLI_DIAGNOSTIC_TEST_HOOKS)
#error "Diagnostic faults are available only in explicit test-enabled builds"
#endif

#include "recipe_support.hpp"

#include <jyppx/liuding/training/trainer.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <string>

namespace jyppx::liuding::cli::recipe_diagnostics {
// Explicit test-only injection after the actual model/loss computation. No
// public command, configuration field, or normal numerical path selects it.
class TestFault final {
 public:
  TestFault() {
    const auto* request =
        std::getenv("JYPPX_LIUDING_TEST_IMAGE_DIAGNOSTIC_FAULT");
    if (!request) return;
    std::size_t size = 0;
    while (size <= 64 && request[size]) ++size;
    recipe_support::require_recipe(size > 0 && size <= 64,
                                   "Invalid diagnostic test fault size");
    fault_.assign(request, size);
    constexpr std::array allowed{"training.nan",
                                 "training.positive_infinity",
                                 "training.negative_infinity",
                                 "validation.nan",
                                 "validation.positive_infinity",
                                 "validation.negative_infinity",
                                 "gradient.nan",
                                 "gradient.infinity",
                                 "storage.failure"};
    recipe_support::require_recipe(
        std::find(allowed.begin(), allowed.end(), fault_) != allowed.end(),
        "Unknown diagnostic test fault");
  }

  void apply(training::StepOutput& output, training::StepPhase phase,
             const std::filesystem::path& root) const {
    if (fault_.empty()) return;
    const bool validation = phase == training::StepPhase::validation;
    if (validation != fault_.starts_with("validation.")) return;
    const bool nan = fault_.ends_with("nan");
    auto value = nan ? std::numeric_limits<double>::quiet_NaN()
                     : std::numeric_limits<double>::infinity();
    if (fault_.find("negative") != std::string::npos) value = -value;
    if (fault_.starts_with("gradient.")) {
      output.loss.register_hook([value](const Tensor& gradient) {
        return torch::full_like(gradient, value);
      });
    } else {
      if (fault_ == "storage.failure") {
        const auto objects = root / "run-store" / "artifacts" / "objects";
        std::filesystem::rename(objects,
                                objects.parent_path() / "objects.saved");
        recipe_support::write_file(objects, recipe_support::bytes("blocked"));
      }
      output.loss = output.loss.mul(value);
    }
  }

 private:
  std::string fault_;
};
}  // namespace jyppx::liuding::cli::recipe_diagnostics

#endif
