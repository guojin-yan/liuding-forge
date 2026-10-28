// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TOOLS_TRANSFER_RECIPE_HPP
#define JYPPX_LIUDING_TOOLS_TRANSFER_RECIPE_HPP

#include "cli.hpp"

#if defined(JYPPX_LIUDING_CLI_IMAGE_RECIPE)
#include <jyppx/liuding/models/weights.hpp>

#include <memory>

namespace jyppx::liuding::cli {
inline constexpr std::string_view kTransferRecipe =
    "jyppx.liuding.imagefolder.transfer/v1";

struct TransferInput {
  models::WeightRecord record;
  Json descriptor;
  std::filesystem::path path;
  std::int64_t source_classes;
  std::string input_identity;
};

bool is_transfer_recipe(const Json& config);
TransferInput inspect_transfer_input(const Json& config);
std::shared_ptr<nn::ModelAdapter> make_transfer_model(
    const TransferInput& input, const Json& config, std::int64_t classes);
Json transfer_state(const nn::ModelAdapter& model);
}  // namespace jyppx::liuding::cli
#endif
#endif
