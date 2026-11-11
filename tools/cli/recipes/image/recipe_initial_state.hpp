// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TOOLS_RECIPE_INITIAL_STATE_HPP
#define JYPPX_LIUDING_TOOLS_RECIPE_INITIAL_STATE_HPP

#include "cli.hpp"

#if defined(JYPPX_LIUDING_CLI_IMAGE_RECIPE)
#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>

namespace jyppx::liuding::cli {
struct RecipeInitialState {
  std::filesystem::path path;
  ArtifactDigest digest;
  Json descriptor;
  std::vector<std::byte> encoded;
};

std::optional<RecipeInitialState> inspect_recipe_initial_state(
    const Json& config, const Json& classes);
Json apply_recipe_initial_state(nn::ModelAdapter& model,
                                RecipeInitialState& input);
Json recipe_model_state_identity(const nn::ModelAdapter& model);
std::vector<std::byte> encode_recipe_initial_state(
    const nn::ModelAdapter& model, const Json& config, const Json& classes);
}  // namespace jyppx::liuding::cli
#endif
#endif
