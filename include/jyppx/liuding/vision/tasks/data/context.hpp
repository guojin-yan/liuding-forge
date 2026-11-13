// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_CONTEXT_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_CONTEXT_HPP
/** @file context.hpp @brief Portable prompt and external-proposal input contexts. */
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>
namespace jyppx::liuding::vision::tasks {
/** @brief Parses bounded vision.input_context/v1 JSON into the canonical typed
 * sample payload. Coordinates refer to the original image; normal task geometry
 * applies afterwards. Text and embeddings bind an ordered label-space identity. */
std::shared_ptr<const data::image::VisionTaskPayload> parse_input_context(std::string_view json);
std::string input_context_json(const data::image::VisionTaskPayload& payload);
/** @brief Encodes literal UTF-8 prompts with verified local native CLIP assets.
 * No weights or tokenizer data are downloaded. The resulting context is usable
 * without text encoder dependencies in the embedding inference profile. */
std::shared_ptr<const data::image::VisionTaskPayload> prepare_text_context(
    const RecipeSpec&,std::span<const std::string> literal_prompts);
}
#endif
