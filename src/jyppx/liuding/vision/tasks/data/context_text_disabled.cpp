// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/data/context.hpp>

namespace jyppx::liuding::vision::tasks {
std::shared_ptr<const data::image::VisionTaskPayload> prepare_text_context(
    const RecipeSpec&, std::span<const std::string>) {
  throw CompatibilityError("Native text encoding requires models_vision");
}
}  // namespace jyppx::liuding::vision::tasks
