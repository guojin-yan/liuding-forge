// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/workflow/export.hpp>
namespace jyppx::liuding::vision::tasks {
std::string predict_vision_bundle(const std::filesystem::path&,
                                  std::span<const data::image::VisionSample>,
                                  std::stop_token) {
  throw CompatibilityError("Offline vision execution requires onnx_runtime");
}
}  // namespace jyppx::liuding::vision::tasks
