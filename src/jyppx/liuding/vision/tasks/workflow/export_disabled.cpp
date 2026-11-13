// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/workflow/export.hpp>
namespace jyppx::liuding::vision::tasks {
std::string export_vision(const TaskBinding&, const RecipeSpec&,
                          const std::filesystem::path&,
                          const VisionExportOptions&) {
  throw CompatibilityError(
      "Vision ONNX export requires the onnx_export component");
}
}  // namespace jyppx::liuding::vision::tasks
