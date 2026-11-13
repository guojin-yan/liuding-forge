// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/data_image/annotations.hpp>
#include <jyppx/liuding/data_image/pixel_spec.hpp>
namespace jyppx::liuding::vision::tasks::detail {
Tensor render_vision_annotations(const Tensor&, const data::image::PixelSpec&,
                                 const data::image::VisionAnnotationRecord&,
                                 std::span<const data::image::VisionCategory>,
                                 std::stop_token) {
  throw CompatibilityError(
      "Annotation rendering requires vision_render; disable annotation output "
      "for raw PNG previews");
}
}  // namespace jyppx::liuding::vision::tasks::detail
