// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/visualization/render.hpp>
namespace jyppx::liuding::vision::tasks::detail {
Tensor render_vision_annotations(
    const Tensor& image, const data::image::PixelSpec& spec,
    const data::image::VisionAnnotationRecord& record,
    std::span<const data::image::VisionCategory> categories,
    std::stop_token stop) {
  return visualization::render_annotations(image, spec, record, categories, {},
                                           {}, {}, stop)
      .pixels;
}
}  // namespace jyppx::liuding::vision::tasks::detail
