// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "yolo_validation.hpp"

namespace jyppx::liuding::vision::tasks::detail {
training::StepOutput YoloDetectionValidation::step(
    TaskBinding& binding, const training::StepInput& input) {
  if (input.context.phase != training::StepPhase::validation)
    throw InvalidArgument("YOLO detection validation is validation-only");
  const auto batch = data::image::image_batch_from_value(input.batch.value);
  if (!batch.images.defined() || batch.images.dim() != 4 || batch.items.empty())
    throw InvalidArgument("YOLO validation requires a packed image batch");
  auto output = binding.step({*binding.model, input.batch, input.invocation,
                              input.target, input.context});
  if (!output.prediction || output.sample_count != batch.items.size())
    throw InvalidArgument(
        "YOLO validation needs the task step's batch prediction");
  auto results = binding.decode(output.prediction->value,
                                {batch.images.size(2), batch.images.size(3)},
                                result_options());
  update(batch, results, input.context.epoch);
  return output;
}
}  // namespace jyppx::liuding::vision::tasks::detail
