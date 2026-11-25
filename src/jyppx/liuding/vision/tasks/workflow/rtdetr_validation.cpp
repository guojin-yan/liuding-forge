// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "rtdetr_validation.hpp"

namespace jyppx::liuding::vision::tasks::detail {
training::StepOutput RtDetrValidation::step(TaskBinding& binding,
                                            const training::StepInput& input) {
  if (input.context.phase != training::StepPhase::validation)
    throw InvalidArgument("RT-DETR inference validation is validation-only");
  const auto batch = data::image::image_batch_from_value(input.batch.value);
  if (!batch.images.defined() || batch.images.dim() != 4 || batch.items.empty())
    throw InvalidArgument("RT-DETR validation requires a packed image batch");
  auto invocation = input.invocation;
  invocation.positional.at(1) = torch::empty({0, 6}, batch.images.options());
  auto prediction = binding.model->invoke(invocation);
  auto results = binding.decode(prediction.value,
                                {batch.images.size(2), batch.images.size(3)},
                                result_options());
  update(batch, results, input.context.epoch);
  training::StepOutput output;
  output.sample_count = batch.items.size();
  return output;
}
}  // namespace jyppx::liuding::vision::tasks::detail
