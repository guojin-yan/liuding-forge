// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_OPEN_VOCABULARY_HPP
#define JYPPX_LIUDING_VISION_TASKS_OPEN_VOCABULARY_HPP
/** @file open_vocabulary.hpp @brief Explicit typed prompt training inputs. */
#include <jyppx/liuding/vision/tasks/data/targets.hpp>
namespace jyppx::liuding::vision::tasks {
struct PromptInputOptions {
  std::int64_t classes=80;
  bool visual=false,prompt_free=false;
  std::string text_encoder_identity;
  std::function<Tensor(const std::vector<std::string>&)> encode_text;
};
/** @brief Requires matching label/text identities per batch; accepts real raw
 * text through the supplied native encoder, or verified precomputed embeddings.
 * Visual class/xyxy boxes rasterize on the same transformed stride-8 canvas.
 * A missing class prompt is rejected rather than silently relabelled. */
training::BatchMapper make_prompt_image_mapper(PromptInputOptions options);
/** @brief Existing TAL/CIoU/DFL and optional instance masks, with supervised
 * foreground proposal-filter BCE only for the prompt-free profile. */
training::StepFunction make_prompt_step(YoloTaskOptions task,bool prompt_free=false);
}
#endif
