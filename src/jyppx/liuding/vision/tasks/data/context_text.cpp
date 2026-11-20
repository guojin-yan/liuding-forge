// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/models/extended/text.hpp>
#include <jyppx/liuding/vision/tasks/data/context.hpp>

#include <nlohmann/json.hpp>

namespace jyppx::liuding::vision::tasks {
std::shared_ptr<const data::image::VisionTaskPayload> prepare_text_context(
    const RecipeSpec& r, std::span<const std::string> prompts) {
  namespace ex = models::extended;
  if (prompts.empty() ||
      prompts.size() != static_cast<std::size_t>(r.model.num_classes) ||
      prompts.size() > 1024 || r.extension.tokenizer_merges.empty() ||
      r.extension.text_encoder_weights.empty())
    throw InvalidArgument(
        "Text context needs complete vocabulary and local tokenizer/text "
        "encoder assets");
  ex::ClipTokenizer tokenizer(r.extension.tokenizer_merges,
                              r.extension.tokenizer_sha256);
  auto encoder = ex::adapt(ex::ClipText());
  auto options = r.extension.teacher_initialization;
  options = {};
  options.weights = r.extension.text_encoder_weights;
  options.sha256 = r.extension.text_encoder_sha256;
  initialize_vision(encoder, options, "clip_text_vit_b32");
  encoder.state().eval();
  torch::NoGradGuard guard;
  auto p = std::make_shared<data::image::VisionTaskPayload>();
  p->text_prompts.assign(prompts.begin(), prompts.end());
  p->text_encoder_id = tokenizer.identity() +
                       ":clip-text-b32:" + r.extension.text_encoder_sha256;
  p->prompt_embeddings =
      encoder.invoke(nn::Invocation{{tokenizer.encode(p->text_prompts)}, {}})
          .value.toTensor();
  const auto identity = nlohmann::json{{"encoder", p->text_encoder_id},
                                       {"literal_prompts", p->text_prompts},
                                       {"normalization", "unit-l2"}}
                            .dump();
  p->label_space_id = sha256_hex(std::as_bytes(std::span(identity)));
  p->validate();
  return p;
}
}  // namespace jyppx::liuding::vision::tasks
