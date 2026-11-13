// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_MODELS_EXTENDED_OPEN_VOCABULARY_HPP
#define JYPPX_LIUDING_VISION_MODELS_EXTENDED_OPEN_VOCABULARY_HPP
/** @file open_vocabulary.hpp @brief Explicit YOLO-World/YOLOE native profiles. */
#include <jyppx/liuding/nn/model_adapter.hpp>
namespace jyppx::liuding::models::extended {
enum class PromptMode { world_text, yoloe_text, yoloe_visual, yoloe_prompt_free };
struct OpenVocabularyOptions {
  PromptMode mode=PromptMode::world_text;
  std::int64_t num_classes=80,embedding=512,mask_channels=32;
  bool segmentation=false;
};
/** @brief YOLOv8-small backbone/PAN, World text-gated CSP neck, normalized
 * contrastive heads; YOLOE residual SwiGLU text adapter, spatial-aware visual
 * prompt extraction or learned vocabulary/proposal filtering. These are
 * separately named native profiles, not interchangeable upstream checkpoints.
 * Inputs: images, prompt embeddings [B,C,512] (empty for visual/free), visual
 * masks [B,C,H/8,W/8] (empty for text/free). Visual masks must be nonempty per
 * class and image. Outputs always raw [B,64+C+M,A], prototypes [B,M,H/4,W/4]
 * (M=0 when detection), and proposal logits [B,1,A] (zero in prompted modes).
 * Frozen-prompt conversion is explicit and eval-only; it creates no asset.
 */
class OpenVocabularyImpl final:public torch::nn::Module {
 public:
  explicit OpenVocabularyImpl(OpenVocabularyOptions options={});
  ~OpenVocabularyImpl() override;
  std::vector<Tensor> forward(const Tensor& images,const Tensor& prompts,const Tensor& visual_masks);
  /** @brief Copies normalized prompt embeddings into the fixed vocabulary;
   * only prompt-free profile, eval mode, matching configured class count. */
  void set_vocabulary(const Tensor& embeddings);
 private:
  struct Impl;std::shared_ptr<Impl> impl_;
};
TORCH_MODULE(OpenVocabulary);
nn::ModelAdapter adapt(OpenVocabulary model);
}
#endif
