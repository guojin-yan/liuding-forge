// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_MODELS_EXTENDED_TEXT_HPP
#define JYPPX_LIUDING_VISION_MODELS_EXTENDED_TEXT_HPP
/** @file text.hpp @brief Native bounded UTF-8 byte BPE and CLIP text encoder. */
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <filesystem>
#include <span>
namespace jyppx::liuding::models::extended {
/** @brief Plain UTF-8 merge asset with explicit SHA-256, never fetched or
 * decompressed implicitly. Profile clip.utf8-lower-bpe/v1 performs Unicode
 * lowercase/whitespace collapse and HTML entities; it does not repair broken
 * encodings (ftfy). Inputs must be valid, normalized UTF-8. Vocabulary follows
 * CLIP byte symbols, end-of-word symbols, 48894 merges, BOS and EOS. */
class ClipTokenizer final {
 public:
  ClipTokenizer(const std::filesystem::path& merges,std::string_view sha256);
  Tensor encode(std::span<const std::string> prompts,bool truncate=false)const;
  const std::string& identity()const noexcept;
 private:
  struct Impl;std::shared_ptr<const Impl> impl_;
};
struct ClipTextOptions {
  std::int64_t vocabulary=49408,context=77,width=512,layers=12,heads=8,embedding=512;
  bool activation_recomputation=false;
};
/** @brief Causal pre-norm CLIP text tower, QuickGELU, learned positions,
 * EOS pooling and projection; outputs normalized [B,D]. Integer tokens are
 * explicit inputs. Tokenizer/weights/config digest must identify any cached
 * embedding. Construction alone provides random weights, never text quality. */
class ClipTextImpl final:public torch::nn::Module {
 public:
  explicit ClipTextImpl(ClipTextOptions options={});
  ~ClipTextImpl() override;
  Tensor forward(const Tensor& tokens);
 private:
  struct Impl;std::shared_ptr<Impl> impl_;
};
TORCH_MODULE(ClipText);
nn::ModelAdapter adapt(ClipText model);
}
#endif
