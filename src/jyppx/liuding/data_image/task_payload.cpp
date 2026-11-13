// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/data_image/task_payload.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <array>
#include <cmath>
namespace jyppx::liuding::data::image {
namespace {
using I = std::int64_t;
ValueSchema Kind(ValueKind kind) { ValueSchema s; s.kind = kind; return s; }
c10::IValue Tuple(std::vector<c10::IValue> fields) {
  return c10::ivalue::Tuple::create(std::move(fields));
}
void Require(bool condition, const char* field) {
  if (!condition) throw InvalidArgument(std::string("Invalid task payload: ") + field);
}
}
std::vector<Tensor> VisionTaskPayload::tensors() const {
  return {multilabel_values, known_labels, depth, depth_valid, proposals,
          proposal_scores, prompt_embeddings, visual_prompts, pretrain_mask,
          track_ids, camera_intrinsics};
}
std::uint64_t VisionTaskPayload::validate(std::uint64_t cap, std::size_t strings,
                                         std::size_t string_bytes) const {
  std::uint64_t bytes = 0;
  for (const auto& t : tensors()) {
    if (!t.defined()) continue;
    Require(t.layout() == torch::kStrided && !t.requires_grad(), "target storage/autograd");
    if (t.numel() > (cap - bytes) / t.element_size())
      throw ResourceExhausted("Task payload tensor budget exceeded");
    bytes += t.numel() * t.element_size();
  }
  if (text_prompts.size() > strings) throw ResourceExhausted("Prompt count exceeded");
  auto text = [&](const std::string& s) {
    if (s.size() > string_bytes) throw ResourceExhausted("Task payload string exceeded");
  };
  for (auto* s : {&label_space_id,&text_encoder_id,&proposal_generator,
                  &proposal_inventory,&sequence_id,&camera_id}) text(*s);
  for (const auto& s : text_prompts) text(s);
  Require(frame_id >= 0 && std::isfinite(timestamp_seconds) && timestamp_seconds >= 0,
          "sequence frame/timestamp");
  Require(std::isfinite(depth_unit_scale) && depth_unit_scale > 0 &&
              static_cast<I>(depth_representation) >= 0 && static_cast<I>(depth_representation) <= 2,
          "depth representation/unit");
  Require(multilabel_values.defined() == known_labels.defined(), "known label presence");
  if (multilabel_values.defined()) {
    Require(multilabel_values.dim() == 1 && multilabel_values.is_floating_point() &&
                known_labels.sizes() == multilabel_values.sizes() &&
                known_labels.scalar_type() == torch::kBool && known_labels.device() == multilabel_values.device(), "multilabel shape");
    Require(torch::isfinite(multilabel_values).all().item<bool>() &&
                multilabel_values.ge(0).all().item<bool>() && multilabel_values.le(1).all().item<bool>(), "multilabel values");
  }
  Require(depth.defined() == depth_valid.defined(), "depth validity presence");
  if (depth.defined()) {
    Require(depth.dim() == 3 && depth.size(0) == 1 && depth.is_floating_point() &&
                depth_valid.sizes() == depth.sizes() && depth_valid.scalar_type() == torch::kBool &&
                depth_valid.device() == depth.device(), "depth shape");
    auto valid = depth.masked_select(depth_valid);
    Require(torch::isfinite(valid).all().item<bool>() &&
                (depth_representation == DepthRepresentation::relative_inverse_depth || valid.gt(0).all().item<bool>()), "depth valid values");
  }
  for (const auto& t : {proposals, visual_prompts}) {
    if (!t.defined()) continue;
    const I columns = t.is_same(proposals) ? 4 : 5;
    Require(t.dim() == 2 && t.size(1) == columns && t.is_floating_point() &&
                torch::isfinite(t).all().item<bool>(), "proposal/visual prompt shape");
    const I first = columns - 4;
    Require((t.slice(1, first + 2, first + 4) >= t.slice(1, first, first + 2)).all().item<bool>(), "proposal box extents");
  }
  if (proposal_scores.defined())
    Require(proposals.defined() && proposal_scores.dim() == 1 && proposal_scores.size(0) == proposals.size(0) &&
                proposal_scores.is_floating_point() && torch::isfinite(proposal_scores).all().item<bool>(), "proposal scores");
  if (prompt_embeddings.defined())
    Require(prompt_embeddings.dim() == 2 && prompt_embeddings.is_floating_point() &&
                torch::isfinite(prompt_embeddings).all().item<bool>(), "prompt embeddings");
  if (pretrain_mask.defined()) Require(pretrain_mask.dim() == 2 && pretrain_mask.scalar_type() == torch::kBool, "pretrain mask");
  if (track_ids.defined()) Require(track_ids.dim() == 1 && track_ids.scalar_type() == torch::kInt64, "track IDs");
  if (camera_intrinsics.defined()) Require(camera_intrinsics.sizes() == torch::IntArrayRef({3,3}) &&
      camera_intrinsics.is_floating_point() && torch::isfinite(camera_intrinsics).all().item<bool>(), "camera intrinsics");
  return bytes;
}
ValueSchema task_payload_schema() {
  auto tensor = ValueSchema::tensor({}); tensor.nullable = true;
  const auto string = Kind(ValueKind::string), integer = Kind(ValueKind::integer), real = Kind(ValueKind::floating);
  return ValueSchema::tuple({integer, ValueSchema::tuple(std::vector<ValueSchema>(11, tensor)),
      ValueSchema::tuple(std::vector<ValueSchema>(6, string)), integer, real, integer, real, integer,
      ValueSchema::list(string)});
}
c10::IValue task_payload_value(const VisionTaskPayload& p) {
  p.validate();
  std::vector<c10::IValue> tensors;
  for (const auto& t : p.tensors()) tensors.push_back(t.defined() ? c10::IValue(t) : c10::IValue{});
  c10::List<std::string> prompts;
  for (const auto& text : p.text_prompts) prompts.push_back(text);
  return Tuple({I{1}, Tuple(std::move(tensors)), Tuple({p.label_space_id,p.text_encoder_id,
      p.proposal_generator,p.proposal_inventory,p.sequence_id,p.camera_id}),
      p.frame_id,p.timestamp_seconds,static_cast<I>(p.depth_representation),p.depth_unit_scale,
      p.semantic_ignore,prompts});
}
std::shared_ptr<const VisionTaskPayload> task_payload_from_value(const c10::IValue& value) {
  validate_value(value, task_payload_schema(), "image.task_payload");
  const auto& a = value.toTupleRef().elements();
  Require(a[0].toInt() == 1, "schema version");
  auto p = std::make_shared<VisionTaskPayload>();
  const auto& tensors = a[1].toTupleRef().elements();
  std::array<Tensor*,11> destinations{&p->multilabel_values,&p->known_labels,&p->depth,&p->depth_valid,
      &p->proposals,&p->proposal_scores,&p->prompt_embeddings,&p->visual_prompts,&p->pretrain_mask,
      &p->track_ids,&p->camera_intrinsics};
  for (std::size_t i=0;i<destinations.size();++i) if (!tensors[i].isNone()) *destinations[i]=tensors[i].toTensor();
  const auto& strings = a[2].toTupleRef().elements();
  std::array<std::string*,6> texts{&p->label_space_id,&p->text_encoder_id,&p->proposal_generator,
      &p->proposal_inventory,&p->sequence_id,&p->camera_id};
  for (std::size_t i=0;i<texts.size();++i) *texts[i]=strings[i].toStringRef();
  p->frame_id=a[3].toInt(); p->timestamp_seconds=a[4].toDouble();
  p->depth_representation=static_cast<DepthRepresentation>(a[5].toInt());
  p->depth_unit_scale=a[6].toDouble(); p->semantic_ignore=a[7].toInt();
  for (const auto& text:a[8].toListRef()) p->text_prompts.emplace_back(text.toStringRef());
  p->validate();
  return p;
}
}  // namespace jyppx::liuding::data::image
