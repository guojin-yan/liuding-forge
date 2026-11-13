// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "checkpoint_codec.hpp"

#include <jyppx/liuding/vision/tasks/training/model_selection.hpp>

#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <utility>

namespace jyppx::liuding::vision::tasks {
namespace {
constexpr std::uint64_t kMagic = 0x314c4553464cULL;
constexpr std::uint64_t kRecipeLimit = 64U << 10U;
constexpr std::uint64_t kControlBytes = 56;
}  // namespace

ModelSelectionCallback::ModelSelectionCallback(ModelSelectionOptions options)
    : options_(std::move(options)) {
  if (!state::Identifier(options_.callback_id) ||
      !state::Identifier(options_.monitor) ||
      !std::isfinite(options_.min_delta) || options_.min_delta < 0)
    throw InvalidArgument("Invalid model selection identity or metric recipe");
  descriptor_ = {
      .type_id = "jyppx.vision.callbacks.model_selection.v1",
      .callback_id = options_.callback_id,
      .subscriptions = {training::CallbackEventKind::epoch_finished},
      .allowed_commands = {training::CallbackCommandKind::request_checkpoint},
      .stateful = true};
  recipe_ =
      nlohmann::json{{"schema", descriptor_.type_id},
                     {"monitor", options_.monitor},
                     {"maximize", options_.maximize},
                     {"min_delta", options_.min_delta},
                     {"save_interval_epochs", options_.save_interval_epochs}}
          .dump();
  validate_checkpoint_participant(*this);
}
const training::CallbackDescriptor& ModelSelectionCallback::descriptor()
    const noexcept {
  return descriptor_;
}
training::CallbackCommandRequest ModelSelectionCallback::handle(
    const training::CallbackEvent& event) {
  if (event.kind != training::CallbackEventKind::epoch_finished ||
      event.phase != training::CallbackPhase::training ||
      event.epoch != state_.observations ||
      event.global_step < state_.last_step)
    throw InvalidArgument(
        "Model selection requires consecutive epoch observations");
  const auto metric = event.metrics.find(options_.monitor);
  if (metric == event.metrics.end() || !std::isfinite(metric->second))
    throw InvalidArgument("Model selection monitor is missing or nonfinite");
  if (state_.observations == (std::numeric_limits<std::uint64_t>::max)())
    throw ResourceExhausted("Model selection observation count overflow");
  auto candidate = state_;
  candidate.last_improved =
      state_.observations == 0 ||
      (options_.maximize
           ? metric->second > state_.best_value + options_.min_delta
           : metric->second < state_.best_value - options_.min_delta);
  if (candidate.last_improved) {
    candidate.best_value = metric->second;
    candidate.best_epoch = event.epoch;
    candidate.best_step = event.global_step;
  }
  candidate.last_step = event.global_step;
  ++candidate.observations;
  const bool save =
      candidate.last_improved ||
      (options_.save_interval_epochs != 0 &&
       candidate.observations % options_.save_interval_epochs == 0);
  training::CallbackCommandRequest command;
  if (save)
    command = {.kind = training::CallbackCommandKind::request_checkpoint,
               .reason = candidate.last_improved ? "monitored metric improved"
                                                 : "periodic epoch checkpoint"};
  state_ = candidate;
  return command;
}
ModelSelectionState ModelSelectionCallback::state() const noexcept {
  return state_;
}
std::string ModelSelectionCallback::participant_id() const {
  return options_.callback_id;
}
StateManifest ModelSelectionCallback::describe_state() const {
  return {
      .participant_id = participant_id(),
      .fields = {{"recipe", "application/vnd.jyppx.model-selection-recipe-v1",
                  kRecipeLimit},
                 {"control", "application/vnd.jyppx.model-selection-control-v1",
                  kControlBytes}}};
}
void ModelSelectionCallback::save_state(StateWriter& writer) const {
  std::vector<std::byte> control;
  state::Append(control, kMagic);
  state::Append(control, state_.observations);
  state::Append(control, std::bit_cast<std::uint64_t>(state_.best_value));
  state::Append(control, state_.best_epoch);
  state::Append(control, state_.best_step);
  state::Append(control, state_.last_step);
  state::Append(control, state_.last_improved ? 1U : 0U);
  writer.write_blob("recipe", std::as_bytes(std::span(recipe_)));
  writer.write_blob("control", control);
}
void ModelSelectionCallback::load_state(const StateReader& reader,
                                        RestorePolicy policy) {
  state::VerifyManifest(reader.manifest(), describe_state(), policy);
  state::VerifyText(reader, "recipe", recipe_, kRecipeLimit);
  const auto bytes = reader.read_blob("control", kControlBytes);
  std::size_t offset = 0;
  if (bytes.size() != kControlBytes || state::Read(bytes, offset) != kMagic)
    throw SerializationError("Malformed model selection control state");
  ModelSelectionState candidate;
  candidate.observations = state::Read(bytes, offset);
  candidate.best_value = std::bit_cast<double>(state::Read(bytes, offset));
  candidate.best_epoch = state::Read(bytes, offset);
  candidate.best_step = state::Read(bytes, offset);
  candidate.last_step = state::Read(bytes, offset);
  const auto improved = state::Read(bytes, offset);
  candidate.last_improved = improved == 1;
  if (improved > 1 || !std::isfinite(candidate.best_value) ||
      candidate.best_step > candidate.last_step)
    throw SerializationError("Invalid model selection values");
  if (candidate.observations == 0) {
    if (candidate.best_value != 0 || candidate.best_epoch != 0 ||
        candidate.best_step != 0 || candidate.last_step != 0 || improved != 0)
      throw SerializationError("Model selection initial state must be zero");
  } else if (candidate.best_epoch >= candidate.observations ||
             candidate.last_improved !=
                 (candidate.best_epoch == candidate.observations - 1) ||
             (candidate.last_improved &&
              candidate.best_step != candidate.last_step)) {
    throw SerializationError("Inconsistent model selection counters");
  }
  state_ = candidate;
}

}  // namespace jyppx::liuding::vision::tasks
