// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "record_function_observer.hpp"

#include <ATen/record_function.h>
#include <memory>
#include <stdexcept>
#include <string_view>

// The prebuilt Release LibTorch omits RecordFunction's debug input flag.
// Match that header ABI here while retaining ASan/UBSan instrumentation.
#ifndef NDEBUG
#error "The RecordFunction bridge requires the prebuilt Release LibTorch ABI"
#endif

namespace jyppx::liuding::tests::model_checkpoint {
namespace {
thread_local CopyObserver active_observer = nullptr;

std::unique_ptr<at::ObserverContext> observe_copy(
    const at::RecordFunction& function) {
  if (active_observer && std::string_view(function.name()) == "aten::copy_" &&
      !function.inputs().empty() && function.inputs().front().isTensor()) {
    active_observer(function.inputs().front().toTensor());
  }
  return nullptr;
}
}  // namespace

std::uint64_t register_copy_observer(CopyObserver observer) {
  if (!observer || active_observer)
    throw std::invalid_argument("Model copy observer must be unique per thread");
  active_observer = observer;
  try {
    return at::addThreadLocalCallback(
        at::RecordFunctionCallback(observe_copy).needsInputs(true));
  } catch (...) {
    active_observer = nullptr;
    throw;
  }
}

void unregister_copy_observer(std::uint64_t handle) noexcept {
  at::removeCallback(handle);
  active_observer = nullptr;
}
}  // namespace jyppx::liuding::tests::model_checkpoint
