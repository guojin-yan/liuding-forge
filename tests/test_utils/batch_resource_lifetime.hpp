// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#pragma once

#include <jyppx/liuding/core/types.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stop_token>

namespace jyppx::liuding::test {

// Observe actual Tensor storage deletion independently of envelope handles.
struct BatchResourceLifetime
    : std::enable_shared_from_this<BatchResourceLifetime> {
  struct Snapshot {
    int live_payloads = 0;
    int payloads_released = 0;
    int resources_acquired = 0;
    int resources_released = 0;
    int admissions_acquired = 0;
    int admissions_released = 0;
    int early_resource_releases = 0;
    int early_admission_releases = 0;
  };

  Tensor tensor(std::int64_t value = 7) {
    auto self = shared_from_this();
    auto* payload = new std::int64_t(value);
    {
      std::lock_guard lock(mutex);
      ++state.live_payloads;
    }
    return torch::from_blob(
        payload, {1},
        [self](void* pointer) {
          delete static_cast<std::int64_t*>(pointer);
          std::lock_guard lock(self->mutex);
          --self->state.live_payloads;
          ++self->state.payloads_released;
        },
        torch::TensorOptions().dtype(torch::kInt64));
  }

  std::shared_ptr<void> lease(bool admission = false) {
    auto self = shared_from_this();
    auto owner =
        std::shared_ptr<void>(new bool(admission), [self](void* pointer) {
          const bool admission = *static_cast<bool*>(pointer);
          delete static_cast<bool*>(pointer);
          std::lock_guard lock(self->mutex);
          if (admission) {
            ++self->state.admissions_released;
            if (self->state.live_payloads)
              ++self->state.early_admission_releases;
          } else {
            ++self->state.resources_released;
            if (self->state.live_payloads)
              ++self->state.early_resource_releases;
          }
          self->ready.notify_all();
        });
    {
      std::lock_guard lock(mutex);
      if (admission)
        ++state.admissions_acquired;
      else
        ++state.resources_acquired;
      ready.notify_all();
    }
    return owner;
  }

  bool wait_for_resources(int count = 1) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, std::chrono::seconds(2),
                          [&] { return state.resources_acquired >= count; });
  }

  bool wait_for_admissions(int count = 1) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, std::chrono::seconds(2),
                          [&] { return state.admissions_acquired >= count; });
  }

  bool wait_for_admission_releases(int count = 1) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, std::chrono::seconds(2),
                          [&] { return state.admissions_released >= count; });
  }

  Snapshot snapshot() const {
    std::lock_guard lock(mutex);
    return state;
  }

  static void wait_for_stop(std::stop_token stop) {
    std::mutex mutex;
    std::condition_variable_any ready;
    std::unique_lock lock(mutex);
    static_cast<void>(ready.wait(lock, stop, [] { return false; }));
  }

 private:
  mutable std::mutex mutex;
  std::condition_variable ready;
  Snapshot state;
};

}  // namespace jyppx::liuding::test
