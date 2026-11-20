// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_RUN_ADMISSION_HPP
#define JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_RUN_ADMISSION_HPP

#include <chrono>
#include <mutex>

namespace jyppx::liuding::onnx_runtime::detail {

// No run options/callback exist until this slot is owned. A waiting caller can
// abandon its admission without terminating the caller that currently owns it.
template <typename Check>
std::unique_lock<std::timed_mutex> AcquireRunAdmission(std::timed_mutex& mutex,
                                                       bool serial,
                                                       Check&& check) {
  std::unique_lock<std::timed_mutex> lock(mutex, std::defer_lock);
  check();
  if (serial) {
    while (!lock.try_lock_for(std::chrono::milliseconds(5))) check();
    check();
  }
  return lock;
}

}  // namespace jyppx::liuding::onnx_runtime::detail

#endif  // JYPPX_LIUDING_DEPLOYMENT_ONNX_RUNTIME_RUN_ADMISSION_HPP
