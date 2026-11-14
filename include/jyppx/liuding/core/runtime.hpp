// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_CORE_RUNTIME_HPP
#define JYPPX_LIUDING_CORE_RUNTIME_HPP

namespace jyppx::liuding {

/**
 * @brief Initializes the supported LibTorch process runtime before use.
 *
 * Call on the main thread before configuring LibTorch threads, starting worker
 * threads, or invoking tensor/model operations. Repeated calls are harmless.
 * Preserves the configured intra-op and inter-op thread counts.
 *
 * On Windows LibTorch 2.14.1 with OpenMP, initializes the auxiliary Caffe2
 * pthreadpool without workers. This avoids its DLL-detach mutex deadlock after
 * Windows has terminated worker threads. OpenMP/MKL computation can still use
 * the caller's requested parallelism; auxiliary pthreadpool work runs inline.
 * Other profiles require no adjustment. This cannot repair a pool initialized
 * earlier by an application or a different library.
 *
 * @throws c10::Error If the LibTorch thread runtime cannot be initialized.
 */
void initialize_runtime();

}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_CORE_RUNTIME_HPP
