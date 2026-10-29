// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_LIUDING_DEPLOYMENT_ONNX_SERVING_BACKEND_HPP
#define JYPPX_LIUDING_DEPLOYMENT_ONNX_SERVING_BACKEND_HPP
/** @file backend.hpp @brief Optional ONNX bundle adapter for the Serving
 * lifecycle. */

#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/serving/runtime.hpp>

namespace jyppx::liuding::serving {

/** @brief Owner decodes verified descriptors and supplies trusted same-schema
 * stages.
 * @note The factory receives stable snapshots and actual graph signatures. It
 * must verify descriptor contents, not only copy their digests. The adapter
 * separately checks all four bindings and exact graph shape/dtype/device/name
 * constraints. Neither the factory nor its callbacks may retain borrowed
 * bundle/signature spans.
 */
using OnnxPipelineFactory = std::function<Pipeline(
    const VerifiedBundle&, std::span<const onnx_runtime::TensorSignature>,
    std::span<const onnx_runtime::TensorSignature>)>;

/** @brief Builds the exact named CPU FP32 input contract for verified
 * signatures.
 * @param signatures Session-verified graph input declarations.
 * @return Keyword-only invocation; dynamic leading batch maps to a dynamic
 * extent.
 * @throws Error For an empty, duplicated or unsupported signature.
 */
nn::InvocationSchema onnx_invocation_schema(
    std::span<const onnx_runtime::TensorSignature> signatures);

/** @brief Builds a closed string-keyed CPU FP32 output dictionary contract.
 * @param signatures Session-verified graph output declarations.
 * @return Dictionary schema with every graph output required by its exact name.
 * @throws Error For an empty, duplicated or unsupported signature.
 */
ValueSchema onnx_result_schema(
    std::span<const onnx_runtime::TensorSignature> signatures);

/** @brief Loads data-only ONNX bundles through an explicit ORT execution
 * provider.
 *
 * This independent optional component depends on serving and onnx_runtime;
 * enabling serving alone does not load or link ORT. All external tensors are
 * dense CPU FP32 because Session uses host I/O. RuntimeDescriptor::device
 * describes that boundary;
 * execution_provider/provider_options/allow_cpu_fallback pin the actual
 * requested EP. GPU transfer and kernel selection belong to ORT; this API does
 * not imply GPU zero-copy, provider qualification or a LibTorch CUDA
 * requirement.
 *
 * Inputs are named keyword tensors and results are owned named tensor
 * dictionaries. The owner factory must decode the four verified schema/pre/post
 * descriptors and return the exact graph boundary. Trusted same-schema
 * transforms are permitted. Bundle integrity, actual
 * dependencies/environment/EP, owner bindings and graph schema are checked
 * before known-vector warmup and publication. No model code, custom-op library,
 * Python, network provider or fallback is discovered implicitly.
 */
class OnnxRuntimeBackend final : public RuntimeBackend {
 public:
  /** @brief Freezes the explicit runtime, provider options and descriptor
   * owner.
   * @param descriptor Exact environment, host boundary and provider
   * declaration.
   * @param session Explicit EP, threading, graph and I/O resource options.
   * @param factory Trusted owner decoder and same-schema preprocessing
   * pipeline.
   * @param model_id Exact ONNX variant id in the bundle.
   * @throws Error For unsupported/inconsistent declarations; device
   * availability is checked by the real Session during load.
   */
  OnnxRuntimeBackend(RuntimeDescriptor descriptor,
                     onnx_runtime::SessionOptions session,
                     OnnxPipelineFactory factory,
                     std::string model_id = "primary");

  /** @brief Returns the exact locked host boundary and provider declaration.
   * @return Owning immutable runtime metadata, including EP options and
   * fallback.
   */
  RuntimeDescriptor descriptor() const override;

  /** @brief Reserves the one verified snapshot containing ONNX bytes once.
   * @param options Snapshot limit used by the subsequent bundle load.
   * @return Positive max_snapshot_bytes; max_state_bytes is native-only and
   * ignored.
   * @note Session borrows those bytes synchronously without an adapter model
   * copy. ORT graph parsing/optimization, allocator and EP workspace are not
   * bounded by this logical payload reservation. Request/warmup I/O uses its
   * separate limits.
   */
  std::uint64_t load_reservation_bytes(
      const LoadOptions& options) const override;

  /** @brief Verifies the bundle, loads the actual Session and checks known
   * vectors.
   * @param directory Immutable data-only ONNX bundle directory.
   * @param verify Integrity, signature and encoded-resource policy.
   * @param warmup Nonempty independently supplied known-vector cases.
   * @param options Snapshot, request/output and cooperative lifecycle limits.
   * @return Ready serialized Predictor sharing the existing close/lease
   * lifecycle.
   * @throws Error For integrity, environment, graph, schema, warmup or resource
   * failure.
   */
  std::shared_ptr<Predictor> load(
      const std::filesystem::path& directory,
      const interop::ModelBundleVerifyOptions& verify,
      std::span<const WarmupCase> warmup,
      const LoadOptions& options = {}) const override;

 private:
  RuntimeDescriptor descriptor_;  ///< Exact host I/O and backend declaration.
  onnx_runtime::SessionOptions
      session_;  ///< Explicit immutable session options.
  OnnxPipelineFactory
      factory_;  ///< Trusted descriptor codec and pipeline owner.
  std::string
      model_id_;  ///< Selected single self-contained ONNX artifact variant.
};

}  // namespace jyppx::liuding::serving
#endif  // JYPPX_LIUDING_DEPLOYMENT_ONNX_SERVING_BACKEND_HPP
