// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_STATIC_INT8_LINEAR_HPP
#define JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_STATIC_INT8_LINEAR_HPP
/** @file static_int8_linear.hpp
 * @brief Native C++ ONNX conversion of a calibrated portable Linear artifact.
 */

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/quantization/quantization.hpp>

namespace jyppx::liuding::onnx_export {

/** @brief An exact source-to-ONNX identity association, without execution
 * claims. */
struct StaticInt8LinearExportResult {
  ExportResult artifact;  ///< Self-contained graph with FP32 input/output.
  ArtifactDigest source_artifact_digest;  ///< Complete canonical Q0 identity.
  std::string profile_id;  ///< Explicit U8U8 static PTQ recipe version.
};

/** @brief Converts a validated Q0 Linear artifact to an integer ONNX graph.
 *
 * Available only when both onnx_export and quantization were enabled. The
 * graph uses QuantizeLinear, U8U8 MatMulInteger/int32, Cast, Mul and optional
 * FP32 bias. It delegates protobuf serialization to LibTorch, never Python.
 * A graph containing integer operators does not prove runtime kernel choice,
 * numerical accuracy, task quality or speedup; qualify those independently.
 *
 * @param source_artifact Complete immutable canonical Q0 bytes for this call.
 * @param expected_source_digest Independently supplied exact source identity.
 * @param example_shape Concrete positive [batch,in_features] signature.
 * @param options Opset 17, batch policy and complete serialized byte ceiling.
 * @param limits Independent Q0 parsing, calibration and payload ceilings.
 * @return Owned ONNX bytes and their complete canonical source association.
 * @throws InvalidArgument For unsupported options, shape or malformed identity.
 * @throws CompatibilityError For identity, Q0 version or serializer mismatch.
 * @throws ResourceExhausted For source, graph or complete output byte bounds.
 * @throws NumericalError For nonrepresentable calibration scales or output.
 * @throws Error For malformed Q0 or backend failures.
 * @note Source spans must remain unchanged for this call. No caller storage
 * is retained or mutated. Hash verification establishes integrity, not trust.
 * Concurrent calls have independent graph state; LibTorch must be configured
 * before calls begin. This is not a hard process-memory quota.
 */
StaticInt8LinearExportResult export_static_int8_linear(
    std::span<const std::byte> source_artifact,
    const ArtifactDigest& expected_source_digest,
    std::span<const std::int64_t> example_shape,
    const ExportOptions& options = {},
    quantization::QuantizationLimits limits = {});

/** @brief Exports ordered per-output-channel Q0 weights as one INT8 Linear.
 *
 * The canonical aggregate has one Q0 row per output, identical calibration
 * bytes on every row, uniform bias presence and an explicit ordered channel
 * count. One U8U8 MatMulInteger accumulates into int32; a [out_features] FP32
 * product-scale vector broadcasts across batch before optional FP32 bias.
 * There is no dequantized FP32 dot product. Activation qparams and the existing
 * per-tensor Q0 format retain their original symmetric semantics.
 *
 * @param source_artifact Complete canonical per-output-channel aggregate.
 * @param expected_source_digest Independent SHA-256 of the complete aggregate,
 * binding row order, dimensions, biases and full calibration lineage.
 * @param example_shape Positive [batch,in_features] concrete signature.
 * @param options Opset 17, batch policy and complete output byte ceiling.
 * @param limits Complete aggregate, channel, feature and calibration ceilings.
 * @return Owned graph and complete aggregate identity; profile is
 * jyppx.linear.ptq.ort.u8u8.per_output_channel/v1. No kernel/quality claim.
 * @throws Error For source identity, order, shape, bias/calibration mismatch,
 * resource bounds, nonrepresentable scales or backend serialization failure.
 * @note Source storage is borrowed only for this call. Independent calls own
 * their graph state. Qualify runtime trace, task accuracy and speed separately.
 */
StaticInt8LinearExportResult export_static_int8_linear_per_output_channel(
    std::span<const std::byte> source_artifact,
    const ArtifactDigest& expected_source_digest,
    std::span<const std::int64_t> example_shape,
    const ExportOptions& options = {},
    quantization::QuantizationLimits limits = {});

}  // namespace jyppx::liuding::onnx_export

#endif  // JYPPX_LIUDING_DEPLOYMENT_ONNX_EXPORT_STATIC_INT8_LINEAR_HPP
