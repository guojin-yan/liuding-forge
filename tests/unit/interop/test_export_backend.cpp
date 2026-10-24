// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/interop/export_backend.hpp>

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace interop = jyppx::liuding::interop;

jyppx::liuding::ArtifactDigest Digest(char identity = 'a') {
  return {.algorithm = jyppx::liuding::ArtifactDigestAlgorithm::sha256,
          .hex = std::string(64U, identity)};
}

jyppx::liuding::ArtifactRef Artifact(
    char identity, std::string logical_name,
    std::string media_type = "application/octet-stream") {
  return {.digest = Digest(identity),
          .size_bytes = 128U,
          .logical_name = std::move(logical_name),
          .media_type = std::move(media_type),
          .producer = "unit-fixture"};
}

std::array<interop::BackendDependency, 6> Dependencies() {
  return {{{.name = "python", .version = "3.12.8"},
           {.name = "torch", .version = "2.5.1+cpu"},
           {.name = "onnx", .version = "1.17.0"},
           {.name = "onnxscript", .version = "0.1.0"},
           {.name = "onnxruntime", .version = "1.20.1"},
           {.name = "safetensors", .version = "0.6.2"}}};
}

interop::BackendDescriptor Descriptor() {
  const auto dependencies = Dependencies();
  return interop::make_onnx_sidecar_descriptor("1.0.0", Digest(), dependencies);
}

interop::ExportRequest Request() {
  interop::ExportRequest request;
  request.request_id = "request-001";
  request.backend_id = "onnx.sidecar";
  request.source_kind = interop::ExportSourceKind::python_factory;
  request.source = Artifact('b', "fixture-factory-code");
  request.state = Artifact('c', "fixture-state", "application/safetensors");
  request.factory_entry_point = "fixture.models:make_model";
  request.invocation_schema =
      Artifact('d', "invocation-schema", "application/json");
  request.verification_cases = {Artifact('e', "case-fixed")};
  request.options = {{"opset", "18"}};
  request.trust_source = true;
  request.environment_lock = Digest();
  return request;
}

interop::BackendProbe Probe(const interop::ExportRequest& request) {
  const auto descriptor = Descriptor();
  interop::BackendProbe probe;
  probe.backend_id = descriptor.backend_id;
  probe.backend_version = descriptor.backend_version;
  probe.protocol_version = descriptor.protocol_version;
  probe.environment_lock = descriptor.environment_lock;
  probe.dependencies = descriptor.dependencies;
  probe.operations = descriptor.operations;
  probe.devices = {"cpu"};
  probe.dtypes = {"float32"};
  probe.dynamic_shapes_verified = true;
  probe.official_onnx_exporter = true;
  probe.onnx_checker = true;
  probe.onnx_runtime = true;
  probe.request_digest = interop::backend_request_digest(request);
  probe.model_preflight_passed = true;
  probe.evidence =
      Artifact('f', "test-only-synthetic-probe", "application/json");
  return probe;
}

interop::ExportCandidate Candidate(const interop::ExportRequest& request) {
  interop::ExportCandidate candidate;
  candidate.request_id = request.request_id;
  candidate.backend_id = request.backend_id;
  candidate.backend_version = "1.0.0";
  candidate.artifact =
      Artifact('b', "test-only-staged-output", "application/onnx");
  auto& verification = candidate.verification;
  verification.request_digest = interop::backend_request_digest(request);
  verification.artifact_digest = candidate.artifact.digest;
  verification.environment_lock = request.environment_lock;
  verification.schema_passed = true;
  verification.checker_passed = true;
  verification.shape_inference = interop::ShapeInferenceStatus::passed;
  verification.runtime_passed = true;
  verification.numerical_passed = true;
  verification.cases_run = request.verification_cases.size();
  verification.cases_passed = request.verification_cases.size();
  verification.dynamic_cases_passed = request.dynamic_shapes ? 1U : 0U;
  verification.absolute_tolerance = request.absolute_tolerance;
  verification.relative_tolerance = request.relative_tolerance;
  verification.evidence =
      Artifact('c', "test-only-synthetic-validation", "application/json");
  return candidate;
}

bool HasIssue(const interop::SupportReport& report, std::string_view code) {
  return std::any_of(report.issues.begin(), report.issues.end(),
                     [&](const auto& issue) { return issue.code == code; });
}

std::string Replace(std::string text, std::string_view from,
                    std::string_view to) {
  const auto position = text.find(from);
  if (position == std::string::npos) {
    ADD_FAILURE() << "Unit fixture mutation did not match its source text";
    return text;
  }
  text.replace(position, from.size(), to);
  return text;
}

TEST(ExportBackendTest, ExplicitDescriptorsDoNotClaimRuntimeAvailability) {
  const auto descriptor = Descriptor();
  EXPECT_EQ(descriptor.role, interop::BackendRole::exporter);
  EXPECT_EQ(descriptor.execution, interop::BackendExecution::out_of_process);
  EXPECT_EQ(descriptor.protocol_version, 1U);
  EXPECT_EQ(descriptor.sources,
            (std::vector{interop::ExportSourceKind::python_factory,
                         interop::ExportSourceKind::exported_program}));
  const auto json = interop::serialize_backend_descriptor(descriptor);
  EXPECT_NE(json.find("\"status\":\"declared-unprobed\""), std::string::npos);
  EXPECT_NE(json.find("\"dependencies\""), std::string::npos);
  EXPECT_NE(json.find("\"allowed_options\""), std::string::npos);
  const auto report = interop::preflight_backend(descriptor, Request());
  EXPECT_EQ(report.decision, interop::SupportDecision::conditional);
  EXPECT_TRUE(HasIssue(report, "environment.unprobed"));
}

TEST(ExportBackendTest,
     EnvironmentProbeRequiresExactRequestBoundModelPreflight) {
  const auto descriptor = Descriptor();
  const auto request = Request();
  auto probe = Probe(request);
  EXPECT_EQ(interop::preflight_backend(descriptor, request, &probe).decision,
            interop::SupportDecision::supported);
  probe.request_digest.reset();
  probe.model_preflight_passed = false;
  const auto environment_only =
      interop::preflight_backend(descriptor, request, &probe);
  EXPECT_EQ(environment_only.decision, interop::SupportDecision::conditional);
  EXPECT_TRUE(HasIssue(environment_only, "model.preflight_required"));
  probe = Probe(request);
  probe.request_digest = Digest('f');
  EXPECT_TRUE(HasIssue(interop::preflight_backend(descriptor, request, &probe),
                       "probe.request_mismatch"));
  probe = Probe(request);
  probe.model_preflight_passed = false;
  probe.unsupported_operators = {"custom::unsupported"};
  const auto unsupported =
      interop::preflight_backend(descriptor, request, &probe);
  EXPECT_EQ(unsupported.decision, interop::SupportDecision::unsupported);
  EXPECT_TRUE(HasIssue(unsupported, "operator.unsupported"));
}

TEST(ExportBackendTest,
     NativeCppStateAndUntrustedExecutableSourcesFailPreflight) {
  const auto descriptor = Descriptor();
  auto request = Request();
  request.source_kind = interop::ExportSourceKind::native_state;
  request.factory_entry_point.clear();
  request.state.reset();
  auto report = interop::preflight_backend(descriptor, request);
  EXPECT_EQ(report.decision, interop::SupportDecision::unsupported);
  EXPECT_TRUE(HasIssue(report, "source.unsupported"));
  request = Request();
  request.trust_source = false;
  report = interop::preflight_backend(descriptor, request);
  EXPECT_TRUE(HasIssue(report, "source.trust_required"));
  request = Request();
  request.source_kind = interop::ExportSourceKind::exported_program;
  request.factory_entry_point.clear();
  request.state.reset();
  const auto probe = Probe(request);
  EXPECT_EQ(interop::preflight_backend(descriptor, request, &probe).decision,
            interop::SupportDecision::supported);
}

TEST(ExportBackendTest, ReportsAllDeclaredAndObservedCapabilityViolations) {
  const auto descriptor = Descriptor();
  auto request = Request();
  request.device = "cuda";
  request.dtype = "float16";
  request.dynamic_shapes = true;
  request.quantization_mode = "int8";
  request.prepost_mode = "embedded";
  request.options = {{"opset", "999"}, {"unrecognized", "value"}};
  auto probe = Probe(request);
  probe.dependencies.erase(probe.dependencies.begin());
  probe.dependencies[0].version = "9.9.9";
  probe.protocol_version = 2U;
  probe.dynamic_shapes_verified = false;
  probe.onnx_runtime = false;
  const auto report = interop::preflight_backend(descriptor, request, &probe);
  EXPECT_EQ(report.decision, interop::SupportDecision::unsupported);
  for (const auto code :
       {"dtype.unsupported", "quantization.unsupported", "prepost.unsupported",
        "option.unsupported", "dependency.missing",
        "dependency.version_mismatch", "probe.protocol_mismatch",
        "probe.device.unsupported", "probe.dtype.unsupported",
        "probe.dynamic_unverified", "onnx.official_validation_unavailable",
        "dynamic_cases.required"})
    EXPECT_TRUE(HasIssue(report, code)) << code;
}

TEST(ExportBackendTest, TorchscriptRegistryIsCompatibilityRuntimeOnly) {
  auto descriptor = interop::make_torchscript_compatibility_descriptor(
      "1.0.0", Digest(), "2.5.1+cpu");
  EXPECT_TRUE(descriptor.compatibility_only);
  EXPECT_TRUE(descriptor.deprecated);
  EXPECT_EQ(descriptor.role, interop::BackendRole::runtime);
  EXPECT_EQ(descriptor.operations,
            (std::vector{interop::BackendOperation::load,
                         interop::BackendOperation::run,
                         interop::BackendOperation::inspect}));
  interop::BackendRegistry exporters(interop::BackendRole::exporter);
  interop::BackendRegistry runtimes(interop::BackendRole::runtime);
  exporters.add(Descriptor());
  runtimes.add(descriptor);
  EXPECT_NE(exporters.find("onnx.sidecar"), nullptr);
  EXPECT_EQ(exporters.find("torchscript.compatibility"), nullptr);
  EXPECT_NE(runtimes.find("torchscript.compatibility"), nullptr);
  EXPECT_THROW(exporters.add(descriptor), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(exporters.add(Descriptor()), jyppx::liuding::InvalidArgument);
  auto request = Request();
  request.backend_id = descriptor.backend_id;
  request.source_kind = interop::ExportSourceKind::torchscript;
  request.state.reset();
  request.factory_entry_point.clear();
  request.target = interop::ArtifactFormat::torchscript;
  request.options.clear();
  request.operation = interop::BackendOperation::trace;
  auto report = interop::preflight_backend(descriptor, request);
  EXPECT_EQ(report.decision, interop::SupportDecision::unsupported);
  EXPECT_TRUE(HasIssue(report, "operation.unsupported"));
  EXPECT_TRUE(HasIssue(report, "trace.opt_in_required"));
  request.trace_opt_in = true;
  EXPECT_EQ(interop::preflight_backend(descriptor, request).decision,
            interop::SupportDecision::unsupported);
}

TEST(ExportBackendTest, RequestProbeAndCandidateProtocolRoundTripCanonically) {
  const auto request = Request();
  const auto probe = Probe(request);
  const auto candidate = Candidate(request);
  const auto request_json = interop::serialize_sidecar_request(request);
  const auto probe_json = interop::serialize_sidecar_probe(probe);
  const auto candidate_json = interop::serialize_sidecar_candidate(candidate);
  EXPECT_EQ(interop::serialize_sidecar_request(
                interop::deserialize_sidecar_request(request_json)),
            request_json);
  EXPECT_EQ(interop::serialize_sidecar_probe(
                interop::deserialize_sidecar_probe(probe_json)),
            probe_json);
  EXPECT_EQ(interop::serialize_sidecar_candidate(
                interop::deserialize_sidecar_candidate(candidate_json)),
            candidate_json);
  EXPECT_NE(candidate_json.find("\"kind\":\"export.candidate\""),
            std::string::npos);
  EXPECT_EQ(candidate_json.find("\"success\""), std::string::npos);
  EXPECT_NO_THROW(interop::validate_export_candidate_metadata(
      Descriptor(), request, probe,
      interop::deserialize_sidecar_candidate(candidate_json)));
}

TEST(ExportBackendTest, RequestIdentityIncludesSchemaCasesOptionsAndTolerance) {
  const auto original = Request();
  const auto digest = interop::backend_request_digest(original);
  auto changed = original;
  changed.relative_tolerance *= 2.0;
  EXPECT_FALSE(interop::backend_request_digest(changed) == digest);
  changed = original;
  changed.invocation_schema->digest = Digest('f');
  EXPECT_FALSE(interop::backend_request_digest(changed) == digest);
  changed = original;
  changed.verification_cases.push_back(Artifact('f', "second-case"));
  EXPECT_FALSE(interop::backend_request_digest(changed) == digest);
  changed = original;
  changed.options["opset"] = "19";
  EXPECT_FALSE(interop::backend_request_digest(changed) == digest);
}

TEST(ExportBackendTest,
     ProtocolRejectsUnknownCriticalDuplicateAndFutureFields) {
  const auto json = interop::serialize_sidecar_request(Request());
  const std::vector invalid{
      Replace(json, "\"protocol_version\":1", "\"protocol_version\":2"),
      Replace(json, "\"schema_version\":1", "\"schema_version\":2"),
      Replace(json, "\"kind\":\"export.request\"", "\"kind\":\"probe.result\""),
      Replace(json, "\"source_kind\":\"python_factory\"",
              "\"source_kind\":\"inferred_cpp_graph\""),
      Replace(json, "\"request_id\":\"request-001\"",
              "\"request_id\":\"request-001\",\"request_id\":\"other\""),
      Replace(json, "\"request_id\":\"request-001\"",
              "\"request_id\":\"request-001\",\"request_\\u0069d\":\"other\""),
      Replace(json, "\"resources\":{",
              "\"resources\":{\"unknown_critical\":true,"),
      Replace(json, "\"source\":{",
              "\"source\":{\"path\":\"unexpected-local-path\","),
      Replace(json, "\"cpu_threads\":1", "\"cpu_threads\":true"),
      Replace(json, "\"memory_bytes\":4294967296", "\"memory_bytes\":-1"),
      Replace(json, "\"trust_source\":true", "\"trust_source\":1"),
      json + "{}",
      json.substr(0U, json.size() - 1U),
      std::string("\xef\xbb\xbf") + json};
  for (const auto& malformed : invalid)
    EXPECT_THROW(interop::deserialize_sidecar_request(malformed),
                 jyppx::liuding::Error);
  EXPECT_THROW(interop::deserialize_sidecar_request(std::string("{\xc0\xaf}")),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(interop::deserialize_sidecar_request(
                   "{\"x\":[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]}"),
               jyppx::liuding::ResourceExhausted);
}

TEST(ExportBackendTest, ProtocolBoundsBytesNodesStringsAndHostIntegerWidths) {
  const auto json = interop::serialize_sidecar_request(Request());
  EXPECT_THROW(interop::deserialize_sidecar_request(json, 16U),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(interop::deserialize_sidecar_request(json, 0U),
               jyppx::liuding::InvalidArgument);
  const auto oversized =
      Replace(json, "\"request-001\"", "\"" + std::string(4097U, 'a') + "\"");
  EXPECT_THROW(interop::deserialize_sidecar_request(oversized),
               jyppx::liuding::ResourceExhausted);
  const auto overflow =
      Replace(json, "\"cpu_threads\":1", "\"cpu_threads\":4294967296");
  EXPECT_THROW(interop::deserialize_sidecar_request(overflow),
               jyppx::liuding::SerializationError);
  const auto fraction =
      Replace(json, "\"cpu_threads\":1", "\"cpu_threads\":1.0");
  EXPECT_THROW(interop::deserialize_sidecar_request(fraction),
               jyppx::liuding::SerializationError);
  std::string nodes = "{\"x\":[";
  for (std::size_t index = 0U; index < 65536U; ++index) nodes += "0,";
  nodes += "0]}";
  EXPECT_THROW(interop::deserialize_sidecar_request(nodes),
               jyppx::liuding::ResourceExhausted);
  auto request = Request();
  request.resources.max_control_bytes = 64U;
  EXPECT_THROW(interop::serialize_sidecar_request(request),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(interop::serialize_sidecar_probe(Probe(Request()), 8U),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(interop::serialize_sidecar_candidate(Candidate(Request()), 8U),
               jyppx::liuding::ResourceExhausted);
}

TEST(ExportBackendTest, DescriptorPinsAndRequestImportIdentifiersAreStrict) {
  auto descriptor = Descriptor();
  descriptor.dependencies[0].version = "latest";
  EXPECT_THROW(interop::validate_backend_descriptor(descriptor),
               jyppx::liuding::InvalidArgument);
  descriptor = Descriptor();
  descriptor.dependencies[0].version = ">=3.12";
  EXPECT_THROW(interop::validate_backend_descriptor(descriptor),
               jyppx::liuding::InvalidArgument);
  descriptor = Descriptor();
  descriptor.dependencies.push_back(descriptor.dependencies[0]);
  EXPECT_THROW(interop::validate_backend_descriptor(descriptor),
               jyppx::liuding::InvalidArgument);
  auto dependencies = Dependencies();
  EXPECT_THROW(interop::make_onnx_sidecar_descriptor(
                   "1.0.0", Digest(), std::span(dependencies).first(5U)),
               jyppx::liuding::InvalidArgument);
  for (const auto entry :
       {"C:/model.py:factory", "model.py:factory()", "a..b:factory",
        "a:factory;command", "a:factory:", "a:.factory"}) {
    auto request = Request();
    request.factory_entry_point = entry;
    EXPECT_THROW(interop::serialize_sidecar_request(request),
                 jyppx::liuding::InvalidArgument);
  }
  auto request = Request();
  request.absolute_tolerance = (std::numeric_limits<double>::quiet_NaN)();
  EXPECT_THROW(interop::serialize_sidecar_request(request),
               jyppx::liuding::InvalidArgument);
  request = Request();
  request.resources.cpu_threads = 0U;
  EXPECT_THROW(interop::serialize_sidecar_request(request),
               jyppx::liuding::InvalidArgument);
}

TEST(ExportBackendTest, CandidateRequiresEveryActualOfficialVerificationStage) {
  const auto request = Request();
  const auto descriptor = Descriptor();
  const auto probe = Probe(request);
  const auto good = Candidate(request);
  const auto rejected = [&](const interop::ExportCandidate& candidate) {
    EXPECT_THROW(interop::validate_export_candidate_metadata(
                     descriptor, request, probe, candidate),
                 jyppx::liuding::CompatibilityError);
  };
  auto invalid = good;
  invalid.verification.schema_passed = false;
  rejected(invalid);
  invalid = good;
  invalid.verification.checker_passed = false;
  rejected(invalid);
  invalid = good;
  invalid.verification.runtime_passed = false;
  rejected(invalid);
  invalid = good;
  invalid.verification.numerical_passed = false;
  rejected(invalid);
  invalid = good;
  invalid.verification.shape_inference =
      interop::ShapeInferenceStatus::not_checked;
  rejected(invalid);
  invalid.verification.shape_inference =
      interop::ShapeInferenceStatus::not_applicable;
  rejected(invalid);
  invalid.verification.shape_inference_reason =
      "official stage not applicable to this graph";
  EXPECT_NO_THROW(interop::validate_export_candidate_metadata(
      descriptor, request, probe, invalid));
  invalid = good;
  invalid.verification.cases_passed = 0U;
  rejected(invalid);
  invalid = good;
  invalid.verification.relative_tolerance *= 10.0;
  rejected(invalid);
  invalid = good;
  invalid.verification.request_digest = Digest('f');
  rejected(invalid);
  invalid = good;
  invalid.verification.artifact_digest = Digest('f');
  rejected(invalid);
  invalid = good;
  invalid.request_id = "stale-request";
  rejected(invalid);
  invalid = good;
  invalid.artifact.media_type = "application/octet-stream";
  rejected(invalid);
}

TEST(ExportBackendTest, DynamicCandidateRequiresBothFixedAndDynamicCases) {
  auto request = Request();
  request.dynamic_shapes = true;
  request.verification_cases.push_back(Artifact('f', "case-dynamic"));
  const auto probe = Probe(request);
  auto candidate = Candidate(request);
  EXPECT_NO_THROW(interop::validate_export_candidate_metadata(
      Descriptor(), request, probe, candidate));
  candidate.verification.dynamic_cases_passed = 0U;
  EXPECT_THROW(interop::validate_export_candidate_metadata(
                   Descriptor(), request, probe, candidate),
               jyppx::liuding::CompatibilityError);
  candidate.verification.dynamic_cases_passed = 2U;
  EXPECT_THROW(interop::validate_export_candidate_metadata(
                   Descriptor(), request, probe, candidate),
               jyppx::liuding::CompatibilityError);
}

}  // namespace
