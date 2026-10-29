// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/checkpoint_manifest.hpp>
#include <jyppx/liuding/interop/file_state_archive.hpp>
#include <jyppx/liuding/interop/model_bundle.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>

#include <torch/cuda.h>
#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::liuding::cli {
namespace {

constexpr std::uint64_t kArtifactBytes = 512ULL * 1024U * 1024U;
constexpr std::uint64_t kMaxTensorElements = 16ULL * 1024U * 1024U;
constexpr std::uint64_t kMaxWorkElements = 1ULL << 30U;

std::string action_for(const Arguments& arguments, std::string_view command) {
  const auto& words = arguments.words();
  if (words.size() != 2U || words.front() != command) {
    usage(command, "Supply exactly one documented subcommand.");
  }
  return words[1];
}

std::filesystem::path existing_artifact(const std::string& name) {
  const auto path = std::filesystem::u8path(name);
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error) || error) {
    throw IoError("Command input is not an existing regular file",
                  "Supply the path of a readable artifact file.");
  }
  const auto size = std::filesystem::file_size(path, error);
  if (error) throw IoError("Could not size the artifact input");
  if (size > kArtifactBytes) {
    throw ResourceExhausted("Artifact input exceeds the CLI limit of 512 MiB");
  }
  return path;
}

std::span<const std::byte> bytes_of(std::string_view value) {
  return std::as_bytes(std::span(value.data(), value.size()));
}

interop::FileStateArchiveOptions archive_options() {
  interop::FileStateArchiveOptions options;
  options.max_artifact_bytes = kArtifactBytes;
  options.max_field_bytes = kArtifactBytes;
  options.manifest_limits.max_total_blob_bytes = kArtifactBytes;
  options.read_only = true;
  return options;
}

Json checkpoint_document(const std::filesystem::path& path) {
  const auto options = archive_options();
  const interop::FileStateArchive archive(path, options);
  // Opening validates the complete container and every blob digest. Never
  // invoke a participant, torch::load, pickle, or executable restore hook.
  return parse_json(interop::checkpoint_artifact_manifest_json(
      archive.manifest(), options.manifest_limits));
}

Json checkpoint_summary(const std::string& path, const Json& manifest) {
  std::uint64_t blob_count = 0U;
  std::uint64_t blob_bytes = 0U;
  for (const auto& participant : manifest.at("participants")) {
    blob_count += participant.at("blobs").size();
    for (const auto& blob : participant.at("blobs")) {
      blob_bytes += blob.at("size_bytes").get<std::uint64_t>();
    }
  }
  return Json{{"path", path},
              {"format", "jyppx-file-state-archive/v1"},
              {"valid", true},
              {"participant_count", manifest.at("participants").size()},
              {"blob_count", blob_count},
              {"blob_bytes", blob_bytes},
              {"manifest_sha256", sha256_hex(bytes_of(manifest.dump()))},
              {"verification",
               {{"artifact_digest", "sha256_verified"},
                {"blob_digests", "sha256_verified"},
                {"participant_deserialization", false}}},
              {"max_artifact_bytes", kArtifactBytes}};
}

Json named_entries(const Json& entries, std::string_view identity) {
  auto result = Json::object();
  for (const auto& entry : entries) {
    result[entry.at(std::string(identity)).get<std::string>()] = entry;
  }
  return result;
}

Json comparison_manifest(const Json& manifest) {
  auto result = manifest;
  auto participants = Json::object();
  for (const auto& participant : manifest.at("participants")) {
    auto entry = participant;
    auto& state = entry.at("state");
    state["fields"] = named_entries(state.at("fields"), "name");
    entry["blobs"] = named_entries(entry.at("blobs"), "field_name");
    const auto participant_id = state.at("participant_id").get<std::string>();
    participants[participant_id] = std::move(entry);
  }
  result["participants"] = std::move(participants);
  return result;
}

std::string pointer_segment(std::string_view key) {
  std::string result;
  for (const char character : key) {
    if (character == '~')
      result += "~0";
    else if (character == '/')
      result += "~1";
    else
      result += character;
  }
  return result;
}

void changes_between(const Json& baseline, const Json& candidate,
                     const std::string& path, Json& changes) {
  if (baseline == candidate) return;
  if (baseline.is_object() && candidate.is_object()) {
    for (const auto& entry : baseline.items()) {
      const auto child = path + "/" + pointer_segment(entry.key());
      if (candidate.contains(entry.key())) {
        changes_between(entry.value(), candidate.at(entry.key()), child,
                        changes);
      } else {
        changes.push_back({{"path", child},
                           {"change", "removed"},
                           {"baseline", entry.value()},
                           {"candidate", nullptr}});
      }
    }
    for (const auto& entry : candidate.items()) {
      if (!baseline.contains(entry.key())) {
        changes.push_back({{"path", path + "/" + pointer_segment(entry.key())},
                           {"change", "added"},
                           {"baseline", nullptr},
                           {"candidate", entry.value()}});
      }
    }
    return;
  }
  changes.push_back({{"path", path},
                     {"change", "modified"},
                     {"baseline", baseline},
                     {"candidate", candidate}});
}

interop::ModelBundleVerifyOptions bundle_options(const Arguments& arguments) {
  const auto trust = arguments.optional("require-trusted").value_or("false");
  if (trust != "true" && trust != "false")
    usage("model.bundle", "--require-trusted accepts true or false.");
  interop::ModelBundleVerifyOptions options;
  options.require_trusted_signature = trust == "true";
  return options;
}

Json bundle_summary(std::string_view operation, const std::string& path,
                    const interop::ModelBundleVerification& verified) {
  const auto signature =
      verified.signature_status ==
              interop::BundleSignatureStatus::unsigned_bundle
          ? "unsigned"
      : verified.signature_status == interop::BundleSignatureStatus::trusted
          ? "trusted"
          : "present_unverified";
  return Json{{"operation", operation},
              {"path", path},
              {"format", "jyppx.model_bundle/v" +
                             std::to_string(verified.manifest.schema_version)},
              {"valid", true},
              {"manifest_sha256", verified.manifest_digest.hex},
              {"verified_file_bytes", verified.verified_file_bytes},
              {"verified_tensors", verified.verified_tensors},
              {"integrity_only_models", verified.integrity_only_models},
              {"signature_status", signature},
              {"authenticated", verified.signature_status ==
                                    interop::BundleSignatureStatus::trusted},
              {"runtime_executed", false},
              {"descriptor_payloads_validated", false},
              {"runtime_requirements_checked", false},
              {"manifest", Json::parse(interop::serialize_model_bundle_manifest(
                               verified.manifest))}};
}

Result pack_bundle(const Arguments& arguments) {
  arguments.allow(
      {"manifest", "manifest-sha256", "source", "output", "require-trusted"});
  const auto options = bundle_options(arguments);
  const ArtifactDigest expected{.hex = arguments.require("manifest-sha256")};
  validate_artifact_digest(expected);
  const auto json = read_text(arguments.require("manifest"),
                              options.limits.max_manifest_bytes);
  if (sha256_hex(bytes_of(json)) != expected.hex)
    throw SerializationError("Bundle source manifest digest mismatch");
  const auto manifest =
      interop::deserialize_model_bundle_manifest(json, options.limits);
  const auto source = std::filesystem::u8path(arguments.require("source"));
  const auto output = arguments.require("output");
  std::vector<interop::BundlePublishSource> sources;
  sources.reserve(manifest.files.size());
  for (const auto& file : manifest.files)
    sources.push_back(
        {.file_id = file.id, .source_path = source / file.relative_path});
  const auto verified = interop::publish_model_bundle(
      std::filesystem::u8path(output), manifest, sources, options);
  auto result = bundle_summary("pack", output, verified);
  result["source_manifest_sha256"] = expected.hex;
  return {std::move(result), 0};
}

}  // namespace

Result model_command(const Arguments& arguments) {
  const auto& words = arguments.words();
  if (words.size() == 3U && words[0] == "model" && words[1] == "onnx")
    return onnx_command(arguments);
  if (words.size() == 3U && words[0] == "model" && words[1] == "bundle") {
    if (words[2] == "pack") return pack_bundle(arguments);
    if (words[2] != "inspect" && words[2] != "verify")
      usage("model.bundle",
            "Choose pack, inspect or verify for a model bundle directory.");
    arguments.allow({"path", "require-trusted"});
    const auto name = arguments.require("path");
    const auto verified = interop::verify_model_bundle_directory(
        std::filesystem::u8path(name), bundle_options(arguments));
    return {bundle_summary(words[2], name, verified), 0};
  }
  const auto action = action_for(arguments, "model");
  if (action != "inspect" && action != "verify") {
    usage("model", "Choose inspect or verify for a safetensors file.");
  }
  arguments.allow({"path"});
  const auto name = arguments.require("path");
  const auto path = existing_artifact(name);
  const auto content =
      read_text(path, static_cast<std::size_t>(kArtifactBytes));
  interop::SafeTensorsLimits limits;
  limits.max_file_bytes = kArtifactBytes;
  limits.max_payload_bytes = kArtifactBytes;
  limits.max_tensor_bytes = kArtifactBytes;
  const interop::SafeTensorsView view(bytes_of(content), limits);
  const auto& directory = view.directory();
  auto tensors = Json::array();
  for (const auto& tensor : directory.tensors) {
    tensors.push_back(
        {{"name", tensor.name},
         {"dtype", std::string(interop::safetensors_dtype_name(tensor.dtype))},
         {"shape", tensor.shape},
         {"data_offsets", Json::array({tensor.data_begin, tensor.data_end})},
         {"size_bytes", tensor.data_end - tensor.data_begin},
         {"sha256", sha256_hex(view.tensor_bytes(tensor.name))}});
  }
  return {Json{{"operation", action},
               {"path", name},
               {"format", "safetensors"},
               {"valid", true},
               {"file_size_bytes", content.size()},
               {"sha256", view.digest().hex},
               {"metadata", directory.metadata},
               {"tensor_count", directory.tensors.size()},
               {"payload_offset", directory.payload_offset},
               {"payload_size_bytes", directory.payload_size},
               {"tensors", std::move(tensors)},
               {"verification",
                {{"structure_and_full_payload_bounds", true},
                 {"sha256", "computed_content_identity"},
                 {"embedded_checksum", false}}},
               {"max_artifact_bytes", kArtifactBytes}},
          0};
}

Result checkpoint_command(const Arguments& arguments) {
  const auto action = action_for(arguments, "checkpoint");
  if (action == "inspect" || action == "verify") {
    arguments.allow({"path"});
    const auto name = arguments.require("path");
    const auto path = existing_artifact(name);
    const auto manifest = checkpoint_document(path);
    auto data = checkpoint_summary(name, manifest);
    data["operation"] = action;
    data["manifest"] = manifest;
    return {std::move(data), 0};
  }
  if (action == "diff") {
    arguments.allow({"baseline", "candidate"});
    const auto baseline_name = arguments.require("baseline");
    const auto candidate_name = arguments.require("candidate");
    const auto baseline_path = existing_artifact(baseline_name);
    const auto candidate_path = existing_artifact(candidate_name);
    // Each archive is released after its validated metadata is copied, so
    // comparing two inputs does not retain two complete blob snapshots.
    const auto baseline = checkpoint_document(baseline_path);
    const auto candidate = checkpoint_document(candidate_path);
    auto changes = Json::array();
    changes_between(comparison_manifest(baseline),
                    comparison_manifest(candidate), "", changes);
    const bool equal = changes.empty();
    return {Json{{"equal", equal},
                 {"baseline", checkpoint_summary(baseline_name, baseline)},
                 {"candidate", checkpoint_summary(candidate_name, candidate)},
                 {"comparison", "complete_manifest_and_verified_blob_sha256"},
                 {"path_format",
                  "JSON Pointer; participants/fields/blobs keyed by identity"},
                 {"changes", std::move(changes)}},
            equal ? 0 : 1};
  }
  usage("checkpoint", "Choose inspect, verify, or diff.");
}

Result benchmark_command(const Arguments& arguments) {
  if (action_for(arguments, "benchmark") != "tensor") {
    usage("benchmark", "Choose tensor.");
  }
  arguments.allow({"device", "size", "samples", "warmup"});
  const auto device_name = arguments.optional("device").value_or("cpu");
  if (device_name != "cpu" && device_name != "cuda") {
    usage("benchmark.device", "Choose cpu or cuda (device index zero).");
  }
  const auto size = arguments.number("size", 1048576U, kMaxTensorElements);
  const auto samples = arguments.number("samples", 20U, 1000U);
  const auto warmup = arguments.number("warmup", 5U, 100U);
  if (size == 0U || samples == 0U) {
    usage("benchmark.tensor",
          "Size and samples must be positive; warmup may be zero.");
  }
  if (size > kMaxWorkElements / (samples + warmup)) {
    usage("benchmark.tensor",
          "Keep size * (samples + warmup) at most 1073741824.");
  }

  try {
    const bool cuda = device_name == "cuda";
    if (cuda && !torch::cuda::is_available()) {
      throw DeviceError("CUDA device zero is unavailable",
                        "Run doctor --device cuda or select --device cpu.");
    }
    const torch::NoGradGuard no_grad;
    const auto device =
        cuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto input =
        torch::full({static_cast<std::int64_t>(size)}, -0.25, options);
    auto output = torch::empty_like(input);
    const auto synchronize = [&] {
      if (cuda) torch::cuda::synchronize(0);
    };
    const auto workload = [&] {
      torch::mul_out(output, input, 2.0);
      output.add_(1.5);
      output.relu_();
    };
    synchronize();
    for (std::uint64_t index = 0U; index < warmup; ++index) {
      workload();
      synchronize();
    }
    std::vector<double> durations;
    durations.reserve(static_cast<std::size_t>(samples));
    for (std::uint64_t index = 0U; index < samples; ++index) {
      const auto started = std::chrono::steady_clock::now();
      workload();
      synchronize();
      const double milliseconds =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started)
              .count();
      if (!std::isfinite(milliseconds) || milliseconds < 0.0) {
        throw InternalError("Benchmark clock produced an invalid duration");
      }
      durations.push_back(milliseconds);
    }
    // Validate the complete result outside the timing window. Exact constants
    // make every expected FP32 output equal to one on both CPU and CUDA.
    const bool correct = output.eq(1.0).all().item<bool>();
    synchronize();
    if (!correct)
      throw NumericalError("Tensor benchmark output verification failed");
    std::sort(durations.begin(), durations.end());
    const auto middle = durations.size() / 2U;
    const double median =
        durations.size() % 2U == 0U
            ? (durations[middle - 1U] + durations[middle]) / 2.0
            : durations[middle];
    const auto p95_index =
        static_cast<std::size_t>((samples * 95U + 99U) / 100U - 1U);
    return {
        Json{{"workload", "jyppx.tensor.scalar_mul_add_relu.fp32/v1"},
             {"operation", "out = relu(input * 2.0 + 1.5); input = -0.25"},
             {"scope", "fixed tensor operation latency"},
             {"device", device.str()},
             {"dtype", "float32"},
             {"shape", Json::array({size})},
             {"samples", samples},
             {"warmup", warmup},
             {"input_output_bytes", size * 2U * sizeof(float)},
             {"sample_completion",
              cuda ? "cuda_device_synchronize_each_sample" : "synchronous_cpu"},
             {"output_verified", true},
             {"timing_unit", "milliseconds"},
             {"median", median},
             {"p95", durations[p95_index]},
             {"min", durations.front()},
             {"max", durations.back()},
             {"percentile_method", "nearest_rank"},
             {"intraop_threads", at::get_num_threads()},
             {"interop_threads", at::get_num_interop_threads()},
             {"build", parse_json(build_fingerprint_json())},
             {"limits",
              {{"max_size", kMaxTensorElements},
               {"max_samples", 1000U},
               {"max_warmup", 100U},
               {"max_total_elements", kMaxWorkElements}}}},
        0};
  } catch (const c10::Error&) {
    std::throw_with_nested(DeviceError(
        "LibTorch could not complete the tensor benchmark",
        "Run doctor for the selected device and check available memory."));
  }
}

}  // namespace jyppx::liuding::cli
