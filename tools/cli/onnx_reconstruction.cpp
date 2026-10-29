// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"

#if defined(JYPPX_LIUDING_CLI_ONNX_EXPORT) && \
    defined(JYPPX_LIUDING_CLI_ONNX_RUNTIME)
#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/file_state_archive.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#if defined(JYPPX_LIUDING_CLI_ONNX_MODELS)
#include <jyppx/liuding/models/registry.hpp>
#endif
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>

namespace jyppx::liuding::cli {
namespace {
constexpr std::uint64_t kCheckpointLimit = 256U << 20U;
constexpr std::uint64_t kTensorLimit = 64U << 20U;

std::span<const std::byte> Bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}

struct Staging {
  std::filesystem::path path;
  Staging() {
    const auto root = std::filesystem::temp_directory_path();
    const auto tick =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (unsigned i = 0; i < 128; ++i) {
      auto candidate = root / ("liuding-onnx-source-" + std::to_string(tick) +
                               "-" + std::to_string(i));
      std::error_code error;
      if (std::filesystem::create_directory(candidate, error)) {
#if !defined(_WIN32)
        std::filesystem::permissions(
            candidate, std::filesystem::perms::owner_all,
            std::filesystem::perm_options::replace, error);
        if (error) {
          std::error_code ignored;
          std::filesystem::remove(candidate, ignored);
          throw IoError("Cannot restrict ONNX source staging permissions");
        }
#endif
        path = std::move(candidate);
        return;
      }
      if (error) throw IoError("Cannot create ONNX source staging directory");
    }
    throw IoError("Cannot allocate ONNX source staging directory");
  }
  ~Staging() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

void Write(const std::filesystem::path& path,
           std::span<const std::byte> bytes) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  stream.close();
  if (!stream) throw IoError("Cannot stage ONNX source artifact");
}

experiment::RunArtifactReference CopyReference(
    experiment::SQLiteRunStore& store, std::string_view run,
    const std::string& id, const std::filesystem::path& output,
    std::uint64_t maximum, bool checkpoint) {
  const auto ref = store.find_reference(run, id);
  if (!ref ||
      ref->kind != (checkpoint ? experiment::RunArtifactKind::checkpoint
                               : experiment::RunArtifactKind::artifact) ||
      (!checkpoint && ref->role != "model.definition"))
    throw CompatibilityError(
        "Select an original run checkpoint and model.definition reference");
  if (ref->artifact.size_bytes == 0 || ref->artifact.size_bytes > maximum)
    throw ResourceExhausted("ONNX source reference exceeds its byte limit");
  experiment::ArtifactStoreLimits cas_limits;
  cas_limits.max_artifact_bytes = maximum;
  experiment::FileArtifactStore cas(store.artifact_root(), cas_limits);
  auto reader = cas.open(ref->artifact.digest);
  if (reader->size_bytes() != ref->artifact.size_bytes)
    throw SerializationError(
        "ONNX source CAS size differs from the run reference");
  std::ofstream stream(output, std::ios::binary);
  Sha256 hash;
  std::array<std::byte, 65536> block{};
  std::uint64_t count = 0;
  while (const auto got = reader->read(block)) {
    if (got > maximum - count)
      throw ResourceExhausted("ONNX source CAS changed beyond its byte limit");
    hash.update(std::span(block).first(got));
    stream.write(reinterpret_cast<const char*>(block.data()),
                 static_cast<std::streamsize>(got));
    count += got;
  }
  stream.close();
  if (!stream) throw IoError("Cannot stage the original run CAS artifact");
  if (count != ref->artifact.size_bytes ||
      sha256_digest_hex(hash.finish()) != ref->artifact.digest.hex)
    throw SerializationError(
        "ONNX source CAS bytes differ from the run reference");
  return *ref;
}

nn::ModelAdapter Construct(const Json& definition) {
  fields(definition, {"format", "schema_version", "participant_id", "factory"});
  if (definition.at("format") != "jyppx.liuding.onnx_model_definition" ||
      unsigned_number(definition.at("schema_version"), 4294967295ULL) != 1 ||
      !definition.at("participant_id").is_string())
    throw SerializationError("Unsupported native ONNX model definition");
  const auto& factory = definition.at("factory");
  if (!factory.is_object() || !factory.contains("id") ||
      !factory.at("id").is_string())
    throw SerializationError("Use an explicit native model factory");
  if (factory.at("id") == "torch.nn.Linear") {
    fields(factory, {"id", "in_features", "out_features", "bias"});
    const auto in = unsigned_number(factory.at("in_features"), 1000000U);
    const auto out = unsigned_number(factory.at("out_features"), 1000000U);
    if (!in || !out || !factory.at("bias").is_boolean())
      throw SerializationError("Invalid native Linear dimensions or bias");
    if (out > kCheckpointLimit / sizeof(float) / (in + 1U))
      throw ResourceExhausted("Native Linear state exceeds 256 MiB");
    return nn::adapt_tensor_holder(torch::nn::Linear(
        torch::nn::LinearOptions(static_cast<std::int64_t>(in),
                                 static_cast<std::int64_t>(out))
            .bias(factory.at("bias").get<bool>())));
  }
  if (factory.at("id") != "jyppx.liuding.models")
    throw CompatibilityError(
        "Unknown native model factory; use export_module for custom C++ "
        "forward code");
  fields(factory, {"id", "name", "num_classes", "input_channels"});
#if defined(JYPPX_LIUDING_CLI_ONNX_MODELS)
  const auto name = factory.at("name").get<std::string>();
  const auto info = models::model_info(name);
  const auto classes = unsigned_number(factory.at("num_classes"), 1000000U);
  const auto channels = unsigned_number(factory.at("input_channels"), 1024U);
  if (!channels || info.features.empty())
    throw CompatibilityError(
        "Select a registered image model and positive input channels");
  // Conservative pre-construction guard; exact state is bounded again below.
  const auto estimate =
      info.default_parameters * sizeof(float) +
      (classes > 1000 ? classes - 1000 : 0) * 8192U * sizeof(float) +
      (channels > 3 ? channels - 3 : 0) * 8192U * 256U * sizeof(float);
  if (estimate > kCheckpointLimit)
    throw ResourceExhausted(
        "Registered model exceeds the bounded CLI construction profile");
  models::ModelOptions options;
  const auto configure = [&](auto& selected) {
    selected.num_classes = static_cast<std::int64_t>(classes);
    selected.input_channels = static_cast<std::int64_t>(channels);
  };
  configure(options.convnext);
  configure(options.densenet);
  configure(options.efficientnet);
  configure(options.efficientnet_v2);
  configure(options.resnet);
  configure(options.mobile_v2);
  configure(options.mobile);
  configure(options.swin);
  configure(options.vit);
  configure(options.dinov2);
  configure(options.dinov3);
  configure(options.dinov3_convnext);
  return models::create_model(name, options);
#else
  throw CompatibilityError(
      "Native model registry is disabled; enable JYPPX_LIUDING_BUILD_MODELS");
#endif
}

std::string Checked(const Arguments& args, const char* name, const char* digest,
                    std::size_t limit) {
  const ArtifactDigest expected{.hex = args.require(digest)};
  validate_artifact_digest(expected);
  auto content = read_text(args.require(name), limit);
  if (sha256_hex(Bytes(content)) != expected.hex)
    throw SerializationError("ONNX source input SHA-256 mismatch");
  return content;
}

std::vector<at::Tensor> ReadInputs(const Json& bindings,
                                   const interop::SafeTensorsView& view,
                                   const std::vector<std::string>& names,
                                   std::set<std::string>& used,
                                   std::uint64_t& remaining) {
  if (!bindings.is_array() || bindings.size() != names.size())
    throw InvalidArgument(
        "Native export inputs must bind every name exactly once");
  std::map<std::string, std::string> mapping;
  for (const auto& binding : bindings) {
    fields(binding, {"name", "tensor"});
    if (!mapping
             .emplace(binding.at("name").get<std::string>(),
                      binding.at("tensor").get<std::string>())
             .second)
      throw InvalidArgument("Duplicate native export input binding");
  }
  std::vector<at::Tensor> result;
  for (const auto& name : names) {
    if (!mapping.contains(name))
      throw InvalidArgument("Missing native export input name");
    const auto& key = mapping.at(name);
    const auto& entries = view.directory().tensors;
    const auto info =
        std::find_if(entries.begin(), entries.end(),
                     [&](const auto& entry) { return entry.name == key; });
    if (info == entries.end() || info->dtype != interop::SafeTensorDtype::f32 ||
        info->shape.empty() || info->shape.size() > 8)
      throw CompatibilityError(
          "Native source inputs require FP32 tensors of rank 1-8");
    const auto bytes = view.tensor_bytes(key);
    if (bytes.size() > remaining)
      throw ResourceExhausted("Native source cases exceed 64 MiB");
    remaining -= bytes.size();
    std::vector<std::int64_t> shape;
    for (const auto extent : info->shape) {
      if (!extent || extent > (1ULL << 32U))
        throw InvalidArgument("Invalid native source dimension");
      shape.push_back(static_cast<std::int64_t>(extent));
    }
    auto tensor = at::empty(shape, at::kFloat);
    static_assert(std::endian::native == std::endian::little);
    std::memcpy(tensor.mutable_data_ptr(), bytes.data(), bytes.size());
    if (!at::isfinite(tensor).all().item<bool>())
      throw InvalidArgument("Native source inputs must be finite");
    result.push_back(std::move(tensor));
    used.insert(key);
  }
  return result;
}

Json Reference(const experiment::RunArtifactReference& ref) {
  return {{"reference_id", ref.reference_id},
          {"sha256", ref.artifact.digest.hex},
          {"size_bytes", ref.artifact.size_bytes}};
}
}  // namespace

Result onnx_export_command(const Arguments& arguments) {
  arguments.allow({"root", "source-run", "checkpoint-reference",
                   "definition-reference", "inputs", "inputs-sha256", "tensors",
                   "tensors-sha256", "metadata", "metadata-sha256", "provider",
                   "provider-option", "timeout-ms", "output"});
  const auto output = std::filesystem::u8path(arguments.require("output"));
  std::error_code error;
  if (std::filesystem::symlink_status(output, error).type() !=
      std::filesystem::file_type::not_found)
    throw IoError("ONNX export output must be a new directory");
  const auto root = std::filesystem::u8path(arguments.require("root"));
  if (!std::filesystem::is_regular_file(root / "runs.sqlite3"))
    throw IoError("Native export requires an original run store");
  experiment::SQLiteRunStore store(root, {.allow_v1_migration = false});
  const auto run = store.find(arguments.require("source-run"));
  if (!run || !run->lineage.source_digest || !run->lineage.config_digest ||
      !run->lineage.data_manifest_digest)
    throw CompatibilityError(
        "Native export requires complete original run lineage");
  Staging staging;
  const auto checkpoint = CopyReference(
      store, run->run_id, arguments.require("checkpoint-reference"),
      staging.path / "checkpoint", kCheckpointLimit, true);
  const auto definition_ref = CopyReference(
      store, run->run_id, arguments.require("definition-reference"),
      staging.path / "definition", 4U << 20U, false);
  Json proof;
  {
    const auto definition =
        parse_json(read_text(staging.path / "definition", 4U << 20U));
    auto model = Construct(definition);
    nn::ModelCheckpointParticipantOptions state_options;
    state_options.participant_id =
        definition.at("participant_id").get<std::string>();
    state_options.max_tensor_bytes = kCheckpointLimit;
    state_options.max_topology_bytes = 4U << 20U;
    state_options.max_total_bytes = kCheckpointLimit;
    nn::ModelCheckpointParticipant participant(model, state_options);
    // FileStateArchive owns the exact staged snapshot, independent of CAS
    // paths.
    {
      interop::FileStateArchive archive(staging.path / "checkpoint",
                                        {.max_artifact_bytes = kCheckpointLimit,
                                         .max_field_bytes = kCheckpointLimit,
                                         .manifest_limits = {},
                                         .read_only = true});
      participant.load_state(archive.reader_for(state_options.participant_id),
                             RestorePolicy::exact_only);
    }
    model.state().eval();
    auto input_document =
        Checked(arguments, "inputs", "inputs-sha256", 4U << 20U);
    auto input_tensors =
        Checked(arguments, "tensors", "tensors-sha256", kTensorLimit);
    Write(staging.path / "inputs", Bytes(input_document));
    Write(staging.path / "input-tensors", Bytes(input_tensors));
    const auto specification = parse_json(input_document);
    fields(specification,
           {"format", "schema_version", "input_names", "output_names",
            "minimum_batch", "maximum_batch", "dynamic_batch",
            "absolute_tolerance", "relative_tolerance", "cases"});
    if (specification.at("format") != "jyppx.liuding.onnx_export_inputs" ||
        unsigned_number(specification.at("schema_version"), 4294967295ULL) !=
            1 ||
        !specification.at("dynamic_batch").is_boolean())
      throw SerializationError("Unsupported native ONNX export inputs");
    onnx_export::ExportOptions options;
    options.input_names =
        specification.at("input_names").get<std::vector<std::string>>();
    options.output_names =
        specification.at("output_names").get<std::vector<std::string>>();
    options.minimum_batch = static_cast<std::int64_t>(
        unsigned_number(specification.at("minimum_batch"), 65536U));
    options.maximum_batch = static_cast<std::int64_t>(
        unsigned_number(specification.at("maximum_batch"), 65536U));
    options.dynamic_batch = specification.at("dynamic_batch").get<bool>();
    for (const auto& names : {options.input_names, options.output_names})
      if (names.empty() || names.size() > 64 ||
          std::set(names.begin(), names.end()).size() != names.size())
        throw InvalidArgument("Use 1-64 unique native export I/O names");
    const auto& samples = specification.at("cases");
    if (!samples.is_array() || samples.empty() || samples.size() > 64)
      throw ResourceExhausted("Native export requires 1-64 source cases");
    interop::SafeTensorsLimits limits;
    limits.max_file_bytes = kTensorLimit;
    limits.max_payload_bytes = kTensorLimit;
    limits.max_tensor_bytes = kTensorLimit;
    limits.max_tensors = 8192U;
    const interop::SafeTensorsView view(Bytes(input_tensors), limits);
    std::uint64_t remaining = kTensorLimit;
    std::set<std::string> used, ids;
    std::vector<std::vector<at::Tensor>> cases;
    for (const auto& sample : samples) {
      fields(sample, {"id", "inputs"});
      const auto id = sample.at("id").get<std::string>();
      if (id.empty() || !ids.insert(id).second)
        throw InvalidArgument("Use unique nonempty source case IDs");
      cases.push_back(ReadInputs(sample.at("inputs"), view, options.input_names,
                                 used, remaining));
    }
    if (used.size() != view.directory().tensors.size())
      throw InvalidArgument("Source tensor file contains undeclared inputs");
    if (options.minimum_batch <= 0 ||
        options.maximum_batch < options.minimum_batch ||
        options.input_names.size() != 1 || options.output_names.size() != 1)
      throw InvalidArgument(
          "Native CLI factories require one input/output and valid batch "
          "bounds");
    // Validate every declared sample before executing any native forward.
    // A later sample must not bypass the graph's batch or fixed-shape profile.
    const auto example_shape = cases.front().front().sizes().vec();
    for (const auto& sample : cases) {
      const auto shape = sample.front().sizes().vec();
      if (shape.size() != example_shape.size() ||
          shape.front() < options.minimum_batch ||
          shape.front() > options.maximum_batch ||
          (!options.dynamic_batch && shape != example_shape) ||
          !std::equal(shape.begin() + 1, shape.end(),
                      example_shape.begin() + 1))
        throw InvalidArgument(
            "All native source cases must match the declared shape profile");
    }
    if (definition.at("factory").at("id") == "torch.nn.Linear") {
      const auto width =
          definition.at("factory").at("out_features").get<std::uint64_t>();
      std::uint64_t output_bytes = 0;
      for (const auto& sample : cases) {
        const auto& input = sample.front();
        const auto in =
            definition.at("factory").at("in_features").get<std::int64_t>();
        if (input.size(-1) != in)
          throw InvalidArgument("Source Linear input width differs");
        const auto rows = static_cast<std::uint64_t>(input.numel() / in);
        if (width > (remaining - output_bytes) / sizeof(float) / rows)
          throw ResourceExhausted(
              "Native Linear outputs exceed the shared 64 MiB payload budget");
        output_bytes += rows * width * sizeof(float);
      }
    }
    const auto forward = [&](std::span<const at::Tensor> values) {
      nn::Invocation invocation;
      for (const auto& value : values)
        invocation.positional.emplace_back(value);
      const auto result = model.invoke(invocation).value;
      if (!result.isTensor())
        throw CompatibilityError(
            "Registry CLI factory must return one Tensor; use export_module "
            "for custom multi-output code");
      return std::vector<at::Tensor>{result.toTensor()};
    };
    auto exported = onnx_export::export_module(model.state().module(), forward,
                                               cases.front(), options);
    if (!exported.report)
      throw InternalError("Native export did not produce its observations");
    exported.report->source = {
        .run_id = run->run_id,
        .source_digest = run->lineage.source_digest,
        .config_digest = run->lineage.config_digest,
        .data_manifest_digest = run->lineage.data_manifest_digest};
    const auto report = onnx_export::serialize_export_report(*exported.report);
    Json suite{{"format", "jyppx.liuding.onnx_verification_cases"},
               {"schema_version", 1},
               {"reference", "native-checkpoint-model"},
               {"absolute_tolerance", specification.at("absolute_tolerance")},
               {"relative_tolerance", specification.at("relative_tolerance")},
               {"cases", Json::array()}};
    std::vector<at::Tensor> storage;
    std::vector<interop::SafeTensorSource> tensors;
    const auto add_tensor = [&](const at::Tensor& tensor,
                                const std::string& key,
                                const std::string& name) {
      if (tensor.scalar_type() != at::kFloat ||
          tensor.device().type() != at::kCPU || tensor.dim() < 1 ||
          tensor.dim() > 8)
        throw CompatibilityError(
            "Native source outputs require CPU FP32 tensors of rank 1-8");
      if (tensor.nbytes() > remaining)
        throw ResourceExhausted(
            "Native source outputs exceed the shared 64 MiB payload budget");
      remaining -= tensor.nbytes();
      storage.push_back(tensor.detach().contiguous());
      const auto& stable = storage.back();
      std::vector<std::uint64_t> shape;
      for (auto extent : stable.sizes())
        shape.push_back(static_cast<std::uint64_t>(extent));
      tensors.push_back(
          {.name = key,
           .dtype = interop::SafeTensorDtype::f32,
           .shape = std::move(shape),
           .data = {static_cast<const std::byte*>(stable.const_data_ptr()),
                    stable.nbytes()}});
      return Json{{"name", name}, {"tensor", key}};
    };
    // Input ownership was already charged when materializing the source cases.
    const torch::NoGradGuard guard;
    for (std::size_t i = 0; i < cases.size(); ++i) {
      Json row{{"id", samples[i].at("id")},
               {"inputs", Json::array()},
               {"expected", Json::array()}};
      for (std::size_t j = 0; j < cases[i].size(); ++j) {
        remaining += cases[i][j].nbytes();
        row["inputs"].push_back(add_tensor(
            cases[i][j], "case" + std::to_string(i) + ".in" + std::to_string(j),
            exported.input_names[j]));
      }
      const auto results = forward(cases[i]);
      if (results.size() != exported.output_names.size())
        throw CompatibilityError("Native source output count changed");
      for (std::size_t j = 0; j < results.size(); ++j)
        row["expected"].push_back(add_tensor(
            results[j], "case" + std::to_string(i) + ".out" + std::to_string(j),
            exported.output_names[j]));
      suite["cases"].push_back(std::move(row));
    }
    Write(staging.path / "model.onnx", exported.onnx);
    Write(staging.path / "report.json", Bytes(report));
    const auto case_text = suite.dump();
    Write(staging.path / "cases.json", Bytes(case_text));
    const auto encoded_tensors =
        interop::serialize_safetensors(tensors, {}, limits);
    Write(staging.path / "cases.safetensors", encoded_tensors);
    proof = {{"format", "jyppx.liuding.onnx_source_reconstruction"},
             {"schema_version", 1},
             {"source", Json::parse(report).at("source")},
             {"checkpoint", Reference(checkpoint)},
             {"definition", Reference(definition_ref)},
             {"participant_id", state_options.participant_id},
             {"exact_model_state_restored", true},
             {"checkpoint_run_association_verified", true},
             {"definition_run_association_verified", true},
             {"original_code_authenticated", false},
             {"original_config_semantics_verified", false},
             {"source_authenticated", false},
             {"evaluation_mode", true},
             {"input_spec_sha256", sha256_hex(Bytes(input_document))},
             {"input_tensors_sha256", sha256_hex(Bytes(input_tensors))},
             {"export_report_sha256", sha256_hex(Bytes(report))},
             {"cases_sha256", sha256_hex(Bytes(case_text))},
             {"tensors_sha256", sha256_hex(encoded_tensors)},
             {"source_case_count", cases.size()}};
  }
  // Release the source model, checkpoint snapshot and reference tensors before
  // allocating the independent runtime Session and bundle verification state.
  std::vector<std::string> words{"model", "onnx", "pack"};
  const auto add = [&](std::string name, std::string value) {
    words.push_back("--" + name);
    words.push_back(std::move(value));
  };
  const auto path_text = [](const std::filesystem::path& path) {
    const auto value = path.generic_u8string();
    return std::string(reinterpret_cast<const char*>(value.data()),
                       value.size());
  };
  add("path", path_text(staging.path / "model.onnx"));
  add("report", path_text(staging.path / "report.json"));
  add("report-sha256", proof.at("export_report_sha256").get<std::string>());
  add("cases", path_text(staging.path / "cases.json"));
  add("cases-sha256", proof.at("cases_sha256").get<std::string>());
  add("tensors", path_text(staging.path / "cases.safetensors"));
  add("tensors-sha256", proof.at("tensors_sha256").get<std::string>());
  for (const auto key : {"metadata", "metadata-sha256", "provider", "output"})
    add(key, arguments.require(key));
  if (const auto timeout = arguments.optional("timeout-ms"))
    add("timeout-ms", *timeout);
  for (const auto& option : arguments.repeated("provider-option"))
    add("provider-option", option);
  std::vector<std::string_view> views(words.begin(), words.end());
  auto result = onnx_pack_reconstructed(
      Arguments(views), proof, staging.path / "definition",
      staging.path / "inputs", staging.path / "input-tensors");
  result.data["operation"] = "model.onnx.export";
  result.data["source_reconstruction"] = std::move(proof);
  return result;
}
}  // namespace jyppx::liuding::cli
#else
namespace jyppx::liuding::cli {
Result onnx_export_command(const Arguments&) {
  throw CompatibilityError(
      "Native ONNX export delivery components are disabled",
      "Enable JYPPX_LIUDING_BUILD_ONNX_EXPORT and "
      "JYPPX_LIUDING_BUILD_ONNX_RUNTIME.");
}
}  // namespace jyppx::liuding::cli
#endif
