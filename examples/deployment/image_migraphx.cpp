// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
/** @file image_migraphx.cpp @brief Fixed-shape, signed native GPU delivery of
 * an independently pinned CIFAR-10 ONNX export and its checkpoint references.
 */
#include "cifar10_common.hpp"

#include <jyppx/liuding/core/runtime.hpp>
#include <jyppx/liuding/deployment/migraphx_serving/backend.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <ATen/Parallel.h>
#include <bit>
#include <chrono>
#include <cmath>
#include <numeric>

namespace {
using namespace jyppx_example::cifar10;
namespace mgx = forge::migraphx_runtime;
namespace serving = forge::serving;
constexpr std::array<std::int64_t, 6> kBatches{1, 2, 4, 8, 16, 128};
constexpr std::string_view kTensorSchema =
    "jyppx.example.migraphx_cifar10_tensor";

std::size_t Batch(const fs::path& text) {
  Require(text == "128" || text == "16",
          "Use fixed batch 128 or 16 explicitly");
  return text == "128" ? 128 : 16;
}
mgx::SessionOptions Options(std::size_t batch) {
  mgx::SessionOptions options;
  options.device_id = 0;
  options.input_name = "image";
  options.input_shape = {batch, 3, 32, 32};
  options.max_batch_size = 128;
  options.max_model_bytes = 64 * kMiB;
  options.max_input_bytes = 8 * kMiB;
  options.max_output_bytes = kMiB;
  return options;
}
Json Schema(std::size_t batch, bool input) {
  return {{"format", kTensorSchema},
          {"schema_version", 1},
          {"dtype", "float32"},
          {"io_device", "cpu"},
          {"name", input ? "image" : "logits"},
          {"shape",
           input ? Json::array({batch, 3, 32, 32}) : Json::array({batch, 10})}};
}
void Environment(const mgx::EnvironmentLock& lock, std::size_t batch) {
  Require(lock.device_id == 0 && lock.input_name == "image" &&
              lock.input_shape == Options(batch).input_shape &&
              lock.output_shape == std::vector<std::size_t>{batch, 10},
          "Environment must bind HIP device 0 and exact CIFAR specialization");
}
serving::WarmupCase Known(const interop::ModelBundleSnapshot& snapshot,
                          std::size_t batch) {
  const auto suite = Json::parse(Text(snapshot.files.at("cases")));
  Require(suite.at("format") == "jyppx.liuding.onnx_verification_cases" &&
              suite.at("schema_version") == 1 &&
              suite.at("reference") == "native-checkpoint-model" &&
              suite.at("absolute_tolerance") == 1e-5 &&
              suite.at("relative_tolerance") == 1e-4 &&
              suite.at("cases").size() == kBatches.size(),
          "Require six original native checkpoint references and tolerances");
  interop::SafeTensorsLimits limits;
  limits.max_file_bytes = limits.max_payload_bytes = limits.max_tensor_bytes =
      64 * kMiB;
  const interop::SafeTensorsView tensors(snapshot.files.at("tensors"), limits);
  serving::WarmupCase selected;
  for (std::size_t i = 0; i < kBatches.size(); ++i) {
    const auto n = kBatches[i];
    const auto& item = suite.at("cases").at(i);
    Require(item.at("id") == "batch" + std::to_string(n) &&
                item.at("inputs").size() == 1 &&
                item.at("expected").size() == 1 &&
                item.at("inputs").at(0).at("name") == "image" &&
                item.at("expected").at(0).at("name") == "logits",
            "Original case identity/names differ");
    auto input = Tensor(
        tensors, item.at("inputs").at(0).at("tensor").get<std::string>());
    auto expected = Tensor(
        tensors, item.at("expected").at(0).at("tensor").get<std::string>());
    Require(input.sizes().vec() == std::vector<std::int64_t>{n, 3, 32, 32} &&
                expected.sizes().vec() == std::vector<std::int64_t>{n, 10},
            "Original case tensor signature differs");
    if (static_cast<std::size_t>(n) == batch)
      selected = {
          .input = {.positional = {}, .keyword = {{"image", std::move(input)}}},
          .expected = std::move(expected),
          .relative_tolerance = 1e-4,
          .absolute_tolerance = 1e-5};
  }
  return selected;
}
void Compare(const at::Tensor& actual, const at::Tensor& expected) {
  Require(actual.scalar_type() == at::kFloat && actual.device().is_cpu() &&
              actual.is_contiguous() && actual.sizes() == expected.sizes(),
          "FP32 owned output signature differs");
  for (std::int64_t i = 0; i < actual.numel(); ++i) {
    const auto x = static_cast<double>(actual.const_data_ptr<float>()[i]);
    const auto y = static_cast<double>(expected.const_data_ptr<float>()[i]);
    Require(std::isfinite(x) && std::isfinite(y) &&
                std::abs(x - y) <= 1e-5 + 1e-4 * std::abs(y),
            "Original checkpoint numerical tolerance exceeded");
  }
}
interop::ModelBundleSnapshot Snapshot(
    const fs::path& directory, std::string_view expected,
    const interop::ModelBundleVerifyOptions& verify) {
  const auto metadata =
      interop::verify_model_bundle_directory(directory, verify);
  Require(metadata.manifest_digest.hex == expected,
          "Independently pinned manifest differs");
  std::vector<std::string_view> ids;
  for (const auto& file : metadata.manifest.files) ids.push_back(file.id);
  return interop::read_model_bundle_snapshot(
      directory, {.hex = std::string(expected)}, ids, verify, 96 * kMiB);
}
std::string ModelId(const interop::ModelBundleManifest& m) {
  Require(m.models.size() == 1 && m.models.front().id == "primary" &&
              m.models.front().format == interop::BundleModelFormat::onnx &&
              m.models.front().file_ids.size() == 1,
          "Require one primary self-contained ONNX graph");
  return m.models.front().file_ids.front();
}
void Pack(std::span<const fs::path> args) {
  Require(args.size() == 4,
          "pack SOURCE_BUNDLE SOURCE_MANIFEST_SHA BATCH OUTPUT");
  const auto batch = Batch(args[2]);
  const auto source = Snapshot(args[0], args[1].string(), Limits());
  const auto& original = source.verification.manifest;
  const auto graph_id = ModelId(original);
  const auto& graph = source.files.at(graph_id);
  Require(
      Json::parse(Text(source.files.at(original.preprocessing.file_id))) ==
              Preprocessing() &&
          Json::parse(Text(source.files.at(original.postprocessing.file_id))) ==
              Postprocessing(),
      "Source image codec differs from this example");
  Json labels = Json::array();
  for (const auto name : kClasses) labels.push_back(name);
  Require(original.labels &&
              Json::parse(Text(source.files.at(original.labels->file_id))) ==
                  Json{{"schema_version", 1}, {"classes", labels}},
          "Source class order differs");
  const auto known = Known(source, batch);
  // Independent references are borrowed from the pinned source, never made by
  // MIGraphX.
  mgx::EnvironmentLock environment;
  mgx::RunResult actual;
  {
    const mgx::Session session(graph, Digest(graph), Options(batch));
    environment = mgx::observe_environment(session);
    Environment(environment, batch);
    const auto& input = known.input.keyword.at("image").toTensor();
    actual = session.run(std::span(input.const_data_ptr<float>(),
                                   static_cast<std::size_t>(input.numel())));
  }
  auto observed = at::empty({static_cast<std::int64_t>(batch), 10}, at::kFloat);
  Require(actual.shape == std::vector<std::size_t>{batch, 10} &&
              actual.values.size() == batch * 10,
          "Pack warmup signature differs");
  std::memcpy(observed.data_ptr<float>(), actual.values.data(),
              observed.nbytes());
  Compare(observed, known.expected.toTensor());
  NewDirectory(args[3]);
  const auto staging = args[3] / "sources";
  Require(fs::create_directory(staging), "Fresh staging directory required");
  auto manifest = original;
  manifest.bundle_id =
      "cifar10-resnet18-migraphx-fixed-" + std::to_string(batch);
  manifest.build = forge::current_build_fingerprint();
  manifest.signatures.clear();
  const auto descriptor = serving::migraphx_runtime_descriptor(environment);
  manifest.runtime = {.backend_id = descriptor.backend_id,
                      .backend_version = descriptor.backend_version,
                      .environment_lock = descriptor.environment_lock,
                      .dependencies = descriptor.dependencies,
                      .devices = {"rocm"},
                      .dtypes = {"float32"},
                      .os_arch = {environment.os_arch},
                      .dynamic_shapes = false};
  auto payloads = source.files;
  const auto replace = [&](std::string_view id,
                           std::span<const std::byte> bytes) {
    payloads.at(std::string(id)) = {bytes.begin(), bytes.end()};
    auto found = std::find_if(manifest.files.begin(), manifest.files.end(),
                              [&](const auto& file) { return file.id == id; });
    Require(found != manifest.files.end(), "Declared source file required");
    found->artifact.digest = Digest(bytes);
    found->artifact.size_bytes = bytes.size();
    found->artifact.producer = "jyppx.example.migraphx-cifar10";
  };
  const auto add = [&](std::string id, std::span<const std::byte> bytes) {
    Require(!payloads.contains(id), "Source already uses reserved evidence id");
    manifest.files.push_back(
        {.id = id,
         .relative_path = "evidence/" + id + ".data",
         .role = interop::BundleFileRole::evidence,
         .artifact = {.digest = Digest(bytes),
                      .size_bytes = bytes.size(),
                      .logical_name = id,
                      .media_type = "application/octet-stream",
                      .producer = "jyppx.example.migraphx-cifar10"}});
    manifest.evidence.push_back({.schema_id = "jyppx.example." + id,
                                 .schema_version = 1,
                                 .file_id = id});
    payloads.emplace(std::move(id),
                     std::vector<std::byte>(bytes.begin(), bytes.end()));
  };
  add("source-manifest", source.manifest_bytes);
  add("source-provenance", source.files.at(original.provenance.file_id));
  add("source-sbom", source.files.at(original.sbom.file_id));
  add("source-evaluation", source.files.at(original.evaluation.file_id));
  replace(manifest.input_schema.file_id, Bytes(Schema(batch, true).dump()));
  replace(manifest.output_schema.file_id, Bytes(Schema(batch, false).dump()));
  manifest.input_schema.schema_id = manifest.output_schema.schema_id =
      kTensorSchema;
  const auto lock = mgx::serialize_environment_lock(environment);
  // Exactly one new public lock; original export reports describe only the
  // source.
  std::size_t locks = 0;
  for (auto& ref : manifest.evidence) {
    if (ref.file_id == "runtime-lock") {
      replace(ref.file_id, Bytes(lock));
      ref.schema_id = std::string(mgx::kEnvironmentLockSchemaId);
      ref.schema_version = 1;
      ++locks;
    } else {
      Require(ref.schema_id != mgx::kEnvironmentLockSchemaId,
              "Source contains an unexpected native runtime lock");
    }
  }
  Require(locks == 1, "Require one source runtime-lock evidence file");
  const Json provenance{
      {"format", "jyppx.example.migraphx_cifar10_provenance"},
      {"schema_version", 1},
      {"source_manifest_sha256", args[1].string()},
      {"source_bundle_id", original.bundle_id},
      {"model_sha256", Digest(graph).hex},
      {"tensors_sha256", Digest(source.files.at("tensors")).hex},
      {"source_authenticated", false},
      {"reexported", false},
      {"source_reports_scope", "original pinned ONNX export only"}};
  replace(manifest.provenance.file_id, Bytes(provenance.dump()));
  manifest.provenance.schema_id = "jyppx.example.migraphx_cifar10_provenance";
  replace(manifest.evaluation.file_id,
          Bytes(Json{{"format", "jyppx.example.migraphx_known_vector"},
                     {"schema_version", 1},
                     {"batch", batch},
                     {"reference", "original native checkpoint"},
                     {"relative_tolerance", 1e-4},
                     {"absolute_tolerance", 1e-5},
                     {"numerical", "passed"},
                     {"full_dataset_evaluated", false}}
                    .dump()));
  manifest.evaluation.schema_id = "jyppx.example.migraphx_known_vector";
  const Json sbom{{"complete_transitive_sbom", false},
                  {"scope", "observed native runtime subset"},
                  {"migraphx_headers", environment.migraphx_header_version},
                  {"hip_headers", environment.hip_header_version},
                  {"libtorch", environment.libtorch_version},
                  {"liuding_forge", environment.adapter_version}};
  replace(manifest.sbom.file_id, Bytes(sbom.dump()));
  manifest.sbom.schema_id = "jyppx.example.migraphx_dependency_subset";
  std::vector<interop::BundlePublishSource> sources;
  for (const auto& [id, bytes] : payloads) {
    const auto path = staging / (id + ".data");
    Write(path, bytes);
    sources.push_back({.file_id = id, .source_path = path});
  }
  const auto published = interop::publish_model_bundle(
      args[3] / "bundle", manifest, sources, Limits());
  std::ofstream trace(args[3] / "dispatch.txt");
  for (const auto& name : actual.instructions) trace << name << '\n';
  trace.close();
  Require(static_cast<bool>(trace), "Cannot save actual pack dispatch");
  Write(args[3] / "warmup.f32", std::as_bytes(std::span(actual.values)));
  WriteJson(args[3] / "pack.json",
            {{"batch", batch},
             {"device_id", 0},
             {"source_manifest_sha256", args[1].string()},
             {"model_sha256", Digest(graph).hex},
             {"tensors_sha256", Digest(source.files.at("tensors")).hex},
             {"manifest_sha256", published.manifest_digest.hex},
             {"numerical", "passed"},
             {"relative_tolerance", 1e-4},
             {"absolute_tolerance", 1e-5},
             {"source_authenticated", false},
             {"training_updates", 0}});
  std::cout << "Packed fixed batch " << batch
            << "; manifest SHA-256=" << published.manifest_digest.hex << '\n';
}

template <class Bundle>
serving::Pipeline Codec(const Bundle& bundle,
                        const mgx::EnvironmentLock& environment,
                        std::string_view expected_manifest, std::size_t batch) {
  Require(bundle.verification.manifest_digest.hex == expected_manifest,
          "Trusted snapshot differs from independent manifest digest");
  Environment(environment, batch);
  const auto bind = [&](const interop::BundleDescriptorRef& ref,
                        std::string_view schema, const Json& content) {
    const auto& bytes = bundle.files.at(ref.file_id);
    Require(ref.schema_id == schema && ref.schema_version == 1 &&
                Json::parse(Text(bytes)) == content,
            "Owner descriptor content differs");
    return serving::DescriptorBinding{.schema_id = ref.schema_id,
                                      .schema_version = ref.schema_version,
                                      .digest = Digest(bytes)};
  };
  const auto& m = bundle.verification.manifest;
  return serving::Pipeline{
      .version = "cifar10-rgb32-migraphx-fixed-v1",
      .input = bind(m.input_schema, kTensorSchema, Schema(batch, true)),
      .output = bind(m.output_schema, kTensorSchema, Schema(batch, false)),
      .preprocessing =
          bind(m.preprocessing, "jyppx.example.cifar10_input", Preprocessing()),
      .postprocessing = bind(m.postprocessing, "jyppx.example.cifar10_logits",
                             Postprocessing()),
      .request_schema =
          serving::migraphx_invocation_schema("image", environment.input_shape),
      .response_schema =
          serving::migraphx_result_schema(environment.output_shape),
      .preprocess = {},
      .postprocess = {}};
}
serving::MIGraphXPipelineFactory Owner(std::string expected_manifest,
                                       std::size_t batch) {
  return [expected_manifest = std::move(expected_manifest), batch](
             const serving::VerifiedBundle& bundle,
             const mgx::EnvironmentLock& environment) {
    return Codec(bundle, environment, expected_manifest, batch);
  };
}
struct Loaded {
  std::shared_ptr<serving::Predictor> predictor;
  serving::WarmupCase known;
  std::string lock, model, tensors, source;
};
struct Prepared {
  Loaded data;
  interop::ModelBundleVerifyOptions verify;
};
Prepared Inspect(std::span<const fs::path> args, std::size_t batch) {
  Require(args.size() == 4, "BUNDLE MANIFEST_SHA POLICY POLICY_SHA required");
  const auto policy =
      signing::Ed25519TrustPolicy::load(args[2], {.hex = args[3].string()});
  const auto verify = signing::trusted_ed25519_options(policy, Limits().limits);
  const auto snapshot = Snapshot(args[0], args[1].string(), verify);
  Require(snapshot.verification.signature_status ==
              interop::BundleSignatureStatus::trusted,
          "Trusted Ed25519 signature required");
  const auto& m = snapshot.verification.manifest;
  const auto graph = ModelId(m);
  const auto lock = Text(snapshot.files.at("runtime-lock"));
  const auto environment = mgx::parse_environment_lock(lock);
  Environment(environment, batch);
  static_cast<void>(Codec(snapshot, environment, args[1].string(), batch));
  auto known = Known(snapshot, batch);
  return {.data = {.predictor = {},
                   .known = std::move(known),
                   .lock = lock,
                   .model = Digest(snapshot.files.at(graph)).hex,
                   .tensors = Digest(snapshot.files.at("tensors")).hex,
                   .source = Digest(snapshot.files.at("source-manifest")).hex},
          .verify = verify};
}
Loaded Load(std::span<const fs::path> args, std::size_t batch,
            Prepared prepared) {
  const auto environment = mgx::parse_environment_lock(prepared.data.lock);
  const serving::MIGraphXRuntimeBackend backend(
      serving::migraphx_runtime_descriptor(environment), Options(batch),
      Owner(args[1].string(), batch));
  serving::LoadOptions load;
  load.max_snapshot_bytes = 96 * kMiB;
  load.request_limits.max_tensor_bytes = 8 * kMiB;
  load.output_limits.max_tensor_bytes = kMiB;
  const std::array warmup{prepared.data.known};
  prepared.data.predictor =
      backend.load(args[0], prepared.verify, warmup, load);
  return std::move(prepared.data);
}
serving::PredictRequest Request(const Loaded& loaded, at::Tensor input,
                                std::string id) {
  serving::PredictRequest request;
  request.input = {.positional = {}, .keyword = {{"image", std::move(input)}}};
  request.context.request_id = std::move(id);
  request.context.model_version = loaded.predictor->info().model_version;
  return request;
}
void Evaluate(std::span<const fs::path> args) {
  Require(args.size() == 10,
          "evaluate BUNDLE128 MANIFEST128 POLICY128 POLICY128_SHA BUNDLE16 "
          "MANIFEST16 POLICY16 POLICY16_SHA DATA OUTPUT");
  // Reject untrusted/mixed specializations and owner descriptors before JIT.
  std::array prepared{Inspect(args.first(4), 128),
                      Inspect(args.subspan(4, 4), 16)};
  Require(prepared[0].data.model == prepared[1].data.model &&
              prepared[0].data.tensors == prepared[1].data.tensors &&
              prepared[0].data.source == prepared[1].data.source,
          "Both fixed bundles must share the pinned graph/reference/source");
  Images images(args[8]);
  // Compile the two backend sessions, each with its own original known vector.
  std::array loaded{Load(args.first(4), 128, std::move(prepared[0])),
                    Load(args.subspan(4, 4), 16, std::move(prepared[1]))};
  NewDirectory(args[9]);
  std::array<at::Tensor, 2> retained;
  for (std::size_t i = 0; i < loaded.size(); ++i) {
    const auto batch = i ? 16 : 128;
    auto response = loaded[i].predictor->predict(
        Request(loaded[i], loaded[i].known.input.keyword.at("image").toTensor(),
                "known"));
    retained[i] = response.output.value.toTensor();
    Compare(retained[i], loaded[i].known.expected.toTensor());
    Write(args[9] / ("known-" + std::to_string(batch) + ".f32"),
          std::as_bytes(
              std::span(retained[i].const_data_ptr<float>(),
                        static_cast<std::size_t>(retained[i].numel()))));
    Write(args[9] / ("environment-" + std::to_string(batch) + ".json"),
          Bytes(loaded[i].lock));
  }
  std::ofstream logits_file(args[9] / "logits.f32", std::ios::binary);
  std::ofstream labels_file(args[9] / "labels.i64", std::ios::binary);
  std::ofstream inventory(args[9] / "samples.tsv");
  inventory << "index\tlabel\tpath\n";
  std::array<std::array<std::uint64_t, 10>, 10> confusion{};
  std::uint64_t correct = 0, calls = 0;
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t offset = 0; offset < kSamples; offset += 128) {
    std::vector<std::size_t> ids(std::min<std::size_t>(128, kSamples - offset));
    std::iota(ids.begin(), ids.end(), offset);
    const auto batch = images.batch(ids);
    const auto& instance = loaded[ids.size() == 128 ? 0 : 1];
    const auto response = instance.predictor->predict(
        Request(instance, batch.images, "images-" + std::to_string(offset)));
    const auto logits = response.output.value.toTensor();
    Require(logits.scalar_type() == at::kFloat && logits.device().is_cpu() &&
                logits.is_contiguous() &&
                logits.sizes().vec() ==
                    std::vector<std::int64_t>{
                        static_cast<std::int64_t>(ids.size()), 10} &&
                at::isfinite(logits).all().item<bool>() &&
                batch.labels.scalar_type() == at::kLong &&
                batch.labels.numel() == static_cast<std::int64_t>(ids.size()),
            "Exact evaluation tensor boundary required");
    Append(logits_file, logits);
    Append(labels_file, batch.labels.contiguous());
    for (std::size_t j = 0; j < ids.size(); ++j) {
      const auto label = batch.labels.const_data_ptr<std::int64_t>()[j];
      Require(label >= 0 && label < 10, "Label outside class map");
      const auto* row = logits.const_data_ptr<float>() + j * 10;
      const auto prediction = std::max_element(row, row + 10) - row;
      ++confusion[static_cast<std::size_t>(label)]
                 [static_cast<std::size_t>(prediction)];
      correct += label == prediction;
      inventory << ids[j] << '\t' << label << '\t' << batch.items[j].image_path
                << '\n';
    }
    ++calls;
    if (offset % 1024 == 0)
      std::cout << "evaluated=" << offset + ids.size() << std::endl;
  }
  logits_file.close();
  labels_file.close();
  inventory.close();
  Require(logits_file && labels_file && inventory,
          "Cannot finish evaluation payloads");
  for (std::size_t i = 0; i < loaded.size(); ++i) {
    Require(loaded[i].predictor->close(), "Serving did not drain");
    loaded[i].predictor.reset();
    Compare(retained[i], loaded[i].known.expected.toTensor());
  }
  const auto accuracy = static_cast<double>(correct) / kSamples;
  WriteJson(
      args[9] / "evaluation.json",
      {{"format", "jyppx.example.cifar10_migraphx_serving_evaluation"},
       {"schema_version", 1},
       {"manifests_sha256", {args[1].string(), args[5].string()}},
       {"policies_sha256", {args[3].string(), args[7].string()}},
       {"signature_status", "trusted"},
       {"provider", "migraphx.gpu"},
       {"device_id", 0},
       {"io_device", "cpu"},
       {"allow_cpu_fallback", false},
       {"model_sha256", loaded[0].model},
       {"tensors_sha256", loaded[0].tensors},
       {"source_manifest_sha256", loaded[0].source},
       {"samples", kSamples},
       {"correct", correct},
       {"accuracy", accuracy},
       {"threshold", 0.65},
       {"passed", accuracy >= 0.65},
       {"confusion_matrix", confusion},
       {"fixed_batches", {128, 16}},
       {"predict_calls", calls},
       {"response_ownership_after_close", true},
       {"training_updates", 0},
       {"seconds", std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - started)
                       .count()},
       {"logits_sha256", Digest(Read(args[9] / "logits.f32", kMiB)).hex},
       {"labels_sha256", Digest(Read(args[9] / "labels.i64", kMiB)).hex}});
  Require(accuracy >= 0.65, "Full validation accuracy is below 0.65");
  std::cout << "samples=" << kSamples << ", accuracy=" << accuracy << '\n';
}
}  // namespace
#if defined(_WIN32)
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  const std::vector<fs::path> args(argv + 1, argv + argc);
  if (args.size() == 1 && args[0] == "--help") {
    std::cout
        << "Commands:\n  pack SOURCE_BUNDLE SOURCE_MANIFEST_SHA BATCH OUTPUT\n"
           "  sign UNSIGNED EXPECTED_MANIFEST_SHA OUTPUT\n"
           "  evaluate BUNDLE128 MANIFEST128 POLICY128 POLICY128_SHA "
           "BUNDLE16 MANIFEST16 POLICY16 POLICY16_SHA DATA OUTPUT\n";
    return 0;
  }
  try {
    forge::initialize_runtime();
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    const torch::NoGradGuard guard;
    Require(std::endian::native == std::endian::little,
            "Little-endian output required");
    Require(!args.empty(), "Use --help for command syntax");
    const auto tail = std::span(args).subspan(1);
    if (args[0] == "pack")
      Pack(tail);
    else if (args[0] == "sign")
      Sign(tail);
    else if (args[0] == "evaluate")
      Evaluate(tail);
    else
      throw std::invalid_argument("Unknown command; use --help");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Image MIGraphX example failed: " << error.what() << '\n';
    return 1;
  }
}
