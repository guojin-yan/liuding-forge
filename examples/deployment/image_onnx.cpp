// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
/** @file image_onnx.cpp @brief Prepares CIFAR-10 inputs, signs a native ONNX
 * delivery and evaluates its trusted FP32 host-I/O Serving instance. */

#include "cifar10_common.hpp"

#include <jyppx/liuding/core/runtime.hpp>
#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/deployment/bundle_signing/provider.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/environment.hpp>
#include <jyppx/liuding/deployment/onnx_serving/backend.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>

#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace jyppx_example::cifar10;
namespace forge = jyppx::liuding;
namespace image = forge::data::image;
namespace interop = forge::interop;
namespace ort = forge::onnx_runtime;
namespace serving = forge::serving;
namespace signing = forge::bundle_signing;
namespace fs = std::filesystem;
using Json = nlohmann::json;
void Prepare(std::span<const fs::path> args) {
  Require(args.size() == 6,
          "prepare DATA CHECKPOINT CHECKPOINT_SHA RESULT RESULT_SHA OUTPUT");
  // A complete pinned checkpoint is inspected, then released before decoding.
  const auto checkpoint_size = [&] {
    const auto bytes = Pinned(args[1], args[2].string(), 256 * kMiB);
    return bytes.size();
  }();
  const auto result_bytes = Pinned(args[3], args[4].string(), 4 * kMiB);
  const auto result = Json::parse(Text(result_bytes));
  Images images(args[0]);
  NewDirectory(args[5]);
  Json inputs{{"format", "jyppx.liuding.onnx_export_inputs"},
              {"schema_version", 1},
              {"input_names", {"image"}},
              {"output_names", {"logits"}},
              {"minimum_batch", 1},
              {"maximum_batch", 128},
              {"dynamic_batch", true},
              {"absolute_tolerance", 1e-5},
              {"relative_tolerance", 1e-4},
              {"cases", Json::array()}};
  std::vector<at::Tensor> storage;
  std::vector<interop::SafeTensorSource> tensors;
  Json inventory = Json::array();
  for (const std::size_t n : {1, 2, 4, 8, 16, 128}) {
    std::vector<std::size_t> ids(n);
    for (std::size_t j = 0; j < n; ++j) ids[j] = j * 997 % kSamples;
    const auto batch = images.batch(ids);
    storage.push_back(batch.images.contiguous());
    const auto& tensor = storage.back();
    const auto key = "batch" + std::to_string(n);
    tensors.push_back({.name = key,
                       .dtype = interop::SafeTensorDtype::f32,
                       .shape = {n, 3, 32, 32},
                       .data = std::as_bytes(std::span(
                           tensor.const_data_ptr<float>(), tensor.numel()))});
    inputs["cases"].push_back(
        {{"id", key}, {"inputs", {{{"name", "image"}, {"tensor", key}}}}});
    inventory.push_back({{"case", key}, {"indices", ids}});
  }
  interop::SafeTensorsLimits tensor_limits;
  tensor_limits.max_file_bytes = tensor_limits.max_payload_bytes =
      tensor_limits.max_tensor_bytes = 64 * kMiB;
  Write(args[5] / "inputs.safetensors",
        interop::serialize_safetensors(tensors, {}, tensor_limits));
  WriteJson(args[5] / "inputs.json", inputs);
  WriteJson(args[5] / "input-indices.json", inventory);
  const auto build = forge::current_build_fingerprint();
  Json labels = Json::array();
  for (auto name : kClasses) labels.push_back(name);
  Json delivery;
  delivery["format"] = "jyppx.liuding.onnx_delivery_metadata";
  delivery["schema_version"] = 1;
  delivery["bundle_id"] = "cifar10-resnet18-cpu-onnx";
  delivery["checkpoint"] = {{"sha256", args[2].string()},
                            {"size_bytes", checkpoint_size}};
  delivery["preprocessing"] =
      Descriptor("jyppx.example.cifar10_input", Preprocessing());
  delivery["postprocessing"] =
      Descriptor("jyppx.example.cifar10_logits", Postprocessing());
  delivery["labels"] = Descriptor("jyppx.example.cifar10_labels",
                                  {{"schema_version", 1}, {"classes", labels}});
  delivery["evaluation"] =
      Descriptor("jyppx.example.training_result",
                 {{"source_sha256", args[4].string()}, {"result", result}});
  delivery["sbom"] = Descriptor("jyppx.example.dependency_subset",
                                {{"complete_transitive_sbom", false},
                                 {"liuding_forge", build.jyppx_version},
                                 {"libtorch", build.libtorch_version}});
  WriteJson(args[5] / "delivery.json", delivery);
  std::cout << "Prepared six real-image cases; no training updates.\n";
}

serving::OnnxPipelineFactory Owner(std::string expected_manifest) {
  return [expected_manifest = std::move(expected_manifest)](
             const serving::VerifiedBundle& bundle,
             std::span<const ort::TensorSignature> inputs,
             std::span<const ort::TensorSignature> outputs) {
    Require(bundle.verification.manifest_digest.hex == expected_manifest,
            "Serving snapshot differs from the independently pinned manifest");
    const auto signature = [](auto values, std::string_view name,
                              std::vector<std::int64_t> shape) {
      Require(values.size() == 1 && values[0].name == name &&
                  values[0].shape == shape && values[0].batch_symbol == "batch",
              "Graph signature differs from the application codec");
    };
    signature(inputs, "image", {-1, 3, 32, 32});
    signature(outputs, "logits", {-1, 10});
    const auto bind = [&](const interop::BundleDescriptorRef& ref,
                          std::string_view schema, const Json& content) {
      const auto& bytes = bundle.files.at(ref.file_id);
      Require(ref.schema_id == schema && ref.schema_version == 1 &&
                  Json::parse(Text(bytes)) == content,
              "Verified descriptor content differs from the owner codec");
      return serving::DescriptorBinding{.schema_id = ref.schema_id,
                                        .schema_version = ref.schema_version,
                                        .digest = Digest(bytes)};
    };
    const auto tensor_schema = [](std::string_view name,
                                  std::vector<std::int64_t> shape) {
      return Json{{"format", "jyppx.liuding.onnx_tensor_schema"},
                  {"schema_version", 1},
                  {"minimum_batch", 1},
                  {"maximum_batch", 128},
                  {"tensors",
                   {{{"name", name},
                     {"shape", shape},
                     {"batch_symbol", "batch"},
                     {"dtype", "float32"}}}}};
    };
    const auto& manifest = bundle.verification.manifest;
    return serving::Pipeline{
        .version = "cifar10-rgb32-nchw-logits-v1",
        .input = bind(manifest.input_schema, "jyppx.liuding.onnx_tensor_schema",
                      tensor_schema("image", {-1, 3, 32, 32})),
        .output =
            bind(manifest.output_schema, "jyppx.liuding.onnx_tensor_schema",
                 tensor_schema("logits", {-1, 10})),
        .preprocessing = bind(manifest.preprocessing,
                              "jyppx.example.cifar10_input", Preprocessing()),
        .postprocessing =
            bind(manifest.postprocessing, "jyppx.example.cifar10_logits",
                 Postprocessing()),
        .request_schema = serving::onnx_invocation_schema(inputs),
        .response_schema = serving::onnx_result_schema(outputs),
        .preprocess = {},
        .postprocess = {}};
  };
}

void Evaluate(std::span<const fs::path> args) {
  Require(args.size() == 6 || args.size() == 7,
          "evaluate BUNDLE MANIFEST_SHA POLICY POLICY_SHA DATA OUTPUT "
          "[CPUExecutionProvider|DmlExecutionProvider]");
  const auto requested_provider =
      args.size() == 7 ? args[6].string() : "CPUExecutionProvider";
  Require(requested_provider == "CPUExecutionProvider" ||
              requested_provider == "DmlExecutionProvider",
          "Use CPUExecutionProvider or DmlExecutionProvider explicitly");
  const auto policy =
      signing::Ed25519TrustPolicy::load(args[2], {.hex = args[3].string()});
  auto verify = signing::trusted_ed25519_options(policy);
  verify.limits = Limits().limits;
  const std::array<std::string_view, 3> files{"cases", "tensors",
                                              "runtime-lock"};
  const auto snapshot = interop::read_model_bundle_snapshot(
      args[0], {.hex = args[1].string()}, files, verify, 96 * kMiB);
  Require(snapshot.verification.signature_status ==
              interop::BundleSignatureStatus::trusted,
          "The bundle must have a trusted Ed25519 signature");
  const auto suite = Json::parse(Text(snapshot.files.at("cases")));
  Require(suite.at("format") == "jyppx.liuding.onnx_verification_cases" &&
              suite.at("schema_version") == 1 &&
              suite.at("reference") == "native-checkpoint-model" &&
              suite.at("absolute_tolerance") == 1e-5 &&
              suite.at("relative_tolerance") == 1e-4 &&
              suite.at("cases").size() == 6,
          "Expected six native checkpoint references with the ONNX tolerances");
  interop::SafeTensorsLimits limits;
  limits.max_file_bytes = limits.max_payload_bytes = limits.max_tensor_bytes =
      64 * kMiB;
  const interop::SafeTensorsView tensors(snapshot.files.at("tensors"), limits);
  std::vector<serving::WarmupCase> warmup;
  constexpr std::array<std::int64_t, 6> batches{1, 2, 4, 8, 16, 128};
  for (std::size_t i = 0; i < batches.size(); ++i) {
    const auto& item = suite.at("cases").at(i);
    Require(item.at("inputs").size() == 1 && item.at("expected").size() == 1 &&
                item.at("inputs").at(0).at("name") == "image" &&
                item.at("expected").at(0).at("name") == "logits",
            "Warmup input/output names differ");
    auto input = Tensor(
        tensors, item.at("inputs").at(0).at("tensor").get<std::string>());
    auto expected = Tensor(
        tensors, item.at("expected").at(0).at("tensor").get<std::string>());
    Require(
        input.sizes().vec() ==
                std::vector<std::int64_t>{batches[i], 3, 32, 32} &&
            expected.sizes().vec() == std::vector<std::int64_t>{batches[i], 10},
        "Warmup batch/shape differs");
    c10::Dict<std::string, at::Tensor> result;
    result.insert("logits", std::move(expected));
    warmup.push_back(
        {.input = {.positional = {}, .keyword = {{"image", std::move(input)}}},
         .expected = std::move(result),
         .relative_tolerance = 1e-4,
         .absolute_tolerance = 1e-5});
  }
  const auto environment =
      ort::parse_environment_lock(Text(snapshot.files.at("runtime-lock")));
  Require(environment.provider.name == requested_provider &&
              !environment.allow_cpu_fallback,
          "Bundle provider must match the explicit request without fallback");
  serving::RuntimeDescriptor descriptor;
  descriptor.backend_id = environment.runtime.backend_id;
  descriptor.backend_version = environment.runtime.backend_version;
  descriptor.format = "onnx";
  descriptor.environment_lock = Digest(snapshot.files.at("runtime-lock"));
  for (const auto& entry : environment.runtime.dependencies)
    descriptor.dependencies.push_back(
        {.name = entry.name, .version = entry.version});
  descriptor.device = at::Device(at::kCPU);
  descriptor.dynamic_shapes = environment.runtime.dynamic_shapes;
  descriptor.execution_provider = environment.provider.name;
  descriptor.provider_options = environment.provider.options;
  descriptor.allow_cpu_fallback = false;
  descriptor.tensor_batching = false;
  descriptor.max_batch_size = 1;
  descriptor.oom_retry_safe = false;
  ort::SessionOptions session;
  session.provider = environment.provider;
  session.max_batch_size = 128;
  session.max_model_bytes = 64 * kMiB;
  session.max_input_bytes = 8 * kMiB;
  session.max_output_bytes = kMiB;
  serving::OnnxRuntimeBackend backend(descriptor, session,
                                      Owner(args[1].string()));
  serving::LoadOptions load;
  load.max_snapshot_bytes = 96 * kMiB;
  load.request_limits.max_tensor_bytes = 8 * kMiB;
  load.output_limits.max_tensor_bytes = kMiB;
  auto predictor = backend.load(args[0], verify, warmup, load);
  Images images(args[4]);
  NewDirectory(args[5]);
  std::ofstream logits_file(args[5] / "logits.f32", std::ios::binary);
  std::ofstream labels_file(args[5] / "labels.i64", std::ios::binary);
  std::ofstream inventory(args[5] / "samples.tsv");
  inventory << "index\tlabel\tpath\n";
  std::array<std::array<std::uint64_t, 10>, 10> confusion{};
  std::uint64_t correct = 0;
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t offset = 0; offset < kSamples; offset += 128) {
    std::vector<std::size_t> ids(std::min<std::size_t>(128, kSamples - offset));
    std::iota(ids.begin(), ids.end(), offset);
    const auto batch = images.batch(ids);
    serving::PredictRequest request;
    request.input = {.positional = {}, .keyword = {{"image", batch.images}}};
    request.context.request_id = "images-" + std::to_string(offset);
    request.context.model_version = predictor->info().model_version;
    const auto response = predictor->predict(request);
    const auto logits = response.output.value.toGenericDict()
                            .at("logits")
                            .toTensor()
                            .contiguous();
    Require(logits.scalar_type() == at::kFloat && logits.device().is_cpu() &&
                logits.sizes().vec() ==
                    std::vector<std::int64_t>{
                        static_cast<std::int64_t>(ids.size()), 10} &&
                at::isfinite(logits).all().item<bool>() &&
                batch.labels.scalar_type() == at::kLong &&
                batch.labels.numel() == static_cast<std::int64_t>(ids.size()),
            "Evaluation output or label contract differs");
    Append(logits_file, logits);
    Append(labels_file, batch.labels.contiguous());
    for (std::size_t j = 0; j < ids.size(); ++j) {
      const auto label = batch.labels.const_data_ptr<std::int64_t>()[j];
      Require(label >= 0 && label < 10, "Evaluation label outside class map");
      const auto* row = logits.const_data_ptr<float>() + j * 10;
      const auto predicted = std::max_element(row, row + 10) - row;
      ++confusion[static_cast<std::size_t>(label)]
                 [static_cast<std::size_t>(predicted)];
      correct += label == predicted;
      inventory << ids[j] << '\t' << label << '\t' << batch.items[j].image_path
                << '\n';
    }
    if (offset % 1024 == 0)
      std::cout << "evaluated=" << offset + ids.size() << std::endl;
  }
  logits_file.close();
  labels_file.close();
  inventory.close();
  Require(static_cast<bool>(logits_file) && static_cast<bool>(labels_file) &&
              static_cast<bool>(inventory),
          "Cannot finish evaluation output");
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  Require(predictor->close(std::chrono::steady_clock::now() +
                           std::chrono::seconds(10)),
          "Serving did not drain");
  const auto accuracy = static_cast<double>(correct) / kSamples;
  WriteJson(
      args[5] / "evaluation.json",
      {{"format", "jyppx.example.cifar10_serving_evaluation"},
       {"schema_version", 1},
       {"manifest_sha256", args[1].string()},
       {"policy_sha256", args[3].string()},
       {"signature_status", "trusted"},
       {"provider", environment.provider.name},
       {"provider_options", environment.provider.options},
       {"io_device", "cpu"},
       {"allow_cpu_fallback", false},
       {"samples", kSamples},
       {"correct", correct},
       {"accuracy", accuracy},
       {"threshold", 0.65},
       {"passed", accuracy >= 0.65},
       {"confusion_matrix", confusion},
       {"warmup_batches", batches},
       {"seconds", elapsed},
       {"training_updates", 0},
       {"logits_sha256", Digest(Read(args[5] / "logits.f32", kMiB)).hex},
       {"labels_sha256", Digest(Read(args[5] / "labels.i64", kMiB)).hex}});
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
    std::cout << "Commands:\n  prepare DATA CHECKPOINT CHECKPOINT_SHA RESULT "
                 "RESULT_SHA OUTPUT\n"
                 "  sign UNSIGNED EXPECTED_MANIFEST_SHA OUTPUT\n"
                 "  evaluate BUNDLE MANIFEST_SHA POLICY POLICY_SHA DATA OUTPUT "
                 "[CPUExecutionProvider|DmlExecutionProvider]\n";
    return 0;
  }
  try {
    jyppx::liuding::initialize_runtime();
    Require(std::endian::native == std::endian::little,
            "This example requires little-endian tensor output");
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    const torch::NoGradGuard guard;
    Require(!args.empty(), "Use --help for command syntax");
    const auto tail = std::span(args).subspan(1);
    if (args[0] == "prepare")
      Prepare(tail);
    else if (args[0] == "sign")
      Sign(tail);
    else if (args[0] == "evaluate")
      Evaluate(tail);
    else
      throw std::invalid_argument("Unknown command; use --help");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Image ONNX example failed: " << error.what() << '\n';
    return 1;
  }
}
