// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/environment.hpp>
#include <jyppx/liuding/deployment/onnx_serving/backend.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/serving/reload.hpp>

#include <torch/torch.h>

#include <ATen/core/Dict.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <c10/core/InferenceMode.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::serving;
namespace ort = jyppx::liuding::onnx_runtime;
using namespace std::chrono_literals;

std::span<const std::byte> Bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}
ArtifactDigest Digest(std::span<const std::byte> bytes) {
  return {.hex = sha256_hex(bytes)};
}
std::string InputText(std::span<const ort::TensorSignature> signatures) {
  const auto schema = onnx_invocation_schema(signatures);
  std::string result = "named-onnx-input-v1;";
  for (const auto& [name, value] : schema.keyword)
    result += name + ":" + canonical_value_schema(value);
  return result;
}

c10::IValue Expected(std::int64_t batch, float gain) {
  c10::Dict<std::string, Tensor> value;
  value.insert("sum", torch::full({batch, 2}, gain + 3.0F));
  value.insert("difference", torch::full({batch, 2}, -2.0F));
  return value;
}
nn::Invocation Input(std::int64_t batch = 1) {
  return {.positional = {},
          .keyword = {{"right", torch::full({batch, 2}, 3.0F)},
                      {"left", torch::ones({batch, 2})}}};
}
WarmupCase Known(float gain = 2.0F) {
  return {.input = Input(), .expected = Expected(1, gain)};
}
PredictRequest Request(std::string version, std::int64_t batch = 1) {
  PredictRequest result;
  result.input = Input(batch);
  result.context.request_id = "onnx-serving-request";
  result.context.model_version = std::move(version);
  return result;
}
Tensor Field(const c10::IValue& output, const char* name) {
  return output.toGenericDict().at(std::string(name)).toTensor();
}

struct Bundle {
  std::filesystem::path root, directory;
  RuntimeDescriptor descriptor;
  interop::ModelBundleManifest manifest;
  std::vector<interop::BundlePublishSource> sources;
  ort::EnvironmentLock environment;
  float gain;

  explicit Bundle(std::string version = "onnx-v1", float value = 2.0F,
                  std::string batch_symbol = "batch")
      : gain(value) {
    static std::atomic<std::uint64_t> sequence{0};
    root = std::filesystem::temp_directory_path() /
           ("liuding-onnx-serving-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(++sequence));
    std::filesystem::create_directories(root / "sources");
    directory = root / "bundle";
    torch::nn::Module module;
    const std::array<Tensor, 2> examples{torch::ones({1, 2}),
                                         torch::ones({1, 2})};
    onnx_export::ExportOptions options;
    options.input_names = {"left", "right"};
    options.output_names = {"sum", "difference"};
    options.maximum_batch = 8;
    for (const auto& name : {"left", "right", "sum", "difference"})
      options.dynamic_axes[name] = {{0, batch_symbol}};
    const auto exported = onnx_export::export_module(
        module,
        [value](std::span<const Tensor> input) {
          return std::vector<Tensor>{input[0] * value + input[1],
                                     input[0] - input[1]};
        },
        examples, options);
    const ort::Session session(exported.onnx, exported.digest);
    environment = ort::observe_environment(session);
    const auto lock = ort::serialize_environment_lock(environment);
    descriptor.backend_id = environment.runtime.backend_id;
    descriptor.backend_version = environment.runtime.backend_version;
    descriptor.format = "onnx";
    descriptor.environment_lock = Digest(Bytes(lock));
    for (const auto& dependency : environment.runtime.dependencies)
      descriptor.dependencies.push_back({dependency.name, dependency.version});
    descriptor.dynamic_shapes = environment.runtime.dynamic_shapes;
    descriptor.execution_provider = environment.provider.name;
    descriptor.provider_options = environment.provider.options;
    descriptor.allow_cpu_fallback = environment.allow_cpu_fallback;
    manifest.bundle_id = std::move(version);
    manifest.build = current_build_fingerprint();
    manifest.runtime = {.backend_id = descriptor.backend_id,
                        .backend_version = descriptor.backend_version,
                        .environment_lock = descriptor.environment_lock,
                        .dependencies = descriptor.dependencies,
                        .devices = environment.runtime.devices,
                        .dtypes = environment.runtime.dtypes,
                        .os_arch = environment.runtime.os_arch,
                        .dynamic_shapes = environment.runtime.dynamic_shapes};
    Add("graph", interop::BundleFileRole::model, exported.onnx);
    manifest.models.push_back({.id = "primary",
                               .format = interop::BundleModelFormat::onnx,
                               .file_ids = {"graph"},
                               .tensors = {}});
    manifest.input_schema = Descriptor("input", interop::BundleFileRole::schema,
                                       InputText(session.inputs()));
    manifest.output_schema = Descriptor(
        "output", interop::BundleFileRole::schema,
        canonical_value_schema(onnx_result_schema(session.outputs())));
    manifest.preprocessing = Descriptor(
        "preprocess", interop::BundleFileRole::prepost, "identity-v1");
    manifest.postprocessing = Descriptor(
        "postprocess", interop::BundleFileRole::prepost, "identity-v1");
    manifest.evaluation =
        Descriptor("evaluation", interop::BundleFileRole::evaluation,
                   "synthetic-known-vector");
    manifest.sbom = Descriptor("sbom", interop::BundleFileRole::sbom,
                               "test-runtime-observation");
    manifest.provenance =
        Descriptor("provenance", interop::BundleFileRole::provenance,
                   "generated-native-cpp-test");
    auto reference =
        Descriptor("runtime-lock", interop::BundleFileRole::evidence, lock);
    reference.schema_id = std::string(ort::kEnvironmentLockSchemaId);
    manifest.evidence.push_back(std::move(reference));
  }
  ~Bundle() {
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
  }
  Bundle(const Bundle&) = delete;
  Bundle& operator=(const Bundle&) = delete;

  void Add(std::string id, interop::BundleFileRole role,
           std::span<const std::byte> bytes) {
    const bool model = role == interop::BundleFileRole::model;
    const auto source = root / "sources" / (id + ".data");
    std::ofstream stream(source, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!stream)
      throw std::runtime_error("Cannot write ONNX Serving test source");
    sources.push_back({.file_id = id, .source_path = source});
    manifest.files.push_back(
        {.id = id,
         .relative_path = "artifacts/" + id + (model ? ".onnx" : ".data"),
         .role = role,
         .artifact = {.digest = Digest(bytes),
                      .size_bytes = bytes.size(),
                      .logical_name = id,
                      .media_type = model ? "application/onnx"
                                          : "application/octet-stream",
                      .producer = "jyppx.test.onnx-serving"}});
  }
  interop::BundleDescriptorRef Descriptor(std::string id,
                                          interop::BundleFileRole role,
                                          std::string_view text) {
    Add(id, role, Bytes(text));
    return {.schema_id = "jyppx.test.onnx." + id,
            .schema_version = 1U,
            .file_id = std::move(id)};
  }
  void Publish() {
    static_cast<void>(
        interop::publish_model_bundle(directory, manifest, sources));
  }
  OnnxPipelineFactory Factory() const {
    return [](const VerifiedBundle& bundle, auto inputs, auto outputs) {
      const auto bind = [&](const interop::BundleDescriptorRef& reference,
                            std::string_view expected) {
        const auto& bytes = bundle.files.at(reference.file_id);
        if (reference.schema_id != "jyppx.test.onnx." + reference.file_id ||
            reference.schema_version != 1U || bytes.size() != expected.size() ||
            !std::equal(bytes.begin(), bytes.end(), Bytes(expected).begin()))
          throw CompatibilityError(
              "Owner descriptor does not match its decoded graph contract");
        return DescriptorBinding{.schema_id = reference.schema_id,
                                 .schema_version = reference.schema_version,
                                 .digest = Digest(bytes)};
      };
      const auto& m = bundle.verification.manifest;
      return Pipeline{
          .version = "onnx-test-identity-v1",
          .input = bind(m.input_schema, InputText(inputs)),
          .output = bind(m.output_schema,
                         canonical_value_schema(onnx_result_schema(outputs))),
          .preprocessing = bind(m.preprocessing, "identity-v1"),
          .postprocessing = bind(m.postprocessing, "identity-v1"),
          .request_schema = onnx_invocation_schema(inputs),
          .response_schema = onnx_result_schema(outputs),
          .preprocess = {},
          .postprocess = {}};
    };
  }
  OnnxRuntimeBackend Backend(OnnxPipelineFactory factory = {},
                             ort::SessionOptions options = {}) const {
    options.max_batch_size = 8;
    return OnnxRuntimeBackend(descriptor, options,
                              factory ? std::move(factory) : Factory());
  }
  std::shared_ptr<Predictor> Load(const LoadOptions& options = {},
                                  OnnxPipelineFactory factory = {}) const {
    const std::array<WarmupCase, 1> cases{Known(gain)};
    return Backend(std::move(factory)).load(directory, {}, cases, options);
  }
};

TEST(OnnxServingTest, NamedMultiIoDynamicBatchOutputsOwnStorageAfterClose) {
  Bundle bundle;
  bundle.Publish();
  auto predictor = bundle.Load();
  EXPECT_EQ(predictor->info().accepted, 0U);
  EXPECT_EQ(predictor->info().runtime.execution_provider,
            "CPUExecutionProvider");
  std::vector<PredictResponse> replies;
  for (const auto batch : {1, 2, 5}) {
    auto request = Request(bundle.manifest.bundle_id, batch);
    replies.push_back(predictor->predict(request));
    request.input.keyword.at("left").toTensor().fill_(-100);
    EXPECT_TRUE(torch::equal(Field(replies.back().output.value, "sum"),
                             torch::full({batch, 2}, 5.0F)));
    EXPECT_TRUE(torch::equal(Field(replies.back().output.value, "difference"),
                             torch::full({batch, 2}, -2.0F)));
  }
  {
    // Serving returns inference tensors; mutate under the matching mode
    // without cloning so this still checks independent response storage.
    c10::InferenceMode guard;
    Field(replies[0].output.value, "sum").fill_(99);
  }
  EXPECT_EQ(Field(replies[1].output.value, "sum")[0][0].item<float>(), 5.0F);
  EXPECT_TRUE(predictor->close());
  EXPECT_EQ(Field(replies[2].output.value, "difference")[0][0].item<float>(),
            -2.0F);
  EXPECT_THROW(predictor->predict(Request(bundle.manifest.bundle_id)),
               NotReady);
}

TEST(OnnxServingTest,
     InputNamesDtypesFiniteValuesAndBatchSymbolsFailBeforeRun) {
  Bundle bundle;
  bundle.Publish();
  auto predictor = bundle.Load();
  auto request = Request(bundle.manifest.bundle_id);
  request.input.keyword.erase("right");
  EXPECT_THROW(predictor->predict(request), InvalidArgument);
  request = Request(bundle.manifest.bundle_id);
  request.input.keyword.emplace("extra", torch::ones({1, 2}));
  EXPECT_THROW(predictor->predict(request), InvalidArgument);
  request = Request(bundle.manifest.bundle_id);
  request.input.keyword.at("left") = torch::ones({1, 2}, torch::kFloat64);
  EXPECT_THROW(predictor->predict(request), InvalidArgument);
  request = Request(bundle.manifest.bundle_id);
  request.input.keyword.at("left") = torch::ones({1, 3});
  EXPECT_THROW(predictor->predict(request), InvalidArgument);
  request = Request(bundle.manifest.bundle_id);
  request.input.keyword.at("left") =
      torch::full({1, 2}, std::numeric_limits<float>::infinity());
  EXPECT_THROW(predictor->predict(request), NumericalError);
  request = Request(bundle.manifest.bundle_id);
  request.input.keyword.at("left") = torch::ones({2, 2});
  EXPECT_THROW(predictor->predict(request), InvalidArgument);
  EXPECT_THROW(predictor->predict(Request(bundle.manifest.bundle_id, 9)),
               ResourceExhausted);
  EXPECT_THROW(predictor->predict(Request(bundle.manifest.bundle_id, 0)),
               ResourceExhausted);
  EXPECT_EQ(predictor->info().health, HealthStatus::ready);
  EXPECT_NO_THROW(predictor->predict(Request(bundle.manifest.bundle_id)));
}

TEST(OnnxServingTest,
     SessionInputAndOutputPayloadLimitsAreAppliedBeforeAllocation) {
  Bundle bundle;
  bundle.Publish();
  ort::SessionOptions session;
  session.max_input_bytes = 16;
  session.max_output_bytes = 16;
  auto backend = bundle.Backend({}, session);
  const std::array<WarmupCase, 1> cases{Known()};
  auto predictor = backend.load(bundle.directory, {}, cases);
  EXPECT_EQ(predictor->request_limits().max_tensor_bytes, 16U);
  EXPECT_THROW(predictor->predict(Request(bundle.manifest.bundle_id, 2)),
               ResourceExhausted);
  EXPECT_EQ(predictor->info().health, HealthStatus::ready);
}

TEST(OnnxServingTest, OwnerGraphSchemaAndAllDescriptorBindingsAreEnforced) {
  Bundle bundle;
  bundle.Publish();
  const auto factory = bundle.Factory();
  for (int failure = 0; failure < 6; ++failure) {
    auto bad = [factory, failure](const VerifiedBundle& verified, auto inputs,
                                  auto outputs) {
      auto pipeline = factory(verified, inputs, outputs);
      if (failure == 0)
        pipeline.request_schema.keyword.at("left").tensor_constraints->dtype =
            torch::kFloat64;
      if (failure == 1) pipeline.response_schema.fields.erase("sum");
      if (failure == 2) pipeline.input.digest.hex.assign(64, '0');
      if (failure == 3) pipeline.output.schema_version = 2;
      if (failure == 4) pipeline.preprocessing.digest.hex.assign(64, '0');
      if (failure == 5) pipeline.postprocessing.schema_id = "unknown.owner";
      return pipeline;
    };
    EXPECT_THROW(bundle.Load({}, bad), CompatibilityError) << failure;
  }
}

TEST(OnnxServingTest,
     ExactActualEnvironmentLockRejectsDeclaredDependencyDrift) {
  Bundle bundle;
  auto& lock = bundle.environment;
  for (auto& dependency : lock.runtime.dependencies)
    if (dependency.name == "onnxruntime") dependency.version += "-unavailable";
  for (auto& dependency : bundle.descriptor.dependencies)
    if (dependency.name == "onnxruntime") dependency.version += "-unavailable";
  bundle.manifest.runtime.dependencies = bundle.descriptor.dependencies;
  const auto encoded = ort::serialize_environment_lock(lock);
  bundle.descriptor.environment_lock = Digest(Bytes(encoded));
  bundle.manifest.runtime.environment_lock = bundle.descriptor.environment_lock;
  auto& file =
      *std::find_if(bundle.manifest.files.begin(), bundle.manifest.files.end(),
                    [](const auto& item) { return item.id == "runtime-lock"; });
  file.artifact.digest = bundle.descriptor.environment_lock;
  file.artifact.size_bytes = encoded.size();
  std::ofstream stream(bundle.root / "sources/runtime-lock.data",
                       std::ios::binary | std::ios::trunc);
  stream << encoded;
  stream.close();
  bundle.Publish();
  EXPECT_THROW(bundle.Load(), CompatibilityError);
}

TEST(OnnxServingTest,
     MissingEnvironmentEvidenceAndUntrustedSignatureAreRejected) {
  Bundle bundle;
  bundle.manifest.evidence.front().schema_id = "jyppx.test.other-evidence";
  bundle.Publish();
  EXPECT_THROW(bundle.Load(), CompatibilityError);
  interop::ModelBundleVerifyOptions verify;
  verify.require_trusted_signature = true;
  const std::array<WarmupCase, 1> cases{Known()};
  EXPECT_THROW(bundle.Backend().load(bundle.directory, verify, cases),
               CompatibilityError);
}

TEST(OnnxServingTest, LoadBudgetsWarmupCancellationAndDeadlineDoNotPublish) {
  Bundle bundle;
  bundle.Publish();
  auto backend = bundle.Backend();
  const std::array<WarmupCase, 1> cases{Known()};
  LoadOptions limits;
  limits.max_snapshot_bytes = 1;
  EXPECT_THROW(backend.load(bundle.directory, {}, cases, limits),
               ResourceExhausted);
  limits = {};
  limits.max_snapshot_bytes = 0;
  EXPECT_THROW(backend.load_reservation_bytes(limits), ResourceExhausted);
  limits = {};
  limits.deadline = std::chrono::steady_clock::now() - 1ms;
  EXPECT_THROW(backend.load(bundle.directory, {}, cases, limits),
               DeadlineExceeded);
  limits = {};
  std::stop_source stop;
  stop.request_stop();
  limits.stop = stop.get_token();
  EXPECT_THROW(backend.load(bundle.directory, {}, cases, limits), Cancelled);
  EXPECT_THROW(backend.load(bundle.directory, {}, {}), InvalidArgument);
  const std::array<WarmupCase, 1> wrong{Known(99)};
  EXPECT_THROW(backend.load(bundle.directory, {}, wrong), NumericalError);
  ort::SessionOptions session;
  session.max_model_bytes = 1;
  EXPECT_THROW(bundle.Backend({}, session).load(bundle.directory, {}, cases),
               ResourceExhausted);
}

TEST(OnnxServingTest,
     SnapshotOutlivesBundlePathsAndOwnerFactoryBorrowedArguments) {
  Bundle bundle;
  bundle.Publish();
  const auto factory = bundle.Factory();
  auto remove = [factory, path = bundle.directory](
                    const VerifiedBundle& verified, auto inputs, auto outputs) {
    auto pipeline = factory(verified, inputs, outputs);
    std::filesystem::remove_all(path);
    return pipeline;
  };
  auto predictor = bundle.Load({}, remove);
  EXPECT_FALSE(std::filesystem::exists(bundle.directory));
  EXPECT_TRUE(torch::equal(
      Field(predictor->predict(Request(bundle.manifest.bundle_id)).output.value,
            "sum"),
      torch::full({1, 2}, 5.0F)));
}

TEST(OnnxServingTest, ExplicitProviderMetadataIsIndependentOfHostTensorDevice) {
  Bundle bundle;
  auto descriptor = bundle.descriptor;
  ort::SessionOptions session;
  descriptor.execution_provider = "CUDAExecutionProvider";
  descriptor.provider_options = {{"device_id", "0"}};
  session.provider = {.name = descriptor.execution_provider,
                      .options = descriptor.provider_options};
  EXPECT_NO_THROW(OnnxRuntimeBackend(descriptor, session, bundle.Factory()));
  EXPECT_EQ(descriptor.device, Device(c10::kCPU));
  descriptor.allow_cpu_fallback = true;
  EXPECT_THROW(OnnxRuntimeBackend(descriptor, session, bundle.Factory()),
               InvalidArgument);
  session.allow_cpu_fallback = true;
  EXPECT_NO_THROW(OnnxRuntimeBackend(descriptor, session, bundle.Factory()));
  descriptor.device = Device(c10::kCUDA, 0);
  EXPECT_THROW(OnnxRuntimeBackend(descriptor, session, bundle.Factory()),
               Error);
}

TEST(OnnxServingTest, ReloadPinsNumericVersionsAndUsesOnnxSnapshotBudget) {
  Bundle first, second("onnx-v2", 5.0F);
  first.Publish();
  second.Publish();
  LoadOptions load;
  load.max_snapshot_bytes = 1U << 20U;
  load.max_state_bytes = 0;
  const auto reservation = first.Backend().load_reservation_bytes(load);
  EXPECT_EQ(reservation, load.max_snapshot_bytes);
  load.max_state_bytes = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(first.Backend().load_reservation_bytes(load), reservation);
  ReloadPolicy policy;
  policy.initial_reservation_bytes = reservation;
  policy.max_resident_bytes = reservation * 2;
  ReloadablePredictor owner(first.Load(load), policy);
  auto old = std::make_unique<PredictionLease>(owner.acquire());
  ReloadOptions reload;
  reload.drain_timeout = 0ms;
  const std::array<WarmupCase, 1> known{Known(5)};
  const auto changed =
      owner.reload(second.Backend(), second.directory, {}, known, load, reload);
  EXPECT_FALSE(changed.previous_drained);
  EXPECT_EQ(owner.statistics().reserved_bytes, reservation * 2);
  EXPECT_EQ(Field(old->predict(Request(first.manifest.bundle_id)).output.value,
                  "sum")[0][0]
                .item<float>(),
            5.0F);
  EXPECT_EQ(
      Field(owner.predict(Request(second.manifest.bundle_id)).output.value,
            "sum")[0][0]
          .item<float>(),
      8.0F);
  EXPECT_THROW(owner.acquire(first.manifest.bundle_id), InvalidArgument);
  EXPECT_THROW(owner.reload(first.Backend(), first.directory, {}, known, load),
               NotReady);
  old.reset();
  EXPECT_TRUE(owner.drain_previous());
  EXPECT_EQ(owner.statistics().reserved_bytes, reservation);
  EXPECT_TRUE(owner.close());
}

class DescriptorOnlyBackend final : public RuntimeBackend {
 public:
  RuntimeDescriptor runtime;
  mutable std::size_t loads = 0;
  explicit DescriptorOnlyBackend(RuntimeDescriptor value)
      : runtime(std::move(value)) {}
  RuntimeDescriptor descriptor() const override { return runtime; }
  std::shared_ptr<Predictor> load(const std::filesystem::path&,
                                  const interop::ModelBundleVerifyOptions&,
                                  std::span<const WarmupCase>,
                                  const LoadOptions&) const override {
    ++loads;
    throw std::logic_error("Incompatible descriptor must fail before loading");
  }
};

TEST(OnnxServingTest,
     ReloadRejectsProviderOptionsAndFallbackBeforeCandidateLoad) {
  Bundle first;
  first.Publish();
  LoadOptions load;
  load.max_snapshot_bytes = 1U << 20U;
  ReloadablePredictor owner(
      first.Load(load), {.initial_reservation_bytes = load.max_snapshot_bytes,
                         .max_resident_bytes = load.max_snapshot_bytes * 2});
  for (int failure = 0; failure < 3; ++failure) {
    auto descriptor = first.descriptor;
    if (failure == 0) descriptor.execution_provider = "CUDAExecutionProvider";
    if (failure == 1) descriptor.provider_options["device_id"] = "1";
    if (failure == 2)
      descriptor.allow_cpu_fallback = !descriptor.allow_cpu_fallback;
    DescriptorOnlyBackend backend(std::move(descriptor));
    EXPECT_THROW(owner.reload(backend, first.directory, {}, {}),
                 CompatibilityError);
    EXPECT_EQ(backend.loads, 0U);
  }
  EXPECT_EQ(owner.info().model_version, first.manifest.bundle_id);
}

TEST(OnnxServingTest,
     ReloadRejectsChangedDynamicSymbolContractAndPreservesOldVersion) {
  Bundle first, changed("onnx-v2", 5.0F, "different-batch");
  first.Publish();
  changed.Publish();
  LoadOptions load;
  load.max_snapshot_bytes = 1U << 20U;
  ReloadPolicy policy;
  policy.initial_reservation_bytes = load.max_snapshot_bytes;
  policy.max_resident_bytes = load.max_snapshot_bytes * 2;
  ReloadablePredictor owner(first.Load(load), policy);
  const std::array<WarmupCase, 1> known{Known(5)};
  EXPECT_THROW(
      owner.reload(changed.Backend(), changed.directory, {}, known, load),
      CompatibilityError);
  EXPECT_EQ(owner.info().model_version, first.manifest.bundle_id);
  EXPECT_EQ(owner.statistics().publications, 0U);
  EXPECT_EQ(owner.statistics().reserved_bytes, load.max_snapshot_bytes);
  EXPECT_EQ(Field(owner.predict(Request(first.manifest.bundle_id)).output.value,
                  "sum")[0][0]
                .item<float>(),
            5.0F);
}

TEST(OnnxServingTest,
     ReloadTotalReservationRejectsBeforeFactoryAndWarmupFailureKeepsOld) {
  Bundle first, second("onnx-v2", 5.0F);
  first.Publish();
  second.Publish();
  LoadOptions load;
  load.max_snapshot_bytes = 1U << 20U;
  ReloadPolicy policy;
  policy.initial_reservation_bytes = load.max_snapshot_bytes;
  policy.max_resident_bytes = load.max_snapshot_bytes;
  ReloadablePredictor full(first.Load(load), policy);
  const std::array<WarmupCase, 1> known{Known(5)};
  EXPECT_THROW(full.reload(second.Backend(), second.directory, {}, known, load),
               ResourceExhausted);
  EXPECT_EQ(full.statistics().resident_instances, 1U);
  policy.max_resident_bytes *= 2;
  ReloadablePredictor owner(first.Load(load), policy);
  const std::array<WarmupCase, 1> wrong{Known(99)};
  EXPECT_THROW(
      owner.reload(second.Backend(), second.directory, {}, wrong, load),
      NumericalError);
  EXPECT_EQ(owner.info().model_version, first.manifest.bundle_id);
  EXPECT_EQ(owner.statistics().reserved_bytes, load.max_snapshot_bytes);
}

TEST(OnnxServingTest, CancelledAndExpiredRequestsKeepSessionReady) {
  Bundle bundle;
  bundle.Publish();
  auto armed = std::make_shared<std::atomic<bool>>(false);
  auto entered = std::make_shared<std::promise<void>>();
  auto release = std::make_shared<std::promise<void>>();
  auto released = release->get_future().share();
  const auto factory = bundle.Factory();
  auto blocking = [factory, armed, entered, released](
                      const VerifiedBundle& verified, auto inputs,
                      auto outputs) {
    auto pipeline = factory(verified, inputs, outputs);
    pipeline.preprocess = [armed, entered,
                           released](const nn::Invocation& input) {
      if (armed->exchange(false)) {
        entered->set_value();
        released.wait();
      }
      return input;
    };
    return pipeline;
  };
  auto predictor = bundle.Load({}, blocking);
  std::stop_source stop;
  armed->store(true);
  auto request = std::async(std::launch::async, [&] {
    return predictor->predict(Request(bundle.manifest.bundle_id),
                              stop.get_token());
  });
  const auto reached = entered->get_future().wait_for(2s);
  if (reached != std::future_status::ready) {
    release->set_value();
    ASSERT_EQ(reached, std::future_status::ready);
  }
  EXPECT_EQ(predictor->info().active, 1U);
  stop.request_stop();
  release->set_value();
  EXPECT_THROW(request.get(), Cancelled);
  EXPECT_EQ(predictor->info().health, HealthStatus::ready);
  EXPECT_NO_THROW(predictor->predict(Request(bundle.manifest.bundle_id)));
  auto expired = Request(bundle.manifest.bundle_id);
  expired.deadline = std::chrono::steady_clock::now() - 1ms;
  EXPECT_THROW(predictor->predict(expired), DeadlineExceeded);
  EXPECT_TRUE(predictor->close());
}

TEST(OnnxServingTest, ObservedCancellationAfterSuccessfulRunKeepsSessionReady) {
  Bundle bundle;
  bundle.Publish();
  auto armed = std::make_shared<std::atomic<bool>>(false);
  auto postprocessed = std::make_shared<std::atomic<bool>>(false);
  auto stop = std::make_shared<std::stop_source>();
  const auto factory = bundle.Factory();
  auto cancel_after_run = [factory, armed, postprocessed, stop](
                              const VerifiedBundle& verified, auto inputs,
                              auto outputs) {
    auto pipeline = factory(verified, inputs, outputs);
    pipeline.postprocess = [armed, postprocessed,
                            stop](const nn::InvocationResult& result) {
      if (armed->exchange(false)) {
        postprocessed->store(true);
        stop->request_stop();
      }
      return result;
    };
    return pipeline;
  };
  auto predictor = bundle.Load({}, cancel_after_run);
  armed->store(true);
  EXPECT_THROW(
      predictor->predict(Request(bundle.manifest.bundle_id), stop->get_token()),
      Cancelled);
  EXPECT_TRUE(postprocessed->load());
  EXPECT_EQ(predictor->info().health, HealthStatus::ready);
  EXPECT_NO_THROW(predictor->predict(Request(bundle.manifest.bundle_id)));
}

TEST(OnnxServingTest, ActiveCloseRetainsSessionUntilInvocationDrains) {
  Bundle bundle;
  bundle.Publish();
  auto armed = std::make_shared<std::atomic<bool>>(false);
  auto entered = std::make_shared<std::promise<void>>();
  auto release = std::make_shared<std::promise<void>>();
  auto released = release->get_future().share();
  const auto factory = bundle.Factory();
  auto blocking = [factory, armed, entered, released](
                      const VerifiedBundle& verified, auto inputs,
                      auto outputs) {
    auto pipeline = factory(verified, inputs, outputs);
    pipeline.preprocess = [armed, entered,
                           released](const nn::Invocation& input) {
      if (armed->exchange(false)) {
        entered->set_value();
        released.wait();
      }
      return input;
    };
    return pipeline;
  };
  auto predictor = bundle.Load({}, blocking);
  armed->store(true);
  auto request = std::async(std::launch::async, [&] {
    return predictor->predict(Request(bundle.manifest.bundle_id));
  });
  const auto reached = entered->get_future().wait_for(2s);
  if (reached != std::future_status::ready) {
    release->set_value();
    ASSERT_EQ(reached, std::future_status::ready);
  }
  EXPECT_FALSE(predictor->close(std::chrono::steady_clock::now()));
  EXPECT_EQ(predictor->info().health, HealthStatus::draining);
  release->set_value();
  EXPECT_EQ(Field(request.get().output.value, "sum")[0][0].item<float>(), 5.0F);
  EXPECT_TRUE(predictor->close());
  EXPECT_EQ(predictor->info().health, HealthStatus::closed);
}
}  // namespace
