// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include "recipe_support.hpp"

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/experiment/config.hpp>
#include <jyppx/liuding/experiment/evaluation.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/random_checkpoint_participant.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/model_bundle.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>
#include <jyppx/liuding/training/classification_evaluation.hpp>
#include <jyppx/liuding/training/diagnostics.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>
#include <jyppx/liuding/training/scheduler_adapter.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>
#include <jyppx/liuding/training/trainer.hpp>

#include <torch/cuda.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/modules/linear.h>

#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stop_token>

namespace jyppx::liuding::cli {
namespace {
using namespace recipe_support;
namespace ex = experiment;
namespace tr = training;
constexpr std::string_view kRecipe = "jyppx.liuding.paired-affine/v1";
constexpr std::string_view kRun = "run.paired-affine";

const std::map<std::string, ValueKind>& field_kinds() {
  static const std::map<std::string, ValueKind> fields{
      {"schema_version", ValueKind::integer},
      {"recipe", ValueKind::string},
      {"train_file", ValueKind::string},
      {"validation_file", ValueKind::string},
      {"data_license", ValueKind::string},
      {"device", ValueKind::string},
      {"seed", ValueKind::integer},
      {"epochs", ValueKind::integer},
      {"batch_size", ValueKind::integer},
      {"workers", ValueKind::integer},
      {"prefetch_batches", ValueKind::integer},
      {"checkpoint_every", ValueKind::integer},
      {"learning_rate", ValueKind::floating},
      {"momentum", ValueKind::floating},
      {"weight_decay", ValueKind::floating},
      {"lr_gamma", ValueKind::floating},
      {"regression_weight", ValueKind::floating},
      {"minimum_accuracy", ValueKind::floating},
      {"maximum_mse", ValueKind::floating},
      {"max_samples", ValueKind::integer},
      {"max_record_bytes", ValueKind::integer},
      {"max_data_bytes", ValueKind::integer},
      {"max_batch_bytes", ValueKind::integer}};
  return fields;
}

Json resolve_config(const std::filesystem::path& path) {
  ex::config::ConfigSchema schema;
  schema.schema_id = std::string(kRecipe);
  schema.root.kind = ValueKind::dictionary;
  for (const auto& [name, kind] : field_kinds()) {
    ValueSchema field;
    field.name = name;
    field.kind = kind;
    schema.root.fields.emplace(name, std::move(field));
  }
  ex::config::ConfigResolver resolver(std::move(schema));
  resolver.add_json_file(path);
  const auto resolved = resolver.resolve();
  auto config = parse_json(resolved.canonical_json());
  const auto integer = [&](const char* name, std::int64_t low,
                           std::int64_t high) {
    const auto value = resolved.integer(std::string("/") + name);
    require_recipe(value >= low && value <= high,
                   std::string("Recipe field out of range: ") + name);
  };
  integer("schema_version", 1, 1);
  integer("seed", 0, 2147483647);
  integer("epochs", 1, 1000);
  integer("batch_size", 1, 1024);
  integer("workers", 0, 8);
  integer("prefetch_batches", 1, 8);
  integer("checkpoint_every", 1, 1000000);
  integer("max_samples", 2, 4096);
  integer("max_record_bytes", 128, 4096);
  integer("max_data_bytes", 1024, 1048576);
  integer("max_batch_bytes", 28, 1048576);
  require_recipe(config.at("recipe").get<std::string>() == kRecipe,
                 "Unsupported paired recipe identity");
  require_recipe(config.at("device") == "cpu" || config.at("device") == "cuda",
                 "Recipe device must be cpu or cuda");
  for (const auto* key :
       {"learning_rate", "momentum", "weight_decay", "lr_gamma",
        "regression_weight", "minimum_accuracy", "maximum_mse"}) {
    const auto value = resolved.floating(std::string("/") + key);
    require_recipe(std::isfinite(value) && value >= 0 && value <= 1,
                   std::string("Recipe field must be in [0,1]: ") + key);
  }
  require_recipe(config.at("learning_rate").get<double>() > 0 &&
                     config.at("momentum").get<double>() < 1 &&
                     config.at("lr_gamma").get<double>() > 0 &&
                     config.at("regression_weight").get<double>() > 0 &&
                     config.at("minimum_accuracy").get<double>() > 0 &&
                     config.at("maximum_mse").get<double>() > 0,
                 "Recipe requires positive learning rate, gamma, regression "
                 "weight and thresholds; momentum < 1");
  require_recipe(config.at("batch_size").get<std::uint64_t>() * 28 <=
                     config.at("max_batch_bytes").get<std::uint64_t>(),
                 "Recipe batch exceeds max_batch_bytes before allocation");
  require_recipe(!config.at("data_license").get<std::string>().empty(),
                 "Declare the dataset license");
  for (const auto* key : {"train_file", "validation_file"}) {
    require_recipe(!config.at(key).get<std::string>().empty(),
                   "Recipe data path must not be empty");
    auto file = std::filesystem::u8path(config.at(key).get<std::string>());
    if (file.is_relative()) file = path.parent_path() / file;
    config[key] = path_text(std::filesystem::absolute(file).lexically_normal());
  }
  require_recipe(config.at("train_file") != config.at("validation_file"),
                 "Training and validation files must differ");
  return config;
}

struct Row {
  std::array<float, 2> left, right;
  float regression;
  std::int64_t category;
};
struct Inputs {
  std::shared_ptr<const std::vector<Row>> train, validation;
  Json manifest;
  ArtifactDigest identity;
  std::uint64_t total_steps;
};

Inputs inspect_inputs(const Json& config) {
  std::uint64_t total_bytes = 0;
  Json manifest{{"schema_version", 1},
                {"dataset", kRecipe},
                {"data_license", config.at("data_license")}};
  const auto load = [&](const char* key, const char* split) {
    const auto path =
        std::filesystem::u8path(config.at(key).get<std::string>());
    const auto identity =
        file_digest(path, config.at("max_data_bytes").get<std::uint64_t>());
    const auto size = std::filesystem::file_size(path);
    require_recipe(
        size <= config.at("max_data_bytes").get<std::uint64_t>() - total_bytes,
        "Recipe dataset exceeds total byte limit");
    total_bytes += size;
    auto rows = std::make_shared<std::vector<Row>>();
    const auto maximum = config.at("max_samples").get<std::size_t>();
    rows->reserve(maximum);
    std::array<std::uint64_t, 2> classes{};
    std::ifstream stream(path, std::ios::binary);
    require_recipe(static_cast<bool>(stream), "Cannot read paired recipe data");
    std::vector<char> line(config.at("max_record_bytes").get<std::size_t>() +
                           1);
    Sha256 hash;
    std::uint64_t count = 0;
    while (stream.peek() != std::char_traits<char>::eof()) {
      stream.getline(line.data(), static_cast<std::streamsize>(line.size()));
      require_recipe(!stream.fail() && !stream.eof(),
                     "Recipe record is oversized or missing its final LF");
      const auto length = static_cast<std::size_t>(stream.gcount());
      require_recipe(
          length != 0 &&
              length <= config.at("max_record_bytes").get<std::size_t>(),
          "Recipe record byte limit exceeded");
      hash.update(bytes(std::string_view(line.data(), length - 1)));
      hash.update(bytes("\n"));
      count += length;
      require_recipe(rows->size() < maximum && count <= size,
                     "Recipe sample/byte limit exceeded");
      const auto value = parse_json(std::string_view(line.data(), length - 1));
      fields(value, {"id", "left", "right", "class_id", "regression"});
      std::ostringstream id;
      id << split << ':' << std::setw(6) << std::setfill('0') << rows->size();
      require_recipe(value.at("id") == id.str(),
                     "Recipe row identity/order differs");
      Row row{};
      const auto feature = [&](const char* field, std::size_t width,
                               float* output, double minimum,
                               double maximum_value) {
        const auto& values = value.at(field);
        require_recipe(values.is_array() && values.size() == width,
                       "Recipe input/target shape differs");
        for (std::size_t i = 0; i < width; ++i) {
          require_recipe(values[i].is_number() && !values[i].is_boolean(),
                         "Recipe feature must be numeric");
          const auto number = values[i].get<double>();
          require_recipe(
              std::isfinite(number) && number >= minimum &&
                  number <= maximum_value &&
                  static_cast<double>(static_cast<float>(number)) == number,
              "Recipe value must be finite, exactly float32 and in its "
              "declared range");
          output[i] = static_cast<float>(number);
        }
      };
      feature("left", 2, row.left.data(), -1, 1);
      feature("right", 2, row.right.data(), -1, 1);
      feature("regression", 1, &row.regression, -19.0 / 16.0, 21.0 / 16.0);
      require_recipe(
          value.at("class_id").is_number_integer() &&
              (value.at("class_id") == 0 || value.at("class_id") == 1),
          "Recipe class_id must be integer 0 or 1");
      row.category = value.at("class_id").get<std::int64_t>();
      ++classes[static_cast<std::size_t>(row.category)];
      rows->push_back(row);
    }
    require_recipe(stream.eof() && count == size &&
                       sha256_digest_hex(hash.finish()) == identity.hex,
                   "Recipe data changed while reading");
    require_recipe(rows->size() >= 2 && classes[0] != 0 && classes[1] != 0,
                   "Each recipe split needs both classes");
    manifest[split] = {{"samples", rows->size()},
                       {"bytes", size},
                       {"sha256", identity.hex},
                       {"class_counts", classes}};
    return std::shared_ptr<const std::vector<Row>>(std::move(rows));
  };
  auto train = load("train_file", "train");
  auto validation = load("validation_file", "validation");
  manifest["data_bytes"] = total_bytes;
  const auto batch = config.at("batch_size").get<std::uint64_t>();
  const auto total = ((train->size() + batch - 1) / batch) *
                     config.at("epochs").get<std::uint64_t>();
  require_recipe(total <= 1000000,
                 "Recipe exceeds one million optimizer updates");
  const auto identity = digest(manifest.dump());
  return {std::move(train), std::move(validation), std::move(manifest),
          identity, total};
}

ValueSchema tensor_schema(std::string name, std::int64_t width,
                          const Device& device, Dtype dtype = torch::kFloat32) {
  TensorSchema schema;
  schema.dtype = dtype;
  schema.device = device;
  schema.shape =
      width == 0
          ? std::vector<std::optional<std::int64_t>>{std::nullopt}
          : std::vector<std::optional<std::int64_t>>{std::nullopt, width};
  return ValueSchema::tensor(std::move(schema), std::move(name));
}

std::shared_ptr<data::DataPipeline> make_pipeline(
    const Json& config, std::shared_ptr<const std::vector<Row>> rows,
    bool validation, const ArtifactDigest& identity, const Device& device) {
  data::DataPipelineOptions options;
  options.pipeline_id = validation ? "recipe.validation" : "recipe.training";
  options.dataset_version = identity.hex;
  options.dataset_size = rows->size();
  options.batch_factory_id = std::string(kRecipe);
  options.batch_size = config.at("batch_size").get<std::size_t>();
  options.num_workers = config.at("workers").get<std::size_t>();
  options.prefetch_batches = config.at("prefetch_batches").get<std::size_t>();
  options.worker_seed = config.at("seed").get<std::uint64_t>();
  options.memory.max_bytes = config.at("max_batch_bytes").get<std::uint64_t>();
  options.memory.max_tensors = 4;
  options.memory.max_value_nodes = 8;
  if (device.is_cuda()) {
    options.memory.pin_memory = true;
    options.memory.target_device = device;
    options.memory.non_blocking = true;
  }
  options.schema = {
      .id = std::string(kRecipe),
      .version = 1,
      .value = ValueSchema::tuple(
          {tensor_schema("left", 2, device), tensor_schema("right", 2, device),
           tensor_schema("class_id", 0, device, torch::kInt64),
           tensor_schema("regression", 1, device)})};
  const auto maximum = options.memory.max_bytes;
  return std::make_shared<data::DataPipeline>(
      std::make_shared<data::samplers::RandomSampler>(
          rows->size(), config.at("seed").get<std::int64_t>()),
      options,
      [rows, maximum](std::span<const std::size_t> indices,
                      const data::DataWorkerContext&, std::stop_token stop) {
        require_recipe(indices.size() <= maximum / 28,
                       "Recipe batch exceeds allocation budget");
        if (stop.stop_requested())
          throw Cancelled("Paired recipe data cancelled");
        const auto count = static_cast<std::int64_t>(indices.size());
        auto left = torch::empty({count, 2}, torch::kFloat32);
        auto right = torch::empty({count, 2}, torch::kFloat32);
        auto labels = torch::empty({count}, torch::kInt64);
        auto regression = torch::empty({count, 1}, torch::kFloat32);
        for (std::size_t i = 0; i < indices.size(); ++i) {
          const auto& row = rows->at(indices[i]);
          std::copy(row.left.begin(), row.left.end(),
                    left.data_ptr<float>() + 2 * i);
          std::copy(row.right.begin(), row.right.end(),
                    right.data_ptr<float>() + 2 * i);
          labels.data_ptr<std::int64_t>()[i] = row.category;
          regression.data_ptr<float>()[i] = row.regression;
        }
        return c10::ivalue::Tuple::create({left, right, labels, regression});
      });
}

class PairedModule final : public torch::nn::Module {
 public:
  PairedModule() {
    classifier = register_module("classifier", torch::nn::Linear(4, 2));
    regressor = register_module("regressor", torch::nn::Linear(4, 1));
  }
  c10::IValue forward(const Tensor& left, const Tensor& right) {
    const auto joined = torch::cat({left, right}, 1);
    return c10::ivalue::Tuple::create({classifier(joined), regressor(joined)});
  }

 private:
  torch::nn::Linear classifier{nullptr}, regressor{nullptr};
};

class Checkpoints final : public tr::CallbackHandler {
 public:
  Checkpoints(std::uint64_t every, std::uint64_t pause, std::uint64_t total)
      : every_(every), pause_(pause), total_(total) {
    descriptor_.type_id = "liuding.recipe.paired-affine.checkpoint-v1";
    descriptor_.callback_id = "recipe.checkpoint";
    descriptor_.subscriptions = {tr::CallbackEventKind::batch_finished};
    descriptor_.allowed_commands = {
        tr::CallbackCommandKind::request_checkpoint};
  }
  const tr::CallbackDescriptor& descriptor() const noexcept override {
    return descriptor_;
  }
  tr::CallbackCommandRequest handle(const tr::CallbackEvent& event) override {
    if (event.phase == tr::CallbackPhase::training &&
        (event.global_step % every_ == 0 || event.global_step == pause_ ||
         event.global_step == total_))
      return {tr::CallbackCommandKind::request_checkpoint,
              "paired recipe safe boundary"};
    return {};
  }

 private:
  tr::CallbackDescriptor descriptor_;
  std::uint64_t every_, pause_, total_;
};

std::vector<std::byte> encode_model(const nn::ModelAdapter& model) {
  std::vector<Tensor> owners;
  std::vector<interop::SafeTensorSource> tensors;
  for (const auto& parameter : model.state().named_parameters()) {
    owners.push_back(parameter.value().detach().to(torch::kCPU).contiguous());
    const auto& tensor = owners.back();
    std::vector<std::uint64_t> shape;
    for (const auto extent : tensor.sizes())
      shape.push_back(static_cast<std::uint64_t>(extent));
    tensors.push_back(
        {.name = "parameter." + parameter.key(),
         .dtype = interop::SafeTensorDtype::f32,
         .shape = std::move(shape),
         .data = {static_cast<const std::byte*>(tensor.const_data_ptr()),
                  tensor.nbytes()}});
  }
  return interop::serialize_safetensors(tensors,
                                        {{"recipe", std::string(kRecipe)}});
}

void attach(ex::SQLiteRunStore& store, const std::filesystem::path& path,
            const std::string& id, std::uint64_t time,
            ex::RunArtifactKind kind = ex::RunArtifactKind::artifact) {
  const auto hash = file_digest(path);
  if (const auto previous = store.find_reference(kRun, id)) {
    require_recipe(previous->artifact.digest.hex == hash.hex,
                   "Recipe CAS artifact retry conflict");
    return;
  }
  store.put_artifact_file(
      kRun, id, kind, id, path,
      {.logical_name = path_text(path.filename()),
       .media_type = "application/octet-stream",
       .producer = std::string(kRecipe),
       .expected_size_bytes = std::filesystem::file_size(path)},
      time, hash);
}

Json publish_bundle(const std::filesystem::path& root,
                    const BuildFingerprint& build, const Json& config,
                    const Json& lineage, const Json& evaluation,
                    std::span<const std::byte> weights) {
  interop::ModelBundleManifest manifest;
  manifest.bundle_id = "paired-affine-v1";
  manifest.build = build;
  manifest.files.push_back(
      {.id = "weights",
       .relative_path = "weights.safetensors",
       .role = interop::BundleFileRole::weights,
       .artifact = {.digest = {.hex = sha256_hex(weights)},
                    .size_bytes = weights.size(),
                    .logical_name = "weights",
                    .media_type = "application/safetensors",
                    .producer = std::string(kRecipe)}});
  interop::BundleModel model;
  model.id = "paired-affine";
  model.file_ids = {"weights"};
  const interop::SafeTensorsView view(weights);
  for (const auto& tensor : view.directory().tensors)
    model.tensors.push_back({.name = tensor.name,
                             .dtype = tensor.dtype,
                             .shape = tensor.shape,
                             .device_intent = "cpu",
                             .shard_file_id = "weights"});
  manifest.models.push_back(std::move(model));
  std::filesystem::create_directories(root / "descriptors");
  std::vector<interop::BundlePublishSource> sources{
      {"weights", root / "model.safetensors"}};
  const auto descriptor = [&](const std::string& id,
                              interop::BundleFileRole role, Json value) {
    value["schema_version"] = 1;
    const auto file = root / "descriptors" / (id + ".json");
    write_json(file, value);
    manifest.files.push_back(
        {.id = id,
         .relative_path = "metadata/" + id + ".json",
         .role = role,
         .artifact = {.digest = file_digest(file),
                      .size_bytes = std::filesystem::file_size(file),
                      .logical_name = id,
                      .media_type = "application/json",
                      .producer = std::string(kRecipe)}});
    sources.push_back({id, file});
    return interop::BundleDescriptorRef{
        .schema_id = "liuding.recipe.paired-affine." + id, .file_id = id};
  };
  manifest.input_schema = descriptor(
      "input", interop::BundleFileRole::schema,
      {{"positional",
        {{{"name", "left"}, {"dtype", "float32"}, {"shape", {nullptr, 2}}},
         {{"name", "right"}, {"dtype", "float32"}, {"shape", {nullptr, 2}}}}}});
  manifest.output_schema = descriptor(
      "output", interop::BundleFileRole::schema,
      {{"tuple",
        {{{"name", "logits"}, {"dtype", "float32"}, {"shape", {nullptr, 2}}},
         {{"name", "regression"},
          {"dtype", "float32"},
          {"shape", {nullptr, 1}}}}}});
  manifest.preprocessing =
      descriptor("preprocess", interop::BundleFileRole::prepost,
                 {{"operation", "identity"}, {"feature_range", {-1, 1}}});
  manifest.postprocessing =
      descriptor("postprocess", interop::BundleFileRole::prepost,
                 {{"logits", "argmax axis 1"}, {"regression", "identity"}});
  manifest.labels = descriptor("labels", interop::BundleFileRole::labels,
                               {{"classes", {"negative", "nonnegative"}}});
  manifest.evaluation =
      descriptor("evaluation", interop::BundleFileRole::evaluation, evaluation);
  manifest.sbom = descriptor(
      "sbom", interop::BundleFileRole::sbom,
      {{"complete", false},
       {"scope", "direct declarations; transitive SBOM pending"},
       {"components",
        {{{"name", "liuding-forge"}, {"version", build.jyppx_version}},
         {{"name", "libtorch"}, {"version", build.libtorch_version}}}}});
  manifest.provenance =
      descriptor("provenance", interop::BundleFileRole::provenance,
                 {{"lineage", lineage},
                  {"config", config},
                  {"initialization", "random; self-authored affine model"}});
  auto os_arch = build.os_arch;
  for (auto& value : os_arch)
    if (value >= 'A' && value <= 'Z')
      value = static_cast<char>(value - 'A' + 'a');
  manifest.runtime = {.backend_id = "libtorch.paired-affine",
                      .backend_version = build.jyppx_version,
                      .environment_lock = digest(build_fingerprint_json(build)),
                      .dependencies = {{"libtorch", build.libtorch_version}},
                      .devices = {config.at("device").get<std::string>()},
                      .dtypes = {"float32", "int64"},
                      .os_arch = {os_arch},
                      .dynamic_shapes = true};
  if (std::filesystem::exists(root / "bundle")) {
    require_recipe(
        interop::serialize_model_bundle_manifest(
            interop::verify_model_bundle_directory(root / "bundle").manifest) ==
            interop::serialize_model_bundle_manifest(manifest),
        "Existing paired recipe bundle differs");
  } else
    interop::publish_model_bundle(root / "bundle", manifest, sources);
  return {{"manifest_sha256", file_digest(root / "bundle/manifest.json").hex},
          {"signed", false},
          {"transitive_sbom_complete", false}};
}

Result execute(const Arguments& arguments, bool resume) {
  const auto config = resolve_config(std::filesystem::absolute(
      std::filesystem::u8path(arguments.require("file"))));
  const auto inputs = inspect_inputs(config);
  const Device device(config.at("device") == "cuda" ? "cuda:0" : "cpu");
  require_recipe(!device.is_cuda() || torch::cuda::is_available(),
                 "CUDA requested but unavailable");
  const auto pause = arguments.number("pause-after", 0, inputs.total_steps);
  require_recipe(pause == 0 || pause < inputs.total_steps,
                 "pause-after must precede final update");
  const auto root = std::filesystem::absolute(
                        std::filesystem::u8path(arguments.require("root")))
                        .lexically_normal();
  require_recipe(
      resume ? std::filesystem::is_directory(root)
             : std::filesystem::create_directory(root),
      "Recipe run needs a new directory; resume needs an existing directory");
  RunLock lock(root);
  const auto build = current_build_fingerprint();
  const auto config_hash = digest(config.dump());
  const Json numerics{{"deterministic_algorithms", true},
                      {"cudnn_benchmark", false},
                      {"tf32_cudnn", false},
                      {"tf32_cublas", false},
                      {"float32_matmul_precision", "highest"},
                      {"intraop_threads", 1},
                      {"interop_threads", 1}};
  const Json lineage{{"config_sha256", config_hash.hex},
                     {"dataset_sha256", inputs.identity.hex},
                     {"build", parse_json(build_fingerprint_json(build))},
                     {"recipe", kRecipe},
                     {"numerics", numerics}};
  if (resume) {
    require_recipe(
        parse_json(read_text(root / "lineage.json")) == lineage &&
            parse_json(read_text(root / "config.json")) == config &&
            parse_json(read_text(root / "dataset.json")) == inputs.manifest,
        "Recipe resume rejected changed config, data content or build");
  } else {
    write_json(root / "lineage.json", lineage);
    write_json(root / "config.json", config);
    write_json(root / "dataset.json", inputs.manifest);
  }
  ex::SQLiteRunStore store(root / "run-store");
  if (!resume) {
    const auto time = now_ms();
    store.start({.run_id = std::string(kRun),
                 .project_id = "liuding-forge",
                 .experiment_id = "paired-affine",
                 .display_name = "Paired classification and regression",
                 .status = ex::RunStatus::running,
                 .created_at_unix_ms = time,
                 .updated_at_unix_ms = time,
                 .lineage = {.parent_run_id = {},
                             .source_digest = build.git_commit,
                             .build_fingerprint = build_fingerprint_json(build),
                             .config_digest = config_hash.hex,
                             .data_manifest_digest = inputs.identity.hex},
                 .tags = {}});
  }
  const auto metadata = store.find(kRun);
  require_recipe(
      metadata.has_value() &&
          metadata->lineage.config_digest == config_hash.hex &&
          metadata->lineage.data_manifest_digest == inputs.identity.hex,
      "Recipe run identity inconsistent");
  const auto time = metadata->created_at_unix_ms;
  if (metadata->status == ex::RunStatus::completed) {
    require_recipe(resume && pause == 0,
                   "Completed recipe accepts resume without pause-after");
    const auto saved = store.find_reference(kRun, "result");
    require_recipe(saved && saved->artifact.digest.hex ==
                                file_digest(root / "result.json").hex,
                   "Completed result differs from CAS evidence");
    const auto result = parse_json(read_text(root / "result.json"));
    require_recipe(result.at("bundle").at("manifest_sha256") ==
                       interop::verify_model_bundle_directory(root / "bundle")
                           .manifest_digest.hex,
                   "Completed bundle differs");
    return {result, result.at("quality_passed").get<bool>() ? 0 : 4};
  }
  require_recipe(metadata->status == ex::RunStatus::running,
                 "Recipe cannot resume a terminal run");
  for (const auto* id : {"lineage", "config", "dataset"})
    attach(store, root / (std::string(id) + ".json"), id, time);
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  at::globalContext().setDeterministicAlgorithms(true, false);
  at::globalContext().setDeterministicCuDNN(true);
  at::globalContext().setBenchmarkCuDNN(false);
  at::globalContext().setAllowTF32CuDNN(false);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setFloat32MatmulPrecision("highest");
  require_recipe(!at::globalContext().allowTF32CuBLAS() &&
                     !at::globalContext().allowTF32CuDNN(),
                 "Paired recipe requires TF32 disabled");
  core::RngGuard rng(config.at("seed").get<std::uint64_t>(), device.is_cuda());
  auto train =
      make_pipeline(config, inputs.train, false, inputs.identity, device);
  auto validation =
      make_pipeline(config, inputs.validation, true, inputs.identity, device);
  auto module = std::make_shared<PairedModule>();
  nn::InvocationSchema input_schema;
  input_schema.positional = {tensor_schema("left", 2, device),
                             tensor_schema("right", 2, device)};
  auto model = std::make_shared<nn::ModelAdapter>(nn::adapt_module(
      module,
      [](PairedModule& concrete, const nn::Invocation& input) {
        return concrete.forward(input.positional.at(0).toTensor(),
                                input.positional.at(1).toTensor());
      },
      input_schema,
      ValueSchema::tuple({tensor_schema("logits", 2, device),
                          tensor_schema("regression", 1, device)})));
  std::vector<optim::OptimizerParameterBinding> bindings;
  for (const auto& parameter : model->state().named_parameters())
    bindings.push_back({parameter.key(), parameter.value()});
  auto optimizer = std::make_shared<optim::SgdOptimizerAdapter>(
      "recipe.sgd",
      std::vector<optim::SgdParameterGroup>{
          {.group_id = "main",
           .parameters = std::move(bindings),
           .options = {
               .learning_rate = config.at("learning_rate").get<double>(),
               .momentum = config.at("momentum").get<double>(),
               .weight_decay = config.at("weight_decay").get<double>()}}});
  auto scheduler = std::make_shared<optim::StepLrScheduler>(
      optimizer, optim::StepLrSchedulerOptions{
                     .scheduler_id = "recipe.lr",
                     .step_size = 1,
                     .gamma = config.at("lr_gamma").get<double>()});
  const tr::BatchMapper mapper = [](const data::BatchEnvelope& batch) {
    const auto& values = batch.value.toTuple()->elements();
    return nn::Invocation{.positional = {values.at(0), values.at(1)},
                          .keyword = {}};
  };
  const tr::TargetMapper targets =
      [](const data::BatchEnvelope& batch) -> std::optional<c10::IValue> {
    const auto& values = batch.value.toTuple()->elements();
    return c10::ivalue::Tuple::create({values.at(2), values.at(3)});
  };
  const tr::StepFunction step =
      [weight = config.at("regression_weight").get<double>()](
          const tr::StepInput& input) {
        const auto output = input.model.invoke(input.invocation);
        const auto& values = output.value.toTuple()->elements();
        const auto& target = input.target->toTuple()->elements();
        const auto logits = values.at(0).toTensor();
        const auto mse =
            (values.at(1).toTensor() - target.at(1).toTensor()).square().mean();
        return tr::StepOutput{
            .loss = torch::nn::functional::cross_entropy(
                        logits, target.at(0).toTensor()) +
                    weight * mse,
            .prediction = output,
            .metrics = {{"accuracy", logits.argmax(1)
                                         .eq(target.at(0).toTensor())
                                         .to(torch::kFloat64)
                                         .mean()
                                         .item<double>()},
                        {"mse", mse.item<double>()}},
            .sample_count = input.batch.sample_ids.size()};
      };
  auto diagnostics = std::make_shared<tr::TrainingDiagnostics>();
  std::stop_source pause_stop;
  tr::Trainer trainer({.trainer_id = "recipe.trainer",
                       .run_id = std::string(kRun),
                       .trace_id = "recipe.paired-affine",
                       .max_epochs = config.at("epochs").get<std::uint64_t>()});
  trainer.prepare(
      {.model = model,
       .optimizers =
           {{.optimizer = optimizer,
             .checkpoint =
                 std::make_shared<optim::SgdOptimizerCheckpointParticipant>(
                     optimizer),
             .scheduler = std::make_shared<tr::SchedulerAdapter>(scheduler)}},
       .training_data = train,
       .training_batch_mapper = mapper,
       .training_target_mapper = targets,
       .training_step = step,
       .validation_data = validation,
       .validation_batch_mapper = mapper,
       .validation_target_mapper = targets,
       .validation_step = step,
       .strategy = std::make_shared<tr::SingleDeviceStrategy>(
           tr::SingleDeviceStrategyOptions{.device = device}),
       .callbacks = std::make_shared<tr::CallbackDispatcher>(
           std::vector<std::shared_ptr<tr::CallbackHandler>>{
               std::make_shared<Checkpoints>(
                   config.at("checkpoint_every").get<std::uint64_t>(), pause,
                   inputs.total_steps)}),
       .model_checkpoint =
           std::make_shared<nn::ModelCheckpointParticipant>(*model),
       .rng_checkpoint = std::make_shared<core::RngCheckpointParticipant>(
           core::RngCheckpointParticipantOptions{.include_cuda =
                                                     device.is_cuda()}),
       .diagnostics = diagnostics,
       .checkpoint_action = [&](const tr::CheckpointCoordinator& coordinator) {
         tr::FileCheckpointStore checkpoint(root / "latest.jyck");
         coordinator.save(checkpoint);
         if (pause != 0 && trainer.global_step() == pause)
           pause_stop.request_stop();
       }});
  if (resume) {
    tr::FileCheckpointStore checkpoint(root / "latest.jyck");
    trainer.restore_checkpoint(checkpoint);
    require_recipe(pause == 0 || pause > trainer.global_step(),
                   "pause-after must exceed restored step");
  }
  const auto fit = trainer.fit(pause_stop.get_token());
  if (pause_stop.stop_requested()) {
    require_recipe(
        fit.status == tr::FitStatus::cancelled && fit.global_step == pause,
        "Recipe failed to pause at durable boundary");
    store.heartbeat(kRun, now_ms());
    trainer.close();
    return {Json{{"status", "paused"},
                 {"global_step", fit.global_step},
                 {"lineage", lineage}},
            0};
  }
  require_recipe(fit.status == tr::FitStatus::completed &&
                     fit.global_step == inputs.total_steps,
                 "Recipe did not complete full configured budget");
  tr::FileCheckpointStore final_checkpoint(root / "final.jyck");
  trainer.save_checkpoint(final_checkpoint);
  const auto weights = encode_model(*model);
  const ArtifactDigest model_hash{.hex = sha256_hex(weights)};
  write_file(root / "model.safetensors", weights);
  auto classifier = nn::adapt_module(
      module,
      [](PairedModule& concrete, const nn::Invocation& invocation) {
        return concrete
            .forward(invocation.positional.at(0).toTensor(),
                     invocation.positional.at(1).toTensor())
            .toTuple()
            ->elements()
            .at(0);
      },
      input_schema, tensor_schema("logits", 2, device));
  validation->set_epoch(0);
  const auto scores = tr::evaluate_classification(
      classifier,
      [&](std::stop_token stop)
          -> std::optional<tr::ClassificationEvaluationBatch> {
        const auto batch = validation->next(stop);
        if (!batch) return std::nullopt;
        return tr::ClassificationEvaluationBatch{
            .invocation = mapper(*batch),
            .targets = batch->value.toTuple()->elements().at(2).toTensor()};
      },
      {.num_classes = 2,
       .max_samples = config.at("max_samples").get<std::uint64_t>()});
  double squared_error = 0;
  std::uint64_t regression_samples = 0;
  validation->set_epoch(0);
  {
    torch::NoGradGuard guard;
    while (const auto batch = validation->next()) {
      const auto output = model->invoke(mapper(*batch))
                              .value.toTuple()
                              ->elements()
                              .at(1)
                              .toTensor();
      const auto expected = batch->value.toTuple()->elements().at(3).toTensor();
      squared_error +=
          (output.to(torch::kFloat64) - expected.to(torch::kFloat64))
              .square()
              .sum()
              .item<double>();
      regression_samples += batch->sample_ids.size();
    }
  }
  require_recipe(scores.overall.accuracy &&
                     scores.overall.samples == inputs.validation->size() &&
                     regression_samples == inputs.validation->size(),
                 "Recipe evaluation did not consume full validation data");
  const double mse = squared_error / static_cast<double>(regression_samples);
  require_recipe(std::isfinite(mse), "Recipe regression MSE is nonfinite");
  const auto accuracy = std::find_if(
      scores.observations.begin(), scores.observations.end(),
      [](const auto& point) {
        return point.definition.metric_id == "classification.accuracy" &&
               point.context.at("slice") == "overall" &&
               point.context.at("source") == "all";
      });
  require_recipe(accuracy != scores.observations.end(),
                 "Missing classification accuracy observation");
  const TelemetryMetricPoint mse_point{
      .definition = {.metric_id = "regression.mse",
                     .direction = TelemetryMetricDirection::minimize,
                     .parameters = {{"reduction",
                                     "mean of squared float64 differences"}},
                     .context_keys = {"split"}},
      .step = {.kind = TelemetryStepKind::sample, .value = regression_samples},
      .observation = {.status = TelemetryValueStatus::finite, .value = mse},
      .context = {{"split", "validation"}},
      .sample_count = regression_samples};
  ex::EvaluationSuite suite{
      .suite_id = "recipe.paired-affine",
      .task = "classification-and-regression",
      .dataset_digest = inputs.identity,
      .environment_id = "paired-affine-" + device.str() + "-" + build.os_arch,
      .seed = config.at("seed").get<std::uint64_t>(),
      .rules = {{.rule_id = "validation-accuracy",
                 .definition = accuracy->definition,
                 .context = accuracy->context,
                 .minimum_samples = scores.overall.samples,
                 .minimum = config.at("minimum_accuracy").get<double>(),
                 .maximum = 1.0},
                {.rule_id = "validation-mse",
                 .definition = mse_point.definition,
                 .context = mse_point.context,
                 .minimum_samples = regression_samples,
                 .minimum = 0.0,
                 .maximum = config.at("maximum_mse").get<double>()}}};
  const auto report = ex::evaluate_observations(
      suite,
      {.evaluation_id = "evaluation.paired-affine",
       .run_id = std::string(kRun),
       .model_digest = model_hash,
       .build_fingerprint = build_fingerprint_json(build),
       .adapter_id = std::string(kRecipe),
       .config_digest = config_hash},
      std::array{*accuracy, mse_point});
  const auto quality = parse_json(ex::quality_report_json(report));
  write_file(root / "evaluation-suite.json",
             bytes(ex::evaluation_suite_json(suite)));
  write_json(root / "evaluation.json", quality);
  write_file(root / "diagnostics.json",
             bytes(tr::training_diagnostics_report_json(
                 tr::capture_training_diagnostics_report(
                     *diagnostics,
                     {.report_id = "diagnostic.paired-affine",
                      .run_id = std::string(kRun),
                      .attempt_id = "attempt.final",
                      .model = model_hash,
                      .configuration = config_hash,
                      .dataset = inputs.identity,
                      .checkpoint = file_digest(root / "final.jyck")},
                     tr::TrainingDiagnosticOutcome::completed))));
  require_recipe(inspect_inputs(config).identity.hex == inputs.identity.hex,
                 "Recipe input files changed during training");
  const auto bundle =
      publish_bundle(root, build, config, lineage, quality, weights);
  const auto next_rng = torch::rand({16}, torch::TensorOptions().device(device))
                            .to(torch::kCPU)
                            .contiguous();
  const auto next_cpu_rng = torch::rand({16}).contiguous();
  const bool passed = report.verdict == ex::QualityVerdict::pass;
  Json result{
      {"schema_version", 1},
      {"recipe", kRecipe},
      {"status", "completed"},
      {"run_id", kRun},
      {"global_step", fit.global_step},
      {"completed_epochs", fit.completed_epochs},
      {"optimizer_steps", optimizer->completed_steps()},
      {"scheduler_steps", scheduler->completed_steps()},
      {"training_metrics", fit.training_metrics},
      {"validation_metrics", fit.validation_metrics},
      {"accuracy", *scores.overall.accuracy},
      {"mse", mse},
      {"confusion", scores.overall.confusion},
      {"validation_samples", regression_samples},
      {"quality_passed", passed},
      {"model_sha256", model_hash.hex},
      {"numerics", numerics},
      {"lineage", lineage},
      {"bundle", bundle},
      {"next_rng_sha256",
       sha256_hex({static_cast<const std::byte*>(next_rng.const_data_ptr()),
                   next_rng.nbytes()})},
      {"next_cpu_rng_sha256",
       sha256_hex({static_cast<const std::byte*>(next_cpu_rng.const_data_ptr()),
                   next_cpu_rng.nbytes()})}};
  write_json(root / "result.json", result);
  for (const auto* name :
       {"model.safetensors", "evaluation-suite.json", "evaluation.json",
        "diagnostics.json", "result.json"})
    attach(store, root / name, std::filesystem::path(name).stem().string(),
           time);
  attach(store, root / "final.jyck", "final-checkpoint", time,
         ex::RunArtifactKind::checkpoint);
  attach(store, root / "bundle/manifest.json", "bundle", time);
  store.transition(kRun, ex::RunStatus::completed, now_ms());
  trainer.close();
  return {result, passed ? 0 : 4};
}
}  // namespace

Result paired_recipe_command(const Arguments& arguments) {
  const auto& action = arguments.words().at(1);
  if (action == "schema") {
    require_recipe(!arguments.optional("file") && !arguments.optional("root") &&
                       !arguments.optional("pause-after"),
                   "recipe schema accepts only --name");
    Json fields = Json::object();
    for (const auto& [name, kind] : field_kinds())
      fields[name] = {{"kind", value_kind_name(kind)}};
    return {Json{{"schema_version", 1},
                 {"schema_id", kRecipe},
                 {"version", 1},
                 {"root", {{"kind", "dictionary"}, {"fields", fields}}}},
            0};
  }
  require_recipe(!arguments.optional("name"),
                 "Use --name only with recipe schema");
  if (action == "check") {
    require_recipe(
        !arguments.optional("root") && !arguments.optional("pause-after"),
        "recipe check accepts only --file");
    const auto config = resolve_config(std::filesystem::absolute(
        std::filesystem::u8path(arguments.require("file"))));
    const auto inputs = inspect_inputs(config);
    return {Json{{"valid", true},
                 {"config", config},
                 {"config_sha256", digest(config.dump()).hex},
                 {"dataset", inputs.manifest},
                 {"dataset_sha256", inputs.identity.hex},
                 {"total_steps", inputs.total_steps}},
            0};
  }
  if (action == "run" || action == "resume")
    return execute(arguments, action == "resume");
  usage("recipe", "Choose schema, check, run or resume.");
}
}  // namespace jyppx::liuding::cli
