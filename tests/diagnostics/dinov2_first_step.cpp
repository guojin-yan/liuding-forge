// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

// Explicit, offline diagnostic; never registered as a default CTest or example.
#include "cli.hpp"
#include "dinov2_compiled_sources.hpp"
#include "recipe_support.hpp"
#include "reference_file.hpp"
#include "transfer_recipe.hpp"

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/data_image/batch_transform.hpp>
#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/foundation/random.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>

#include <torch/torch.h>

#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace jyppx::liuding::cli {
// Exact bounded helpers, extracted from cli_main.cpp. This target does not
// link cli_objects, so the normal CLI main and dispatcher are not duplicated.
#include "cli_input_support.inc"

namespace {
namespace fs = std::filesystem;
namespace image = data::image;
using namespace recipe_support;

static_assert(std::endian::native == std::endian::little);
constexpr std::string_view kWeightsSha =
    "f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca68b29db8a68c8c84c9810e5";
constexpr std::string_view kConfigSha =
    "187d72db61fce5ebe8da5d053aa2423f038c6355f35939d4aa0f49e3d5cb0eea";
constexpr std::size_t kParameterCount = 176;

// Exact production inspect_inputs/make_pipeline definitions, without Trainer,
// checkpoint or task-quality logic. Both extractions are source-hash bound by
// CMake and the execution driver, and must be refreshed if their source moves.
#include "recipe_capture_support.inc"

void Require(bool value, std::string_view reason) {
  if (!value) throw std::runtime_error(std::string(reason));
}

Json ReadJson(const fs::path& path) { return parse_json(read_text(path)); }

Json CompiledSources() { return parse_json(kDinoV2CompiledSources); }

std::vector<std::int64_t> Shape(const Tensor& tensor) {
  return {tensor.sizes().begin(), tensor.sizes().end()};
}

std::span<const std::byte> TensorBytes(const Tensor& tensor) {
  Require(tensor.device().is_cpu() && tensor.is_contiguous(),
          "Diagnostic tensor must be packed on CPU");
  return {static_cast<const std::byte*>(tensor.const_data_ptr()),
          tensor.nbytes()};
}

std::string TensorSha(const Tensor& tensor) {
  return sha256_hex(TensorBytes(tensor));
}

Json TensorDescriptor(const Tensor& tensor, std::string_view file) {
  const auto type = tensor.scalar_type();
  Require(type == torch::kFloat32 || type == torch::kInt64,
          "Only raw F32/I64 diagnostic tensors are supported");
  return {{"file", file},
          {"dtype", type == torch::kFloat32 ? "F32" : "I64"},
          {"shape", Shape(tensor)},
          {"size_bytes", tensor.nbytes()},
          {"sha256", TensorSha(tensor)}};
}

Json WriteTensor(const fs::path& root, std::string_view name,
                 const Tensor& tensor) {
  const auto packed = tensor.detach().contiguous();
  write_file(root / fs::u8path(name), TensorBytes(packed));
  return TensorDescriptor(packed, name);
}

Tensor ReadTensor(const fs::path& root, const Json& descriptor,
                  std::string_view file, const std::vector<std::int64_t>& shape,
                  torch::ScalarType type) {
  Require(descriptor.at("file").get<std::string>() == file &&
              descriptor.at("shape").get<std::vector<std::int64_t>>() == shape,
          "Reference tensor identity or complete shape differs");
  Require(descriptor.at("dtype") == (type == torch::kFloat32 ? "F32" : "I64"),
          "Reference tensor dtype differs");
  auto result = torch::empty(shape, torch::TensorOptions().dtype(type));
  Require(result.nbytes() <= 16U * 1024U * 1024U,
          "Single diagnostic tensor exceeds 16 MiB");
  const auto path = root / fs::u8path(file);
  Require(fs::file_size(path) == result.nbytes() &&
              descriptor.at("size_bytes") == result.nbytes(),
          "Reference tensor byte size differs");
  std::ifstream input(path, std::ios::binary);
  input.read(static_cast<char*>(result.mutable_data_ptr()),
             static_cast<std::streamsize>(result.nbytes()));
  Require(input && input.peek() == std::char_traits<char>::eof(),
          "Cannot read complete diagnostic tensor");
  Require(TensorSha(result) == descriptor.at("sha256").get<std::string>(),
          "Diagnostic tensor digest differs");
  return result;
}

std::string ParameterFile(std::string_view phase, std::size_t index) {
  std::ostringstream name;
  name << phase << '/' << std::setw(3) << std::setfill('0') << index << ".f32";
  return name.str();
}

Json LoadContract(const fs::path& path) {
  auto contract = ReadJson(path);
  Require(contract.at("schema_id") ==
                  "jyppx.liuding.dinov2-real-first-step-diagnostic" &&
              contract.at("schema_version") == 1 &&
              contract.at("device") == "cpu" &&
              contract.at("optimizer_updates") == 1 &&
              contract.at("original_config_sha256").get<std::string>() ==
                  kConfigSha,
          "Expected the fixed one-step CPU diagnostic contract");
  const auto config_path =
      fs::u8path(contract.at("original_config_file").get<std::string>());
  Require(file_digest(config_path).hex == kConfigSha &&
              ReadJson(config_path) == contract.at("original_config"),
          "Record 245 original configuration changed");
  const auto& original = contract.at("original_config");
  Require(original.at("seed") == 20261029 && original.at("epochs") == 3 &&
              original.at("batch_size") == 8 &&
              original.at("image_size") == 224 &&
              original.at("minimum_accuracy") == 0.65 &&
              original.at("backbone_mode") == "finetune" &&
              original.at("batch_norm") == "eval" &&
              original.at("model") == "vit_small_patch14_dinov2" &&
              original.at("backbone_learning_rate") == 0.0001 &&
              original.at("head_learning_rate") == 0.01 &&
              original.at("momentum") == 0.9 &&
              original.at("weight_decay") == 0.0001,
          "Frozen data/seed/budget/optimizer fields differ");
  const Json tolerances{{"loss_rtol", 3e-4},     {"loss_atol", 3e-4},
                        {"gradient_rtol", 3e-3}, {"gradient_atol", 3e-5},
                        {"updated_rtol", 3e-4},  {"updated_atol", 3e-5}};
  Require(contract.at("tolerances") == tolerances &&
              contract.at("full_task_updates") == 375 &&
              contract.at("minimum_accuracy") == 0.65,
          "Frozen numerical tolerances or quality gate differ");
  Require(contract.at("compiled_sources") == CompiledSources(),
          "Gate configure-time source bindings differ; reconfigure/rebuild");
  return contract;
}

Json RelocatedConfig(const Json& contract) {
  auto config = contract.at("original_config");
  for (const auto* key :
       {"train_root", "validation_root", "weights_file", "weights_metadata"}) {
    const auto path =
        fs::u8path(contract.at("paths").at(key).get<std::string>());
    Require(path.is_absolute(), "Diagnostic input paths must be absolute");
    config[key] = path_text(path);
  }
  // This is a CPU numerical reproduction of the old grouped experiment.
  // The unchanged original CUDA/full-task JSON remains separately hash bound.
  config["device"] = "cpu";
  return config;
}

void ConfigureCpu() {
  Require(std::string_view(build_info().libtorch_version) == "2.14.1",
          "Expected the fixed native LibTorch 2.14.1 build");
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  at::globalContext().setDeterministicAlgorithms(true, false);
  at::globalContext().setDeterministicCuDNN(true);
  at::globalContext().setBenchmarkCuDNN(false);
  at::globalContext().setAllowTF32CuDNN(false);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setFloat32MatmulPrecision("highest");
}

struct InitialState {
  std::unique_ptr<core::RngGuard> rng;
  std::shared_ptr<nn::ModelAdapter> model;
  std::map<std::string, Tensor> parameters;
  image::ImageBatch batch;
  Json dataset;
  Json first_batch;
  Json inventory;
};

InitialState Prepare(const Json& contract) {
  const auto config = RelocatedConfig(contract);
  auto inputs = inspect_inputs(config);
  Require(inputs.total_steps == 375 &&
              inputs.manifest == contract.at("expected_dataset"),
          "Complete task dataset content or 375-update budget differs");
  const auto transfer = inspect_transfer_input(config);
  Require(transfer.record.artifact->digest.hex == kWeightsSha &&
              transfer.record.artifact->size_bytes == 88240512 &&
              transfer.descriptor.at("position_antialias") == false,
          "Expected the existing fixed official DINOv2-S converted weights");
  const auto declared_descriptor =
      contract.at("weights_metadata_sha256").get<std::string>();
  Require(
      file_digest(fs::u8path(config.at("weights_metadata").get<std::string>()))
              .hex == declared_descriptor,
      "Weight input descriptor changed");
  auto rng = std::make_unique<core::RngGuard>(
      config.at("seed").get<std::uint64_t>(), false);
  image::ImageResourceBudgetLimits limits;
  const auto maximum = config.at("image_budget_bytes").get<std::uint64_t>();
  limits.max_encoded_bytes = maximum;
  limits.max_decoded_bytes = maximum;
  limits.max_codec_scratch_bytes = maximum;
  limits.max_pinned_bytes = maximum;
  limits.max_gpu_bytes = maximum;
  limits.max_pipeline_inflight_bytes = maximum;
  limits.max_pipeline_inflight_batches =
      config.at("prefetch_batches").get<std::uint64_t>() + 2;
  auto budget = std::make_shared<image::ImageResourceBudget>(limits);
  // Preserve the recipe's initialization order, including both exemplars.
  auto training = make_pipeline(config, inputs.train, false, inputs.identity,
                                Device(torch::kCPU), budget, &transfer);
  auto validation =
      make_pipeline(config, inputs.validation, true, inputs.identity,
                    Device(torch::kCPU), budget, &transfer);
  auto model = make_transfer_model(transfer, config, 10);
  model->state().to(Device(torch::kCPU));
  model->state().train();
  Require(model->state().is_training() && model->state().buffers().empty(),
          "DINOv2 finetune must train with no persistent buffers");
  std::map<std::string, Tensor> parameters;
  for (const auto& row : model->state().named_parameters()) {
    Require(row.value().requires_grad() &&
                row.value().scalar_type() == torch::kFloat32 &&
                row.value().device().is_cpu() && row.value().is_contiguous(),
            "Expected every native parameter trainable, contiguous F32 CPU");
    Require(parameters.emplace(row.key(), row.value()).second,
            "Duplicate parameter identity");
  }
  Require(parameters.size() == kParameterCount, "Expected exactly 176 groups");
  Json inventory = Json::array();
  {
    models::test::MappedReference mapped(transfer.path, 88240512);
    Require(sha256_hex(mapped) == kWeightsSha, "Mapped weights changed");
    const interop::SafeTensorsView weights(mapped);
    Require(weights.directory().tensors.size() == 174,
            "Expected the complete 174-tensor unmasked backbone");
    for (const auto& [name, parameter] : parameters) {
      if (name.starts_with("backbone.")) {
        const auto source = weights.tensor_bytes(name.substr(9));
        const auto actual = TensorBytes(parameter);
        Require(source.size() == actual.size() &&
                    std::equal(source.begin(), source.end(), actual.begin()),
                "Native initial backbone differs from fixed weight bytes");
      } else {
        Require(name == "head.weight" || name == "head.bias",
                "Unexpected parameter outside backbone/head");
      }
      inventory.push_back(
          {{"name", name},
           {"shape", Shape(parameter)},
           {"dtype", "F32"},
           {"elements", parameter.numel()},
           {"size_bytes", parameter.nbytes()},
           {"initial_sha256", TensorSha(parameter)},
           {"learning_rate", name.starts_with("head.") ? 0.01 : 0.0001}});
    }
  }
  training->set_epoch(0);
  const auto envelope = training->next();
  Require(envelope.has_value() && envelope->sample_ids.size() == 8 &&
              envelope->epoch == 0 && envelope->sequence == 0,
          "Expected the first real batch in epoch zero");
  auto batch = image::image_batch_from_value(envelope->value);
  Require(Shape(batch.images) == std::vector<std::int64_t>({8, 3, 224, 224}) &&
              Shape(batch.labels) == std::vector<std::int64_t>({8}) &&
              batch.items.size() == 8 &&
              batch.images.scalar_type() == torch::kFloat32 &&
              batch.images.device().is_cpu() &&
              batch.labels.device().is_cpu() &&
              batch.labels.scalar_type() == torch::kInt64 &&
              torch::isfinite(batch.images).all().item<bool>(),
          "Unexpected real batch shape, labels, dtype or nonfinite pixels");
  Json selected = Json::array();
  for (std::size_t item = 0; item < 8; ++item) {
    const auto index = envelope->sample_ids[item];
    Require(index < inputs.train.records().size(), "Invalid source index");
    const auto& source =
        inputs.train.records()[static_cast<std::size_t>(index)];
    Require(source.labels.size() == 1 &&
                batch.items[item].image_path == source.path &&
                batch.labels[static_cast<std::int64_t>(item)]
                        .item<std::int64_t>() ==
                    static_cast<std::int64_t>(source.labels[0]),
            "Actual batch identity and labels differ from indexed source");
    const auto file = fs::u8path(config.at("train_root").get<std::string>()) /
                      fs::u8path(source.path);
    selected.push_back(
        {{"index", index},
         {"relative_path", source.path},
         {"class", inputs.train.categories()
                       .at(static_cast<std::size_t>(source.labels[0]))
                       .name},
         {"label", source.labels[0]},
         {"sha256", file_digest(file).hex}});
  }
  Require(selected == contract.at("expected_first_batch"),
          "First batch differs from the eight real samples in record 245");
  training->close();
  validation->close();
  return {std::move(rng),      std::move(model), std::move(parameters),
          std::move(batch),    inputs.manifest,  std::move(selected),
          std::move(inventory)};
}

Json CompareTensor(const Tensor& actual, const Tensor& expected,
                   double relative, double absolute) {
  Require(actual.sizes() == expected.sizes() &&
              actual.scalar_type() == torch::kFloat32 &&
              expected.scalar_type() == torch::kFloat32 &&
              torch::isfinite(actual).all().item<bool>() &&
              torch::isfinite(expected).all().item<bool>(),
          "Comparison has invalid shape/dtype or nonfinite values");
  const auto packed = actual.detach().contiguous();
  const auto mask =
      torch::isclose(packed, expected, relative, absolute, false).contiguous();
  const auto* lhs = packed.const_data_ptr<float>();
  const auto* rhs = expected.const_data_ptr<float>();
  const auto* matches = mask.const_data_ptr<bool>();
  std::uint64_t failed = 0;
  double max_error = 0, error_square = 0, expected_square = 0;
  Json samples = Json::array();
  for (std::int64_t index = 0; index < packed.numel(); ++index) {
    const auto a = static_cast<double>(lhs[index]);
    const auto b = static_cast<double>(rhs[index]);
    const auto error = std::abs(a - b);
    max_error = std::max(max_error, error);
    error_square += error * error;
    expected_square += b * b;
    if (!matches[index]) {
      ++failed;
      if (samples.size() < 8)
        samples.push_back({{"flat_index", index},
                           {"native", a},
                           {"reference", b},
                           {"absolute_error", error}});
    }
  }
  return {{"passed", failed == 0},
          {"elements", packed.numel()},
          {"failed_elements", failed},
          {"max_abs_error", max_error},
          {"error_l2", std::sqrt(error_square)},
          {"reference_l2", std::sqrt(expected_square)},
          {"native_sha256", TensorSha(packed)},
          {"reference_sha256", TensorSha(expected)},
          {"first_failed_elements", samples}};
}

void Capture(const fs::path& contract_path, const fs::path& root) {
  const auto contract = LoadContract(contract_path);
  Require(root.is_absolute() && !fs::exists(root),
          "Capture directory must be a new absolute path");
  fs::create_directory(root);
  auto state = Prepare(contract);
  const auto input = WriteTensor(root, "input.f32", state.batch.images);
  const auto labels = WriteTensor(root, "labels.i64", state.batch.labels);
  const auto head_weight =
      WriteTensor(root, "head-weight.f32", state.parameters.at("head.weight"));
  const auto head_bias =
      WriteTensor(root, "head-bias.f32", state.parameters.at("head.bias"));
  write_json(
      root / "capture.json",
      {{"schema_version", 1},
       {"status", "zero_step_captured"},
       {"device", "cpu"},
       {"seed", 20261029},
       {"contract_sha256", file_digest(contract_path).hex},
       {"compiled_sources", CompiledSources()},
       {"build",
        parse_json(build_fingerprint_json(current_build_fingerprint()))},
       {"original_config_sha256", kConfigSha},
       {"weights_sha256", kWeightsSha},
       {"dataset", state.dataset},
       {"first_batch", state.first_batch},
       {"parameters", state.inventory},
       {"input", input},
       {"labels", labels},
       {"head_weight", head_weight},
       {"head_bias", head_bias},
       {"optimizer_updates", 0},
       {"head_initialization", "direct native capture before forward/SGD"},
       {"task_accuracy_gate_closed", false},
       {"full_v1_0_complete", false}});
  std::cout
      << "Captured exact zero-step head and actual first batch; no update.\n";
}

bool Compare(const fs::path& contract_path, const fs::path& capture_root,
             const fs::path& reference_root, const fs::path& report_path) {
  const auto contract = LoadContract(contract_path);
  const auto capture = ReadJson(capture_root / "capture.json");
  const auto reference = ReadJson(reference_root / "reference.json");
  Require(report_path.is_absolute() && !fs::exists(report_path),
          "Comparison report must be a new absolute file path");
  Require(
      capture.at("contract_sha256") == file_digest(contract_path).hex &&
          reference.at("contract_sha256") == file_digest(contract_path).hex &&
          reference.at("capture_sha256") ==
              file_digest(capture_root / "capture.json").hex &&
          reference.at("initial_state_verified") == true &&
          reference.at("input") == capture.at("input") &&
          reference.at("labels") == capture.at("labels") &&
          reference.at("parameters").size() == kParameterCount &&
          reference.at("optimizer_updates") == 1,
      "Reference/capture provenance or parameter coverage differs");
  auto state = Prepare(contract);
  Require(state.inventory == capture.at("parameters") &&
              state.first_batch == capture.at("first_batch"),
          "Reconstructed zero-step state or real batch differs");
  for (const auto& [key, file, tensor] :
       std::vector<std::tuple<std::string, std::string, Tensor>>{
           {"input", "input.f32", state.batch.images},
           {"labels", "labels.i64", state.batch.labels},
           {"head_weight", "head-weight.f32",
            state.parameters.at("head.weight")},
           {"head_bias", "head-bias.f32", state.parameters.at("head.bias")}}) {
    const auto captured = ReadTensor(capture_root, capture.at(key), file,
                                     Shape(tensor), tensor.scalar_type());
    Require(TensorSha(tensor) == TensorSha(captured),
            "Capture and comparison inputs/head are not byte-identical");
  }
  std::vector<optim::SgdParameterGroup> groups(2);
  groups[0].group_id = "backbone";
  groups[1].group_id = "head";
  for (std::size_t index = 0; index < groups.size(); ++index) {
    groups[index].options.learning_rate = index == 0 ? 0.0001 : 0.01;
    groups[index].options.momentum = 0.9;
    groups[index].options.weight_decay = 0.0001;
  }
  for (const auto& [name, parameter] : state.parameters)
    groups[name.starts_with("head.") ? 1U : 0U].parameters.push_back(
        {name, parameter});
  optim::SgdOptimizerAdapter optimizer("diagnostic.sgd", std::move(groups));
  optimizer.zero_grad();
  const auto logits =
      state.model->invoke({.positional = {state.batch.images}, .keyword = {}})
          .value.toTensor();
  const auto loss =
      torch::nn::functional::cross_entropy(logits, state.batch.labels);
  const auto expected_logits =
      ReadTensor(reference_root, reference.at("logits"), "logits.f32", {8, 10},
                 torch::kFloat32);
  const auto expected_loss = ReadTensor(reference_root, reference.at("loss"),
                                        "loss.f32", {1}, torch::kFloat32);
  const auto loss_check =
      CompareTensor(loss.reshape({1}), expected_loss, 3e-4, 3e-4);
  const auto logits_check = CompareTensor(logits, expected_logits, 3e-4, 3e-4);
  loss.backward();
  Json checks = Json::array();
  std::size_t index = 0, gradients_passed = 0, updates_passed = 0;
  for (const auto& [name, parameter] : state.parameters) {
    const auto& row = reference.at("parameters").at(index);
    Require(row.at("name") == name && row.at("shape") == Shape(parameter) &&
                row.at("initial_sha256") ==
                    state.inventory.at(index).at("initial_sha256") &&
                parameter.grad().defined(),
            "Reference parameter mapping, initial bytes or native gradient "
            "missing");
    const auto expected = ReadTensor(reference_root, row.at("gradient"),
                                     ParameterFile("gradient", index),
                                     Shape(parameter), torch::kFloat32);
    const auto gradient_check =
        CompareTensor(parameter.grad(), expected, 3e-3, 3e-5);
    if (gradient_check.at("passed").get<bool>()) ++gradients_passed;
    checks.push_back({{"name", name},
                      {"shape", Shape(parameter)},
                      {"gradient", gradient_check}});
    ++index;
  }
  optimizer.step();
  Require(optimizer.completed_steps() == 1,
          "Expected exactly one native SGD update");
  index = 0;
  for (const auto& [name, parameter] : state.parameters) {
    const auto& row = reference.at("parameters").at(index);
    const auto expected = ReadTensor(reference_root, row.at("updated"),
                                     ParameterFile("updated", index),
                                     Shape(parameter), torch::kFloat32);
    const auto update_check = CompareTensor(parameter, expected, 3e-4, 3e-5);
    if (update_check.at("passed").get<bool>()) ++updates_passed;
    checks.at(index)["updated"] = update_check;
    ++index;
  }
  const auto passed = gradients_passed == kParameterCount &&
                      updates_passed == kParameterCount &&
                      loss_check.at("passed").get<bool>() &&
                      logits_check.at("passed").get<bool>();
  write_json(
      report_path,
      {{"schema_version", 1},
       {"status", passed ? "passed" : "failed"},
       {"device", "cpu"},
       {"contract_sha256", file_digest(contract_path).hex},
       {"capture_sha256", file_digest(capture_root / "capture.json").hex},
       {"reference_sha256", file_digest(reference_root / "reference.json").hex},
       {"compiled_sources", CompiledSources()},
       {"parameters_compared", kParameterCount},
       {"gradients_passed", gradients_passed},
       {"updates_passed", updates_passed},
       {"comparison", "every element of every gradient and updated parameter"},
       {"loss", loss_check},
       {"logits", logits_check},
       {"parameter_checks", checks},
       {"tolerances", contract.at("tolerances")},
       {"optimizer_updates", 1},
       {"full_task_updates", 375},
       {"minimum_accuracy", 0.65},
       {"task_accuracy_gate_closed", false},
       {"full_v1_0_complete", false}});
  std::cout << "Full-gradient groups: " << gradients_passed
            << "/176; full-update groups: " << updates_passed
            << "/176; numerical gate: " << (passed ? "passed" : "failed")
            << "; task-quality gate remains open.\n";
  return passed;
}
}  // namespace

int FirstStepMain(std::span<const std::filesystem::path> arguments) {
  if (arguments.size() == 1 && arguments[0] == "--help") {
    std::cout << "Capture: --capture CONTRACT NEW_CAPTURE_DIRECTORY\n"
                 "Compare: --compare CONTRACT CAPTURE_DIRECTORY "
                 "REFERENCE_DIRECTORY NEW_REPORT\n";
    return 0;
  }
  ConfigureCpu();
  if (arguments.size() == 3 && arguments[0] == "--capture") {
    Capture(arguments[1], arguments[2]);
    return 0;
  }
  if (arguments.size() == 5 && arguments[0] == "--compare")
    return Compare(arguments[1], arguments[2], arguments[3], arguments[4]) ? 0
                                                                           : 4;
  throw std::runtime_error("Invalid diagnostic arguments; see --help");
}
}  // namespace jyppx::liuding::cli

#if defined(_WIN32)
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  try {
    const std::vector<std::filesystem::path> arguments(argv + 1, argv + argc);
    return jyppx::liuding::cli::FirstStepMain(arguments);
  } catch (const std::exception& error) {
    std::cerr << "DINOv2 first-step diagnostic incomplete: " << error.what()
              << '\n';
    return 2;
  }
}
