// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/training/ema.hpp>
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>

#include <torch/cuda.h>

#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <nlohmann/json.hpp>

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;
using J = nlohmann::json;
J Reference() {
  J result;
  std::ifstream(std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) /
                "vision/ema_original.json") >>
      result;
  return result;
}
Tensor Value(const J& field) {
  const auto type = field["dtype"].get<std::string>();
  if (type == "torch.float32")
    return torch::tensor(field["values"].get<std::vector<float>>());
  if (type == "torch.float64")
    return torch::tensor(field["values"].get<std::vector<double>>(),
                         torch::kFloat64);
  if (type == "torch.int64")
    return torch::tensor(field["values"].get<std::int64_t>());
  return torch::tensor(field["values"].get<bool>());
}
class EmaFixtureState final : public StateReader, public StateWriter {
 public:
  explicit EmaFixtureState(const t::ModelEma& ema)
      : descriptor(ema.describe_state()) {
    ema.save_state(*this);
  }
  const StateManifest& manifest() const noexcept override { return descriptor; }
  bool contains(std::string_view name) const override {
    return blobs.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max) const override {
    const auto& result = blobs.at(std::string(name));
    if (result.size() > max) throw ResourceExhausted("fixture read limit");
    return result;
  }
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    blobs.emplace(std::string(name),
                  std::vector<std::byte>(data.begin(), data.end()));
  }
  StateManifest descriptor;
  std::map<std::string, std::vector<std::byte>> blobs;
};
struct Domain {
  std::shared_ptr<nn::ModelAdapter> source, shadow;
  std::unique_ptr<t::ModelEma> ema;
  explicit Domain(Device device = Device(torch::kCPU),
                  t::ModelEmaUpdateRule rule =
                      t::ModelEmaUpdateRule::multiply_add_floating_only) {
    const auto reference = Reference();
    auto make = [&] {
      auto module = std::make_shared<torch::nn::Module>();
      for (const auto& [key, field] : reference["initial"].items()) {
        auto value = Value(field).to(device);
        if (key == "weight")
          module->register_parameter(key, value);
        else
          module->register_buffer(key, value);
      }
      auto state = std::make_shared<nn::ModelStateView>(module);
      return std::make_shared<nn::ModelAdapter>(
          state,
          [module](const nn::Invocation&) -> c10::IValue {
            return module->named_parameters()["weight"];
          },
          nn::InvocationSchema{}, ValueSchema::tensor());
    };
    source = make();
    shadow = make();
    t::ModelEmaOptions options;
    options.update_rule = rule;
    ema = std::make_unique<t::ModelEma>(source, shadow, options);
  }
  void step(const J& row) {
    torch::NoGradGuard guard;
    for (const auto& [key, field] : row["source"].items()) {
      auto target = key == "weight" ? source->state().named_parameters()[key]
                                    : source->state().named_buffers()[key];
      target.copy_(Value(field).to(target.device()));
    }
    ema->on_optimizer_update(ema->completed_updates() + 1);
  }
  void compare(const J& expected) {
    for (const auto& [key, field] : expected.items()) {
      const auto actual = key == "weight"
                              ? shadow->state().named_parameters()[key]
                              : shadow->state().named_buffers()[key];
      EXPECT_TRUE(torch::equal(actual.cpu(), Value(field))) << key;
    }
  }
};
void OriginalReference(Device device) {
  torch::set_num_threads(1);
  const auto reference = Reference();
  Domain domain(device);
  EXPECT_DOUBLE_EQ(domain.ema->current_decay(), 0);
  for (const auto& row : reference["updates"]) {
    domain.step(row);
    domain.compare(row["expected"]);
    EXPECT_NEAR(domain.ema->current_decay(), row["decay"].get<double>(), 1e-16);
    EXPECT_FALSE(domain.shadow->state().is_training());
    EXPECT_FALSE(
        domain.shadow->state().named_parameters()["weight"].requires_grad());
  }
  EXPECT_EQ(domain.ema->completed_updates(), 6U);
}
TEST(VisionEma, OriginalCpuReferencePreservesInitialIntegralState) {
  OriginalReference(Device(torch::kCPU));
}
TEST(VisionEma, OriginalCudaReferencePreservesInitialIntegralState) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA/ROCm required";
  OriginalReference(Device(torch::kCUDA, 0));
}
TEST(VisionEma, BatchedCudaUpdatesMatchScalarBytesAcrossDtypesAndChunks) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA/ROCm required";
  const torch::NoGradGuard no_grad;
  for (const auto rule : {t::ModelEmaUpdateRule::lerp_copy_integral,
                          t::ModelEmaUpdateRule::multiply_add_floating_only}) {
    const auto make = [] {
      auto module = std::make_shared<torch::nn::Module>();
      for (int index = 0; index < 265; ++index) {
        // Include odd tails, scalar buffers, both floating groups, >128
        // entries and products exceeding the normal 4 MiB batch bound.
        const auto count = index < 2 ? 1048583 : 3 + (index % 17) * 101;
        auto value =
            (torch::arange(count, torch::kFloat64).remainder(97) - 48) / 31.7;
        value = value.to(Device(torch::kCUDA, 0),
                         index % 2 ? torch::kFloat64 : torch::kFloat32);
        module->register_parameter("p" + std::to_string(index), value);
      }
      module->register_buffer("scalar", torch::tensor(1.234567f).cuda());
      module->register_buffer("counter", torch::tensor(std::int64_t{7}).cuda());
      module->register_buffer("flag", torch::tensor(true).cuda());
      module->register_buffer("cpu", torch::tensor({.3125, -.777}));
      auto state = std::make_shared<nn::ModelStateView>(module);
      return std::make_shared<nn::ModelAdapter>(
          state, [](const nn::Invocation&) -> c10::IValue { return Tensor{}; },
          nn::InvocationSchema{}, ValueSchema::tensor());
    };
    auto source = make(), shadow = make();
    t::ModelEma ema(source, shadow,
                    {.decay = .9999, .ramp_updates = 2.7, .update_rule = rule});
    std::vector<Tensor> live, expected, actual;
    const auto collect = [&](const auto& from, const auto& to) {
      for (const auto& item : from) {
        live.push_back(item.value());
        actual.push_back(to[item.key()]);
        expected.push_back(item.value().clone());
      }
    };
    collect(source->state().named_parameters(),
            shadow->state().named_parameters());
    collect(source->state().named_buffers(), shadow->state().named_buffers());
    for (std::uint64_t step = 1; step <= 12; ++step) {
      for (std::size_t index = 0; index < live.size(); ++index) {
        if (live[index].is_floating_point())
          live[index].mul_(1.00003).add_(.0031 * step);
        else
          live[index].fill_(step % 2);
      }
      ema.on_optimizer_update(step);
      const auto decay = ema.current_decay();
      for (std::size_t index = 0; index < live.size(); ++index) {
        if (live[index].is_floating_point()) {
          if (rule == t::ModelEmaUpdateRule::multiply_add_floating_only) {
            expected[index].mul_(decay);
            expected[index].add_(live[index] * (1.0 - decay));
          } else {
            expected[index].lerp_(live[index], 1.0 - decay);
          }
        } else if (rule == t::ModelEmaUpdateRule::lerp_copy_integral) {
          expected[index].copy_(live[index]);
        }
        EXPECT_TRUE(torch::equal(
            actual[index].cpu().reshape({-1}).view(torch::kUInt8),
            expected[index].cpu().reshape({-1}).view(torch::kUInt8)))
            << "rule=" << static_cast<int>(rule) << " step=" << step
            << " tensor=" << index;
      }
    }
    const EmaFixtureState before(ema);
    auto restored_shadow = make();
    t::ModelEma restored(
        source, restored_shadow,
        {.decay = .9999, .ramp_updates = 2.7, .update_rule = rule});
    restored.load_state(before, RestorePolicy::exact_only);
    EXPECT_EQ(EmaFixtureState(restored).blobs, before.blobs);
    source->state().module().register_buffer("late", torch::ones({1}));
    EXPECT_THROW(ema.on_optimizer_update(13), CompatibilityError);
    EXPECT_EQ(ema.completed_updates(), 12U);
    for (std::size_t index = 0; index < actual.size(); ++index)
      EXPECT_TRUE(torch::equal(
          actual[index].cpu().reshape({-1}).view(torch::kUInt8),
          expected[index].cpu().reshape({-1}).view(torch::kUInt8)));
  }
}
TEST(VisionEma, RestoreContinuesExactlyAndRejectsDifferentRuleTransactionally) {
  const auto rows = Reference()["updates"];
  Domain original;
  for (int i = 0; i < 3; ++i) original.step(rows[i]);
  EmaFixtureState snapshot(*original.ema);
  Domain restored;
  restored.ema->load_state(snapshot, RestorePolicy::exact_only);
  EXPECT_EQ(restored.ema->completed_updates(), 3U);
  Domain legacy(Device(torch::kCPU), t::ModelEmaUpdateRule::lerp_copy_integral);
  const EmaFixtureState before(*legacy.ema);
  EXPECT_THROW(legacy.ema->load_state(snapshot, RestorePolicy::exact_only),
               CompatibilityError);
  const EmaFixtureState after(*legacy.ema);
  EXPECT_EQ(before.blobs, after.blobs);
  for (int i = 3; i < 6; ++i) {
    original.step(rows[i]);
    restored.step(rows[i]);
    const EmaFixtureState a(*original.ema), b(*restored.ema);
    EXPECT_EQ(a.blobs, b.blobs);
  }
  restored.compare(rows[5]["expected"]);
}
TEST(VisionEma,
     DefaultRuleKeepsPublishedStateAndInvalidRuleDoesNotMutateShadow) {
  Domain legacy(Device(torch::kCPU), t::ModelEmaUpdateRule::lerp_copy_integral);
  legacy.step(Reference()["updates"][0]);
  EXPECT_EQ(
      legacy.shadow->state().named_buffers()["counter"].item<std::int64_t>(),
      1);
  const EmaFixtureState state(*legacy.ema);
  const auto& bytes = state.blobs.at("ema_recipe");
  const auto json = J::parse(
      std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  EXPECT_EQ(json, (J{{"schema", "vision.model-ema/v1"},
                     {"source_model_id", "model.main"},
                     {"decay", .9999},
                     {"ramp_updates", 2000.},
                     {"floating", "ramped-fp32-fp64"},
                     {"integral", "copy"}}));
  const auto previous =
      legacy.shadow->state().named_parameters()["weight"].clone();
  t::ModelEmaOptions invalid;
  invalid.update_rule = static_cast<t::ModelEmaUpdateRule>(255);
  EXPECT_THROW(t::ModelEma(legacy.source, legacy.shadow, invalid),
               InvalidArgument);
  EXPECT_TRUE(torch::equal(
      previous, legacy.shadow->state().named_parameters()["weight"]));
}
TEST(VisionEma, RecipeBindsOnlyExplicitUpdateRuleAndRejectsUnknownValues) {
  t::RecipeSpec recipe;
  recipe.training.source = std::filesystem::absolute("fixture.json");
  const auto json = t::recipe_json(recipe), id = t::recipe_identity(recipe);
  EXPECT_EQ(json.find("update_rule"), std::string::npos);
  recipe.ema_update_rule = t::ModelEmaUpdateRule::multiply_add_floating_only;
  const auto roundtrip =
      t::parse_recipe(t::recipe_json(recipe), std::filesystem::current_path());
  EXPECT_EQ(roundtrip.ema_update_rule, recipe.ema_update_rule);
  EXPECT_EQ(t::recipe_identity(roundtrip), t::recipe_identity(recipe));
  EXPECT_NE(id, t::recipe_identity(recipe));
  recipe.ema_update_rule = t::ModelEmaUpdateRule::lerp_copy_integral;
  EXPECT_EQ(t::recipe_json(recipe), json);
  for (const auto& value : {"null", "1", "true", "[]", "\"other\""}) {
    auto bad = J::parse(json);
    bad["ema"]["update_rule"] = J::parse(value);
    EXPECT_THROW(t::parse_recipe(bad.dump(), std::filesystem::current_path()),
                 InvalidArgument);
  }
}
}  // namespace
}  // namespace jyppx::liuding::vision::tests
