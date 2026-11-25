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
