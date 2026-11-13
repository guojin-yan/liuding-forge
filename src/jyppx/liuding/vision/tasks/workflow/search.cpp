// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "artifact_io.hpp"
#include "run_ledger.hpp"

#include <jyppx/liuding/data_image/sample_random.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/workflow/search.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
namespace jyppx::liuding::vision::tasks {
namespace {
using J = nlohmann::json;
double& Continuous(RecipeSpec& r, const std::string& name) {
  if (name == "learning_rate") return r.optimizer.learning_rate;
  if (name == "momentum") return r.optimizer.momentum;
  if (name == "weight_decay") return r.optimizer.weight_decay;
  if (name == "label_smoothing") return r.label_smoothing;
  if (name == "final_lr_factor") return r.scheduler.final_factor;
  if (name == "warmup_start_factor") return r.scheduler.warmup_start_factor;
  auto& a = r.data.augmentation;
  if (name == "horizontal_flip") return a.horizontal_flip_probability;
  if (name == "vertical_flip") return a.vertical_flip_probability;
  if (name == "scale") return a.spatial.scale_gain;
  if (name == "translate") return a.spatial.translate_fraction;
  if (name == "hsv_h") return a.photo.hue_shift;
  if (name == "hsv_s") return a.photo.saturation_gain;
  if (name == "hsv_v") return a.photo.value_gain;
  if (name == "grayscale") return a.photo.grayscale_probability;
  if (name == "blur") return a.photo.blur_probability;
  if (name == "median_blur") return a.photo.median_blur_probability;
  if (name == "clahe") return a.photo.clahe_probability;
  if (name == "mosaic") return a.multi_source.mosaic_probability;
  if (name == "mixup") return a.multi_source.mixup_probability;
  if (name == "cutmix") return a.multi_source.cutmix_probability;
  if (name == "copypaste") return a.multi_source.copypaste_probability;
  if (name == "degrees") return a.projective.degrees;
  if (name == "shear") return a.projective.shear_degrees;
  if (name == "perspective") return a.projective.perspective;
  auto& c = r.data.classification;
  if (name == "classification.crop_min_area") return c.crop_min_area;
  if (name == "classification.brightness") return c.brightness;
  if (name == "classification.contrast") return c.contrast;
  if (name == "classification.saturation") return c.saturation;
  if (name == "classification.randaugment_magnitude")
    return c.randaugment_magnitude;
  if (name == "classification.erasing") return c.erasing_probability;
  if (name == "classification.mixup") return c.mixup_probability;
  if (name == "classification.cutmix") return c.cutmix_probability;
  throw InvalidArgument("Unsupported vision search field: " + name);
}
double Field(RecipeSpec& r, const std::string& name) {
  if (name == "close_epochs")
    return static_cast<double>(r.data.augmentation.multi_source.close_epochs);
  if (name == "classification.randaugment_operations")
    return r.data.classification.randaugment_operations;
  return Continuous(r, name);
}
void Set(RecipeSpec& r, const std::string& name, double value) {
  if (name == "close_epochs" ||
      name == "classification.randaugment_operations") {
    if (!std::isfinite(value) || value < 0 || value > 1000000 ||
        std::floor(value) != value)
      throw InvalidArgument(
          "Search integer field requires an exact bounded nonnegative integer");
    if (name == "close_epochs")
      r.data.augmentation.multi_source.close_epochs =
          static_cast<std::uint64_t>(value);
    else
      r.data.classification.randaugment_operations =
          static_cast<std::uint32_t>(value);
  } else
    Continuous(r, name) = value;
}
void Publish(const std::filesystem::path& p, const J& value) {
  detail::publish_text(p, value.dump(2));
}
J ReadJson(const std::filesystem::path& p) {
  auto bytes = detail::read_artifact(p, 64U << 20U);
  return J::parse(bytes.begin(), bytes.end());
}
RecipeSpec Apply(const RecipeSpec& base, const J& candidate) {
  auto r = base;
  for (const auto& p : candidate.items())
    Set(r, p.key(), p.value().get<double>());
  check_recipe(r);
  static_cast<void>(describe_binding(r.model));
  return r;
}
std::string Identity(const RecipeSpec& base, const SearchOptions& o) {
  J space = J::array();
  for (const auto& p : o.space)
    space.push_back({p.field, static_cast<int>(p.distribution), p.lower,
                     p.upper, p.choices, p.condition_field, p.condition_value,
                     p.allow_enable});
  const auto text = J{
      {"revision", "native-vision-search/counter-evolution-v1"},
      {"base", recipe_identity(base)},
      {"space", space},
      {"seed", o.seed},
      {"trials", o.trials},
      {"elite", o.elite_size},
      {"evolution", o.evolution},
      {"sigma", o.mutation_sigma},
      {"objective", o.objective},
      {"maximize",
       o.maximize}}.dump();
  return sha256_hex(std::as_bytes(std::span(text)));
}
}  // namespace
SearchObservation execute_vision_trial(const RecipeSpec& r,
                                       const std::filesystem::path& path,
                                       const std::string& objective,
                                       TrialTrainingExecutor launcher,
                                       std::stop_token stop) {
  check_recipe(r);
  const std::set<std::string> objectives{"training.loss",   "validation.loss",
                                         "validation.map",  "validation.ap50",
                                         "validation.top1", "validation.top5"};
  if (!objectives.contains(objective))
    throw InvalidArgument("Unknown native vision trial objective");
  const bool has_validation = !r.extension.training_manifest.empty()
                                  ? !r.extension.validation_manifest.empty()
                              : r.model.task == TaskKind::classify
                                  ? !r.classification_validation_root.empty()
                                  : r.validation.has_value();
  if (objective.starts_with("validation.") && !has_validation)
    throw InvalidArgument(
        "Validation search objective requires a validation split");
  if ((objective == "validation.map" || objective == "validation.ap50") &&
      r.model.task == TaskKind::classify)
    throw InvalidArgument(
        "Detection AP search objective is inapplicable to classification");
  if ((objective == "validation.top1" || objective == "validation.top5") &&
      r.model.task != TaskKind::classify)
    throw InvalidArgument(
        "Classification search objective is inapplicable to this task");
  if (objective != "training.loss" && objective != "validation.loss" &&
      !r.full_validation_metrics)
    throw InvalidArgument(
        "Task metric search requires full_validation_metrics");
  std::filesystem::create_directories(path);
  const auto identity = recipe_identity(r);
  auto score = [&](const J& result, const std::filesystem::path& checkpoint) {
    const auto dot = objective.find('.');
    if (dot == std::string::npos || (objective.substr(0, dot) != "training" &&
                                     objective.substr(0, dot) != "validation"))
      throw InvalidArgument("Trial objective needs training.* or validation.*");
    const auto& metrics = result.at(
        objective.starts_with("validation.") ? "validation" : "training");
    auto key = objective.substr(dot + 1);
    if (metrics.contains(objective)) key = objective;
    if (!metrics.contains(key))
      throw InvalidArgument("Native trial did not report selected objective: " +
                            objective);
    if (!std::filesystem::exists(checkpoint))
      throw IoError("Scored trial checkpoint is missing");
    SearchObservation observation{metrics.at(key).get<double>(), checkpoint,
                                  result.at("run_identity").get<std::string>()};
    Publish(path / "observation.json",
            {{"recipe_identity", identity},
             {"objective_name", objective},
             {"objective", observation.objective},
             {"checkpoint", checkpoint.generic_string()},
             {"run_identity", observation.run_identity}});
    return observation;
  };
  if (std::filesystem::exists(path / "observation.json")) {
    const auto row = ReadJson(path / "observation.json");
    if (row.at("recipe_identity") != identity ||
        row.at("objective_name") != objective)
      throw CompatibilityError("Persisted trial observation identity changed");
    auto checkpoint =
        std::filesystem::u8path(row.at("checkpoint").get<std::string>());
    if (!std::filesystem::exists(checkpoint))
      throw IoError("Persisted trial checkpoint is missing");
    return SearchObservation{row.at("objective").get<double>(), checkpoint,
                             row.at("run_identity").get<std::string>()};
  }
  if (std::filesystem::exists(path / "trial.recipe.json")) {
    if (ReadJson(path / "trial.recipe.json") != J::parse(recipe_json(r)))
      throw CompatibilityError("Pending trial recipe differs");
  } else
    Publish(path / "trial.recipe.json", J::parse(recipe_json(r)));
  std::filesystem::path resume, attempt;
  for (std::size_t i = 0; i < 1024; ++i) {
    attempt = path / ("attempt-" + std::to_string(i));
    if (!std::filesystem::exists(attempt)) break;
    if (std::filesystem::exists(attempt / "training.json")) {
      auto result = ReadJson(attempt / "training.json");
      const auto status = result.at("status").get<int>();
      if (status == static_cast<int>(training::FitStatus::completed) ||
          status == static_cast<int>(training::FitStatus::stopped))
        return score(result, attempt / "last.checkpoint");
    }
    if (std::filesystem::exists(attempt / "last.checkpoint"))
      resume = attempt / "last.checkpoint";
    if (i == 1023)
      throw ResourceExhausted("Native trial exceeded 1024 recovery attempts");
  }
  TrainRequest request{
      .recipe = r, .output = attempt, .resume = resume, .stop = stop};
  if (launcher)
    launcher(request);
  else
    static_cast<void>(train_vision(request));
  const auto result = ReadJson(attempt / "training.json");
  const auto status =
      static_cast<training::FitStatus>(result.at("status").get<int>());
  if (status != training::FitStatus::completed &&
      status != training::FitStatus::stopped)
    throw Cancelled("Trial did not reach a scoring boundary");
  return score(result, attempt / "last.checkpoint");
}
SearchResult search_vision(const RecipeSpec& base, const SearchOptions& o,
                           const std::filesystem::path& output,
                           TrialExecutor execute, std::stop_token stop) {
  check_recipe(base);
  if (output.empty() || o.space.empty() || o.space.size() > 64 ||
      o.trials < 1 || o.trials > 100000 || o.elite_size < 1 ||
      o.elite_size > 1000 || !std::isfinite(o.mutation_sigma) ||
      o.mutation_sigma <= 0 || o.mutation_sigma > 1 || o.objective.empty())
    throw InvalidArgument("Invalid native search space/budget");
  std::set<std::string> names;
  auto copy = base;
  for (const auto& p : o.space) {
    if (p.field.starts_with("classification.") &&
        base.model.task != TaskKind::classify)
      throw InvalidArgument(
          "Classification search field requires a classification task");
    if (base.model.task == TaskKind::classify &&
        std::set<std::string>{"mosaic", "mixup", "cutmix", "copypaste",
                              "close_epochs", "scale", "translate", "degrees",
                              "shear", "perspective", "hsv_h", "hsv_s", "hsv_v",
                              "horizontal_flip", "vertical_flip", "grayscale",
                              "blur", "median_blur", "clahe"}
            .contains(p.field))
      throw InvalidArgument(
          "Classification augmentation search uses classification.* fields");
    const double original = Field(copy, p.field);
    if (!std::isfinite(p.lower) || !std::isfinite(p.upper) ||
        p.lower > p.upper || !std::isfinite(p.condition_value) ||
        (!p.condition_field.empty() && !names.contains(p.condition_field)) ||
        !names.insert(p.field).second ||
        (p.distribution == SearchDistribution::log_uniform && p.lower <= 0) ||
        static_cast<unsigned>(p.distribution) > 3)
      throw InvalidArgument("Invalid search range/condition/order");
    if (p.distribution == SearchDistribution::integer &&
        (std::ceil(p.lower) != p.lower || std::floor(p.upper) != p.upper ||
         p.upper - p.lower > 1000000))
      throw InvalidArgument("Invalid integer search range");
    if (p.distribution == SearchDistribution::discrete &&
        (p.choices.empty() || p.choices.size() > 4096))
      throw InvalidArgument("Discrete search needs bounded choices");
    for (auto v : p.choices)
      if (!std::isfinite(v)) throw InvalidArgument("Nonfinite search choice");
    if (original == 0 && !p.allow_enable && p.field != "weight_decay" &&
        p.field != "label_smoothing")
      throw InvalidArgument(
          "Search must explicitly allow enabling a disabled field: " + p.field);
  }
  if (!execute)
    execute = [&o, stop](const RecipeSpec& r, const std::filesystem::path& path,
                         std::size_t) {
      return execute_vision_trial(r, path, o.objective, {}, stop);
    };
  const auto identity = Identity(base, o);
  std::filesystem::create_directories(output);
  const auto path = output / "search.json";
  J state = {{"schema", "vision.search/v1"},
             {"identity", identity},
             {"completed", J::array()},
             {"pending", nullptr}};
  if (!std::filesystem::exists(path) &&
      std::filesystem::exists(output / "search.json.previous"))
    std::filesystem::rename(output / "search.json.previous", path);
  if (std::filesystem::exists(path)) {
    if (std::filesystem::file_size(path) > 64U << 20U)
      throw ResourceExhausted("Native search state exceeds 64 MiB");
    std::ifstream stream(path);
    stream >> state;
    if (state.at("schema") != "vision.search/v1" ||
        state.at("identity") != identity || !state.at("completed").is_array() ||
        state["completed"].size() > o.trials)
      throw CompatibilityError("Native search restore identity mismatch");
  }
  while (state["completed"].size() < o.trials) {
    if (stop.stop_requested())
      throw Cancelled("Native vision search cancelled");
    const auto index = state["completed"].size();
    if (state["pending"].is_null()) {
      data::image::SampleRandom random({.dataset_id = identity,
                                        .sample_id = std::to_string(index),
                                        .transform_id = "vision-search/v1",
                                        .seed = o.seed});
      J candidate = J::object();
      J parent = nullptr;
      if (o.evolution && index >= o.elite_size) {
        std::vector<std::size_t> order(index);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
          const double x = state["completed"][a]["objective"],
                       y = state["completed"][b]["objective"];
          return o.maximize ? x > y : x < y;
        });
        parent = state["completed"][order[random.bounded(
            (std::min)(o.elite_size, index), 0)]]["candidate"];
      }
      for (std::size_t i = 0; i < o.space.size(); ++i) {
        const auto& p = o.space[i];
        if (!p.condition_field.empty() &&
            (!candidate.contains(p.condition_field) ||
             candidate[p.condition_field] != p.condition_value))
          continue;
        double value = 0;
        const auto draw = 8 * i + 1;
        if (p.distribution == SearchDistribution::discrete)
          value = p.choices[random.bounded(p.choices.size(), draw)];
        else if (!parent.is_null() && parent.contains(p.field)) {
          const double normal =
              std::sqrt(-2 * std::log((std::max)(random.unit(draw), 1e-15))) *
              std::cos(2 * std::acos(-1.) * random.unit(draw + 1));
          if (p.distribution == SearchDistribution::log_uniform)
            value = std::exp(
                std::clamp(std::log(parent[p.field].get<double>()) +
                               normal * o.mutation_sigma *
                                   (std::log(p.upper) - std::log(p.lower)),
                           std::log(p.lower), std::log(p.upper)));
          else
            value =
                std::clamp(parent[p.field].get<double>() +
                               normal * o.mutation_sigma * (p.upper - p.lower),
                           p.lower, p.upper);
        } else
          value = p.distribution == SearchDistribution::log_uniform
                      ? std::exp(std::log(p.lower) +
                                 random.unit(draw) *
                                     (std::log(p.upper) - std::log(p.lower)))
                      : p.lower + random.unit(draw) * (p.upper - p.lower);
        if (p.distribution == SearchDistribution::integer)
          value = std::round(value);
        candidate[p.field] = value;
      }
      static_cast<void>(Apply(base, candidate));
      state["pending"] = {{"index", index}, {"candidate", candidate}};
      Publish(path, state);
    }
    if (state["pending"]["index"] != index)
      throw SerializationError("Native search pending cursor mismatch");
    const auto candidate = state["pending"]["candidate"];
    auto recipe = Apply(base, candidate);
    auto trial = output / ("trial-" + std::to_string(index));
    auto observation = execute(recipe, trial, index);
    if (!std::isfinite(observation.objective) ||
        observation.run_identity.empty() || observation.checkpoint.empty())
      throw NumericalError(
          "Trial returned missing/nonfinite metric or artifact identity");
    if (!std::filesystem::is_regular_file(observation.checkpoint))
      throw IoError("Trial returned a missing checkpoint file");
    state["completed"].push_back(
        {{"index", index},
         {"candidate", candidate},
         {"objective", observation.objective},
         {"run_identity", observation.run_identity},
         {"checkpoint", observation.checkpoint.generic_string()}});
    Publish(trial / "search.observation.json", state["completed"].back());
    Publish(trial / "search.recipe.json", J::parse(recipe_json(recipe)));
    detail::RunLedger ledger(
        output / "experiment",
        "vision.search." + identity + "." + std::to_string(index),
        recipe_identity(recipe), identity,
        {{"training_run", observation.run_identity},
         {"objective", o.objective}},
        false);
    ledger.file(trial / "search.recipe.json", "recipe");
    ledger.file(trial / "search.observation.json", "observation");
    ledger.file(observation.checkpoint, "scored.checkpoint", true, stop);
    ledger.finish(experiment::RunStatus::completed);
    state["pending"] = nullptr;
    Publish(path, state);
  }
  std::size_t best = 0;
  for (std::size_t i = 1; i < state["completed"].size(); ++i) {
    const double x = state["completed"][i]["objective"],
                 y = state["completed"][best]["objective"];
    if (o.maximize ? x > y : x < y) best = i;
  }
  const auto& row = state["completed"][best];
  auto winner = Apply(base, row["candidate"]);
  Publish(output / "best.recipe.json", J::parse(recipe_json(winner)));
  Publish(output / "best.json", row);
  return {winner,
          {row["objective"].get<double>(),
           std::filesystem::u8path(row["checkpoint"].get<std::string>()),
           row["run_identity"].get<std::string>()},
          state["completed"].size()};
}
}  // namespace jyppx::liuding::vision::tasks
