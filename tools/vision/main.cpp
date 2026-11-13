// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/experiment/local_execution/process.hpp>
#include <jyppx/liuding/vision/tasks/data/context.hpp>
#include <jyppx/liuding/vision/tasks/extended.hpp>
#include <jyppx/liuding/vision/tasks/data/extended.hpp>
#include <jyppx/liuding/vision/tasks/tracking/workflow.hpp>
#include <jyppx/liuding/vision/tasks/tracking/metrics.hpp>

#include <jyppx/liuding/data_image/geometry_trace.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/vision/tasks/training/weight_conversion.hpp>
#include <jyppx/liuding/vision/tasks/workflow/export.hpp>
#include <jyppx/liuding/vision/tasks/workflow/output.hpp>
#include <jyppx/liuding/vision/tasks/workflow/preflight.hpp>
#include <jyppx/liuding/vision/tasks/workflow/search.hpp>

#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace {
namespace t = jyppx::liuding::vision::tasks;
namespace im = jyppx::liuding::data::image;
namespace j = jyppx::liuding;
using J = nlohmann::json;
constexpr std::array<const char*,10> tasks{"detect","classify","segment","pose","obb","classify_multilabel","semantic_segment","depth","pretrain_simmim","reid"};
class Arguments {
  std::map<std::string, std::string> values;

 public:
  Arguments(const std::vector<std::string>& args) {
    static const std::set<std::string> flags{
        "random-init", "text",           "render", "crops",
        "decode",      "no-annotations", "fuse"};
    for (std::size_t i = 2; i < args.size(); ++i) {
      const auto& key = args[i];
      if (!key.starts_with("--") || key.size() < 3)
        throw j::InvalidArgument("Expected --option");
      auto name = key.substr(2);
      if (values.contains(name))
        throw j::InvalidArgument("Repeated option: " + key);
      if (flags.contains(name))
        values[name] = "true";
      else {
        if (i + 1 == args.size() || args[i + 1].starts_with("--"))
          throw j::InvalidArgument("Missing value: " + key);
        values[name] = args[++i];
      }
    }
  }
  std::string take(const std::string& name, std::string fallback = {}) {
    auto found = values.find(name);
    if (found == values.end()) return fallback;
    auto value = found->second;
    values.erase(found);
    return value;
  }
  std::string require(const std::string& name) {
    auto value = take(name);
    if (value.empty()) throw j::InvalidArgument("Required option: --" + name);
    return value;
  }
  bool flag(const std::string& name) { return take(name) == "true"; }
  std::uint64_t number(const std::string& name, std::uint64_t fallback) {
    auto text = take(name);
    if (text.empty()) return fallback;
    std::uint64_t value;
    auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
      throw j::InvalidArgument("Invalid unsigned integer: --" + name);
    return value;
  }
  void check() const {
    if (!values.empty())
      throw j::InvalidArgument("Unknown or inapplicable option: --" +
                               values.begin()->first);
  }
};
std::filesystem::path Path(const std::string& text) {
  return std::filesystem::u8path(text);
}
std::string Utf8(const std::filesystem::path& path) {
  auto text = path.generic_u8string();
  return {reinterpret_cast<const char*>(text.data()), text.size()};
}
std::filesystem::path Executable() {
#ifdef _WIN32
  std::wstring value(32768, L'\0');
  const auto length = GetModuleFileNameW(nullptr, value.data(),
                                         static_cast<DWORD>(value.size()));
  if (!length || length >= value.size())
    throw j::IoError("Cannot identify the vision CLI executable");
  value.resize(length);
  return std::filesystem::path(value);
#elif defined(__linux__)
  std::error_code error;
  auto value = std::filesystem::read_symlink("/proc/self/exe", error);
  if (error || !value.is_absolute())
    throw j::IoError("Cannot identify the vision CLI executable");
  return value;
#else
  throw j::CompatibilityError("Vision process trials require Windows or Linux");
#endif
}
std::string Read(const std::filesystem::path& path) {
  auto size = std::filesystem::file_size(path);
  if (size > (64U << 20U))
    throw j::ResourceExhausted("CLI JSON exceeds 64 MiB");
  std::ifstream input(path, std::ios::binary);
  std::string text(size, '\0');
  input.read(text.data(), static_cast<std::streamsize>(size));
  if (!input) throw j::IoError("Cannot read JSON");
  return text;
}
void Emit(const std::string& text, const std::string& output) {
  if (output.empty()) {
    std::cout << text << '\n';
    return;
  }
  auto path = Path(output);
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file << text << '\n';
  file.close();
  if (!file.good()) throw j::IoError("Cannot write CLI result");
}
t::TaskBinding Binding(t::RecipeSpec& recipe, Arguments& args) {
  const auto weights = args.take("weights"), digest = args.take("sha256");
  const bool random = args.flag("random-init");
  if (!weights.empty()) {
    recipe.initialization.weights = std::filesystem::absolute(Path(weights));
    recipe.initialization.sha256 = digest;
  } else if (!digest.empty())
    throw j::InvalidArgument("--sha256 needs --weights");
  if (recipe.initialization.weights.empty() && !random)
    throw j::InvalidArgument(
        "Select recipe initialization or --weights/--sha256; --random-init "
        "explicitly selects untrained weights");
  if (random && !recipe.initialization.weights.empty())
    throw j::InvalidArgument(
        "--random-init conflicts with initialized weights");
  torch::manual_seed(recipe.seed);
  auto binding = t::bind_task(recipe);
  binding.model->state().to(recipe.device);
  t::initialize_vision(*binding.model, recipe.initialization,
                       recipe.model.architecture);
  return binding;
}
im::VisionSample Image(
    const std::filesystem::path& path,
    t::ExifPolicy policy = t::ExifPolicy::reject_non_identity) {
  im::ImageDecodeOptions options;
  options.channels = im::ImageChannels::rgb;
  auto decoded = im::read_image_file(path, options);
  im::VisionSample sample;
  sample.image = std::move(decoded.tensor);
  sample.pixels = decoded.pixels;
  sample.width = decoded.width;
  sample.height = decoded.height;
  auto name = path.generic_u8string();
  sample.image_path.assign(reinterpret_cast<const char*>(name.data()),
                           name.size());
  return t::orient_vision_sample(std::move(sample), path, policy);
}
t::SearchOptions Search(const J& json) {
  t::SearchOptions options;
  static const std::set<std::string> fields{
      "space",     "seed",           "trials",    "elite_size",
      "evolution", "mutation_sigma", "objective", "maximize"};
  for (const auto& field : json.items())
    if (!fields.contains(field.key()))
      throw j::InvalidArgument("Unknown search option: " + field.key());
  options.seed = json.value("seed", options.seed);
  options.trials = json.value("trials", options.trials);
  options.elite_size = json.value("elite_size", options.elite_size);
  options.evolution = json.value("evolution", options.evolution);
  options.mutation_sigma = json.value("mutation_sigma", options.mutation_sigma);
  options.objective = json.value("objective", options.objective);
  options.maximize = json.value("maximize", options.maximize);
  for (const auto& entry : json.at("space")) {
    static const std::set<std::string> allowed{
        "field",   "distribution",    "lower",           "upper",
        "choices", "condition_field", "condition_value", "allow_enable"};
    for (const auto& field : entry.items())
      if (!allowed.contains(field.key()))
        throw j::InvalidArgument("Unknown search parameter: " + field.key());
    t::SearchParameter p;
    p.field = entry.at("field").get<std::string>();
    auto distribution = entry.value("distribution", std::string("uniform"));
    if (distribution == "uniform")
      p.distribution = t::SearchDistribution::uniform;
    else if (distribution == "log_uniform")
      p.distribution = t::SearchDistribution::log_uniform;
    else if (distribution == "integer")
      p.distribution = t::SearchDistribution::integer;
    else if (distribution == "discrete")
      p.distribution = t::SearchDistribution::discrete;
    else
      throw j::InvalidArgument("Unknown search distribution");
    p.lower = entry.value("lower", p.lower);
    p.upper = entry.value("upper", p.upper);
    p.choices = entry.value("choices", p.choices);
    p.condition_field = entry.value("condition_field", p.condition_field);
    p.condition_value = entry.value("condition_value", p.condition_value);
    p.allow_enable = entry.value("allow_enable", p.allow_enable);
    options.space.push_back(std::move(p));
  }
  return options;
}
int Run(const std::vector<std::string>& argv) {
  if (argv.size() < 2 || argv[1] == "--help" || argv[1] == "help") {
    std::cout
        << "Liuding Forge native vision suite\n"
           "  version\n"
           "  catalog [--output catalog.json]\n"
           "  template --model yolov8n --task detect --classes 80 --output "
           "recipe.json\n"
           "  check --recipe recipe.json [--output resolved.json]\n"
           "  train --recipe recipe.json --output run [--pause-after N]\n"
           "  resume --recipe recipe.json --checkpoint last.checkpoint "
           "--output new-run\n"
           "  val --recipe recipe.json --weights weights.safetensors --sha256 "
           "HASH [--output metrics.json]\n"
           "  predict --recipe recipe.json --weights weights.safetensors "
           "--sha256 HASH --source image.jpg --output predictions [--warmup 3 "
           "--text --render --crops]\n"
           "  tune --recipe recipe.json --search search.json --output "
           "search-run [--execution process|in-process --trial-timeout 3600]\n"
           "  export --recipe recipe.json --weights weights.safetensors "
           "--sha256 HASH --output bundle [--fuse]\n"
           "  bundle-predict --bundle bundle --source image.jpg [--output "
           "result.json]\n"
           "  inspect --recipe recipe.json [--decode] [--output audit.json]\n"
           "  preview --recipe recipe.json --output preview [--samples 16 "
           "--epoch 0 --no-annotations]\n"
           "  preflight --recipe recipe.json [--batches 1,2,4] [--output "
           "preflight.json]\n"
           "  convert-data --recipe recipe.json --output "
           "annotations.coco.json\n"
           "  convert-weights --source upstream.pt --sha256 HASH --root model "
           "--output weights.safetensors [--mapping mapping.json --metadata "
           "source.json]\n"
           "  prepare-prompts --recipe recipe.json --prompts literal-prompts.json --output context.json\n"
           "  calibrate --recipe multilabel.json --weights student.safetensors --sha256 HASH --output thresholds.json\n"
           "  extract-encoder --recipe recipe.json --weights state.safetensors --sha256 HASH --output encoder.safetensors\n"
           "  track --recipe detector.json --sequence frames.json --tracker byte_track --output tracks\n"
           "  track-eval --ground-truth gt.txt --predictions tracks.txt [--output metrics.json]\n"
           "  map-weights --recipe recipe.json --source weights.safetensors "
           "--sha256 HASH [--output mapping.json]\n";
    return 0;
  }
  const auto& command = argv[1];
  Arguments args(argv);
  if (command == "version" || command == "--version") {
    args.check();
    std::cout << j::version() << '\n';
    return 0;
  }
  if (command == "catalog") {
    auto output = args.take("output");
    args.check();
    J values = J::array();
    for (const auto& entry : t::vision_catalog())
      values.push_back(
          {{"architecture", entry.architecture},
           {"task", tasks.at(static_cast<std::size_t>(entry.task))},
           {"variant", entry.variant},
           {"revision", entry.revision},
           {"component", entry.component},
           {"input_divisor", entry.input_divisor},{"fixed_input_size",entry.fixed_input_size},{"minimum_input_size",entry.minimum_input_size},{"output_contract",entry.output_contract},{"requires_external_proposals",entry.requires_external_proposals},
           {"end_to_end", entry.end_to_end},
           {"native_training", entry.native_training},
           {"pretrained_weights", entry.pretrained_weights},
           {"reference_qualified", entry.reference_qualified},
           {"quality_qualified", entry.quality_qualified}});
    Emit(values.dump(2), output);
    return 0;
  }
  if (command == "template") {
    t::RecipeSpec recipe;
    recipe.model.architecture = args.require("model");
    auto task = args.take("task", "detect");
    recipe.model.task = t::parse_task(task);
    recipe.model.variant = args.take("variant", "p5");
    recipe.model.num_classes = args.number("classes", 80);
    recipe.optimizer_name = "auto";
    recipe.training.format = t::DetectionDatasetFormat::yolo;
    recipe.training.source = std::filesystem::absolute("dataset.yaml");
    if (recipe.model.task == t::TaskKind::classify)
      recipe.classification_root = std::filesystem::absolute("dataset/train");
    auto output = args.require("output");
    args.check();
    auto capability=t::describe_binding(recipe.model);
    if(capability.component=="models_vision"||static_cast<unsigned>(recipe.model.task)>4){recipe.schema_version=2;recipe.extension.training_manifest=std::filesystem::absolute("dataset.train.json");recipe.extension.validation_manifest=std::filesystem::absolute("dataset.val.json");}
    if(capability.fixed_input_size[0])recipe.data.letterbox.canvas={static_cast<std::uint32_t>(capability.fixed_input_size[1]),static_cast<std::uint32_t>(capability.fixed_input_size[0])};
    if(recipe.model.architecture.starts_with("simmim_vit"))recipe.extension.model_patch_size=16;
    Emit(t::recipe_json(recipe), output);
    return 0;
  }
  if (command == "convert-weights") {
    t::WeightConversionOptions options;
    options.source = Path(args.require("source"));
    options.output = Path(args.require("output"));
    options.sha256 = args.require("sha256");
    options.root = args.take("root", "model");
    auto mapping = args.take("mapping"), metadata = args.take("metadata");
    if (!mapping.empty()) {
      auto value = J::parse(Read(Path(mapping)));
      if (value.contains("key_mapping")) value = value.at("key_mapping");
      options.key_mapping = value.get<std::map<std::string, std::string>>();
    }
    if (!metadata.empty())
      options.metadata = J::parse(Read(Path(metadata)))
                             .get<std::map<std::string, std::string>>();
    args.check();
    std::cout << t::convert_vision_weights(options) << '\n';
    return 0;
  }
  if (command == "bundle-predict") {
    auto bundle = Path(args.require("bundle"));
    auto configuration = t::parse_recipe(
        J::parse(Read(bundle / "manifest.json")).at("recipe").dump(), bundle);
    auto sample =
        Image(Path(args.require("source")), configuration.exif_policy);
    auto context=args.take("sample-data");if(!context.empty())sample.task_payload=t::parse_input_context(Read(Path(context)));
    auto output = args.take("output");
    args.check();
    Emit(t::predict_vision_bundle(bundle, std::span(&sample, 1)), output);
    return 0;
  }
  if(command=="track-eval") {namespace tr=t::tracking;auto gt=tr::read_mot_sequence(Path(args.require("ground-truth")),true);auto pred=tr::read_mot_sequence(Path(args.require("predictions")),false);auto output=args.take("output");args.check();std::map<std::int64_t,tr::TrackingFrame> merged;for(auto& f:gt)merged[f.frame_id]=std::move(f);for(auto& f:pred){auto& row=merged[f.frame_id];row.frame_id=f.frame_id;row.predictions=std::move(f.predictions);}std::vector<tr::TrackingFrame> frames;for(auto& [id,f]:merged)frames.push_back(std::move(f));Emit(J(tr::evaluate_tracking(frames)).dump(2),output);return 0;}
  auto recipe = t::load_recipe(Path(args.require("recipe")));
  if(command=="prepare-prompts"){auto prompts=J::parse(Read(Path(args.require("prompts")))).get<std::vector<std::string>>();auto output=args.require("output");args.check();Emit(t::input_context_json(*t::prepare_text_context(recipe,prompts)),output);return 0;}
  if(command=="extract-encoder"){auto binding=Binding(recipe,args);auto output=Path(args.require("output"));args.check();Emit(J{{"sha256",t::save_simmim_encoder(*binding.model,output,recipe.model.architecture)},{"artifact","encoder_initialization"}}.dump(2),{});return 0;}
  if(command=="track"){namespace tr=t::tracking;auto binding=Binding(recipe,args);auto source=Path(args.require("sequence"));auto sequence=J::parse(Read(source));auto output=Path(args.require("output"));auto kind=args.take("tracker","byte_track");auto reid_path=args.take("reid-recipe");args.check();if(sequence.at("schema")!="vision.sequence/v1"||sequence.at("frames").size()>1000000||(kind!="byte_track"&&kind!="bot_sort"))throw j::InvalidArgument("Invalid tracking sequence/profile");if(std::filesystem::exists(output)&&!std::filesystem::is_empty(output))throw j::InvalidArgument("Track output must be new/empty");std::filesystem::create_directories(output);tr::TrackerOptions options;options.kind=kind=="bot_sort"?tr::TrackerKind::bot_sort:tr::TrackerKind::byte_track;tr::Tracker tracker(options);tracker.reset(sequence.at("sequence_id").get<std::string>());std::optional<t::RecipeSpec> reid_recipe;std::optional<t::TaskBinding> reid;std::optional<im::VisionSample> previous;
    if(options.kind==tr::TrackerKind::bot_sort){if(reid_path.empty())throw j::InvalidArgument("BoT-SORT requires --reid-recipe with initialized weights");reid_recipe=t::load_recipe(Path(reid_path));if(reid_recipe->initialization.weights.empty())throw j::InvalidArgument("ReID recipe needs verified weights");reid=t::bind_task(*reid_recipe);reid->model->state().to(reid_recipe->device);t::initialize_vision(*reid->model,reid_recipe->initialization,reid_recipe->model.architecture);}
    std::vector<tr::TrackingFrame> frames;for(const auto& row:sequence.at("frames")){auto sample=Image(source.parent_path()/Path(row.at("image").get<std::string>()),recipe.exif_policy);auto id=row.at("frame_id").get<std::int64_t>();auto result=tr::track_vision_frame(tracker,binding,sample,recipe,id,options,reid?&*reid:nullptr,reid_recipe?&*reid_recipe:nullptr,previous?&*previous:nullptr);frames.push_back({id,{},std::move(result.tracks)});previous=std::move(sample);}tr::write_mot_sequence(output/"tracks.txt",frames);Emit(tracker.snapshot(),Utf8(output/"tracker.json"));Emit(J{{"schema","vision.tracking_run/v1"},{"frames",frames.size()},{"tracker",kind},{"sequence_id",sequence.at("sequence_id")}}.dump(2),Utf8(output/"summary.json"));return 0;}

  if (command == "map-weights") {
    auto source = Path(args.require("source"));
    auto digest = args.require("sha256");
    auto output = args.take("output");
    args.check();
    auto binding = t::bind_task(recipe);
    Emit(t::propose_vision_weight_mapping(*binding.model, source, digest,
                                          recipe.model.architecture),
         output);
    return 0;
  }
  if (command == "check") {
    auto output = args.take("output");
    args.check();
    auto capability = t::describe_binding(recipe.model);
    if (recipe.data.letterbox.canvas.width % capability.input_divisor ||
        recipe.data.letterbox.canvas.height % capability.input_divisor)
      throw j::InvalidArgument("Image size violates architecture divisor");
    Emit(t::recipe_json(recipe), output);
    return 0;
  }
  if (command == "train" || command == "resume") {
    t::TrainRequest request;
    request.recipe = recipe;
    request.output = Path(args.require("output"));
    if (command == "resume") request.resume = Path(args.require("checkpoint"));
    request.pause_after_updates = args.number("pause-after", 0);
    args.check();
    auto result = t::train_vision(request);
    std::cout << J{{"run_identity", result.run_identity},
                   {"checkpoint", result.last_checkpoint.generic_string()},
                   {"epochs", result.fit.completed_epochs},
                   {"updates", result.fit.global_step},
                   {"status", static_cast<int>(result.fit.status)}}
                     .dump(2)
              << '\n';
    return 0;
  }
  if (command == "tune") {
    auto options = Search(J::parse(Read(Path(args.require("search")))));
    auto output = std::filesystem::absolute(Path(args.require("output")));
    auto execution = args.take("execution", "process");
    auto seconds = args.number("trial-timeout", 3600);
    args.check();
    if ((execution != "process" && execution != "in-process") || seconds < 1 ||
        seconds > 604800)
      throw j::InvalidArgument("Invalid search execution or timeout");
    t::TrialExecutor executor;
    if (execution == "process") {
      auto executable = Executable();
      executor = [executable, seconds, objective = options.objective](
                     const t::RecipeSpec& r, const std::filesystem::path& path,
                     std::size_t) {
        return t::execute_vision_trial(
            r, path, objective,
            [executable, seconds](const t::TrainRequest& request) {
              j::experiment::LocalProcessOptions launch;
              launch.executable = executable;
              launch.working_directory =
                  std::filesystem::absolute(request.output.parent_path());
              launch.inherit_environment = true;
              launch.timeout = std::chrono::milliseconds(seconds * 1000);
              const auto config =
                  launch.working_directory / "trial.recipe.json";
              launch.arguments = {request.resume.empty() ? "train" : "resume",
                                  "--recipe", Utf8(config), "--output",
                                  Utf8(request.output)};
              if (!request.resume.empty()) {
                launch.arguments.push_back("--checkpoint");
                launch.arguments.push_back(Utf8(request.resume));
              }
              auto result =
                  j::experiment::LocalProcessRunner{}.run(launch, request.stop);
              const auto base = Utf8(request.output);
              Emit(result.stdout_capture, base + ".stdout.log");
              Emit(result.stderr_capture, base + ".stderr.log");
              J usage = nullptr;
              if (result.resource_usage) {
                const auto& u = *result.resource_usage;
                usage = {
                    {"source",
                     u.source ==
                             j::experiment::LocalProcessResourceSource::windows_process
                         ? "windows_process"
                         : "linux_wait4"},
                    {"user_cpu_us",
                     u.user_cpu_time_us ? J(*u.user_cpu_time_us) : J(nullptr)},
                    {"system_cpu_us", u.system_cpu_time_us
                                          ? J(*u.system_cpu_time_us)
                                          : J(nullptr)},
                    {"peak_resident_memory_bytes",
                     u.peak_resident_memory_bytes
                         ? J(*u.peak_resident_memory_bytes)
                         : J(nullptr)}};
              }
              Emit(J{{"schema", "vision.trial_process/v1"},
                     {"status", static_cast<int>(result.status)},
                     {"exit_code",
                      result.exit_code ? J(*result.exit_code) : J(nullptr)},
                     {"signal", result.signal ? J(*result.signal) : J(nullptr)},
                     {"process_id", result.process_id},
                     {"stdout_bytes", result.stdout_bytes},
                     {"stderr_bytes", result.stderr_bytes},
                     {"elapsed_ms", result.elapsed.count()},
                     {"resource_usage", usage},
                     {"tree_cleanup", result.process_tree_cleanup_confirmed}}
                       .dump(2),
                   base + ".process.json");
              if (request.stop.stop_requested())
                throw j::Cancelled("Vision search child cancelled");
              if (result.status != j::experiment::LocalProcessStatus::exited ||
                  !result.exit_code || *result.exit_code != 0)
                throw j::IoError(
                    "Vision search child failed; inspect the attempt process "
                    "receipt and logs");
            });
      };
    }
    auto result = t::search_vision(recipe, options, output, executor);
    std::cout << J{{"completed", result.completed},
                   {"objective", result.best.objective},
                   {"checkpoint", result.best.checkpoint.generic_string()}}
                     .dump(2)
              << '\n';
    return 0;
  }
  if (command == "preview") {
    auto output = Path(args.require("output"));
    t::PreviewOptions options;
    options.samples = args.number("samples", 16);
    options.epoch = args.number("epoch", 0);
    options.annotations = !args.flag("no-annotations");
    args.check();
    std::cout << t::preview_vision(recipe, output, options) << '\n';
    return 0;
  }
  if (command == "convert-data" || command == "inspect") {
    auto output = args.take("output");
    bool decode = args.flag("decode");
    args.check();
    if (recipe.model.task == t::TaskKind::classify)
      throw j::InvalidArgument(
          "Use ImageFolder for classification; object annotation conversion "
          "needs a vision task");
    if(!recipe.extension.training_manifest.empty()) {if(command=="convert-data")throw j::InvalidArgument("Task manifests already use canonical native targets; object-only COCO conversion would drop fields");auto source=t::read_task_dataset(recipe.extension.training_manifest);J report{{"schema","vision.task_inspection/v1"},{"identity",source.identity},{"samples",source.dataset->size()},{"categories",source.categories.size()},{"decoded",decode}};if(decode)for(std::size_t i=0;i<source.dataset->size();++i)source.dataset->get(i);Emit(report.dump(2),output);return 0;}
    auto source = t::read_detection_dataset(recipe.training);
    Emit(command == "convert-data" ? t::coco_annotations(source.annotations)
                                   : t::inspect_dataset(source, decode),
         output);
    return 0;
  }
  if (command == "preflight") {
    t::PreflightOptions options;
    auto batches = args.take("batches");
    if (!batches.empty()) {
      options.batch_candidates.clear();
      std::size_t offset = 0;
      while (offset < batches.size()) {
        auto end = batches.find(',', offset);
        auto part = batches.substr(
            offset, end == std::string::npos ? end : end - offset);
        std::size_t count = 0;
        auto parsed =
            std::from_chars(part.data(), part.data() + part.size(), count);
        if (parsed.ec != std::errc{} || parsed.ptr != part.data() + part.size())
          throw j::InvalidArgument("Invalid batch candidate list");
        options.batch_candidates.push_back(count);
        if (end == std::string::npos) break;
        offset = end + 1;
      }
    }
    auto output = args.take("output");
    args.check();
    auto data = t::load_vision_data(recipe);
    std::vector<im::VisionSample> samples;
    for (std::size_t i = 0;
         i < (std::min)(data.training->size(), std::size_t{4}); ++i)
      samples.push_back(data.training->get(i));
    const auto report = t::preflight_vision(recipe, samples, options);
    Emit(report, output);
    return J::parse(report).at("recommended_batch").is_null() ? 2 : 0;
  }
  if (command == "val"||command=="calibrate") {
    auto binding = Binding(recipe, args);
    auto output = args.take("output");
    args.check();
    auto data = t::load_vision_data(recipe);
    if (!data.validation)
      throw j::InvalidArgument(
          "Validation requires an explicit disjoint split");
    if(command=="calibrate"){Emit(t::calibrate_vision_multilabel(binding,*data.validation,recipe,data.validation_identity),output);return 0;}
    Emit(t::evaluate_vision(binding, *data.validation, recipe,
                            data.validation_identity),
         output);
    return 0;
  }
  if (command == "export") {
    auto binding = Binding(recipe, args);
    auto output = Path(args.require("output"));
    t::VisionExportOptions options;
    options.batch_size = args.number("batch", 1);
    options.rcnn_proposals = args.number("proposals", 300);
    options.weight_role = args.take("role", "raw");
    options.fuse = args.flag("fuse");
    args.check();
    std::cout << t::export_vision(binding, recipe, output, options) << '\n';
    return 0;
  }
  if (command == "predict") {
    auto binding = Binding(recipe, args);
    auto sample = Image(Path(args.require("source")), recipe.exif_policy);
    auto context=args.take("sample-data");if(!context.empty())sample.task_payload=t::parse_input_context(Read(Path(context)));
    auto output = Path(args.require("output"));
    t::PredictionOutputOptions options;
    options.text = args.flag("text");
    options.render = args.flag("render");
    options.crops = args.flag("crops");
    auto warmup = args.number("warmup", 0);
    args.check();
    auto prediction = t::predict_vision_profiled(binding, std::span(&sample, 1),
                                                 recipe, warmup);
    const auto& result = prediction.results.at(0);
    const auto& time = prediction.timing;
    t::save_prediction(sample, result, output / "prediction", options);
    std::cout << J{{"timing_ms",
                    {{"preprocess", time.preprocess_ms},
                     {"inference", time.inference_ms},
                     {"postprocess", time.postprocess_ms},
                     {"warmup", time.warmup_ms}}},
                   {"warmup_iterations", warmup},
                   {"synchronization", "device_synchronized_wall_time"},
                   {"result", J::parse(t::result_json(result))}}
                     .dump(2)
              << '\n';
    return 0;
  }
  throw j::InvalidArgument("Unknown vision command: " + command);
}
int Entry(const std::vector<std::string>& argv) {
  try {
    return Run(argv);
  } catch (const std::exception& error) {
    std::cerr << "vision: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  std::vector<std::string> args;
  for (int i = 0; i < argc; ++i) {
    auto size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1,
                                    nullptr, 0, nullptr, nullptr);
    if (size <= 0) return 1;
    std::string text(size, '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, text.data(),
                        size, nullptr, nullptr);
    text.pop_back();
    args.push_back(std::move(text));
  }
  return Entry(args);
}
#else
int main(int argc, char** argv) { return Entry({argv, argv + argc}); }
#endif
