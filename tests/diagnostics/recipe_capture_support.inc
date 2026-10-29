// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

// Diagnostic-only exact production fragment; orchestration checks for drift.
struct Inputs {
  image::ManifestIndex train;
  image::ManifestIndex validation;
  Json manifest;
  ArtifactDigest identity;
  std::uint64_t total_steps;
};
Inputs inspect_inputs(const Json& config) {
  image::ImageFolderOptions options;
  options.extensions = {".jpg", ".jpeg", ".png"};
  options.limits.max_images = config.at("max_samples").get<std::size_t>();
  options.limits.max_categories = 1024;
  const auto train_root =
      std::filesystem::u8path(config.at("train_root").get<std::string>());
  const auto val_root =
      std::filesystem::u8path(config.at("validation_root").get<std::string>());
  auto train = image::index_image_folder(train_root, options);
  auto validation = image::index_image_folder(val_root, options);
  require_recipe(train.categories().size() >= 2,
                 "Classification requires at least two classes");
  const auto classes = [](const image::ManifestIndex& index) {
    Json result = Json::array();
    for (const auto& category : index.categories())
      result.push_back(category.name);
    return result;
  };
  require_recipe(classes(train) == classes(validation),
                 "Training and validation class maps must match exactly");
  const auto batch = config.at("batch_size").get<std::size_t>();
  if (!is_transfer_recipe(config) || config.at("batch_norm") == "train")
    require_recipe(
        train.records().size() >= 2 && train.records().size() % batch != 1,
        "Training BatchNorm requires batches of at least two; choose "
        "another batch_size or use the transfer recipe with batch_norm=eval");
  const auto total_steps = ((train.records().size() + batch - 1) / batch) *
                           config.at("epochs").get<std::uint64_t>();
  require_recipe(total_steps <= 1000000,
                 "Recipe exceeds one million optimizer updates");
  std::uint64_t data_bytes = 0;
  const auto split = [&](const image::ManifestIndex& index,
                         const std::filesystem::path& root) {
    Sha256 hash;
    hash.update(
        bytes(config.at("recipe").get<std::string>() + "/content/v1\n"));
    for (const auto& record : index.records()) {
      const auto file = root / std::filesystem::u8path(record.path);
      const auto size = std::filesystem::file_size(file);
      const auto maximum = config.at("max_data_bytes").get<std::uint64_t>();
      require_recipe(size <= maximum - data_bytes,
                     "Dataset exceeds max_data_bytes");
      data_bytes += size;
      const auto content =
          file_digest(file, config.at("max_image_bytes").get<std::uint64_t>());
      hash.update(bytes(Json{{"path", record.path},
                             {"labels", record.labels},
                             {"bytes", size},
                             {"sha256", content.hex}}
                            .dump() +
                        "\n"));
    }
    return Json{{"samples", index.records().size()},
                {"inventory_sha256", index.source_sha256()},
                {"content_sha256", sha256_digest_hex(hash.finish())}};
  };
  const auto train_manifest = split(train, train_root);
  const auto validation_manifest = split(validation, val_root);
  Json manifest{{"schema_version", 1},
                {"classes", classes(train)},
                {"train", train_manifest},
                {"validation", validation_manifest},
                {"data_license", config.at("data_license")},
                {"image_bytes", data_bytes},
                {"identity_scope",
                 "ordered paths, labels and complete encoded image bytes"}};
  const auto identity = digest(manifest.dump());
  return {std::move(train), std::move(validation), std::move(manifest),
          identity, total_steps};
}

std::shared_ptr<data::DataPipeline> make_pipeline(
    const Json& config, image::ManifestIndex index, bool validation,
    const ArtifactDigest& identity, const Device& device,
    const std::shared_ptr<image::ImageResourceBudget>& budget,
    const TransferInput* transfer) {
  image::ImageDatasetOptions source_options;
  source_options.resource_budget = budget;
  source_options.image.channels = image::ImageChannels::rgb;
  source_options.image.dtype = image::ImageDtype::float32_normalized;
  source_options.image.limits.max_input_bytes =
      config.at("max_image_bytes").get<std::size_t>();
  source_options.image.limits.max_output_bytes =
      config.at("image_budget_bytes").get<std::size_t>();
  auto source = std::make_shared<image::ImageDataset>(
      std::filesystem::u8path(
          config.at(validation ? "validation_root" : "train_root")
              .get<std::string>()),
      std::move(index), source_options);
  const auto size = config.at("image_size").get<std::uint32_t>();
  const image::ImageGeometryStep resize{
      .kind = image::ImageGeometryKind::resize, .extent = {size, size}};
  image::JointTransformOptions transform;
  transform.interpolation = image::ImageInterpolation::bilinear;
  if (transfer) transform.interpolation = image::ImageInterpolation::bicubic;
  transform.resource_budget = budget;
  image::ImageCollateOptions collate;
  collate.resource_budget = budget;
  const auto per_batch =
      config.at("image_budget_bytes").get<std::uint64_t>() /
      (config.at("prefetch_batches").get<std::uint64_t>() + 2);
  collate.limits.max_metadata_bytes = static_cast<std::size_t>(
      std::min<std::uint64_t>(1U << 20U, per_batch / 4));
  collate.limits.max_output_bytes =
      per_batch - collate.limits.max_metadata_bytes;
  collate.limits.max_working_bytes = collate.limits.max_output_bytes;
  const auto preprocess =
      transfer ? std::optional(transfer->record.input_spec->image)
               : std::nullopt;
  image::ImageBatchTransformOptions numeric;
  numeric.resource_budget = budget;
  numeric.limits = collate.limits;
  if (preprocess) {
    numeric.image.mean = preprocess->mean;
    numeric.image.std = preprocess->std;
  }
  const auto factory = [source, resize, transform, collate, preprocess,
                        numeric](std::span<const std::size_t> indices,
                                 const data::DataWorkerContext& context,
                                 std::stop_token stop) {
    std::vector<image::VisionSample> samples;
    std::vector<image::ImageBatchKey> keys;
    samples.reserve(indices.size());
    keys.reserve(indices.size());
    for (const auto index : indices) {
      auto sample = source->get(index, stop);
      keys.push_back({sample.image_path, context.epoch, 0});
      if (preprocess) {
        const auto geometry = models::resolve_image_preprocess_geometry(
            *preprocess, {sample.height, sample.width});
        const std::array steps{
            image::ImageGeometryStep{
                .kind = image::ImageGeometryKind::resize,
                .extent = {geometry.resized_size[1], geometry.resized_size[0]}},
            image::ImageGeometryStep{
                .kind = image::ImageGeometryKind::crop,
                .extent = {geometry.output_size[1], geometry.output_size[0]},
                .left = geometry.crop_left,
                .top = geometry.crop_top}};
        samples.push_back(
            image::transform_sample(sample, steps, transform, stop));
      } else {
        samples.push_back(image::transform_sample(sample, std::span(&resize, 1),
                                                  transform, stop));
      }
    }
    auto batch = image::collate_images(samples, keys, collate, stop);
    if (preprocess) batch = image::transform_image_batch(batch, numeric, stop);
    return image::image_batch_value(batch, collate.limits);
  };
  data::DataPipelineOptions options;
  options.pipeline_id = validation ? "recipe.validation" : "recipe.training";
  options.dataset_version = identity.hex;
  options.dataset_size = source->size();
  options.batch_factory_id = config.at("recipe").get<std::string>();
  if (transfer) options.batch_factory_id += "/" + transfer->input_identity;
  options.batch_size = config.at("batch_size").get<std::size_t>();
  options.num_workers = config.at("workers").get<std::size_t>();
  options.prefetch_batches = config.at("prefetch_batches").get<std::size_t>();
  options.worker_seed = config.at("seed").get<std::uint64_t>();
  if (device.is_cuda()) {
    options.memory.pin_memory = true;
    options.memory.target_device = device;
    options.memory.non_blocking = true;
  }
  {
    const std::array<std::size_t, 1> first{0};
    const auto exemplar = image::image_batch_from_value(factory(first, {}, {}));
    options.schema = image::image_batch_schema(exemplar, device);
  }
  options.admission_factory =
      image::make_image_batch_byte_admission_factory(budget, collate.limits);
  options.resource_lease_factory =
      image::make_image_batch_resource_lease_factory(budget, collate.limits);
  return std::make_shared<data::DataPipeline>(
      std::make_shared<data::samplers::RandomSampler>(
          source->size(), config.at("seed").get<std::int64_t>()),
      options, factory);
}
