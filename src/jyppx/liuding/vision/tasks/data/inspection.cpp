// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../workflow/artifact_io.hpp"

#include <jyppx/liuding/data_image/sample_random.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/data/inspection.hpp>
#include <jyppx/liuding/vision/tasks/data/orientation.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <list>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
namespace jyppx::liuding::vision::tasks {
namespace {
namespace im = data::image;
using J = nlohmann::json;
void Stop(std::stop_token s) {
  if (s.stop_requested()) throw Cancelled("Vision data operation cancelled");
}
std::string Digest(std::string_view v) {
  return sha256_hex(std::as_bytes(std::span(v)));
}
std::vector<std::byte> Bytes(const std::filesystem::path& path,
                             std::uint64_t maximum) {
  auto size = std::filesystem::file_size(path);
  if (size > maximum)
    throw ResourceExhausted("Cache/source artifact byte bound exceeded");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw IoError("Cannot read cache/source artifact");
  return bytes;
}
class Rect final : public data::samplers::Sampler {
 public:
  std::vector<im::ImageExtent> extents;
  std::size_t batch;
  std::uint64_t seed, epoch = 0;
  std::string identity;
  Rect(std::vector<im::ImageExtent> e, std::size_t b, std::uint64_t s)
      : extents(std::move(e)), batch(b), seed(s) {
    if (extents.empty() || extents.size() > 1000000 || !batch || batch > 4096)
      throw InvalidArgument("Invalid rect sampler bounds");
    J dimensions = J::array();
    for (auto v : extents) {
      if (!v.width || !v.height)
        throw InvalidArgument(
            "Rect sampler needs known positive image extents");
      dimensions.push_back({v.width, v.height});
    }
    identity =
        Digest(J{"rect/sorted-batches-v1", batch, seed, dimensions}.dump());
  }
  std::vector<std::size_t> indices() const override {
    std::vector<std::size_t> sorted(extents.size());
    std::iota(sorted.begin(), sorted.end(), 0);
    std::stable_sort(sorted.begin(), sorted.end(), [&](auto a, auto b) {
      return static_cast<double>(extents[a].height) / extents[a].width <
             static_cast<double>(extents[b].height) / extents[b].width;
    });
    const auto full = sorted.size() / batch;
    std::vector<std::size_t> blocks(full);
    std::iota(blocks.begin(), blocks.end(), 0);
    im::SampleRandom random({.dataset_id = identity,
                             .sample_id = "batches",
                             .transform_id = "rect/shuffle-v1",
                             .seed = seed,
                             .epoch = epoch});
    for (std::size_t i = blocks.size(); i > 1; --i)
      std::swap(blocks[i - 1],
                blocks[static_cast<std::size_t>(random.bounded(i, i))]);
    std::vector<std::size_t> output;
    output.reserve(sorted.size());
    for (auto b : blocks)
      output.insert(output.end(), sorted.begin() + b * batch,
                    sorted.begin() + (b + 1) * batch);
    output.insert(output.end(), sorted.begin() + full * batch, sorted.end());
    return output;
  }
  std::size_t size() const override { return extents.size(); }
  void set_epoch(std::uint64_t e) override { epoch = e; }
  bool supports_epoch_control() const override { return true; }
  bool supports_state() const override { return true; }
  std::string configuration_identity() const override { return identity; }
  std::string save_state() const override { return std::to_string(epoch); }
  void load_state(const std::string& text) override {
    std::uint64_t value;
    auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
      throw SerializationError("Malformed rect sampler state");
    epoch = value;
  }
};
}  // namespace
struct ImageDecodeCache::Impl {
  DecodeCacheOptions options;
  mutable std::mutex mutex;
  DecodeCacheStats stats;
  std::uint64_t clock = 0;
  struct Ram {
    im::DecodedImage image;
    std::uint64_t tick;
  };
  std::map<std::string, Ram> ram;
  struct Disk {
    std::filesystem::path path;
    std::uint64_t bytes, tick;
  };
  std::map<std::string, Disk> disk;
  explicit Impl(DecodeCacheOptions o) : options(std::move(o)) {
    if (options.decoder_revision.empty() ||
        options.decoder_revision.size() > 4096 ||
        options.ram_bytes > (64ULL << 30U) ||
        options.disk_bytes > (1ULL << 40U) ||
        (options.disk_bytes && !options.directory.is_absolute()))
      throw InvalidArgument("Invalid decode cache identity/budget/directory");
    if (options.disk_bytes) {
      std::filesystem::create_directories(options.directory);
      std::size_t entries = 0;
      for (const auto& entry :
           std::filesystem::directory_iterator(options.directory)) {
        if (++entries > 1000000)
          throw ResourceExhausted("Cache directory entry bound exceeded");
        const auto name = entry.path().stem().string();
        if (entry.path().extension() != ".lfpixels" || name.size() != 64 ||
            !entry.is_regular_file())
          continue;
        auto bytes = entry.file_size();
        disk[name] = {entry.path(), bytes, clock++};
        stats.disk_bytes += bytes;
      }
      EvictDisk(0);
    }
  }
  void EvictDisk(std::uint64_t next) {
    while (next <= options.disk_bytes &&
           stats.disk_bytes > options.disk_bytes - next && !disk.empty()) {
      auto old = std::min_element(disk.begin(), disk.end(),
                                  [](const auto& a, const auto& b) {
                                    return a.second.tick < b.second.tick;
                                  });
      std::filesystem::remove(old->second.path);
      stats.disk_bytes -= old->second.bytes;
      disk.erase(old);
    }
  }
};
ImageDecodeCache::ImageDecodeCache(DecodeCacheOptions o)
    : impl_(std::make_unique<Impl>(std::move(o))) {}
ImageDecodeCache::~ImageDecodeCache() = default;
DecodeCacheStats ImageDecodeCache::stats() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->stats;
}
im::DecodedImage ImageDecodeCache::read(const std::filesystem::path& path,
                                        const im::ImageDecodeOptions& options,
                                        std::stop_token stop) {
  Stop(stop);
  auto& s = *impl_;
  if (options.channels != im::ImageChannels::rgb ||
      options.layout != im::ImageLayout::chw ||
      options.dtype != im::ImageDtype::preserve || options.pin_memory ||
      options.resource_budget)
    throw InvalidArgument(
        "Decode cache supports unpinned CPU RGB uint8 CHW without an aggregate "
        "decoder budget");
  const auto encoded = Bytes(path, options.limits.max_input_bytes);
  const auto source_digest = sha256_hex(encoded);
  const auto key = Digest(
      J{s.options.decoder_revision, source_digest,
        static_cast<int>(options.format), static_cast<int>(options.channels),
        static_cast<int>(options.layout), static_cast<int>(options.dtype),
        options.limits.max_dimension, options.limits.max_pixels,
        options.limits.max_output_bytes}
          .dump());
  std::unique_lock lock(s.mutex);
  if (auto it = s.ram.find(key); it != s.ram.end()) {
    it->second.tick = s.clock++;
    ++s.stats.ram_hits;
    auto copy = it->second.image;
    copy.tensor = copy.tensor.clone();
    return copy;
  }
  im::DecodedImage image;
  bool loaded = false;
  if (auto it = s.disk.find(key); it != s.disk.end()) {
    try {
      auto bytes =
          Bytes(it->second.path, options.limits.max_output_bytes + 512);
      std::size_t offset = 0;
      auto number = [&]() {
        if (offset + 8 > bytes.size())
          throw SerializationError("Truncated cache header");
        std::uint64_t v = 0;
        for (unsigned i = 0; i < 8; ++i)
          v |= std::uint64_t(std::to_integer<unsigned char>(bytes[offset++]))
               << (8 * i);
        return v;
      };
      if (number() != 0x31584550464cULL)
        throw SerializationError("Unknown decode cache format");
      const auto width = number(), height = number(), size = number();
      if (!width || !height || width > options.limits.max_dimension ||
          height > options.limits.max_dimension ||
          width > options.limits.max_pixels / height ||
          width * height > options.limits.max_output_bytes / 3 ||
          size != width * height * 3 || bytes.size() != 32 + 64 + size)
        throw SerializationError("Invalid decode cache shape/payload");
      std::string digest(reinterpret_cast<const char*>(bytes.data() + 32), 64);
      auto payload = std::span(bytes).subspan(96);
      if (sha256_hex(payload) != digest)
        throw SerializationError("Corrupt decode cache pixel digest");
      image.tensor = torch::empty({3, static_cast<std::int64_t>(height),
                                   static_cast<std::int64_t>(width)},
                                  torch::kUInt8);
      std::memcpy(image.tensor.data_ptr(), payload.data(), payload.size());
      image.width = static_cast<std::uint32_t>(width);
      image.height = static_cast<std::uint32_t>(height);
      image.channels = 3;
      image.layout = im::ImageLayout::chw;
      image.pixels.dtype = torch::kUInt8;
      image.pixels.layout = im::ImageLayout::chw;
      image.pixels.color = im::PixelColorModel::rgb;
      image.pixels.range = im::PixelValueRange::unsigned_full_scale;
      it->second.tick = s.clock++;
      ++s.stats.disk_hits;
      loaded = true;
    } catch (const Error&) {
      ++s.stats.invalid;
      if (s.options.reject_corrupt) throw;
      std::filesystem::remove(it->second.path);
      s.stats.disk_bytes -= it->second.bytes;
      s.disk.erase(it);
    }
  }
  if (!loaded) {
    ++s.stats.misses;
    image = im::decode_image(encoded, options, stop);
    if (image.tensor.scalar_type() != torch::kUInt8)
      throw InvalidArgument(
          "Decode cache native profile requires uint8 sources");
    const auto size = static_cast<std::uint64_t>(image.tensor.numel());
    if (s.options.disk_bytes && size + 96 <= s.options.disk_bytes) {
      auto final = s.options.directory / (key + ".lfpixels");
      s.EvictDisk(size + 96);
      std::vector<std::byte> encoded_pixels;
      encoded_pixels.reserve(size + 96);
      for (auto number :
           {0x31584550464cULL, static_cast<unsigned long long>(image.width),
            static_cast<unsigned long long>(image.height),
            static_cast<unsigned long long>(size)})
        for (unsigned i = 0; i < 8; ++i)
          encoded_pixels.push_back(
              static_cast<std::byte>((number >> (8 * i)) & 255));
      auto data = std::span(
          static_cast<const std::byte*>(image.tensor.const_data_ptr()),
          static_cast<std::size_t>(size));
      auto digest = sha256_hex(data);
      auto header = std::as_bytes(std::span(digest));
      encoded_pixels.insert(encoded_pixels.end(), header.begin(), header.end());
      encoded_pixels.insert(encoded_pixels.end(), data.begin(), data.end());
      Stop(stop);
      detail::publish_artifact(final, encoded_pixels);
      s.disk[key] = {final, size + 96, s.clock++};
      s.stats.disk_bytes += size + 96;
    }
  }
  const auto bytes = static_cast<std::uint64_t>(image.tensor.numel());
  if (s.options.ram_bytes && bytes <= s.options.ram_bytes) {
    while (s.stats.ram_bytes > s.options.ram_bytes - bytes && !s.ram.empty()) {
      auto old = std::min_element(s.ram.begin(), s.ram.end(),
                                  [](const auto& a, const auto& b) {
                                    return a.second.tick < b.second.tick;
                                  });
      s.stats.ram_bytes -= old->second.image.tensor.numel();
      s.ram.erase(old);
    }
    auto retained = image;
    retained.tensor = retained.tensor.clone();
    s.ram.emplace(key, Impl::Ram{std::move(retained), s.clock++});
    s.stats.ram_bytes += bytes;
  }
  Stop(stop);
  return image;
}
std::string inspect_dataset(const DetectionDatasetImport& source, bool decode,
                            std::stop_token stop) {
  const auto& a = source.annotations;
  J per_class = J::array(), issues = J::array(), content = J::array();
  std::vector<std::uint64_t> counts(a.categories.size());
  std::array<std::uint64_t, 3> sizes{};
  std::map<std::string, std::uint64_t> extents;
  std::map<std::string, std::string> digests;
  std::uint64_t exif_rotated = 0, unreadable = 0;
  std::uint64_t negative = 0, objects = 0, ignore = 0, crowd = 0, pose = 0,
                regions = 0, obb = 0;
  std::set<std::string> paths;
  for (const auto& record : a.images) {
    Stop(stop);
    if (!paths.insert(record.image_path).second)
      issues.push_back(
          {{"path", record.image_path}, {"issue", "duplicate_path"}});
    if (record.objects.empty()) ++negative;
    ++extents[std::to_string(record.width) + "x" +
              std::to_string(record.height)];
    for (const auto& object : record.objects) {
      ++objects;
      if (object.category >= counts.size())
        throw InvalidArgument("Dataset class index outside frozen vocabulary");
      ++counts[object.category];
      ignore += object.ignore;
      crowd += object.crowd;
      pose += !object.keypoints.empty();
      regions += object.rle.has_value() || !object.polygons.empty();
      obb += object.oriented_box.has_value();
      if (!object.box)
        issues.push_back({{"path", record.image_path},
                          {"object", object.source_id},
                          {"issue", "missing_box"}});
      else {
        const auto b = *object.box;
        const double area = (b.x_max - b.x_min) * (b.y_max - b.y_min);
        if (!std::isfinite(area) || b.x_max <= b.x_min || b.y_max <= b.y_min)
          issues.push_back({{"path", record.image_path},
                            {"object", object.source_id},
                            {"issue", "degenerate_box"}});
        else
          ++sizes[area < 32 * 32 ? 0 : area < 96 * 96 ? 1 : 2];
        if (b.x_min < 0 || b.y_min < 0 || b.x_max > record.width ||
            b.y_max > record.height)
          issues.push_back({{"path", record.image_path},
                            {"object", object.source_id},
                            {"issue", "out_of_bounds"}});
      }
    }
    if (issues.size() > 100000)
      throw ResourceExhausted("Data inspection issue bound exceeded");
    if (decode) try {
        const auto path =
            source.image_root / std::filesystem::u8path(record.image_path);
        auto bytes = Bytes(path, 256U << 20U);
        im::ImageDecodeOptions options;
        options.channels = im::ImageChannels::rgb;
        auto image = im::decode_image(bytes, options, stop);
        if (image.width != record.width || image.height != record.height)
          issues.push_back({{"path", record.image_path},
                            {"issue", "decoded_dimensions_mismatch"}});
        const auto orientation = image_exif_orientation(path);
        if (orientation != 1) {
          ++exif_rotated;
          issues.push_back(
              {{"path", record.image_path},
               {"issue", "non_identity_exif_requires_coordinate_policy"},
               {"orientation", orientation}});
        }
        auto digest = sha256_hex(bytes);
        content.push_back({record.image_path, digest});
        auto [it, inserted] = digests.emplace(digest, record.image_path);
        if (!inserted)
          issues.push_back({{"path", record.image_path},
                            {"issue", "duplicate_image_content"},
                            {"first", it->second}});
      } catch (const Cancelled&) {
        throw;
      } catch (const std::exception& error) {
        ++unreadable;
        issues.push_back({{"path", record.image_path},
                          {"issue", "unreadable_image"},
                          {"reason", error.what()}});
      }
  }
  for (std::size_t i = 0; i < counts.size(); ++i)
    per_class.push_back({{"name", a.categories[i].name},
                         {"source_id", a.categories[i].source_id},
                         {"objects", counts[i]}});
  return J{{"schema", "vision.data_audit/v1"},
           {"metadata_identity", source.metadata_identity},
           {"images", a.images.size()},
           {"objects", objects},
           {"negative", negative},
           {"ignored", ignore},
           {"crowd", crowd},
           {"pose_objects", pose},
           {"region_objects", regions},
           {"obb_objects", obb},
           {"box_size_counts", sizes},
           {"image_extents", extents},
           {"exif_non_identity", exif_rotated},
           {"unreadable", unreadable},
           {"classes", per_class},
           {"issues", issues},
           {"image_content", content}}
      .dump(2);
}
std::shared_ptr<data::samplers::Sampler> make_rect_sampler(
    std::vector<im::ImageExtent> extents, std::size_t batch,
    std::uint64_t seed) {
  return std::make_shared<Rect>(std::move(extents), batch, seed);
}
}  // namespace jyppx::liuding::vision::tasks
