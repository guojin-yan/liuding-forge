// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/tasks/data/orientation.hpp>

#include <fstream>
namespace jyppx::liuding::vision::tasks {
namespace {
namespace im = data::image;
using I = std::int64_t;
Tensor Orient(const Tensor& input, std::uint8_t orientation, I y, I x) {
  switch (orientation) {
    case 1:
      return input;
    case 2:
      return input.flip({x});
    case 3:
      return input.flip({x, y});
    case 4:
      return input.flip({y});
    case 5:
      return input.transpose(y, x);
    case 6:
      return input.transpose(y, x).flip({x});
    case 7:
      return input.transpose(y, x).flip({x, y});
    case 8:
      return input.transpose(y, x).flip({y});
    default:
      throw SerializationError("Invalid EXIF orientation");
  }
}
im::ImagePoint Point(im::ImagePoint p, double w, double h, std::uint8_t o) {
  switch (o) {
    case 1:
      return p;
    case 2:
      return {w - p.x, p.y};
    case 3:
      return {w - p.x, h - p.y};
    case 4:
      return {p.x, h - p.y};
    case 5:
      return {p.y, p.x};
    case 6:
      return {h - p.y, p.x};
    case 7:
      return {h - p.y, w - p.x};
    case 8:
      return {p.y, w - p.x};
    default:
      throw SerializationError("Invalid EXIF point orientation");
  }
}
im::ImageMaskRle Rle(const im::ImageMaskRle& source, std::uint8_t orientation) {
  const auto pixels = std::uint64_t(source.width) * source.height;
  if (pixels > 67108864)
    throw ResourceExhausted("EXIF mask exceeds pixel bound");
  auto flat = torch::empty({static_cast<I>(pixels)}, torch::kUInt8);
  auto* values = flat.data_ptr<std::uint8_t>();
  std::uint64_t offset = 0;
  bool foreground = false;
  for (auto count : source.counts) {
    if (count > pixels - offset)
      throw InvalidArgument("Invalid EXIF source RLE");
    std::fill(values + offset, values + offset + count, foreground ? 1 : 0);
    offset += count;
    foreground = !foreground;
  }
  if (offset != pixels) throw InvalidArgument("Invalid RLE extent");
  auto image = Orient(flat.view({source.width, source.height}).transpose(0, 1),
                      orientation, 0, 1);
  auto output = image.transpose(0, 1).contiguous();
  values = output.data_ptr<std::uint8_t>();
  im::ImageMaskRle result;
  result.width = static_cast<std::uint32_t>(image.size(1));
  result.height = static_cast<std::uint32_t>(image.size(0));
  foreground = false;
  std::uint64_t count = 0;
  for (std::uint64_t i = 0; i < pixels; ++i) {
    bool next = values[i] != 0;
    if (next != foreground) {
      result.counts.push_back(count);
      count = 0;
      foreground = next;
    }
    ++count;
  }
  result.counts.push_back(count);
  return result;
}
}  // namespace
std::uint8_t image_exif_orientation(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw IoError("Cannot read image orientation metadata");
  auto byte = [&]() {
    auto v = input.get();
    if (v == std::char_traits<char>::eof())
      throw SerializationError("Truncated JPEG metadata");
    return static_cast<unsigned>(v);
  };
  if (byte() != 0xff || byte() != 0xd8) return 1;
  for (std::size_t sections = 0; sections < 4096; ++sections) {
    if (byte() != 0xff)
      throw SerializationError("Invalid JPEG marker before image scan");
    auto marker = byte();
    while (marker == 0xff) marker = byte();
    if (marker == 0xda || marker == 0xd9) return 1;
    if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    const auto high = byte();
    const auto low = byte();
    const auto size = (high << 8) | low;
    if (size < 2) throw SerializationError("Invalid JPEG section length");
    std::vector<unsigned char> data(size - 2);
    input.read(reinterpret_cast<char*>(data.data()), data.size());
    if (!input) throw SerializationError("Truncated JPEG section");
    if (marker != 0xe1 || data.size() < 14 ||
        std::string(reinterpret_cast<char*>(data.data()), 6) !=
            std::string("Exif\0\0", 6))
      continue;
    const std::size_t base = 6;
    const bool little = data[base] == 'I' && data[base + 1] == 'I';
    if (!little && !(data[base] == 'M' && data[base + 1] == 'M'))
      throw SerializationError("Invalid EXIF byte order");
    auto read = [&](std::size_t at, std::size_t n) {
      if (at > data.size() || n > data.size() - at)
        throw SerializationError("EXIF offset exceeds APP1");
      std::uint32_t v = 0;
      for (std::size_t i = 0; i < n; ++i)
        v |= std::uint32_t(data[at + i]) << (8 * (little ? i : n - 1 - i));
      return v;
    };
    if (read(base + 2, 2) != 42)
      throw SerializationError("Invalid EXIF TIFF header");
    auto ifd = std::uint64_t(base) + read(base + 4, 4);
    if (ifd > data.size()) throw SerializationError("Invalid EXIF IFD offset");
    auto count = read(ifd, 2);
    if (count > (data.size() - ifd - 2) / 12)
      throw SerializationError("EXIF IFD directory exceeds APP1");
    for (std::uint32_t i = 0; i < count; ++i) {
      auto at = ifd + 2 + 12 * i;
      if (read(at, 2) == 0x112) {
        if (read(at + 2, 2) != 3 || read(at + 4, 4) != 1)
          throw SerializationError("Invalid EXIF orientation field");
        auto value = read(at + 8, 2);
        if (value < 1 || value > 8)
          throw SerializationError("EXIF orientation must be 1..8");
        return static_cast<std::uint8_t>(value);
      }
    }
    return 1;
  }
  throw ResourceExhausted("JPEG orientation marker bound exceeded");
}
im::VisionSample orient_vision_sample(im::VisionSample sample,
                                      const std::filesystem::path& path,
                                      ExifPolicy policy, std::stop_token stop) {
  if (stop.stop_requested()) throw Cancelled("Image orientation cancelled");
  if (policy == ExifPolicy::ignore) return sample;
  const auto orientation = image_exif_orientation(path);
  if (orientation == 1) return sample;
  if (policy == ExifPolicy::reject_non_identity)
    throw InvalidArgument(
        "Non-identity EXIF needs explicit encoded/already_oriented label "
        "coordinates: " +
        path.string());
  const auto w = sample.width, h = sample.height;
  const bool transpose = orientation >= 5;
  const im::ImageExtent output{transpose ? h : w, transpose ? w : h};
  auto axes = sample.pixels.layout == im::ImageLayout::chw
                  ? std::array<I, 2>{1, 2}
                  : std::array<I, 2>{0, 1};
  sample.image =
      Orient(sample.image, orientation, axes[0], axes[1]).contiguous();
  if (sample.paired_target || sample.mask_target)
    throw InvalidArgument(
        "Vision EXIF orientation requires object/classification sources "
        "without paired raster targets");
  if (sample.annotation) {
    auto record =
        std::make_shared<im::VisionAnnotationRecord>(*sample.annotation);
    if (policy == ExifPolicy::encoded_coordinates &&
        (record->width != w || record->height != h))
      throw InvalidArgument(
          "Encoded-coordinate annotations differ from encoded pixel extent");
    if (policy == ExifPolicy::already_oriented_coordinates) {
      if (record->width != output.width || record->height != output.height)
        throw InvalidArgument(
            "Display-coordinate annotations differ from EXIF oriented extent");
    } else
      for (auto& object : record->objects) {
        if (object.box) {
          auto b = *object.box;
          std::array<im::ImagePoint, 4> corners{{{b.x_min, b.y_min},
                                                 {b.x_max, b.y_min},
                                                 {b.x_max, b.y_max},
                                                 {b.x_min, b.y_max}}};
          for (auto& p : corners) p = Point(p, w, h, orientation);
          im::ImageBox box{static_cast<double>(output.width),
                           static_cast<double>(output.height), 0, 0};
          for (auto p : corners) {
            box.x_min = (std::min)(box.x_min, p.x);
            box.y_min = (std::min)(box.y_min, p.y);
            box.x_max = (std::max)(box.x_max, p.x);
            box.y_max = (std::max)(box.y_max, p.y);
          }
          object.box = box;
        }
        for (auto& ring : object.polygons)
          for (auto& p : ring) p = Point(p, w, h, orientation);
        for (auto& keypoint : object.keypoints)
          if (keypoint.visibility)
            keypoint.point = Point(keypoint.point, w, h, orientation);
        if (object.oriented_box)
          for (auto& p : *object.oriented_box) p = Point(p, w, h, orientation);
        if (object.rle) object.rle = Rle(*object.rle, orientation);
      }
    record->width = output.width;
    record->height = output.height;
    sample.annotation = std::move(record);
  }
  sample.width = output.width;
  sample.height = output.height;
  sample.geometry.emplace(output);
  return sample;
}
}  // namespace jyppx::liuding::vision::tasks
