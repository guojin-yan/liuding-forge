// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/data/composition.hpp>
#include <jyppx/liuding/vision/tasks/data/targets.hpp>

#include <torch/nn/functional/vision.h>

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
namespace jyppx::liuding::vision::tasks {
namespace {
namespace im = data::image;
using I = std::int64_t;
using Matrix = std::array<double, 9>;
void Stop(std::stop_token s) {
  if (s.stop_requested())
    throw Cancelled("Multi-source vision augmentation cancelled");
}
void Probability(double p) {
  if (!std::isfinite(p) || p < 0 || p > 1)
    throw InvalidArgument("Augmentation probability outside [0,1]");
}
std::string Hash(const nlohmann::json& j) {
  auto text = j.dump();
  return sha256_hex(std::as_bytes(std::span(text)));
}
Tensor Chw(const im::VisionSample& s) {
  return s.pixels.layout == im::ImageLayout::chw ? s.image
                                                 : s.image.permute({2, 0, 1});
}
void Check(const im::VisionSample& s, std::uint64_t max_pixels,
           std::size_t max_objects) {
  if (!s.image.defined() || !s.image.is_cpu() ||
      s.image.scalar_type() != torch::kUInt8 ||
      s.pixels.color != im::PixelColorModel::rgb || !s.width || !s.height ||
      s.width > 8192 || s.height > 8192 || s.paired_target || s.mask_target)
    throw InvalidArgument(
        "Composition requires bounded CPU RGB uint8 samples without "
        "paired/semantic targets");
  auto t = Chw(s);
  if (t.dim() != 3 || t.size(0) != 3 || t.size(1) != s.height ||
      t.size(2) != s.width)
    throw InvalidArgument("Composition pixel shape mismatch");
  if (static_cast<std::uint64_t>(s.width) * s.height > max_pixels ||
      (s.annotation && s.annotation->objects.size() > max_objects))
    throw ResourceExhausted("Composition pixel/object budget exceeded");
}
Matrix Multiply(const Matrix& a, const Matrix& b) {
  Matrix c{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        c[3 * i + j] += a[3 * i + k] * b[3 * k + j];
  return c;
}
void Sources(im::VisionSample& s) {
  if (!s.annotation) return;
  auto a = std::make_shared<im::VisionAnnotationRecord>(*s.annotation);
  for (auto& o : a->objects)
    if (o.augmentation_sources.empty())
      o.augmentation_sources.push_back(
          {.image_path = s.image_path,
           .object_id = o.source_id,
           .matrix = s.geometry ? s.geometry->matrix()
                                : Matrix{1, 0, 0, 0, 1, 0, 0, 0, 1}});
  s.annotation = std::move(a);
}
im::VisionSample Synthetic(const im::VisionSample& base, Tensor pixels,
                           std::vector<im::VisionObject> objects,
                           const std::string& identity) {
  auto result = base;
  result.image = std::move(pixels);
  result.pixels.layout = im::ImageLayout::chw;
  result.geometry.reset();
  result.width = static_cast<std::uint32_t>(result.image.size(2));
  result.height = static_cast<std::uint32_t>(result.image.size(1));
  result.image_path = "synthetic/" + identity;
  auto a = std::make_shared<im::VisionAnnotationRecord>();
  a->image_path = result.image_path;
  a->width = result.width;
  a->height = result.height;
  a->objects = std::move(objects);
  I id = 1;
  for (auto& object : a->objects) object.source_id = id++;
  result.annotation = std::move(a);
  // The source tensors remain immutable; output owns independent storage.
  result.image_resource_leases.reset();
  return result;
}
std::vector<im::ImagePoint> Clip(std::vector<im::ImagePoint> polygon, double w,
                                 double h) {
  for (int side = 0; side < 4; ++side) {
    if (polygon.empty()) break;
    auto input = std::move(polygon);
    polygon.clear();
    const auto distance = [&](im::ImagePoint p) {
      return side == 0 ? p.x : side == 1 ? w - p.x : side == 2 ? p.y : h - p.y;
    };
    auto a = input.back();
    double da = distance(a);
    for (auto b : input) {
      const double db = distance(b);
      if ((da >= 0) != (db >= 0)) {
        const double t = da / (da - db);
        polygon.push_back({a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)});
      }
      if (db >= 0) polygon.push_back(b);
      a = b;
      da = db;
    }
  }
  return polygon;
}
double Area(const std::vector<im::ImagePoint>& p) {
  double v = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const auto& a = p[i];
    const auto& b = p[(i + 1) % p.size()];
    v += a.x * b.y - a.y * b.x;
  }
  return std::abs(v) * .5;
}
im::ImageMaskRle Rle(const Tensor& input) {
  auto m = input.to(torch::kCPU, torch::kBool).contiguous();
  auto p = m.accessor<bool, 2>();
  im::ImageMaskRle r{static_cast<std::uint32_t>(m.size(1)),
                     static_cast<std::uint32_t>(m.size(0)),
                     {}};
  bool flag = false;
  std::uint64_t run = 0;
  for (I x = 0; x < m.size(1); ++x)
    for (I y = 0; y < m.size(0); ++y)
      if (p[y][x] == flag)
        ++run;
      else {
        r.counts.push_back(run);
        run = 1;
        flag = !flag;
      }
  r.counts.push_back(run);
  return r;
}
Tensor Mask(const im::VisionObject& object, I h, I w) {
  auto result = torch::zeros({h, w}, torch::kFloat32);
  auto p = result.accessor<float, 2>();
  if (object.rle) {
    if (object.rle->height != h || object.rle->width != w)
      throw InvalidArgument("Composition RLE size mismatch");
    std::uint64_t cursor = 0;
    bool flag = false;
    for (auto run : object.rle->counts) {
      if (run > static_cast<std::uint64_t>(h * w) - cursor)
        throw InvalidArgument("Composition RLE overrun");
      if (flag)
        for (std::uint64_t i = 0; i < run; ++i)
          p[(cursor + i) % h][(cursor + i) / h] = 1;
      cursor += run;
      flag = !flag;
    }
    if (cursor != static_cast<std::uint64_t>(h * w))
      throw InvalidArgument("Composition RLE incomplete");
  } else {
    if (object.polygons.empty()) {
      if (!object.box)
        throw InvalidArgument("Composition object lacks box/region geometry");
      const auto b = *object.box;
      for (I y = (std::max)(I{0}, static_cast<I>(std::ceil(b.y_min - .5)));
           y < (std::min)(h, static_cast<I>(std::ceil(b.y_max - .5))); ++y)
        for (I x = (std::max)(I{0}, static_cast<I>(std::ceil(b.x_min - .5)));
             x < (std::min)(w, static_cast<I>(std::ceil(b.x_max - .5))); ++x)
          p[y][x] = 1;
      return result;
    }
    for (I y = 0; y < h; ++y)
      for (const auto& ring : object.polygons) {
        std::vector<double> xs;
        for (std::size_t i = 0; i < ring.size(); ++i) {
          auto a = ring[i], b = ring[(i + 1) % ring.size()];
          if ((a.y <= y + .5 && b.y > y + .5) ||
              (b.y <= y + .5 && a.y > y + .5))
            xs.push_back(a.x + (y + .5 - a.y) * (b.x - a.x) / (b.y - a.y));
        }
        std::sort(xs.begin(), xs.end());
        for (std::size_t i = 0; i + 1 < xs.size(); i += 2)
          for (I x = (std::max)(I{0}, static_cast<I>(std::ceil(xs[i] - .5)));
               x < (std::min)(w, static_cast<I>(std::ceil(xs[i + 1] - .5)));
               ++x)
            p[y][x] = 1;
      }
  }
  return result;
}
void RasterObject(im::VisionObject& object, const Tensor& mask) {
  auto positive = (mask > 0.5).nonzero();
  if (!positive.numel()) {
    object.box.reset();
    return;
  }
  auto lo = std::get<0>(positive.min(0)), hi = std::get<0>(positive.max(0));
  object.box = im::ImageBox{lo[1].item<double>(), lo[0].item<double>(),
                            hi[1].item<double>() + 1, hi[0].item<double>() + 1};
  object.rle = Rle(mask > .5);
  object.mask_pixel_area = positive.size(0);
  object.area = static_cast<double>(positive.size(0));
  object.polygons.clear();
}
}  // namespace
std::string multisource_identity(const MultiSourceOptions& o) {
  for (double p :
       {o.mosaic_probability, o.mixup_probability, o.cutmix_probability,
        o.copypaste_probability, o.max_occlusion})
    Probability(p);
  if (!o.total_epochs || o.close_epochs > o.total_epochs || !o.max_objects ||
      o.max_objects > 100000 || !o.max_pixels)
    throw InvalidArgument("Invalid multi-source schedule/budget");
  return Hash({"multisource/quadrant-uniform-v1", o.mosaic_probability,
               o.mixup_probability, o.cutmix_probability,
               o.copypaste_probability, o.total_epochs, o.close_epochs,
               o.max_occlusion, o.max_objects, o.max_pixels});
}
im::VisionSample augment_multisource(
    const im::VisionSample& primary, std::size_t count,
    const SampleLoader& load, const MultiSourceOptions& o,
    const LetterBoxOptions& letterbox, const im::SampleRandom& random,
    std::uint64_t epoch, std::stop_token stop) {
  auto identity = multisource_identity(o);
  Check(primary, o.max_pixels, o.max_objects);
  Stop(stop);
  if (epoch >= o.total_epochs - o.close_epochs ||
      (o.mosaic_probability == 0 && o.mixup_probability == 0 &&
       o.cutmix_probability == 0 && o.copypaste_probability == 0))
    return primary;
  if (static_cast<std::uint64_t>(letterbox.canvas.width) *
          letterbox.canvas.height >
      o.max_pixels)
    throw ResourceExhausted("Composition output canvas exceeds pixel bound");
  if (!count || !load)
    throw InvalidArgument(
        "Multi-source augmentation requires an identified source loader");
  auto result = primary;
  const auto partner = [&](std::uint64_t draw) {
    auto s = load(static_cast<std::size_t>(random.bounded(count, draw)), stop);
    Check(s, o.max_pixels, o.max_objects);
    Sources(s);
    if (s.categories && primary.categories) {
      if (s.categories->size() != primary.categories->size())
        throw InvalidArgument("Composition class map mismatch");
      for (std::size_t i = 0; i < s.categories->size(); ++i)
        if ((*s.categories)[i].name != (*primary.categories)[i].name ||
            (*s.categories)[i].source_id != (*primary.categories)[i].source_id)
          throw InvalidArgument("Composition class order mismatch");
    }
    return s;
  };
  if (random.unit(0) < o.mosaic_probability) {
    const auto w = letterbox.canvas.width, h = letterbox.canvas.height;
    const auto cx = static_cast<std::uint32_t>(w * (.35 + .3 * random.unit(1))),
               cy = static_cast<std::uint32_t>(h * (.35 + .3 * random.unit(2)));
    auto pixels =
        torch::full({3, h, w}, letterbox.padding_value, torch::kUInt8);
    std::vector<im::VisionObject> objects;
    for (std::uint32_t tile = 0; tile < 4; ++tile) {
      auto s = tile ? partner(3 + tile) : primary;
      Sources(s);
      const auto x = tile % 2 ? cx : 0, y = tile / 2 ? cy : 0,
                 tw = tile % 2 ? w - cx : cx, th = tile / 2 ? h - cy : cy;
      auto target = letterbox;
      target.canvas = {tw, th};
      s = letterbox_sample(s, target, stop);
      auto options = target.geometry;
      options.image_padding = letterbox.padding_value;
      const std::array steps{
          im::ImageGeometryStep{.kind = im::ImageGeometryKind::pad,
                                .left = x,
                                .top = y,
                                .right = w - x - tw,
                                .bottom = h - y - th}};
      auto placed = im::transform_sample(s, steps, options, stop);
      pixels.slice(1, y, y + th)
          .slice(2, x, x + tw)
          .copy_(Chw(placed).slice(1, y, y + th).slice(2, x, x + tw));
      if (placed.annotation)
        objects.insert(objects.end(), placed.annotation->objects.begin(),
                       placed.annotation->objects.end());
      if (objects.size() > o.max_objects)
        throw ResourceExhausted("Mosaic object bound exceeded");
    }
    result = Synthetic(primary, pixels, std::move(objects),
                       random.identity() + "-mosaic");
  }
  if (random.unit(10) < o.mixup_probability) {
    Sources(result);
    auto other = letterbox_sample(partner(11), letterbox, stop);
    auto base = letterbox_sample(result, letterbox, stop);
    const double a = .35 + .3 * random.unit(12);
    auto pixels = (Chw(base).to(torch::kFloat32) * a +
                   Chw(other).to(torch::kFloat32) * (1 - a))
                      .round()
                      .clamp(0, 255)
                      .to(torch::kUInt8);
    std::vector<im::VisionObject> objects =
        base.annotation ? base.annotation->objects
                        : std::vector<im::VisionObject>{};
    for (auto& object : objects)
      for (auto& source : object.augmentation_sources) source.weight *= a;
    if (other.annotation)
      for (auto object : other.annotation->objects) {
        for (auto& source : object.augmentation_sources) source.weight *= 1 - a;
        objects.push_back(std::move(object));
      }
    if (objects.size() > o.max_objects)
      throw ResourceExhausted("MixUp object bound exceeded");
    result = Synthetic(primary, pixels, std::move(objects),
                       random.identity() + "-mixup");
  }
  const bool cut = random.unit(20) < o.cutmix_probability,
             paste = random.unit(30) < o.copypaste_probability;
  if (cut || paste) {
    Sources(result);
    auto base = letterbox_sample(result, letterbox, stop),
         other = letterbox_sample(partner(21), letterbox, stop);
    auto pixels = Chw(base).clone();
    const I h = base.height, w = base.width;
    auto selection = torch::zeros({h, w}, torch::kBool);
    if (cut) {
      const I x = static_cast<I>(random.unit(22) * w * .7),
              y = static_cast<I>(random.unit(23) * h * .7);
      const I cw = (std::max)(I{1},
                              static_cast<I>((.2 + .4 * random.unit(24)) * w)),
              ch = (std::max)(I{1},
                              static_cast<I>((.2 + .4 * random.unit(25)) * h));
      selection.slice(0, y, (std::min)(h, y + ch))
          .slice(1, x, (std::min)(w, x + cw))
          .fill_(true);
    }
    std::vector<im::VisionObject> objects =
        base.annotation ? base.annotation->objects
                        : std::vector<im::VisionObject>{};
    if (other.annotation)
      for (auto object : other.annotation->objects) {
        Stop(stop);
        if (!object.keypoints.empty() || object.oriented_box)
          throw InvalidArgument(
              "CutMix/CopyPaste does not accept pose/OBB targets");
        if (paste && !cut && object.polygons.empty() && !object.rle)
          throw InvalidArgument("CopyPaste requires genuine instance regions");
        auto mask = Mask(object, h, w) > .5;
        if (cut) mask = mask & selection;
        bool accept = mask.any().item<bool>();
        if (paste && !cut) {
          for (const auto& existing : objects) {
            auto receiver = Mask(existing, h, w) > .5;
            const auto area = receiver.sum().item<double>();
            if (area > 0 && (mask & receiver).sum().item<double>() / area >
                                o.max_occlusion) {
              accept = false;
              break;
            }
          }
        }
        if (!accept) continue;
        pixels = torch::where(mask.unsqueeze(0), Chw(other), pixels);
        selection |= mask;
        RasterObject(object, mask);
        if (object.box) objects.push_back(std::move(object));
      }
    const auto originals =
        base.annotation ? base.annotation->objects.size() : 0U;
    std::vector<im::VisionObject> visible;
    for (std::size_t i = 0; i < objects.size(); ++i) {
      auto object = objects[i];
      if (i < originals) {
        if (!object.keypoints.empty() || object.oriented_box)
          throw InvalidArgument(
              "CutMix/CopyPaste does not accept pose/OBB targets");
        auto mask = Mask(object, h, w) > .5;
        mask = mask & (~selection);
        RasterObject(object, mask);
      }
      if (object.box) visible.push_back(std::move(object));
    }
    if (visible.size() > o.max_objects)
      throw ResourceExhausted("CutMix/CopyPaste object bound exceeded");
    result = Synthetic(primary, pixels, std::move(visible),
                       random.identity() + "-paste");
  }
  Stop(stop);
  return result;
}
std::string projective_identity(const ProjectiveOptions& o) {
  if (!std::isfinite(o.degrees) || o.degrees < 0 || o.degrees > 180 ||
      !std::isfinite(o.shear_degrees) || o.shear_degrees < 0 ||
      o.shear_degrees > 45 || !std::isfinite(o.perspective) ||
      o.perspective < 0 || o.perspective > .001 ||
      !std::isfinite(o.scale_gain) || o.scale_gain < 0 || o.scale_gain > .9 ||
      !std::isfinite(o.translate_fraction) || o.translate_fraction < 0 ||
      o.translate_fraction > .5 || !std::isfinite(o.min_box_area) ||
      o.min_box_area < 0 || !o.max_pixels || !o.max_objects)
    throw InvalidArgument("Invalid projective augmentation");
  Probability(o.min_visibility);
  return Hash({"projective/bilinear-boundary-v1", o.degrees, o.shear_degrees,
               o.perspective, o.scale_gain, o.translate_fraction,
               o.min_box_area, o.min_visibility, o.max_pixels, o.max_objects});
}
im::VisionSample augment_projective(const im::VisionSample& source,
                                    const ProjectiveOptions& o,
                                    const im::SampleRandom& random,
                                    double padding, std::stop_token stop) {
  static_cast<void>(projective_identity(o));
  Check(source, o.max_pixels, o.max_objects);
  if (!std::isfinite(padding) || padding < 0 || padding > 255)
    throw InvalidArgument("Projective padding out of uint8 range");
  if (o.degrees == 0 && o.shear_degrees == 0 && o.perspective == 0 &&
      o.scale_gain == 0 && o.translate_fraction == 0)
    return source;
  auto s = source;
  Sources(s);
  const double w = s.width, h = s.height;
  const double angle =
                   (2 * random.unit(0) - 1) * o.degrees * std::acos(-1.) / 180,
               scale = 1 + (2 * random.unit(1) - 1) * o.scale_gain;
  Matrix center{1, 0, -w * .5, 0, 1, -h * .5, 0, 0, 1},
      rotation{scale * std::cos(angle),
               -scale * std::sin(angle),
               0,
               scale * std::sin(angle),
               scale * std::cos(angle),
               0,
               0,
               0,
               1};
  Matrix shear{
      1,
      std::tan((2 * random.unit(2) - 1) * o.shear_degrees * std::acos(-1.) / 180),
      0,
      std::tan((2 * random.unit(3) - 1) * o.shear_degrees * std::acos(-1.) / 180),
      1,
      0,
      0,
      0,
      1};
  Matrix perspective{1,
                     0,
                     0,
                     0,
                     1,
                     0,
                     (2 * random.unit(4) - 1) * o.perspective,
                     (2 * random.unit(5) - 1) * o.perspective,
                     1};
  Matrix translate{
      1, 0, w * (.5 + (2 * random.unit(6) - 1) * o.translate_fraction),
      0, 1, h * (.5 + (2 * random.unit(7) - 1) * o.translate_fraction),
      0, 0, 1};
  auto matrix = Multiply(
      translate,
      Multiply(perspective, Multiply(shear, Multiply(rotation, center))));
  auto m = torch::tensor(std::vector<double>(matrix.begin(), matrix.end()),
                         torch::kFloat64)
               .view({3, 3});
  auto inverse = torch::linalg_inv(m).to(torch::kFloat32);
  auto xs = (torch::arange(static_cast<I>(w), torch::kFloat32) + .5)
                .view({1, -1})
                .expand({static_cast<I>(h), static_cast<I>(w)});
  auto ys = (torch::arange(static_cast<I>(h), torch::kFloat32) + .5)
                .view({-1, 1})
                .expand_as(xs);
  auto points = torch::stack({xs, ys, torch::ones_like(xs)}, -1)
                    .matmul(inverse.transpose(0, 1));
  auto denominator = points.select(2, 2);
  if ((denominator.abs() < 1e-6).any().item<bool>())
    throw InvalidArgument("Projective warp crosses its horizon");
  auto grid = torch::stack({points.select(2, 0) / denominator * (2 / w) - 1,
                            points.select(2, 1) / denominator * (2 / h) - 1},
                           2)
                  .unsqueeze(0);
  auto pixels = torch::nn::functional::grid_sample(
                    (Chw(s).to(torch::kFloat32) - padding).unsqueeze(0), grid,
                    torch::nn::functional::GridSampleFuncOptions()
                        .mode(torch::kBilinear)
                        .padding_mode(torch::kZeros)
                        .align_corners(false))
                    .squeeze(0)
                    .add(padding)
                    .round()
                    .clamp(0, 255)
                    .to(torch::kUInt8);
  const auto map = [&](im::ImagePoint p) {
    const double z = matrix[6] * p.x + matrix[7] * p.y + matrix[8];
    if (z <= 1e-6)
      throw InvalidArgument("Projective target crosses its horizon");
    return im::ImagePoint{(matrix[0] * p.x + matrix[1] * p.y + matrix[2]) / z,
                          (matrix[3] * p.x + matrix[4] * p.y + matrix[5]) / z};
  };
  std::vector<im::VisionObject> objects;
  if (s.annotation)
    for (auto object : s.annotation->objects) {
      Stop(stop);
      if (!object.box) continue;
      if (object.oriented_box && (o.shear_degrees > 0 || o.perspective > 0))
        throw InvalidArgument(
            "OBB cannot use shear/perspective without changing its rectangle "
            "semantics");
      const auto b = *object.box;
      std::vector<im::ImagePoint> box{
          map({b.x_min, b.y_min}), map({b.x_max, b.y_min}),
          map({b.x_max, b.y_max}), map({b.x_min, b.y_max})};
      auto clipped = Clip(box, w, h);
      if (clipped.size() < 3 || Area(clipped) < o.min_box_area ||
          Area(clipped) < Area(box) * o.min_visibility)
        continue;
      im::ImageBox bounds{w, h, 0, 0};
      for (auto p : clipped) {
        bounds.x_min = (std::min)(bounds.x_min, p.x);
        bounds.y_min = (std::min)(bounds.y_min, p.y);
        bounds.x_max = (std::max)(bounds.x_max, p.x);
        bounds.y_max = (std::max)(bounds.y_max, p.y);
      }
      object.box = bounds;
      std::vector<std::vector<im::ImagePoint>> rings;
      for (auto ring : object.polygons) {
        for (auto& p : ring) p = map(p);
        ring = Clip(std::move(ring), w, h);
        if (ring.size() >= 3) rings.push_back(std::move(ring));
      }
      object.polygons = std::move(rings);
      if (object.oriented_box) {
        bool outside = false;
        for (auto& p : *object.oriented_box) {
          p = map(p);
          outside |= p.x < 0 || p.y < 0 || p.x > w || p.y > h;
        }
        if (outside) continue;
      }
      for (auto& k : object.keypoints)
        if (k.visibility) {
          k.point = map(k.point);
          if (k.point.x < 0 || k.point.y < 0 || k.point.x > w || k.point.y > h)
            k = {{0, 0}, 0};
        }
      if (object.rle) {
        // Rasterize the original RLE; polygon coordinates above are already
        // transformed.
        im::VisionObject original;
        original.rle = object.rle;
        auto mask = Mask(original, static_cast<I>(h), static_cast<I>(w));
        auto warped =
            torch::nn::functional::grid_sample(
                mask.view({1, 1, static_cast<I>(h), static_cast<I>(w)}), grid,
                torch::nn::functional::GridSampleFuncOptions()
                    .mode(torch::kNearest)
                    .padding_mode(torch::kZeros)
                    .align_corners(false))
                .squeeze();
        RasterObject(object, warped);
        if (!object.box) continue;
      }
      for (auto& mapping : object.augmentation_sources)
        mapping.matrix = Multiply(matrix, mapping.matrix);
      objects.push_back(std::move(object));
    }
  return Synthetic(source, pixels, std::move(objects),
                   random.identity() + "-warp");
}
}  // namespace jyppx::liuding::vision::tasks
