// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "image_batch_detail.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <new>

namespace jyppx::liuding::data::image {
namespace {
using I = c10::IValue;
using Values = std::vector<I>;
using detail::BatchLimit;
using detail::InvalidBatch;

struct Budget {
  const ImageBatchLimits& limits;
  std::size_t nodes = 0;
  std::size_t bytes = 0;
  std::uint64_t tensor_bytes = 0;
  void Node(std::size_t payload = 0) {
    if (nodes == limits.max_metadata_nodes ||
        payload > limits.max_metadata_bytes - bytes)
      BatchLimit("batch.metadata");
    ++nodes;
    bytes += payload;
  }
};

// Only scalar/list/tuple metadata: the data memory stage never transfers
// annotation coordinates, source IDs, extents, offsets or geometry state.
struct Writer {
  Budget budget;
  I Int(std::int64_t v) {
    budget.Node(8);
    return I(v);
  }
  I Unsigned(std::uint64_t v) { return Int(std::bit_cast<std::int64_t>(v)); }
  I Size(std::size_t v) {
    if (v > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
      BatchLimit("batch.integer");
    return Int(static_cast<std::int64_t>(v));
  }
  I Real(double v) {
    budget.Node(8);
    return I(v);
  }
  I Bool(bool v) {
    budget.Node(1);
    return I(v);
  }
  I None() {
    budget.Node();
    return I();
  }
  I String(const std::string& v) {
    if (v.size() > budget.limits.max_string_bytes) BatchLimit("batch.string");
    budget.Node(v.size());
    return I(v);
  }
  I Tuple(Values values) {
    budget.Node();
    return I(c10::ivalue::Tuple::create(std::move(values)));
  }
  I List(Values values) {
    budget.Node();
    c10::impl::GenericList result(c10::AnyType::get());
    for (auto& v : values) result.push_back(std::move(v));
    return I(result);
  }
  I OptionalInt(const std::optional<std::int64_t>& v) {
    return v ? Int(*v) : None();
  }
  I OptionalString(const std::optional<std::string>& v) {
    return v ? String(*v) : None();
  }
  I Point(ImagePoint p) { return Tuple({Real(p.x), Real(p.y)}); }
  I Box(const std::optional<ImageBox>& b) {
    return b ? Tuple({Real(b->x_min), Real(b->y_min), Real(b->x_max),
                      Real(b->y_max)})
             : None();
  }
  I Pixels(const PixelSpec& p) {
    return Tuple({Int(static_cast<std::int64_t>(p.dtype)),
                  Int(static_cast<std::int64_t>(p.color)),
                  Int(static_cast<std::int64_t>(p.alpha)),
                  Int(static_cast<std::int64_t>(p.range))});
  }
  I Categories(
      const std::shared_ptr<const std::vector<VisionCategory>>& categories) {
    if (!categories) return None();
    Values result;
    for (const auto& c : *categories) {
      Values names, skeleton;
      if (c.name.empty()) InvalidBatch("batch.category_name");
      for (const auto& name : c.keypoint_names) names.push_back(String(name));
      for (const auto& link : c.skeleton) {
        if (link[0] >= c.keypoint_names.size() ||
            link[1] >= c.keypoint_names.size())
          InvalidBatch("batch.skeleton");
        skeleton.push_back(Tuple({Size(link[0]), Size(link[1])}));
      }
      result.push_back(
          Tuple({Int(c.source_id), String(c.name), Bool(c.is_thing),
                 List(std::move(names)), List(std::move(skeleton))}));
    }
    return List(std::move(result));
  }
  I Object(const VisionObject& o) {
    Values polygons, keypoints;
    for (const auto& ring : o.polygons) {
      Values points;
      for (const auto p : ring) points.push_back(Point(p));
      polygons.push_back(List(std::move(points)));
    }
    I obb = None(), rle = None();
    if (o.oriented_box) {
      Values points;
      for (const auto p : *o.oriented_box) points.push_back(Point(p));
      obb = Tuple(std::move(points));
    }
    if (o.rle) {
      Values runs;
      for (const auto run : o.rle->counts) runs.push_back(Unsigned(run));
      rle =
          Tuple({Int(o.rle->width), Int(o.rle->height), List(std::move(runs))});
    }
    for (const auto& p : o.keypoints)
      keypoints.push_back(
          Tuple({Real(p.point.x), Real(p.point.y), Int(p.visibility)}));
    return Tuple({Int(o.source_id), Size(o.category), Box(o.box),
                  List(std::move(polygons)), std::move(obb), std::move(rle),
                  List(std::move(keypoints)), o.area ? Real(*o.area) : None(),
                  o.mask_pixel_area ? Unsigned(*o.mask_pixel_area) : None(),
                  Bool(o.oriented_box_to_polygon), Bool(o.polygon_partitioned),
                  OptionalInt(o.group_id), Bool(o.crowd), Bool(o.ignore),
                  Bool(o.difficult), Bool(o.truncated)});
  }
  I Annotation(const std::shared_ptr<const VisionAnnotationRecord>& r) {
    if (!r) return None();
    Values objects;
    for (const auto& o : r->objects) objects.push_back(Object(o));
    return Tuple({Int(r->source_image_id), String(r->image_path), Int(r->width),
                  Int(r->height), List(std::move(objects)),
                  OptionalString(r->panoptic_mask_path),
                  Bool(r->panoptic_mask_materialized)});
  }
  I Trace(const std::optional<GeometryTrace>& trace) {
    if (!trace) return None();
    const auto s = trace->state();
    return Tuple({Int(s.source.width), Int(s.source.height),
                  Int(s.output.width), Int(s.output.height),
                  Box(s.source_region),
                  Tuple({Real(s.coefficients[0]), Real(s.coefficients[1]),
                         Real(s.coefficients[2]), Real(s.coefficients[3])}),
                  Int(s.operations)});
  }
  I Mask(const std::optional<ImageBatchMaskInfo>& m) {
    if (!m) return None();
    Values palette;
    for (const auto& p : m->palette)
      palette.push_back(
          Tuple({Int(p.rgb[0]), Int(p.rgb[1]), Int(p.rgb[2]), Int(p.alpha)}));
    return Tuple({Int(static_cast<std::int64_t>(m->encoding)),
                  Int(m->source_bit_depth), Bool(m->source_interlaced),
                  List(std::move(palette)), OptionalInt(m->ignore_id),
                  OptionalInt(m->void_id)});
  }
  I Item(const ImageBatchItem& item) {
    if (item.key.id.empty() || item.key.id.size() > 4096)
      InvalidBatch("batch.sample_id");
    return Tuple({String(item.key.id), Unsigned(item.key.occurrence),
                  Unsigned(item.key.view), Int(item.valid_extent.width),
                  Int(item.valid_extent.height), String(item.image_path),
                  OptionalString(item.target_path),
                  Categories(item.label_categories),
                  Categories(item.categories), Annotation(item.annotation),
                  Trace(item.geometry), Mask(item.mask)});
  }
};

ValueSchema Kind(ValueKind kind, bool nullable = false) {
  ValueSchema value;
  value.kind = kind;
  value.nullable = nullable;
  return value;
}
ValueSchema Optional(ValueSchema v) {
  v.nullable = true;
  return v;
}
ValueSchema Tuple(std::vector<ValueSchema> fields) {
  return ValueSchema::tuple(std::move(fields));
}
ValueSchema List(ValueSchema entry) {
  return ValueSchema::list(std::move(entry));
}

ValueSchema ItemSchema() {
  const auto integer = Kind(ValueKind::integer),
             real = Kind(ValueKind::floating),
             boolean = Kind(ValueKind::boolean),
             string = Kind(ValueKind::string);
  const auto point = Tuple({real, real}),
             box = Optional(Tuple({real, real, real, real}));
  const auto category = Tuple({integer, string, boolean, List(string),
                               List(Tuple({integer, integer}))});
  const auto categories = Optional(List(category));
  const auto object =
      Tuple({integer, integer, box, List(List(point)),
             Optional(Tuple({point, point, point, point})),
             Optional(Tuple({integer, integer, List(integer)})),
             List(Tuple({real, real, integer})), Optional(real),
             Optional(integer), boolean, boolean, Optional(integer), boolean,
             boolean, boolean, boolean});
  const auto annotation =
      Optional(Tuple({integer, string, integer, integer, List(object),
                      Optional(string), boolean}));
  const auto trace =
      Optional(Tuple({integer, integer, integer, integer, box,
                      Tuple({real, real, real, real}), integer}));
  const auto mask =
      Optional(Tuple({integer, integer, boolean,
                      List(Tuple({integer, integer, integer, integer})),
                      Optional(integer), Optional(integer)}));
  return Tuple({string, integer, integer, integer, integer, string,
                Optional(string), categories, categories, annotation, trace,
                mask});
}

// Accessors are used only after the complete ValueSchema has accepted types
// and arities. Range checks still apply to every narrowed integer.
const auto& Fields(const I& v) { return v.toTupleRef().elements(); }
std::uint64_t Unsigned(const I& v) {
  return std::bit_cast<std::uint64_t>(v.toInt());
}
std::uint32_t Dimension(const I& v) {
  const auto n = v.toInt();
  if (n <= 0 || n > std::numeric_limits<std::uint32_t>::max())
    InvalidBatch("batch.dimension");
  return static_cast<std::uint32_t>(n);
}
std::size_t Index(const I& v) {
  const auto n = v.toInt();
  if (n < 0 ||
      static_cast<std::uint64_t>(n) > std::numeric_limits<std::size_t>::max())
    InvalidBatch("batch.index");
  return static_cast<std::size_t>(n);
}
std::uint8_t Byte(const I& v) {
  const auto n = v.toInt();
  if (n < 0 || n > 255) InvalidBatch("batch.byte");
  return static_cast<std::uint8_t>(n);
}
std::optional<std::int64_t> OptionalInt(const I& v) {
  return v.isNone() ? std::nullopt : std::optional<std::int64_t>(v.toInt());
}
std::optional<std::string> OptionalString(const I& v) {
  return v.isNone() ? std::nullopt
                    : std::optional<std::string>(v.toStringRef());
}
ImagePoint Point(const I& v) {
  const auto& a = Fields(v);
  return {a[0].toDouble(), a[1].toDouble()};
}
std::optional<ImageBox> Box(const I& v) {
  if (v.isNone()) return {};
  const auto& a = Fields(v);
  return ImageBox{a[0].toDouble(), a[1].toDouble(), a[2].toDouble(),
                  a[3].toDouble()};
}
PixelSpec Pixels(const I& v) {
  const auto& a = Fields(v);
  if (a[0].toInt() < 0 ||
      a[0].toInt() >= static_cast<std::int64_t>(torch::ScalarType::NumOptions))
    InvalidBatch("batch.dtype");
  if (a[1].toInt() < 0 || a[1].toInt() > 5 || a[2].toInt() < 0 ||
      a[2].toInt() > 2 || a[3].toInt() < 0 || a[3].toInt() > 2)
    InvalidBatch("batch.pixel_enum");
  PixelSpec p;
  p.dtype = static_cast<Dtype>(a[0].toInt());
  p.color = static_cast<PixelColorModel>(a[1].toInt());
  p.alpha = static_cast<PixelAlphaMode>(a[2].toInt());
  p.range = static_cast<PixelValueRange>(a[3].toInt());
  return p;
}
std::shared_ptr<const std::vector<VisionCategory>> Categories(const I& v) {
  if (v.isNone()) return {};
  auto result = std::make_shared<std::vector<VisionCategory>>();
  for (const auto& entry : v.toListRef()) {
    const auto& a = Fields(entry);
    VisionCategory c;
    c.source_id = a[0].toInt();
    c.name = a[1].toStringRef();
    c.is_thing = a[2].toBool();
    if (c.name.empty()) InvalidBatch("batch.category_name");
    for (const auto& name : a[3].toListRef())
      c.keypoint_names.push_back(name.toStringRef());
    for (const auto& link : a[4].toListRef()) {
      const auto& p = Fields(link);
      if (Index(p[0]) >= c.keypoint_names.size() ||
          Index(p[1]) >= c.keypoint_names.size())
        InvalidBatch("batch.skeleton");
      c.skeleton.push_back({Index(p[0]), Index(p[1])});
    }
    result->push_back(std::move(c));
  }
  return result;
}
VisionObject Object(const I& v) {
  const auto& a = Fields(v);
  VisionObject o;
  o.source_id = a[0].toInt();
  o.category = Index(a[1]);
  o.box = Box(a[2]);
  for (const auto& ring : a[3].toListRef()) {
    std::vector<ImagePoint> points;
    for (const auto& p : ring.toListRef()) points.push_back(Point(p));
    o.polygons.push_back(std::move(points));
  }
  if (!a[4].isNone()) {
    std::array<ImagePoint, 4> points;
    for (std::size_t i = 0; i < 4; ++i) points[i] = Point(Fields(a[4])[i]);
    o.oriented_box = points;
  }
  if (!a[5].isNone()) {
    const auto& r = Fields(a[5]);
    ImageMaskRle rle{Dimension(r[0]), Dimension(r[1]), {}};
    for (const auto& run : r[2].toListRef())
      rle.counts.push_back(Unsigned(run));
    o.rle = std::move(rle);
  }
  for (const auto& p : a[6].toListRef()) {
    const auto& k = Fields(p);
    if (k[2].toInt() < 0 || k[2].toInt() > 2) InvalidBatch("batch.visibility");
    o.keypoints.push_back({{k[0].toDouble(), k[1].toDouble()}, Byte(k[2])});
  }
  if (!a[7].isNone()) o.area = a[7].toDouble();
  if (!a[8].isNone()) o.mask_pixel_area = Unsigned(a[8]);
  o.oriented_box_to_polygon = a[9].toBool();
  o.polygon_partitioned = a[10].toBool();
  o.group_id = OptionalInt(a[11]);
  o.crowd = a[12].toBool();
  o.ignore = a[13].toBool();
  o.difficult = a[14].toBool();
  o.truncated = a[15].toBool();
  return o;
}
std::shared_ptr<const VisionAnnotationRecord> Annotation(const I& v) {
  if (v.isNone()) return {};
  const auto& a = Fields(v);
  auto r = std::make_shared<VisionAnnotationRecord>();
  r->source_image_id = a[0].toInt();
  r->image_path = a[1].toStringRef();
  r->width = Dimension(a[2]);
  r->height = Dimension(a[3]);
  for (const auto& o : a[4].toListRef()) r->objects.push_back(Object(o));
  r->panoptic_mask_path = OptionalString(a[5]);
  r->panoptic_mask_materialized = a[6].toBool();
  return r;
}
std::optional<GeometryTrace> Trace(const I& v) {
  if (v.isNone()) return {};
  const auto& a = Fields(v);
  const auto operations = a[6].toInt();
  if (operations < 0 || operations > 128)
    InvalidBatch("batch.trace_operations");
  GeometryTraceState s;
  s.source = {Dimension(a[0]), Dimension(a[1])};
  s.output = {Dimension(a[2]), Dimension(a[3])};
  s.source_region = Box(a[4]);
  s.operations = static_cast<std::uint16_t>(operations);
  for (std::size_t i = 0; i < 4; ++i)
    s.coefficients[i] = Fields(a[5])[i].toDouble();
  return GeometryTrace::from_state(s);
}
std::optional<ImageBatchMaskInfo> Mask(const I& v) {
  if (v.isNone()) return {};
  const auto& a = Fields(v);
  if (a[0].toInt() < 0 || a[0].toInt() > 3) InvalidBatch("batch.mask_encoding");
  ImageBatchMaskInfo m;
  m.encoding = static_cast<MaskEncoding>(a[0].toInt());
  m.source_bit_depth = Byte(a[1]);
  m.source_interlaced = a[2].toBool();
  for (const auto& p : a[3].toListRef()) {
    const auto& q = Fields(p);
    m.palette.push_back({{Byte(q[0]), Byte(q[1]), Byte(q[2])}, Byte(q[3])});
  }
  m.ignore_id = OptionalInt(a[4]);
  m.void_id = OptionalInt(a[5]);
  return m;
}
ImageBatchItem Item(const I& v) {
  const auto& a = Fields(v);
  ImageBatchItem item;
  item.key = {a[0].toStringRef(), Unsigned(a[1]), Unsigned(a[2])};
  if (item.key.id.empty() || item.key.id.size() > 4096)
    InvalidBatch("batch.sample_id");
  item.valid_extent = {Dimension(a[3]), Dimension(a[4])};
  item.image_path = a[5].toStringRef();
  item.target_path = OptionalString(a[6]);
  item.label_categories = Categories(a[7]);
  item.categories = Categories(a[8]);
  item.annotation = Annotation(a[9]);
  item.geometry = Trace(a[10]);
  item.mask = Mask(a[11]);
  return item;
}

void Inspect(const I& value, Budget& budget, std::size_t depth = 0) {
  if (depth > 32) BatchLimit("batch.value_depth");
  budget.Node(value.isString() ? value.toStringRef().size() : 8);
  if (value.isString() &&
      value.toStringRef().size() > budget.limits.max_string_bytes)
    BatchLimit("batch.string");
  if (value.isTensor()) {
    const auto t = value.toTensor();
    if (!t.defined() || t.layout() != torch::kStrided)
      InvalidBatch("batch.tensor");
    const auto count = static_cast<std::uint64_t>(t.numel());
    if (count > (budget.limits.max_output_bytes - budget.tensor_bytes) /
                    t.element_size())
      BatchLimit("batch.tensor_bytes");
    const auto bytes = count * t.element_size();
    budget.tensor_bytes += bytes;
  } else if (value.isTuple()) {
    for (const auto& v : Fields(value)) Inspect(v, budget, depth + 1);
  } else if (value.isList()) {
    for (const auto& v : value.toListRef()) Inspect(v, budget, depth + 1);
  } else if (value.isGenericDict()) {
    for (const auto& e : value.toGenericDict()) {
      if (!e.key().isString()) InvalidBatch("batch.dictionary_key");
      Inspect(e.key(), budget, depth + 1);
      Inspect(e.value(), budget, depth + 1);
    }
  } else if (!value.isNone() && !value.isInt() && !value.isDouble() &&
             !value.isBool() && !value.isString())
    InvalidBatch("batch.value_kind");
}

I TensorList(const std::vector<Tensor>& tensors) {
  c10::impl::GenericList list(c10::TensorType::get());
  for (const auto& tensor : tensors) list.push_back(tensor);
  return I(list);
}
std::vector<Tensor> Tensors(const I& v) {
  std::vector<Tensor> result;
  for (const auto& t : v.toListRef()) result.push_back(t.toTensor());
  return result;
}

data::BatchSchema Schema(ImageBatchLayout layout, const PixelSpec& pixels,
                         const std::optional<PixelSpec>& paired, bool masks,
                         Device device) {
  const bool ragged = layout == ImageBatchLayout::ragged;
  const auto tensor = [&](Dtype dtype, int rank, bool photo = false,
                          int channels = 0) {
    TensorSchema constraints;
    constraints.dtype = dtype;
    constraints.device = device;
    constraints.rank = rank;
    constraints.shape = std::vector<std::optional<std::int64_t>>(
        static_cast<std::size_t>(rank));
    if (photo) (*constraints.shape)[ragged ? 0 : 1] = channels;
    auto v = ValueSchema::tensor(constraints);
    return ragged ? List(v) : v;
  };
  const auto pixel =
      Tuple({Kind(ValueKind::integer), Kind(ValueKind::integer),
             Kind(ValueKind::integer), Kind(ValueKind::integer)});
  auto labels = ValueSchema::tensor(
      {.dtype = torch::kInt64,
       .device = device,
       .rank = 1,
       .shape = std::vector<std::optional<std::int64_t>>{std::nullopt}});
  auto root = ValueSchema::dictionary(
      {{"format", Kind(ValueKind::integer)},
       {"layout", Kind(ValueKind::integer)},
       {"pixels", pixel},
       {"paired_pixels", paired ? pixel : Kind(ValueKind::none)},
       {"images",
        tensor(pixels.dtype, ragged ? 3 : 4, true, pixels.channels())},
       {"paired",
        paired ? tensor(paired->dtype, ragged ? 3 : 4, true, paired->channels())
               : Kind(ValueKind::none)},
       {"masks",
        masks ? tensor(torch::kInt64, ragged ? 2 : 3) : Kind(ValueKind::none)},
       {"valid", tensor(torch::kBool, ragged ? 2 : 3)},
       {"labels", labels},
       {"label_offsets", List(Kind(ValueKind::integer))},
       {"items", List(ItemSchema())}},
      false);
  const auto semantics = [](const PixelSpec& p) {
    return std::to_string(static_cast<int>(p.color)) + ":" +
           std::to_string(static_cast<int>(p.alpha)) + ":" +
           std::to_string(static_cast<int>(p.range));
  };
  root.semantic_tag =
      "liuding.image-batch.v1:" + std::to_string(static_cast<int>(layout)) +
      ":" + semantics(pixels) +
      ":paired=" + (paired ? semantics(*paired) : "none") +
      ":mask=" + (masks ? "int64" : "none");
  return {"liuding.image-batch", 1, std::move(root)};
}
}  // namespace

c10::IValue image_batch_value(const ImageBatch& batch,
                              const ImageBatchLimits& limits) {
  try {
    detail::CheckImageBatch(batch, limits);
    Writer w{{limits}};
    Values items, offsets;
    for (const auto& item : batch.items) items.push_back(w.Item(item));
    for (const auto offset : batch.label_offsets)
      offsets.push_back(w.Int(offset));
    const bool ragged = batch.layout == ImageBatchLayout::ragged;
    c10::impl::GenericDict value(c10::StringType::get(), c10::AnyType::get());
    value.insert("format", w.Int(1));
    value.insert("layout", w.Int(static_cast<std::int64_t>(batch.layout)));
    value.insert("pixels", w.Pixels(batch.pixels));
    value.insert("paired_pixels", batch.paired_pixels
                                      ? w.Pixels(*batch.paired_pixels)
                                      : w.None());
    value.insert("images",
                 ragged ? TensorList(batch.image_list) : I(batch.images));
    value.insert("paired", batch.paired_pixels
                               ? (ragged ? TensorList(batch.paired_image_list)
                                         : I(*batch.paired_images))
                               : w.None());
    value.insert("masks",
                 batch.items[0].mask
                     ? (ragged ? TensorList(batch.mask_list) : I(*batch.masks))
                     : w.None());
    value.insert("valid",
                 ragged ? TensorList(batch.valid_list) : I(batch.valid));
    value.insert("labels", batch.labels);
    value.insert("label_offsets", w.List(std::move(offsets)));
    value.insert("items", w.List(std::move(items)));
    I result(value);
    Budget budget{limits};
    Inspect(result, budget);
    return result;
  } catch (const std::bad_alloc&) {
    BatchLimit("batch.encode_allocate");
  }
}

ImageBatch image_batch_from_value(const c10::IValue& value,
                                  const ImageBatchLimits& limits) {
  detail::CheckBatchLimits(limits);
  try {
    Budget budget{limits};
    Inspect(value, budget);
    if (!value.isGenericDict()) InvalidBatch("batch.value_dictionary");
    const auto dict = value.toGenericDict();
    const auto get = [&](const char* key) -> I {
      if (!dict.contains(key)) InvalidBatch("batch.value_field");
      return dict.at(key);
    };
    if (!get("format").isInt() || get("format").toInt() != 1 ||
        !get("layout").isInt())
      InvalidBatch("batch.value_version");
    const auto mode = get("layout").toInt();
    if (mode < 0 || mode > 2) InvalidBatch("batch.layout");
    const auto pixel_schema =
        Tuple({Kind(ValueKind::integer), Kind(ValueKind::integer),
               Kind(ValueKind::integer), Kind(ValueKind::integer)});
    ValueSchemaLimits schema_limits;
    schema_limits.max_runtime_nodes = limits.max_metadata_nodes;
    validate_value_schema(pixel_schema);
    validate_value(get("pixels"), pixel_schema, "image.pixels", schema_limits);
    ImageBatch b;
    b.layout = static_cast<ImageBatchLayout>(mode);
    b.pixels = Pixels(get("pixels"));
    if (!get("paired_pixels").isNone()) {
      validate_value(get("paired_pixels"), pixel_schema, "image.paired_pixels",
                     schema_limits);
      b.paired_pixels = Pixels(get("paired_pixels"));
    }
    const auto images = get("images");
    Device device(torch::kCPU);
    if (b.layout == ImageBatchLayout::ragged) {
      if (!images.isList() || images.toListRef().empty() ||
          !images.toListRef()[0].isTensor())
        InvalidBatch("batch.images");
      device = images.toListRef()[0].toTensor().device();
    } else {
      if (!images.isTensor()) InvalidBatch("batch.images");
      device = images.toTensor().device();
    }
    validate_batch_value(value,
                         Schema(b.layout, b.pixels, b.paired_pixels,
                                !get("masks").isNone(), device),
                         schema_limits);
    if (b.layout == ImageBatchLayout::ragged) {
      b.image_list = Tensors(images);
      b.valid_list = Tensors(get("valid"));
      if (b.paired_pixels) b.paired_image_list = Tensors(get("paired"));
      if (!get("masks").isNone()) b.mask_list = Tensors(get("masks"));
    } else {
      b.images = images.toTensor();
      b.valid = get("valid").toTensor();
      if (b.paired_pixels) b.paired_images = get("paired").toTensor();
      if (!get("masks").isNone()) b.masks = get("masks").toTensor();
    }
    b.labels = get("labels").toTensor();
    for (const auto& offset : get("label_offsets").toListRef())
      b.label_offsets.push_back(offset.toInt());
    for (const auto& item : get("items").toListRef())
      b.items.push_back(Item(item));
    detail::CheckImageBatch(b, limits);
    return b;
  } catch (const std::bad_alloc&) {
    BatchLimit("batch.decode_allocate");
  }
}

std::shared_ptr<void> image_batch_resource_lease(
    const c10::IValue& value, std::shared_ptr<ImageResourceBudget> budget,
    const ImageBatchLimits& limits, std::stop_token stop) {
  if (!budget) return {};
  auto batch = image_batch_from_value(value, limits);
  auto leases = detail::ReserveBatchResources(batch, budget, limits, stop);
  return std::static_pointer_cast<void>(std::move(leases));
}

data::BatchResourceLeaseFactory make_image_batch_resource_lease_factory(
    std::shared_ptr<ImageResourceBudget> budget, ImageBatchLimits limits) {
  return [budget = std::move(budget), limits](
             const c10::IValue& value, const data::BatchMemoryMetadata&,
             std::stop_token stop) {
    return image_batch_resource_lease(value, budget, limits, stop);
  };
}

data::BatchAdmissionFactory make_image_batch_admission_factory(
    std::shared_ptr<ImageResourceBudget> budget) {
  return [budget = std::move(budget)](std::stop_token stop) {
    if (!budget) return std::shared_ptr<void>{};
    auto lease = std::make_shared<ImageResourceLease>(budget->reserve_wait(
        ImageResourceKind::pipeline_inflight_batches, 1, stop));
    return std::static_pointer_cast<void>(std::move(lease));
  };
}

data::BatchSchema image_batch_schema(const ImageBatch& batch, Device device) {
  if (batch.layout != ImageBatchLayout::stack &&
      batch.layout != ImageBatchLayout::pad &&
      batch.layout != ImageBatchLayout::ragged)
    InvalidBatch("batch.layout");
  return Schema(batch.layout, batch.pixels, batch.paired_pixels,
                batch.masks.has_value() || !batch.mask_list.empty(), device);
}

ImageBatch prepare_image_batch(const ImageBatch& batch,
                               const data::BatchMemoryOptions& options,
                               const ImageBatchLimits& limits,
                               std::shared_ptr<ImageResourceBudget> budget) {
  const auto prepared =
      data::prepare_batch_memory(image_batch_value(batch, limits), options);
  auto result = image_batch_from_value(prepared.value, limits);
  result.memory = prepared.metadata;
  if (budget)
    result.resource_lease_sets.push_back(
        detail::ReserveBatchResources(result, budget, limits));
  return result;
}

ImageBatch transfer_image_batch(ImageBatch prepared,
                                const data::BatchMemoryOptions& options,
                                const ImageBatchLimits& limits,
                                std::shared_ptr<ImageResourceBudget> budget) {
  auto staged = data::transfer_batch_memory(
      {image_batch_value(prepared, limits), prepared.memory}, options);
  auto result = image_batch_from_value(staged.value, limits);
  result.memory = staged.metadata;
  if (budget)
    result.resource_lease_sets.push_back(
        detail::ReserveBatchResources(result, budget, limits));
  return result;
}
}  // namespace jyppx::liuding::data::image
