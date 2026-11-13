// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../workflow/artifact_io.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/vision/tasks/training/weight_conversion.hpp>

#include <bit>
#include <caffe2/serialize/inline_container.h>
#include <cstring>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
namespace jyppx::liuding::vision::tasks {
namespace {
using I = std::int64_t;
using J = nlohmann::json;
struct Value;
using V = std::shared_ptr<Value>;
struct Value {
  enum Kind {
    none,
    integer,
    text,
    global,
    sequence,
    dictionary,
    object,
    tensor,
    storage,
    mark
  } kind = none;
  I number = 0;
  std::string string;
  std::vector<V> items;
  std::map<std::string, V> fields;
  Tensor data;
};
class Pickle {
  const WeightConversionOptions& options;
  std::stop_token stop;
  caffe2::serialize::PyTorchStreamReader& archive;
  std::span<const std::byte> bytes;
  std::size_t position = 0, nodes = 0;
  std::vector<V> stack;
  std::map<std::uint64_t, V> memo;
  std::map<std::string, Tensor> storages;
  [[noreturn]] void fail(const std::string& text) const {
    throw SerializationError("Native weight pickle: " + text + " at byte " +
                             std::to_string(position));
  }
  V make(Value::Kind kind = Value::none) {
    if (++nodes > options.max_nodes)
      throw ResourceExhausted("Weight pickle node budget exceeded");
    auto v = std::make_shared<Value>();
    v->kind = kind;
    return v;
  }
  unsigned byte() {
    if (position == bytes.size()) fail("truncated opcode/payload");
    return std::to_integer<unsigned char>(bytes[position++]);
  }
  std::uint64_t number(unsigned count) {
    std::uint64_t n = 0;
    for (unsigned i = 0; i < count; ++i) n |= std::uint64_t(byte()) << (8 * i);
    return n;
  }
  std::string text(std::uint64_t n) {
    if (n > bytes.size() - position || n > (16U << 20U))
      fail("invalid string bound");
    std::string s(reinterpret_cast<const char*>(bytes.data() + position),
                  static_cast<std::size_t>(n));
    position += n;
    return s;
  }
  std::string line() {
    std::string s;
    while (true) {
      auto b = byte();
      if (b == '\n') break;
      if (s.size() >= 4096) fail("global name too long");
      s += static_cast<char>(b);
    }
    return s;
  }
  V pop() {
    if (stack.empty()) fail("empty stack");
    auto v = stack.back();
    stack.pop_back();
    return v;
  }
  std::vector<V> marked() {
    auto at = stack.size();
    while (at && stack[at - 1]->kind != Value::mark) --at;
    if (!at) fail("missing MARK");
    std::vector<V> values(stack.begin() + at, stack.end());
    stack.resize(at - 1);
    return values;
  }
  I integer(const V& v) {
    if (v->kind != Value::integer) fail("integer required");
    return v->number;
  }
  std::string string(const V& v) {
    if (v->kind != Value::text && v->kind != Value::global)
      fail("string/global required");
    return v->string;
  }
  std::string key(const V& v) {
    if (v->kind == Value::integer)
      return std::string(1, '\x01') + "integer:" + std::to_string(v->number);
    return string(v);
  }
  std::vector<I> dimensions(const V& v) {
    if (v->kind != Value::sequence || v->items.size() > 32)
      fail("bounded dimension tuple required");
    std::vector<I> result;
    for (auto p : v->items) {
      auto n = integer(p);
      if (n < 0) fail("negative size/stride");
      result.push_back(n);
    }
    return result;
  }
  void pairs(const V& target, const std::vector<V>& values) {
    if (values.size() % 2) fail("odd dictionary entries");
    for (std::size_t i = 0; i < values.size(); i += 2)
      target->fields[key(values[i])] = values[i + 1];
  }
  V persistent(const V& tuple) {
    if (tuple->kind != Value::sequence || tuple->items.size() != 5 ||
        string(tuple->items[0]) != "storage")
      fail("unsupported persistent object");
    const auto type = string(tuple->items[1]), key = string(tuple->items[2]);
    const auto count = integer(tuple->items[4]);
    if (key.empty() ||
        key.find_first_not_of("0123456789") != std::string::npos || count < 0)
      fail("invalid storage identity");
    static const std::map<std::string, torch::ScalarType> types{
        {"torch.FloatStorage", torch::kFloat32},
        {"torch.DoubleStorage", torch::kFloat64},
        {"torch.HalfStorage", torch::kFloat16},
        {"torch.BFloat16Storage", torch::kBFloat16},
        {"torch.LongStorage", torch::kInt64},
        {"torch.IntStorage", torch::kInt32},
        {"torch.ShortStorage", torch::kInt16},
        {"torch.CharStorage", torch::kInt8},
        {"torch.ByteStorage", torch::kUInt8},
        {"torch.BoolStorage", torch::kBool}};
    const auto found = types.find(type);
    if (found == types.end()) fail("unsupported storage dtype: " + type);
    const auto width = c10::elementSize(found->second);
    if (static_cast<std::uint64_t>(count) > options.max_bytes / width)
      throw ResourceExhausted("Weight storage byte bound exceeded");
    const auto record = "data/" + key;
    const auto size = static_cast<std::uint64_t>(count) * width;
    if (!archive.hasRecord(record) || archive.getRecordSize(record) != size)
      fail("storage payload size differs from metadata");
    auto [it, inserted] = storages.try_emplace(key);
    if (inserted) {
      it->second =
          torch::empty({count}, torch::TensorOptions().dtype(found->second));
      if (size) archive.getRecord(record, it->second.data_ptr(), size);
    } else if (it->second.scalar_type() != found->second ||
               it->second.numel() != count)
      fail("storage alias metadata mismatch");
    auto v = make(Value::storage);
    v->data = it->second;
    return v;
  }
  V reduce(const V& callable, const V& args) {
    if (callable->kind != Value::global || args->kind != Value::sequence)
      fail("unsupported REDUCE form");
    const auto name = callable->string;
    if (name == "collections.OrderedDict") {
      auto v = make(Value::dictionary);
      if (!args->items.empty()) {
        if (args->items.size() != 1 || args->items[0]->kind != Value::sequence)
          fail("OrderedDict arguments");
        for (auto pair : args->items[0]->items) {
          if (pair->kind != Value::sequence || pair->items.size() != 2)
            fail("OrderedDict pair");
          pairs(v, pair->items);
        }
      }
      return v;
    }
    if (name == "__builtin__.set" || name == "builtins.set" ||
        name == "builtins.frozenset") {
      auto v = make(Value::sequence);
      if (args->items.size() == 1)
        v->items = args->items[0]->items;
      else if (!args->items.empty())
        fail("set arguments");
      return v;
    }
    if (name == "torch.Size") {
      if (args->items.size() != 1 || args->items[0]->kind != Value::sequence ||
          args->items[0]->items.size() > 64)
        fail("invalid inert Size metadata");
      for (const auto& item : args->items[0]->items)
        static_cast<void>(integer(item));
      auto v = make(Value::sequence);
      v->items = args->items[0]->items;
      return v;
    }
    if (name == "__builtin__.getattr" || name == "builtins.getattr") {
      // Old Ultralytics checkpoints store Detect.forward as head metadata.
      // Keep an opaque marker; never resolve a class, attribute or callable.
      if (args->items.size() != 2 || args->items[0]->kind != Value::global ||
          !args->items[0]->string.starts_with("ultralytics.nn.modules.head.") ||
          args->items[1]->kind != Value::text ||
          args->items[1]->string != "forward")
        fail("unsupported inert method metadata");
      auto v = make(Value::object);
      v->string = "inert:" + args->items[0]->string + ".forward";
      return v;
    }
    if (name == "torch._utils._rebuild_parameter" ||
        name == "torch._utils._rebuild_parameter_with_state") {
      if (args->items.empty() || args->items[0]->kind != Value::tensor)
        fail("Parameter argument");
      return args->items[0];
    }
    if (name == "torch._utils._rebuild_tensor_v2" ||
        name == "torch._utils._rebuild_tensor") {
      if (args->items.size() < 4 || args->items[0]->kind != Value::storage)
        fail("tensor rebuild arguments");
      const auto& storage = args->items[0]->data;
      auto offset = integer(args->items[1]);
      auto shape = dimensions(args->items[2]),
           stride = dimensions(args->items[3]);
      if (shape.size() != stride.size() || offset < 0 ||
          offset > storage.numel())
        fail("tensor view layout");
      std::uint64_t last = offset, elements = 1;
      for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] &&
            elements > options.max_bytes / static_cast<std::uint64_t>(shape[i]))
          throw ResourceExhausted("Tensor element bound exceeded");
        elements *= shape[i];
        if (shape[i] > 1) {
          const auto extent = static_cast<std::uint64_t>(shape[i] - 1);
          if (stride[i] && extent > (options.max_bytes - last) /
                                        static_cast<std::uint64_t>(stride[i]))
            fail("stride address overflow");
          last += extent * stride[i];
        }
      }
      if (elements && last >= static_cast<std::uint64_t>(storage.numel()))
        fail("tensor view exceeds storage");
      auto v = make(Value::tensor);
      v->data = storage.as_strided(shape, stride, offset);
      return v;
    }
    fail("unsupported function reduction: " + name);
  }

 public:
  Pickle(const WeightConversionOptions& o, std::stop_token s,
         caffe2::serialize::PyTorchStreamReader& reader,
         std::span<const std::byte> input)
      : options(o), stop(s), archive(reader), bytes(input) {}
  V parse() {
    while (position < bytes.size()) {
      if ((position & 4095) == 0 && stop.stop_requested())
        throw Cancelled("Weight conversion cancelled");
      const auto code = byte();
      V value;
      switch (code) {
        case 0x80: {
          auto version = byte();
          if (version < 2 || version > 5) fail("protocol must be 2..5");
          break;
        }
        case 0x95:
          if (number(8) > bytes.size() - position)
            fail("FRAME exceeds document");
          break;
        case '.': {
          auto result = pop();
          if (!stack.empty() || position != bytes.size())
            fail("trailing stack/data");
          return result;
        }
        case '(':
          stack.push_back(make(Value::mark));
          break;
        case 'N':
          stack.push_back(make());
          break;
        case 0x88:
        case 0x89:
          value = make(Value::integer);
          value->number = code == 0x88;
          stack.push_back(value);
          break;
        case 'K':
        case 'M':
        case 'J': {
          auto n = number(code == 'K' ? 1 : code == 'M' ? 2 : 4);
          value = make(Value::integer);
          value->number =
              code == 'J' ? static_cast<std::int32_t>(n) : static_cast<I>(n);
          stack.push_back(value);
          break;
        }
        case 0x8a:
        case 0x8b: {
          const auto n = number(code == 0x8a ? 1 : 4);
          if (n > 8) fail("LONG exceeds int64");
          auto raw = number(static_cast<unsigned>(n));
          if (n && n < 8 && (raw & (1ULL << (8 * n - 1))))
            raw |= (~0ULL) << (8 * n);
          value = make(Value::integer);
          value->number = static_cast<I>(raw);
          stack.push_back(value);
          break;
        }
        case 'G':
          number(8);
          stack.push_back(make());
          break;  // Non-tensor scalar training metadata.
        case 'X':
        case 0x8c:
        case 0x8d:
        case 'U':
        case 'T':
        case 'B':
        case 'C': {
          auto n = number(code == 0x8c || code == 'U' || code == 'C' ? 1
                          : code == 0x8d                             ? 8
                                                                     : 4);
          value = make(Value::text);
          value->string = text(n);
          stack.push_back(value);
          break;
        }
        case 'c': {
          auto module = line(), name = line();
          value = make(Value::global);
          value->string = module + "." + name;
          stack.push_back(value);
          break;
        }
        case 0x93: {
          auto name = string(pop()), module = string(pop());
          value = make(Value::global);
          value->string = module + "." + name;
          stack.push_back(value);
          break;
        }
        case ')':
        case ']':
        case '}':
        case 0x8f:
          stack.push_back(
              make(code == '}' ? Value::dictionary : Value::sequence));
          break;
        case 't':
        case 'l':
        case 'd': {
          auto items = marked();
          value = make(code == 'd' ? Value::dictionary : Value::sequence);
          if (code == 'd')
            pairs(value, items);
          else
            value->items = std::move(items);
          stack.push_back(value);
          break;
        }
        case 0x85:
        case 0x86:
        case 0x87: {
          const auto count = code - 0x84;
          if (stack.size() < count) fail("tuple stack underflow");
          value = make(Value::sequence);
          value->items.assign(stack.end() - count, stack.end());
          stack.resize(stack.size() - count);
          stack.push_back(value);
          break;
        }
        case 'a': {
          auto item = pop();
          if (stack.empty()) fail("APPEND target");
          stack.back()->items.push_back(item);
          break;
        }
        case 'e':
        case 0x90: {
          auto items = marked();
          if (stack.empty()) fail("APPENDS target");
          auto& target = stack.back()->items;
          target.insert(target.end(), items.begin(), items.end());
          break;
        }
        case 's': {
          auto v = pop(), k = pop();
          if (stack.empty()) fail("SETITEM target");
          stack.back()->fields[key(k)] = v;
          break;
        }
        case 0x91:
          value = make(Value::sequence);
          value->items = marked();
          stack.push_back(value);
          break;
        case 'u': {
          auto items = marked();
          if (stack.empty()) fail("SETITEMS target");
          pairs(stack.back(), items);
          break;
        }
        case 'q':
        case 'r':
        case 0x94: {
          auto id = code == 0x94 ? memo.size() : number(code == 'q' ? 1 : 4);
          if (stack.empty() || id > options.max_nodes || memo.contains(id))
            fail("invalid/repeated memo identity");
          memo.emplace(id, stack.back());
          break;
        }
        case 'h':
        case 'j': {
          auto id = number(code == 'h' ? 1 : 4);
          if (!memo.contains(id)) fail("missing memo reference");
          stack.push_back(memo.at(id));
          break;
        }
        case 'Q':
          value = persistent(pop());
          stack.push_back(value);
          break;
        case 'R': {
          auto args = pop(), callable = pop();
          stack.push_back(reduce(callable, args));
          break;
        }
        case 0x81: {
          auto args = pop(), cls = pop();
          if (cls->kind != Value::global || args->kind != Value::sequence ||
              !args->items.empty())
            fail("only inert empty module construction is supported");
          value = make(Value::object);
          value->string = cls->string;
          stack.push_back(value);
          break;
        }
        case 'b': {
          auto state = pop();
          if (stack.empty() || (stack.back()->kind != Value::object &&
                                stack.back()->kind != Value::dictionary))
            fail("BUILD requires inert object");
          auto target = stack.back();
          if (state->kind == Value::dictionary)
            target->fields.insert(state->fields.begin(), state->fields.end());
          else if (state->kind == Value::sequence) {
            for (auto part : state->items)
              if (part->kind == Value::dictionary)
                target->fields.insert(part->fields.begin(), part->fields.end());
          } else
            fail("unsupported object state");
          break;
        }
        case '0':
          pop();
          break;
        case '2':
          if (stack.empty()) fail("DUP empty stack");
          stack.push_back(stack.back());
          break;
        default:
          fail("unsupported opcode " + std::to_string(code));
      }
      if (stack.size() > options.max_nodes || memo.size() > options.max_nodes)
        throw ResourceExhausted("Pickle stack/memo budget exceeded");
    }
    fail("missing STOP");
  }
};
void Collect(const V& node, const std::string& prefix,
             std::map<std::string, Tensor>& tensors,
             std::set<const Value*>& active, std::size_t maximum,
             std::size_t depth = 0) {
  if (depth > 128 || !active.insert(node.get()).second)
    throw SerializationError("Cyclic/deep module state");
  auto add = [&](const std::string& key, const V& value) {
    if (value->kind != Value::tensor) return;
    if (tensors.size() >= maximum)
      throw ResourceExhausted("Converted tensor count exceeded");
    if (!tensors.emplace(prefix + key, value->data).second)
      throw SerializationError("Duplicate tensor key");
  };
  if (node->kind == Value::object) {
    for (auto name : {"_parameters", "_buffers"})
      if (auto it = node->fields.find(name); it != node->fields.end())
        for (auto [key, value] : it->second->fields) add(key, value);
    if (auto it = node->fields.find("_modules"); it != node->fields.end())
      for (auto [key, value] : it->second->fields)
        if (value->kind != Value::none)
          Collect(value, prefix + key + ".", tensors, active, maximum,
                  depth + 1);
  } else if (node->kind == Value::dictionary)
    for (auto [key, value] : node->fields) add(key, value);
  else
    throw CompatibilityError(
        "Selected checkpoint root is neither state_dict nor inert nn.Module");
  active.erase(node.get());
}
interop::SafeTensorDtype Dtype(torch::ScalarType t) {
  switch (t) {
    case torch::kFloat32:
      return interop::SafeTensorDtype::f32;
    case torch::kFloat64:
      return interop::SafeTensorDtype::f64;
    case torch::kFloat16:
      return interop::SafeTensorDtype::f16;
    case torch::kBFloat16:
      return interop::SafeTensorDtype::bf16;
    case torch::kInt64:
      return interop::SafeTensorDtype::i64;
    case torch::kInt32:
      return interop::SafeTensorDtype::i32;
    case torch::kInt16:
      return interop::SafeTensorDtype::i16;
    case torch::kInt8:
      return interop::SafeTensorDtype::i8;
    case torch::kUInt8:
      return interop::SafeTensorDtype::u8;
    case torch::kBool:
      return interop::SafeTensorDtype::boolean;
    default:
      throw CompatibilityError("Unsupported conversion tensor dtype");
  }
}
}  // namespace
std::string convert_vision_weights(const WeightConversionOptions& o,
                                   std::stop_token stop) {
  if (o.source.empty() || o.output.empty() || o.sha256.size() != 64 ||
      !o.max_bytes || o.max_bytes > (4ULL << 30U) || !o.max_nodes ||
      o.max_nodes > 10000000 || !o.max_tensors || o.max_tensors > 65536)
    throw InvalidArgument("Invalid bounded weight conversion options");
  if (std::filesystem::exists(o.output))
    throw InvalidArgument("Converted weight output must be new");
  auto encoded = detail::read_artifact(o.source, o.max_bytes);
  if (sha256_hex(encoded) != o.sha256)
    throw CompatibilityError("Source checkpoint SHA-256 mismatch");
  std::string content(reinterpret_cast<const char*>(encoded.data()),
                      encoded.size());
  std::istringstream stream(content, std::ios::binary);
  caffe2::serialize::PyTorchStreamReader archive(&stream);
  std::uint64_t total = 0;
  auto records = archive.getAllRecords();
  if (records.size() > o.max_tensors + 128)
    throw ResourceExhausted("ZIP record count exceeded");
  for (const auto& name : records) {
    auto size = archive.getRecordSize(name);
    if (size > o.max_bytes - total)
      throw ResourceExhausted("ZIP expanded byte bound exceeded");
    total += size;
  }
  if (archive.hasRecord("byteorder")) {
    auto [data, size] = archive.getRecord("byteorder");
    if (std::string(static_cast<const char*>(data.get()), size) != "little")
      throw CompatibilityError(
          "Only little-endian PyTorch storage is supported");
  }
  if (!archive.hasRecord("data.pkl") ||
      archive.getRecordSize("data.pkl") > (64U << 20U))
    throw CompatibilityError(
        "Expected a bounded PyTorch data.pkl ZIP checkpoint");
  auto [data, size] = archive.getRecord("data.pkl");
  auto root = Pickle(o, stop, archive,
                     {static_cast<const std::byte*>(data.get()), size})
                  .parse();
  if (o.root != "root") {
    auto found = root->fields.find(o.root);
    if (found == root->fields.end() || found->second->kind == Value::none)
      throw CompatibilityError("Selected checkpoint root is absent: " + o.root);
    root = found->second;
  }
  std::map<std::string, Tensor> tensors;
  std::set<const Value*> active;
  Collect(root, "", tensors, active, o.max_tensors);
  if (tensors.empty())
    throw CompatibilityError("No weights in selected checkpoint root");
  std::vector<Tensor> copies;
  std::vector<interop::SafeTensorSource> sources;
  J mapping = J::object();
  std::set<std::string> targets, used;
  total = 0;
  copies.reserve(tensors.size());
  for (const auto& [key, tensor] : tensors) {
    if (stop.stop_requested()) throw Cancelled("Weight conversion cancelled");
    auto found = o.key_mapping.find(key);
    auto target = found == o.key_mapping.end() ? key : found->second;
    if (found != o.key_mapping.end()) used.insert(key);
    if (target.empty() || !targets.insert(target).second)
      throw InvalidArgument("Converted tensor names collide");
    auto value = tensor.detach();
    if (o.floating_to_float32 && value.is_floating_point())
      value = value.to(torch::kFloat32);
    copies.push_back(value.contiguous());
    const auto& cpu = copies.back();
    if (cpu.nbytes() > o.max_bytes - total)
      throw ResourceExhausted("Converted payload byte bound exceeded");
    total += cpu.nbytes();
    sources.push_back({target,
                       Dtype(cpu.scalar_type()),
                       {cpu.sizes().begin(), cpu.sizes().end()},
                       {static_cast<const std::byte*>(cpu.const_data_ptr()),
                        static_cast<std::size_t>(cpu.nbytes())}});
    mapping[key] = target;
  }
  if (used.size() != o.key_mapping.size())
    throw InvalidArgument(
        "Conversion mapping contains an absent source tensor");
  auto metadata = o.metadata;
  metadata["source_sha256"] = o.sha256;
  metadata["source_root"] = o.root;
  metadata["converter"] = "native-inert-pytorch-zip/v1";
  interop::SafeTensorsLimits limits;
  limits.max_file_bytes = o.max_bytes;
  limits.max_payload_bytes = o.max_bytes;
  auto artifact = interop::serialize_safetensors(sources, metadata, limits);
  detail::publish_artifact(o.output, artifact);
  auto report = J{
      {"schema", "vision.weight_conversion/v1"},
      {"source_sha256", o.sha256},
      {"output_sha256", sha256_hex(artifact)},
      {"root", o.root},
      {"keys", mapping},
      {"metadata", metadata},
      {"model_compatibility_checked",
       false}}.dump(2);
  detail::publish_text(o.output.string() + ".json", report);
  return report;
}
}  // namespace jyppx::liuding::vision::tasks
