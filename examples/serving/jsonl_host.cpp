// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
#include "linear_bundle.hpp"

#include <charconv>
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
namespace {
using Json = nlohmann::json;
using namespace jyppx::liuding;
using namespace jyppx::liuding::serving;
namespace example = jyppx::liuding::examples::serving;
constexpr std::size_t kMaxLine = 4096;
Json Parse(std::string_view text) {
  std::vector<std::set<std::string>> objects;
  auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
    if (depth > 8)
      throw SerializationError("JSON nesting exceeds example limit");
    if (event == Json::parse_event_t::object_start) objects.emplace_back();
    if (event == Json::parse_event_t::key &&
        (objects.empty() ||
         !objects.back().insert(value.get<std::string>()).second))
      throw SerializationError("Duplicate JSON member");
    if (event == Json::parse_event_t::object_end) objects.pop_back();
    return true;
  };
  return Json::parse(text.begin(), text.end(), callback, true, false);
}
void Members(const Json& object,
             std::initializer_list<std::string_view> allowed) {
  if (!object.is_object()) throw SerializationError("Expected JSON object");
  for (const auto& item : object.items())
    if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
      throw SerializationError("Unknown JSON member");
}
ProtocolAdapter Protocol(Device device) {
  return {.protocol_id = "jyppx.example.json-linear",
          .schema_version = 1,
          .decode =
              [device](std::span<const std::byte> bytes) {
                const auto object = Parse(std::string_view(
                    reinterpret_cast<const char*>(bytes.data()), bytes.size()));
                Members(object, {"input"});
                const auto& input = object.at("input");
                if (!input.is_array() || input.size() != 2 ||
                    !input[0].is_number() || !input[1].is_number())
                  throw SerializationError("Expected two numeric input values");
                const double a = input[0].get<double>(),
                             b = input[1].get<double>();
                if (!std::isfinite(a) || !std::isfinite(b) ||
                    std::abs(a) > 1e6 || std::abs(b) > 1e6)
                  throw InvalidArgument("Example input range exceeded");
                return nn::Invocation{
                    .positional = {torch::tensor(
                        {{static_cast<float>(a), static_cast<float>(b)}},
                        torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(device))},
                    .keyword = {}};
              },
          .encode =
              [](const nn::InvocationResult& result) {
                const auto output = result.value.toTensor().cpu().contiguous();
                const auto* values = output.data_ptr<float>();
                const auto text = Json{{"schema_version", 1},
                                       {"output", {values[0], values[1]}}}
                                      .dump();
                const auto bytes = example::Bytes(text);
                return std::vector<std::byte>(bytes.begin(), bytes.end());
              },
          .authorize =
              [](const RequestContext& context) {
                return context.project_id == "local-example" &&
                       context.principal_id == "local-operator";
              }};
}
enum class ReadResult { end, line, oversized };
ReadResult ReadLine(std::string& line) {
  line.clear();
  bool oversized = false;
  char ch = 0;
  while (std::cin.get(ch)) {
    if (ch == '\n') return oversized ? ReadResult::oversized : ReadResult::line;
    if (line.size() < kMaxLine)
      line.push_back(ch);
    else
      oversized = true;
  }
  if (line.empty() && !oversized) return ReadResult::end;
  return oversized ? ReadResult::oversized : ReadResult::line;
}
void Reply(ServiceStatus status, std::string_view id = {},
           const Json& body = nullptr) {
  std::cout << Json{{"schema_version", 1},
                    {"request_id", id},
                    {"status", service_status_name(status)},
                    {"body", body}}
                   .dump()
            << '\n';
  std::cout.flush();
}
}  // namespace
int main(int argc, char** argv) {
  try {
    bool demo = false, health_only = false, manifest_only = false;
    std::filesystem::path directory;
    Device device(c10::kCPU);
    for (int i = 1; i < argc; ++i) {
      const std::string_view argument = argv[i];
      if (argument == "--demo")
        demo = true;
      else if (argument == "--health")
        health_only = true;
      else if (argument == "--manifest")
        manifest_only = true;
      else if (argument == "--bundle" && i + 1 < argc)
        directory = std::filesystem::path(argv[++i]);
      else if (argument == "--cuda" && i + 1 < argc) {
        const std::string_view text = argv[++i];
        int index = -1;
        const auto result =
            std::from_chars(text.data(), text.data() + text.size(), index);
        if (result.ec != std::errc{} ||
            result.ptr != text.data() + text.size() || index < 0 || index > 127)
          throw InvalidArgument("Expected explicit CUDA device index");
        device = Device(c10::kCUDA, static_cast<c10::DeviceIndex>(index));
      } else {
        std::cerr << "Usage: liuding-forge-example-serving-jsonl (--demo | "
                     "--bundle PATH) "
                     "[--cuda N] [--health | --manifest]\n";
        return 2;
      }
    }
    if (manifest_only) {
      std::cout << serving_preview_manifest_json() << '\n';
      return 0;
    }
    if (demo == !directory.empty())
      throw InvalidArgument("Select exactly one local model source");
    std::unique_ptr<example::BundleFixture> generated;
    if (demo) {
      generated = std::make_unique<example::BundleFixture>(device);
      generated->Publish();
      directory = generated->directory;
    }
    NativeStateBackend backend(example::Descriptor(device), example::Factory());
    const std::array<WarmupCase, 1> warmup{example::KnownVector(device)};
    auto predictor = backend.load(directory, {}, warmup);
    auto batcher = std::make_shared<DynamicBatcher>(predictor);
    InferenceServiceAdapter service(
        predictor, Protocol(device),
        {.max_request_bytes = kMaxLine, .max_response_bytes = kMaxLine},
        batcher);
    if (health_only) {
      const auto health = service.health();
      std::cout << Json{{"live", health.live},
                        {"ready", health.ready},
                        {"streaming", health.streaming},
                        {"model_version", health.model.model_version},
                        {"device", health.model.runtime.device.str()}}
                       .dump()
                << '\n';
      return service.close() ? 0 : 1;
    }
    std::string line;
    for (;;) {
      const auto read = ReadLine(line);
      if (read == ReadResult::end) break;
      if (read == ReadResult::oversized) {
        Reply(ServiceStatus::resource_exhausted);
        continue;
      }
      std::string id;
      try {
        const auto request = Parse(line);
        Members(request, {"request_id", "input", "deadline_ms"});
        if (!request.contains("request_id") ||
            !request["request_id"].is_string())
          throw SerializationError("Expected request_id string");
        id = request["request_id"].get<std::string>();
        if (id.empty() || id.size() > 256 ||
            std::any_of(id.begin(), id.end(),
                        [](unsigned char c) { return c < 0x20U || c > 0x7eU; }))
          throw InvalidArgument("Expected printable bounded request ID");
        auto context = example::Request(device).context;
        context.request_id = id;
        context.project_id = "local-example";
        TransportContext transport{
            .correlation = context, .authenticated = true, .deadline = {}};
        if (request.contains("deadline_ms")) {
          if (!request["deadline_ms"].is_number_integer())
            throw SerializationError("Expected integer deadline_ms");
          const auto timeout = request["deadline_ms"].get<std::int64_t>();
          if (timeout < 0 || timeout > 60000)
            throw InvalidArgument("Deadline range is 0..60000 ms");
          transport.deadline = std::chrono::steady_clock::now() +
                               std::chrono::milliseconds(timeout);
        }
        const auto body = Json{{"input", request.at("input")}}.dump();
        const auto response = service.handle(example::Bytes(body), transport);
        Reply(response.status, id,
              response.status == ServiceStatus::ok
                  ? Parse(std::string_view(
                        reinterpret_cast<const char*>(response.body.data()),
                        response.body.size()))
                  : Json(nullptr));
      } catch (const Error& error) {
        Reply(error.code() == ErrorCode::resource_exhausted
                  ? ServiceStatus::resource_exhausted
                  : ServiceStatus::invalid_argument,
              id);
      } catch (const Json::exception&) {
        Reply(ServiceStatus::invalid_argument, id);
      } catch (...) {
        Reply(ServiceStatus::internal, id);
      }
    }
    return service.close() ? 0 : 1;
  } catch (const Error& error) {
    std::cerr << "Serving host initialization: "
              << error_code_name(error.code()) << '\n';
    return 1;
  } catch (...) {
    std::cerr << "Serving host initialization failed\n";
    return 1;
  }
}
