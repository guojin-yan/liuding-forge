// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"

#include <jyppx/liuding/foundation/build_config.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace jyppx::liuding::cli {

[[noreturn]] void usage(std::string_view operation, std::string_view hint) {
  throw InvalidArgument(
      "Invalid command line",
      ErrorDetails{.component = "tools.cli",
                   .operation = std::string(operation),
                   .expected = "documented noninteractive arguments",
                   .actual = "invalid or incomplete command",
                   .hint = std::string(hint)});
}

Arguments::Arguments(std::span<const std::string_view> arguments) {
  if (arguments.size() > 1024U) usage("argv", "Use at most 1024 arguments.");
  std::size_t bytes = 0U;
  for (std::size_t i = 0U; i < arguments.size(); ++i) {
    const auto argument = arguments[i];
    if (argument.size() > 65536U - bytes)
      usage("argv", "Keep arguments below 64 KiB in total.");
    bytes += argument.size();
    if (!argument.starts_with("--")) {
      if (argument.starts_with('-'))
        usage("argv", "Use the documented long options.");
      words_.emplace_back(argument);
      continue;
    }
    const auto separator = argument.find('=');
    const auto name = argument.substr(
        2U, separator == std::string_view::npos ? separator : separator - 2U);
    if (name.empty()) usage("argv", "Empty option names are invalid.");
    if (name == "help" || name == "version") {
      if (separator != std::string_view::npos)
        usage("argv", "Help and version take no value.");
      options_[std::string(name)].emplace_back("true");
    } else {
      std::string_view value;
      if (separator != std::string_view::npos)
        value = argument.substr(separator + 1U);
      else {
        if (i + 1U == arguments.size() || arguments[i + 1U].starts_with("--")) {
          usage("argv", "Every option except help/version requires a value.");
        }
        value = arguments[++i];
        if (value.size() > 65536U - bytes)
          usage("argv", "Keep arguments below 64 KiB in total.");
        bytes += value.size();
      }
      if (value.empty()) usage("argv", "Option values cannot be empty.");
      options_[std::string(name)].emplace_back(value);
    }
  }
  if (const auto format = optional("format")) {
    if (*format != "json" && *format != "text")
      usage("format", "Choose text or json.");
    json_ = *format == "json";
  }
}

void Arguments::allow(std::initializer_list<std::string_view> names) const {
  for (const auto& [name, values] : options_) {
    static_cast<void>(values);
    if (name != "format" &&
        std::find(names.begin(), names.end(), name) == names.end()) {
      usage("option", "Remove an option unsupported by this command.");
    }
  }
}

const std::vector<std::string>& Arguments::words() const noexcept {
  return words_;
}

std::optional<std::string> Arguments::optional(std::string_view name) const {
  const auto found = options_.find(std::string(name));
  if (found == options_.end()) return std::nullopt;
  if (found->second.size() != 1U)
    usage("option", "Do not repeat singleton options.");
  return found->second.front();
}

std::string Arguments::require(std::string_view name) const {
  const auto value = optional(name);
  if (!value) usage("option", "Supply all required options shown by --help.");
  return *value;
}

const std::vector<std::string>& Arguments::repeated(
    std::string_view name) const {
  static const std::vector<std::string> empty;
  const auto found = options_.find(std::string(name));
  return found == options_.end() ? empty : found->second;
}

std::uint64_t Arguments::number(std::string_view name,
                                std::uint64_t default_value,
                                std::uint64_t maximum) const {
  const auto value = optional(name);
  if (!value) return default_value;
  std::uint64_t parsed = 0U;
  const auto result =
      std::from_chars(value->data(), value->data() + value->size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value->data() + value->size() ||
      parsed > maximum) {
    usage("number", "Supply a bounded nonnegative decimal integer.");
  }
  return parsed;
}

bool Arguments::json_format() const noexcept { return json_; }

std::string read_text(const std::filesystem::path& path,
                      std::size_t maximum_bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw IoError("Could not open command input",
                  "Check the input path and permissions.");
  const auto length = input.tellg();
  if (length < 0) throw IoError("Could not size command input");
  if (static_cast<std::uint64_t>(length) > maximum_bytes) {
    throw ResourceExhausted("Command input exceeds the byte limit");
  }
  std::string value(static_cast<std::size_t>(length), '\0');
  input.seekg(0);
  input.read(value.data(), static_cast<std::streamsize>(value.size()));
  if (!input || input.peek() != std::char_traits<char>::eof()) {
    throw IoError("Command input changed or could not be read completely");
  }
  return value;
}

std::string read_text(const std::string& utf8_path, std::size_t maximum_bytes) {
  return read_text(std::filesystem::u8path(utf8_path), maximum_bytes);
}

void fields(const Json& value, std::initializer_list<std::string_view> required,
            std::initializer_list<std::string_view> optional) {
  if (!value.is_object()) throw SerializationError("Expected a JSON object");
  for (const auto field : required) {
    if (!value.contains(std::string(field)))
      throw SerializationError("Missing required JSON field");
  }
  for (const auto& item : value.items()) {
    if (std::find(required.begin(), required.end(), item.key()) ==
            required.end() &&
        std::find(optional.begin(), optional.end(), item.key()) ==
            optional.end()) {
      throw SerializationError("Unknown JSON field");
    }
  }
}

std::uint64_t unsigned_number(const Json& value, std::uint64_t maximum) {
  if (!(value.is_number_unsigned() ||
        (value.is_number_integer() && value.get<std::int64_t>() >= 0))) {
    throw SerializationError("Expected a nonnegative JSON integer");
  }
  const auto result = value.get<std::uint64_t>();
  if (result > maximum)
    throw ResourceExhausted("JSON integer exceeds the supported range");
  return result;
}

Json parse_json(std::string_view value) {
  if (value.size() > 8U * 1024U * 1024U)
    throw ResourceExhausted("Command JSON input exceeds 8 MiB");
  std::size_t nodes = 0U;
  std::vector<std::set<std::string>> keys;
  const auto callback = [&](int depth, Json::parse_event_t event,
                            Json& parsed) {
    if (depth > 32 || ++nodes > 262144U)
      throw ResourceExhausted("Command JSON structure limit exceeded");
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key &&
        (keys.empty() ||
         !keys.back().insert(parsed.get<std::string>()).second)) {
      throw SerializationError("Duplicate JSON field");
    }
    if (event == Json::parse_event_t::object_end && !keys.empty())
      keys.pop_back();
    if (parsed.is_string() &&
        parsed.get_ref<const std::string&>().size() > 65536U) {
      throw ResourceExhausted("Command JSON string exceeds 64 KiB");
    }
    return true;
  };
  try {
    return Json::parse(value.begin(), value.end(), callback, true, false);
  } catch (const Json::exception&) {
    throw SerializationError("Malformed command JSON or UTF-8");
  }
}

Result execute(const Arguments& arguments) {
  const auto& words = arguments.words();
  if (arguments.optional("version")) {
    arguments.allow({"version"});
    if (!words.empty()) usage("version", "Use --version without a command.");
    return {Json{{"display_name", "六丁炼算"},
                 {"display_name_en", "Liuding Forge"},
                 {"brand", "Liuding Forge"},
                 {"artifact_prefix", "liuding-forge"},
                 {"project_alias", "JYPPX Training Framework"},
                 {"version", version()},
                 {"build", parse_json(build_fingerprint_json())}},
            0};
  }
  if (arguments.optional("help") || words.empty()) {
    arguments.allow({"help"});
    return {
        Json{{"brand", "六丁炼算 (Liuding Forge)"},
             {"usage", "liuding-forge <command> [options] --format text|json"},
             {"commands",
              Json::array(
                  {"doctor [--device cpu|cuda] [--expected-libtorch VERSION]",
                   "config lint|render|explain --schema PATH [--file PATH] "
                   "[--set /path=JSON] [--path POINTER]",
                   "config migrate --input PATH --plan PATH --to-version N "
                   "[--mode preview|apply] [--output NEW_DIRECTORY]",
                   "config migrate --mode rollback --receipt DIRECTORY "
                   "--plan PATH --output NEW_DIRECTORY",
                   "api scan --root SOURCE_DIRECTORY [--max-files N] "
                   "[--max-bytes N] [--max-findings N]",
                   "run start --root PATH --metadata PATH",
                   "run list --root PATH [--after ID] [--limit N]",
                   "run show --root PATH --id ID",
                   "run transition --root PATH --id ID --status NAME "
                   "[--time-ms N]",
                   "run recover --root PATH --stale-ms N [--time-ms N]",
                   "run verify --root PATH",
                   "run events|metrics|export --root PATH --id ID [options]",
                   "run backup --root PATH --destination PATH",
                   "run restore --backup PATH --root PATH",
                   "evaluation run --suite PATH --input PATH [--output PATH] "
                   "[--html PATH]",
                   "evaluation compare --baseline PATH --candidate PATH",
                   "evaluation show --report PATH",
                   "data inspect|verify --input PATH",
                   "checkpoint inspect|verify --path PATH",
                   "checkpoint diff --baseline PATH --candidate PATH",
                   "model inspect|verify --path PATH",
                   "model bundle inspect|verify --path DIRECTORY",
                   "benchmark tensor --device cpu|cuda [options]",
                   "offline run --root PATH [options]"})}},
        0};
  }
  if (words.front() == "doctor" && words.size() == 1U) return doctor(arguments);
  if (words.front() == "model") return model_command(arguments);
  if (words.size() != 2U)
    usage("command", "Use a documented command and subcommand.");
  if (words.front() == "config") return config_command(arguments);
  if (words.front() == "api") return api_command(arguments);
  if (words.front() == "run") return run_command(arguments);
  if (words.front() == "evaluation") return evaluation_command(arguments);
  if (words.front() == "data") return data_command(arguments);
  if (words.front() == "checkpoint") return checkpoint_command(arguments);
  if (words.front() == "benchmark") return benchmark_command(arguments);
  if (words.front() == "offline") return offline_command(arguments);
  usage("command", "Use --help to list supported commands.");
}

int error_exit(ErrorCode code) noexcept {
  if (code == ErrorCode::cancelled) return 130;
  if (code == ErrorCode::deadline_exceeded) return 124;
  if (code == ErrorCode::invalid_argument || code == ErrorCode::serialization ||
      code == ErrorCode::compatibility || code == ErrorCode::not_implemented)
    return 2;
  return 3;
}

int main_entry(std::span<const std::string_view> values) {
  bool requested_json = false;
  for (std::size_t i = 0U; i < values.size(); ++i) {
    if (values[i].starts_with("--format="))
      requested_json = values[i].substr(9U) == "json";
    else if (values[i] == "--format" && i + 1U < values.size())
      requested_json = values[++i] == "json";
  }
  const auto visible = [](std::string_view value) {
    constexpr std::string_view hex = "0123456789abcdef";
    std::string result;
    result.reserve(value.size());
    for (std::size_t index = 0U; index < value.size(); ++index) {
      auto byte = static_cast<unsigned char>(value[index]);
      const bool c1 = byte == 0xc2U && index + 1U < value.size() &&
                      static_cast<unsigned char>(value[index + 1U]) >= 0x80U &&
                      static_cast<unsigned char>(value[index + 1U]) <= 0x9fU;
      if (c1) byte = static_cast<unsigned char>(value[++index]);
      if (byte < 0x20U || byte == 0x7fU || c1) {
        result += "\\u00";
        result.push_back(hex[byte >> 4U]);
        result.push_back(hex[byte & 0x0fU]);
      } else
        result.push_back(static_cast<char>(byte));
    }
    return result;
  };
  const auto render_text = [&](const auto& self, const Json& value,
                               std::size_t indent) -> void {
    const std::string padding(indent, ' ');
    if (value.is_object()) {
      if (value.empty()) std::cout << padding << "(empty)\n";
      for (const auto& [name, field] : value.items()) {
        std::cout << padding << visible(name) << ':';
        if (field.is_structured()) {
          std::cout << '\n';
          self(self, field, indent + 2U);
        } else
          std::cout << ' '
                    << (field.is_string()
                            ? visible(
                                  field.template get_ref<const std::string&>())
                            : field.dump())
                    << '\n';
      }
    } else if (value.is_array()) {
      if (value.empty()) std::cout << padding << "(none)\n";
      for (const auto& field : value) {
        std::cout << padding << '-';
        if (field.is_structured()) {
          std::cout << '\n';
          self(self, field, indent + 2U);
        } else
          std::cout << ' '
                    << (field.is_string()
                            ? visible(
                                  field.template get_ref<const std::string&>())
                            : field.dump())
                    << '\n';
      }
    } else
      std::cout << padding
                << (value.is_string()
                        ? visible(value.get_ref<const std::string&>())
                        : value.dump())
                << '\n';
  };
  const auto render_error = [&](const Json& envelope) {
    if (requested_json)
      std::cout << envelope.dump() << '\n';
    else {
      const auto& error = envelope.at("error");
      std::cout << "error (" << visible(error.at("name").get<std::string>())
                << "): " << visible(error.at("message").get<std::string>())
                << '\n';
      if (error.contains("hint") &&
          !error.at("hint").get<std::string>().empty()) {
        std::cout << "hint: " << visible(error.at("hint").get<std::string>())
                  << '\n';
      }
    }
  };
  try {
    const Arguments arguments(values);
    const auto result = execute(arguments);
    const Json envelope{
        {"schema_version", 1U}, {"status", "ok"}, {"data", result.data}};
    if (arguments.json_format())
      std::cout << envelope.dump() << '\n';
    else
      render_text(render_text, result.data, 0U);
    return result.exit_code;
  } catch (const Error& error) {
    const auto& details = error.details();
    const Json envelope{{"schema_version", 1U},
                        {"status", "error"},
                        {"error",
                         {{"code", static_cast<unsigned>(error.code())},
                          {"name", error_code_name(error.code())},
                          {"message", error.message()},
                          {"component", details.component},
                          {"operation", details.operation},
                          {"hint", details.hint}}}};
    render_error(envelope);
    return error_exit(error.code());
  } catch (const Json::exception&) {
    render_error(Json{{"schema_version", 1U},
                      {"status", "error"},
                      {"error",
                       {{"code", 4U},
                        {"name", "serialization"},
                        {"message", "Invalid command JSON value"},
                        {"hint", "Check the documented input schema."}}}});
    return 2;
  } catch (const std::exception&) {
    render_error(
        Json{{"schema_version", 1U},
             {"status", "error"},
             {"error",
              {{"code", 11U},
               {"name", "internal"},
               {"message", "Command execution failed"},
               {"hint", "Run doctor and inspect the local environment."}}}});
    return 3;
  }
}
}  // namespace jyppx::liuding::cli

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) try {
  std::vector<std::string> storage;
  if (argc > 1) storage.reserve(static_cast<std::size_t>(argc - 1));
  for (int i = 1; i < argc; ++i) {
    const int length =
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, nullptr,
                            0, nullptr, nullptr);
    if (length <= 0) return 2;
    std::string value(static_cast<std::size_t>(length), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1,
                            value.data(), length, nullptr, nullptr) != length)
      return 2;
    value.pop_back();
    storage.push_back(std::move(value));
  }
  std::vector<std::string_view> arguments;
  arguments.reserve(storage.size());
  for (const auto& value : storage) arguments.emplace_back(value);
  return jyppx::liuding::cli::main_entry(arguments);
} catch (...) {
  // Argument allocation and even error rendering can fail before main_entry
  // reaches its normal structured-error boundary. No exception escapes main.
  std::fputs("Command failed before a response could be rendered.\n", stderr);
  return 3;
}
#else
int main(int argc, char* argv[]) try {
  std::vector<std::string_view> arguments;
  if (argc > 1) arguments.reserve(static_cast<std::size_t>(argc - 1));
  for (int i = 1; i < argc; ++i) arguments.emplace_back(argv[i]);
  return jyppx::liuding::cli::main_entry(arguments);
} catch (...) {
  std::fputs("Command failed before a response could be rendered.\n", stderr);
  return 3;
}
#endif
