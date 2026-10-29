// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

// Diagnostic-only exact production fragment; orchestration checks for drift.
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
