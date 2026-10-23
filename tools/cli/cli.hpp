// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TOOLS_CLI_HPP
#define JYPPX_TOOLS_CLI_HPP

#include <jyppx/liuding/foundation/error.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::cli {
using Json = nlohmann::json;

struct Result {
  Json data;
  int exit_code = 0;
};

class Arguments {
 public:
  explicit Arguments(std::span<const std::string_view> arguments);
  void allow(std::initializer_list<std::string_view> names) const;
  const std::vector<std::string>& words() const noexcept;
  std::optional<std::string> optional(std::string_view name) const;
  std::string require(std::string_view name) const;
  const std::vector<std::string>& repeated(std::string_view name) const;
  std::uint64_t number(std::string_view name, std::uint64_t default_value,
                       std::uint64_t maximum) const;
  bool json_format() const noexcept;

 private:
  std::vector<std::string> words_;
  std::map<std::string, std::vector<std::string>> options_;
  bool json_ = false;
};

[[noreturn]] void usage(std::string_view operation, std::string_view hint);
std::string read_text(const std::filesystem::path& path,
                      std::size_t maximum_bytes = 8U * 1024U * 1024U);
std::string read_text(const std::string& utf8_path,
                      std::size_t maximum_bytes = 8U * 1024U * 1024U);
Json parse_json(std::string_view text);
void fields(const Json& value, std::initializer_list<std::string_view> required,
            std::initializer_list<std::string_view> optional = {});
std::uint64_t unsigned_number(const Json& value, std::uint64_t maximum);
Result doctor(const Arguments& arguments);
Result config_command(const Arguments& arguments);
Result api_command(const Arguments& arguments);
Result run_command(const Arguments& arguments);
Result evaluation_command(const Arguments& arguments);
Result data_command(const Arguments& arguments);
Result checkpoint_command(const Arguments& arguments);
Result model_command(const Arguments& arguments);
Result benchmark_command(const Arguments& arguments);
Result offline_command(const Arguments& arguments);

}  // namespace jyppx::liuding::cli

#endif  // JYPPX_TOOLS_CLI_HPP
