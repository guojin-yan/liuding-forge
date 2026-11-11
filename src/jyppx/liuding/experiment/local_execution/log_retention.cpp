// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "log_retention.hpp"

#include "support.hpp"

#include <algorithm>
#include <string_view>

namespace jyppx::liuding::experiment::local_execution_detail {
namespace {
LogRetention read_policy(const Json& value) {
  require_recipe(value.is_string(), "Log retention policy must be a string");
  if (value == "always") return LogRetention::always;
  if (value == "on_failure") return LogRetention::on_failure;
  if (value == "never") return LogRetention::never;
  throw InvalidArgument("Unknown local log retention policy");
}

std::string_view policy_name(LogRetention policy) {
  switch (policy) {
    case LogRetention::always:
      return "always";
    case LogRetention::on_failure:
      return "on_failure";
    case LogRetention::never:
      return "never";
  }
  throw InvalidArgument("Unknown local log retention policy");
}

bool retained(LogRetention policy, bool successful) {
  return policy == LogRetention::always ||
         (policy == LogRetention::on_failure && !successful);
}
}  // namespace

LocalLogRetention read_log_retention(const Json& value) {
  fields(value, {"stdout", "stderr"});
  return {.stdout_policy = read_policy(value.at("stdout")),
          .stderr_policy = read_policy(value.at("stderr"))};
}

Json retain_local_log(LogRetention policy, bool successful,
                      std::string& capture) {
  const bool keep = retained(policy, successful);
  Json receipt{{"policy", policy_name(policy)},
               {"retained", keep},
               {"captured_bytes", capture.size()},
               {"captured_sha256", digest(capture).hex}};
  if (!keep) capture.clear();
  return receipt;
}

void validate_log_retention(const Json& receipt, LogRetention policy,
                            bool successful, std::uint64_t observed_bytes,
                            std::uint64_t capture_limit,
                            const ArtifactRef& stored) {
  fields(receipt, {"policy", "retained", "captured_bytes", "captured_sha256"});
  require_recipe(
      receipt.at("policy") == policy_name(policy) &&
          receipt.at("retained").is_boolean() &&
          receipt.at("retained").get<bool>() == retained(policy, successful),
      "Local log receipt differs from the frozen retention policy");
  const auto count =
      unsigned_number(receipt.at("captured_bytes"), capture_limit);
  require_recipe(count == std::min(observed_bytes, capture_limit),
                 "Local log capture differs from its observed bounded prefix");
  require_recipe(receipt.at("captured_sha256").is_string(),
                 "Local log capture digest must be a string");
  const ArtifactDigest captured{
      .hex = receipt.at("captured_sha256").get<std::string>()};
  validate_artifact_digest(captured);
  const auto empty = digest("");
  require_recipe(count != 0 || captured == empty,
                 "Empty local log capture has a different digest");
  const bool keep = retained(policy, successful);
  require_recipe(stored.size_bytes == (keep ? count : 0) &&
                     stored.digest == (keep ? captured : empty),
                 "Local log content differs from its retention receipt");
}

}  // namespace jyppx::liuding::experiment::local_execution_detail
