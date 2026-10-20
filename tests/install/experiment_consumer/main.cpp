#include <jyppx/experiment/config.hpp>

#include <cstdint>
#include <string>

int main() {
  c10::impl::GenericDict defaults(c10::StringType::get(), c10::AnyType::get());
  defaults.insert("epochs", std::int64_t{2});
  defaults.insert("token", "secret://env/consumer/token");
  jyppx::experiment::config::ConfigSchema schema{
      .schema_id = "consumer.config",
      .version = 1,
      .root = jyppx::ValueSchema::dictionary(
          {{"epochs", jyppx::ValueSchema{.kind = jyppx::ValueKind::integer}},
           {"token", jyppx::ValueSchema{.kind = jyppx::ValueKind::string}}}),
      .defaults = c10::IValue(defaults),
      .environment_bindings = {},
      .secret_paths = {"/token"},
  };
  jyppx::experiment::config::ConfigResolver resolver(schema);
  resolver.add_cli_override("/epochs", "3");
  const auto resolved = resolver.resolve();
  const auto capability =
      jyppx::experiment::config::make_config_capability(schema);
  return resolved.integer("/epochs") == 3 &&
                 resolved.explain("/token").redacted &&
                 capability.scope_id == "consumer.config" &&
                 resolved.canonical_json().find("secret://") ==
                     std::string::npos
             ? 0
             : 1;
}
