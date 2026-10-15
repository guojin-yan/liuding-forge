#include <jyppx/core/device.hpp>
#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/version.hpp>

#include <string>

int main() {
  const jyppx::InvalidArgument error(
      "consumer probe", jyppx::ErrorDetails{.component = "InstallConsumer"});
  const bool valid_version = std::string(jyppx::version()) == "0.3.0";
  const bool valid_error = error.code() == jyppx::ErrorCode::invalid_argument &&
                           error.details().component == "InstallConsumer";
  const bool valid_device = jyppx::parse_device("cpu").is_cpu();
  return valid_version && valid_error && valid_device ? 0 : 1;
}
