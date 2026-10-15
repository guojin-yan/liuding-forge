#include <jyppx/core/device.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/version.hpp>

#include <cstddef>
#include <string>
#include <vector>

int main() {
  const jyppx::InvalidArgument error(
      "consumer probe", jyppx::ErrorDetails{.component = "InstallConsumer"});
  const bool valid_version = std::string(jyppx::version()) == "0.3.0";
  const bool valid_error = error.code() == jyppx::ErrorCode::invalid_argument &&
                           error.details().component == "InstallConsumer";
  const bool valid_device = jyppx::parse_device("cpu").is_cpu();
  const auto indices = jyppx::data::samplers::SequentialSampler(3).indices();
  const bool valid_data = indices == std::vector<std::size_t>({0, 1, 2});
  return valid_version && valid_error && valid_device && valid_data ? 0 : 1;
}
