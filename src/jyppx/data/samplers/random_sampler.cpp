#include <jyppx/data/samplers/random_sampler.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace jyppx::data::samplers {

namespace {

constexpr char kStateMagic[] = "JYPPX-RANDOM-SAMPLER/v1";

uint64_t random_seed() {
  std::random_device device;
  return (static_cast<uint64_t>(device()) << 32) ^
         static_cast<uint64_t>(device());
}

}  // namespace

RandomSampler::RandomSampler(size_t size, Optional<int64_t> seed)
    : size_(size),
      base_seed_(seed.has_value() ? static_cast<std::uint64_t>(*seed)
                                  : random_seed()),
      generator_(static_cast<std::mt19937_64::result_type>(base_seed_)) {
  if (size_ == 0) {
    throw std::invalid_argument("RandomSampler size must be positive");
  }
}

std::vector<size_t> RandomSampler::indices() const {
  std::vector<size_t> result(size_);
  std::iota(result.begin(), result.end(), size_t{0});

  std::shuffle(result.begin(), result.end(), generator_);
  return result;
}

size_t RandomSampler::size() const { return size_; }

void RandomSampler::set_epoch(std::uint64_t epoch) {
  generator_.seed(
      static_cast<std::mt19937_64::result_type>(base_seed_ + epoch));
}

std::string RandomSampler::configuration_identity() const {
  return "jyppx.sampler.random/v1:size=" + std::to_string(size_);
}

std::string RandomSampler::save_state() const {
  std::ostringstream stream;
  stream << kStateMagic << '\n' << base_seed_ << '\n' << generator_;
  return stream.str();
}

void RandomSampler::load_state(const std::string& state) {
  std::string magic;
  std::uint64_t restored_seed = 0;
  auto restored = generator_;
  std::istringstream stream(state);
  if (!std::getline(stream, magic) || magic != kStateMagic ||
      !(stream >> restored_seed) || !(stream >> restored)) {
    throw std::invalid_argument("Invalid RandomSampler state");
  }
  stream >> std::ws;
  if (!stream.eof()) {
    throw std::invalid_argument("RandomSampler state contains trailing data");
  }
  base_seed_ = restored_seed;
  generator_ = restored;
}

}  // namespace jyppx::data::samplers
