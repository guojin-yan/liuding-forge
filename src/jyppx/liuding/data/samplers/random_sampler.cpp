// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#include <jyppx/liuding/data/detail/error.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>

#include <algorithm>
#include <charconv>
#include <limits>
#include <numeric>
#include <random>

namespace jyppx::liuding::data::samplers {

namespace {

constexpr char kStateMagic[] = "JYPPX-RANDOM-SAMPLER/v2";
static_assert(std::numeric_limits<size_t>::digits <= 64);

std::uint64_t next_word(std::uint64_t& state) {
  // All arithmetic is unsigned modulo 2^64; no standard-library distribution
  // or engine serialization participates in the portable contract.
  auto mixed = (state += UINT64_C(0x9e3779b97f4a7c15));
  mixed = (mixed ^ (mixed >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  mixed = (mixed ^ (mixed >> 27)) * UINT64_C(0x94d049bb133111eb);
  return mixed ^ (mixed >> 31);
}

bool read_decimal(std::string_view& input, std::uint64_t& value,
                  bool last = false) {
  const auto separator = input.find('\n');
  if ((separator == std::string_view::npos) != last) return false;
  const auto token = input.substr(0, separator);
  if (token.empty() || (token.size() > 1 && token.front() == '0')) return false;
  const auto [end, error] =
      std::from_chars(token.data(), token.data() + token.size(), value);
  if (error != std::errc{} || end != token.data() + token.size()) return false;
  if (!last) input.remove_prefix(separator + 1);
  return true;
}

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
      generator_state_(base_seed_) {
  if (size_ == 0) {
    ::jyppx::liuding::data::detail::throw_invalid(
        "data.random_sampler", "constructor.size", "positive population",
        "zero", "Provide a dataset with at least one sample.");
  }
}

std::vector<size_t> RandomSampler::indices() const {
  std::vector<size_t> result(size_);
  std::iota(result.begin(), result.end(), size_t{0});

  for (size_t remaining = size_; remaining > 1; --remaining) {
    const auto bound = static_cast<std::uint64_t>(remaining);
    const auto threshold = (std::uint64_t{0} - bound) % bound;
    auto word = next_word(generator_state_);
    while (word < threshold) word = next_word(generator_state_);
    std::swap(result[remaining - 1], result[static_cast<size_t>(word % bound)]);
  }
  return result;
}

size_t RandomSampler::size() const { return size_; }

void RandomSampler::set_epoch(std::uint64_t epoch) {
  generator_state_ = base_seed_ + epoch;
}

std::string RandomSampler::configuration_identity() const {
  return "jyppx.sampler.random/v2:algorithm=" + std::string(algorithm_id()) +
         ":size=" + std::to_string(size_);
}

std::string RandomSampler::save_state() const {
  return std::string(kStateMagic) + '\n' + std::to_string(size_) + '\n' +
         std::to_string(base_seed_) + '\n' + std::to_string(generator_state_);
}

void RandomSampler::load_state(const std::string& state) {
  std::string_view input(state);
  constexpr std::string_view header = "JYPPX-RANDOM-SAMPLER/v2\n";
  std::uint64_t restored_size = 0;
  std::uint64_t restored_seed = 0;
  std::uint64_t restored_state = 0;
  if (!input.starts_with(header)) {
    ::jyppx::liuding::data::detail::throw_state_error(
        "data.random_sampler", "load_state", "unsupported state version");
  }
  input.remove_prefix(header.size());
  if (!read_decimal(input, restored_size) || restored_size != size_ ||
      !read_decimal(input, restored_seed) ||
      !read_decimal(input, restored_state, true)) {
    ::jyppx::liuding::data::detail::throw_state_error(
        "data.random_sampler", "load_state",
        "invalid population, seed, or generator word");
  }
  base_seed_ = restored_seed;
  generator_state_ = restored_state;
}

}  // namespace jyppx::liuding::data::samplers
