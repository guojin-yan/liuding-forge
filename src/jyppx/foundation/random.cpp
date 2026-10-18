#include "jyppx/foundation/random.hpp"

#include "jyppx/foundation/error.hpp"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <string>

namespace jyppx::core {
namespace {

std::vector<at::Generator> cuda_generators() {
  std::vector<at::Generator> generators;
  const auto count =
      torch::cuda::is_available() ? static_cast<int64_t>(at::getNumGPUs()) : 0;
  generators.reserve(static_cast<size_t>(count));
  for (int64_t index = 0; index < count; ++index) {
    generators.push_back(at::globalContext().defaultGenerator(
        at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(index))));
  }
  return generators;
}

void validate_generator_state(const Tensor& state, const Tensor& reference,
                              const char* name) {
  if (!state.defined() || state.device().type() != torch::kCPU ||
      state.scalar_type() != torch::kUInt8 || state.dim() != 1 ||
      state.numel() != reference.numel()) {
    throw InvalidArgument(
        std::string(name) + " RNG state has an incompatible representation",
        ErrorDetails{
            .component = "RNG",
            .operation = "validate_state",
            .expected =
                "defined one-dimensional CPU UInt8 state with matching size",
            .actual = state.defined() ? state.toString() : "undefined tensor",
            .hint = "Restore a state produced by get_rng_state on a compatible "
                    "runtime.",
        });
  }
}

void apply_rng_state(const RngState& state, bool include_cuda) {
  auto generators =
      include_cuda ? cuda_generators() : std::vector<at::Generator>{};
  auto cpu_generator = at::detail::getDefaultCPUGenerator();
  cpu_generator.set_state(state.cpu.contiguous());
  for (size_t i = 0; i < generators.size(); ++i) {
    generators[i].set_state(state.cuda[i].contiguous());
  }
}

void seed_generators(std::uint64_t seed, bool include_cuda) {
  auto generators =
      include_cuda ? cuda_generators() : std::vector<at::Generator>{};
  auto cpu_generator = at::detail::getDefaultCPUGenerator();
  cpu_generator.set_current_seed(seed);
  for (auto& generator : generators) generator.set_current_seed(seed);
}

[[noreturn]] void translate_backend(const c10::Error& cause,
                                    const char* operation) {
  detail::rethrow_backend_error(
      cause,
      {.component = "RNG",
       .operation = operation,
       .failure_kind = detail::BackendFailureKind::device,
       .expected = "compatible CPU/CUDA default generator state",
       .hint = "Check the active device topology and LibTorch/CUDA build."});
}

void seed_generators_atomically(std::uint64_t seed, bool include_cuda,
                                const RngState& previous) {
  try {
    seed_generators(seed, include_cuda);
  } catch (const c10::Error& cause) {
    try {
      apply_rng_state(previous, include_cuda);
    } catch (...) {
    }
    translate_backend(cause, "manual_seed");
  }
}

std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

}  // namespace

RngState get_rng_state(bool include_cuda) {
  try {
    RngState state;
    state.cpu = at::detail::getDefaultCPUGenerator().get_state().cpu().clone();
    if (include_cuda) {
      for (auto& generator : cuda_generators()) {
        state.cuda.push_back(generator.get_state().cpu().clone());
      }
    }
    return state;
  } catch (const Error&) {
    throw;
  } catch (const c10::Error& cause) {
    translate_backend(cause, "get_state");
  }
}

void validate_rng_state(const RngState& state, bool include_cuda) {
  if (state.format_version != kRngStateFormatVersion) {
    throw CompatibilityError(
        "Unsupported RNG state format",
        ErrorDetails{
            .component = "RNG",
            .operation = "validate_state",
            .expected = std::to_string(kRngStateFormatVersion),
            .actual = std::to_string(state.format_version),
            .hint = "Migrate the checkpoint before restoring RNG state.",
        });
  }
  const auto current = get_rng_state(include_cuda);
  validate_generator_state(state.cpu, current.cpu, "CPU");
  if (include_cuda && state.cuda.size() != current.cuda.size()) {
    throw CompatibilityError(
        "CUDA RNG state topology does not match this process",
        ErrorDetails{
            .component = "RNG",
            .operation = "validate_state",
            .expected = std::to_string(current.cuda.size()) + " CUDA states",
            .actual = std::to_string(state.cuda.size()) + " CUDA states",
            .hint = "Restore with the same visible-device mapping or disable "
                    "CUDA restore.",
        });
  }
  if (!include_cuda && !state.cuda.empty()) {
    throw InvalidArgument(
        "CUDA RNG state supplied while CUDA restore is disabled",
        "Pass include_cuda=true or supply a CPU-only state.");
  }
  for (size_t i = 0; i < state.cuda.size(); ++i) {
    validate_generator_state(state.cuda[i], current.cuda[i], "CUDA");
  }
}

void set_rng_state(const RngState& state, bool include_cuda) {
  validate_rng_state(state, include_cuda);
  const auto current = get_rng_state(include_cuda);
  try {
    apply_rng_state(state, include_cuda);
  } catch (const c10::Error& cause) {
    try {
      apply_rng_state(current, include_cuda);
    } catch (...) {
    }
    translate_backend(cause, "set_state");
  }
}

void manual_seed(std::uint64_t seed, bool include_cuda) {
  const auto previous = get_rng_state(include_cuda);
  seed_generators_atomically(seed, include_cuda, previous);
}

std::uint64_t derive_seed(std::uint64_t base_seed, std::uint64_t stream_id,
                          std::uint64_t epoch,
                          std::uint64_t worker_id) noexcept {
  auto value = splitmix64(base_seed);
  value = splitmix64(value ^ stream_id);
  value = splitmix64(value ^ epoch);
  return splitmix64(value ^ worker_id);
}

RngGuard::RngGuard(std::uint64_t seed, bool include_cuda)
    : previous_(get_rng_state(include_cuda)), include_cuda_(include_cuda) {
  seed_generators_atomically(seed, include_cuda_, previous_);
}

RngGuard::~RngGuard() noexcept {
  try {
    set_rng_state(previous_, include_cuda_);
  } catch (...) {
  }
}

}  // namespace jyppx::core
