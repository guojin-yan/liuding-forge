#include <jyppx/distributed/distributed.hpp>

namespace jyppx {
namespace distributed {

namespace {

struct GroupState {
  bool initialized = false;
  std::string backend = "single-process";
  int rank = 0;
  int world_size = 1;
};

GroupState& state() {
  static GroupState state;
  return state;
}

}  // namespace

void init_process_group(const std::string& backend, int rank, int world_size) {
  if (backend != "single-process") {
    throw NotImplemented(
        "distributed backend '" + backend +
            "' requires c10d support, which is not available in this build; "
            "use backend='single-process' (rank=0, world_size=1)",
        ErrorDetails{
            .component = "distributed",
            .operation = "init_process_group",
            .expected = "backend=single-process",
            .actual = backend,
            .hint = "Build with c10d support for cross-process training"});
  }
  if (world_size != 1 || rank != 0) {
    throw InvalidArgument(
        "the single-process backend only supports rank=0 and world_size=1; "
        "cross-process training requires a c10d backend build",
        ErrorDetails{.component = "distributed",
                     .operation = "init_process_group",
                     .expected = "rank=0, world_size=1",
                     .actual = "rank=" + std::to_string(rank) +
                               ", world_size=" + std::to_string(world_size),
                     .hint = "Use a c10d backend for multiple processes"});
  }
  auto& group = state();
  group.initialized = true;
  group.backend = backend;
  group.rank = rank;
  group.world_size = world_size;
}

bool is_initialized() { return state().initialized; }

int get_rank() { return state().rank; }

int get_world_size() { return state().world_size; }

void barrier() {
  // A single-process group is trivially synchronized.
}

Tensor average_gradients(Tensor gradient) {
  // The only constructible group in this build has world_size == 1. Keeping
  // this operation branch-free makes that current capability explicit; a
  // future c10d implementation will own the collective reduction path.
  return gradient;
}

}  // namespace distributed
}  // namespace jyppx
