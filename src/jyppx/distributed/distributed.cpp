#include <jyppx/distributed/distributed.hpp>

#include <stdexcept>

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

} // namespace

void init_process_group(const std::string& backend, int rank, int world_size) {
    if (backend != "single-process") {
        throw std::runtime_error(
            "distributed backend '" + backend +
            "' requires c10d support, which is not available in this build; "
            "use backend='single-process' (rank=0, world_size=1)");
    }
    if (world_size != 1 || rank != 0) {
        throw std::runtime_error(
            "the single-process backend only supports rank=0 and world_size=1; "
            "cross-process training requires a c10d backend build");
    }
    auto& group = state();
    group.initialized = true;
    group.backend = backend;
    group.rank = rank;
    group.world_size = world_size;
}

bool is_initialized() {
    return state().initialized;
}

int get_rank() {
    return state().rank;
}

int get_world_size() {
    return state().world_size;
}

void barrier() {
    // A single-process group is trivially synchronized.
}

Tensor average_gradients(Tensor gradient) {
    if (state().world_size <= 1) {
        return gradient;
    }
    throw std::runtime_error(
        "gradient all-reduce requires c10d support, which is not available in "
        "this build");
}

} // namespace distributed
} // namespace jyppx
