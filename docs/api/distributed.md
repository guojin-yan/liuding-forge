# API: Distributed (Phase 5 groundwork)

Types live in `jyppx::distributed`.

## Process group

`#include <jyppx/distributed/distributed.hpp>`

```cpp
namespace jyppx::distributed {
void init_process_group(const std::string& backend = "single-process",
                        int rank = 0, int world_size = 1);
bool is_initialized();
int get_rank();
int get_world_size();
void barrier();
Tensor average_gradients(Tensor gradient);
}
```

This is a **single-process reference implementation**: it accepts
`init_process_group("single-process", 0, 1)` so distributed-aware code paths
run unchanged on one process. Requesting any other backend or a larger world
size throws `std::runtime_error` — cross-process collectives require a c10d
backend build.

## DistributedDataParallel wrapper

`#include <jyppx/distributed/ddp.hpp>`

```cpp
template <typename ModuleT>
class DistributedDataParallel {
public:
    explicit DistributedDataParallel(std::shared_ptr<ModuleT> module);
    template <typename... Args> auto forward(Args&&... args);
    std::shared_ptr<ModuleT> module() const;
};
```

Wraps a typed module: `forward` delegates to the wrapped module and every
parameter gradient is routed through `distributed::average_gradients` via
autograd hooks. With the single-process backend the hooks are no-ops, so
DDP-style code runs locally; a c10d backend build turns the hooks into real
all-reduces.

```cpp
#include <jyppx/distributed/ddp.hpp>

auto model = std::make_shared<MyNet>();
jyppx::distributed::DistributedDataParallel<MyNet> ddp(model);
auto loss = some_loss(ddp.forward(input), target);
loss.backward();   // gradients pass through the hook chain
optimizer->step();
```

## DistributedSampler

`#include <jyppx/data/samplers/weighted_random_sampler.hpp>`

Shards an epoch across ranks (see [Data](data.md#samplers)):

```cpp
jyppx::data::samplers::DistributedSampler sampler(
    dataset->size(),
    jyppx::distributed::get_rank(),
    jyppx::distributed::get_world_size(),
    /*shuffle=*/true, /*seed=*/42);
sampler.set_epoch(epoch);  // call once per training epoch
```

Every rank receives the same number of samples (padded unless
`drop_last`-style truncation is used), and ranks partition the epoch
disjointly.
