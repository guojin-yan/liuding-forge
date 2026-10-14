# API: Optimizers and Schedulers

All types live in the `jyppx::optim` namespace.

## Optimizer base

`#include <jyppx/optim/optimizer.hpp>`

```cpp
struct ParamGroup {
    TensorList params;                      // parameters in this group
    std::map<std::string, double> options;  // per-group hyperparameters
};

class Optimizer {
public:
    explicit Optimizer(const TensorList& params);  // throws on empty list
    virtual void step() = 0;
    void zero_grad(bool set_to_none = false);      // false zeroes, true clears
    std::vector<ParamGroup>& param_groups();
    void add_param_group(ParamGroup group);
    std::vector<double> get_lrs() const;           // throws if a group lacks "lr"
    void set_lr(double lr);
    void set_lrs(const std::vector<double>& lrs);
    int64_t step_count() const;
    virtual std::map<std::string, torch::Tensor> state_dict() const;
    virtual void load_state_dict(const std::map<std::string, torch::Tensor>& state);
};
```

`add_param_group()` rejects empty groups, undefined tensors, duplicate tensor
identities, non-finite options, and negative learning rates. Omitted options
inherit from the first group.

`state_dict()` round-trips hyperparameters (as float64) and per-parameter
optimizer state (e.g. momentum buffers), keyed by stable parameter order, so a
fresh optimizer can resume an interrupted run. Loading requires the current
format metadata, validates parameter-group and parameter counts plus every
declared state count, and replaces buffers atomically instead of merging them
with stale state.

## Optimizers

Every optimizer takes the parameter list plus an options struct and validates
the hyperparameters at construction. All of them are validated numerically
against `torch::optim` (float64, tolerance 1e-9) by the unit tests.

```cpp
struct SGDOptions   { double lr = 0.01; double momentum = 0.0; double dampening = 0.0;
                      double weight_decay = 0.0; bool nesterov = false; };
struct AdamOptions  { double lr = 0.001; double beta1 = 0.9; double beta2 = 0.999;
                      double eps = 1e-8; double weight_decay = 0.0; bool amsgrad = false; };
struct AdamWOptions { double lr = 0.001; double beta1 = 0.9; double beta2 = 0.999;
                      double eps = 1e-8; double weight_decay = 0.01; bool amsgrad = false; };
struct RMSpropOptions { double lr = 0.01; double alpha = 0.99; double eps = 1e-8;
                        double weight_decay = 0.0; double momentum = 0.0;
                        bool centered = false; };
struct AdagradOptions { double lr = 0.01; double lr_decay = 0.0;
                        double weight_decay = 0.0; double eps = 1e-10; };
```

```cpp
#include <jyppx/optim/sgd.hpp>       // also adam.hpp, adamw.hpp, rmsprop.hpp, adagrad.hpp

auto optimizer = std::make_shared<jyppx::optim::Adam>(
    model->parameters(), jyppx::optim::AdamOptions{.lr = 1e-3, .weight_decay = 1e-4});
```

Notes:

- **AdamW** applies decoupled weight decay (`param *= 1 - lr * wd`) like
  `torch.optim.AdamW`.
- **RMSprop momentum** follows the `torch::optim` C++ semantics: the buffer
  accumulates the normalized gradient `grad / avg` and the step is
  `-lr * buffer`. This coincides with `torch.optim.rmsprop` in Python on the
  first step and differs from the second step on.
- Mutating `options()` (e.g. `optimizer->options().lr = 0.01`) is picked up
  on the next `step()`; schedulers use `set_lr`/`set_lrs` instead.

## Learning-rate schedulers

`#include <jyppx/optim/schedulers/...>`

```cpp
class LRScheduler {
public:
    explicit LRScheduler(std::shared_ptr<Optimizer> optimizer);  // captures base lrs
    virtual void step() = 0;             // epoch-driven schedulers
    virtual void step(double metric);    // metric-driven (ReduceLROnPlateau)
    virtual bool requires_metric() const;
    double get_last_lr() const;          // first parameter group
    const std::vector<double>& get_last_lrs() const;
    int64_t get_step() const;
    std::map<std::string, double> state_dict() const;
    void load_state_dict(const std::map<std::string, double>& state);
};
```

Epoch-based schedulers reject `step(metric)`; `ReduceLROnPlateau` rejects the
metric-less `step()`. Every `step()` also writes the new learning rate back
into the optimizer.

```cpp
#include <jyppx/optim/schedulers/step_lr.hpp>

auto scheduler = std::make_shared<jyppx::optim::schedulers::StepLR>(
    optimizer, /*step_size=*/2, /*gamma=*/0.1);
scheduler->step();   // once per epoch
```

| Scheduler | Parameters | Behavior |
|-----------|-----------|----------|
| `StepLR` | `step_size`, `gamma` | `lr = base * gamma^(step / step_size)` |
| `ExponentialLR` | `gamma` | `lr = base * gamma^step` |
| `CosineAnnealingLR` | `T_max`, `eta_min` | cosine decay from base lr to `eta_min` |
| `MultiStepLR` | `milestones`, `gamma` | decays by gamma at each milestone |
| `ReduceLROnPlateau` | options below | decays when a metric stops improving |
| `OneCycleLR` | `max_lr`, `total_steps` | one-cycle warmup and cooldown |
| `CosineAnnealingWarmRestarts` | `T_0`, `T_mult`, `eta_min` | cosine cycles with restarts |
| `LinearLR` | `start_factor`, `end_factor`, `total_iters` | linear factor schedule |
| `CyclicLR` | `base_lr`, `max_lr`, `step_size_up` | triangular or exponential cycles |
| `PolynomialLR` | `total_iters`, `power` | polynomial decay |
| `ConstantLR` | `factor`, `total_iters` | constant factor for a prefix |
| `LambdaLR` | callable factor | user-defined factor per step |
| `SequentialLR` / `ChainedScheduler` | scheduler list | compose schedules |
| `WarmupScheduler` | `warmup_iters`, `start_factor`, optional scheduler | linear warmup wrapper |

`OneCycleLR` is stepped once per optimizer update. Its two-phase and
three-phase boundaries, cosine/linear interpolation, and inverse momentum or
Adam beta1 schedule follow PyTorch. `CyclicLR` supports triangular,
triangular2, exp-range, and custom scale functions, with optional inverse
momentum or beta1 cycling.

```cpp
struct ReduceLROnPlateauOptions {
    std::string mode = "min";            // "min" or "max"
    double factor = 0.1;
    int patience = 10;
    double threshold = 1e-4;
    std::string threshold_mode = "rel";  // "rel" or "abs"
    double min_lr = 0.0;
};

jyppx::optim::schedulers::ReduceLROnPlateauOptions options;
options.mode = "min";     // "min" or "max"
options.factor = 0.5;
options.patience = 2;

auto plateau = std::make_shared<jyppx::optim::schedulers::ReduceLROnPlateau>(
    optimizer, options);
// after each validation:
plateau->step(val_loss);
```

`ReduceLROnPlateau` also exposes `best()` and `num_bad_epochs()`.
All schedulers can serialize their step counter and current per-group rates;
`ReduceLROnPlateau` additionally stores its best metric and bad-epoch counter.
Loading a scheduler state also restores the optimizer's learning rates.
