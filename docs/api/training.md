# API: Training

Types live in `jyppx::training` (callbacks additionally in
`jyppx::training::callbacks`, metrics in `jyppx::training::metrics`, AMP in
`jyppx::training::amp`).

## Trainer

`#include <jyppx/training/trainer.hpp>`

```cpp
struct TrainerOptions {
    using ForwardFn = std::function<Tensor(const Tensor&)>;
    using LossFn = std::function<Tensor(const Tensor&, const Tensor&)>;

    int max_epochs = 10;
    int max_steps = -1;                      // stop after N optimizer steps
    Device device = torch::kCPU;
    double gradient_clip_val = 0.0;          // 0 disables clipping
    std::string gradient_clip_mode = "norm"; // "norm" or "value"
    bool use_amp = false;                    // enables scoped autocast
    at::ScalarType amp_dtype = at::kHalf;    // CUDA Half/BFloat16, CPU BFloat16
    amp::GradScalerOptions amp_options;
    int accumulation_steps = 1;              // gradient accumulation
    ForwardFn forward_fn;                    // set automatically by the template ctor
    LossFn loss_fn = mse_loss;               // replace for other objectives
    std::vector<std::shared_ptr<Callback>> callbacks;
    uint64_t max_checkpoint_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
};
```

Construction. The template constructor accepts any
`std::shared_ptr<ModuleT>` whose `ModuleT` derives from `torch::nn::Module`
and exposes `Tensor forward(const Tensor&)`; it type-erases the forward call
automatically:

```cpp
jyppx::training::Trainer trainer(model, optimizer,
                                 jyppx::training::TrainerOptions{.max_epochs = 20});
```

A base `torch::nn::Module` pointer (no typed `forward`) is also accepted; in
that case provide the forward function explicitly:

```cpp
jyppx::training::Trainer trainer(base_model, optimizer, options);
trainer.set_forward_fn([](const Tensor& input) { return module_call(input); });
// or pass TrainerOptions::forward_fn at construction
```

Training and evaluation:

```cpp
trainer.fit(train_loader);                    // fit(loader, val_loader*) optional
trainer.fit(train_loader, &val_loader);
double val = trainer.validate(val_loader);    // mean loss, no callbacks state change
double test = trainer.test(test_loader);      // same loop, callback-free
```

`fit` runs the callback lifecycle (below), moves the model to the configured
device, applies gradient accumulation (loss divided by `accumulation_steps`,
rescaled when the last group is smaller), gradient clipping and max-step
limits. Validation temporarily switches the model to eval mode and restores
the previous training mode afterwards.

Loop state accessors: `global_step()`, `current_epoch()`,
`last_train_loss()`, `last_val_loss()`, `has_validation_loss()`,
`should_stop()`, `request_stop()` (soft stop after the current epoch),
`model()`, `optimizer()`, `scheduler()`, `grad_scaler()`.

### Checkpoints

```cpp
trainer.save_checkpoint("run.pt");    // model, optimizer, scheduler/scaler, losses
trainer.load_checkpoint("run.pt");    // restores weights + all loop state
// fitting again resumes at next_epoch; optimizer moments are restored too
```

Checkpoints are strict version-2 LibTorch archives. They contain model and
optimizer state, scheduler and float16 `GradScaler` state, Trainer counters and
losses, callback state, CPU/CUDA RNG state, and restorable train/validation
DataLoader state. Saving is allowed only at an epoch boundary, is bounded by
`max_checkpoint_bytes`, and atomically replaces the destination file.

`load_checkpoint()` first parses and validates the complete archive, current
format version, training/device topology, callback order, RNG topology, and
component presence. Applying the state is transactional: if any participant
rejects its state, all already-applied model, optimizer, scheduler, scaler,
callback, RNG, loader, and Trainer state is rolled back. Older unversioned
archives are intentionally rejected rather than routed through a compatibility
reader. A metric-driven scheduler consumes the validation loss after each
validation epoch and requires a validation loader.

### Custom loss

```cpp
options.loss_fn = [](const Tensor& prediction, const Tensor& target) {
    return torch::mse_loss(prediction, target);   // must return a scalar tensor
};
```

## Loss functions

Use `#include <jyppx/nn/losses.hpp>` for the remaining legacy loss group and
`#include <jyppx/nn/loss/focal_loss.hpp>`,
`#include <jyppx/nn/loss/contrastive_loss.hpp>` or
`#include <jyppx/nn/loss/dice_loss.hpp>` or
`#include <jyppx/nn/loss/label_smoothing_cross_entropy.hpp>` for migrated
focused adapters. The migrated APIs are owned by and directly linkable as
`jyppx::nn`.

The focused headers provide both the functional and module forms of
`FocalLoss`, `ContrastiveLoss`, `DiceLoss` and
`LabelSmoothingCrossEntropy`. All focused losses accept batch reduction
`"none"`, `"mean"` or `"sum"`.

The complete label-smoothing shape, dtype, device, empty-input and error
contract is recorded in the
[semantic matrix](losses/label_smoothing_cross_entropy.md).
The contrastive pair contract is recorded in the
[contrastive semantic matrix](losses/contrastive_loss.md).
The Dice task modes, aggregation boundaries and masking policies are recorded
in the [Dice semantic matrix](losses/dice_loss.md).

```cpp
namespace F = jyppx::nn::functional;

// Focal loss (Lin et al., 2017) over {N, C} logits and class indices.
auto focal = F::focal_loss(logits, target, /*alpha=*/0.25, /*gamma=*/2.0);

// Soft Dice loss with explicit task and aggregation semantics.
jyppx::nn::DiceLossOptions dice_options;
dice_options.mode = jyppx::nn::DiceMode::multiclass;
dice_options.ignore_index = -1;
auto dice = F::dice_loss(logits, mask, dice_options);

// Cross entropy with label smoothing (matches torch's label_smoothing).
auto smoothed = F::label_smoothing_cross_entropy(logits, target, 0.1);

// Metric-learning losses.
auto contrastive = F::contrastive_loss(x1, x2, pair_labels, /*margin=*/1.0);
```

For triplet margin loss, use LibTorch's `torch::nn::functional::triplet_margin_loss`
or `torch::nn::TripletMarginLoss` directly. The removed JYPPX wrapper added no
semantics and hid LibTorch's `p`, `eps` and `swap` options.

Notes:

- Binary focal inputs are selected by matching input/target shape. Dice mode is
  always explicit so binary, multiclass and multilabel shapes cannot be
  confused by rank alone.
- `focal_loss` with `gamma = 0` reduces to alpha-weighted cross entropy.
- All functions keep computations in tensor ops, so gradients flow to the
  logits; the tests verify gradients are finite and hand-computed values
  match to 1e-12.

## Callbacks

`#include <jyppx/training/callback.hpp>` and `.../callbacks/...`

```cpp
class Callback {
public:
    virtual void on_fit_start(Trainer*) {}
    virtual void on_fit_end(Trainer*) {}
    virtual void on_epoch_start(Trainer*) {}
    virtual void on_epoch_end(Trainer*) {}
    virtual void on_train_epoch_start(Trainer*) {}
    virtual void on_train_epoch_end(Trainer*) {}
    virtual void on_validation_epoch_start(Trainer*) {}
    virtual void on_validation_epoch_end(Trainer*) {}
    virtual void on_train_batch_start(Trainer*, int batch_idx) {}
    virtual void on_train_batch_end(Trainer*, int batch_idx, double loss) {}
    virtual void on_validation_batch_start(Trainer*, int batch_idx) {}
    virtual void on_validation_batch_end(Trainer*, int batch_idx) {}
    virtual std::string state_key() const { return {}; }
    virtual CallbackState state_dict() const { return {}; }
    virtual void load_state_dict(const CallbackState&) {}
};
```

Callbacks with a non-empty `state_key()` are checkpoint participants. Their
order and key must match during restore, and `load_state_dict()` must validate
before mutation so Trainer rollback can restore the entry state.

Built-in callbacks:

- `ProgressCallback()` — per-epoch progress lines.
- `LoggingCallback(log_every_n_steps = 50, log_file = "")` — batch/epoch
  logging through the framework logger (optionally into a log file).
- `LRMonitor()` — logs the learning rate at each epoch start.
- `TensorBoardCallback(log_dir, log_every_n_steps = 50)` — writes real
  TensorBoard event files (batch loss, epoch losses, learning rates).
- `EarlyStopping(monitor = "val_loss", patience = 10, min_delta = 0.0,
  mode = "min")` — requests a stop after `patience` non-improving epochs;
  monitors `"val_loss"` or `"train_loss"`. Exposes `should_stop()`/`reset()`.
- `ModelCheckpoint(filepath, monitor = "val_loss", mode = "min",
  save_top_k = 1)` — retains the best K full Trainer checkpoints (`-1` saves
  every eligible epoch and `0` disables saving), creates parent directories,
  and publishes only after all callbacks have observed the epoch boundary.
- `LRSchedulerCallback(scheduler)` — steps epoch-based schedulers at each
  epoch end and passes the validation loss to metric-driven schedulers
  (redundant with `trainer.set_scheduler`; both guard against double stepping
  the same scheduler).

Attach with `trainer.add_callback(std::make_shared<...>(...))`.

## Metrics

`#include <jyppx/training/metric.hpp>` and `.../metrics/...`

```cpp
class Metric {
public:
    explicit Metric(const std::string& name);
    virtual void update(const Tensor& preds, const Tensor& targets) = 0;
    virtual double compute() = 0;
    virtual void reset() = 0;
    const std::string& name() const;
};
```

Built-ins:

| Metric | Input shape | Notes |
|--------|-------------|-------|
| `Accuracy` | 1D scores (binarized `> 0`) or 2D logits (argmax) | running mean |
| `TopKAccuracy(k)` | `{batch, classes}` scores + class indices | target in top-k |
| `Precision` / `Recall` / `F1Score` | same as Accuracy | macro-averaged over the accumulated confusion matrix |
| `MSE` / `MAE` | any matching shapes | accumulated over all elements |
| `R2Score` | any matching shapes | exact whole-set R² |
| `AverageMeter` | scalar / tensor mean | weighted running average |
| `MetricCollection` | — | named bundle, `compute()` → `map<string, double>` |

```cpp
jyppx::training::metrics::MetricCollection metrics;
metrics.add(std::make_shared<jyppx::training::metrics::Accuracy>());
metrics.add(std::make_shared<jyppx::training::metrics::TopKAccuracy>(5));
metrics.update(logits, targets);
auto values = metrics.compute();   // {"accuracy": ..., "top5_accuracy": ...}
```

## Mixed precision (AMP)

`#include <jyppx/training/amp/grad_scaler.hpp>` and `.../amp/autocast.hpp`

- `GradScaler` mirrors `torch.cuda.amp.GradScaler`: `scale(loss)`,
  `unscale(optimizer)`, `step(optimizer)` (skips and backs off on inf/NaN
  gradients), `update()` (grows the scale after `growth_interval` clean
  steps), `get_scale()`, `found_inf()`, `state_dict()`/`load_state_dict()`.
- `AutocastGuard` is an RAII wrapper that restores both the prior enabled flag
  and dtype, including nested scopes:
  `amp::AutocastGuard guard{c10::DeviceType::CUDA, at::kHalf};`

Set `TrainerOptions::use_amp = true` and choose `amp_dtype`. CUDA accepts Half
or BFloat16; CPU accepts BFloat16 only. Float16 uses `GradScaler`; BFloat16 does
not. The trainer runs forward passes under autocast, scales float16 loss before
backward, unscales before gradient clipping and routes optimizer steps through
the scaler. It works with gradient accumulation and clipping.
