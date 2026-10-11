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
    bool use_amp = false;                    // CUDA only, validated at construction
    amp::GradScalerOptions amp_options;
    int accumulation_steps = 1;              // gradient accumulation
    ForwardFn forward_fn;                    // set automatically by the template ctor
    LossFn loss_fn = mse_loss;               // replace for other objectives
    std::vector<std::shared_ptr<Callback>> callbacks;
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
trainer.save_checkpoint("run.pt");    // model, optimizer state, counters
trainer.load_checkpoint("run.pt");    // restores weights + state + next_epoch
// fitting again resumes at next_epoch; optimizer moments are restored too
```

Checkpoints are LibTorch archives; `load_checkpoint` restores
`global_step`, `current_epoch` and `next_epoch` so continued training picks
up where the save left off.

### Custom loss

```cpp
options.loss_fn = [](const Tensor& prediction, const Tensor& target) {
    return torch::mse_loss(prediction, target);   // must return a scalar tensor
};
```

## Loss functions

`#include <jyppx/nn/losses.hpp>`

Functional implementations in `jyppx::nn::functional`, plus criterion
modules (`nn::FocalLoss`, `nn::DiceLoss`, `nn::LabelSmoothingCrossEntropy`,
`nn::ContrastiveLoss`, `nn::TripletLoss`) whose `forward` calls them.
All functions accept `reduction` = `"none"`, `"mean"` or `"sum"` (Dice is
mean-only).

```cpp
namespace F = jyppx::nn::functional;

// Focal loss (Lin et al., 2017) over {N, C} logits and class indices.
auto focal = F::focal_loss(logits, target, /*alpha=*/0.25, /*gamma=*/2.0);

// Soft Dice loss for segmentation (logits in, per-class Dice averaged).
auto dice = F::dice_loss(logits, mask, /*smooth=*/1.0);

// Cross entropy with label smoothing (matches torch's label_smoothing).
auto smoothed = F::label_smoothing_cross_entropy(logits, target, 0.1);

// Metric-learning losses.
auto contrastive = F::contrastive_loss(x1, x2, pair_labels, /*margin=*/1.0);
auto triplet = F::triplet_margin_loss(anchor, positive, negative, 1.0);
```

Notes:

- Binary inputs (logits and a `{0, 1}` float target of the same shape) take
  a sigmoid path in `focal_loss` and `dice_loss`; multiclass inputs take a
  softmax/cross-entropy path.
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
};
```

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
  save_top_k = 1)` — saves a full trainer checkpoint whenever the monitored
  metric improves; creates parent directories.
- `LRSchedulerCallback(scheduler)` — calls `scheduler->step()` at each epoch
  end (redundant with `trainer.set_scheduler`; both guard against double
  stepping the same scheduler).

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
- `AutocastGuard` is an RAII wrapper around the ATen autocast state:
  `amp::AutocastGuard guard{c10::DeviceType::CUDA, at::kHalf};`

Set `TrainerOptions::use_amp = true` (requires a CUDA device; validated at
construction) and the trainer runs forward passes under autocast, scales the
loss before backward, unscales before gradient clipping and routes optimizer
steps through the scaler. Works together with gradient accumulation and
clipping. See the AMP tutorial for the full flow.
