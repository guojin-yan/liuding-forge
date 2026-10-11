# Tutorial 4: Callbacks, Checkpoints and Metrics

Callbacks hook into the training lifecycle without touching the loop. This
tutorial assembles the standard production setup: progress reporting,
checkpointing, early stopping and metric tracking.

## The callback lifecycle

`jyppx::training::Callback` fires these hooks in order for each epoch:

```
fit_start
  epoch_start
    train_epoch_start
      train_batch_start / train_batch_end   (per batch)
    train_epoch_end
    validation_epoch_start                  (when a val loader is given)
      validation_batch_start / validation_batch_end
    validation_epoch_end
  epoch_end
fit_end
```

Derive from `Callback` and override what you need — the trainer pointer
gives access to `global_step()`, `last_train_loss()`, `last_val_loss()` and
`request_stop()`.

## Built-in callbacks

```cpp
#include <jyppx/jyppx.hpp>

using namespace jyppx::training;

auto checkpoint = std::make_shared<callbacks::ModelCheckpoint>(
    "runs/linear/best.pt",   // filepath (directories are created)
    "val_loss",              // monitored metric
    "min",                   // smaller is better
    /*save_top_k=*/1);

auto early_stopping = std::make_shared<callbacks::EarlyStopping>(
    "val_loss", /*patience=*/5, /*min_delta=*/1e-4);

trainer.add_callback(std::make_shared<callbacks::ProgressCallback>());
trainer.add_callback(checkpoint);
trainer.add_callback(early_stopping);
```

- `ModelCheckpoint` writes a full trainer checkpoint whenever the monitored
  metric improves — resume later with `trainer.load_checkpoint(path)`.
- `EarlyStopping` calls `trainer->request_stop()` once `patience` epochs
  pass without an improvement of at least `min_delta`; `fit` then returns
  after the current epoch. Both callbacks expose `reset()` for reuse.

Both monitor `"val_loss"` (default) or `"train_loss"`; they skip epochs
without a validation pass for `"val_loss"`.

## Tracking metrics

Metrics accumulate across `update` calls and report through `compute`:

```cpp
#include <jyppx/training/metrics/accuracy.hpp>
#include <jyppx/training/metrics/metric_collection.hpp>

jyppx::training::metrics::MetricCollection metrics;
metrics.add(std::make_shared<jyppx::training::metrics::Accuracy>());
metrics.add(std::make_shared<jyppx::training::metrics::TopKAccuracy>(3));
```

Drive them from a callback so training and validation are measured the same
way:

```cpp
class MetricCallback : public jyppx::training::Callback {
public:
    explicit MetricCallback(jyppx::training::metrics::MetricCollection& metrics)
        : metrics_(metrics) {}

    void on_validation_epoch_start(jyppx::training::Trainer*) override {
        metrics_.reset();
    }
    void on_validation_batch_end(jyppx::training::Trainer* trainer, int) override {
        // Re-run the forward pass in eval mode, or cache predictions in
        // on_validation_batch_start from a custom Trainer subclass.
        metrics_.update(cached_logits_, cached_targets_);
    }
    double last_score() const { return metrics_.compute().at("accuracy"); }

private:
    jyppx::training::metrics::MetricCollection& metrics_;
    jyppx::Tensor cached_logits_, cached_targets_;
};
```

`AverageMeter` is the lightweight alternative for running scalars (e.g. the
per-epoch mean loss):

```cpp
jyppx::training::metrics::AverageMeter meter;
meter.update(loss_value, /*count=*/batch_size);
meter.average();
```

## Putting it together

```cpp
jyppx::training::Trainer trainer(model, optimizer,
                                 {.max_epochs = 100, .gradient_clip_val = 1.0});
trainer.fit(train_loader, &val_loader);

if (early_stopping->should_stop()) {
    std::cout << "stopped early at epoch " << trainer.current_epoch() << '\n';
}
trainer.load_checkpoint("runs/linear/best.pt");   // restore the best model
```

## What to read next

- Tutorial 5 enables mixed precision on CUDA.
