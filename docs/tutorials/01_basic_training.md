# Tutorial 1: Basic Training

This tutorial trains a linear model on a synthetic dataset using the
`Trainer` — the smallest complete JYPPX program.

## Prerequisites

- A project configured with `find_package(jyppx_training_framework CONFIG REQUIRED)`
  and linked against `jyppx_training_framework::jyppx_training_framework`
  (see the README for the CMake setup).

## 1. Define a model

Models are ordinary `torch::nn::Module`s; the Trainer consumes them through
a type-erased forward call:

```cpp
#include <jyppx/jyppx.hpp>

class LinearModel : public torch::nn::Module {
public:
    LinearModel()
        : linear(register_module("linear", torch::nn::Linear(2, 1))) {}

    jyppx::Tensor forward(const jyppx::Tensor& input) {
        return linear->forward(input);
    }

    torch::nn::Linear linear{nullptr};
};
```

## 2. Build a dataset and loader

```cpp
torch::manual_seed(7);
auto inputs = torch::rand({32, 2});
auto targets = 2.0 * inputs.select(1, 0) - inputs.select(1, 1) + 0.5;

auto dataset = std::make_shared<jyppx::data::TensorDataset>(
    inputs, targets.view({32, 1}));

jyppx::data::DataLoader loader(dataset, {.batch_size = 8});
```

`DataLoader` deduces the sample type from the dataset, so no explicit
template argument is needed. The batch type for a `TensorDataset` exposes
`.data` and `.targets`.

## 3. Create the trainer

```cpp
auto model = std::make_shared<LinearModel>();
auto optimizer = std::make_shared<jyppx::optim::Adam>(
    model->parameters(), jyppx::optim::AdamOptions{.lr = 0.05});

jyppx::training::Trainer trainer(
    model,
    optimizer,
    jyppx::training::TrainerOptions{.max_epochs = 20});
```

## 4. Fit and read the result

```cpp
trainer.fit(loader);
std::cout << "final loss=" << trainer.last_train_loss() << '\n';
```

`fit` runs the full loop: batching, forward, loss, backward, optimizer step,
gradient handling, and the callback lifecycle. After it returns,
`last_train_loss()` holds the mean training loss of the final epoch,
`global_step()` the number of optimizer steps.

For the same run with a validation pass each epoch:

```cpp
jyppx::data::DataLoader val_loader(dataset, {.batch_size = 8});
trainer.fit(loader, &val_loader);   // pointer to the validation loader
```

`last_val_loss()` then holds the latest validation loss.

## 5. Custom objectives

The default loss is mean squared error. Replace it through the options:

```cpp
jyppx::training::TrainerOptions options;
options.max_epochs = 20;
options.loss_fn = [](const jyppx::Tensor& prediction,
                     const jyppx::Tensor& target) {
    return torch::nn::functional::cross_entropy(prediction, target);
};
```

The loss function must return a scalar tensor.

## Complete example

`examples/basic_trainer.cpp` in the repository implements exactly this flow
and prints `final loss=...`. Build it with `JYPPX_BUILD_EXAMPLES=ON`.

## What to read next

- Tutorial 2 covers datasets, splits and multi-worker loading.
- Tutorial 3 compares optimizers and learning-rate schedulers.
- Tutorial 4 adds callbacks: checkpointing, early stopping and metrics.
