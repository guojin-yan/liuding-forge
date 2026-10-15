# API: Model Adapter

Header: `#include <jyppx/nn/model_adapter.hpp>`

LibTorch modules do not share a virtual `forward()` signature. JYPPX therefore
keeps model state and model invocation as separate contracts:

- `ModelStateView` retains shared ownership of a `torch::nn::Module` and
  exposes parameters, buffers, mode, device, and dtype operations.
- `Invocation` carries positional and keyword `c10::IValue` arguments.
- `ValueSchema` and `InvocationSchema` validate tensor, tuple, list,
  dictionary, and scalar boundaries before and after execution.
- `ModelAdapter` couples a state view with an explicit `ModelInvoker`; it does
  not guess a forward signature from module inheritance.

## Tensor models

The narrow helper for the common `Tensor -> Tensor` shape accepts either a
concrete implementation pointer or a LibTorch holder:

```cpp
torch::nn::Linear linear(8, 3);

jyppx::nn::TensorSchema input;
input.dtype = torch::kFloat32;
input.rank = 2;
input.shape = std::vector<std::optional<std::int64_t>>{std::nullopt, 8};

jyppx::nn::TensorSchema output;
output.dtype = torch::kFloat32;
output.rank = 2;
output.shape = std::vector<std::optional<std::int64_t>>{std::nullopt, 3};

auto adapter = jyppx::nn::adapt_tensor_holder(linear, input, output);
auto result = adapter.invoke({.positional = {torch::ones({4, 8})}});
torch::Tensor predictions = result.value.toTensor();
```

The adapter retains the module implementation even if the original holder or
shared pointer is released.

## Multi-input and structured output

Use the typed `adapt_module` overload when the forward contract is not
`Tensor -> Tensor`. The callable receives the concrete module and the already
validated invocation:

```cpp
jyppx::nn::InvocationSchema inputs;
inputs.positional = {
    jyppx::nn::ValueSchema::tensor({}, "left"),
    jyppx::nn::ValueSchema::tensor({}, "right"),
};

auto outputs = jyppx::nn::ValueSchema::tuple({
    jyppx::nn::ValueSchema::tensor({}, "sum"),
    jyppx::nn::ValueSchema::tensor({}, "difference"),
});

auto adapter = jyppx::nn::adapt_module(
    model,
    [](MyModule& module, const jyppx::nn::Invocation& call) {
      auto left = call.positional[0].toTensor();
      auto right = call.positional[1].toTensor();
      return c10::IValue(c10::ivalue::Tuple::create(
          {module.combine(left, right), left - right}));
    },
    inputs, outputs);
```

`c10::IValue` is only the in-process carrier. Persistent artifacts and remote
protocols require separately versioned schemas and codecs.

## Validation and errors

Malformed schemas, missing keywords, unexpected dictionary fields, and
dtype/device/rank/shape mismatches raise `jyppx::InvalidArgument`. Inputs are
validated before user code runs. Results are validated before they leave the
adapter. A `c10::Error` raised during forward execution is translated to
`jyppx::BackendError` with the original exception retained as a nested cause.

The adapter does not add synchronization. Concurrent calls or state mutation
are valid only when the concrete module and invoker explicitly support them.
