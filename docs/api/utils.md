# API: Utilities

## Device helpers

`#include <jyppx/liuding/core/device.hpp>`

```cpp
namespace jyppx::liuding { Device get_default_device();                 // cuda:0 when available, else cpu bool cuda_is_available(); int cuda_device_count(); Device parse_device(const std::string& s);   // "cpu", "cuda", "cuda:1" }
```

## Tensor utilities

`#include <jyppx/liuding/core/tensor_utils.hpp>`

```cpp
namespace jyppx::liuding {
bool has_nan(const Tensor& tensor);
bool has_inf(const Tensor& tensor);
bool is_finite(const Tensor& tensor);        // no NaN and no Inf
TensorList clone_tensors(const TensorList& tensors);
TensorList to_device(const TensorList& tensors, const Device& device);
}
```

`clone_tensors` produces independent copies (mutations of the copies never
affect the originals).

## Core types

`#include <jyppx/liuding/core/types.hpp>`

```cpp
namespace jyppx::liuding {
using Tensor = torch::Tensor;
using TensorList = std::vector<Tensor>;
using Device = torch::Device;
using Dtype = torch::Dtype;
template <typename T> using Optional = std::optional<T>;
}
```

## Parameter initialization

JYPPX does not wrap LibTorch's parameter initialization API. Include
`<torch/nn/init.h>` and select the complete upstream options directly:

```cpp
auto weight = torch::empty(
    {64, 32}, torch::TensorOptions().requires_grad(true));
torch::nn::init::kaiming_uniform_(
    weight, 0.0, torch::kFanIn, torch::kLeakyReLU);
```

The removed JYPPX wrapper delegated every operation to LibTorch while hiding
the typed fan mode and nonlinearity options, so it did not provide a distinct
framework contract.

## Version

`#include <jyppx/liuding/foundation/version.hpp>`

```cpp
jyppx::liuding::version();          // "0.3.0"
jyppx::liuding::version_major();    // 0 (plus version_minor(), version_patch())
```
