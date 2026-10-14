# API: Utilities

## Device helpers

`#include <jyppx/core/device.hpp>`

```cpp
namespace jyppx {
Device get_default_device();                 // cuda:0 when available, else cpu
bool cuda_is_available();
int cuda_device_count();
Device parse_device(const std::string& s);   // "cpu", "cuda", "cuda:1"
}
```

## Tensor utilities

`#include <jyppx/core/tensor_utils.hpp>`

```cpp
namespace jyppx {
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

`#include <jyppx/core/types.hpp>`

```cpp
namespace jyppx {
using Tensor = torch::Tensor;
using TensorList = std::vector<Tensor>;
using Device = torch::Device;
using Dtype = torch::Dtype;
template <typename T> using Optional = std::optional<T>;
}
```

## Logger

`#include <jyppx/utils/logger.hpp>`

```cpp
enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class Logger {  // thread-safe singleton
public:
    static Logger& instance();
    void set_level(LogLevel level);
    void set_file(const std::string& filepath);   // appends; throws if unopenable
    void debug/info/warning/error(const std::string& message);
};

#define LOG_DEBUG(msg)  // plus LOG_INFO / LOG_WARNING / LOG_ERROR macros
```

Messages below the configured level are dropped; every accepted line is
written to `stdout` and, when configured, to the log file with a
`[timestamp] [LEVEL]` prefix.

## Parameter initialization

`#include <jyppx/nn/init.hpp>`

```cpp
namespace jyppx::nn::init {
void uniform_(Tensor& t, double a = 0.0, double b = 1.0);
void normal_(Tensor& t, double mean = 0.0, double std = 1.0);
void constant_(Tensor& t, double val);
void zeros_(Tensor& t);
void ones_(Tensor& t);
void xavier_uniform_(Tensor& t, double gain = 1.0);
void xavier_normal_(Tensor& t, double gain = 1.0);
void kaiming_uniform_(Tensor& t, double a = 0.0,
                      const std::string& mode = "fan_in");
void kaiming_normal_(Tensor& t, double a = 0.0,
                     const std::string& mode = "fan_in");
}
```

All functions delegate to `torch::nn::init` (exact PyTorch formulas, leaky
relu gain, NoGradGuard inside — safe on `requires_grad` weights). Notes for
this LibTorch version:

- `mode` is `"fan_in"` or `"fan_out"`; anything else throws.
- Fan-based initializers require tensors with at least 2 dimensions (1D
  tensors are rejected, unlike `torch.nn.init` in Python).

```cpp
auto weight = torch::empty({64, 32}, torch::TensorOptions().requires_grad(true));
jyppx::nn::init::kaiming_uniform_(weight);  // He initialization
```

## Version

`#include <jyppx/foundation/version.hpp>`

```cpp
jyppx::version();          // "0.3.0"
jyppx::version_major();    // 0 (plus version_minor(), version_patch())
```
