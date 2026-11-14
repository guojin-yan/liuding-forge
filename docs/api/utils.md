# 设备、Tensor 与基础类型

## 设备查询

包含 `<jyppx/liuding/core/device.hpp>`，使用 `jyppx::liuding` 命名空间：

```cpp
Device get_default_device();
bool cuda_is_available();
int cuda_device_count();
Device parse_device(const std::string& value);
```

默认设备优先选择可用 CUDA，否则选择 CPU。`parse_device()` 接受 LibTorch 设备语法，如 `cpu`、`cuda`、`cuda:1`；输入长度为1–64字节且不能包含 NUL，非法语法报 `InvalidArgument`。解析成功不代表目标设备当前可用。

## Tensor 检查与复制

包含 `<jyppx/liuding/core/tensor_utils.hpp>`：

```cpp
bool has_nan(const Tensor& tensor);
bool has_inf(const Tensor& tensor);
bool is_finite(const Tensor& tensor);
TensorList clone_tensors(const TensorList& tensors);
TensorList to_device(const TensorList& tensors, const Device& device);
```

未定义 Tensor 的 `has_nan/has_inf/is_finite` 均返回 false。`clone_tensors()` 为各已定义项创建独立存储并保留未定义项；`to_device()` 保留列表次序和未定义项，迁移失败保留后端异常。数值扫描和迁移可能触发设备同步，不能当作零成本操作。

## 类型与初始化

`<jyppx/liuding/core/types.hpp>` 提供 Tensor、TensorList、Device、Dtype 与 Optional 等别名。参数初始化直接使用 LibTorch 完整接口：

```cpp
#include <torch/nn/init.h>

auto weight = torch::empty({64, 32}, torch::TensorOptions().requires_grad(true));
torch::nn::init::kaiming_uniform_(weight, 0.0, torch::kFanIn, torch::kLeakyReLU);
```

`<jyppx/liuding/foundation/version.hpp>` 的 `version()`、`version_major/minor/patch()` 查询实际构建的项目版本，不表示已发布版本或 ABI 兼容资格。
