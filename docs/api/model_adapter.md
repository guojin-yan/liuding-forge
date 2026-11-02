# API：模型适配器

头文件：`#include <jyppx/liuding/nn/model_adapter.hpp>`。

LibTorch 模块没有统一的虚函数 `forward()` 签名。六丁炼算分别约定模型状态和调用方式：

- `ModelStateView` 共享持有 `torch::nn::Module`，提供参数、缓冲区、训练模式、设备和 dtype 操作。
- `Invocation` 通过 `c10::IValue` 携带位置参数和关键字参数。
- `ValueSchema` 和 `InvocationSchema` 在执行前后检查张量、元组、列表、字典和标量边界。
- `ModelAdapter` 将状态视图与显式 `ModelInvoker` 组合；调用签名由调用者指定。

## 单张量输入输出

常见的 `Tensor -> Tensor` 模型可以通过具体实现的共享指针或 LibTorch holder 适配：

```cpp
torch::nn::Linear linear(8, 3);

jyppx::liuding::TensorSchema input;
input.dtype = torch::kFloat32;
input.rank = 2;
input.shape = std::vector<std::optional<std::int64_t>>{std::nullopt, 8};

jyppx::liuding::TensorSchema output;
output.dtype = torch::kFloat32;
output.rank = 2;
output.shape = std::vector<std::optional<std::int64_t>>{std::nullopt, 3};

auto adapter = jyppx::liuding::nn::adapt_tensor_holder(linear, input, output);
auto result = adapter.invoke({.positional = {torch::ones({4, 8})}});
torch::Tensor predictions = result.value.toTensor();
```

原 holder 或共享指针释放后，适配器仍持有模块实现。

## 设备与 dtype 迁移

加载权重后，如需保持每个参数和缓冲区的 dtype，只传入目标设备：

```cpp
adapter.state().to(jyppx::liuding::Device("cuda:0"));
adapter.state().eval();
```

`to(device)` 递归迁移参数和缓冲区并保留各自 dtype。`to(dtype)` 和 `to(device, dtype)` 直接遵循 LibTorch C++ `Module::to` 的转换语义，递归转换所有参数和缓冲区，包括 BatchNorm 的整数计数器；调用者应明确选择需要的状态类型。严格[权重导入](model_weight_import.md)要求目标 dtype 与映射一致，后续迁移也应保留这一约定。

迁移不会改写输入或输出 schema；设备或 dtype 约束改变时，调用者需同步更新 schema，并将输入张量放到匹配的设备。状态迁移和推理、反向、优化器更新之间的互斥及必要同步由调用者负责。

## 多输入与结构化输出

其他调用签名使用带类型的 `adapt_module` 重载。回调接收具体模块和已通过 schema 检查的调用参数：

```cpp
jyppx::liuding::nn::InvocationSchema inputs; inputs.positional = {
    jyppx::liuding::ValueSchema::tensor({}, "left"),
    jyppx::liuding::ValueSchema::tensor({}, "right"),
};

auto outputs = jyppx::liuding::ValueSchema::tuple({
    jyppx::liuding::ValueSchema::tensor({}, "sum"),
    jyppx::liuding::ValueSchema::tensor({}, "difference"),
});

auto adapter = jyppx::liuding::nn::adapt_module(
    model,
    [](MyModule& module, const jyppx::liuding::nn::Invocation& call) {
      auto left = call.positional[0].toTensor();
      auto right = call.positional[1].toTensor();
      return c10::IValue(c10::ivalue::Tuple::create(
          {module.combine(left, right), left - right}));
    },
    inputs, outputs);
```

`c10::IValue` 只用于进程内传递；持久制品和远程协议需要独立版本化的 schema 与编解码器。

## 校验与错误

schema 格式错误、缺失关键字、多余字典字段，以及 dtype、设备、rank 或 shape 不匹配均抛出 `jyppx::liuding::InvalidArgument`。适配器在运行用户代码前检查输入，返回前检查输出。forward 中的 `c10::Error` 转换为 `jyppx::liuding::BackendError`，原异常保留为嵌套原因。

只有具体模块和 invoker 明确支持相应访问方式时，才能并发调用或修改状态；适配器不提供内部同步。
