# TorchScript 兼容运行时

公共头文件为 `<jyppx/liuding/interop/torchscript_compatibility.hpp>`。`TorchScriptCompatibilityRuntime` 通过 LibTorch JIT 加载并执行明确受信的既有归档；该接口属于 deprecated 兼容支持，需要 interop 与 nn，不提供 tracing/export 或插件发现。通用导出使用 [ONNX](onnx.md)。

```cpp
#include <jyppx/liuding/interop/torchscript_compatibility.hpp>

namespace forge = jyppx::liuding;
namespace interop = forge::interop;

forge::nn::InvocationSchema input{
    .positional = {forge::ValueSchema::tensor(), forge::ValueSchema::tensor()},
    .keyword = {{"scale", forge::ValueSchema{
        .kind = forge::ValueKind::floating, .required = false}}}};
interop::TorchScriptLoadOptions options;
options.trust_source = true;
options.device = forge::Device(c10::kCPU);

auto runtime = interop::TorchScriptCompatibilityRuntime::load(
    archive_bytes, trusted_sha256, input,
    forge::ValueSchema::tuple({forge::ValueSchema::tensor(),
                              forge::ValueSchema::tensor()}), options);
auto result = runtime.invoke({.positional = {x, y}, .keyword = {}});
runtime.close();
```

## 加载与调用

库先复制完整输入字节，再对同一私有快照计算摘要并交给 JIT stream loader，不重新打开路径。`trust_source` 默认 false，检查制品同样要求受信加载；SHA-256 只确认字节，生产者信任由调用方决定。快照期间保持输入静止，制品必须与所链接 LibTorch 兼容。

位置 schema 是实际 forward 参数的必需前缀，其余参数使用真实 JIT 名称声明 keyword。只有归档确有合法默认值且 schema 的 required=false 才允许省略；默认值加载时验证并复制，每次调用再独立快照。未知/缺失参数、隐式 int-to-float 或设备回退均拒绝。

| 输入/输出 | 支持边界 |
| --- | --- |
| 设备 | 无索引 CPU 或实际可用的显式 cuda:N；拒绝隐式 CUDA、MPS、meta 和 fallback |
| Tensor | 输入须 contiguous dense strided；输出可为稠密视图并复制为独立连续存储；支持 bool、uint8、int8/16/32/64、float16、bfloat16、float32/64 |
| 标量 | 精确 int、float、bool、str、None |
| 容器 | 固定 unnamed tuple、同类型 List[T]、封闭 schema 的 Dict[str,T]、显式 nullable 的 Optional[T] |
| 不支持 | sparse/quantized/complex/conj/negative view、Any/custom class、named tuple、future、可变参数和 keyword-only JIT 方法 |

边界类型可表示不代表所有 JIT 算子均支持对应 dtype/device。运行设备的环境 autocast 必须关闭；内核不支持时直接失败，不自动转换。

## 状态与资源

输入及可变容器在执行前复制，不保留字段间共享存储关系；输出为无 autograd 的独立推理快照，可在 close 后使用。脚本可修改私有模型状态，调用方须保证图和宿主算子不启动未跟踪的后台修改；串行调用不构成沙箱。

同一 runtime 串行执行。`close()` 等待当前调用并释放模型和默认值，可重复调用；move/析构需独占所有权。forward 前拒绝保持 ready，进入 forward 后的异常、结果不匹配、资源拒绝或取消会关闭实例，不回滚脚本的部分私有修改。后端诊断保留为嵌套原因。

stop/deadline 在锁等待和后端工作前后协作检查，不能抢占归档解析、forward、拷贝或 kernel。CUDA 成功返回包括设备同步。limits 约束归档字节、已知模型参数/缓冲、输入输出快照与遍历元数据；状态在加载后计数，输出在 forward 后计数，不能限制任意算子 workspace、解压或进程 RSS。
