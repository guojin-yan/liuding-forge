# 状态格式与后端边界

`jyppx::liuding::interop` 负责权重、状态和模型制品的格式契约。制品摘要用于确认字节身份，格式描述、能力声明和执行验证分别检查。

| 公共接口 | 用途 |
|---|---|
| `safetensors.hpp` | 有界读写 13 种按字节寻址的 tensor 类型、标量、空 tensor 和字符串元数据。 |
| `weight_import_report.hpp` | 生成 strict 或显式 partial 权重映射计划与版本化诊断。 |
| `model_weight_import.hpp` | 可选 nn 适配器，验证候选权重后应用到 CPU/CUDA 模型。 |
| `model_bundle.hpp` | 发布不可变模型包，校验完整文件、tensor 清单、路径、大小与摘要。 |
| `checkpoint_manifest.hpp`、`file_state_archive.hpp` | 版本化 participant manifest 与完整性校验、原子文件存储。 |
| `torchscript_compatibility.hpp` | 可选 nn 适配器，加载显式受信的已有 JIT 制品并执行受限推理。 |
| `export_backend.hpp` | 显式后端注册、能力描述、preflight 与既有协议元数据编解码。 |

safetensors 不包含文件校验和或设备身份，应使用独立可信的 `ArtifactRef`/摘要校验完整文件。读写器保留原始小端位表示，不创建模型、推断 forward 图或执行代码；不支持的 FP8、亚字节和复数类型会明确失败。重复字段、未知 tensor 字段、非法 UTF-8、形状溢出、重叠或有空隙的数据区、尾随数据与资源超限均会拒绝。

model bundle 默认保存数据与描述，不从包中执行任意脚本或动态库。完整清单、来源和签名要求必须分别验证；签名字段或测试 verifier 不能代替真实可信签名。strict 权重导入拒绝缺失、多余、重复、shape/dtype 不匹配的条目；partial 必须显式请求并保留映射报告。

`BackendRegistry` 需要显式注册。descriptor 只声明能力，不能证明依赖已安装或模型执行成功；缺少实际环境 probe 时为 `conditional`，超出 source/format/operation、dtype/device/options 范围时为 `unsupported`。旧 sidecar 的 request/probe/candidate 元数据契约仍可解析，候选校验要求请求、制品、环境与独立证据精确绑定；JSON 成功与协议测试不能证明当前导出能力，也不代表继续提供 Python 生产导出入口。

`onnx_export` 提供类似 `torch.onnx.export` 的通用 C++ 入口：传入用户自定义模型或 callable、example inputs、输入输出名称和动态 batch 声明，由原生 trace 前端捕获图，经算子级 registry/lowering 转为 ONNX，再交给 LibTorch 2.14.1 官方 serializer。接口支持多输入输出；新增网络只要组合已支持的标准算子，就复用同一导出器，无需逐模型手写整图。自定义或尚不支持的算子可注册 C++ lowering 扩展，缺少转换规则时返回明确的算子与失败原因。Linear、ResNet18 是首批 FP32/opset 17/自包含制品的验收模型，不构成模型白名单。

`onnx_export` 与 `onnx_runtime` 独立可选、默认 OFF。运行显式选择 execution provider，非 CPU provider 默认禁止 CPU fallback；CPU 是首批执行证据门槛，其他 provider 的可用性、算子执行与数值结果分别验收。trace 会实际执行 example forward，只覆盖捕获路径，调用方须遵守状态与副作用约束；不承诺 PyTorch 全算子或将任意 `.pt` 文件直接转换为 ONNX。构建选项、接口与限制见 [ONNX 导出与运行](onnx.md)。

ONNX 产品导出、用户 API、安装与运行路径不依赖 Python；独立 Python checker 或参考数据差分仅用于验收。仓库内存量遥测、数据准备和 Hugging Face 转换工具的 Python 产品路径仍待原生迁移，不能据此宣称全功能 C++ 闭环已经完成。

`TorchScriptCompatibilityRuntime` 使用官方 `torch::jit::load`/`forward`，加载前快照并校验精确字节，且必须显式设置 `trust_source=true`。摘要不认证可执行制品的生产者，inspect 同样需要受信加载。调用接受连续的位置参数前缀与精确关键字名，省略参数须有制品内经校验的默认值。运行复用 `nn::Invocation`/`ValueSchema`；nn 关闭时不提供该适配器。

JIT 边界只接受声明支持的 CPU/显式 `cuda:N` dense tensor、标量、固定 tuple、同类型 list、闭合字符串键 dictionary 与 Optional/None；拒绝 Any/classes、额外关键字、隐式设备回退与环境 autocast。输入、默认值和输出采用独立快照，不保留跨字段存储别名；调用串行执行，进入 forward 后失败会关闭实例。字节/遍历限制、取消与 deadline 是协作边界，不约束所有算子分配，也不能抢占 JIT/CUDA kernel。完整契约见 [TorchScript 兼容运行时](torchscript_compatibility.md)。

TorchScript 仅兼容已有制品。通用 C++ ONNX 前端的图捕获与算子转换独立验收，不把已有 JIT 制品可加载当作通用导出成功。
