# 部署指南

六丁炼算以 CMake 包交付，使用 `find_package(jyppx CONFIG REQUIRED COMPONENTS ...)` 选择所需组件，公共头文件位于 `<jyppx/liuding/...>`。按应用需要选择入口：

| 用途 | CMake 组件 | 接入说明 |
| --- | --- | --- |
| 运行 C++ 训练程序 | `training` | [训练 API](../api/training.md) |
| 从 C++ 模型导出 ONNX | `onnx_export` | [ONNX 导出与推理](../api/onnx.md) |
| 直接执行 ONNX | `onnx_runtime` | [部署示例](../../examples/deployment/README.md) |
| 模型包 Ed25519 签名与外部策略验签 | `bundle_signing` | [签名指南与 C++ 示例](bundle-signing.md) |
| native bundle 本地推理 | `serving` | [Serving API](../api/serving.md) |
| ONNX bundle 本地推理与版本切换 | `onnx_serving` | [本地服务示例](../../examples/serving/README.md) |

例如，安装包已启用 ONNX Serving 时，应用链接对应组件即可：

```cmake
find_package(jyppx CONFIG REQUIRED COMPONENTS onnx_serving)
add_executable(my_service main.cpp)
target_link_libraries(my_service PRIVATE jyppx::liuding::onnx_serving)
```

构建库时，`JYPPX_LIUDING_BUILD_ONNX_EXPORT`、`JYPPX_LIUDING_BUILD_ONNX_RUNTIME`、`JYPPX_LIUDING_BUILD_SERVING` 和 `JYPPX_LIUDING_BUILD_ONNX_SERVING` 均默认关闭。ONNX Serving bridge 要求同时开启 `SERVING` 与 `ONNX_RUNTIME`，并通过 `JYPPX_LIUDING_ONNX_RUNTIME_ROOT` 提供匹配的 ORT SDK；它不依赖导出器，仅启用 `serving` 也不引入 ORT。运行从 C++ 导出模型的 ONNX Serving 示例时，还须开启 `ONNX_EXPORT` 和 `JYPPX_LIUDING_BUILD_EXAMPLES`。

需要签名交付时，显式开启 `JYPPX_LIUDING_BUILD_BUNDLE_SIGNING` 并提供匹配的 OpenSSL SDK；该组件也默认关闭。签名发布到新目录，外部信任策略须绑定独立获得的原始 SHA-256。CLI 的 bundle 校验及 ONNX smoke/verify 可复用该策略，在加载推理模型前核对全部声明签名；C++ Serving 加载器通过相同 `ModelBundleVerifyOptions` 接入。具体配置和命令见[模型包签名](bundle-signing.md)。

交付使用 Release 构建，固定编译器、C++20 标准库、LibTorch 以及选用的 ORT/CUDA 版本。Windows 随应用分发匹配的 LibTorch/ORT DLL 和 MSVC runtime，避免加载系统旧版 ORT；Linux 通过 rpath 或加载路径使用对应共享库。已验证组合见 [`compatibility.json`](../../compatibility.json)，macOS 尚未支持，其他组合及 Provider 需单独验收，不能由 CPU 结果推导 GPU 资格。

训练程序应限制数据 worker、预取和训练 batch，将带来源信息的检查点保存到持久存储，并在目标环境验证小任务和恢复流程。`SingleDeviceStrategy` 支持单进程 CPU 或单 GPU；多进程分布式训练尚未提供。

本地服务应校验输入 schema、设置请求与快照预算、核对 bundle 和环境锁，并通过独立已知向量完成 warmup。逻辑字节限制不等于 RSS/显存硬配额，实际延迟、吞吐和峰值需在部署环境测量。HTTP/TLS、认证和服务控制由应用负责；示例仅演示本地进程集成。
