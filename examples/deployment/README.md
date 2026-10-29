# 从原生检查点到 ONNX 推理

这个离线 C++ 示例演示如何保存模型参数与 buffer、在新模型实例中严格恢复，再用通用 `export_module()` 导出双输入、双输出和动态 batch，最后通过 ONNX Runtime 核对结果。源码见 [checkpoint_onnx.cpp](checkpoint_onnx.cpp)，完整接口与支持边界见 [ONNX API](../../docs/api/onnx.md)。运行时不下载文件，也不调用 Python。

## 构建与运行

先按[构建指南](../../CONTRIBUTING.md#构建与验证)准备匹配的 LibTorch、编译器和 ONNX Runtime SDK，在兼容的现有 CMake 配置中启用 `JYPPX_LIUDING_BUILD_EXAMPLES`、`JYPPX_LIUDING_BUILD_NN`、`JYPPX_LIUDING_BUILD_INTEROP`、`JYPPX_LIUDING_BUILD_ONNX_EXPORT` 与 `JYPPX_LIUDING_BUILD_ONNX_RUNTIME`，并设置 `JYPPX_LIUDING_ONNX_RUNTIME_ROOT`。不需要启用 `models` 组件。

```sh
cmake --build out/build/your-profile --config Release --target jyppx_liuding_deployment_checkpoint --parallel 2
```

使用构建后的 `liuding-forge-example-checkpoint-onnx`，将产物写入仓库外的全新绝对目录；其父目录须已存在。Windows 示例：

```powershell
$output = Join-Path $env:TEMP ('liuding-deployment-' + [guid]::NewGuid().ToString('N'))
& ./out/build/your-profile/examples/Release/liuding-forge-example-checkpoint-onnx.exe --output $output
```

Linux 单配置构建的示例：

```sh
./out/build/your-profile/examples/liuding-forge-example-checkpoint-onnx --output /tmp/liuding-deployment-example-001
```

Windows 运行目录需包含当前 SDK 的 LibTorch DLL 与 ONNX Runtime DLL；源码构建的示例 target 会复制所需运行库。Linux 应确保动态加载器找到该配置使用的对应共享库。`--help` 查看命令；成功返回 0，参数错误返回 2，导出、恢复、文件或数值校验失败返回 1。程序拒绝覆盖已有目录，执行中失败可能保留新目录中的部分产物。

## 学习流程

模型把两路三维特征作为 `left`、`right` 输入，输出校正特征 `calibrated = relu(left * gain + right + offset)` 和差值 `difference = left - right`。`gain` 是注册参数，`offset` 是注册 buffer；示例使用固定演示状态，便于以独立标量公式核对，不涉及训练质量或预训练权重。

1. 用 `ModelCheckpointParticipant` 描述模型状态，在 `FileStateArchive` 事务中保存 `checkpoint.jyck`，随后销毁原模型。
2. 构造相同 C++ 模型拓扑，以 `exact_only` 恢复。检查点保存状态，不保存或推断 C++ `forward`。
3. 设置 `eval()`，将双输入调用封装为返回 `std::vector<at::Tensor>` 的 lambda，传给通用 `export_module()`。以 batch 2 trace，并为四个 I/O 的第零维使用同一个动态符号 `batch`。
4. 释放原生模型和检查点快照，显式创建 `CPUExecutionProvider` Session。输入以名称绑定，示例故意交换提交顺序。
5. 对 batch 1、2、5 的两个输出逐元素比较独立标量参考，并读取同一个 Session 的实际 ORT trace，确认执行过 CPU kernel。

输出目录包含 `checkpoint.jyck`、`model.onnx`、`export-report.json`、`verification-report.json`、`runtime-profile.json`，以及检查点写入器保留的锁文件。成功时终端显示 `All outputs match the independent scalar oracle.`，数值报告中的 `verification.numerical` 为 `passed`。

要接入自己的模型，替换模型类和样例数据，保持原生构造拓扑与保存时一致，并用独立参考输出替换标量公式。示例仅验证上述 CPU FP32 工作流；动态声明不替代全部 batch、数据或控制流验证，其他 Provider 需单独选择与验证。设置的模型、检查点和 I/O 字节限制不是进程内存上限，数值相符也不等于任务准确率、吞吐或显存资格。

在安装后的独立应用中使用同样代码时，链接这四个组件：

```cmake
find_package(jyppx CONFIG REQUIRED COMPONENTS nn interop onnx_export onnx_runtime)
target_link_libraries(my_app PRIVATE
    jyppx::liuding::nn jyppx::liuding::interop
    jyppx::liuding::onnx_export jyppx::liuding::onnx_runtime)
```
