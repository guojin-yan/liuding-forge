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

## 已训练的 CIFAR-10 模型交付

[image_onnx.cpp](image_onnx.cpp) 连接已有原生训练记录、检查点重建导出、Ed25519 签名和本地 CPU Serving。它要求完整的 CIFAR-10 验证目录（10000 张图、每类 1000 张，类别顺序为 `airplane automobile bird cat deer dog frog horse ship truck`）、原训练的 `final.jyck`、`result.json` 及包含 `final-checkpoint`、`model-definition` 引用的原 `run-store`。检查点须属于十分类 ResNet18；图片处理与[分类配方](../../docs/tutorials/image_classification_recipe.md)一致：RGB float32 `[0,1]`、bilinear resize 32×32、NCHW，不添加 ImageNet 归一化。程序不会训练、下载数据或权重。

在现有兼容配置中另启用 `JYPPX_LIUDING_BUILD_DATA_IMAGE`、`JYPPX_LIUDING_BUILD_BUNDLE_SIGNING` 和 `JYPPX_LIUDING_BUILD_ONNX_SERVING`；CLI 的检查点重建导出还需 `models`、`experiment` 和相关训练组件。组件依赖及 SDK 路径按[构建指南](../../CONTRIBUTING.md#构建与验证)配置，不通过额外启用全部组件解决依赖。示例 target 默认不参与 ALL，显式构建：

```sh
cmake --build out/build/your-profile --config Release --target jyppx_cli jyppx_liuding_image_onnx --parallel 1
```

以下为 Linux 单配置命令。将数据、训练和输出绝对路径改为自己的路径；所有输出目录必须尚不存在，父目录须已存在。摘要由操作者在其可信输入上计算；下例 `sha` 仅用于本地文件身份绑定：

```sh
sha() { sha256sum "$1" | cut -d ' ' -f 1; }
example=./out/build/your-profile/examples/liuding-forge-example-image-onnx
cli=./out/build/your-profile/tools/cli/liuding-forge

"$example" prepare /data/cifar10/validation \
  /train/final.jyck "$(sha /train/final.jyck)" \
  /train/result.json "$(sha /train/result.json)" /work/image-inputs

"$cli" model onnx export --root /train/run-store \
  --source-run run.image-classification \
  --checkpoint-reference final-checkpoint --definition-reference model-definition \
  --inputs /work/image-inputs/inputs.json \
  --inputs-sha256 "$(sha /work/image-inputs/inputs.json)" \
  --tensors /work/image-inputs/inputs.safetensors \
  --tensors-sha256 "$(sha /work/image-inputs/inputs.safetensors)" \
  --metadata /work/image-inputs/delivery.json \
  --metadata-sha256 "$(sha /work/image-inputs/delivery.json)" \
  --provider CPUExecutionProvider --timeout-ms 180000 \
  --output /work/unsigned-onnx --format json

"$example" sign /work/unsigned-onnx \
  "$(sha /work/unsigned-onnx/manifest.json)" /work/signed-onnx
```

`prepare` 产生 batch 1、2、4、8、16、128 的真实图片输入和完整交付元数据；导出命令从原记录严格恢复 `model.main`，用同一原生模型产生参考输出，再在释放原生状态后运行 ORT。动态 batch 范围为 1..128，逐元素容差为 `atol=1e-5`、`rtol=1e-4`。训练结果摘要只绑定所提供文件，不能自行证明其语义；依赖描述明确是已知版本子集，不是完整传递 SBOM。

`sign` 只在进程内生成临时 Ed25519 私钥，输出签名包、`public-trust-policy.json` 和 `signing.json`。这是本地教学授权，不是生产发布密钥治理；进程退出后不能使用同一私钥重新签名。操作者应保存终端显示的 public policy 摘要及自己刚生成的包清单摘要，并通过可信配置将这两个预期值交给新的消费者进程；不能从接收到的不可信包中自动接受授权摘要。

```sh
# Set these to the digests authorized by the local producer above.
manifest_sha=AUTHORIZED_SIGNED_MANIFEST_SHA256
policy_sha=AUTHORIZED_PUBLIC_POLICY_SHA256

"$cli" model onnx smoke --bundle /work/signed-onnx/bundle \
  --bundle-sha256 "$manifest_sha" --model-id primary \
  --report-id export-report --cases-id cases --tensors-id tensors \
  --provider CPUExecutionProvider --timeout-ms 180000 \
  --trust-policy /work/signed-onnx/public-trust-policy.json \
  --trust-policy-sha256 "$policy_sha" --format json

"$example" evaluate /work/signed-onnx/bundle "$manifest_sha" \
  /work/signed-onnx/public-trust-policy.json "$policy_sha" \
  /data/cifar10/validation /work/serving-evaluation
```

`evaluate` 核验签名及独立预期摘要，在同次可信快照中读取原生参考和环境锁，解码并核对四个 owner 描述符与实际图签名，通过六种 batch 预热后逐批读取全部 10000 张图。Serving 接收预先准备好的 CPU NCHW Tensor；图片解码和 resize 在外部 C++ 数据流程完成。输出 `logits.f32`（little-endian 10000×10）、`labels.i64`（10000）、`samples.tsv` 及含混淆矩阵、摘要和准确率的 `evaluation.json`。准确率低于 0.65 返回非零，部分结果保留；成功返回 0。

本案例仅验证 CPU FP32 ORT 推理及本地 Serving，不声明其他 Provider、GPU 零复制、自动合批、OOM 拆批恢复或生产签名资格。96 MiB 快照和 I/O 限制是逻辑载荷预算，不是进程 RSS 上限。确认交付包及结果摘要后，可删除重复的展开导出包和中间图片输入，保留原训练资产与唯一交付副本；不要将运行库或编译树存入记录目录。
