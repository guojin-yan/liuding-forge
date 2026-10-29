# 命令行工具

`liuding-forge`（Windows 为 `liuding-forge.exe`）提供环境检查、配置解析、本地运行管理、训练配方、实际评估及模型/检查点校验。启用 `JYPPX_LIUDING_BUILD_TOOLS=ON` 时构建并安装到 `bin/`；它需要完整的八核心 SDK，缺少组件时明确报错，不隐式开启其他组件。关闭 tools 不影响库的使用。

用 `liuding-forge --help --format text` 查看命令，用 `liuding-forge --version --format json` 查看实际构建身份。命令接受 `--format text|json`；JSON 成功响应使用下面的统一 envelope：

```json
{"schema_version":1,"status":"ok","data":{}}
```

错误响应为 `status: "error"` 和包含稳定错误码、名称、消息、组件、操作及提示的 `error` 对象。退出码 `0` 表示成功，`1` 表示已完成但未通过/不完整的检查，`2` 表示输入或功能不支持，`3` 表示操作失败，`4` 表示训练配方已完成但质量门槛未通过，`7` 表示本地任务仍活跃或需要确认恢复，`124` 表示超时，`130` 表示取消。

## 常用命令

```sh
liuding-forge doctor --device cpu --format text
liuding-forge doctor --device cuda --expected-libtorch 2.14.1 --format json
liuding-forge config lint --schema config-schema.json --file training.json --format text
liuding-forge config render --schema config-schema.json --set /training/rate=0.01 --format json
liuding-forge run list --root local-runs --format text
liuding-forge model verify --path weights.safetensors --format json
liuding-forge model bundle verify --path model-bundle --format json
liuding-forge checkpoint verify --path checkpoint.jyck --format json
```

`doctor` 实际执行 Tensor、autograd 和序列化操作；CUDA 不可用或 LibTorch 版本不符时返回失败，不安装驱动或自动换用 CPU。配置 schema 声明字段类型、默认值、环境绑定及可脱敏字段，解析器按默认值、配置文件、环境绑定及 CLI 覆盖的顺序合并。配置与运行的完整约定见[本地运行存储](../api/local_run_storage.md)和[评估](../api/evaluation.md)。

## 训练与恢复

```sh
liuding-forge recipe schema --name paired-affine --format json
liuding-forge recipe check --file paired-affine.local.json --format json
liuding-forge recipe run --file paired-affine.local.json --root runs/example --pause-after 256 --format json
liuding-forge recipe resume --file paired-affine.local.json --root runs/example --format json
```

双输入分类/回归配方使用核心 SDK；[完整教程](../tutorials/multi_io_recipe.md)包括离线数据、配置、两类损失、评估及新进程恢复。图像分类配方额外需要 `models` 和 `data_image`，省略 `--name` 的 `recipe schema` 返回其配置 schema，详见[ResNet18 教程](../tutorials/image_classification_recipe.md)。配方命令按配置中的 `recipe` 标识选择入口，不会根据数据文件名猜测任务。

## 模型包与可选运行时

原生 C++ 的 `model onnx export` 可从训练检查点恢复模型，经通用导出与实际 ORT 数值验证后发布 bundle；已有 ONNX 使用 `model onnx pack`，包内数值复验使用 `model onnx smoke`。所需组件、输入格式和支持边界见 [ONNX 导出与推理](../api/onnx.md)。

`model bundle inspect|verify` 有界检查目录、文件大小、摘要和 safetensors 映射，分别报告内容完整性、签名信任及运行时兼容性。CLI 尚未配置密码学信任提供者：无签名报告 `unsigned`，已有签名报告 `present_unverified`，`--require-trusted true` 拒绝未受信任的包。应用可以提供库的签名验证回调；schema 的实际解释和推理仍由对应运行时负责。

启用 `JYPPX_LIUDING_INSTALL_TELEMETRY_SIDECAR=ON` 可单独安装遥测 sidecar 源码与锁定依赖声明；CMake 不安装 Python 包，调用方显式提供解释器和配置，见[进程传输](../api/telemetry_sidecar_process.md)。更完整的本地组合示例见[离线工作流](../examples/offline_workflow.md)。

## 本地任务

```sh
liuding-forge local execute --root local-runs --id trial-001 --spec local-run.json --format json
liuding-forge local show --root local-runs --id trial-001 --format json
liuding-forge local recover --root local-runs --id trial-001 --format json
liuding-forge local cancel --root local-runs --id trial-001 --format json
```

任务按冻结的 spec 直接启动可信可执行文件，原始 stdout/stderr 和独立 attempt 保存到现有 run/CAS，最多尝试 16 次；只对声明的非零退出码重试。取消、超时和输出超限回收进程树，不自动重试。`recover` 可补齐已确认结果的生命周期提交；活跃 worker 或缺少确认结果的 attempt 返回退出码 7，避免重复运行。完整 [spec 示例与进程约定](../development/local_process_runner.md)包含环境、字节和时间限制；当前不自动决定训练检查点续训，也不提供 grid/random 搜索。
