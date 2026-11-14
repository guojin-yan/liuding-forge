# 官方格式互通验证

此目录保存可选开发测试使用的 safetensors 和 TorchScript 独立验证工具、依赖锁与制品来源。它们不安装到 C++ SDK，也不参与产品训练、权重转换或 ONNX 导出。

| 文件 | 用途 |
| --- | --- |
| `requirements-safetensors.lock` | 固定 CPython 3.11 的 Windows amd64、Linux x86_64 官方 wheel URL 和 SHA-256 |
| `dependency-metadata.json` | wheel 的版本、兼容标签、大小、来源与包内许可证摘要 |
| `verify_safetensors_interop.py` | 官方 Python raw bytes API 与已编译 C++ 用例的双向验证 |
| `verify_torchscript_compatibility.py` | 为可信的既有 TorchScript 格式构造独立参考并验证原生兼容入口，见[TorchScript 说明](../../docs/api/torchscript_compatibility.md) |

## 准备 safetensors 测试环境

使用独立的 64 位 CPython 3.11 虚拟环境；Linux 还需满足 manylinux_2_17 的 glibc 条件。锁文件中的平台 marker 可能在不支持的平台全部跳过，因此先核对实际解释器，再检查安装的包版本，不能只看 pip 的退出码。

从仓库根目录执行，下文的 `<venv-python>` 替换为实际解释器路径：

```text
<venv-python> -c "import platform, struct, sys; assert sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8; assert (sys.platform, platform.machine()) in (('win32', 'AMD64'), ('linux', 'x86_64')); print(sys.version)"
<venv-python> -m pip --isolated install --no-deps --require-hashes -r scripts/interop/requirements-safetensors.lock
<venv-python> -c "import importlib.metadata, safetensors; assert importlib.metadata.version('safetensors') == '0.6.2'; assert callable(safetensors.serialize) and callable(safetensors.deserialize)"
<venv-python> -m pip check
```

这里只使用 safetensors 的 raw `serialize` / `deserialize`，不选择 NumPy、Torch 或测试 extras。若与 TensorBoard 共用环境，其 NumPy 依赖由遥测测试自己的[依赖锁](../../tests/reference/telemetry/requirements.lock)管理。虚拟环境放在被忽略的 `out/envs/` 或仓库外；无需修改系统 Python。

`cp38-abi3` 是 wheel 的稳定扩展 ABI 标签，包本身要求 Python ≥3.9；此测试锁收窄为 CPython 3.11。其他 Python minor、架构、PyPy、macOS 和 musl 不在锁定范围内。解释器的实际发行来源、patch 版本和 pip 版本随每次测试记录，源码元数据不保存某台机器的安装状态。

## 执行互通验证

启用可选官方互通测试并构建后，提供包含 `SafeTensorsTest.OfficialPythonBidirectionalInterchange` 的真实 C++ 测试程序：

```text
<venv-python> scripts/interop/verify_safetensors_interop.py --cpp-test-executable <absolute-cpp-test-executable> --work-dir <external-evidence-directory> --jyppx-commit <tested-source-commit>
```

驱动生成官方 Python 输入，调用 C++ 用例，再用官方 Python 解码 C++ 输出；核对 dtype、shape、原始字节和 metadata。环境安装成功不代替互通测试，也不授予 GPU 或其他模型资格。

环境探测、pip 安装报告、验收 JSON 和日志保存在被忽略的 `out/` 或仓库外。依赖元数据只保留可移植的输入身份；历史执行结果不混入锁文件或源码目录。

## 来源与许可

wheel 来自 [PyPI safetensors 0.6.2](https://pypi.org/pypi/safetensors/0.6.2/json) 返回的官方 URL，源码项目为 [huggingface/safetensors](https://github.com/huggingface/safetensors)。包内 Apache-2.0 许可证的两平台原始摘要分别保留在元数据中；单独分发 wheel 时应保留其许可证及适用告知。这些记录不替代完整传递依赖、CVE 或再分发审查。
