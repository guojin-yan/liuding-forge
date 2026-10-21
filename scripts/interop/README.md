# 官方 safetensors Python 互通环境

此目录仅锁定可选测试依赖 `safetensors==0.6.2`，供 C++ 与官方 Python
raw `serialize` / `deserialize` API 双向验证使用。目标为 **64 位 CPython 3.11**
的 Windows amd64 与 Ubuntu x86_64（glibc ≥ 2.17）。两个平台分别固定官方 PyPI
wheel URL 和 SHA-256，不选择 NumPy、PyTorch 或测试 extras。与 telemetry 共用的
验证环境因 TensorBoard 包含 NumPy；safetensors raw API 本身不依赖它。

`requirements-safetensors.lock` 是安装入口；`dependency-metadata.json` 记录下载来源、
wheel 大小、哈希、兼容标签、许可证和验证状态。
[`optional-runtime-provenance.json`](optional-runtime-provenance.json) 另行记录两平台
实际解释器来源、精确版本、pip 安装报告摘要、所选 wheel URL/哈希及 Collector。
这些依赖不进入核心构建或安装的 C++ SDK。

2026-10-21 已从官方 PyPI 查询元数据，下载两个 wheel 到
`out/deps/safetensors-0.6.2/`，核对实际大小与 SHA-256，并读取 wheel 内的
`METADATA`、`WHEEL` 和 Apache-2.0 `LICENSE`。随后已在两个独立环境完成哈希锁安装、
版本/导入/raw API 存在性检查及 `pip check`，均无依赖冲突：

| 平台 | 解释器来源及版本 | 可复用环境解释器（仓库根目录相对路径） |
| --- | --- | --- |
| Windows amd64 | Python Software Foundation 官方 NuGet `python` 3.11.9 | `out/envs/telemetry-py311/Scripts/python.exe` |
| Ubuntu x86_64 | Astral python-build-standalone CPython 3.11.17，release `20261020` | `out/envs/telemetry-py311/bin/python` |

两端 pip 均固定为 25.2，safetensors 均为 0.6.2。解释器以已核对摘要的便携制品解压，
没有替换系统 Python、修改全局 PATH 或调用 WindowsApps 商店别名。Ubuntu 制品由
python-build-standalone 项目发布，不是 PSF 发布的 Linux 二进制。
完整环境与安装报告保存在 `out/deps/optional-runtime-provenance/`。
**这次依赖准备没有编译 C++、执行双向互通或正式章节测试**；这些结果仍由统一章节验收记录。

## 锁定的制品

| 平台 | 官方 wheel | SHA-256 |
| --- | --- | --- |
| Windows amd64 | `safetensors-0.6.2-cp38-abi3-win_amd64.whl` | `c7b214870df923cbc1593c3faee16bec59ea462758699bd3fee399d00aac072c` |
| Ubuntu x86_64 | `safetensors-0.6.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `8045db2c872db8f4cbe3faa0495932d89c38c899c603f21e9b6486951a5ecb8f` |

`cp38-abi3` 是稳定扩展 ABI 标签，可用于 CPython 3.11；包自身声明
`Requires-Python: >=3.9`。锁文件主动收窄到 3.11，其他 Python minor、架构、PyPy、
macOS、musl/Alpine 均不在本锁的验证范围。实际解释器 patch 和 pip 版本需随验收记录。

两个 wheel 的基础依赖都为空；`Requires-Dist` 中的 NumPy、Torch 等条目全部由
extras 条件启用。驱动只使用 raw bytes API 和 Python 标准库，无需选择 `[numpy]`、
`[torch]`、`[testing]` 或 `[all]`。单独安装本锁时可使用 `--no-deps`，避免扩展范围。

## 准备解释器与安装

从代码仓库根目录执行。当前两端已准备上述环境，可直接使用其中的解释器。
首次重建时，先按 provenance 核对并解压相应的便携解释器，再创建被 Git 忽略的
`out/envs/telemetry-py311/`；已有环境时无需重复创建。

**先检查解释器再安装**：PEP 508 marker 在不支持的平台可能全部跳过，pip 仍返回 0；
这个退出码不能当作安装成功。

Windows PowerShell：

```powershell
# 仅首次创建环境时执行：
# & ./out/deps/cpython-3.11.9/nuget/tools/python.exe -m venv out/envs/telemetry-py311
$safetensorsPython = Join-Path (Get-Location) 'out/envs/telemetry-py311/Scripts/python.exe'
& $safetensorsPython -c "import platform, struct, sys; assert sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8 and sys.platform == 'win32' and platform.machine() == 'AMD64'; print(sys.version); print(platform.platform())"
if ($LASTEXITCODE -ne 0) { throw 'Unsupported safetensors test interpreter' }
& $safetensorsPython -m pip --version
& $safetensorsPython -m pip --isolated install --no-deps --require-hashes -r scripts/interop/requirements-safetensors.lock
if ($LASTEXITCODE -ne 0) { throw 'Locked safetensors installation failed' }
```

Ubuntu bash：

```bash
# 仅首次创建环境时执行：
# out/deps/cpython-3.11.17+20261020/python/bin/python3.11 -m venv out/envs/telemetry-py311
safetensors_python="$PWD/out/envs/telemetry-py311/bin/python"
"$safetensors_python" -c "import platform, struct, sys; assert sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8 and sys.platform == 'linux' and platform.machine() == 'x86_64'; print(sys.version); print(platform.platform()); print(platform.libc_ver())" || exit 1
"$safetensors_python" -m pip --version
"$safetensors_python" -m pip --isolated install --no-deps --require-hashes -r scripts/interop/requirements-safetensors.lock || exit 1
```

Ubuntu 的 glibc/兼容标签还由 pip 的 wheel 兼容性检查把关；无法接受 manylinux_2_17
wheel 时应停止，不回退到源码、Rust 编译或其他平台 wheel。锁内只有两个直接 URL，
`--no-index` 不会让直接 URL 变成离线安装；若使用离线分发，应另行保留上述哈希检查。

可选预取命令（只下载当前受支持平台的 wheel，不执行包代码）：

```text
<venv-python> -m pip --isolated download --no-deps --require-hashes --dest out/deps/safetensors-0.6.2 -r scripts/interop/requirements-safetensors.lock
```

## 版本探测与整章验证

安装后记录以下探测输出。它检查实际解释器、包版本与 raw API 是否存在，不替代互通测试。

```text
<venv-python> -c "import importlib.metadata, platform, sys, safetensors; assert importlib.metadata.version('safetensors') == '0.6.2'; assert callable(safetensors.serialize) and callable(safetensors.deserialize); print(sys.version); print(platform.platform()); print(importlib.metadata.version('safetensors'))"
<venv-python> -m pip --version
<venv-python> -m pip check
```

章节集成时，由既有增量构建生成定义了
`JYPPX_SAFETENSORS_PYTHON_INTEROP=1` 的专用 C++ GTest executable。它必须包含
`SafeTensorsTest.OfficialPythonBidirectionalInterchange`。常规单测不要因启用此宏而
直接发现并执行该分支；该用例的输入与环境由 Python 驱动设置。

完成 `scripts/interop/verify_safetensors_interop.py` 的正式集成并提供实际 executable 后执行：

```text
<venv-python> scripts/interop/verify_safetensors_interop.py --cpp-test-executable <absolute-cpp-test-executable> --work-dir <chapter-evidence-directory> --jyppx-commit <tested-source-commit>
```

驱动会生成官方 Python 输入，调用已编译的指定 C++ 用例，再由官方 Python 解码
C++ 输出；检查 13 种 dtype、标量、空 tensor、shape、原始字节与 metadata。
它拒绝错误包版本、缺失用例和非零退出码，并在运行前移除自己的旧 C++ 输出，避免旧文件
被误当作成功。保存生成的结构化证据与日志，分别记录 Windows 和 Ubuntu 的结果；
Linux 的 CUDA 主构建验收另按章节进行，此 Python raw-bytes 检查本身不验证 CUDA。

本目录的依赖准备只验证安装、版本和 API 存在性。它不触发编译、完整测试、GPU 检查、
ONNX 环境安装或自动包下载；两平台正式双向结果须另行补入章节验收证据。

## 许可证与来源

官方来源为 [PyPI 0.6.2 release metadata](https://pypi.org/pypi/safetensors/0.6.2/json)
和其返回的 `files.pythonhosted.org` wheel；项目源代码在
[huggingface/safetensors](https://github.com/huggingface/safetensors)。

PyPI 的 `license` 与 `license_expression` 字段为空，但其 Apache 分类与两个实际 wheel
内的 `safetensors-0.6.2.dist-info/licenses/LICENSE` 均指向 Apache-2.0。两个 LICENSE
的原始字节大小和哈希因换行不同，已分别记录，不能用一个平台的 LICENSE 哈希代替另一个。
单独再分发 wheel 时保留原包许可证与适用 notices；本记录不代表已完成原生传递依赖的
完整许可证、CVE 或 SBOM 审计。
