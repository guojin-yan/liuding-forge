# Third-Party Notices

本项目贡献者拥有并有权授权的贡献采用 Apache-2.0。第三方依赖、上游代码和其他材料保留各自适用条款，本声明不将其重新许可为 Apache-2.0。下列摘要不能代替许可原文；依赖在此列出不表示每个 SDK 都包含其源码或二进制。清单中的 `pending` 表示对应审查尚未完成。上游名称仅用于说明来源与兼容性，不表示关联、认证或背书。

## Runtime Dependency

### LibTorch / PyTorch

上游为 <https://github.com/pytorch/pytorch>，提供 Tensor、autograd、设备和运行后端。PyTorch 主体使用 BSD 风格许可，随包组件分别适用其自身条款。

锁定的 Windows `2.14.1+cu126` ZIP 未包含许可证或 notice 文件；已保留其构建提交 `5c4886908584029761b579af026dcfb627c84070` 的 PyTorch 源码 `LICENSE`。候选 `2.5.1+cu124` 同样尚未完成随包组件审查；对应依赖状态继续为 `pending`。

该 Windows cu126 包的本地完整 ZIP 摘要与锁定值一致，安装目录的 37 个 DLL 也与 ZIP 对应成员逐一一致。进一步核对确认 `uv.dll` 来自 libuv 1.40.0 `h8ffe710_0`，`zlibwapi.dll` 来自 `zlib123dllx64.zip`，`cudart64_12.dll` 来自 CUDA Runtime 12.6.77，`cupti64_2024.3.2.dll` 与 `nvperf_host.dll` 来自 CUPTI 12.6.80，旧 `nvToolsExt64_1.dll` 来自 PyTorch 构建使用的 `NvToolsExt.7z`；这些 DLL 均与相应来源包的成员逐字节一致。

下表链接使用源码树路径；对应 SDK 中已安装原文的位置见本节后面的安装说明。

| 随包组件 | 许可与保留原文 |
| --- | --- |
| libuv 1.40.0 | [主 MIT 原文](third_party/licenses/libuv-1.40.0-LICENSE)，另保留 [tree.h BSD 声明](third_party/licenses/libuv-1.40.0-tree-SOURCE-NOTICE.txt)、[inet.c ISC 声明](third_party/licenses/libuv-1.40.0-inet-SOURCE-NOTICE.txt)及 [旧 MSVC 头声明](third_party/licenses/libuv-1.40.0-stdint-msvc2008-SOURCE-NOTICE.txt)。旧头声明的留存不表示该头已编入当前 DLL。 |
| zlib 1.2.3 / MiniZip 1.01e | [二进制包 README](third_party/licenses/zlib-1.2.3-binary-README.txt)、[Zlib 许可](third_party/licenses/zlib-1.2.3-SOURCE-NOTICE.txt)，以及 MiniZip 的 [zip](third_party/licenses/minizip-1.01e-zip-SOURCE-NOTICE.txt)、[unzip](third_party/licenses/minizip-1.01e-unzip-SOURCE-NOTICE.txt)、[ioapi](third_party/licenses/minizip-1.01e-ioapi-SOURCE-NOTICE.txt)、[crypt](third_party/licenses/minizip-1.01e-crypt-SOURCE-NOTICE.txt)原始声明。原生数据工具的 zlib 1.3.2 记录不能替代此处旧 DLL 的来源。 |
| CUDA / CUPTI | `LicenseRef-NVIDIA-CUDA`：[已匹配 cudart/CUPTI 包的完整 LICENSE](third_party/licenses/nvidia-cuda-12.6-cudart-cupti-LICENSE.txt)，包含协议、补充和第三方声明；另保留 [CUDA 12.6.0 官方协议网页](third_party/licenses/nvidia-cuda-12.6.0-EULA.html)。 |
| cuDNN 9.10.2.21 | `LicenseRef-NVIDIA-cuDNN`：[版本化归档 LICENSE](third_party/licenses/nvidia-cudnn-9.10.2.21-archive-LICENSE.txt)及 [9.10.2 官方协议网页](third_party/licenses/nvidia-cudnn-9.10.2-EULA.html)。 |
| 旧 NvToolsExt | `LicenseRef-NVIDIA-NvToolsExt-Legacy`：[该旧包的 NVIDIA 专有声明](third_party/licenses/nvidia-nvtoolsext-legacy-SOURCE-NOTICE.txt)，指向相应 NVIDIA 协议；不能套用新版 NVTX3 的开源许可。 |
| Intel OpenMP | 官方 2026.0.0 wheel 的[根 EULA](third_party/licenses/intel-openmp-2026.0.0-LICENSE.txt)（`LicenseRef-Intel-Developer-Tools`）与[完整第三方通知](third_party/licenses/intel-openmp-2026.0.0-third-party-programs.txt)。通知包括 LLVM OpenMP 的 `Apache-2.0 WITH LLVM-exception`、BSD/MIT 组件及 oneTBB/Thread Composability Manager 的 `LicenseRef-Intel-Simplified-Software-2022` 原文；准确适用范围见下文。 |

本地两个 Intel OpenMP DLL 的版本资源为 `20260213`，有效签名属于 Intel；其全部 PE 节和去除签名封装差异后的文件主体，与官方 2026.0.0 和 2026.1.0 wheel 对应 DLL 一致，但完整文件摘要不一致，无法唯一确定其原始分发包版本。两份官方 wheel 的根 EULA 相同；保留这些参考原文不等于已经确认具体 DLL 的全部再分发资格。

Intel 包内是混合条款，不能将整个 OpenMP 笼统归为专有 EULA 或纯 LLVM 许可。EULA 的 Materials 定义排除 Third Party Software，第 2.2 条明确这些软件按其独立条款管理；第三方通知中的 LLVM OpenMP 等开源授权不能被根 EULA 无差别覆盖。当前尚未将本地 DLL 的各组成部分与这些授权逐项对应，通知列出的全部组件也不表示每一项都实际编入当前 DLL。

对实际由 Developer Tools EULA 管理的 Materials，该协议允许有条件的内部个人或企业开发使用；分发须限于已授权的 Redistributables，作为自有产品的一部分，二进制下游协议须禁止对这些 Intel 材料逆向、反编译和反汇编，并传递相应责任限制。两个 wheel 中未找到该定义所指的 `redist.txt`、`redist-rt.txt` 或同类文件清单；需要依赖该 EULA 的分发范围和合同条件仍未闭合，不能只附 EULA 就视为完成。

Intel EULA 第 3.1(xi) 条限制将其 Materials 直接或间接用于特定 SaaS 或 service bureau 服务；这不自动给适用独立开源许可的部分新增同类限制。对外提供托管训练或推理服务时，应先确认实际使用部分的授权，再结合服务架构判断，不能笼统推导“整个 OpenMP 禁止 SaaS”或“根 EULA 的所有限制均不适用”。

CUDA/cuDNN 并非一概禁止商业使用，但受各自 NVIDIA 协议的用途、平台和再分发条件限制。cuDNN 补充条款允许符合协议的应用分发 runtime `.dll` / `.so`，其开发用途限 NVIDIA GPU 系统；CUDA 需按相应可再分发清单核对实际文件。尚未确认 `nvperf_host.dll` 的具体分发授权范围，不能仅因它出现在开发运行目录就认定可随包转发。旧 NvToolsExt 的专有声明也不能替代对适用 NVIDIA 协议的核对。

cuDNN 版本化 LICENSE 从官方 9.10.2.21 归档通过 HTTP Range 取得；索引中的整个归档摘要来自 NVIDIA 官方版本清单，本次未下载并重算该完整归档，也未逐个与该厂商包核对 cuDNN DLL 的 SHA-256。这一证据边界与上述已完整匹配的 DLL 不同。当前声明和依赖清单只列已识别条款，并非可任选其一或已穷尽 LibTorch 静态嵌入组件及全部第三方 notice，相关依赖继续为 `pending`。

Ubuntu 开发构建独立锁定 Linux `2.14.1+cu126` 官方包，SHA-256 为 `78409e1d17eb439efab01bbdde385ef3531ac3fc651ceda1117317ae9903dc7c`。该摘要经过本地完整归档复核，不是厂商另行发布的摘要；二进制附带组件的 notice 审查仍为 `pending`，开发构建结果不代表 GA 资格。

Ubuntu CPU 开发 profile 另锁定 `2.14.1+cpu` 原始 ZIP，127853266 字节、SHA-256 `0f02329a5266153cffa9dc12926ff2221fa626207b894d40bd61905cc2079821`；它与 CPU `2.5.1` 候选分别记录。其构建提交 `5c4886908584029761b579af026dcfb627c84070` 的 [PyTorch 主许可原文](third_party/licenses/pytorch-2.14.1-cpu-build-5c488690-LICENSE.txt) 来自上游源码文档；该留存不表示 ZIP 内含相同许可文件，随包第三方组件、完整 notice 与再分发审查继续为 `pending`。

CPU 包实际附带的第三方头文件按原 ZIP 成员摘要与构建提交的固定子模块版本登记在 [CPU 组件来源清单](third_party/licenses/libtorch-cpu-2.14.1/components.json)，对应完整原文位于 [libtorch-cpu-2.14.1](third_party/licenses/libtorch-cpu-2.14.1/)。其中 protobuf、pybind11、FBGEMM、TensorPipe、Kineto 和包内 QNNPACK 头使用 BSD-3-Clause；cpuinfo、其内置 clog、NNPACK、pthreadpool 使用 BSD-2-Clause；FP16、FXdiv、psimd、fmt 使用 MIT；SLEEF 使用 BSL-1.0。它们保留各自版权与署名，不能统一改称 PyTorch 主体 BSD 许可，也不表示这些上游组件都已编入本框架 SDK。该清单只补充既有外部 LibTorch 包的许可来源，不启用或恢复框架已延期的量化功能。

oneDNN 的实际头文件对应 PyTorch 固定 iDeep 子模块进一步固定的 oneDNN 提交 `80afa71049cd69a3df32adcccb623b12cd7baa22`；保留其 [Apache-2.0 原文](third_party/licenses/libtorch-cpu-2.14.1/oneDNN-LICENSE.txt)和[完整第三方声明](third_party/licenses/libtorch-cpu-2.14.1/oneDNN-THIRD-PARTY-PROGRAMS.txt)，后者的各条款分别管理相应材料。ITT API 上游提供 BSD-3-Clause 或 GPL-2.0-only 双重许可，两个原文均保留；这不把 GPL 条款赋给全部 PyTorch 或无关的 SDK 代码。

同一 CPU 原包的 `libgomp.so.1` 已按原始代码、数据及加载语义对应到 AlmaLinux `8.5.0-28.el8_10.alma.1`，精确源码 RPM、发行版补丁及官方 PyTorch `$ORIGIN` RPATH 修改材料见 [CPU 组件来源清单](third_party/licenses/libtorch-cpu-2.14.1/components.json)。保留其 [GPLv3 原文](third_party/licenses/libtorch-cpu-2.14.1/gcc-8.5.0-COPYING3.txt)与 [GCC Runtime Library Exception 3.1](third_party/licenses/libtorch-cpu-2.14.1/gcc-8.5.0-COPYING.RUNTIME.txt)；例外不免除分发运行库本身的适用义务。原 ELF 完整字节并不相同，实际打包执行与下游完整对应源码交付仍待完成，不能只给源码链接就视为已履行。

该 CPU 包的 `libtorch_cpu.so` 内嵌版本函数已确认 oneMKL `2024.2`、Product Build `20240605`；这不等于已确定全部静态组成或原始 Intel 分发包。完整第三方告知、传递依赖、漏洞及再分发审查仍为 `pending`，上述来源识别不授予整个 CPU 包或正式发布资格。

当前安装后的 CMake 包通过 `find_dependency(Torch REQUIRED)` 要求调用方提供外部 LibTorch。SDK 安装对 2.14.1 和 2.5.1 保留对应 PyTorch 源码主许可证；Windows 2.14.1 cu126 配置还附带上述已留存的随包运行库原文，位置为 `share/doc/jyppx/licenses/`。这些文件的留存不关闭随包组件审查，也不代替需要实际履行的下游合同条件。

构建脚本会将 Torch `lib/` 下全部 DLL 暂存到开发运行目录；这个操作不构成再分发资格确认。若交付制品包含 LibTorch 二进制，必须按实际交付文件逐项满足相应许可和再分发条件，提供对应构建的完整版权、许可与 notice；不能直接以开发运行目录或这份摘要替代审核。

### ONNX Runtime（可选）

`jyppx::liuding::onnx_runtime` 使用 Microsoft 官方 ONNX Runtime 1.23.2 CPU SDK，许可证为 MIT；公共头文件不暴露 ORT 类型。原生 `onnx_export` 使用 LibTorch 自带的官方 C++ ONNX 序列化器，不引入 Python 或独立 protobuf SDK。上游项目为 <https://github.com/microsoft/onnxruntime>。

- Windows x64 官方归档：`onnxruntime-win-x64-1.23.2.zip`，SHA-256 `0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd`。
- Linux x64 官方归档：`onnxruntime-linux-x64-1.23.2.tgz`，SHA-256 `1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b`。

两个归档均已完整复核摘要，并逐字节保留各自的 `LICENSE` 与 `ThirdPartyNotices.txt`，文件位于 `third_party/licenses/onnxruntime-1.23.2-{windows,linux}-x64-*`，来源、文件摘要和长度记录在许可证清单中。分发相应 ORT 二进制时必须同时保留该构建的完整许可与第三方声明。上述摘要来自本地完整归档复核；许可文件留存不代表依赖安全审查完成。

当前下载锁只选择 CPU SDK。Provider 查询或显式选择接口不代表 GPU 执行资格；CUDA、TensorRT、OpenVINO 等需要各自匹配的官方 SDK、依赖和实际执行验证。请求的 Provider 缺失或尚无接入实现时会明确失败；CPU 分区回退由调用者显式决定。

### SQLite

- Project: https://sqlite.org/
- Role: embedded local run metadata, transactions, and CAS coordination inside
  `jyppx::liuding::experiment`
- Version selected by the current build: 3.54.1
- Source: https://sqlite.org/2026/sqlite-amalgamation-3540100.zip
- Upstream status: public domain (https://sqlite.org/copyright.html)
- The pinned amalgamation archive contains no license file; the upstream
  copyright page and the exact copyright/blessing excerpt from pinned
  `sqlite3.h` are retained in `third_party/licenses/` with their audit digests.
- Distribution rule: the amalgamation is compiled into the experiment static
  library; public JYPPX headers do not expose SQLite types. The upstream source
  disclaims copyright and provides a blessing rather than a license requirement.
  Preserve this provenance and the pinned digest in source distributions.

## Test Dependency

### GoogleTest

- Project: https://github.com/google/googletest
- Role: build-time unit and integration testing only
- Version selected by the current build: 1.14.0
- Upstream license: BSD 3-Clause
- The exact `googletest-1.14.0/LICENSE` entry is retained in
  `third_party/licenses/` and the dependency manifest marks its notice audit
  recorded. GoogleTest is test-only.
- Distribution rule: GoogleTest is fetched for tests and is not installed as
  part of the JYPPX SDK. Any future redistribution must include its upstream
  copyright notice, license conditions and disclaimer.

## Build Dependency

### JSON for Modern C++ (nlohmann/json)

- Project: https://github.com/nlohmann/json
- Role: strict JSON parsing and serialization inside interop, experiment and the
  optional `jyppx::liuding::data_image` annotation readers
- Version selected by the current build: 3.12.0
- Upstream license: MIT
- Distribution rule: the header is compiled into the experiment config static
  library and is not exposed by JYPPX public headers. Source or binary
  redistribution must retain the upstream copyright notice and MIT license.
  SDK installations with interop, experiment or data_image include the original notice at
  `share/doc/jyppx/licenses/nlohmann-json-3.12.0-LICENSE.MIT`. Experiment SDK
  installations also retain the original SQLite copyright/blessing provenance
  in that directory.

### TinyXML2

- Project: https://github.com/leethomason/tinyxml2
- Version: 11.0.0; license: Zlib
- Role: private Pascal VOC XML parsing in optional `jyppx::liuding::data_image`
- The pinned source archive and the exact original `LICENSE.txt` are recorded in
  the dependency and retained-license manifests. The original is installed at
  `share/doc/jyppx/licenses/tinyxml2-11.0.0-LICENSE.txt` with this profile.
- Preserve the upstream copyright and license text in redistributions. Parser
  objects are embedded in the image static library; public headers expose no
  TinyXML2 types. Release security review is separate from this notice record.

### yaml-cpp

- Project: https://github.com/jbeder/yaml-cpp
- Version: 0.9.0; license: MIT
- Role: private data-only YOLO dataset YAML parsing in optional `jyppx::liuding::data_image`
- The pinned source archive and exact original `LICENSE` are recorded in the
  dependency and retained-license manifests. The original is installed at
  `share/doc/jyppx/licenses/yaml-cpp-0.9.0-LICENSE.txt` with this profile.
- Preserve the original copyright notice and MIT license in source and binary
  redistributions. No YAML parser types enter public headers. JYPPX rejects
  aliases, anchors and explicit tags and never executes download instructions;
  release security review remains separate from this notice record.

### stb_image

- Project: https://github.com/nothings/stb
- Version: 2.30 (2024-05-31); license: MIT or Unlicense (public domain dedication)
- Role: private JPEG, PNG, BMP, GIF, TGA, PNM and related still-image decoding
  inside optional `jyppx::liuding::data_image`
本地版本 `2.30+liuding.1` 基于不可变上游提交 `013ac3beddff3dbffafd5177e7972067cd2b5083`，修正 JPEG marker 失败传播并拒绝缺少 scan 的图像；修改范围和上游原件摘要见[来源说明](third_party/stb/README.md)，当前文件摘要见 `dependency-manifest.json`。许可及署名保持，原文保留在 `third_party/licenses/stb_image-2.30-LICENSE.txt`，启用 data-image profile 时随 SDK 安装；公共头不暴露 stb_image 类型。
- Decoder limits are enforced before third-party allocation. HDR/PIC and other
  formats that need a different scalar contract remain outside this profile;
  callers receive a structured decode error rather than an implicit conversion.

### stb_image_write（图像编码）

可选 `jyppx::liuding::data_image` 的 PNG/JPEG 输出私有编译 stb_image_write 1.16；上游为 <https://github.com/nothings/stb>，可选择 MIT 或 Unlicense。该 writer 独立于 stb_image decoder 登记，公共头不暴露其类型。已入库头的路径、长度和 SHA-256 记录在 `dependency-manifest.json`；来源已固定到不可变上游提交 `1ee679ca2ef753a528db5ba6801e1067b40481b8`，对应上游 header 与入库文件逐字节一致。

头文件内完整的双选许可、版权与免责声明原样保留在 `third_party/licenses/stb_image_write-1.16-LICENSE.txt`，启用 data-image profile 时安装到 `share/doc/jyppx/licenses/stb_image_write-1.16-LICENSE.txt`。分发对应源代码或二进制时保留完整原文；许可留存不代表安全或正式发布资格。

### zlib（原生数据准备工具）

可选原生数据准备工具使用显式选择的 zlib 1.3.2 私有静态 codec 子集（`Z_SOLO`），适用 Zlib 许可证，上游为 <https://github.com/madler/zlib>；它不是 Core 默认运行依赖。精确源归档身份记录在 `dependency-manifest.json`。

归档成员 `zlib-1.3.2/LICENSE` 的完整原文原样保留在 `third_party/licenses/zlib-1.3.2-LICENSE.txt`。启用相应工具时，构建规则核验调用者显式 SDK 的同一原文并安装为 `share/doc/jyppx/licenses/zlib-1.3.2-LICENSE.txt`；再分发对应库和工具时保留原文。notice 记录完成不代表 codec 子集的漏洞适用性、实际编译/链接、传递依赖或正式发布资格已完成审查。

### OpenSSL

- Project: https://openssl-library.org/; version: 3.5.9; license: Apache-2.0.
- Role: private static EVP Ed25519 backend for the optional `jyppx::liuding::bundle_signing` component. Source archive integrity is pinned in `dependency-manifest.json`; original terms are retained in `third_party/licenses/openssl-3.5.9-LICENSE.txt` and installed with the enabled component. Public headers expose no OpenSSL types. Retaining the license does not establish a complete vulnerability audit or FIPS qualification.

## Telemetry test references

遥测官方参考仅用于维护测试，不由 C++ SDK 安装；精确 wheel 锁位于 `tests/reference/telemetry/requirements.lock`。旧 Python 产品 sidecar 已退役，产品导出使用 C++ worker。Werkzeug 3.1.9 与 protobuf 6.33.5 主许可为 BSD-3-Clause，Pillow 13.0.0 主许可为 MIT-CMU，setuptools 83.0.0 主许可为 MIT；Pillow 和 setuptools 的随包组件另外适用各自条款。

同一锁定环境还包括 TensorBoard 2.20.0、opentelemetry-proto 1.37.0、absl-py 2.3.1、grpcio 1.75.1、tensorboard-data-server 0.7.2（主许可 Apache-2.0），Markdown 3.9、NumPy 2.3.3、MarkupSafe 3.0.2（主许可 BSD-3-Clause），packaging 25.0（Apache-2.0 或 BSD-2-Clause）及 typing-extensions 4.15.0（PSF-2.0）。根许可摘要不覆盖每个平台 wheel 的全部内置库；例如 NumPy 随包数值库、GCC runtime 例外和 Pillow 图像/字体库条款需要结合实际制品核对，不能据某个内置许可文本推导整个框架的许可。

选定 Werkzeug、protobuf、Pillow 和 setuptools Windows/Linux 或通用 wheel 的原始许可与 notice 已保存在 `third_party/licenses/optional-sidecar/`，其制品和文件摘要见 `third_party/licenses/manifest.json`。分发这些 wheel 时应保留完整原文；其余依赖及整个 Python 运行环境的传递许可审查尚未完成，不能以这份主许可列表代替。

## 模型参考与源码来源

模型参考脚本使用 torchvision 0.24.1 和 0.29.1（BSD-3-Clause）、固定 timm 提交 `af97f99efef62bd9bd7eaa908fffe5b7a2d98e2c`（根许可 Apache-2.0），以及固定 Meta DINOv3 提交 `6876159a11b4df116f30f667f8c9888617df0751`（自定义 DINOv3 License）。timm 中 ConvNeXt、Swin 等文件另保留 Meta、Microsoft 等原始来源声明；根许可不统一覆盖所有模型权重。

锁定 timm 源码中的 ConvNeXt、Swin 文件明确保留 MIT 声明，依赖清单因此同时列出 Apache-2.0 与 MIT。两个 torchvision 源归档还各含 `test/assets/damaged_jpeg/TensorFlow-LICENSE`，保留 TensorFlow Authors 2019 的 Apache-2.0 告知，原文件分别存为 [0.24.1 原文](third_party/licenses/torchvision-0.24.1-TensorFlow-SOURCE-NOTICE.txt)和 [0.29.1 原文](third_party/licenses/torchvision-0.29.1-TensorFlow-SOURCE-NOTICE.txt)，[Apache-2.0 完整协议](LICENSE)继续保留。该附加条款来自源归档的测试图像资产，不能解释为所有 torchvision 模型或权重统一改为 Apache-2.0；这两份参考归档告知在源树中留存，测试图像不属于当前 SDK 安装集合。上述补充只修正已观察到的归档许可身份，四个模型参考的源码来源和分发审查仍为 pending。

对应许可原文、部分源文件头和架构参考关系见源码树中的 [模型来源与许可说明](third_party/licenses/MODEL_PROVENANCE.md)。启用 `models` 的 SDK 将该说明和相关原文安装到 `share/doc/jyppx/licenses/models/`。版权及署名包括 torchvision 的 Soumith Chintala、timm 的 Ross Wightman、ConvNeXt/DINOv3 的 Meta Platforms, Inc. and affiliates，以及 Swin Transformer 的 Microsoft；完整措辞以保留的原文为准。

DINOv3 协议覆盖模型代码、权重及其他 DINO Materials，不仅限于预训练权重；使用、复制、修改和分发须遵守其协议。已确认参考测试执行官方源码，C++ ViT 和 DINOv3 ConvNeXt 后续按官方源码作过兼容适配；当前实现的完全独立来源尚未证实，是否复制、翻译或实质改写受保护表达仍需具体判断。分发适用材料或其衍生作品时须按 DINOv3 协议提供并附带协议原文，不能宣称已确认只适用 Apache-2.0。四个模型参考条目继续为 `pending`，具体边界见上述模型来源说明；关闭编译选项或随机参考测试通过均不能替代来源判定。

## Reference Projects

Git 仓库外 `refer/` 目录中的项目是研究输入，不因存放于本机就成为随本仓库分发的依赖；是否产生许可义务取决于实际使用或复用内容。已识别的参考项目包括 Ultralytics（AGPL-3.0）、YoloSharp/SwapFighter（Apache-2.0）及 ggml（MIT）。实际复制、翻译、修改、链接或再分发时，应记录具体来源、适用版本和许可，并完成相应审查；不能据“仅作参考”推定复用内容已获宽松许可。

## 数据集与模型权重

框架代码、模型实现、具体权重和数据集分别核对权利来源。Fashion-MNIST 的 MIT 原文随数据准备配方保留；MNIST、CIFAR-10、Penn-Fudan 等材料仍须按各自发布方的实际授权和使用方式审查，下载 URL 或 SHA-256 只证明来源与文件身份，不授予数据使用或再分发权。

当前仓库不分发预训练权重，Penn-Fudan 的图片、掩码及训练权重也在仓库外。后续交付预训练、微调、量化或转换后的权重及模型包时，必须保留具体制品所适用的授权和声明；转换为 safetensors、ONNX 或本项目格式不会自动改变原许可。模型注册、随机数值对齐或权重摘要登记不表示已经取得权重分发或商业用途资格。


## MNIST 可选 HTTPS 下载

默认关闭的 `JYPPX_LIUDING_DOWNLOAD_MNIST` 仅为产品工具增加显式 HTTPS 下载，离线准备及安装后的公共库不依赖网络 SDK。Windows 通过现有 Windows SDK 链接系统 WinHTTP，使用 Windows 系统证书验证；WinHTTP 与 SDK 适用 Microsoft 的相应条款，本项目不复制或分发 winhttp.dll。

Linux 使用调用者显式选择的共享系统 libcurl >= 7.81.0 SDK，适用 [curl license](https://curl.se/docs/copyright.html)，上游为 <https://github.com/curl/curl>。SDK 使用的 OpenSSL、GnuTLS、解析器和其他传递依赖分别适用各自条款；本项目不下载、嵌入或安装这些库及 CA 证书。如果后续制品要捆绑任何对应二进制，须先留存所选构建的完整许可证、版权和第三方声明并完成依赖审查，不能用系统链接的说明代替再分发义务。

已保留锁定 Ubuntu `7.81.0-1ubuntu1.30` 包内的 [runtime copyright](third_party/licenses/curl-7.81.0-1ubuntu1.30-mnist-runtime-copyright) 和 [headers copyright](third_party/licenses/curl-7.81.0-1ubuntu1.30-mnist-headers-copyright) 完整原文；对应 `.deb`、内部成员、长度及摘要见许可索引。这只完成这两个包的 notice 留存，不授予其他 SDK、传递依赖、安全或再分发资格。

工具声明位于 [network-dependencies.json](tools/datasets/network-dependencies.json)；Linux 配置记录所选 curlver.h 与解析后共享库的 SHA-256。该可选系统 SDK 接口不代表任意发行版、TLS 后端或版本已经取得发布资格。


### 可选原生 TensorBoard

启用 `JYPPX_LIUDING_BUILD_TELEMETRY_NATIVE` 时，使用 nanopb 0.4.9.2（Zlib 许可证）的官方运行库和 TensorBoard 2.20.0（Apache-2.0）的官方 schema 生成代码。运行库的符号使用项目私有前缀，原始依赖源码不修改。安装包附带对应许可证；nanopb 生成器、protoc 和独立 Python reader 仅用于开发，不属于产品运行时依赖。

### 原生 OTLP 协议

同一可选组件使用 OpenTelemetry proto 1.7.0 的官方 schema 生成私有描述符（Apache-2.0；Copyright 2019, OpenTelemetry Authors；Copyright 2020, OpenTelemetry Authors），复用 nanopb 0.4.9.2 运行库；不链接 OpenTelemetry C++ SDK 或 Python SDK。完整许可证位于 `third_party/licenses/opentelemetry-proto-v1.7.0-LICENSE`，各 schema 与开发期生成工具的来源摘要见 `scripts/telemetry/native_schema.lock.json`。离线协议准备不包含 HTTP/TLS 传输。依赖与原始许可证摘要均纳入正式清单；许可证留存不表示已完成依赖安全或发布审计。

### 可选原生 OTLP HTTP

独立可选 `telemetry_http` 使用系统 WinHTTP（Windows）或显式选择的共享系统 libcurl >= 7.81（Linux），不复制或分发网络库、TLS 库及 CA 证书。Linux curl 适用 [curl license](https://curl.se/docs/copyright.html)，其 TLS、DNS 和其他依赖各自适用对应条款；Windows 系统 SDK/runtime 适用 Microsoft 相应条款。安装消费者也需显式选择自己的系统 curl SDK。该系统链接策略不代表已完成所选发行版包的 CVE、许可证或发布资格审查。

锁定 Ubuntu `7.81.0-1ubuntu1.30` 的 [runtime copyright](third_party/licenses/curl-7.81.0-1ubuntu1.30-otlp-runtime-copyright) 和 [headers copyright](third_party/licenses/curl-7.81.0-1ubuntu1.30-otlp-headers-copyright) 按本组件单独登记，均来自精确 `.deb` 内部完整原文。notice 留存不代替 curl 许可政策、传递依赖或安全审计；其他选择仍需匹配实际包的证据。

当前 Linux OTLP 发送路径显式设置 `CURLAUTH_NONE`、忽略 netrc、关闭代理和重定向，每次请求独立创建并释放 easy/multi 句柄，不提供凭据输入或连接共享接口。对 [CVE-2026-19931](https://curl.se/docs/CVE-2026-19931.html) 的审查须限定这个调用路径；无认证和请求隔离不能推导所选系统 libcurl、MNIST 下载路径或其他调用者已完成安全资格。
