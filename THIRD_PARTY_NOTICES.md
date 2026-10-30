# Third-Party Notices

JYPPX Training Framework is licensed under the Apache License, Version 2.0.
Third-party software remains subject to its own license terms. A dependency's
presence here does not mean its source or binaries are redistributed with every
JYPPX package.

## Runtime Dependency

### LibTorch / PyTorch

- Project: https://github.com/pytorch/pytorch
- Role: tensor, autograd, device and runtime backend
- Upstream license: BSD-style license with separately licensed bundled
  third-party components
- The locked Windows `2.14.1+cu126` archive was inspected on 2026-10-21 and
  contains no license-like entry. Its PyTorch source `LICENSE` at build commit
  `5c4886908584029761b579af026dcfb627c84070` is retained in
  `third_party/licenses/`, but does not close the bundled-component notice
  audit. The candidate `2.5.1+cu124` installation has the same unresolved
  notice boundary. Both manifest entries remain pending.
- Ubuntu 开发构建独立锁定 Linux `2.14.1+cu126` 官方包，SHA-256 为 `78409e1d17eb439efab01bbdde385ef3531ac3fc651ceda1117317ae9903dc7c`。该摘要经过本地完整归档复核，不是厂商另行发布的摘要；二进制附带组件的 notice 审查仍为 pending，开发构建结果不代表 GA 资格。
- Distribution rule: when a JYPPX artifact redistributes LibTorch binaries,
  it must also redistribute the exact license and notice files supplied with
  that LibTorch build. This summary does not replace those files.

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
- The vendored header is pinned by SHA-256 in `dependency-manifest.json`; its
  upstream license block is retained in
  `third_party/licenses/stb_image-2.30-LICENSE.txt` and installed with the
  data-image profile. No stb_image types enter public headers.
- Decoder limits are enforced before third-party allocation. HDR/PIC and other
  formats that need a different scalar contract remain outside this profile;
  callers receive a structured decode error rather than an implicit conversion.

### OpenSSL

- Project: https://openssl-library.org/; version: 3.5.9; license: Apache-2.0.
- Role: private static EVP Ed25519 backend for the optional `jyppx::liuding::bundle_signing` component. Source archive integrity is pinned in `dependency-manifest.json`; original terms are retained in `third_party/licenses/openssl-3.5.9-LICENSE.txt` and installed with the enabled component. Public headers expose no OpenSSL types. Retaining the license does not establish a complete vulnerability audit or FIPS qualification.

## Optional Python Sidecar

The optional sidecar runtime is separately deployed and is not installed by the
C++ SDK. Its exact wheel allowlist is
`tools/telemetry_sidecar/requirements.lock`. The 2026-10-21 security remediation
pins Werkzeug 3.1.9 (BSD-3-Clause), protobuf 6.33.5 (BSD-3-Clause), Pillow
13.0.0 (HPND and separately licensed bundled components), and setuptools 83.0.0
(MIT and separately licensed bundled vendors). Exact original license/notice
entries for the selected Windows/Linux and universal wheels are retained
under `third_party/licenses/optional-sidecar/`, with artifact and file digests
in `third_party/licenses/manifest.json`. Preserve the full originals when
redistributing those wheels; this list does not establish a complete transitive
license review of the entire optional Python runtime.

## Reference Projects

Projects stored outside the Git repository under the local `refer/` directory
are research inputs, not JYPPX dependencies or redistributed source. Their
licenses do not change the JYPPX project license. Any future source reuse,
linking or binary redistribution requires a separate provenance and license
review before integration.


## MNIST 可选 HTTPS 下载

默认关闭的 `JYPPX_LIUDING_DOWNLOAD_MNIST` 仅为产品工具增加显式 HTTPS 下载，离线准备及安装后的公共库不依赖网络 SDK。Windows 通过现有 Windows SDK 链接系统 WinHTTP，使用 Windows 系统证书验证；WinHTTP 与 SDK 适用 Microsoft 的相应条款，本项目不复制或分发 winhttp.dll。

Linux 使用调用者显式选择的共享系统 libcurl >= 7.81.0 SDK，适用 [curl license](https://curl.se/docs/copyright.html)，上游为 <https://github.com/curl/curl>。SDK 使用的 OpenSSL、GnuTLS、解析器和其他传递依赖分别适用各自条款；本项目不下载、嵌入或安装这些库及 CA 证书。如果后续制品要捆绑任何对应二进制，须先留存所选构建的完整许可证、版权和第三方声明并完成依赖审查，不能用系统链接的说明代替再分发义务。

工具声明位于 [network-dependencies.json](tools/datasets/network-dependencies.json)；Linux 配置记录所选 curlver.h 与解析后共享库的 SHA-256。该可选系统 SDK 接口不代表任意发行版、TLS 后端或版本已经取得发布资格。


### 可选原生 TensorBoard

启用 `JYPPX_LIUDING_BUILD_TELEMETRY_NATIVE` 时，使用 nanopb 0.4.9.2（Zlib 许可证）的官方运行库和 TensorBoard 2.20.0（Apache-2.0）的官方 schema 生成代码。运行库的符号使用项目私有前缀，原始依赖源码不修改。安装包附带对应许可证；nanopb 生成器、protoc 和独立 Python reader 仅用于开发，不属于产品运行时依赖。

### 原生 OTLP 协议

同一可选组件使用 OpenTelemetry proto 1.7.0 的官方 schema 生成私有描述符（Apache-2.0；Copyright 2019, OpenTelemetry Authors；Copyright 2020, OpenTelemetry Authors），复用 nanopb 0.4.9.2 运行库；不链接 OpenTelemetry C++ SDK 或 Python SDK。完整许可证位于 `third_party/licenses/opentelemetry-proto-v1.7.0-LICENSE`，各 schema 与开发期生成工具的来源摘要见 `scripts/telemetry/native_schema.lock.json`。离线协议准备不包含 HTTP/TLS 传输。依赖与原始许可证摘要均纳入正式清单；许可证留存不表示已完成依赖安全或发布审计。
