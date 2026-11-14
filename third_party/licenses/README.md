# 第三方许可原文

本目录保留从固定归档、vendored header 或准确上游文档取得的许可、版权、notice 和来源摘录。[manifest.json](manifest.json) 记录每份原文的字节数、SHA-256、来源 URL、制品摘要以及适用的归档成员或摘录字节区间。原文通过 Git `-text` 属性保留上游字节，不调整换行、措辞或原始署名；本 README、模型来源说明和 JSON 索引是项目维护的元数据。

## 范围与索引

当前保留材料覆盖 GoogleTest、nlohmann/json、SQLite、TinyXML2、yaml-cpp、stb_image/write、zlib、Ubuntu curl、OpenSSL、ONNX Runtime、原生遥测 schema/runtime、部分可选 Python wheel、PyTorch 主许可证、模型参考声明以及 NVIDIA 协议。完整文件集合及准确版本以 [manifest.json](manifest.json) 和项目根目录 `dependency-manifest.json` 为准，不以根许可证推定整个二进制或传递依赖的许可已经覆盖。

| 原文 | 来源与用途 |
| --- | --- |
| [GoogleTest](googletest-1.14.0-LICENSE) | 固定 1.14.0 源归档中的 BSD-3-Clause 原文；仅测试使用。 |
| Ubuntu curl 7.81.0-1ubuntu1.30：[MNIST runtime](curl-7.81.0-1ubuntu1.30-mnist-runtime-copyright)、[headers](curl-7.81.0-1ubuntu1.30-mnist-headers-copyright)，[OTLP runtime](curl-7.81.0-1ubuntu1.30-otlp-runtime-copyright)、[headers](curl-7.81.0-1ubuntu1.30-otlp-headers-copyright) | 精确 runtime/dev `.deb` 的完整 copyright 按四个依赖条目分别留存；相同原文不表示传递依赖或许可政策/CVE 审查已完成。不捆绑这些系统库及 SDK。 |
| [nlohmann/json](nlohmann-json-3.12.0-LICENSE.MIT) | 固定 3.12.0 源归档中的 MIT 原文。 |
| [SQLite 头部摘录](sqlite-3.54.1-copyright-blessing.txt)及[上游版权页](sqlite-copyright-page.txt) | 固定 3.54.1 `sqlite3.h` 最前 294 字节和官方 public domain / blessing 说明。 |
| [PyTorch 2.14.1](pytorch-2.14.1-build-5c488690-LICENSE.txt)及 [2.5.1](pytorch-2.5.1-LICENSE.txt) | 分别来自构建提交 `5c4886908584029761b579af026dcfb627c84070` 与 `v2.5.1` 的源码主许可证；不能替代随包组件声明。 |
| [模型来源与许可](MODEL_PROVENANCE.md) | 固定 torchvision、timm、Meta DINOv3 的许可原文、部分文件头与继承声明索引；C++ 源码来源仍待核查。 |
| [CUDA 12.6.0 EULA](nvidia-cuda-12.6.0-EULA.html)及 [cuDNN 9.10.2 EULA](nvidia-cudnn-9.10.2-EULA.html) | NVIDIA 官方版本化网页原文；供核对 Windows cu126 基线条款，不构成完整 DLL 再分发清单。 |
| [CUDA cudart/CUPTI 原包 LICENSE](nvidia-cuda-12.6-cudart-cupti-LICENSE.txt)、[cuDNN 9.10.2.21 原包 LICENSE](nvidia-cudnn-9.10.2.21-archive-LICENSE.txt)及[旧 NvToolsExt 声明](nvidia-nvtoolsext-legacy-SOURCE-NOTICE.txt) | Windows cu126 随包运行库的版本化材料；旧 NvToolsExt 使用专有声明。cuDNN 整包摘要取自官方清单，未作本地完整复核。 |
| [Intel OpenMP EULA](intel-openmp-2026.0.0-LICENSE.txt)及[第三方通知](intel-openmp-2026.0.0-third-party-programs.txt) | 官方 2026.0.0 wheel 的混合许可原文，含独立 LLVM/BSD/MIT 条款；EULA 排除按独立条款管理的第三方软件。本地 DLL 与候选包主体相同、签名封装不同，各部分授权范围仍需对应。 |
| [libuv 1.40.0](libuv-1.40.0-LICENSE)、[zlib 1.2.3 包 README](zlib-1.2.3-binary-README.txt)及[许可](zlib-1.2.3-SOURCE-NOTICE.txt) | DLL 已与相应二进制包逐字节匹配；索引还保留 libuv 外部代码和 MiniZip 1.01e 的相关原始声明。 |

## 安装与分发

根目录 `NOTICE` 和 `THIRD_PARTY_NOTICES.md` 概述项目及第三方授权边界。CMake 按启用组件安装所需原文：2.14.1/2.5.1 的 PyTorch 主许可位于 `share/doc/jyppx/licenses/`，Windows 2.14.1 cu126 同处另附 NVIDIA 协议；启用 `models` 时，模型来源说明和所列模型许可安装到 `share/doc/jyppx/licenses/models/`，其余组件沿用各自安装规则。源码包仍应保留其实际包含材料的适用声明，不因组件关闭而自动免除义务。

已检查的 Windows LibTorch 归档缺少许可/notice 文件。Windows 2.14.1 cu126 安装现在附带上述新增运行库原文；已匹配部分 DLL 的来源，但完整静态嵌入组件、Intel 各部分独立授权及需要依赖其 EULA 的分发范围和下游合同等仍未闭合，相关状态继续为 `pending`。具体适用条件见根目录 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)；当前安装后的 CMake 包仍要求调用方另行提供 Torch。

索引中 `artifact_sha256` 记录原文所在制品的身份。cuDNN 9.10.2.21 的该字段是厂商声明的整包摘要，原文通过官方归档的 HTTP Range 取得，未重算完整归档摘要；其他来源包的摘要身份保留在对应清单中。原文文件自身的长度和摘要均由本地实际字节计算。libuv 的旧 MSVC 头、MiniZip 辅助文件及 Intel 第三方通知的留存范围不用于推断每一项都已编入当前 LibTorch。

`optional-sidecar/` 保留从已核对 Werkzeug 3.1.9、protobuf 6.33.5、Pillow 13.0.0 和 setuptools 83.0.0 wheel 中抽取的 24 份原始许可/notice，包括 setuptools 的内置 vendor 声明，以及分别保留的 Pillow Windows/Linux 原文。Pillow 13.0.0 主许可为 MIT-CMU，内置组件另有条款；这些记录不表示整个可选 Python 环境及传递许可已审查完成。

`recorded` 只表示对应 notice 原文与来源记录已完成；它不表示完成漏洞审查、设备支持验证或正式发布资格认定。模型参考条目保持 `pending`，完整 DINOv3 协议的留存也不替代具体 C++ 源码是否属于衍生作品的判断。

## 校验

在项目根目录运行以下命令，同时检查依赖锁与原文索引：

```sh
python scripts/validation/dependency_manifest.py --file dependency-manifest.json
```

校验器核对准确字节、长度、摘要、可移植路径和 `recorded` 条目的原文覆盖，并拒绝重复 JSON 键及文件链接/别名。通过只说明记录与当前文件一致，不认证发布者身份，也不授予任何再分发权。
