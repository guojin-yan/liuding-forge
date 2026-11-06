# 准备与验证 SDK 发布候选

六丁炼算（Liuding Forge）的候选包使用 `liuding-forge-<version>-<platform>.zip`；CPack 也使用 `liuding-forge` 前缀。安装头文件、`find_package(jyppx)`、`jyppx::liuding::` 目标和发布/SBOM 协议标识保持稳定。

离线维护工具 `scripts/release/prepare_release.py` 从明确的安装前缀准备候选，`verify_release.py` 独立核验声明、摘要、证据绑定及解包内容。二者不进入产品安装，不构建或运行 SDK、不下载依赖、不扫描 CVE，也不生成项目密钥、签名或发布资产。

当前依赖清单仍为 **draft**，来源完整性和第三方告知审查尚未全部完成，会阻断 GA。分发特定 LibTorch 构建时，必须核对它实际随附的许可、告知和传递依赖，不能只依赖项目的告知摘要。

## 输入与源码身份

先执行 `cmake --install`，并在打包期间独占安装前缀、输入文档及输出父目录，保持输入静止。提供精确版本、可移植的平台/构建标签、完整的 40/64 位小写源码提交、明确的 clean/dirty 状态和固定的 `SOURCE_DATE_EPOCH`。这些是发布者声明；工具不会用另一份当前工作树推断二进制身份。

安装的 `include/jyppx/liuding/foundation/build_config.hpp` 必须与基础语义版本一致，不能把 0.3.0 SDK 重标为 1.0.0。下文以 0.3.0 SDK 的 `0.3.0-rc.1` 为例；平台标签须替换为实际环境，不能据示例推断支持资格。RC 标签会阻断 GA。

生成的 in-toto Statement v1 / SLSA provenance v1 描述安装前缀打包操作。原始构建来源、工具链、开关、LibTorch/ABI/CRT/CUDA 事实和验收结果须另附真实证据。

先独立生成安装清单：

```sh
python scripts/release/prepare_release.py inventory \
  --prefix /work/install --output /work/install-inventory.json
```

返回值包含规范化清单字节的 SHA-256。清单按路径排序，记录相对路径、大小、SHA-256、可移植的 0644/0755 模式和总字节数。将该摘要写入证据索引，并单独计算原始 `dependency-manifest.json` 字节的摘要；格式变化也会改变绑定。

## 证据索引

schema 1 索引是 UTF-8 JSON，结构如下：

```json
{
  "schema_version": 1,
  "release": {
    "version": "0.3.0-rc.1",
    "platform": "linux-x86_64-cuda-cu126-cxx11-abi1",
    "source": {"commit": "FULL_SOURCE_COMMIT", "dirty": false}
  },
  "installation_inventory_sha256": "EXACT_INVENTORY_SHA256",
  "dependency_lock_sha256": "EXACT_LOCK_SHA256",
  "checks": [{
    "id": "windows",
    "status": "pending",
    "details": "等待该精确源码和依赖锁的实际验证",
    "artifacts": []
  }],
  "dependency_audit": {
    "status": "pending",
    "audited_dependency_ids": [],
    "blocking_licenses": [],
    "blocking_vulnerabilities": [],
    "artifacts": []
  }
}
```

执行前替换全部提交/摘要占位符；`release` 必须与准备参数完全一致。安装清单摘要可为 `null`，但会新增 GA 待办；非空时必须匹配。

每个证据引用必须恰含 `path`、`size`、`sha256`，例如 `{"path":"logs/ctest.log","size":123,"sha256":"64位小写十六进制摘要"}`。路径相对索引所在目录。准备工具读取、核摘要并将原文件纳入候选。`passed` 检查至少需要一份实际文件；重复检查、未知字段、冲突引用、缺文件和摘要不符均被拒绝。

第三方源码伴随包可作为 `dependency_audit.artifacts` 的证据附入候选，必须核验实际整包和包内来源清单的摘要，并随候选保留原件。[CPU 组件来源清单](../../third_party/licenses/libtorch-cpu-2.14.1/components.json)的 `source_materials_bundle` 登记已组装 libgomp 材料包的精确身份；当前尚未建立公开分发机制，不能因此将依赖审计改为 `passed`，完整 Corresponding Source、实际打包执行、CVE 和再分发资格仍须分别完成。

检查状态只能为 `passed`、`pending`、`failed`。缺失或未通过的必需检查会阻断 GA：

| 检查 ID | 所需证据 |
| --- | --- |
| `windows` | 精确 Windows 配置、构建、测试及运行库/工具链 |
| `ubuntu-cuda` | 精确 Ubuntu 构建、测试和真实 CUDA 设备执行 |
| `compatibility-matrix` | 全部声明支持的环境组合 |
| `install-consumer` | 干净安装消费及可选组件 ON/OFF 消费 |
| `sanitizers` | 支持的 sanitizer 配置及已审查的排除项 |
| `fuzz` | 当前解析器、编解码器及输入边界的模糊测试 |
| `static-analysis` | 源码、秘密信息及静态分析检查 |
| `coverage-risk-review` | 当前覆盖率分母、结果及风险决定 |
| `performance-memory-build-size` | 性能、资源、构建和包大小预算审查 |
| `documentation-api-examples` | API、示例、迁移和故障排查审查 |
| `upgrade-rollback` | Beta/RC 升级与回退演练 |
| `release-blockers` | 没有未解决的 P0/P1 发布阻断 |
| `offline-rebuild` | 干净离线重建的等价证据 |

依赖审计必须为 `passed`，枚举清单中每个精确依赖 ID、提供实际审计材料，并且没有未解决的 `blocking_licenses` 或 `blocking_vulnerabilities`。每项依赖必须具有已核来源摘要、精确 URL 和告知审查。

发布许可策略接受 `Apache-2.0`、`BSD-2-Clause`、`BSD-3-Clause`、`MIT`、`ISC`、`Zlib`、`BSL-1.0`、`CC0-1.0`、`Unlicense`、`blessing` 和 `curl`；其他或自定义许可须单独审查。`curl` 要求保留完整版权及许可声明，禁止未经许可使用权利人名称宣传；实际包、其他文件及传递依赖仍须分别核验。接受许可标识不会关闭告知、安全或签名门槛。

发行版回补版本须绑定实际包摘要、完整发行版安全记录、源码补丁及适用配置。上游基础版本号或 `not-affected` 标签不能单独证明安全；缺少源码、配置或调用路径依据的例外继续待审。空漏洞阻断列表是发布者审计声明，工具不会自动扫描 CVE 或认证日志真实性。

## 准备候选

```sh
python scripts/release/prepare_release.py prepare \
  --prefix /work/install --out /work/candidate \
  --version 0.3.0-rc.1 --platform linux-x86_64-cuda-cu126-cxx11-abi1 \
  --source-commit FULL_SOURCE_COMMIT --source-state clean \
  --source-date-epoch 1750000000 \
  --evidence-index /work/evidence/index.json \
  --dependency-lock dependency-manifest.json \
  --license LICENSE --notices THIRD_PARTY_NOTICES.md
```

输出目录必须尚不存在，且不能位于安装前缀内部。工具独占新目录并拒绝覆盖；中断后遗留的部分输出须独立验证，不能直接视为完整候选。

输出包括：

- `liuding-forge-VERSION-PLATFORM.zip`：`payload/` 下排序、确定性的无压缩 ZIP/ZIP64 普通文件，无链接、注释、墙钟时间戳或隐式依赖下载。
- `inventory.json`、`dependency-manifest.json`、`LICENSE`、`THIRD_PARTY_NOTICES.md`：精确内容清单和依赖/许可声明。
- `evidence-index.json` 与 `evidence/`：索引及按实际大小/摘要复制的证据。
- `sbom.cdx.json`：CycloneDX 1.5 声明依赖清单，保留完整性/告知的待审状态；传递依赖图的完整性须有独立审计。
- `provenance.intoto.json`、`release-manifest.json`、`signing-payload.json`：身份、打包与证据绑定及全部未签名 GA 阻断。签名载荷绑定除自身和外置签名以外的所有候选文件。

工具生成的 JSON 使用排序键、紧凑 UTF-8 和末尾 LF；该 LF 属于签名字节。原依赖与证据 JSON 字节保持不变。相同输入重复准备会得到相同输出字节，但不据此声明原 SDK 构建可复现。

## 独立验证与外部签名

```sh
python scripts/release/verify_release.py --directory /work/candidate \
  --expected-version 0.3.0-rc.1 \
  --expected-source-commit FULL_SOURCE_COMMIT \
  --expected-platform linux-x86_64-cuda-cu126-cxx11-abi1
```

验证核对预期身份、完整文件集合、全部元数据/证据摘要、依赖绑定、重新生成的 SBOM/provenance，以及私有临时目录中解出的实际字节和版本头。压缩、加密、非普通或未知/重复条目、路径穿越、Windows 设备/数据流名、大小写别名、ZIP 头不一致、数据重叠/空洞、注释、首尾附加字节和大小/摘要/CRC 错误均被拒绝。解出的内容不会执行。

候选审查通过后，由获授权的发布者使用已有受保护发布密钥和批准的外部 OpenSSL，对 `signing-payload.json` 精确字节签名。RSA/ECDSA 密钥的 `dgst -sha256` 示例：

```sh
/approved/bin/openssl dgst -sha256 \
  -sign /protected/release-private.pem \
  -out /work/signatures/candidate.sig /work/candidate/signing-payload.json
```

下面的 GA 验证适用于另行准备、基础版本匹配的正式 `1.0.0` 候选；给 RC 签名不会改变版本或授予 GA。签名和公钥放在候选目录外，公钥信任及文件摘要须经独立可信渠道取得：

```sh
python scripts/release/verify_release.py --directory /work/final-candidate \
  --expected-version 1.0.0 --expected-source-commit FULL_SOURCE_COMMIT \
  --expected-platform linux-x86_64-cuda-cu126-cxx11-abi1 \
  --signature /work/signatures/final-candidate.sig \
  --trusted-public-key /trusted/project-release-public.pem \
  --trusted-public-key-sha256 INDEPENDENTLY_OBTAINED_PUBLIC_KEY_FILE_SHA256 \
  --openssl /approved/bin/openssl --require-ga
```

验证固定公钥 PEM 的完整字节摘要，以明确的 OpenSSL 路径、固定 argv、无 shell 和 30 秒截止验签；缺签名、签名错误或公钥摘要不符均被拒绝。工具不使用隐式本地/OpenSSL 信任库或生成发布密钥。有效签名仅关闭 `signature.pending`，依赖和验收门槛仍须通过。Ed25519、Sigstore、PKCS11 需另行审查的 verifier adapter。

候选准备和有效候选验证即使 GA 待办未清，也返回 0；须读取 JSON 的 `ga_ready`、`ga_blockers`。指定 `--require-ga` 后，仍有阻断返回 1，全部声明门槛及可信签名通过才返回 0。格式错误、不安全或不一致输入返回 2；缺验证器/密钥不会视为签名成功。

## 资源与维护边界

两工具拒绝符号链接、Windows reparse point 和硬链接输入，路径组件限可移植 ASCII。每次读取前后核文件身份、大小和修改时间；调用方仍须保证独占静止输入。验证需预留解包磁盘空间。

默认上限为十万个文件/目录、单个安装/证据文件 4 GiB、安装内容/归档各 32 GiB、证据总量 32 GiB、单份 JSON/许可/告知或 ZIP 中央目录 32 MiB、JSON 深度 24、含键两百万节点、单字符串 8192 UTF-8 字节。Python 维护 API 只允许收紧上限。工具拒绝压缩，接受的展开比例固定为 1，不做可执行格式反序列化。

章节验收时在 Windows 和 Ubuntu 各运行一次既有脚本测试，并完成约定的构建/测试/安装验证，复用兼容构建树。可选 OpenSSL 测试使用临时测试密钥；跳过不构成正式签名证据。
