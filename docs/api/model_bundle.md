# 模型包

头文件：`<jyppx/liuding/interop/model_bundle.hpp>`。`jyppx::liuding::interop` 提供数据型模型包的清单、完整性校验、快照读取、签名策略和原子发布，默认使用 v2，继续严格读写 v1。部署执行见 [ONNX](onnx.md)。

## 内容与格式

模型包目录包含 `manifest.json` 和完整文件清单。每个文件具有唯一 ID、可移植相对路径、角色、长度和完整 SHA-256，必须恰好被一个模型或描述符引用；未声明的文件、目录和未引用项均拒绝。描述符由所属模块定义独立 schema，interop 只校验外层结构和字节身份。

清单必须声明构建身份、模型、输入/输出 schema、预处理/后处理、evaluation、SBOM、provenance 和 runtime requirements；labels 与权重导入报告可选。依赖使用固定版本，不接受通配符。预处理为 identity 时也须显式声明。压缩与加密目前均为 `none`。

v2 另有必须显式提供的 `evidence` 数组，允许为空。导出报告、样例 Tensor、数值报告、trace 等证据文件使用角色 `BundleFileRole::evidence`（10），各由一个 `BundleDescriptorRef` 声明所属 schema ID、正整数版本和文件 ID；按文件 ID 确定性排序。interop 校验外层关系和完整字节，不解释证据内容，也不据此授予任务、数值或性能资格。

v1 不接受 `evidence` 字段或角色 10。升级已有包时，将 `schema_version` 设为 2、显式增加数组并重新签名；降为 v1 前必须没有 evidence 引用或角色 10 文件，不能静默丢弃证据。未知版本、未知字段和重复归属均拒绝。

| 模型格式 | 校验范围 |
| --- | --- |
| safetensors | 校验每个 shard 的完整摘要及全部 Tensor 名称、dtype、shape 和 shard 关联；同一模型中的 Tensor 名称唯一 |
| ONNX | 仅支持一个自包含模型文件，media type 为 `application/onnx`；不接受 external data。interop 只校验字节完整性，`integrity_only_models` 明确计数；官方 checker、图支持和数值另由部署模块验证 |

`shard_safetensors_for_bundle()` 将完整 Tensor 按编码后字节上限分组，不拆分 Tensor，不改变类型。返回值同时拥有编码字节、文件清单和 Tensor 关联；调用方保存源文件后参与发布。编码会分配完整输出，属于有界内存编码器，不能当作流式写入器。

## 校验与读取

```cpp
#include <jyppx/liuding/interop/model_bundle.hpp>
#include <array>

namespace io = jyppx::liuding::interop;
auto checked = io::verify_model_bundle_directory("artifacts/model-v1");

// 使用调用方独立持有的原始 manifest SHA-256；不能用重编码结果代替。
jyppx::liuding::ArtifactDigest expected{.hex = "完整的64位小写SHA256"};
const std::array<std::string_view, 1> selected{"primary-model"};
auto snapshot = io::read_model_bundle_snapshot(
    "artifacts/model-v1", expected, selected, {}, 256ULL << 20U);
// snapshot.files.at("primary-model") 是已校验的同一份字节，随后直接交给运行端。
// snapshot.manifest_bytes 保留原始编码和空白；不会按路径重新加载模型。
```

`verify_model_bundle_directory()` 返回已校验的元数据和原始 manifest 摘要，不授予随后重新打开路径的保证。需要加载内容时使用 `read_model_bundle_snapshot()`：传入预期 manifest 摘要和唯一文件 ID，完整校验包内所有文件，只保留选中内容和原始 manifest。空选择仅保留 manifest；未知或重复 ID 拒绝。返回结果独立拥有字节，后续文件被替换不会改变结果，任何失败均不返回部分快照。

`max_snapshot_bytes` 限制原始 manifest 与全部选中 payload 的合计字节，默认 512 MiB；在读取选中数据前预检声明的合计大小。该上限不包含解析元数据、allocator 开销和一个未选中的 safetensors 临时 shard，也不是 RSS 上限。普通目录校验只暂存一个 safetensors shard，其他数据流式校验。

## 发布

已有清单和本地文件可用原生 C++ CLI 发布：

```sh
liuding-forge model bundle pack --manifest manifest.json --manifest-sha256 SHA256 --source ./files --output ./model-v2
liuding-forge model bundle verify --path ./model-v2
```

`--manifest-sha256` 是调用方独立持有的原始清单摘要。源目录按清单相对路径提供文件，命令只复制声明的文件；逐文件校验大小、摘要和结构，并以 no-follow、私有暂存和不覆盖目标的原子发布完成交付。父目录须已存在，目标须为新目录。输出清单使用 canonical 编码，`source_manifest_sha256` 与发布后的 `manifest_sha256` 分别报告，可能不同。`pack` 不执行模型或证据内容；默认只报告完整性与签名存在状态。启用原生签名组件后，可传入 `--trust-policy` 与 `--trust-policy-sha256` 验证已有签名，见[签名指南](../user-guide/bundle-signing.md)。

`publish_model_bundle(destination, manifest, sources, options)` 要求父目录已存在，并为每个清单文件提供一个只读源路径。实现独占创建同级暂存目录，复制、校验、写入 canonical manifest、检查完整目录并 flush，然后以 no-replace rename 发布。既有目标（含空目录）及竞争创建的目标都不会被覆盖；下一版本应使用新目录名。

支持 Windows 本地盘与 Linux；不支持 UNC/device 路径或非原子复制降级。Linux 使用 `renameat2(RENAME_NOREPLACE)` 并 fsync 子目录和父目录；Windows 使用 `FlushFileBuffers` 与 `MOVEFILE_WRITE_THROUGH`。文件系统必须提供对应语义。rename 之前失败会尝试清理私有暂存，既有包保持完整；rename 成功后的持久化错误可能留下完整目标，应先核验再重试。崩溃遗留的暂存目录不会自动收养或删除，目录发布不管理应用的活动版本指针。

## 路径与资源

包内路径仅使用小写 ASCII 字母、数字、`_`、`-`、`.`、`/`；每段最多 128 字节、最多 16 段。拒绝空段、`.`/`..`、Windows 设备名、尾随点、绝对路径、盘符、反斜杠、alternate stream、大小写别名及文件/目录前缀冲突。外层可信本地目录和源路径可含中文、空格。

原生句柄逐级拒绝 symlink/reparse point，文件须为单硬链接的普通文件。Windows 读取时拒绝写入/删除共享；Linux 比对读前后的大小、链接数、mtime/ctime。读取前后都核对目录清单。调用方须保证可信父目录权限与操作期间的命名空间稳定；这些机制不构成 OS 沙箱，也不防护特权修改、挂载替换或受损内核。

默认上限为 manifest 16 MiB、1,024 文件、32 模型、65,536 Tensor、rank 32、单维/元素数 2³²、单文件 512 MiB、数据文件合计 4 GiB；`max_total_bytes` 不含单独计费的 manifest。字符串最多 4,096 字节，路径默认 1,024 字节，最多 16 签名、单个签名字节最多 2,048。JSON 深度、节点数和 Tensor 展开量也独立受限。资源超限抛 `ResourceExhausted`，编码错误抛 `SerializationError`，不支持的版本/编码抛 `CompatibilityError`，文件系统或 no-follow 错误抛 `IoError`。

## 签名与信任

`model_bundle_signing_payload()` 根据清单版本返回域前缀 `JYPPX-MODEL-BUNDLE-SIGNATURE-V1` 或 `JYPPX-MODEL-BUNDLE-SIGNATURE-V2`、一个换行字节和去掉整个 signatures 字段后的 canonical JSON。清单中的摘要、大小、构建、模型/Tensor 关联、描述符、evidence 和 runtime requirements 全部被覆盖；键和基于身份的数组确定性排序。版本变化需要重新签名，不能跨版本沿用签名字节。

调用方可通过 `ModelBundleVerifyOptions::signature_verifier` 提供密码实现与可信 key 策略。回调收到实际完整签名字节，必须验证每个声明的签名；拒绝会失败，抛错会转换为 `BackendError`。默认关闭的 `bundle_signing` 组件提供基于 OpenSSL EVP 的原生 Ed25519 实现，头文件为 `<jyppx/liuding/deployment/bundle_signing/provider.hpp>`，链接 `jyppx::liuding::bundle_signing`。`Ed25519Signer` 显式导入 PKCS#8，`Ed25519TrustPolicy` 绑定独立授权的原始策略摘要，`trusted_ed25519_options()` 将策略接入同一校验流程；签名、CLI 与学习示例见[签名指南](../user-guide/bundle-signing.md)。

| 策略与输入 | 结果 |
| --- | --- |
| 无签名、默认策略 | `unsigned_bundle` |
| 有签名、未提供 verifier | `present_unverified` |
| 所有签名经显式 verifier 接受 | `trusted` |
| 要求可信签名但缺签名或 verifier | 拒绝 |
| verifier 拒绝或抛错 | 拒绝 |

完整 SHA-256 仅证明内容身份；签名存在也不代表已认证。快照读取沿用同一信任策略。两个版本均没有插件或可执行角色，会拒绝常见脚本、动态库、可执行文件、pickle、Torch archive 和 zip 名称及常见可执行魔数；这不是通用恶意文件识别器，包内文件不会被自动导入、反序列化为对象或执行。
