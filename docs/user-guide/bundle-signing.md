# 原生模型包签名

启用可选的 `bundle_signing` 组件后，可以使用 C++ 为已有模型包生成 Ed25519 签名，并通过外部授权公钥验证模型、描述符和证据清单是否被修改。签名不替代模型格式检查、推理结果验证或生产密钥管理。

构建时设置 `JYPPX_LIUDING_BUILD_BUNDLE_SIGNING=ON` 和 `JYPPX_LIUDING_BUILD_INTEROP=ON`，通过 `JYPPX_LIUDING_OPENSSL_ROOT` 显式提供 OpenSSL 3.5.9 静态 SDK 的绝对路径；默认关闭，不从 PATH 或 Python 环境寻找加密库。Windows SDK 使用 `/MD`。安装包会携带私有静态 Crypto 后端及其许可证。应用使用 `find_package(jyppx CONFIG REQUIRED COMPONENTS bundle_signing)`，链接 `jyppx::liuding::bundle_signing`，包含 `<jyppx/liuding/deployment/bundle_signing/provider.hpp>`，无须另外配置 OpenSSL SDK 或包含 OpenSSL 头文件。

## 签名与验证

私钥使用显式指定的 PKCS#8 Ed25519 PEM 文件；加密文件通过 `PassphraseCallback` 传入口令。组件不提示输入口令，不调用外部程序，不把私钥放进模型包。应用必须保护原始私钥文件、口令来源及调用方持有的输入副本。

```cpp
namespace signing = jyppx::liuding::bundle_signing;
namespace interop = jyppx::liuding::interop;

// expected_policy_sha 来自应用独立授权的可信配置。
const auto policy = signing::Ed25519TrustPolicy::load(
    policy_path, expected_policy_sha);
const auto signer = signing::Ed25519Signer::from_pkcs8_file(private_key_path);
const auto publication = signing::sign_model_bundle_directory(
    source_bundle, expected_source_manifest_sha, new_bundle,
    signer, policy);

// 消费者只需要授权公钥策略，无须私钥。
const auto verified = interop::verify_model_bundle_directory(
    new_bundle, signing::trusted_ed25519_options(policy));
```

CLI 使用同一外部策略：

```text
liuding-forge model bundle sign --source ./unsigned --manifest-sha256 <源清单SHA256> --output ./signed --private-key-file ./private.pem --trust-policy ./policy.json --trust-policy-sha256 <可信策略SHA256>
liuding-forge model bundle verify --path ./signed --trust-policy ./policy.json --trust-policy-sha256 <可信策略SHA256>
```

CLI 签名入口当前仅接收未加密 PKCS#8 私钥文件；加密私钥会拒绝，不提示口令。需要加密私钥的应用使用 C++ `PassphraseCallback`。`model bundle verify/pack` 以及基于模型包的 `model onnx smoke/verify` 可传入 `--trust-policy` 和 `--trust-policy-sha256`；二者必须成对提供。提供策略后默认要求可信签名，可通过 `--require-trusted false` 显式允许无签名包，但每条已声明的签名仍必须通过同一策略。直接指定裸 ONNX 文件的入口不接收模型包策略。

策略文件最多 4096 字节，包含固定格式名、版本 1、算法 `ed25519`、原始公钥、由公钥 SHA-256 导出的 `key_id` 和 `revoked`。加载时必须提供策略原始文件的预期 SHA-256；从同一个不可信下载来源读取文件和摘要不能建立授权。轮换公钥或撤销时，更新应用信任配置中的策略和独立摘要。

公钥必须使用规范编码，并拒绝已知小阶点；其余签名验证由 OpenSSL 完成。这不表示提供独立的完整素数阶子群检测。当前策略明确授权一个公钥。模型包必须具有该 key 的有效签名，所有声明的签名都必须通过验证；未知 key、撤销 key、重复 signer、算法替换及一真一假的签名组合都会失败。现有有效签名会保留。签名覆盖现有版本化 canonical payload，包含完整模型、描述符和证据引用；不修改签名域或模型包格式。

签名操作先按预期 manifest SHA-256 完整验证源包，再流式发布到源包外的新目录，并在发布前用相同策略复验。已有输出目录不会被替换。提交前失败会尝试清理临时目录；原子改名后的持久化错误可能留下完整的新包。

## 学习示例

`examples/deployment/bundle_signing.cpp` 创建极小 safetensors 包、生成内存临时 key、写出公开策略、签名到新目录，并演示普通文件 hash 一致时，修改清单元数据仍然会被真实验签拒绝。

示例仅在构建树中生成，需要同时启用 `JYPPX_LIUDING_BUILD_EXAMPLES=ON`，不会随库安装。以已配置的 `build` 目录为例：

```text
cmake --build build --config Release --target jyppx_liuding_bundle_signing_example --parallel 2
```

Linux 在构建树运行 `./build/examples/liuding-forge-example-bundle-signing`，Windows 运行 `./build/examples/Release/liuding-forge-example-bundle-signing.exe`；可在命令末尾传入 `./new-signing-demo` 保留输出。

无参数运行会使用新建临时目录并在结束时清理；指定的新目录会保留 `unsigned/`、`signed/`、`tampered/` 及公开策略，便于查看。示例私钥只存在于进程内存，退出后无法复用。这些数据仅用于学习签名与信任边界，不代表模型精度、运行时支持或生产发布认证。

本组件使用 OpenSSL EVP 的 Ed25519 one-shot 实现，不预先对 payload 做 SHA-256，不手写密码算法。精确 SDK 与平台支持以依赖锁和已验收矩阵为准；当前接口不提供证书链、KMS、阈值签名、时间戳或 FIPS 资格。

密码操作使用独立 OpenSSL library context 和显式内置 `default` provider，隔离宿主的全局 provider/property 设置；实际加载的 OpenSSL 完整版本号必须与编译 SDK 相同。该检查只核对版本身份，不能替代依赖来源、许可证或安全审查。
