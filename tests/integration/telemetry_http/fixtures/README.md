# 独立 HTTPS 夹具

这些证书和公开的测试私钥只用于本机回环协议测试，禁止用于任何真实服务；`server-key.pem` 是专门生成且故意公开的 RSA 2048 测试密钥，不含开发机现有证书或凭据。

`self-signed.pem` 是自签名的非 CA 服务端证书；`unknown-ca-chain.pem` 包含由 `unknown-ca.pem` 签发的服务端证书及该未知 CA。两种服务端证书的 SAN 均包含 `localhost`、`127.0.0.1`、`::1`，有效期为 2020-01-01 至 2040-01-01，包含 serverAuth EKU 和数字签名用途；未知 CA 的私钥不保留。

独立 Python TLS 客户端先仅在自己的 SSLContext 中信任对应测试证书或 CA，以验证匹配主机名、完整握手和服务端密钥；原生 C++ 传输仍使用系统信任，必须两次拒绝相同事件且不确认导出。测试不改动系统或当前用户信任库，不关闭链或主机名验证。

上述静态夹具只验证自签名和未知 CA 拒绝。Linux 的独立 [Trusted TLS 门禁](../../../scripts/test_native_otlp_trusted_tls.py)在私有用户/挂载命名空间内，将临时 CA 加入只读测试信任 bundle，执行可信链成功、过期拒绝和 IP 主机名不符拒绝；返回后核对主机信任文件的字节、inode 和时间戳保持。原生客户端始终使用系统证书链及主机名验证，命名空间退出即释放挂载。

该门禁需要 `openssl`、`unshare`、`mount` 和可用的非特权用户/挂载命名空间；缺失前置条件会明确失败。启用 `telemetry_http` 和测试后构建 `jyppx_test_otlp_http_harness`，再用 `ctest --test-dir <构建目录> -R '^NativeOtlpHttpTrustedTls$' --output-on-failure` 执行。动态证书、原生输出和独立验证结果保存在构建目录的 `tests/unit/trusted-tls/`，各次执行使用独立子目录。

这组门禁不验证 Windows 的可信链成功、证书撤销或 Collector 持久 ACK。HTTPS 协议响应确认仍遵循原生传输的进程内去重合同。
