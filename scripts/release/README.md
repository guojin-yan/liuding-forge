# 离线发布准备

这些维护工具从 SDK 安装前缀准备候选并独立核验，不作为产品命令安装。未签名候选仅需 Python 3.10+ 标准库；认证发布还需明确指定成熟的 OpenSSL 可执行文件，以及通过独立可信渠道取得的 RSA/ECDSA 公钥和文件 SHA-256。

完整结构、证据、签名命令、资源限制及退出码见[发布流程](../../docs/development/releasing.md)。

```sh
python scripts/release/prepare_release.py inventory --prefix /opt/jyppx-stage \
  --output /work/installation-inventory.json
python scripts/release/prepare_release.py prepare --help
python scripts/release/verify_release.py --help
python scripts/release/check_sdk_transition.py --help
python tests/scripts/test_release.py
```

最后一条命令使用合成夹具。可选 OpenSSL 测试在发布输出之外生成临时测试密钥，不产生受信任的项目签名。若测试环境的 `PATH` 不含 OpenSSL，可将 `JYPPX_RELEASE_TEST_OPENSSL` 设为明确的验证器路径。

`check_sdk_transition.py` 是另行显式运行的维护验收：使用两套真实安装的 C++ CLI 和原生双输入配方，验证跨构建 exact 续训拒绝、拒绝前后状态保持、切回原 SDK 完整恢复，以及候选独立任务与完成态重放。它不安装 SDK、不迁移或伪造 lineage，也不使用 Python 训练；所需动态库由调用方提供。
