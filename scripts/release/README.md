# Offline release preparation

These maintainer tools prepare and independently inspect SDK installation prefixes. They are not installed as product commands. Python 3.10+ and its standard library suffice for unsigned candidates. Authenticating a release additionally requires an explicitly selected mature OpenSSL executable and a trusted RSA/ECDSA public key with an independently obtained SHA-256 pin.

See [the release workflow](../../docs/development/releasing.md) for schemas, evidence requirements, signature commands, resource limits, and exit statuses.

```sh
python scripts/release/prepare_release.py inventory --prefix /opt/jyppx-stage \
  --output /work/installation-inventory.json
python scripts/release/prepare_release.py prepare --help
python scripts/release/verify_release.py --help
python tests/scripts/test_release.py
```

The last command uses synthetic fixtures. Its optional OpenSSL test creates an ephemeral test key outside release output; it never produces a trusted project signature. Set `JYPPX_RELEASE_TEST_OPENSSL` to an explicit verifier executable when the test environment does not put OpenSSL on `PATH`.
