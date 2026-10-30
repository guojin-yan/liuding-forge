# 开发与验证脚本

此目录存放维护、独立参考和验证脚本。面向用户的原生 C++ 命令位于 [tools](../tools/README.md)，示例位于 [examples](../examples/)；此目录的脚本不作为已安装产品的运行入口。

从仓库根目录运行命令；Python 脚本按自身路径定位默认输入。

| 目录 | 用途 |
| --- | --- |
| `validation/` | 兼容与依赖声明、组件依赖图、头文件守卫、检查点夹具、coverage、sanitizer 和 fuzz 验证 |
| `benchmarks/` | Windows/CUDA 采样编排与同环境基线比较，方法见[资源基线](benchmarks/RESOURCE_BASELINE.md) |
| `interop/` | 官方 safetensors/ONNX 独立验证、测试依赖锁和来源记录 |
| `release/` | 离线候选准备、安装清单、依赖与证据关联及独立验证 |
| `datasets/` | 存量 MNIST 数据准备脚本；paired-affine、Fashion-MNIST、CIFAR-10 使用 `liuding-forge data prepare`，独立参考位于 `tests/reference/` |

```sh
python scripts/validation/compatibility.py --file compatibility.json --validate
python scripts/validation/dependency_manifest.py --file dependency-manifest.json
python scripts/validation/header_guards.py
python tests/scripts/test_component_graph.py
bash scripts/validation/run-coverage.sh
```

```powershell
pwsh -NoProfile -File scripts/benchmarks/Run-Baseline.ps1
```

脚本测试位于 `tests/scripts/`，遥测互操作验证位于 `tests/integration/telemetry_sidecar/`。生成物使用被忽略的 `out/`；开发报告和原始验收证据放在仓库外的方案记录目录。
