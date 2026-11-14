# 开发与验证脚本

此目录存放维护、独立参考和验证脚本。面向用户的原生 C++ 命令位于 [tools](../tools/README.md)，示例位于 [examples](../examples/)；此目录的脚本不作为已安装产品的运行入口。

从仓库根目录运行命令；Python 脚本按自身路径定位默认输入。

| 目录 | 用途 |
| --- | --- |
| `validation/` | 兼容与依赖声明、组件依赖图、头文件守卫、检查点夹具、coverage、sanitizer 和 fuzz 验证 |
| `benchmarks/` | Windows/CUDA 采样编排与同环境基线比较，方法见[资源基线](benchmarks/RESOURCE_BASELINE.md) |
| `interop/` | 官方 safetensors/TorchScript 独立验证、测试依赖锁和来源记录 |
| `release/` | 离线候选准备、安装清单、依赖与证据关联及独立验证 |
| `maintenance/` | 将固定来源的 YOLO 架构配置转换为随源码保存的数值定义，生成物保留上游来源和许可 |
| `telemetry/` | 按锁定的官方 schema 重新生成原生遥测协议描述符，见[维护说明](telemetry/README.md) |

这些脚本是可重复使用的维护工具；依赖锁和制品来源元数据应保留，某次运行的安装报告、环境探测、验收日志和本机路径应保存到仓库外。`format_code.sh` 提供批量格式化入口，运行前先检查差异；它会改写源码。

数据准备使用原生 `liuding-forge data prepare`，本地接入和显式下载见[数据准备示例](../examples/data/README.md)；独立数据参考保留在 `tests/reference/`。

训练和搜索的 JSON 参数模板位于 [recipes](../recipes/)，不属于此目录的维护脚本。维护脚本、脚本测试和生成日志均不安装到 SDK；产品功能由原生 C++ 库及 CLI 提供。

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
