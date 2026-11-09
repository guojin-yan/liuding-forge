# SDK 候选范围与迁移边界

本页说明当前实现及使用边界。平台、编译器、LibTorch、设备和测试范围以[兼容矩阵](../../compatibility.json)为准；组件存在或局部测试通过，不代表完整 v1.0 已通过正式发布验收。

## 已实现的主要合同

SDK 聚合 foundation、core、interop、data、nn、optim、training、experiment 八个 Core，使用 `find_package(jyppx)` 与 `jyppx::liuding::<component>`。Models、Vision、部署和原生遥测组件独立选择，`sdk` 不自动启用它们。产品命令为 `liuding-forge`；开发维护命令归仓库 `scripts/`，不作为安装后的产品入口。

本地存储使用 SQLite WAL/FULL 事务和内容寻址制品库。进程崩溃恢复验证覆盖已提交与未提交边界，不授予断电、故障存储固件或网络文件系统资格；备份使用显式一致性快照，不能只复制打开的数据库文件。评估报告绑定套件、输入与来源，本地报告通过不自动授予部署资格。

本地执行和有界 grid/random 搜索已实现，按顺序运行原生 C++ 训练程序，固定程序身份、计划、预算和选优指标。恢复、retry 和外部清理确认须满足所用任务协议；通用 RunSpec、所有配方及完整资源矩阵仍须各自验收。使用方法见 [CLI](cli.md)。

遥测使用 C++ `tensorboard-native` 或 `opentelemetry-native`，不安装或调用 Python runtime；旧 Python backend、`--sidecar-config` 和 Python 安装入口明确拒绝。必需证据与可采样观测分开处理，失败保留原始事件供显式重放；进程清理和截止边界见[运行遥测](../api/run_telemetry_cli.md)。HTTP 响应或进程成功不等于远端持久 ACK 或 exactly-once。

原生 C++ 通用 ONNX 导出、独立 ONNX Runtime 和本地 Serving 已实现，分别由可选组件提供，导出和推理无需 Python。导出按已支持算子组合及显式 lowering 工作，缺失算子明确报错；不承诺直接转换任意 `.pt` 文件。ORT 的 CPU 证据不授予未验证 Provider 或设备资格；详见 [ONNX](../api/onnx.md)与[部署](deployment.md)。

本地 Serving 支持显式模型加载、预测及同 runtime 的受控版本切换，具体后端的 batch、资源、取消和 drain 合同见[Serving API](../api/serving.md)。逻辑字节预算不等于进程 RSS、显存或后端 workspace 的硬上限；不能由组件冒烟推导真实任务数值、质量或性能资格。

## 格式、执行与信任边界

Safetensors 和模型 bundle 是有边界的数据格式。摘要验证用于核对内容，生产者身份依赖外部信任的签名验证；bundle 通过不证明模型的任务质量、运行时或设备兼容。真实签名组件可用不代表正式发布资产已完成受信签名。

TorchScript 仅用于显式可信既有制品的 LibTorch JIT 加载与 forward，须提供完整制品摘要；它不是沙箱或通用 tracing/export API。取消和 deadline 在声明边界检查，不能抢占正在运行的算子或 CUDA kernel；CUDA 成功返回前同步设备。详见[互操作运行时](../api/interop_backends.md)。

## 升级与回退

旧包别名、全局 logger、旧 DataLoader/ThreadSafeQueue、单优化器 Trainer 和分布式占位接口已退役。检查点参与者、数据 epoch/cursor、模型调用及优化器拓扑需按当前合同迁移，不能依赖命名空间替换保持行为兼容。

`liuding-forge api scan` 提供有界词法诊断，包含注释和未激活代码，不是 AST 自动迁移或 API/ABI 兼容证明。配置迁移使用应用拥有的 schema 标识、显式版本链和逐步类型校验，默认预览；apply 与 rollback 各写入新目录，核对原文及摘要收据，不覆盖原配置。详见[配置迁移](../api/config_migration.md)。

检查点只接受参与者明确声明的当前/N−1 schema，未知版本或不兼容拓扑在修改状态前拒绝。保留原应用、依赖锁、数据和检查点供回退；新写出的状态不保证任意旧版本可读。SDK 升级、应用状态回退与正式 Beta/RC 演练分别验收。

## 尚未授予的资格与延期范围

模型架构、官方权重及许可、数值、完整训练质量、恢复、设备和资源资格分别判断；某个模型或任务通过，不能外推全部模型、四类配方或平台矩阵。当前完整覆盖率、性能资源、依赖许可与安全、SBOM/provenance、正式资产签名及 Beta/RC 升级回退仍属于正式发布门槛。

多进程分布式训练、复杂或并行 HPO、远端 runner、平台注册/推广、HTTP/gRPC/TLS 服务托管及部署控制面不属于当前候选范围。macOS 未取得 v1.0 支持资格；Windows、Ubuntu CPU/CUDA 按实际组合单独记录。全部 INT8 量化已延期，当前产品不提供量化入口。
