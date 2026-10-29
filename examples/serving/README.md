# 本地推理示例

示例需要开启 `JYPPX_LIUDING_BUILD_SERVING` 和 examples 构建。前两例使用生成的 safetensors 和应用显式拥有的 C++ 模型工厂，ONNX 示例直接在 C++ 中导出并发布模型。全部使用独立已知向量，不下载模型，不运行 Python，不启动网络监听器。

## JSONL 请求

`liuding-forge-example-serving-jsonl` 使用官方 nlohmann JSON parser、双特征模型、bundle 验证、warmup、有界调度与服务适配器，通过 stdin/stdout 处理本地请求。

```text
liuding-forge-example-serving-jsonl --demo
{"request_id":"example-1","input":[1,2]}
{"request_id":"example-2","input":[3,4],"deadline_ms":5000}
```

结果为 `status:ok`，输出分别为 `[2,-6]` 和 `[6,-12]`。`--demo --health` 查询实际 readiness，`--manifest` 输出能力声明，`--cuda 0` 选择精确可用设备，`--bundle PATH` 加载已经发布的兼容示例 bundle；路径属于进程配置，不从请求中读取。

每行最多 4 KiB，嵌套深度最多 8，重复或未知字段被拒绝；input 必须是两个 [-1e6,1e6] 范围内的有限数。deadline 为 0..60000 毫秒的协作期限。身份为当前本地操作员和固定 project，远程认证/TLS 由宿主负责。JSONL 主循环顺序执行，动态 batch 的并发行为由独立 C++ 测试和安装消费者覆盖。

## 同 runtime 版本切换

```text
liuding-forge-example-serving-reload
```

[reload.cpp](reload.cpp) 生成两个 CPU bundle：v1 权重为 `[2,-3]`，v2 为 `[5,7]`。示例先获取 v1 lease，再旁路加载、验证、warmup 和发布 v2，展示旧 lease 仍得到 `[2,-3]`，新请求得到 `[5,7]`。释放旧 lease 后显式 drain，最后关闭当前实例。

预期输出包含 `pinned example-linear-v1: [2,-3]`、`current example-linear-v2: [5,7]` 和 `resident_instances=1`。`peak_reserved_bytes` 是双实例声明预算，不是实测 RSS/显存。示例不声称跨后端/设备切换，也不证明训练精度或生产负载性能；完整约束见[Serving API](../../docs/api/serving.md)。

## ONNX bundle 与多输入输出

[onnx.cpp](onnx.cpp) 演示从 C++ 模型到 ONNX bundle，再到本地推理与受控版本切换。除 Serving 和 examples 外，还需要启用 `onnx_export`、`onnx_runtime` 和 `onnx_serving` 组件，并准备对应版本的 ONNX Runtime SDK；构建配置见 [ONNX API](../../docs/api/onnx.md)。

```text
liuding-forge-example-serving-onnx
```

示例使用通用 `export_module()` 导出两个实际模型：`sum = left * gain + right`、`difference = left - right`，其中 v1 的 gain 为 2，v2 为 5。输入 `left`、`right` 与具名输出都是 `[batch, 2]` 的 CPU FP32 Tensor，首维共享动态 batch 符号，运行上限为 8。模型权重来自演示常量；没有训练精度或预训练模型效果的含义。

学习代码可以按三个阶段阅读：`Bundle` 生成实际 ONNX、四个 owner 描述符和环境锁并发布数据型 bundle；`Owner()` 解码固定描述符并核对图名称、形状、dtype 和设备；`main()` 用独立标量参考完成 warmup、预测和版本切换。环境锁使用公共 `observe_environment()`、`serialize_environment_lock()`，加载时核对实际版本、平台与 Provider 配置；摘要和版本匹配不会使示例的未签名 bundle 获得来源认证。

发布 v2 前持有的 lease 继续使用 gain=2，新的预测使用 gain=5。batch 为 1、2、5 时，两个版本的全部输出分别与 C++ 标量运算得到的独立参考逐元素比较；释放旧 lease 后 drain，最后关闭当前版本。成功输出包含：

```text
batch=1: pinned v1 gain=2, current v2 gain=5; both named outputs match the independent scalar oracle
batch=2: pinned v1 gain=2, current v2 gain=5; both named outputs match the independent scalar oracle
batch=5: pinned v1 gain=2, current v2 gain=5; both named outputs match the independent scalar oracle
provider=CPUExecutionProvider, resident_instances=1, peak_reserved_bytes=8388608
C++ ONNX bundle serving, dynamic batches and version drain passed.
```

两个 bundle 位于各自独占的系统临时目录，示例结束时自动清理。ORT 图 workspace、优化器及 Provider 内部分配不受 `peak_reserved_bytes` 的逻辑快照预算约束。此示例使用显式 CPU Provider 与主机 I/O，不启用动态合批或 OOM 重试；其他 Provider 需要独立 SDK、驱动和执行资格验证。
