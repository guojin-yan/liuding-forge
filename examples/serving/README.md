# 本地推理示例

示例需要开启 `JYPPX_LIUDING_BUILD_SERVING` 和 examples 构建。全部使用生成的 safetensors、应用显式拥有的 C++ 模型工厂与独立已知向量，不下载模型，不运行 Python，不启动网络监听器。

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
