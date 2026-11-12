# 检查点制品清单

公共头为 `<jyppx/liuding/interop/checkpoint_manifest.hpp>`，链接 `jyppx::liuding::interop`。`CheckpointArtifactManifest`将与存储无关的参与者状态绑定到实际构建指纹、schema、字段长度、相对路径和SHA-256摘要。

schema-v1 codec按稳定顺序生成紧凑JSON，并在解析时拒绝重复或未知字段。限额覆盖JSON字节、深度、节点、字符串、参与者、blob数、路径长度和声明总字节；还检查必需字段完整性、字段上限、路径唯一性和摘要语法。相对路径须符合可移植规则，拒绝绝对路径、父目录跳转、反斜杠和盘符。

```cpp
#include <jyppx/liuding/interop/checkpoint_manifest.hpp>

// manifest须已填写有效的build、参与者清单与每个实际blob的身份。
const auto json =
    jyppx::liuding::interop::checkpoint_artifact_manifest_json(manifest);
const auto decoded =
    jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(json);
```

codec仅校验元数据，不读取文件或计算blob摘要。实际读取、完整性验证、临时文件和原子发布由已实现的 `FileStateArchive` / `FileCheckpointStore`完成，见[持久化检查点](durable_checkpoint.md)。自定义容器必须在暴露reader前独立核对实际长度和摘要，再交给[协调器](checkpoint_coordinator.md)恢复；清单解析成功不表示字节真实、来源可信或整个训练域可恢复。
