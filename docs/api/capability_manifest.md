# 能力清单

公共头文件为 `<jyppx/liuding/foundation/capability.hpp>`，链接 `jyppx::liuding::foundation`。`CapabilityManifest` 使用 schema1 描述一个实际构建或配置实例的能力；`BuildFingerprint` 标识二进制和后端构建，两者分别记录身份与能力。

`CapabilityDescriptor` 包含唯一 feature ID、正版本号、status（available/experimental/unsupported）、implementation（native/delegated/adapted/unsupported）、source（build/runtime/configured-instance）、backend、devices、dtypes、schemas、operations、数值限制与说明。

`make_capability_manifest()` 接受显式构建指纹，`make_current_capability_manifest()` 绑定当前指纹；二者验证并规范化完整条目。重复 feature ID、集合值或限额名会拒绝，unsupported 条目不能声明可执行 operations。字符串与集合受固定资源预算约束。

`capability_manifest_json()` 验证副本并按条目、集合、限额和说明生成确定性紧凑 JSON，不修改调用者对象。数值限额保持 JSON number。

## 构建能力与运行实例

`current_capability_manifest()` 返回 foundation 中显式登记的16个构建组件项，包括未开启的 unsupported 项；它不是全部 CMake target 的自动枚举。Models、Serving、图像、ONNX 和签名项按实现中的 experimental 标记报告，聚合 target 不独立登记；Vision 模型/任务组合另用[视觉 catalog](vision/index.md)查询。`current_capability_manifest_json()` 返回同一清单的规范 JSON。

领域组件将自身描述转换为同一 foundation schema，例如 `describe_data_capability()` 描述配置后的数据能力。构建项存在不证明某个模型、dtype、设备或操作已经执行，应结合所选实例与[兼容矩阵](../../compatibility.json)判断。
