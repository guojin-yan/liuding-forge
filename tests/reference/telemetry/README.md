# 遥测独立测试参考

此目录只供维护 checker 使用，不安装、不作为产品运行入口。`official_mapping.py` 保留已冻结官方 TensorBoard writer/OTLP schema 映射，删除旧 stdin/stdout 服务及 backend CLI；直接执行会明确拒绝。产品使用 C++ `tensorboard-native`、`opentelemetry-native`。

`requirements.lock` 与 `dependency-metadata.json` 保持原锁定的 wheel 身份；用独立 CPython 3.11 环境执行 `pip --require-hashes --only-binary=:all:`。原生测试读取官方事件文件、解码 protobuf，并由独立 Collector fixture 检查互操作；Python 不参与产品导出。
