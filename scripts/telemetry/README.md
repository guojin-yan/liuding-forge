# 遥测 schema 维护

这里的 Python 脚本仅供维护人员调用官方代码生成器，产品构建和运行不调用它。`native_schema.lock.json` 固定官方 schema、nanopb 0.4.9.2 的运行库与生成器源码，以及 protoc 33.5 发布包的 SHA-256。维护环境需安装 `protobuf==6.33.5`，并离线准备锁中指定的官方源码和 protoc ZIP；不使用 LibTorch 私有 protobuf SDK。

官方 schema 按原始路径放入 `schema-sources/tensorboard/` 和 `schema-sources/otlp/`。例如 TensorBoard Event 的路径是 `schema-sources/tensorboard/tensorboard/compat/proto/event.proto`。锁中的 URL 指向不可变提交，获取文件时核对 SHA-256。

```sh
python scripts/telemetry/generate_native_schema.py \
  --protoc /tools/protoc/bin/protoc \
  --protoc-archive /tools/protoc-33.5-linux-x86_64.zip \
  --protoc-include /tools/protoc/include \
  --nanopb-source /sources/nanopb-0.4.9.2 \
  --schema-sources /sources/schema-sources \
  --family all
```

Windows 将 `--protoc` 指向 `protoc.exe`，并使用对应的官方 win64 ZIP。脚本核对发布包、解压出的可执行文件和 includes、源码与 schema，使用官方 protoc/nanopb generator 生成私有包名前缀的 C 描述符；只统一换行符，不修改生成的 token。生产构建直接使用已入库的描述符，不需要生成器。`--family otlp` 和 `--family all` 仅生成协议描述符，不表示相应的 C++ 后端已经实现或通过验收。

生成后应在 Windows、Ubuntu 集中批次中检查官方 reader 互通、安装消费及符号隔离。仅成功生成代码不能证明运行时互通。

生成器同时维护输出目录中的 `generated.sha256`，逐项记录完整 schema 闭包的路径及 SHA-256；卫生检查要求清单、文件集合和官方生成标识全部匹配，才允许这些机器生成文件省略产品文件头。首次生成使用 `--family all`；单独更新某一族时，未选族必须与已有清单完全一致，不能借重新写清单接受其改动。不要手改生成代码或清单。
