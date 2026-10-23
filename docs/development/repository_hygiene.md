# 仓库卫生

源码树只保存代码和稳定项目元数据。运行时数据、日志、构建产物、检查点、TensorBoard event、性能分析输出和下载数据不得写入源码根目录。

## 目录职责

- 公共 API：include/jyppx/liuding/<component>/
- 实现：src/jyppx/liuding/<component>/
- 测试：tests/<level>/<component>/
- 构建辅助：cmake/ 和 scripts/
- 产品工具：tools/，包括 CLI 和可选运行时适配器
- 脚本测试：tests/scripts/；运行时后端与发布验收：tests/integration/
- 生成文件、依赖下载和隔离环境：被忽略的 out/

tools/ 只保存随库交付的产品工具。构建检查、CI、coverage、性能基线和发布编排归 scripts/；对应测试和验收 fixture 归 tests/，不会安装到库中。可选运行时的依赖锁应与对应运行时放在一起，资格审查专用的锁文件归相应脚本和测试所有者维护。

开发交付报告、依赖审计、历史 benchmark 结果、验证日志和原始验收证据统一放在外部 v1.0-plan/06_records/。仓库保留开发人员运行所需的脚本和简洁操作说明，但不得在 docs/ 中提交带日期的报告、本地测量结果或一次性验证证据。

不得创建顶层数字目录或临时目录。新增顶层目录必须登记所有者、用途和保留策略。

## 测试数据和输出

单元测试应在内存或操作系统临时目录中生成确定性数据。小型、经过审阅的 fixture 放在 tests/fixtures/；审阅后的参考输出放在 tests/reference_data/。下载数据和大型数据集放在仓库外的 JYPPX_DATA_ROOT，并提供 manifest 和 checksum。

测试输出放在构建目录的 Testing/ 或唯一的操作系统临时目录中；测试必须清理自己创建的临时文件。

## 构建和清理

新构建目录使用 out/build/<preset>/。保留当前活动构建和必要的一个回滚构建，保存简短结果摘要后删除更旧的构建目录。递归清理前必须解析并核对绝对路径，禁止通过未解析变量或通配符删除。

## Git 门禁

运行仓库卫生门禁：

~~~text
ctest --test-dir <build-dir> -C Release -R RepositoryHygiene --output-on-failure
~~~

门禁拒绝顶层数字目录、旧式顶层构建树、根目录日志和临时文件、被 Git 跟踪的构建/运行时路径、未经登记且超过 5 MiB 的测试数据、缺失法律文件，以及偏离 Apache-2.0-only 的项目许可证元数据。门禁也会拒绝没有完整版本化实现批准的退役占位 API。被 .gitignore 忽略不代表可以污染源码树。
