# stb 私有实现来源

`stb_image.h` 基于上游 2.30、不可变提交 [`013ac3beddff3dbffafd5177e7972067cd2b5083`](https://github.com/nothings/stb/blob/013ac3beddff3dbffafd5177e7972067cd2b5083/stb_image.h)，原文件 SHA-256 为 `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3`。本地版本 `2.30+liuding.1` 保留原许可及署名，只修改 `stbi__decode_jpeg_image`：marker 解析失败返回失败；至少一次 scan 成功解析后才允许发布像素，拒绝 SOF 后直接 EOI 的输入。当前文件大小和 SHA-256 以[依赖清单](../../dependency-manifest.json)为准，不能把本地修改后的摘要当作上游原件摘要。

解码边界通过[原生测试](../../tests/unit/data_image/test_decoder.cpp)检查 JPEG 表长度、采样比例、缺少 scan、16 位 PNM 通道扩展及截断输入，并验证失败后的资源释放与后续有效解码。这些边界回归不替代原始漏洞样本、sanitizer/fuzz 或完整依赖安全审计。

`stb_image_write.h` 仍为上游 1.16、不可变提交 [`1ee679ca2ef753a528db5ba6801e1067b40481b8`](https://github.com/nothings/stb/blob/1ee679ca2ef753a528db5ba6801e1067b40481b8/stb_image_write.h)的未修改文件，与解码器分别登记。两份文件的 MIT/Unlicense 许可选择、版权及免责声明保持；再分发声明见[第三方告知](../../THIRD_PARTY_NOTICES.md)。
