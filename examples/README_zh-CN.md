# 光谱示例

[English](README.md) | 简体中文

| 文件 | 内容 |
| --- | --- |
| [toy_input.hdf5](toy_input.hdf5) | 4 个符合 BayeSED 输入结构的合成星系光谱 |
| [toy_output.hdf5](toy_output.hdf5) | 对应的 GU/GV/GI 模拟光谱、测光与发射线测量，HDF5 schema 3.0 |
| [make_toy_input.py](make_toy_input.py) | 生成另一份小型输入文件的脚本 |

这些 SED 和星系参数是任意构造的接口示例，不代表经过定标的真实星系样本。第 0 行故意设置为无效 SED，输出保留其 ID 与行号并标记 `data_mask=2`；第 1–3 行包含模拟光谱。

## 打开随附文件

按[安装指南](../docs/INSTALL_zh-CN.md)安装 CESS 后，在仓库根目录运行：

```bash
cess-viewer examples/toy_output.hdf5 --lang zh
```

选择源行号 **1** 查看有效光谱。也可以打开输入文件：

```bash
cess-viewer examples/toy_input.hdf5 --lang zh
```

浏览器需要 `viewer` 可选依赖与可用的 Tcl/Tk，详见[浏览器指南](../docs/VIEWER_zh-CN.md)。

## 重新计算

使用随附输入，并选择一个新的输出目录：

```bash
cess simulate -i examples/toy_input.hdf5 -o toy_results_new --all-effects --seed 1214
```

结果为 `toy_results_new/CSST_grism_toy_input_4.hdf5`，全部 4 行保存在同一文件中。示例使用 standard 模式、随包默认配置及全部 4 个可选效应。已有输出不会被覆盖；重新运行时应另选目录。

随附输出中的来源路径已改为仓库相对路径，方便公开分发；科学数组、掩膜、ID 和种子保持生成时的数值。重新计算会记录你所用环境的 provenance；随附文件不用于续跑旧任务。

也可以自己生成另一份小型输入：

```bash
python examples/make_toy_input.py toy_input_new.hdf5 --sources 4 --invalid-first
```

输入文件的内容身份改变后，即使种子相同，随机流也可能改变。
