# CESS v0.9.0 一维接口示例

[English](README.md) | [简体中文](README_zh-CN.md)
`make_toy_input.py` 生成一份可公开的微型 BayeSED-shaped HDF5。
连续谱、线形和星系参数均为任意设定，仅用于安装与接口演示，不用于科学定标。
实际研究仍使用满足[输入契约](../docs/USER_GUIDE_zh-CN.md)的星系模板。

## 1. 准备环境和工作目录

先按[安装指南](../docs/INSTALL_zh-CN.md)安装 CESS，并激活选定的 Python3.12 环境。
选择一个新的可写工作目录并进入它。以下 `toy.hdf5`、`results`、`run.json`、`lines` 和图像
都相对此工作目录；`/path/to/CESS` 须替换为自己的源码位置。

```bash
python -c 'import sys, cess; print(sys.executable, cess.__version__, cess.__file__)'
python /path/to/CESS/examples/make_toy_input.py toy.hdf5 --sources 4 --invalid-first
```

## 2. 模拟完整文件

```bash
cess simulate -i toy.hdf5 -o results --all-effects --seed 1214 --manifest run.json
```

默认1个worker；四个源写入同一份 `results/CSST_grism_toy_4.hdf5`。
源0被故意设置为无效SED，应保留ID与行号并标记 `data_mask=2`；其模拟结果为无效值，
其他三个源正常处理。`--all-effects` 同时启用四个科学开关；不加它时这些开关默认关闭。

不指定JSON时使用包内默认仪器资源。现代入口默认输出schema3.0；
需要旧布局时另选新的输出目录和manifest，并增加 `--output-layout legacy`。
已有输入、产品和manifest不会被静默覆盖。再次练习时换一个新工作目录或使用下述续跑方式。

## 3. 读取与绘图

在同一工作目录、已安装CESS的Python环境或Notebook中执行：

```python
from cess.io.products import open_product, read_product_summary

print(read_product_summary('results/CSST_grism_toy_4.hdf5'))
with open_product('results/CSST_grism_toy_4.hdf5') as product:
    print(product.catalog['ID'][:])
    print(product.catalog['data_mask'][:])
    print(product.catalog['input_row_indices'][:])
    wave = product.band('GV')['wave'][:]                 # Angstrom
    flux = product.band('GV')['flux_ujy_with_noise'][1]   # microJy
```

绘制有效的第1行光谱；需要 `plot` 可选依赖：

```bash
python -m cess.cli.demo results/CSST_grism_toy_4.hdf5 1 emissionline --save spectrum.png
```

## 4. 发射线后处理

```bash
cess el-detect -i results/CSST_grism_toy_4.hdf5 -o lines --manifest lines.json --seed 1214
```

该步骤读取既有模拟产品并写出独立线测量产品，不重新生成含噪光谱。
EL-only文件不复制父产品的全部光谱大数组；保留父模拟文件以供后续读取和解释来源。

## 5. 清单与续跑

确认旧worker已经停止，保留首次运行的参数后，可先只读预检再续跑：

```bash
cess simulate -i toy.hdf5 -o results --all-effects --seed 1214 --manifest run.json --resume --dry-run
cess simulate -i toy.hdf5 -o results --all-effects --seed 1214 --manifest run.json --resume
```

已经完成且核验通过的任务会保留；缺失产品才会重新计算。输入、配置、源区间、chunk、
输出布局、科学开关、种子或源码/资源身份改变时，不应沿用原清单。
新任务的 `--dry-run` 不附加manifest；同时使用新 `--dry-run --manifest` 会创建清单。

未安装包、直接运行源码时，将命令中的 `cess` 替换为
`env PYTHONPATH=/path/to/CESS/src python -s -B -m cess`，解释器仍须已安装所需依赖。
正式星表批处理、并行、分片合并、日志和常见错误见[Bash 命令指南](../docs/COMMANDS_zh-CN.md)。
