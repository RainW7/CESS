# CESS v0.9.0 中文说明

[English](README.md) | 简体中文

**CSST Emulator for Slitless Spectroscopy — CSST 一维无缝光谱模拟器**

版本：v0.9.0 · Python 3.12

CESS 读取 BayeSED 生成的星系光谱模板和星表，在给定的仪器响应、观测条件及形态模型下，
快速生成 GU、GV、GI 三个波段的一维模拟无缝光谱。它面向需要处理大量星系的研究任务：
把已有星系 SED 转换为含噪光谱、误差、信噪比和相关测量，而不要求先生成整幅二维探测器图像。

本文介绍 v0.9.0 的功能、输入输出和模型边界。首次使用先读
[安装指南](docs/INSTALL_zh-CN.md)和[快速开始](docs/QUICKSTART_zh-CN.md)；
可直接复制并逐步理解的命令见 [Bash 命令指南](docs/COMMANDS_zh-CN.md)，
完整参数和产品解释见[中文用户指南](docs/USER_GUIDE_zh-CN.md)。

## 目录

1. [版本与定位](#1-版本与定位)
2. [主要功能](#2-主要功能)
3. [计算流程与模型](#3-计算流程与模型)
4. [输入与输出](#4-输入与输出)
5. [相对旧版的主要变化](#5-相对旧版的主要变化)
6. [运行环境与快速体验](#6-运行环境与快速体验)
7. [程序组织](#7-程序组织)
8. [运行与并行计算](#8-运行与并行计算)
9. [适用范围](#9-适用范围)
10. [文档与许可](#10-文档与许可)

安装并激活Python环境后，在包含 `input.hdf5` 的工作目录执行：

```bash
cess simulate -i input.hdf5 -o results --all-effects
```

默认处理完整输入文件，写出一份包含全部源行的光谱 HDF5；异常源保留行号并标记无效。
将 `input.hdf5` 换成输入目录即可处理目录内全部 HDF5，默认每个输入文件对应一份光谱结果。
默认 `-j 1` 按顺序处理；内存充足时，指定 `-j N` 可让最多 N 个进程同时处理不同文件，仍是一输入一输出。
`--all-effects` 同时打开四个科学开关；详见[快速上手](docs/QUICKSTART_zh-CN.md)。

## 1. 版本与定位

| 项目 | 当前状态 |
| --- | --- |
| 软件版本 | `0.9.0`：继承并重构一维模拟与处理流程 |
| 主要输入 | 既有 BayeSED HDF5 星系模板库、JSON 配置和相应资源文件 |
| 主要输出 | GU/GV/GI 一维光谱 HDF5、可选测光与发射线结果、运行清单 |
| Python 支持范围 | `>=3.12,<3.13`，在所用平台创建独立环境 |
| 当前现代输出 | 分组式 HDF5 schema 3.0，可选择兼容 schema 2.5 |

CESS 的主要用途包括星系样本可观测性评估、谱信噪比分布研究、发射线检测与测量方法测试，
以及波长定标误差对一维结果影响的对照实验。模型参数与资源应按具体研究目的核实。
软件本身不执行 BayeSED 拟合，也不自动完成星系红移测量或科学样本的完备度定标。

## 2. 主要功能

| 功能 | 可以完成的任务 | 关键说明 |
| --- | --- | --- |
| 一维光谱模拟 | 从星系 SED 生成 GU/GV/GI 模型谱、含噪谱和误差 | 保留现有 BayeSED 输入接口 |
| 仪器响应与探测器噪声 | 折叠 throughput，计算电子期望，加入天空、暗电流与读出噪声 | 明确电子率、电子计数、流量和误差的区别 |
| 形态展宽 | 根据星系形态参数及预存形态资源模拟谱展宽 | 形态参数来自现有经验关系和抽样，并非直接导入任意真实二维形态图 |
| 波长定标误差 | 在同一标称像元网格上对谱进行位置重采样 | 传播方差、有效掩膜和相邻像元协方差 |
| `paired-wce` 对照 | 同时生成无 WCE 和有 WCE 的配对光谱 | 两分支共享噪声，适合比较系统误差影响 |
| 七波段测光 | 计算 NUV/u/g/r/i/z/y 响应加权测光及其观测量 | 分别保存模型值、含噪观测值、误差、S/N 和状态 |
| 零级积分测光 | standard 模式计算 GU/GV/GI 零级通带的积分观测量 | 与旧 `_0th` 一维谱诊断数组分开定义 |
| 发射线分析 | 检测峰、真值匹配、漏检/误报诊断，以及窗口线通量和 S/N | 主要面向 GV/GI；候选峰不自动等同于已确认的原子跃迁 |
| 既有产品后处理 | 对 CESS 光谱运行独立 `el-detect` | 支持读取 2.5 / 3.0 产品，继承源行和父产品定义 |
| 批量调度 | 文件、目录、glob、源行区间、chunk 和有界多进程 | 不需要手工粘贴输入文件列表 |
| 可复现与续跑 | 显式种子、逐源随机流、来源记录、严格 manifest 校验 | 已存在产品不会被自动覆盖 |
| 结果读取与展示 | 统一 reader、光谱展示、分层抽样、子集导出和分片合并 | 参数按字段名和单位读取；合并无需重新模拟 |
| HDF5 桌面浏览器 | 结构/数据浏览、逐源光谱、X-Y绘图和CSV/图片导出 | 可选Tk界面，支持中英文，输入文件只读 |
| 进阶形态工具 | 正值 MGE 库生成、核验，以及可选联合 strip 一维提取 | 使用前需显式声明孔径、PSF/LSF 等约定 |

现代 `simulate` 的四个科学开关 **默认均关闭**：`--morphology`、`--wavelength-error`、
`--photometry`、`--emission-lines`。需要哪些效应，应在命令中明确启用；
`standard` 是默认模式，`--seed` 默认值为 0。

## 3. 计算流程与模型

### 3.1 从输入 SED 到一维产品

```text
BayeSED HDF5 + JSON 配置 + throughput/背景/形态资源
    ↓ 选择文件与原始源行，校验字段和单位
rest 波长 → 观测波长；光谱平滑与像元积分
    ↓ 折叠原生响应曲线，计算模型源电子率
星系形态选择、色散方向展宽与空间提取约定
    ↓ 源 + 天空 + 暗电流 + 读出噪声
模型/含噪光谱、误差、S/N 和掩膜
    ↓ 可选波长重采样及协方差传播
可选测光、零级积分、发射线检测与窗口测量
    ↓ 原子发布、记录 provenance 和更新 manifest
版本化 HDF5 产品 + JSON 任务记录
```

输入波长由 μm 转换为 Å，并按输入红移映射到观测系。输入 SED 的流量归一化沿用现有
BayeSED 库约定；不要把任意本征光度谱直接当作 μJy 输入。源响应采用波长积分，
不能用最终输出像元中心的一个 throughput 数值代替所有上游响应计算。

### 3.2 噪声、波长误差和信噪比

噪声包含源和背景光子的 Poisson 涨落、暗电流以及 Gaussian 读出噪声。
空间抽取像元数和曝光次数参与相应噪声面积的计算。扣除平均背景后的含噪信号允许为负，
保留低信噪比情况下正常的负涨落。

WCE（wavelength calibration error）目前通过共同像元网格上的线性重采样实现，
不是一套完整的仪器波长解拟合。重采样使用平方权重传播方差，并记录当前模型产生的
相邻像元 covariance。需要积分多个像元时，应使用误差和可用协方差；
不能始终按像元独立假设直接累加方差。

`paired-wce` 分支共享同一次噪声实现，不能视为两次独立曝光直接相加。
像元 S/N、波段平均 S/N、旧峰附近平均 S/N，以及窗口积分线 S/N 是不同定义，
科学分析时应选择对应字段。

### 3.3 形态和空间抽取

默认 `spectral_extraction.mode=legacy_full` 继承历史全源光近似。
`explicit` 允许显式提供源光比例和空间像元数；`mge_strip` 使用经过核验的正值
Gaussian 源分量计算联合空间 strip 和色散核，记录孔径源光比例、有限核质量及来源。

这几种模式不能通过简单重命名混用。MGE 模式需要配套库和明确的 PSF/LSF 角色，
不因选择该模式就自动获得真实 CSST PSF 定标。
详细定义见 [形态与 MGE 模型](docs/MGE_MODELS_zh-CN.md)。

## 4. 输入与输出

### 4.1 输入契约

CESS 接受的是符合现有 BayeSED 约定的 HDF5，而不是任意结构的 HDF5 文件。

| 输入 | 内容与单位 |
| --- | --- |
| `ID` | 长度 N 的源标识；结合原始行索引追踪来源 |
| `parameters` / `parameters_name` | N×K 参数矩阵及字段名，含 RA/DEC、G/R/Z 星等、质量和红移等所需列 |
| `best_fit/wavelength_rest` | 长度 W 的共同 rest 波长，现有约定为 μm |
| `best_fit/z` | 长度 N 的红移 |
| `best_fit/spec_*` | N×W 的 SED fν，现有约定为 μJy；用于模拟的模型数据集应明确 |
| 外部 JSON | 输入输出路径、仪器/曝光/背景配置、资源路径及可选模型定义 |
| 配套资源 | 光谱/测光效率、背景模板、GEMS 形态参考及形态库等 |

参数的精确列名、最小结构及配置路径规则见用户指南。更换模板库时需要检查波长覆盖、
源轴对齐、单位和参数语义；文件能打开不代表输入契约已经满足。

### 4.2 HDF5 schema 3.0

```text
product.hdf5
├── catalog/       # ID、原始行、参数表、提取及形态诊断
├── spectra/       # GU、GV、GI 的一维数组与共享网格
├── photometry/    # 可选七波段测光
├── zero_order/    # standard 的真实零级积分测光
├── lines/         # 发射线候选、匹配及窗口测量
└── provenance/    # 输入、配置、版本、源码/资源身份与布局映射
```

六个根组都有结构定义；某项未启用或不适用时以状态元数据表达，不能仅凭组存在就认为有测量。
常见光谱数组为 `(source,pixel)`，波长为 Å，流量/误差通常为 μJy；
GU/GV/GI 的像元数量可以不同。参数表按字段名、`axis_order` 和单位解读。

| 常见字段 | 含义与使用注意 |
| --- | --- |
| `wave` | 当前波段共同观测波长网格 |
| `flux_ujy` | 未折叠 throughput/形态之前的 photon-equivalent bin fν |
| `response_flux_ujy` | 已折叠响应的通量定义，不应重复乘 throughput |
| `source_electron_rate_pre_morph` | 形态处理之前的源电子率，按字段单位使用 |
| `flux_elec` | 无噪模型电子期望 |
| `flux_ujy_model` / `flux_ujy_with_noise` | 按名义响应标尺表示的模型谱与含噪谱 |
| `ferr` / `snr` | 对应光谱的误差与像元 S/N |
| `spec_mask` / `snr_mask` | 对应定义下的有效性标记 |
| `covariance_lag1` | 当前 WCE 线性算子下的谱内相邻像元协方差 |

可选模型下提取的谱没有自动除以孔径光比例校回全源光。字段可用性随模式与开关变化。
历史 `_0th` 一维数组仍是一级 grid 上的诊断 proxy；真实零级通带积分保存在 `zero_order`，
不能把两者当作相同物理产品。

`--output-layout legacy` 可输出 schema 2.5；CLI 和 `simulate_file(...)` 默认使用 3.0。统一 reader 可读两种布局；
旧文件转换另写新文件，不修改原件。抽样导出有自己的 schema，不能冒充完整模拟产品。

## 5. 相对旧版的主要变化

本次 v0.9.0 重构以继承原有一维能力为前提，重点改善可维护性、运行接口和结果可追溯性。

| 方向 | 重构后的做法 |
| --- | --- |
| 程序组织 | 从大量脚本内重复逻辑迁移为 `src/cess` 包，按光谱、探测器、形态、测光、发射线、I/O 和流程分类 |
| 配置管理 | 显式加载配置与资源，外部 JSON 相对路径以配置文件位置为基准，减少工作目录耦合 |
| 任务入口 | 统一 `cess simulate` / `cess el-detect` / `cess merge`，支持目录、glob和显式分片 |
| 多进程 | 有界 worker pool、完整等待和失败状态记录，避免无上限创建子进程 |
| 数据完整性 | 修复无效源参数复用、WCE 结果长度错位、空发射线结果和测光参数维度等问题 |
| 数值处理 | 改进响应/像元积分、读出噪声与负涨落处理，传播重采样误差和协方差 |
| 测光与线测量 | 响应加权测光、区分模型和观测值，保留误报/漏检以及窗口测量定义 |
| 结果格式 | 分组式版本化 HDF5、命名字段与统一 reader，减少固定列号和重复数组依赖 |
| 性能与重复代码 | 资源缓存、有界形态统计批处理，共用准备、写出和结果收集，减少重复计算和维护成本 |
| 可复现性 | 分组件的逐源随机流、输入/源码/资源哈希、明确的配置和严格续跑身份 |
| 环境 | 从旧 `grizli` / Python3.7.11 迁移到独立 Python3.12 环境，按核心/工具/绘图/渲染拆分依赖 |

故障修复和数值模型改进可能使结果与 v0.8.5 不同。

## 6. 运行环境与快速体验

使用独立的 Python3.12 环境。已有 Conda 的新用户可依次执行；
`/path/to/CESS` 须替换为已下载或解压的源码根目录：

```sh
conda create -n cess python=3.12 pip
conda activate cess
cd /path/to/CESS
python -m pip install '.[tools,plot]'
python -m pip check
python -c 'import sys, cess; print(sys.executable, cess.__version__, cess.__file__)'
cess --help
```

安装命令自动安装 CESS 及核心、绘图和抽样依赖。以后每个新终端执行 `conda activate cess` 即可。
已有可用环境无需重建；也可以选择 venv，详见[安装指南](docs/INSTALL_zh-CN.md)。
这里安装的是已经下载或解压的源码，不要求同名软件已经上传到 PyPI。
仅需核心模拟时安装 `.`；从 wheel/sdist 安装、可选依赖和版本升级见安装指南。
核心依赖为 NumPy、SciPy、Astropy、h5py、Specutils 和 tqdm；精确版本见
[Python3.12 核心依赖](requirements/core-py312.txt)。pandas 用于抽样工具，Matplotlib 用于展示，
GalSim 用于部分旧形态渲染/建库工具；常规预存库一维模拟不要求加载 GalSim。

仓库已提供可直接下载的 [toy 输入星表](examples/toy_input.hdf5)和[对应模拟结果](examples/toy_output.hdf5)，
可以直接使用输入运行模拟，或用浏览器查看结果，无需先生成示例文件。
文件说明和运行方法见[示例指南](examples/README_zh-CN.md)。

以下命令接上面的安装步骤，在已激活环境和仓库根目录执行；示例文件均相对该目录。
首次运行选择尚未使用的新文件名与输出目录：

```sh
python examples/make_toy_input.py toy.hdf5 --sources 4 --invalid-first
cess simulate -i toy.hdf5 -o toy_results --manifest toy_results/run.json \
  --all-effects --seed 1214 -j 1
```

toy 数据只检验接口，连续谱、发射线和星系参数为任意设定，不用于科学定标。
不指定 JSON 时使用包内默认资源；真实任务应明确确认所用配置。重复计算不会覆盖已有输出；
续跑需保留原参数并加 `--resume`，全新实验则选择新的 output 和 manifest。

安装后的 `cess` 与 `python -m cess` 提供同一现代入口。
安装后可在任意工作目录调用 `cess`，相对输入/输出路径以当前目录为基准。
命令中的 `\` 用于换行，必须是该行最后一个字符；路径包含空格时应加引号。
新环境安装、真实库批量处理、结果读取、抽样和服务器步骤见[Bash 命令指南](docs/COMMANDS_zh-CN.md)与中文用户指南。

### 可选 HDF5 图形浏览器

`cess-viewer` 提供只读的结构/属性查看、逐源 GU/GV/GI 光谱、通用 X-Y 绘图和图片/CSV导出。
支持中英文界面；不参与模拟计算。需要桌面环境、Matplotlib及可用的Tcl/Tk。
在激活的环境中、源码根目录安装并启动：

```bash
python -m pip install '.[viewer]'
cess-viewer /path/to/product.hdf5 --lang zh
```

不带文件路径可打开空窗口后选文件。Conda缺少Tk时先运行 `conda install tk`，
再用 `python -m tkinter` 检查；无桌面的Gravity批计算不需要它。
完整说明见[HDF5浏览器指南](docs/VIEWER_zh-CN.md)。

## 7. 程序组织

```text
CESS/
├── src/cess/
│   ├── cli/            # 命令行、展示
│   ├── config/         # 配置加载与校验
│   ├── spectra/        # 单位、数值积分和光谱算子
│   ├── instrument/     # 背景、探测器噪声、波长误差
│   ├── morphology/     # 形态抽样、预存库、MGE及参考计算
│   ├── photometry/     # 响应加权测光
│   ├── lines/          # 峰检测、匹配和窗口测量
│   ├── pipeline/       # standard/paired/EL流程及公共结果整理
│   ├── io/             # HDF5产品、选择、原子写出、清单与抽样
│   └── data/           # 包内默认配置与仪器/形态资源
├── tests/              # 数值、契约、兼容性和工作流测试
├── requirements/       # 当前Python3.12依赖分层
├── examples/           # 可公开的小型接口示例
├── scripts/gravity/    # 服务器预检与PBS任务脚本
└── docs/               # 安装、使用、输出格式与模型定义
```

## 8. 运行与并行计算

默认每个输入文件对应一个任务及一份光谱 HDF5。CESS 使用单机多进程处理不同输入文件或
显式设置的源行分片；`-j` / `--workers` 指定计算进程数，默认1，不会自动拆分单个文件。
单个文件未分片时只有一个任务，提高worker数也不能让多个进程同时处理该任务。

例如，在内存足够容纳两个完整输入的主机上，可从任意工作目录运行：

```bash
cess simulate -i /path/to/catalogues -o /path/to/new_results --all-effects -j 2
```

每个worker完成一个文件后领取下一个文件。运行前会显示输入数、任务数、请求与规划worker上限和
每worker线程设置，任务开始/完成记录包含PID与耗时；省略 `-j` 时会提示多个任务将顺序执行。

内存受限或希望并行处理单个文件时，可显式分片，再将结果合成一份文件。
下例在包含 `input.hdf5` 的工作目录执行，`chunks` 和 `merged` 均相对此目录：

```bash
cess simulate -i input.hdf5 -o chunks --all-effects --chunk-size 100 -j 2
cess merge -i chunks -o merged/complete.hdf5 --input-source input.hdf5 --dry-run
cess merge -i chunks -o merged/complete.hdf5 --input-source input.hdf5
```

`--chunk-size 100` 指定每个任务最多处理100个原始源行，仍会处理全部输入。
`merge` 按原输入行序真正合并 HDF5，使用有限内存流式写出，无需重算；合并文件自包含，
不依赖原分片。可选 `--input-source` 校验是否覆盖指定输入的全部行，`--dry-run` 仅预检。
分片和合并文件应放在不同目录，避免将合并结果混入待合并输入；各输入文件分别合并。

实际并发受worker上限、可执行任务数及可选内存预算共同限制。现代CLI默认
`--threads-per-worker 1`，在spawn进程启动前设置常见数值库的线程环境，计算结束后恢复父进程环境。
这是数值库线程请求，不是物理CPU绑定，也不保证整个进程树只占用指定数量的核。
先用小样本和 `--dry-run` 查看任务内存估计，再选择worker数；大文件同时运行时内存会累加，
不能只按本机CPU核数设置并发。CESS不会自动用满所有CPU，`--memory-budget-gib` 是保守规划约束，
不是实时内存硬限。

Gravity上需要分别设置调度器申请的CPU数，以及CESS实际启动的worker数。
现有PBS脚本会将常见数值库线程限制为1，并检查workers不超过用户声明的CPU上限；
输入可为目录，未设 `CESS_CHUNK_SIZE` 时按完整文件调度；具体设置见[Gravity说明](scripts/gravity/README_zh-CN.md)。
CPU和内存资源仍需通过服务器实际支持的PBS指令申请。当前并行方式在一个主机上运行，
不提供MPI跨节点分发。具体本地和服务器例子见[用户指南的CPU设置](docs/USER_GUIDE_zh-CN.md#61-cpu进程数与线程设置)。

仪器背景、PSF、灵敏度和科学选择函数仍应针对研究用途进行定标与评估。

## 9. 适用范围

目前适合在明确近似下进行快速一维样本模拟。使用时尤其应确认：

- 输入模板的 LSF 默认保持 `unknown`。已知标称分辨率并不能自动确定其线展宽形状，需根据模板库元数据设置。
- 当前上游 Gaussian 光谱 FWHM 由 `*_dlambda` 控制，`*_Res` 数字不参与该算子；
  不应仅修改 `*_Res` 就认为完成了分辨率重配置，也不应把 `*_dlambda` 直接等同于最终输出采样间隔。
- 默认背景采用固定模板；`background_v_mag` 在该模式中是来源标签，不能单独改变它来重标定背景。
  当前没有按观测时间和太阳相对几何标定的黄道光模型。
- 延展源形态、有限孔径、PSF、模板LSF与仪器LSF需采用一致定义。MGE/strip 模式仍含
  空间分布与SED可分、无色梯度及指定PSF形式等近似。
- 当前快速一维流程不包含整场二维源重叠污染、CCD边缘投影及真实二维提取过程。

## 10. 文档与许可

- [快速开始](docs/QUICKSTART_zh-CN.md)：安装后的最短操作流程。
- [Bash 命令指南](docs/COMMANDS_zh-CN.md)：路径、引号、单文件/目录、并行、分片合并、日志与续跑。
- [中文用户指南](docs/USER_GUIDE_zh-CN.md)：环境、输入、配置、运行、读取与常见问题。
- [输出物理量](docs/USER_GUIDE_zh-CN.md#9-读取产品及物理量)：字段语义与使用边界。
- [输出布局说明](docs/OUTPUT_LAYOUT_zh-CN.md)：schema3.0、统一reader与转换约定。
- [HDF5桌面浏览器](docs/VIEWER_zh-CN.md)：结构、逐源光谱、通用绘图和导出。
- [科学模型与适用边界](docs/SCIENCE_MODELS_zh-CN.md)及[形态/MGE 模型](docs/MGE_MODELS_zh-CN.md)。
- [Python3.12环境与复现](docs/INSTALL_zh-CN.md)。
- [Gravity操作说明](scripts/gravity/README_zh-CN.md)：用户预检、服务器资源设置和qsub。
- [版本变化](CHANGELOG_zh-CN.md)。
- [MIT 许可证](LICENSE)与[随包资源说明](RESOURCE_NOTICE_zh-CN.md)。

项目仓库：[RainW7/CESS](https://github.com/RainW7/CESS)。CESS 源码和文档采用 MIT 许可。
随包数据与依赖软件的说明见[资源说明](RESOURCE_NOTICE_zh-CN.md)。
