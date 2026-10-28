# CESS 一维模拟用户指南

[English](USER_GUIDE.md) | [简体中文](USER_GUIDE_zh-CN.md)

用户指南，2026-10-28。适用于 CESS `0.9.0`，科学 revision `20261027-D5`。

软件定位、功能总览和版本变化见[中文README](../README_zh-CN.md)。本页按实际操作顺序展开。
首次使用命令行，先阅读[Bash 命令手册](COMMANDS_zh-CN.md)：其中逐项解释路径、引号、换行、并行、日志和续跑。只需要最短入口时可先看[快速上手](QUICKSTART_zh-CN.md)。

| 目标 | 本指南位置 |
| --- | --- |
| 创建环境、安装、检查解释器 | 第2节；详细安装方案见[安装指南](INSTALL_zh-CN.md) |
| 单文件、目录、并行或源行子集 | 第6节；可复制的完整命令见[命令手册](COMMANDS_zh-CN.md) |
| 查看仪器参考、LSF与波长网格 | 第5节及[仪器模型指南](INSTRUMENT_MODEL_251106917_zh-CN.md) |
| 断点续跑和预检 | 第7节 |
| 发射线后处理 | 第8节 |
| 读取 HDF5、展示和抽样 | 第9–10节 |
| 科学模型、形态库和 Gravity | 第12节及其链接 |

## 1. 从什么输入得到什么结果

CESS 把已有 BayeSED 星系模板库和仪器配置转换为 GU/GV/GI 一维无缝光谱。它计算光谱响应、电子期望和观测噪声，输出标称流量、误差、S/N、掩膜及来源记录；可以另外启用形态影响、波长定标误差、七波段测光和发射线检测。源形态还使用配置指定的 GEMS/形态库等资源，单独一个 SED 文件不足以替代完整运行配置。

现代 `cess` / `python -m cess` 默认写分组式 HDF5 schema 3.0；`--output-layout legacy` 写兼容 schema 2.5。软件版本、修复 revision 和文件 schema 是不同编号，不能用其中一个推断另两个。

## 2. 创建 cess 环境、安装及运行

当前包要求 Python >=3.12,<3.13。本文 `/path/to/...` 和 `/absolute/...` 均为通用占位路径，
执行前替换为自己的实际路径。`/path/to/CESS` 是含 `setup.cfg`、`pyproject.toml` 的源码根目录；
相对数据路径以命令执行时的工作目录为基准。

### 2.1 首次安装：Conda 环境

已有 Conda 的用户可创建一个名为 `cess` 的独立环境，再安装 CESS。若该环境已存在且Python版本合适，直接激活即可：

```bash
conda create -n cess python=3.12 pip
conda activate cess
cd /path/to/CESS
python -m pip install '.[tools,plot]'
python -m pip check
python -c 'import sys, cess; print(sys.executable, cess.__version__, cess.__file__)'
cess --version
cess --help
```

`conda create` 创建环境，安装后它会出现在 `conda env list` 中。
`python -m pip install '.[tools,plot]'` 自动安装 CESS、核心依赖，以及抽样和绘图包；
无需手工逐个安装 NumPy、Astropy 等依赖。保留引号以适用于 zsh。
`pip check` 应无依赖冲突，打印的解释器和 CESS 路径应属于所选环境。
这些检查确认安装和入口，不代表科学模型已经定标。

仅需一维核心可安装 `python -m pip install .`。`tools` 包含 pandas，`plot` 包含 Matplotlib；
可选 `render` 的 GalSim 用于部分旧渲染/库生成工具，常规预存库一维模拟不必安装它。
完整依赖、开发测试安装和平台边界见[安装指南](INSTALL_zh-CN.md)。

### 2.2 再次使用与 venv 方案

每个新终端只需激活已创建的 Conda 环境，之后可从自己的数据工作目录运行：

```bash
conda activate cess
cd /path/to/data
cess simulate -i input.hdf5 -o results --all-effects
```

此处 `input.hdf5` 和 `results` 都相对刚进入的 `/path/to/data` 目录。
默认处理整个文件，并写出一份包含全部源行的光谱 HDF5；异常源保留原行号并标记无效。
若要处理目录内所有输入文件，将 `-i input.hdf5` 改为 `-i input_directory`；
默认每个输入文件对应一份光谱结果。`--all-effects` 同时启用四个科学开关。
默认只启用1个计算进程，目录文件会依次处理。内存足够时增加 `-j N`，即可同时处理最多N个文件，
每个文件仍写出一份完整结果。

已有可用的环境无需重建。不使用 Conda 时，可按[venv 安装步骤](INSTALL_zh-CN.md#12-使用-python-venv)
创建独立环境；新终端执行 `source /path/to/cess-env/bin/activate`。
venv 不出现在 `conda env list` 中。两种环境方案任选一种，退出分别用 `conda deactivate` 和 `deactivate`。
不要将 Mac 环境目录或 ARM64 lock 复制到 Linux 后当作可用环境。

### 2.3 本指南后续示例的准备

下文默认使用已安装的 `cess` 命令。为运行仓库内 toy 脚本和展示示例，在已激活的环境中设置：

```bash
REPO=/path/to/CESS
PY="$(python -c 'import sys; print(sys.executable)')"
"$PY" --version
cess simulate --help
```

`REPO` 是实际源码根目录，`PY` 指向当前激活环境的解释器。这些普通变量仅在当前终端有效。
源码开发时可将 `cess` 替换为 `env PYTHONPATH="$REPO/src" "$PY" -s -B -m cess`；
解释器仍须有对应依赖。当前 Mac ARM64 [平台锁](migration_py312_20261023/requirements-macos-arm64-py312.lock)
只记录该平台依赖，其他平台独立安装和验收。

## 3. 首次运行：公开 toy 示例

toy 是任意设定的连续谱、发射线和星系参数，只检查接口；它不能代表真实星系总体或用于科学定标。下面创建一个全新的临时工作目录，并生成4个源，其中第一个源故意无效，用于观察无效行的保留行为。请在同一终端顺序执行；`unset CESS_CONFIG` 使本例使用包内默认配置，若此前设置过自定义值，应先自行记录。

```bash
CESS_GUIDE_RUN=$(mktemp -d "${TMPDIR:-/tmp}/cess-guide.XXXXXX")
cd "$CESS_GUIDE_RUN"
unset CESS_CONFIG

"$PY" -s -B "$REPO/examples/make_toy_input.py" \
  "$CESS_GUIDE_RUN/toy.hdf5" --sources 4 --invalid-first

# 先预览：不带manifest，不创建科学产品或新清单。
cess simulate --input "$CESS_GUIDE_RUN/toy.hdf5" \
  --output "$CESS_GUIDE_RUN/spectra" --seed 1214 --workers 1 --dry-run \
  --morphology --wavelength-error --photometry --emission-lines

# 再实际执行：首次使用全新的output和manifest。
cess simulate --input "$CESS_GUIDE_RUN/toy.hdf5" \
  --output "$CESS_GUIDE_RUN/spectra" --manifest "$CESS_GUIDE_RUN/run.json" \
  --seed 1214 --workers 1 \
  --morphology --wavelength-error --photometry --emission-lines
```

此例 standard 产品为 `"$CESS_GUIDE_RUN/spectra/CSST_grism_toy_4.hdf5"`，任务状态保存在 `run.json`。无效源仍占原来的行位置，相关数值和掩膜表示不可用；不要删除该行后再用原 ID 或行号对照。若选择真实样本，先做少量源，检查输出定义和掩膜，再扩大计算规模。

### 输入异常源的处理

模拟前批量检查每个源实际用于计算的 `RA`、`DEC`、`MAG_Z`、`log(M*)[0,1]_{MAL}`、`z_{MAL}`，以及 SED 和 `best_fit/z`。不修改 HDF5，不删除或重新排列行。非法源使用 `data_mask=2`、非零 `input_quality_flags`，模拟谱为 NaN，谱掩膜为 false；原始 ID、行号及传出的输入参数仍保留原值。

当前 `cess.input-quality.v1` 使用宽松的数值安全范围：RA 为 0–360 deg，DEC 为 −90–90 deg，MAG_Z 为 −100–100 AB mag，log10(M*/Msun) 为 −5–20，两种红移各为 1e-8–10；均含端点，并要求计算参数有限。这些是工程保护范围，不是星系选择条件，也不证明处在范围内的上游模型科学正确。不要求 `z_{MAL}` 与 `best_fit/z` 相等。MAG_G/R 仅原样传出，缺失或 `-999` 等值不作为废弃整条 SED 的理由。

`input_quality_flags` 是 uint32 位掩膜，可用按位与读取多个同时存在的原因：1=计算参数非有限，2=RA 越界，4=DEC 越界，8=MAG_Z 越界，16=对数质量越界，32=形态红移越界，64=SED 非有限/负值/超过已有 1e100 上限，128=光谱红移越界，256=三波段所需波长或 LSF 支持不足。字段位置在 schema3.0 的 `/catalog` 下，legacy2.5 则在根目录；属性保存版本、定义和完整策略。缺列、数组形状错误或共享波长不合法属于文件契约错误，仍会明确报错。

## 4. 真实 BayeSED 输入契约

输入必须符合当前 BayeSED-shaped HDF5 约定，不能把任意 HDF5 或二维光谱 FITS 直接传入。N为源数、K为参数列数、L为共同静止系波长采样数。

| HDF5 路径 | 形状 | 当前约定 |
| --- | --- | --- |
| `ID` | `(N,)` | 整数、bytes或字符串标识；保留原dtype/编码，不要求用ID代替原始行定位 |
| `parameters` | `(N,K)` | 星表参数表；列与下行名称一一对应 |
| `parameters_name` | `(K,)` | 名称可为bytes/字符串，大小写和特殊字符必须正确 |
| `best_fit/wavelength_rest` | `(L,)` | 静止系波长，单位μm |
| `best_fit/z` | `(N,)` | 用于谱波长红移变换的无量纲红移 |
| `best_fit/spec_*` | `(N,L)` | 输入fν，单位μJy；当前选取`best_fit`遍历中第一个以`spec_`开头的dataset |

当前 runner 使用的必需参数列为 `RA`、`DEC`、`MAG_G`、`MAG_R`、`MAG_Z`、`log(M*)[0,1]_{MAL}`、`z_{MAL}`。RA/DEC以度表示，星等沿用上游输入定义，质量列是上游星质量对数参数；请核对其以太阳质量为基准的约定。它们并非任意自定义别名。`best_fit/z` 和表内 `z_{MAL}` 分别被不同环节使用，不能因为名字相近就只保留一个。

观察波长由静止系μm转为Å后乘 `(1+z)`；SED沿用上游已完成观测归一化的输入契约，CESS不会在这里再自动补一个距离或红移流量因子。若导入的是别的代码生成的静止系光度密度，需要先完成独立单位和归一化适配。多个 `spec_*` 并存时应确认实际选中的模型，不能凭文件名假定。

源轴须与ID一致，波长轴须与谱长一致。源按原始行切片后再处理，三个光谱波段的有效性合并；不满足条件的源仍保留位置。只有命令规划成功，不足以证明全部数组、科学量或输入覆盖范围都正确。公开结构示例见[生成脚本](../examples/make_toy_input.py)，真实库的科学归一化仍由其来源负责。

## 5. 配置文件、资源与路径

配置优先级为显式 `--config`，其次非空环境变量 `CESS_CONFIG`，最后包内默认JSON。`--config` 本身若是相对路径，先相对调用时cwd定位；**外部JSON中的相对资源及运行目录，相对该JSON所在目录解析**。包内默认JSON是特例：仪器资源相对包内data目录，而其中四个输入/输出目录相对调用cwd，避免往site-packages写产品。CLI显式 `--input/--output/--manifest` 的相对路径仍相对调用cwd。

推荐生产任务把输入、输出、manifest和外部JSON都写成绝对路径。不要只把默认JSON复制到别处而不修改其相对资源路径；这样 `basic_file/...` 会转而在新JSON目录下查找。可以保留完整资源布局，或将资源字段改为真实绝对路径。资源不存在通常在实际使用时才报错，普通CLI dry-run不能替代资源读取验收。

外部配置应由完整模板修改，不要把几行演示JSON当作可独立运行配置。重要项包括：

| 参数/类别 | 定义与注意事项 |
| --- | --- |
| `radi`、`expt`、`expnum` | 口径半径cm、单次曝光s、曝光次数；不能把半径当直径 |
| `arcsecperpix` | arcsec/pixel；与所用形态库的尺度假设一致 |
| `dc_value`、`read_noise` | 暗电流e-/s/pixel；每次读出的单像元RMS读出噪声e- |
| `gu/gv/gi` 的波长字段 | 波长和线宽均用Å；在新采样模式中，`min_wl/max_wl`定义bin外边界，不是首末中心 |
| `instrument_lsf` | 随包为`central_resolving_power`、`wavelength_law=constant_fwhm`；按`*_central_wl / *_Res`推导目标Gaussian宽度 |
| `spectral_sampling` | 随包为`resolution_elements`、`pixels_per_resolution_element=4.0`；以实际输出几何为准 |
| `spectral_window` | 随包为`bandpass`，按实际波长推导分析窗口；此时`init_idx/end_idx`为推导缓存 |
| 背景和模板LSF | `fixed_template`天空；输入模板采用Gaussian R=300及`template_lsf_broader_policy=preserve`，结果取决于该输入库假设 |
| throughput、背景、GEMS、形态库 | 使用实际配置路径；七波段和零级资源按产品与开关读取 |
| `spectral_extraction` | 默认`legacy_full`，进阶模式见第12节 |

随包D5配置采用 [arXiv:2511.06917v2](https://arxiv.org/html/2511.06917#S2) 表1的中央波长和中央R值，按这两个字段推导目标Gaussian FWHM，并在各自波段内保持该宽度。有效宽度写入`*_dlambda`，仅修改这个推导缓存不会改变参考模型。论文没有证明各波段全带恒定R，也没有提供完整仪器LSF；参考值、采样与假设见[仪器模型指南](INSTRUMENT_MODEL_251106917_zh-CN.md)。

模板设置`{"mode": "resolving_power", "value": 300.0}`是为所用BayeSED库采用的Gaussian近似，不是仪器论文的测量结论。GU约3522.65 Å以上，模板已经比目标更宽；默认`template_lsf_broader_policy=preserve`保留已有宽度，并提示目标未达到。若必须在此情况下停止，可选择`error`，但卷积不能将它修复为更窄的谱线。其他输入库须相应设置`template_lsf`。

缺少新仪器/采样/窗口键的外部配置仍沿用早期`fixed_fwhm` / `legacy` / `indices`约定；在该仪器模式中，`*_dlambda`仍控制宽度，`*_Res`作描述。验证会记录有效配置，不会使任一约定自动成为仪器定标。

为方便审阅，当前包内默认JSON的部分数值如下；它们是**当前程序配置**，不是本文独立认证的CSST仪器规格。实际外部配置可以不同：

| 默认项 | 数值 |
| --- | --- |
| `radi` | 100 cm |
| 光谱`expt × expnum` | 150 s × 4 = 600 s |
| `arcsecperpix` | 0.074 arcsec/pixel |
| `dc_value` / `read_noise` | 0.02 e-/s/pixel / 5 e-/pixel/exposure RMS |
| GU/GV/GI 波长外边界 | 2550–4100 / 4000–6400 / 6200–10000 Å |
| GU/GV/GI 中央波长 / R | 3370 / 287；5250 / 232；8100 / 207，波长单位Å |
| GU/GV/GI 推导 `*_dlambda` | 11.74216028 / 22.62931034 / 39.13043478 Å；目标像元积分前Gaussian FWHM |
| GU/GV/GI 光谱bin数 | 529 / 425 / 389；保留精确外边界，中心取中点 |
| GU/GV/GI 实际像元宽度 | 2.93005671 / 5.64705882 / 9.76863753 Å |
| `background_scale` | 1.0，固定背景模板倍率 |

七波段测光和真实零级积分使用各自的`photometry_exposures`、`zero_order_exposures`，不能把所有产品都按上述600 s解释。
Throughput、天空、GEMS及形态库文件保留原有内容；论文中的R80不被替换为PSF FWHM。新网格也会改变按像元定义的线窗口与波长误差偏移所对应的波长跨度；保留分析参数不代表保留其以Å表示的宽度，也不代表检出阈值经过独立定标。

## 6. 模式、开关和批处理

| CLI选项 | 默认值及行为 |
| --- | --- |
| `--mode` | `standard`；另选`paired-wce` |
| `--seed` | **0**；接受非负整数。示例1214只是明确选择，不是默认值 |
| `--workers` | 1；有界进程数，实际不超过可运行任务数 |
| `--threads-per-worker` | 1；在spawn启动前请求常见数值库使用的线程数，不是CPU绑定 |
| `--source-start/--source-stop` | 起点0，终点默认源总数；左闭右开原始行区间 |
| `--chunk-size` | 默认不分片，每个输入的所选区间作为一个任务及一份光谱产品；设置后限制每个产品的源行数 |
| `--memory-budget-gib` | 默认不设置；按估计工作内存限制并发，不是操作系统硬内存上限 |
| 四个科学开关 | `--morphology`、`--wavelength-error`、`--photometry`、`--emission-lines`均默认关闭 |

`paired-wce` 将同一次噪声realization用于未移位/波长误差对照；两分支不是独立曝光。若希望考察WCE，应显式打开 `--wavelength-error`。七波段测光由 `--photometry` 控制；发射线后处理另有子命令。四开关是运行选择，不代表关闭时所有资源都不再需要，例如现有形态准备仍可能查库。

以下例子均从任意cwd调用，替换 `/absolute/...` 为真实路径：

```bash
# 真实库中原始第100到119行；先用新位置运行小样本。
cess simulate --config /absolute/runtime.json --input /absolute/catalogue.hdf5 \
  --output /absolute/new_subset --manifest /absolute/subset.json \
  --source-start 100 --source-stop 120 --chunk-size 10 --workers 1 --seed 1214 \
  --mode standard --morphology --wavelength-error --photometry --emission-lines

# 目录输入：假设内存足以容纳两个完整输入，两个worker按文件并发。
cess simulate -i /absolute/catalogues -o /absolute/new_results --all-effects -j 2

# glob须加引号，让CLI发现输入；这里仅预览显式分片的批处理计划。
cess simulate --config /absolute/runtime.json --input '/absolute/catalogues/*.hdf5' \
  --output /absolute/new_batch --chunk-size 100 --workers 2 --seed 1214 --dry-run

# 共享噪声的WCE对照，使用独立输出和清单。
cess simulate --config /absolute/runtime.json --input /absolute/catalogue.hdf5 \
  --output /absolute/new_paired --manifest /absolute/paired.json \
  --source-stop 4 --workers 1 --seed 1214 --mode paired-wce \
  --morphology --wavelength-error --photometry --emission-lines
```

输入还可为目录或多个文件；目录只发现其直接子级的 `.hdf5/.h5`。同一批两个输入若有相同stem，可能产生相同输出名，CLI会要求拆成独立运行。所有输入共享本次源区间，故短文件不能使用超过其源数的stop。生产建议每个输入文件一个manifest，避免大清单在每个任务结束时全量重写造成额外开销。

可复现随机流绑定seed、输入内容、原始源行、ID和随机分量；采用固定seed时，改变worker/chunk不应改变对应源的随机实现。但“得到同一随机源”与“允许沿用旧清单”是两件事：改变chunk会改变任务计划，续跑将拒绝。多worker也不保证线性加速，须同时考虑数值库线程、内存、I/O和任务大小。

默认未指定 `--source-start/--source-stop` 时处理全部源；`-j` 只控制已有任务的并发数，
不会自动拆分单个文件。若一个输入未显式分片，只有一个计算任务。希望保留一输入一输出时，
使用本节的目录命令即可；需要控制单任务内存或对单个文件并行时，再使用 `--chunk-size`。

### 分片结果合并为完整文件

例如，将一个输入每100行分片计算，再合成一份包含全部原始源行的 HDF5：

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/new_chunks \
  --all-effects --chunk-size 100 -j 2
cess merge -i /path/to/new_chunks -o /path/to/merged/complete.hdf5 \
  --input-source /path/to/data/input.hdf5 --dry-run
cess merge -i /path/to/new_chunks -o /path/to/merged/complete.hdf5 \
  --input-source /path/to/data/input.hdf5
```

`merge` 根据原输入行号排序，检查分片兼容性后真正写出合并数据，不重新模拟。
它采用有限内存的流式合并，最终 HDF5 自包含，不依赖原分片文件。
`--input-source input.hdf5` 是可选的整表覆盖检查；需要确认没有漏行时建议保留。
`--dry-run` 只做预检，移除此参数后才写出产品。
将分片目录与合并目录分开，不要把合并结果混入待合并输入。
一个目录中若包含多个原始输入的结果，应分别选择同一输入的分片进行合并。
`merge` 支持普通和 paired-wce 模拟产品的 schema2.5/3.0，不合并 EL-only 后处理产品。
波长、响应等共享数组必须一致，源行区间必须连续且不重叠。`--copy-rows` 只限制复制块的行数，
不是新的模拟分片；最终发射线完整性核验仍需读取峰表，内存随峰数量增长。
合并会保留分片的原始模拟身份和来源，不把它伪装成重新执行的原生模拟任务；不能用合并文件顶替原任务manifest的产品。

### 6.1 CPU进程数与线程设置

当前并行方式是在同一主机上启动多个Python计算进程。每个进程处理一个完整任务，任务结束后领取后续任务；父进程还负责调度、核验和清单更新。没有MPI跨节点分发，也不对CPU核做固定绑定。

| 设置 | 控制什么 | 不代表什么 |
| --- | --- | --- |
| `--workers N` | 最多N个计算进程；默认1，实际还受任务数及内存预算限制 | 不等于自动申请N个CPU，也不保证整个进程树恰好只占N个核 |
| `--threads-per-worker N` | 默认1；在spawn启动前设置常见数值库线程请求 | 不保证所有后端都遵从，也不将worker绑定到物理核 |
| `--chunk-size M` | 每个任务最多M个原始输入行，生成更多可独立运行的任务 | 不是线程数，也不是每个源都占一个CPU |
| `--memory-budget-gib` | 按估计单任务内存降低并发 | 不读取实时剩余内存，也不是操作系统硬内存限额 |

**文件级并行。**在已激活的CESS环境中，假设内存足以容纳两个完整输入，可以从任意工作目录运行：

```bash
cess simulate -i /absolute/catalogues -o /absolute/new_parallel \
  --all-effects -j 2 --threads-per-worker 1
```

每个worker处理一个完整文件，结束后领取下一个文件。现代CLI使用 `ProcessPoolExecutor` 的spawn进程，
在进程启动前设置OpenBLAS、OpenMP、MKL、Accelerate、NumExpr和BLIS的常见线程环境变量，默认均为1；
在进程池关闭后恢复父进程环境。通常无需再手工export；具体后端是否遵从取决于所用数值库。
一个worker是一个Python计算进程，不能严格等同于一个物理CPU核。

实际运行前的摘要显示输入数、任务数、请求workers与规划有效并发上限及每worker线程设置；任务开始与完成日志包含
PID、源区间和耗时。未指定 `-j` 时默认1；多个任务会提示按顺序执行。`--dry-run` 保留JSON规划输出，
可检查其中的 `workers`、`threads_per_worker` 和各任务的 `estimated_peak_gib`。

先用代表性小样本和内存规划确定并发。以当前默认模型的规划估计为例，约2.3万源的任务为8.5 GiB，
76,654源约27.8 GiB；这是估计值，不能当作实测RSS或操作系统硬上限。32GB机器同时运行大文件时，
不能只按CPU核数选择worker数。`--memory-budget-gib` 按最大任务估计保守降低并发；若一个任务也放不下，
应减小chunk或使用内存足够的主机。若一个文件有1000行、chunk100，则有10个任务；
单个文件未分片时只有一个任务，即使设置8 workers也只运行一个计算进程。

**Gravity运行。**大量独立星表可直接使用 [Torque 作业数组](../scripts/gravity/ARRAY_zh-CN.md)：每文件一个 1 CPU 子作业，MzLS 和 DECaLS 分别申请内存，每个文件保存独立恢复清单。准备、完整文件试跑和全量提交见该指南。
自定义单节点批次仍可使用 `cess.pbs`；先通过PBS申请同一计算主机的CPU和内存，再设置CESS进程数。这是两个独立步骤。
假设已分配8个CPU，且内存足以容纳所选任务，可以设置：

```bash
# 在提交终端设置；其余绝对路径和科学开关按Gravity说明配置。
export CESS_INPUT=/path/to/catalogues
export CESS_CPUS=8
export CESS_WORKERS=8
unset CESS_CHUNK_SIZE  # 每个输入文件一个完整任务；设正整数才启用分片
```

`CESS_CPUS`只是脚本用于检查`CESS_WORKERS <= CESS_CPUS`的用户声明，**不会申请或自动核验PBS实际分配的CPU数量**。例如申请8核却只设`CESS_WORKERS=1`仍只启动一个计算worker；只设`CESS_CPUS=8`却未申请8核，也不意味着获得了8核资源。线程限为1时，8个忙碌worker通常对应约8路计算并发，但父进程和其他辅助工作仍会产生额外开销。

基础 `cess.pbs` 没有预设资源指令。Gravity 的[官方作业说明](https://jupyter.gravity.sjtu.edu.cn/doc/#/Basic/Job)使用 Torque，单主机8 CPU的申请示意为：

```text
#PBS -q normal
#PBS -l nodes=1:ppn=8
#PBS -l mem=<requested_memory>
#PBS -l walltime=12:00:00
```

`<requested_memory>` 必须替换为根据任务估算、本站允许的实际资源量；12小时也只是格式示例，
不是全量任务保证。不要使用 `select=...` 的 PBS Pro/OpenPBS 写法。队列当前限制仍以 `qstat -q`、`qstat -f -Q` 和实际账户权限为准。

当前脚本接受单个HDF5或输入目录，固定每worker数值库线程请求为1；还可设置可选的
`CESS_MEMORY_BUDGET_GIB` 将保守内存预算传给CLI，具体见[Gravity说明](../scripts/gravity/README_zh-CN.md)。
在设置好所有CESS路径、开关和资源后，先运行 `preflight.sh --dry-run`，再按本站方式 `qsub`。
没有申请多节点后自动把workers分散到各节点的机制，应为单个CESS作业申请一个主机；
也可按独立输入或批次提交不同作业，各用独立输出和manifest。

## 7. dry-run与严格续跑

科学revision D5改变了LSF/网格配置。保留旧产品，D5计算应选择新输出与manifest路径，不能用`--resume`继续旧科学revision；Gravity批次也须重新prepare一个新运行目录。随机种子相同不代表改变物理设置后仍得到旧光谱。

三种预览有不同文件行为：不带manifest的新任务 `--dry-run` 只打印规划；新任务同时带 `--manifest` 会创建任务清单，并可能触及父目录/锁文件，但不计算科学产品；已有任务 `--resume --dry-run --manifest ...` 走严格只读核验，不创建新输出或改写旧清单。若已做过第二种预览，后续实际执行应加 `--resume`，而不是再次把该清单当新任务。

续跑前先确认旧worker已停止。对第3节toy，重复原参数并增加resume即可：

```bash
cess simulate --input "$CESS_GUIDE_RUN/toy.hdf5" \
  --output "$CESS_GUIDE_RUN/spectra" --manifest "$CESS_GUIDE_RUN/run.json" \
  --seed 1214 --workers 1 \
  --morphology --wavelength-error --photometry --emission-lines --resume --dry-run

# 核验通过后移除 --dry-run，保留 --resume，即执行剩余任务。
```

严格身份包括输入路径/内容、完整有效配置、开关、seed、模式、源码、资源、布局、输出目录及源区间/任务分片。完整产品会按结构、语义和校验信息核验后跳过；损坏或漂移会报错，不能靠删manifest假装为同一任务。没有manifest就不能使用resume；也不能把旧revision清单拿来续跑新代码。

**可以调整workers、`--threads-per-worker` 及 `--memory-budget-gib`**：这些调度设置没有放入snapshot。
调整后任务身份仍须完全一致，且新内存预算能容纳一个任务。原随机流不随worker数改变，
不同数值库/线程配置仍可能存在浮点舍入差异。不能同时改chunk、选源区间、路径、开关或科学配置；
若要改变这些，应使用新的输出目录和manifest。预览只规划/核验，不是科学正确性验收。

## 8. 发射线后处理

`el-detect` 读取已有CESS谱，不需要重算光谱。它支持旧/新布局父产品，默认输出3.0，保留原catalogue行与直接父产品行的映射；父产品有WCE分支时还处理相应诊断。产品保存线测量和来源，不复制全部父光谱；因此不能把它当完整光谱产品交给所有展示工具。

```bash
cess el-detect --input "$CESS_GUIDE_RUN/spectra/CSST_grism_toy_4.hdf5" \
  --output "$CESS_GUIDE_RUN/lines" --manifest "$CESS_GUIDE_RUN/lines.json" \
  --workers 1 --seed 1214
```

真实任务可增加 `--config /absolute/runtime.json`、源行区间和分片选项。这里的选源索引是父产品内行号，原星表行号由映射保留；父产品本身若是切片，两者未必相同。缺少父误差或认证协方差时，程序不能凭空补齐有效线S/N，应读取对应质量标记。

峰检测、匹配、漏检、假阳性和积分窗口测量是不同信息。`detect_el_snr` 是历史局部平均像元S/N，窗口积分flux/error/SNR位于线测量结构。它们不自动等于已分解的原子线总通量或已确认线身份，不能直接用一个历史列阈值替代科学选择函数。

## 9. 读取产品及物理量

schema3.0主要根组为 `catalog`、`spectra`、`photometry`、`zero_order`、`lines`、`provenance`。组可能处于关闭或不适用状态，存在组名不意味着其中一定有有效测量。使用统一reader可减少对2.5/3.0磁盘位置的依赖；离开with后不要再使用尚未取出的HDF5 dataset。

```python
from cess.io.products import open_product, read_product_summary

path = "/absolute/path/simulation.hdf5"
print(read_product_summary(path))
with open_product(path) as product:
    ids = product.catalog["ID"][:]
    table = product.parameter_table(required=("RA", "DEC"))
    ra = table["RA"]
    gv = product.band("GV")
    wave = gv["wave"][:]                         # Å
    flux = gv["flux_ujy_with_noise"][1]           # μJy，第2个源
    error = gv["ferr"][1]                        # μJy
    available = gv["spec_mask"][1]
    print(dict(gv["wave"].attrs))
    print(dict(product.provenance.attrs))
    row = product.read_row(1)
```

运行该Python代码同样要使用已安装环境，或设置 `PYTHONPATH=/path/to/CESS/src`。大产品宜按行读取；示例中的整列ID/RA按实际需要缩小。按字段名、单位和 `axis_order` 解释参数表，不硬编码某列序号；不要用变长峰表长度推断源数。重复ID、切片和EL子集都应结合行映射定位。

`flux_ujy` 是响应/形态之前的photon-equivalent bin fν；响应折叠量不能再乘一次throughput。`flux_elec` 表示无噪电子期望，观测谱和 `ferr` 使用标称响应标尺。含噪负值可以合法出现，不应统一截零；误差、mask和谱内协方差须共同解释。当前 `snr` 不是普遍适用的每Å或每分辨单元S/N。

历史 `*_0th` 谱数组只是把零级效率作用到一级色散网格上的proxy。真实零级通带积分位于 `zero_order`，也不是二维零级图像。standard和paired的适用产品不同，paired不能据空组被误认成已有真实零级测光。七波段测光还应区分model、observed、误差及状态；未提供、未检出和无效不是同一状态。

若需显式把标识为2.5的历史文件转换为3.0，可使用 `convert_product(source, destination)`；它写新目标，不修改源文件，也不重新计算科学数组。任意更老/无明确schema的历史文件不应被假定可直接转换。接口和布局详见[输出说明](OUTPUT_LAYOUT_zh-CN.md)。

## 10. 展示和分层抽样

展示需要plot依赖。toy的第0行无效，下面选择第1行；`--save` 保存PNG而不打开GUI。图中的输入模型需要源BayeSED文件仍可定位，迁移产品后可显式给 `--model-input`。

```bash
env PYTHONPATH="$REPO/src" "$PY" -s -B -m cess.cli.demo \
  "$CESS_GUIDE_RUN/spectra/CSST_grism_toy_4.hdf5" 1 emissionline \
  --model-input "$CESS_GUIDE_RUN/toy.hdf5" --save "$CESS_GUIDE_RUN/spectrum.png"

cess-extract \
  /absolute/simulation.hdf5 --output /absolute/new_sample.hdf5 \
  --target 100 --n-bins 10 --seed 1214
```

抽样先要求GV和GI的mean S/N均≥3，再按红移quantile分层按比例无放回抽取。`--n-bins` 控制红移分层数，不是S/N分箱；样本不足时不能强行获得目标数。它沿用历史选择规则，不是积分线S/N选择，也不保证科学完备性。抽样CLI自己的seed默认未设置，需显式指定以便复现；不要与模拟CLI默认0混淆。

导出格式标为 `extracted-mock-spectra-1.1`，保留可用误差、掩膜及来源，但不是完整模拟输出的等价副本。并非每个toy都会产生满足阈值的抽样源，首次无合格源时应检查选择条件，而不是调整数据使其必定入选。

### 交互式 HDF5 浏览

安装可选 `viewer` extra 和 Tcl/Tk 后，运行 `cess-viewer /path/to/product.hdf5 --lang zh`。
四页界面用于结构/属性和分页数据、逐源谱、X-Y/直方图及说明；可以按行号或ID选源，保存图和CSV。
同轴三波段叠画不会校正或合并波段；CSV不包含完整协方差、掩膜与provenance。
筛选后绘图数据按当前快照导出，稳健显示范围不会删除导出点。
paired-wce的光谱页显示基础分支，`*_wce` 可在结构或X-Y页查看。
安装、操作、大文件和排错详见[浏览器指南](VIEWER_zh-CN.md)。

## 11. 常见故障与排查顺序

| 现象 | 优先核对 |
| --- | --- |
| 找不到`cess`或导入了旧版本 | 打印`sys.executable`及`cess.__file__`；检查所选Conda或venv环境、安装入口与PYTHONPATH |
| `No input HDF5 files match` | 绝对路径、glob引号、目录直接子级及文件后缀；不能把示例占位路径原样执行 |
| 缺参数列/SED字段 | 对照第4节真实路径、列名大小写和数组轴；检查首个`spec_*`是否预期模型 |
| 找不到throughput/pickle | 确认JSON相对路径基准；复制JSON后是否同步资源或改为绝对路径 |
| 清单/产品已存在 | 区分新任务与原任务续跑；新配置用新位置，相同任务核验后resume |
| resume snapshot/plan不匹配 | 对照代码、输入、资源、配置、区间和chunk；不要手改清单绕过核验 |
| 内存预算容不下单任务 | 先减chunk，再考虑worker；估计不是RSS保证，仍须小样本实测 |
| 数据为NaN、线表为空 | 检查原谱/波段覆盖、源mask、开关、阈值及质量标记；不能直接判定程序错误 |
| 展示找不到原始模型 | 保留或重新指定BayeSED输入，使用实际`--model-input`路径 |

保留完整命令、cwd、版本、配置、日志和manifest，先区分路径/输入契约错误、实现错误和模型局限。不要为了得到预期图形删除异常源或替换未说明的物理默认值。若无法复现，先缩小到明确的源区间，再比较对应行的输入、输出字段与来源。

## 12. 进阶形态模型与Gravity

`mge_strip` 是显式选择的联合空间strip模型，需要经核验的positive source-basis bundle及明确的PSF/LSF角色。其源光比例F只进入源计数一次，空间像元数N决定背景/暗电流/读出噪声面积；有限色散窗口质量M不会自动归一化回1。关闭形态色散重分布不等于关闭该孔径模型的F/N。

请按[一维科学模型说明](SCIENCE_MODELS_zh-CN.md)、[MGE形态模型指南](MGE_MODELS_zh-CN.md)和[MGE库格式](MGE_BUNDLE_FORMAT.md)理解并合入完整配置。示例空间像元、核半宽及 `pre_psf` 都是待核验假设，不能直接认作CSST定标值。库像元尺度、PSF声明及配置必须一致，也不能混用旧pickle bundle与正值库格式。

大批星表按[Gravity作业数组指南](../scripts/gravity/ARRAY_zh-CN.md)运行；自定义单节点批次仍可按[Gravity概览](../scripts/gravity/README_zh-CN.md)使用`preflight.sh`和`cess.pbs`。两种方式均要求绝对路径、版本身份、受限worker与线程、独立日志及严格resume；新任务预检省略manifest以免生成清单。提交前须核对Gravity当前队列和账户限制，由用户自行`qsub`。D5应重新prepare新批次，不要修改旧运行的冻结配置；本地payload核验不代表已完成服务器资源或全星表验收。

源码和文档采用[MIT许可](../LICENSE)，随包资源范围和来源见[资源说明](../RESOURCE_NOTICE_zh-CN.md)及[资源清单](RESOURCE_MANIFEST.json)。

常用快捷命令见[快速上手](QUICKSTART_zh-CN.md)，Bash语法与逐项操作见[命令手册](COMMANDS_zh-CN.md)：`-i/-o/-j`分别等价`--input/--output/--workers`；`simulate --all-effects`同时启用四开关，原默认不变。
