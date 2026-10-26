# CESS 安装与运行环境

[English](INSTALL.md) | [简体中文](INSTALL_zh-CN.md)

适用于 CESS v0.9.0 一维版本。软件要求 Python >=3.12,<3.13。
请先取得本版源码或本地/发布页提供的 wheel、sdist；是否已在线发布以仓库实际 release 为准。
以下步骤从这些文件安装，不要求 PyPI 上已经发布同名软件。

Linux、macOS ARM64 和 x86_64 应分别使用对应平台的解释器与依赖，
不能复制 Mac 的 Conda/venv 目录作为服务器环境。

## 1. 从零创建环境并安装 CESS

先下载或解压 CESS 源码；`/path/to/CESS` 表示包含 `pyproject.toml` 和 `setup.cfg` 的仓库根目录。
所有 `/path/to/...` 均为占位路径，须替换成自己的实际位置。以下两种环境方案任选一种；
已有可用的 Python3.12 CESS 环境时直接激活即可，无需重复创建；升级软件仍需重新安装新版本。
依赖尚未安装时，pip 通常需要联网下载适用于当前平台的依赖。

### 1.1 使用 Conda 创建名为 cess 的环境

适用于已安装并能在终端使用 Conda 的用户。依次执行：

```bash
conda create -n cess python=3.12 pip
conda activate cess
cd /path/to/CESS
python -m pip install '.[tools,plot]'
```

`conda create` 创建环境；`python -m pip install` 把 CESS 和依赖安装到当前激活环境。
`.[tools,plot]` 会自动安装核心科学计算依赖、抽样用 pandas 和绘图用 Matplotlib，
无需逐个手工安装。方括号外保留引号，以适用于 zsh。
该 Conda 环境创建成功后会出现在 `conda env list` 中。

### 1.2 使用 Python venv

已有 `python3.12` 时也可使用独立 venv。先退出当前激活的 Conda 环境，
再执行以下步骤；`source` 激活方式适用于 macOS/Linux 的 Bash 或 zsh：

```bash
python3.12 --version
python3.12 -m venv --prompt cess /path/to/cess-env
source /path/to/cess-env/bin/activate
cd /path/to/CESS
python -m pip install '.[tools,plot]'
```

venv 使用路径管理，**不会出现在 `conda env list` 中**。它同样可以安装并运行 CESS。

### 1.3 检查安装并首次运行

完成任一方案后，在该环境中执行：

```bash
python --version
python -m pip check
python -c 'import sys, cess; print(sys.executable, cess.__version__, cess.__file__)'
cess --help
cess simulate --help
```

Python 版本应为 3.12.x；安装本版后 `cess.__version__` 应为 `0.9.0`。
`pip check` 应无依赖冲突，解释器与 CESS 包路径应属于所选环境。
帮助信息正常显示说明命令入口可用。接下来运行[公开 toy 示例](../examples/README_zh-CN.md)，
再用自己的星表模拟；安装检查不替代科学模型验证。

### 1.4 从 wheel 或源码压缩包安装

先按上述任一种方案激活 Python3.12 环境；以下使用绝对路径，可从任意工作目录执行。
将文件路径替换为实际下载位置，只选一种安装方式：

```bash
# wheel：安装 CESS 与核心依赖。
python -m pip install /path/to/dist/cess-0.9.0-py3-none-any.whl
```

需要绘图和抽样时，wheel 也可带 extras，完整路径必须放在引号内：

```bash
python -m pip install '/path/to/dist/cess-0.9.0-py3-none-any.whl[tools,plot]'
```

sdist 是源码压缩包，可解压后按源码目录安装，也可直接安装：

```bash
python -m pip install '/path/to/dist/cess-0.9.0.tar.gz[tools,plot]'
python -m pip check
```

wheel 无需手动解压。源码安装通常需要构建工具；pip 按 `pyproject.toml` 创建构建环境。
throughput、背景和形态资源随 CESS 包安装，常规一维运行无需手工复制到当前目录。
真实 BayeSED 星表不在软件包内，需要另行准备。

### 1.5 选择可选功能

从源码根目录安装时，按需求选择：

| 安装命令 | 用途 |
| --- | --- |
| `python -m pip install .` | 核心一维模拟、产品读取、分片合并及命令行 |
| `python -m pip install '.[tools,plot]'` | 核心加抽样/导出与绘图，适合常规交互使用 |
| `python -m pip install '.[render]'` | 额外安装 GalSim，供相关形态渲染/建库工具使用 |
| `python -m pip install '.[tools,plot,render,test]'` | 开发、完整测试与相关可选工具 |

常规预存库一维模拟不必安装 GalSim。当前 Python3.12 依赖以根目录包声明为准，
不使用 `CESS_v0.8.6` 中的历史环境文件。

### 1.6 安装可选桌面 viewer

在CESS源码根目录运行 `python -m pip install '.[viewer]'`，随后可用
`cess-viewer /path/to/product.hdf5 --lang zh` 打开文件。`viewer` 只增加Matplotlib，
NumPy/h5py使用核心依赖。它还需要图形桌面和可用的Tkinter/Tcl/Tk；Tkinter不是pip包。
Conda缺少Tk时可在激活环境运行 `conda install tk`，用 `python -m tkinter` 检查；
其他Python发行版按该发行版方式安装Tk支持。服务器批处理不需要GUI。
详见[HDF5浏览器指南](VIEWER_zh-CN.md)。

## 2. 依赖分层

| 用途 | 当前声明的依赖 |
| --- | --- |
| 核心 | astropy8.0.2、h5py3.16.0、numpy2.5.4、scipy1.18.1、specutils2.4.0、tqdm4.70.1 |
| `tools` | pandas3.1.0，用于分层抽样/导出 |
| `plot` | matplotlib3.11.2，用于光谱展示 |
| `viewer` | matplotlib 3.11.2；可选Tk桌面浏览器，Tcl/Tk由Python环境提供 |
| `render` | galsim2.8.5，用于部分旧形态渲染/库生成工具 |
| `test` | pytest9.1.1及声明的打包工具链 |

最终依赖以仓库根`setup.cfg`、`pyproject.toml`和[requirements目录](../requirements)为准。
核心模拟从预存库读取形态或使用已生成MGE库时，不要求加载GalSim。
声明兼容的依赖版本不等于已经验证所有平台组合；安装失败时先核对解释器、架构和对应wheel。

可复现的 macOS ARM64 依赖记录见[平台锁](migration_py312_20261023/requirements-macos-arm64-py312.lock)。
该锁排除CESS自身，软件由具体wheel/源码版本独立确定；该锁不是Gravity/Linux的安装清单，
也不表示已经验证所有新建Conda环境或平台组合。

## 3. 选择运行环境

每个新终端重新激活所选择的环境。Conda 方案执行：

```bash
conda activate cess
```

venv 方案执行：

```bash
source /path/to/cess-env/bin/activate
```

二者择一使用。退出 Conda 环境用 `conda deactivate`，退出 venv 用 `deactivate`。
历史`grizli` / Python3.7.11环境不用于安装当前包。

从任意cwd使用当前源码：

```bash
env PYTHONPATH=/path/to/CESS/src python -s -B -m cess --help
```

使用已安装包则无需PYTHONPATH。核查`sys.executable`、`cess.__file__`和`cess.__version__`，
可以避免选错解释器或混淆源码版与安装版。

### 3.1 升级已有环境中的 CESS

激活目标环境后，可从新源码目录安装；以下命令中的 `.` 指当前源码根目录：

```bash
cd /path/to/CESS
python -m pip install --upgrade '.[tools,plot]'
python -m pip check
python -c 'import sys, cess; print(sys.executable, cess.__version__, cess.__file__)'
```

也可把安装目标换为新的 wheel 文件。只下载新源码或执行 `git pull`，不会自动更新环境中已安装的包。
如果本地修改源码却保持相同版本号，需要明确重新安装；已确认依赖齐全且版本匹配时，可在源码根目录用
`python -m pip install --force-reinstall --no-deps .`。`--no-deps` 不适合用于首次创建环境。

开发时也可使用上面的 `PYTHONPATH` 方式直接运行源码。该方式改变本次调用的导入位置，
不等同于已经升级环境里的安装副本。新版本不能继续使用源码身份不同的旧运行manifest。

## 4. 科学输入与并发

软件包不包含用户的真实BayeSED星表。首次安装可用[公开toy示例](../examples/README_zh-CN.md)
检查接口；真实输入契约、配置路径和命令见[用户指南](USER_GUIDE_zh-CN.md)。
包内默认资源随包查找，外部JSON中的相对资源路径以该JSON目录为基准。

CLI 默认每worker请求1个数值库线程。文件级并发用 `-j N`，默认1；增加worker前应确认同时运行的任务内存。
具体见 [CPU设置](USER_GUIDE_zh-CN.md#61-cpu进程数与线程设置)与[Bash 命令指南](COMMANDS_zh-CN.md)。
当前不是MPI跨节点程序。
Gravity使用独立Linux环境和[服务器脚本](../scripts/gravity/README_zh-CN.md)，先确认本站资源申请语法。

## 5. 从源码运行测试

源码分发包含正式tests、必要兼容入口和小型历史源码oracle，不需要私有BayeSED输入或整套开发存档。
在源码根目录和已激活、具备测试依赖的环境中运行：

```bash
env PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -s -B -m pytest -q -p no:cacheprovider tests
```

wheel主要供安装使用；源码分发包含测试与必要兼容文件，运行上述测试应使用源码根目录。
实现测试和仪器/样本科学定标有不同作用；具体研究仍需核对PSF/LSF、背景及响应资源的定义。
