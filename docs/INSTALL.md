# Installation and runtime environment

[English](INSTALL.md) | [简体中文](INSTALL_zh-CN.md)

This guide applies to the one-dimensional CESS v0.9.0 release. CESS requires Python >=3.12,<3.13.
Download source or a distribution archive from the [CESS repository](https://github.com/RainW7/CESS). The commands below install from that source or archive.

Use an interpreter and dependencies built for the target platform: Linux, macOS ARM64, or x86_64.
Copying a Mac Conda/venv directory to a server does not create a usable server environment.

## 1. Create an environment and install CESS

Download or extract the CESS source first. `/path/to/CESS` denotes the source root containing
`pyproject.toml` and `setup.cfg`. All `/path/to/...` paths are placeholders: replace them with real locations.
Choose one of the two environment options below. If a working Python 3.12 CESS environment already exists,
activate it instead of creating it again; upgrading CESS still requires installing the new version.
When dependencies are not already available, pip normally needs network access to download platform-specific packages.

### 1.1 Create a Conda environment named cess

If Conda is installed and available in your terminal, run these commands in order:

```bash
conda create -n cess python=3.12 pip
conda activate cess
cd /path/to/CESS
python -m pip install '.[tools,plot]'
```

`conda create` creates the environment; `python -m pip install` installs CESS and dependencies into the active environment.
`.[tools,plot]` installs the core scientific dependencies, pandas for sampling/export, and Matplotlib for plotting.
There is no need to install each dependency manually. Keep the quotes around the argument, including when using zsh.
The newly created Conda environment appears in `conda env list`.

### 1.2 Using Python venv

If `python3.12` is already available, you can use a standalone venv instead. Leave any active Conda environment first,
then run the following commands. Activation with `source` works in Bash or zsh on macOS/Linux:

```bash
python3.12 --version
python3.12 -m venv --prompt cess /path/to/cess-env
source /path/to/cess-env/bin/activate
cd /path/to/CESS
python -m pip install '.[tools,plot]'
```

A venv is identified by its path and **does not appear in `conda env list`**. It can install and run CESS in the same way.

### 1.3 Check the installation and run the first example

After completing either installation option, run these commands in that environment:

```bash
python --version
python -m pip check
python -c 'import sys, cess; print(sys.executable, cess.__version__, cess.__file__)'
cess --help
cess simulate --help
```

Python should be 3.12.x, and installing this release should report `cess.__version__` as `0.9.0`.
`pip check` should report no broken requirements. The interpreter and CESS package paths should belong to the selected environment.
Working help output confirms that the command entry point is available. Next run the [public toy example](../examples/README.md)
before using your own catalogue. Installation checks do not validate the scientific model.

### 1.4 Install a wheel or source archive

Activate a Python 3.12 environment as described above. These absolute-path examples work from any working directory.
Replace the download paths with real files and choose one installation method:

```bash
# Wheel: install CESS and its core dependencies.
python -m pip install /path/to/dist/cess-0.9.0-py3-none-any.whl
```

To include plotting and sampling, add extras to the wheel specification and quote the complete path:

```bash
python -m pip install '/path/to/dist/cess-0.9.0-py3-none-any.whl[tools,plot]'
```

An sdist is a source archive. Extract it and install from its source directory, or install it directly:

```bash
python -m pip install '/path/to/dist/cess-0.9.0.tar.gz[tools,plot]'
python -m pip check
```

Do not extract a wheel manually. Source installation normally needs build tools; pip prepares a build environment
according to `pyproject.toml`. Throughput, background, and morphology resources are installed with CESS and do not need
to be copied into the current directory for normal 1D runs. Real BayeSED catalogues are not included and must be supplied separately.

### 1.5 Choose optional functionality

From the source root, choose an installation command appropriate to your needs:

| Installation command | Purpose |
| --- | --- |
| `python -m pip install .` | Core 1D simulation, product reading, chunk merging, and command-line tools |
| `python -m pip install '.[tools,plot]'` | Core plus sampling/export and plotting; suitable for typical interactive use |
| `python -m pip install '.[render]'` | Also install GalSim for the relevant morphology rendering/library tools |
| `python -m pip install '.[tools,plot,render,test]'` | Development, full tests, and the related optional tools |

Normal 1D simulation with precomputed morphology libraries does not require GalSim.
Use the current package declarations for Python 3.12 dependencies.

### 1.6 Install the optional desktop viewer

From the CESS source root, run `python -m pip install '.[viewer]'`, then use
`cess-viewer /path/to/product.hdf5 --lang en`. The `viewer` extra adds Matplotlib;
NumPy/h5py are core dependencies. A graphical desktop and working Tkinter/Tcl/Tk are also required.
Tkinter is not a pip package. In Conda, use `conda install tk` in the activated environment if needed,
and check `python -m tkinter`; for other Python distributions use their Tk installation method.
Batch simulation on a headless server does not require GUI support.
See the [HDF5 viewer guide](VIEWER.md).

## 2. Dependency groups

| Group | Declared dependencies |
| --- | --- |
| Core | astropy 8.0.2, h5py 3.16.0, numpy 2.5.4, scipy 1.18.1, specutils 2.4.0, tqdm 4.70.1 |
| `tools` | pandas 3.1.0 for stratified sampling/export |
| `plot` | matplotlib 3.11.2 for spectrum plots |
| `viewer` | matplotlib 3.11.2 for the optional Tk viewer; Tcl/Tk is supplied by the Python environment |
| `render` | galsim 2.8.5 for selected legacy rendering/library-generation tools |
| `test` | pytest 9.1.1 and the declared packaging toolchain |

The authoritative requirements are the root `setup.cfg`, `pyproject.toml`, and [requirements directory](../requirements).
Core simulation using precomputed profiles or an already generated MGE library does not need to load GalSim.
If installation fails, check the interpreter, architecture, and availability of the corresponding wheels first.

A reproducible macOS ARM64 dependency record is available in the [platform lock file](migration_py312_20261023/requirements-macos-arm64-py312.lock).
It excludes CESS itself, whose identity is determined separately by the selected wheel or source version.
Use the requirement profiles above for other platforms, including Linux servers.

## 3. Select the runtime environment

Activate your chosen environment in each new terminal. For Conda, use:

```bash
conda activate cess
```

For a venv, use:

```bash
source /path/to/cess-env/bin/activate
```

Choose one approach. Leave Conda with `conda deactivate`, or leave a venv with `deactivate`.
The historical `grizli` / Python3.7.11 environment is not used to install the current package.

To run the current source from any working directory:

```bash
env PYTHONPATH=/path/to/CESS/src python -s -B -m cess --help
```

An installed package does not need `PYTHONPATH`. Check `sys.executable`, `cess.__file__`, and `cess.__version__`
to avoid selecting the wrong interpreter or confusing source execution with an installed copy.

### 3.1 Upgrade CESS in an existing environment

Activate the target environment, then install from the new source directory. The dot in this command denotes the current source root:

```bash
cd /path/to/CESS
python -m pip install --upgrade '.[tools,plot]'
python -m pip check
python -c 'import sys, cess; print(sys.executable, cess.__version__, cess.__file__)'
```

You can replace the installation target with the new wheel file. Merely downloading source or running `git pull` does not update
the package already installed in an environment. If you edit local source without changing its version number, explicitly reinstall it.
When the dependencies are already installed and match the requirements, run
`python -m pip install --force-reinstall --no-deps .` from the source root. `--no-deps` is not appropriate for first-time environment setup.

During development, the `PYTHONPATH` command above also runs source directly. It changes the import location for that invocation;
it does not upgrade the installed copy. A new version cannot resume an old run manifest bound to a different source identity.

## 4. Scientific input and concurrency

The package does not include users' real BayeSED catalogues. On first installation, use the [public toy example](../examples/README.md)
to check the interface. The real input contract, configuration paths, and run commands are described in the [user guide](USER_GUIDE.md).
Bundled default resources are located within the package; relative paths in an external JSON are resolved from that JSON's directory.

The CLI requests one numeric-library thread per worker by default. File-level concurrency uses `-j N`, with a default of 1.
Before increasing the worker count, ensure sufficient memory for the simultaneous tasks. See
[CPU settings](USER_GUIDE.md#61-cpu-processes-and-threads) and the [Bash command guide](COMMANDS.md).
The current program does not distribute MPI work across nodes.
Gravity uses its own Linux environment and the [server scripts](../scripts/gravity/README.md). For catalogue campaigns, use the [per-file Torque job arrays](../scripts/gravity/ARRAY.md) and check the current queue and account limits.

## 5. Run tests from source

From the source root, with an active environment containing the test dependencies, run:

```bash
env PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -s -B -m pytest -q -p no:cacheprovider tests
```

Run tests from the source checkout. For scientific applications, also check the PSF/LSF, background, and response-resource definitions.
