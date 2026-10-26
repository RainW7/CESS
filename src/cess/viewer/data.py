"""Read-only HDF5 access adapted from the supplied CESS viewer. No GUI imports."""
from __future__ import annotations

from collections import OrderedDict
import csv
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

BANDS = ("GU", "GV", "GI")

def decode(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace")
    return str(value)


def fmt_num(value) -> str:
    """把标量格式化成紧凑字符串。"""
    if isinstance(value, (bytes, np.bytes_)):
        return decode(value)
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return str(np.float64(value))
        av = abs(float(value))
        if av != 0.0 and (av < 1e-3 or av >= 1e6):
            return f"{float(value):.4e}"
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= 80 else text[:77] + "…"


def attr_to_text(value, maxlen: int = 1600) -> str:
    """把 HDF5 属性渲染成一行文本（长内容截断）。"""
    if isinstance(value, (bytes, np.bytes_)):
        text = decode(value)
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in "SUO":
            items = [decode(v) if isinstance(v, (bytes, np.bytes_)) else str(v) for v in value.ravel()[:80]]
            more = "" if value.size <= 80 else f", …共 {value.size} 项"
            text = "[" + ", ".join(items) + more + "]"
        else:
            text = np.array2string(value, threshold=24, edgeitems=4, precision=8)
    else:
        text = str(value)
    if len(text) > maxlen:
        text = text[:maxlen] + f" …(已截断，原长 {len(text)} 字符)"
    return text


def human_bytes(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


class Variable:
    """一个可以被绘图的变量。

    kind:
        'vec'    —— 一维数据集整体
        'col'    —— 二维表的某一列（index = 列号）
        'field'  —— 结构化数组的某个字段（index = 字段名）
        'col2d'  —— 二维数据集，绘图时用「像素下标」选定列
    """

    __slots__ = ("label", "path", "kind", "index", "units", "size")

    def __init__(self, label, path, kind, index=None, units="", size=0):
        self.label = label
        self.path = path
        self.kind = kind
        self.index = index
        self.units = units
        self.size = size


class ProductModel:
    """只读的 HDF5 产品封装：结构遍历、列/行读取、变量注册。"""

    def __init__(self, path: str, cache_bytes: int = 16 * 1024 * 1024):
        if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or cache_bytes < 0:
            raise ValueError("cache_bytes must be a nonnegative integer")
        self.path = str(Path(path).expanduser().absolute())
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self.cache_limit_bytes = cache_bytes
        self._wave: dict = {}
        self._vars: dict | None = None
        self._ids: np.ndarray | None = None
        self._paths: list[str] | None = None
        self.h5 = h5py.File(self.path, "r")
        try:
            self.nsrc = self._infer_nsrc()
        except BaseException:
            self.h5.close()
            raise

    # ---- 生命周期 ----
    def close(self) -> None:
        self.h5.close()
        self._cache.clear()
        self._cache_bytes = 0
        self._wave.clear()
        self._ids = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    # ---- 基础访问 ----
    def __contains__(self, path) -> bool:
        return path in self.h5

    def get(self, path):
        return self.h5[path]

    def read(self, path: str, key=None):
        """Read a requested selection; cache fixed-size values within a byte budget."""
        ckey = (path, "ALL" if key is None else key)
        try:
            if ckey in self._cache:
                self._cache.move_to_end(ckey)
                return self._cache[ckey]
        except TypeError:
            ckey = None
        dataset = self.h5[path]
        value = dataset[()] if key is None else dataset[key]
        self._remember(ckey, value)
        return value

    def _remember(self, key, value):
        array = np.asarray(value)
        if (key is None or array.dtype.hasobject or not self.cache_limit_bytes
                or array.nbytes > self.cache_limit_bytes):
            return
        while self._cache and (self._cache_bytes + array.nbytes > self.cache_limit_bytes
                               or len(self._cache) >= 256):
            _, removed = self._cache.popitem(last=False)
            self._cache_bytes -= np.asarray(removed).nbytes
        self._cache[key] = value
        self._cache_bytes += array.nbytes

    def field(self, path: str, name: str) -> np.ndarray:
        """Read one compound field without loading unrelated fields."""
        key = (path, "FIELD", name)
        if key in self._cache:
            self._cache.move_to_end(key)
            return np.asarray(self._cache[key])
        value = np.asarray(self.h5[path].fields(name)[()])
        self._remember(key, value)
        return value

    def vector(self, path: str) -> np.ndarray:
        return np.asarray(self.read(path, None))

    def column(self, path: str, j: int) -> np.ndarray:
        dataset = self.h5[path]
        if dataset.ndim != 2 or not 0 <= int(j) < dataset.shape[1]:
            raise IndexError("Column outside a two-dimensional dataset: " + path)
        return np.asarray(self.read(path, (slice(None), j)))

    def row(self, path: str, i: int) -> np.ndarray:
        dataset = self.h5[path]
        if dataset.ndim == 0 or not 0 <= int(i) < dataset.shape[0]:
            raise IndexError("Row outside dataset: " + path)
        return np.asarray(self.read(path, i))

    def first_path(self, *paths: str) -> str | None:
        """返回第一个存在的路径（兼容不同版本的布局）。"""
        for path in paths:
            if not path:
                continue
            try:
                if path in self.h5:
                    return path
            except Exception:
                continue
        return None

    def band_root(self, band: str) -> str | None:
        """波段数据所在组。

        新布局（layout 3.0）在 ``/spectra/<band>``；旧版 CESS 产品把
        ``GU/GV/GI`` 直接放在文件根下，这里一并兼容。
        """
        for path in (f"spectra/{band}", band):
            if path in self.h5 and isinstance(self.h5[path], h5py.Group):
                return path
        return None

    def band_wave(self, band: str) -> np.ndarray:
        if band not in self._wave:
            root = self.band_root(band)
            path = self.first_path(f"{root}/wave") if root else None
            self._wave[band] = self.vector(path) if path else np.array([])
        return self._wave[band]

    def unit_of(self, path: str) -> str:
        try:
            obj = self.h5[path]
        except KeyError:
            return ""
        for key in ("unit", "units"):
            if key in obj.attrs:
                return attr_to_text(obj.attrs[key], 60)
        return ""

    def table_columns(self, path: str):
        """Read named table columns, including the historical ``name`` attribute."""
        try:
            obj = self.h5[path]
        except KeyError:
            return None
        key = next((name for name in ("columns", "name") if name in obj.attrs), None)
        if key is None:
            return None
        names = [decode(v) for v in np.atleast_1d(obj.attrs[key]).ravel()]
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
            names = names[:obj.shape[1]]
        units = ([decode(v) for v in np.atleast_1d(obj.attrs["units"]).ravel()]
                 if "units" in obj.attrs else [])
        out = []
        for i, name in enumerate(names):
            out.append((name, units[i] if i < len(units) else ""))
        return out

    def cell(self, path: str, row: int, column):
        """One scalar of a named table.

        Handles both the current ``(source, column)`` layout and the historical
        ``(column, source[, ...])`` layout, so tables from older products can be
        inspected without special-casing at the call site.
        """
        dataset = self.h5[path]
        if isinstance(column, str):
            columns = self.table_columns(path) or []
            names = [name for name, _unit in columns]
            if column not in names:
                raise KeyError(column)
            index = names.index(column)
        else:
            index = int(column)
        if dataset.ndim <= 1:
            return float(np.ravel(dataset)[row])
        if dataset.ndim == 2:
            if index < dataset.shape[1]:
                return float(dataset[row, index])
            return float(dataset[index, row])
        # (columns, sources, ...) such as parameters_phot (21, nsrc, 1)
        if index < dataset.shape[0]:
            return float(np.ravel(dataset[index])[row])
        return float(np.ravel(dataset[..., row])[index])

    # ---- 源数 ----
    def _infer_nsrc(self) -> int:
        for cand in ("catalog/ID", "catalog/data_mask", "catalog/parameters_desi"):
            if cand in self.h5:
                dataset = self.h5[cand]
                if isinstance(dataset, h5py.Dataset) and dataset.ndim >= 1:
                    return int(dataset.shape[0])
        best = 0
        found = False
        for band in BANDS:
            group = self.band_root(band)
            if group is not None:
                for key in self.h5[group]:
                    ds = self.h5[group][key]
                    if isinstance(ds, h5py.Dataset) and ds.ndim == 2:
                        found = True
                        best = max(best, int(ds.shape[0]))
        if found:
            return best
        # 兜底：兼容不是 CESS 布局的文件（例如 BayeSED 拟合星表）。
        # 先找根级的 ID 类一维数组，再退化为“最大的二维数据集的行数”。
        for name in ("ID", "id", "IDs", "ids", "target_id", "source_id", "obj_id"):
            if name in self.h5:
                ds = self.h5[name]
                if isinstance(ds, h5py.Dataset) and ds.ndim == 1:
                    return int(ds.shape[0])
        rows = 0
        for path in self.dataset_paths():
            try:
                ds = self.h5[path]
            except Exception:
                continue
            if isinstance(ds, h5py.Dataset) and ds.ndim == 2 and ds.shape[1] > 1:
                rows = max(rows, int(ds.shape[0]))
        return rows

    # ---- 结构与元数据 ----
    def dataset_paths(self) -> list[str]:
        if self._paths is None:
            found: list[str] = []

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    found.append(name)

            self.h5.visititems(visit)
            found.sort()
            self._paths = found
        return self._paths

    def children(self, path: str = ""):
        group = self.h5[path] if path else self.h5
        groups, datasets = [], []
        for key in group.keys():
            child = group[key]
            full = f"{path}/{key}" if path else key
            (groups if isinstance(child, h5py.Group) else datasets).append((full, key, child))
        groups.sort(key=lambda t: t[1].lower())
        datasets.sort(key=lambda t: t[1].lower())
        return groups + datasets

    def info_text(self, path: str) -> str:
        obj = self.h5[path] if path else self.h5
        lines = [f"路径 : /{path}" if path else "路径 : /（文件根）"]
        if isinstance(obj, h5py.Dataset):
            itemsize = obj.dtype.itemsize
            lines.append(
                f"类型 : Dataset   shape={obj.shape}   dtype={obj.dtype}   "
                f"约 {human_bytes(int(obj.size) * itemsize)}"
            )
            lines.append(f"分块 : {obj.chunks}   压缩 : {obj.compression}")
        else:
            lines.append(f"类型 : Group   直接成员 {len(obj.keys())} 个")
        if obj.attrs:
            lines.append("")
            lines.append("属性：")
            for key in sorted(obj.attrs):
                value = obj.attrs[key]
                if key in ("columns", "name", "units") and isinstance(value, np.ndarray):
                    items = [decode(v) for v in value.ravel()]
                    lines.append(f"  {key} = [{len(items)} 项] " + ", ".join(items[:40])
                                 + ("" if len(items) <= 40 else ", …"))
                else:
                    lines.append(f"  {key} = {attr_to_text(value)}")
        columns = self.table_columns(path) if isinstance(obj, h5py.Dataset) else None
        if columns:
            lines.append("")
            lines.append("列定义：")
            for i, (name, unit) in enumerate(columns):
                lines.append(f"  [{i:2d}] {name:<26s} {unit}")
        if isinstance(obj, h5py.Dataset) and obj.dtype.names:
            lines.append("")
            lines.append("结构化字段：")
            for name in obj.dtype.names:
                lines.append(f"  {name:<26s} {obj.dtype[name]}")
        return "\n".join(lines)

    # ---- 变量注册 ----
    def pixel_labels(self, path: str, count: int):
        """为二维数据集的每个像素列生成显示名（尽量带上波长）。"""
        wave = None
        group = path.rsplit("/", 1)[0] if "/" in path else ""
        if group:
            candidate = f"{group}/wave"
            if candidate in self.h5:
                try:
                    wave = np.asarray(self.h5[candidate][()])
                    if wave.ndim != 1:
                        wave = None
                except Exception:
                    wave = None
        labels = []
        for j in range(count):
            if wave is not None and j < wave.size:
                labels.append(f"像素 {j} (λ={wave[j]:.1f} Å)")
            else:
                labels.append(f"像素 {j}")
        return labels

    def column_variable(self, path: str, j: int) -> Variable:
        """为二维数据的第 j 列构造一个 Variable（结构页点选列时用）。"""
        obj = self.h5[path]
        columns = self.table_columns(path)
        if columns and j < len(columns):
            name, unit = columns[j]
            label = f"{path} :: {name}" + (f" [{unit}]" if unit else "")
        else:
            label = f"{path} :: {self.pixel_labels(path, obj.shape[1])[j]}"
            unit = self.unit_of(path)
        size = int(obj.shape[0])
        spec = Variable(label, path, "col", j, unit, size)
        if self._vars is not None:
            self._vars.setdefault(label, spec)
        return spec

    def build_variables(self) -> dict:
        """枚举可以绘图的变量，返回 label -> Variable。"""
        if self._vars is not None:
            return self._vars
        table: dict[str, Variable] = {}

        def add(spec: Variable) -> None:
            table.setdefault(spec.label, spec)

        for path in self.dataset_paths():
            if path.startswith("provenance/"):
                continue
            ds = self.h5[path]
            unit = self.unit_of(path)
            if ds.ndim == 0:
                continue
            if ds.dtype.names:
                for name in ds.dtype.names:
                    if ds.dtype[name].kind not in "fiub":
                        continue
                    label = f"{path} :: {name}"
                    add(Variable(label, path, "field", name, "", int(ds.shape[0])))
                continue
            if ds.dtype.kind == "O":
                continue
            if ds.ndim == 1:
                label = path + (f" [{unit}]" if unit else "")
                if int(ds.shape[0]) != self.nsrc:
                    label += f" (N={ds.shape[0]})"
                add(Variable(label, path, "vec", None, unit, int(ds.shape[0])))
                continue
            if ds.ndim == 2 and ds.shape[1]:
                columns = self.table_columns(path)
                if columns:
                    for j, (name, col_unit) in enumerate(columns):
                        label = f"{path} :: {name}" + (f" [{col_unit}]" if col_unit else "")
                        add(Variable(label, path, "col", j, col_unit, int(ds.shape[0])))
                else:
                    label = (
                        f"{path} :: 二维 {ds.shape[0]}×{ds.shape[1]} 像素列"
                        + (f" [{unit}]" if unit else "")
                    )
                    add(Variable(label, path, "col2d", None, unit, int(ds.shape[0])))
        self._vars = table
        return table

    def variable_map(self) -> dict:
        return self.build_variables()

    def resolve(self, spec: Variable, pixel: int = 0) -> np.ndarray:
        if spec.kind == "vec":
            return self.vector(spec.path)
        if spec.kind == "col":
            return self.column(spec.path, int(spec.index))
        if spec.kind == "field":
            return self.field(spec.path, str(spec.index))
        if spec.kind == "col2d":
            ds = self.h5[spec.path]
            if ds.ndim != 2 or not ds.shape[1]:
                raise IndexError("No pixel columns in dataset: " + spec.path)
            j = max(0, min(int(pixel), int(ds.shape[1]) - 1))
            return self.column(spec.path, j)
        raise ValueError(f"未知变量类型: {spec.kind}")

    def row_vector(self, path: str, row: int) -> np.ndarray:
        """二维数据集的某一源行（整条记录）。"""
        ds = self.h5[path]
        if ds.ndim != 2 or not 0 <= int(row) < ds.shape[0]:
            raise IndexError("Source row outside a two-dimensional dataset: " + path)
        return np.asarray(self.read(path, (int(row), slice(None))))

    def _wavelength_spec(self, spec):
        if spec is None or spec.kind != "vec":
            return False
        basename = spec.path.rsplit("/", 1)[-1].lower()
        return ("wave" in basename or "lambda" in basename)

    def _spectral_matrix(self, spec, length):
        if spec is None or spec.kind not in ("col", "col2d"):
            return False
        dataset = self.h5[spec.path]
        return (isinstance(dataset, h5py.Dataset) and dataset.ndim == 2
                and dataset.shape[1] == length and not self.table_columns(spec.path))

    def match_axis_lengths(self, x_spec, x, y_spec, y, row: int):
        """Match a wavelength vector to a spectral row, even when N equals W."""
        if self._wavelength_spec(x_spec) and self._spectral_matrix(y_spec, np.size(x)):
            return x, self.row_vector(y_spec.path, row)
        if self._wavelength_spec(y_spec) and self._spectral_matrix(x_spec, np.size(y)):
            return self.row_vector(x_spec.path, row), y
        return x, y

    def resolve_xy(self, x_spec, y_spec, pixel=0, row=0):
        """Resolve plot data without reading an unused whole source column.

        ``row_used`` identifies wavelength-versus-spectrum selection. Otherwise
        both selections keep their original vector/column meanings. A length
        mismatch is returned for the caller to explain rather than guessed away.
        """
        if self._wavelength_spec(x_spec):
            x = self.resolve(x_spec, pixel)
            if self._spectral_matrix(y_spec, x.size):
                return x, self.row_vector(y_spec.path, row), int(row)
        if self._wavelength_spec(y_spec):
            y = self.resolve(y_spec, pixel)
            if self._spectral_matrix(x_spec, y.size):
                return self.row_vector(x_spec.path, row), y, int(row)
        return self.resolve(x_spec, pixel), self.resolve(y_spec, pixel), None

    def resolve_for_axis(self, spec, pixel=0, row_used=None, length=None):
        """Resolve colour/error data on the same spectral row when applicable."""
        if (row_used is not None and length is not None
                and self._spectral_matrix(spec, length)):
            return self.row_vector(spec.path, row_used)
        return self.resolve(spec, pixel)

    # ---- ID 检索 ----
    def ids_str(self) -> np.ndarray | None:
        if self._ids is None:
            id_path = self.first_path("catalog/ID", "ID")
            if id_path is None:
                return None
            raw = np.asarray(self.h5[id_path][()])
            if raw.ndim != 1:
                return None
            self._ids = np.array([decode(v) for v in raw])
        return self._ids

    def row_of_id(self, text: str) -> int | None:
        ids = self.ids_str()
        if ids is None:
            return None
        text = str(text).strip()
        for i, value in enumerate(ids):
            if value == text:
                return i
        lowered = text.lower()
        for i, value in enumerate(ids):
            if value.lower().startswith(lowered):
                return i
        return None

    # ---- 源信息 ----
    def band_available(self, band: str) -> bool:
        return self.band_root(band) is not None

    def spectrum_row(self, band: str, dataset: str, row: int):
        root = self.band_root(band)
        if root is None:
            return None
        path = f"{root}/{dataset}"
        if path not in self.h5:
            return None
        try:
            return self.row(path, row)
        except Exception:
            return None

    def measured_lines(self, band: str, row: int):
        """该源在该波段全部实测谱线（lines/<band>/line_measurements/truth）。"""
        path = self.first_path(f"lines/{band}/line_measurements/truth",
                               f"{band}/line_measurements/truth")
        if path is None:
            return np.array([]), np.array([])
        try:
            sel = self.field(path, "source_row") == int(row)
            if not sel.any():
                return np.array([]), np.array([])
            indices = np.flatnonzero(sel)
            table = self.h5[path].fields(["center_wave", "snr", "valid"])[indices]
            waves = np.asarray(table["center_wave"], dtype=float)
            snrs = np.asarray(table["snr"], dtype=float)
            valid = np.asarray(table["valid"]).astype(bool)
            keep = np.isfinite(waves) & valid
            return waves[keep], snrs[keep]
        except Exception:
            return np.array([]), np.array([])

    def line_peaks(self, band: str, row: int, detected: bool):
        """返回该源在某个波段的发射线波长（intrinsic 或 detected）。"""
        group = self.first_path(f"lines/{band}", band)
        if group is None:
            return np.array([])
        prefix = "detect_el" if detected else "intri_el"
        id_path = f"{group}/{prefix}_id"
        wave_path = f"{group}/{prefix}_wave"
        if id_path not in self.h5 or wave_path not in self.h5:
            return np.array([])
        try:
            ids = self.vector(id_path)
        except Exception:
            return np.array([])
        if ids.size == 0:
            return np.array([])
        hits = np.nonzero(ids == row)[0]
        if hits.size == 0:
            return np.array([])
        out = []
        try:
            waves = self.h5[wave_path][hits]
        except (IndexError, TypeError, ValueError):
            return np.array([])
        for item in waves:
            try:
                out.extend(np.asarray(item, dtype=float).ravel().tolist())
            except Exception:
                continue
        result = np.array(out, dtype=float)
        return result[np.isfinite(result)]


def validate_export_path(path, source_path):
    """Reject HDF5 destinations and aliases of the read-only source file."""
    target = Path(path).expanduser().absolute()
    if target.suffix.lower() in (".h5", ".hdf5", ".hdf", ".he5"):
        raise ValueError("Export to a CSV or image file, not an HDF5 destination")
    if source_path is not None:
        source = Path(source_path).expanduser().absolute()
        if target.resolve() == source.resolve():
            raise ValueError("Export cannot replace the open source file")
        if target.exists() and source.exists() and os.path.samefile(target, source):
            raise ValueError("Export cannot replace a link to the open source file")
    return target


def _csv_value(value):
    if isinstance(value, (bytes, np.bytes_)):
        return decode(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path, headers, rows, source_path):
    """Atomically write untruncated CSV values, protecting the source HDF5.

    Existing CSV destinations may be replaced after the GUI's save dialog;
    failures leave the previous destination untouched. IEEE floating values
    use Python's round-trip representation rather than display formatting.
    """
    target = validate_export_path(path, source_path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                prefix="." + target.name + ".", suffix=".tmp", dir=target.parent,
                delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow([_csv_value(value) for value in headers])
            for row in rows:
                writer.writerow([_csv_value(value) for value in row])
            stream.flush()
            os.fsync(stream.fileno())
        validate_export_path(target, source_path)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target

