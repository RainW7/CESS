#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional bilingual Tk/Matplotlib desktop viewer, adapted from the supplied script.

Use ``cess-viewer [file.hdf5] --lang en|zh``. Data access and export helpers live
in ``cess.viewer.data``; normal CESS simulation imports never load this GUI.
"""

from __future__ import annotations

from .data import (ProductModel, Variable, decode, fmt_num, attr_to_text, human_bytes,
                   write_csv, validate_export_path)
from .expressions import filter_mask
from .ui import (configure_theme, initial_geometry, ScrollableFrame,
                 CollapsibleFrame, ToolTip, RoundedButton, BACKGROUND, PANEL, TEXT, MUTED)

import argparse
import json
import os
import re
import sys
import tempfile

# --------------------------------------------------------------------------
# matplotlib：先解决缓存目录不可写的问题，再选择 TkAgg 后端
# --------------------------------------------------------------------------


def _ensure_mpl_config_dir() -> None:
    """让 matplotlib 的缓存目录可写，否则某些受限环境会退回临时目录。"""
    if os.environ.get("MPLCONFIGDIR"):
        return
    candidates = [
        os.path.join(os.path.expanduser("~"), ".cache", "cess-h5-viewer-mpl"),
        os.path.join(os.path.expanduser("~"), ".matplotlib"),
    ]
    for cand in candidates:
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, ".write_probe")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("ok")
            os.remove(probe)
            os.environ["MPLCONFIGDIR"] = cand
            return
        except OSError:
            continue
    fallback = os.path.join(tempfile.gettempdir(), "cess-h5-viewer-mpl")
    os.makedirs(fallback, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = fallback


_ensure_mpl_config_dir()


# --------------------------------------------------------------------------
# 界面设置的持久化（窗口大小、分隔条、上次文件）
# --------------------------------------------------------------------------

SETTINGS_VERSION = 1


def settings_path() -> str:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "cess-viewer", "settings.json")


def load_settings() -> dict:
    try:
        with open(settings_path(), encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(data: dict) -> bool:
    """尽力保存；配置目录不可写时静默失败，不影响使用。"""
    target = settings_path()
    temporary = target + ".tmp"
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=1)
        os.replace(temporary, target)
        return True
    except OSError:
        try:
            os.remove(temporary)
        except OSError:
            pass
        return False


_GEOMETRY_RE = re.compile(r"^(\d+)x(\d+)([+-]-?\d+)([+-]-?\d+)$")


def usable_geometry(root, geometry: str) -> str:
    """只接受落在当前屏幕内、尺寸合理的窗口几何串。"""
    match = _GEOMETRY_RE.match(str(geometry or ""))
    if not match:
        return ""
    width, height = int(match.group(1)), int(match.group(2))
    x = int(match.group(3))
    y = int(match.group(4))
    try:
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
    except Exception:
        return ""
    if width < 900 or height < 600:
        return ""
    if width > screen_w or height > screen_h:
        return ""
    if x < 0 or y < 0 or x > screen_w - width or y > screen_h - height:
        return ""
    return f"{width}x{height}{x:+d}{y:+d}"


import numpy as np  # noqa: E402
import h5py  # noqa: E402

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

import matplotlib  # noqa: E402

# FigureCanvasTkAgg selects the GUI canvas without changing the global backend.
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.transforms import blended_transform_factory  # noqa: E402


# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

# 界面语言：zh / en（en 时所有文案走 t() 翻译表，字体也换成拉丁字体）
LANG = "en"

# zh → en 的界面文案表（由 generate 脚本灌入；缺项时 t() 原样返回中文）
TRANSLATIONS: dict[str, str] = {
    ' 个。\n请选择属于同一行轴（源或像素）的变量；\n若想看某个源的一维光谱，请到「一维光谱」页，或让 X 用波段波长向量、\nY 用二维谱的像素列（此时取「源行号」对应的那一行）。': ' elements.\nPlease choose variables on the same row axis (source or pixel);\nFor a single source spectrum, use the Spectrum tab, or set X to a band wavelength vector\nand Y to a pixel column of a 2D spectrum (that source row is used).',
    '界面语言：zh 中文（默认）/ en English；运行时也可用「语言 / Language」菜单切换': 'UI language: zh Chinese (default) / en English; you can also switch it at runtime from the “Language” menu.',
    '源行号用于行向变量（二维谱的像素列会取这一行）。着色变量 / Y 误差选「（不使用）」即可清除。': 'The source row applies to row-wise variables (a 2D spectrum’s pixel column uses that row).\nChoose “(none)” for color / Y error to clear it.',
    '请按住 ⌘ / Shift 在列表里选中两列（第 1 列为 X，第 2 列为 Y）。': 'Hold ⌘ / Shift to select two columns (1st = X, 2nd = Y).',
    '（多条件请用 & / |，并给每个条件加括号，例如 (x>22) & (y>20)）': '(for multiple conditions use & / | and parenthesize each, e.g. (x>22) & (y>20))',
    '电子率 source_electron_rate_pre_morph': 'Electron rate source_electron_rate_pre_morph',
    '含噪流量 flux_ujy_with_noise': 'Noisy flux flux_ujy_with_noise',
    '请先切到「一维光谱」或「X-Y 绘图」页再保存。': 'Switch to the Spectrum or X-Y Plot tab before saving.',
    '请先在「列 / 像素 / 字段」列表里选择一列。': 'Select a column in the "Columns / pixels / fields" list first.',
    '默认显示：源在天球上的分布（可自行更换 X/Y）': 'Default: sky distribution of sources (change X/Y freely)',
    ' 个离群点超出显示范围（可取消“稳健范围”）': ' outliers beyond display range (uncheck "robust range")',
    'HDF5 产品文件（省略则尝试默认示例文件）': 'HDF5 product file (omit to try the default sample file)',
    '没有可用波段（文件中缺少 /spectra）': 'No bands available (file lacks /spectra)',
    '无缝': 'overlay',
    '分波段': 'per band',
    '这个文件不是 CESS 光谱产品；可到「X-Y 绘图」页选「波长向量 + 二维光谱」，用源行号画单个源的光谱。':
        'This file is not a CESS spectral product. Use the "X-Y Plot" tab with a wavelength '
        'vector plus a 2-D spectrum, and pick the source row to plot one source spectrum.',
    '默认显示：当前源行的一维光谱（用上面的「源行号」切换源，或换 X/Y 画别的）':
        'Default: 1-D spectrum of the current source row (switch source with "Source row")',
    '默认显示：当前源行的一维光谱（用上面的「源行号」切换源）':
        'Default: 1-D spectrum of the current source row (switch source with "Source row")',
    '类型 : Dataset   shape=': 'Type: Dataset   shape=',
    '筛选表达式（x / y / np 可用）': 'Filter expression (x / y / np)',
    '模型流量 flux_ujy_model': 'Model flux flux_ujy_model',
    'CESS HDF5 浏览与可视化工具': 'CESS HDF5 Browser and Viewer',
    'GI 红端 5800–10500 Å': 'GI red 5800–10500 Å',
    '列 / 像素 / 字段（可多选两列）': 'Columns / pixels / fields (pick two)',
    '类型 : Group   直接成员 ': 'Type: Group   Direct members: ',
    'GU 蓝端 2500–4200 Å': 'GU blue 2500–4200 Å',
    'GV 绿端 3800–6600 Å': 'GV green 3800–6600 Å',
    'X 与 Y 长度不同，无法导出。': 'X and Y lengths differ; cannot export.',
    '分析掩膜 (spec_mask)': 'Analysis mask (spec_mask)',
    '稳健范围 (0.2–99.8%)': 'Robust range (0.2–99.8%)',
    '筛选表达式的结果长度与数据不一致': 'Filter expression result length does not match data',
    '2D 直方图 (hexbin)': '2D histogram (hexbin)',
    '叠加 _0th 诊断代理（虚线）': 'Overlay _0th diagnostic proxy (dashed)',
    '请先选择 X、Y 变量并绘图。': 'Select X and Y variables and plot first.',
    '选择 CESS HDF5 产品': 'Select CESS HDF5 product',
    '电子计数 flux_elec': 'Electron counts flux_elec',
    'CESS HDF5 浏览器': 'CESS HDF5 Viewer',
    '入射流量 flux_ujy': 'Incident flux flux_ujy',
    '导出该源光谱 (CSV)…': 'Export spectrum (CSV)…',
    '当前没有可导出的光谱数据。': 'No spectrum data to export.',
    '该对象不支持作为绘图变量。': 'This object cannot be used as a plot variable.',
    '语言 / Language': 'Language',
    '光谱提取  valid=': 'Spectrum extraction  valid=',
    '请选择 X 与 Y 变量': 'Select X and Y variables',
    '   |   数据集 ': '   |   Datasets ',
    '垂直间距 hspace': 'Vertical spacing (hspace)',
    '打开 HDF5 文件…': 'Open HDF5 file…',
    '水平间距 wspace': 'Horizontal spacing (wspace)',
    '没有找到 ID 以 “': 'No source with ID starting with “',
    '路径 : /（文件根）': 'Path: / (file root)',
    '   |   源数 ': '   |   Sources ',
    ' …(已截断，原长 ': ' …(truncated, original length ',
    'shape / 成员': 'shape / members',
    '底边距 bottom': 'Bottom margin (bottom)',
    '长度不一致：X 有 ': 'Length mismatch: X has ',
    ' 个元素，Y 有 ': ' elements, Y has ',
    ' 点（筛选后，原 ': ' points (after filter, of ',
    'Y 误差 (可选)': 'Y error (optional)',
    'log10(计数)': 'log10(count)',
    '右边距 right': 'Right margin (right)',
    '着色变量 (可选)': 'Color by (optional)',
    '筛选后没有数据点。': 'No data points after filtering.',
    '选中两列 → 画图': 'Plot both columns',
    '（请选择左侧对象）': '(select an object on the left)',
    '   压缩 : ': '   Compression: ',
    '  全为 NaN': '  all NaN',
    ' 点在显示范围外': ' points outside display range',
    '左边距 left': 'Left margin (left)',
    '无法打开文件：\n': 'Cannot open file:\n',
    '未知变量类型: ': 'Unknown variable type: ',
    '筛选表达式无效：': 'Invalid filter expression: ',
    '行号      ': 'Row       ',
    '（无匹配数据集）': '(no matching datasets)',
    ' :: 二维 ': ' :: 2D ',
    ' · max ': ' · max ',
    ' · min ': ' · min ',
    '” 开头的源。': '” found.',
    '三波段同轴叠画': '3-band overlay',
    '信噪比 snr': 'SNR snr',
    '切换语言失败：': 'Failed to switch language: ',
    '发射线    ': 'Emission lines    ',
    '导出当前数据…': 'Export current data…',
    '已保存图 → ': 'Saved figure → ',
    '文件不存在：\n': 'File not found:\n',
    '没有有效数据。': 'No valid data.',
    '测光 mag ': 'Photometry mag ',
    '结构 / 数据': 'Structure / Data',
    '设为 X 变量': 'Set as X',
    '设为 Y 变量': 'Set as Y',
    '零级 mag ': 'Zeroth-order mag ',
    '顶边距 top': 'Top margin (top)',
    '   波段 ': '   bands ',
    ' / 真值 ': ' / truth ',
    ' · 中位 ': ' · median ',
    ' · 像素 ': ' · pixel ',
    ' · 源行 ': ' · source row ',
    '1D 直方图': '1D histogram',
    '1σ 误差带': '1σ error band',
    'X-Y 绘图': 'X-Y Plot',
    '保存当前图…': 'Save current figure…',
    '其中数据集 ': 'Datasets among them: ',
    '导出绘图数据': 'Export plot data',
    '纵向对数坐标': 'Log y axis',
    '结构化字段：': 'Structured fields:',
    '设为着色变量': 'Set as color',
    '路径 : /': 'Path: /',
    '过滤变量列表': 'Filter variable list',
    '（仅显示前 ': '(showing first ',
    '，Y 长度 ': ', Y length ',
    '   约 ': '   ~',
    ' / 共 ': ' of ',
    ' 行 → ': ' rows → ',
    ', …共 ': ', … ',
    ': 检测 ': ': detected ',
    'X 长度 ': 'X length ',
    '◀ 上一页': '◀ Prev',
    '下一页 ▶': 'Next ▶',
    '保存失败：': 'Save failed: ',
    '写入失败：': 'Write failed: ',
    '分块 : ': 'Chunks: ',
    '变量与样式': 'Variables & style',
    '导出数据…': 'Export data…',
    '已关闭文件': 'File closed',
    '未打开文件': 'No file opened',
    '标注发射线': 'Mark emission lines',
    '直接成员 ': 'Direct members: ',
    '过滤路径:': 'Filter path:',
    '（不使用）': '(none)',
    '；有效点 ': '; valid points ',
    ' 像素列': ' pixel columns',
    ' 字符)': ' chars)',
    ' 参考峰': ' reference peak',
    ' 匹配噪声峰': ' matched noisy peak',
    ' 项] ': ' items] ',
    'x 对数': 'Log x',
    'y 对数': 'Log y',
    '一维光谱': 'Spectrum',
    '像素下标': 'Pixel index',
    '关闭文件': 'Close file',
    '列定义：': 'Column definitions:',
    '反转 y': 'Invert y',
    '子图参数': 'Subplot params',
    '对象信息': 'Object info',
    '导出光谱': 'Export spectrum',
    '已导出 ': 'Exported ',
    '恢复默认': 'Reset defaults',
    '所有文件': 'All files',
    '数据类型': 'Dtype',
    '数据预览': 'Data preview',
    '有效点 ': 'Valid points ',
    '模型曲线': 'Model curve',
    '清空筛选': 'Clear filter',
    '读取失败': 'Read failed',
    '质量标志': 'Quality flags',
    '（源行 ': ' (source row ',
    ' 列）': ' columns)',
    ' 模型': ' model',
    ' 点）': ' points)',
    '保存图': 'Save figure',
    '像素 ': 'Pixel ',
    '分波段': 'per band',
    '字段 ': 'Fields: ',
    '属性：': 'Attributes:',
    '数据集': 'Dataset',
    '有效 ': 'valid ',
    '源信息': 'Source info',
    '源行 ': 'Source row ',
    '源行号': 'Source row',
    '源选择': 'Source',
    '物理量': 'Quantity',
    '等比例': 'Equal aspect',
    ' 个': ' items',
    ' 点': ' points',
    ' 项': ' items',
    '中文': 'Chinese',
    '关闭': 'Close',
    '单位': 'Unit',
    '形状': 'Shape',
    '成员': 'Member',
    '折线': 'Line',
    '散点': 'Scatter',
    '文件': 'File',
    '文本': 'Text',
    '无缝': 'overlay',
    '显示': 'Display',
    '查找': 'Find',
    '标量': 'scalar',
    '波段': 'Band',
    '清空': 'Clear',
    '类型': 'Type',
    '结构': 'Structure',
    '绘图': 'Plot',
    '网格': 'Grid',
    '行 ': 'Rows ',
    '视图': 'View',
    '计数': 'Count',
    '说明': 'Help',
    '退出': 'Quit',
    '错误': 'Error',
    '阶梯': 'Step',
    '值': 'Value',
    '列': 'Col',
    '组': 'Group',
    '行': 'Row',
    '）': ')',
    '；': '; ',
}

# 后续版本新增的界面文案（主表来自自动扫描脚本，这里补手写条目）
TRANSLATIONS.update({
    '复制路径': 'Copy path',
    '设为 X': 'Set X',
    '设为 Y': 'Set Y',
    '设为颜色': 'Set color',
    '展开此节点': 'Expand node',
    '折叠此节点': 'Collapse node',
    '在「一维光谱」页查看这一行': 'View this row in the Spectrum tab',
    '复制整行': 'Copy the whole row',
    '导出本页…': 'Export page…',
    '导出预览页': 'Export preview page',
    '每页': 'Page',
    '已复制该行到剪贴板': 'Row copied to the clipboard',
    '多源对比': 'Compare sources',
    '虚线对比': 'Dashed',
    '归一化': 'Normalise',
    '不归一化': 'None',
    '按中位流量': 'Median flux',
    '按 1σ': 'Median 1σ',
    '流量 / 中位 1σ': 'flux / median 1σ',
    '流量 / 各自中位': 'flux / own median',
    '纵轴截断至 1–99% 分位': 'y axis clipped to 1–99%',
    '    · 按 1σ 归一': '    · normalised by 1σ',
    '    · 按中位流量归一': '    · normalised by median flux',
    '加入对比': 'Add to compare',
    '显示对比': 'Show compare',
    '跳到行': 'Go to row',
    '上一页': 'Previous page',
    '下一页': 'Next page',
})


# TRANSLATIONS 里的键可能是一整句（控件文案），也可能是句子片段
# （例如 "源行 "、"有效点 "），用于翻译动态拼接出来的句子。
_PHRASES: list[tuple[str, str]] = []

_CJK_RE = None


def _has_cjk(text: str) -> bool:
    global _CJK_RE
    if _CJK_RE is None:
        import re

        _CJK_RE = re.compile(r"[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]")
    return bool(_CJK_RE.search(text))


def tf(zh: str, en: str, **kw) -> str:
    """按当前语言选模板并格式化。

    ``t()`` 只能整串/片段替换，遇到 ``f"{n} 点（源行 {row}）"`` 这类动态
    拼出来的句子就无能为力，所以这类地方改用双模板。
    """
    template = en if LANG.startswith("en") else zh
    try:
        return template.format(**kw)
    except (KeyError, IndexError, ValueError):
        return zh.format(**kw)


def refresh_phrases() -> None:
    """重建片段替换表（长键优先，避免短键先吃掉长键的一部分）。"""
    global _PHRASES
    _PHRASES = sorted(TRANSLATIONS.items(), key=lambda kv: -len(kv[0]))


def t(text):
    """取界面文案的当前语言版本。

    zh 时原样返回；en 时先精确查表，查不到再按片段替换（用于 f-string
    拼出来的动态文案，如 "源行 12/23116   波段 GU,GV,GI   无缝"）。
    """
    if not LANG.startswith("en") or not isinstance(text, str) or not text:
        return text
    hit = TRANSLATIONS.get(text)
    if hit is not None:
        return hit
    if not _has_cjk(text):
        return text
    out = text
    for zh, en in _PHRASES:
        if zh and zh in out:
            out = out.replace(zh, en)
    return out


APP_TITLE = "CESS HDF5 浏览器"

BANDS = ("GU", "GV", "GI")
BAND_COLORS = {"GU": "#1f6fb4", "GV": "#2e9e4f", "GI": "#c8342b"}
BAND_RANGES = {"GU": (2500.0, 4200.0), "GV": (3800.0, 6600.0), "GI": (5800.0, 10500.0)}

_BAND_LABELS_ZH = {
    "GU": "GU 蓝端 2500–4200 Å",
    "GV": "GV 绿端 3800–6600 Å",
    "GI": "GI 红端 5800–10500 Å",
}
BAND_LABELS: dict[str, str] = dict(_BAND_LABELS_ZH)

# 光谱标签页可显示的量：(界面文字, dataset 后缀, 单位)
_QUANTITIES_ZH = [
    ("含噪流量 flux_ujy_with_noise", "flux_ujy_with_noise", "μJy"),
    ("模型流量 flux_ujy_model", "flux_ujy_model", "μJy"),
    ("入射流量 flux_ujy", "flux_ujy", "μJy"),
    ("电子计数 flux_elec", "flux_elec", "e-"),
    ("电子率 source_electron_rate_pre_morph", "source_electron_rate_pre_morph", "electron s^-1 cm^-2"),
    ("信噪比 snr", "snr", ""),
]
QUANTITIES: list[tuple[str, str, str]] = list(_QUANTITIES_ZH)
QUANTITY_DS: dict[str, str] = {label: ds for label, ds, _ in QUANTITIES}
QUANTITY_UNIT: dict[str, str] = {label: unit for label, _, unit in QUANTITIES}

# X-Y 页的图型（界面显示值即逻辑值，双语时整表一起变）
_KINDS_ZH = ["散点", "折线", "阶梯", "1D 直方图", "2D 直方图 (hexbin)"]
KINDS: list[str] = list(_KINDS_ZH)
# 可选变量下拉里的“清空”项
NONE_CHOICE = "（不使用）"

PREVIEW_PAGE = 200

# 多源对比用的配色（灰、砖红、青、紫），与波段色区分开
COMPARE_COLORS = ("#7a7a7a", "#c2612a", "#2f7f8f", "#8a4f9e")


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------








def _shorten(text: str, limit: int = 38) -> str:
    """超长的数据集名做中间省略，保留可辨识的首尾。"""
    if len(text) <= limit:
        return text
    head = max(8, limit // 2 - 2)
    tail = max(6, limit - head - 1)
    return f"{text[:head]}…{text[-tail:]}"




def pick_cjk_font() -> str | None:
    """挑选一个既能显示中文、又不缺 Å/− 等符号的字体。

    matplotlib 会自动在 font.sans-serif 列表里做逐字回退，但首选字体最好
    本身就覆盖我们需要的字符，否则终端会刷一堆缺字形警告。
    """
    try:
        from matplotlib import font_manager
        from matplotlib.ft2font import FT2Font
    except Exception:
        return None
    candidates = (
        "Heiti SC",
        "STHeiti",
        "Arial Unicode MS",
        "PingFang SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "SimHei",
        "DejaVu Sans",
    )
    probe = "光谱流量波长源发射线Å−μσ±→"
    by_name: dict[str, str] = {}
    for font in font_manager.fontManager.ttflist:
        by_name.setdefault(font.name, font.fname)

    best = None
    best_missing = len(probe) + 1
    for name in candidates:
        path = by_name.get(name)
        if not path:
            continue
        try:
            charmap = FT2Font(path).get_charmap()
        except Exception:
            continue
        missing = sum(1 for ch in probe if ord(ch) not in charmap)
        if missing == 0:
            return name
        if missing < best_missing:
            best, best_missing = name, missing
    return best


_FONT = pick_cjk_font()
_FONT_FAMILIES: list[str] = ([_FONT] if _FONT else []) + [
    "Helvetica",
    "Arial",
    "DejaVu Sans",
]
matplotlib.rcParams["font.sans-serif"] = _FONT_FAMILIES
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.grid"] = False


def configure_language(lang: str) -> None:
    """切换界面语言，并重建所有依赖语言的常量与字体。"""
    global LANG, APP_TITLE, _FONT_FAMILIES
    LANG = "en" if str(lang).strip().lower().startswith("en") else "zh"
    APP_TITLE = t("CESS HDF5 浏览器")
    BAND_LABELS.clear()
    BAND_LABELS.update({band: t(text) for band, text in _BAND_LABELS_ZH.items()})
    QUANTITIES[:] = [(t(label), ds, unit) for label, ds, unit in _QUANTITIES_ZH]
    QUANTITY_DS.clear()
    QUANTITY_DS.update({label: ds for label, ds, _ in QUANTITIES})
    QUANTITY_UNIT.clear()
    QUANTITY_UNIT.update({label: unit for label, _, unit in QUANTITIES})
    KINDS[:] = [t(kind) for kind in _KINDS_ZH]
    if LANG.startswith("en"):
        # 拉丁字体打头；末尾仍挂一个中文字体，万一有漏译的中文也不会变成方块
        cjk = pick_cjk_font()
        _FONT_FAMILIES = ["Helvetica", "Arial", "DejaVu Sans"] + (
            [cjk] if cjk and cjk != "DejaVu Sans" else []
        )
    else:
        cjk = pick_cjk_font()
        _FONT_FAMILIES = ([cjk] if cjk else []) + ["Helvetica", "Arial", "DejaVu Sans"]
    matplotlib.rcParams["font.sans-serif"] = _FONT_FAMILIES
    # 拉丁字体自带 U+2212，用真正的减号更好看
    matplotlib.rcParams["axes.unicode_minus"] = LANG.startswith("en")
    refresh_phrases()
    install_text_hooks()


def apply_tk_fonts(root) -> None:
    """Update Tk fonts on every language switch, including a return to Chinese."""
    try:
        import tkinter.font as tkfont
    except Exception:
        return
    family = "Helvetica" if LANG.startswith("en") else (pick_cjk_font() or "TkDefaultFont")
    for name in (
        "TkDefaultFont", "TkTextFont", "TkMenuFont",
        "TkHeadingFont", "TkCaptionFont", "TkTooltipFont",
    ):
        try:
            tkfont.nametofont(name).configure(family=family)
        except Exception:
            pass


# --------------------------------------------------------------------------
# 文案出口钩子：让 t() 自动作用到所有控件与图上文字，不必逐处改调用
# --------------------------------------------------------------------------

_HOOKED = False
_TEXT_KEYS = ("text", "label", "title", "message", "s", "xlabel", "ylabel")


def _tr_kwargs(kwargs: dict) -> dict:
    if not LANG.startswith("en") or not kwargs:
        return kwargs
    for key in _TEXT_KEYS:
        value = kwargs.get(key)
        if isinstance(value, str) and value:
            kwargs[key] = t(value)
    labels = kwargs.get("labels")
    if isinstance(labels, (list, tuple)) and labels:
        kwargs["labels"] = type(labels)(
            t(item) if isinstance(item, str) else item for item in labels
        )
    values = kwargs.get("values")
    if isinstance(values, (list, tuple)) and values:
        kwargs["values"] = type(values)(
            t(item) if isinstance(item, str) else item for item in values
        )
    return kwargs


def _wrap_method(cls, name: str, positional: bool = False) -> None:
    """包装一个方法，翻译它的文字参数。

    ``positional=True`` 仅用于「位置参数本身就是显示文字」的绘图接口
    （如 ``ax.text(x, y, "文字")``）；Tk 的 Treeview/Notebook 等方法第一个
    位置参数是控件 ID，翻译了会变成无效索引，所以默认不碰位置参数。
    """
    original = getattr(cls, name, None)
    if original is None:
        return
    if getattr(original, "_i18n_hooked", False):
        if not positional or getattr(original, "_i18n_positional", False):
            return
        original = original.__wrapped__

    def wrapper(self, *args, **kwargs):
        if LANG.startswith("en"):
            if positional:
                args = tuple(t(a) if isinstance(a, str) else a for a in args)
            kwargs = _tr_kwargs(dict(kwargs))
        return original(self, *args, **kwargs)

    wrapper._i18n_hooked = True
    wrapper._i18n_positional = positional
    wrapper.__wrapped__ = original
    setattr(cls, name, wrapper)


def install_text_hooks() -> None:
    """把 t() 挂到控件与绘图 API 上（幂等，可重复调用）。"""
    global _HOOKED
    if _HOOKED:
        return
    _HOOKED = True

    widget_classes = (
        tk.Label, ttk.Label, tk.Button, ttk.Button, tk.Checkbutton,
        ttk.Checkbutton, tk.Radiobutton, ttk.Radiobutton, ttk.LabelFrame,
        tk.Message, ttk.Menubutton,
    )
    for cls in widget_classes:
        for name in ("configure", "config"):
            _wrap_method(cls, name)

    for name in ("add_command", "add_cascade", "add_checkbutton", "add_radiobutton"):
        _wrap_method(tk.Menu, name)
    _wrap_method(ttk.Notebook, "add")
    for name in ("heading", "column", "insert", "set", "item"):
        _wrap_method(ttk.Treeview, name)
    for cls in (tk.Tk, tk.Toplevel):
        _wrap_method(cls, "title")

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure as MplFigure
    from matplotlib.colorbar import Colorbar

    for name in ("set_title", "set_xlabel", "set_ylabel", "text", "annotate",
                 "plot", "scatter", "step", "fill_between", "errorbar",
                 "axvline", "axhline", "legend", "hist", "hexbin"):
        _wrap_method(Axes, name)
    for name in ("suptitle", "text", "legend"):
        _wrap_method(MplFigure, name)
    # 这些接口的位置参数就是图上文字
    for name in ("text", "annotate", "legend", "set_title", "set_xlabel", "set_ylabel"):
        _wrap_method(Axes, name, positional=True)
    for name in ("suptitle", "text", "legend"):
        _wrap_method(MplFigure, name, positional=True)
    _wrap_method(Colorbar, "set_label", positional=True)

    for name in ("showinfo", "showwarning", "showerror", "askyesno",
                 "askokcancel", "askquestion"):
        original = getattr(messagebox, name, None)
        if original is None or getattr(original, "_i18n_hooked", False):
            continue

        def make(func):
            def wrapper(title=None, message=None, **options):
                return func(t(title) if isinstance(title, str) else title,
                            t(message) if isinstance(message, str) else message,
                            **options)

            wrapper._i18n_hooked = True
            return wrapper

        setattr(messagebox, name, make(original))


def localize_tree(widget) -> None:
    """把已建好的控件树里的文案翻成目标语言。

    Tk 控件在构造时传的 ``text=`` 不走 configure()，钩子拦不到，所以界面
    建好后再扫一遍；之后动态更新的文案由 configure 钩子负责。
    """
    if not LANG.startswith("en"):
        return
    try:
        text = widget.cget("text")
    except Exception:
        text = None
    if isinstance(text, str) and text:
        try:
            widget.configure(text=t(text))
        except Exception:
            pass

    if isinstance(widget, tk.Menu):
        try:
            last = widget.index("end")
        except Exception:
            last = None
        if last is not None:
            for index in range(last + 1):
                try:
                    label = widget.entrycget(index, "label")
                except Exception:
                    continue
                if label:
                    try:
                        widget.entryconfigure(index, label=t(label))
                    except Exception:
                        pass
    elif isinstance(widget, ttk.Notebook):
        try:
            for tab_id in widget.tabs():
                label = widget.tab(tab_id, "text")
                if label:
                    widget.tab(tab_id, text=t(label))
        except Exception:
            pass
    elif isinstance(widget, (tk.Tk, tk.Toplevel)):
        try:
            title = widget.title()
            if title:
                widget.title(t(title))
        except Exception:
            pass

    try:
        children = widget.winfo_children()
    except Exception:
        return
    for child in children:
        localize_tree(child)


def safe_tight_layout(fig) -> None:
    """tight_layout 在坐标轴范围极端时会抛异常，绘图不该因此中断。"""
    try:
        fig.tight_layout()
    except Exception:
        pass


def robust_ylim(ax, values, qlo: float = 0.5, qhi: float = 99.5,
                ratio: float = 3.0) -> bool:
    """当数据里混有极端离群值时，把 y 轴收窄到分位数范围。

    只在实际范围比分位数跨度大 ``ratio`` 倍以上时才调整，正常数据不动。
    返回是否真的收窄了（调用方据此提示“纵轴已截断”）。
    """
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size < 20:
        return False
    lo, hi = (float(v) for v in np.percentile(vals, [qlo, qhi]))
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return False
    full = float(vals.max() - vals.min())
    if not np.isfinite(full) or full <= ratio * span:
        return False
    pad = 0.12 * span
    ax.set_ylim(lo - pad, hi + pad)
    return True


def percentile_limits(x: np.ndarray, y: np.ndarray, qlo: float = 0.2, qhi: float = 99.8):
    """返回 (xlim, ylim)，用分位数忽略极端离群点。"""
    def lim(arr):
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        lo, hi = (float(v) for v in np.percentile(vals, [qlo, qhi]))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if lo == hi:
            return (lo - 1.0, hi + 1.0)
        pad = 0.05 * (hi - lo)
        return (lo - pad, hi + pad)

    return lim(x), lim(y)


# --------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------






# --------------------------------------------------------------------------
# 通用表格预览控件
# --------------------------------------------------------------------------


class TablePreview(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.headers: list[str] = []
        self.rows: list[list] = []
        self.on_header_click = None
        self.on_row_click = None
        self.tree = ttk.Treeview(self, show="headings", height=9)
        yscroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.tree.bind("<Button-1>", self._on_click, add="+")
        self.tree.bind("<Button-2>", self._on_right_click, add="+")
        self.tree.bind("<Button-3>", self._on_right_click, add="+")
        self.tree.bind("<Control-Button-1>", self._on_right_click, add="+")

    def _on_click(self, event):
        if self.tree.identify_region(event.x, event.y) == "heading":
            column = self.tree.identify_column(event.x)
            try:
                index = int(column.lstrip("#")) - 1
            except ValueError:
                return
            if self.on_header_click:
                self.on_header_click(index)

    def _on_right_click(self, event):
        if not self.on_row_click:
            return
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        self.tree.selection_set(iid)
        self.tree.focus(iid)
        self.on_row_click(event)

    def show(self, headers, rows, widths=None):
        self.tree.delete(*self.tree.get_children())
        headers = [str(h) for h in headers]
        self.headers = headers
        self.rows = [list(values) for values in rows]
        self.tree.configure(columns=headers)
        for i, header in enumerate(headers):
            self.tree.heading(header, text=header)
            width = widths[i] if widths and i < len(widths) else max(70, min(240, 12 + 9 * len(header)))
            self.tree.column(header, width=width, anchor="w", stretch=False)
        for values in self.rows:
            self.tree.insert("", "end", values=[fmt_num(v) for v in values])

    def clear(self):
        self.tree.delete(*self.tree.get_children())
        self.headers = []
        self.rows = []


# --------------------------------------------------------------------------
# 标签页 1：结构与数据
# --------------------------------------------------------------------------


class StructureTab(ttk.Frame):
    def __init__(self, master, app: "App"):
        super().__init__(master, padding=4)
        self.app = app
        self.current_path: str | None = None
        self.offset = 0
        self._selecting = False
        self.columns: list[tuple[str, str, int | None]] = []  # (显示名, kind, index)
        self._view = None        # 当前预览类型，表头点击排序时用
        self._sort = None        # (key, ascending, 列名)

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)
        self.split_paned = paned

        # ---------- 左：树 ----------
        left = ttk.Frame(paned, padding=(0, 0, 4, 0))
        paned.add(left, weight=1)

        top = ttk.Frame(left)
        top.pack(fill="x")
        ttk.Label(top, text="过滤路径:").pack(side="left")
        self.filter_var = tk.StringVar()
        entry = ttk.Entry(top, textvariable=self.filter_var)
        entry.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(top, text="清空", width=6, command=lambda: self.filter_var.set("")).pack(side="left")
        self.filter_var.trace_add("write", lambda *_: self._apply_filter())

        tree_wrap = ttk.Frame(left)
        tree_wrap.pack(fill="both", expand=True, pady=4)
        self.tree = ttk.Treeview(
            tree_wrap, columns=("shape", "dtype"), show="tree headings", selectmode="browse"
        )
        self.tree.heading("#0", text="结构")
        self.tree.heading("shape", text="shape / 成员")
        self.tree.heading("dtype", text="dtype")
        self.tree.column("#0", width=220, stretch=True)
        self.tree.column("shape", width=105, anchor="w", stretch=False)
        self.tree.column("dtype", width=95, anchor="w", stretch=False)
        yscroll = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(tree_wrap, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree_wrap.rowconfigure(0, weight=1)
        tree_wrap.columnconfigure(0, weight=1)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewOpen>>", self._on_open)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Button-2>", self._on_tree_menu)
        self.tree.bind("<Button-3>", self._on_tree_menu)
        self.tree.bind("<Control-Button-1>", self._on_tree_menu)
        self.menu_path: str | None = None

        # ---------- 右：信息 + 列 + 预览（三块之间可上下拖动分隔） ----------
        right = ttk.PanedWindow(paned, orient="vertical")
        paned.add(right, weight=3)
        self.right_panes = right

        info_box = ttk.LabelFrame(right, text="对象信息", padding=4)
        right.add(info_box, weight=2)
        info_bar = ttk.Frame(info_box)
        info_bar.pack(side="top", fill="x", pady=(0, 2))
        ttk.Button(info_bar, text=tf("复制", "Copy"), width=7,
                   command=self._copy_info).pack(side="right")
        ttk.Button(info_bar, text="A+", width=3,
                   command=lambda: self._zoom_info(1)).pack(side="right", padx=(0, 4))
        ttk.Button(info_bar, text="A−", width=3,
                   command=lambda: self._zoom_info(-1)).pack(side="right", padx=(0, 2))
        self._info_size = 10
        self.info = tk.Text(info_box, height=7, wrap="word", font=("Menlo", self._info_size), borderwidth=0, highlightthickness=0, background=PANEL, foreground=TEXT, padx=8, pady=6)
        info_scroll = ttk.Scrollbar(info_box, orient="vertical", command=self.info.yview)
        self.info.configure(yscrollcommand=info_scroll.set, state="disabled")
        info_scroll.pack(side="right", fill="y")
        self.info.pack(side="left", fill="both", expand=True)

        middle = ttk.Frame(right, padding=(0, 4))
        right.add(middle, weight=0)
        col_box = ttk.LabelFrame(middle, text="列 / 像素 / 字段（可多选两列）", padding=4)
        col_box.pack(fill="both", expand=True)
        self.col_list = tk.Listbox(
            col_box, height=4, selectmode="extended", exportselection=False,
            font=("Menlo", 10), borderwidth=0, highlightthickness=0, background=PANEL, foreground=TEXT, activestyle="none",
        )
        col_scroll = ttk.Scrollbar(col_box, orient="vertical", command=self.col_list.yview)
        self.col_list.configure(yscrollcommand=col_scroll.set)
        self.col_list.pack(side="left", fill="both", expand=True)
        col_scroll.pack(side="right", fill="y")
        self.col_list.bind("<<ListboxSelect>>", self._on_column_select)

        btn = ttk.Frame(middle)
        btn.pack(fill="x", pady=(4, 0))
        for i, (label, callback) in enumerate((
            (tf("设为 X", "Set X"), lambda: self._assign("x")),
            (tf("设为 Y", "Set Y"), lambda: self._assign("y")),
            (tf("绘制两列", "Plot columns"), self._plot_two),
            (tf("设为颜色", "Set color"), lambda: self._assign("c")))):
            btn.columnconfigure(i % 2, weight=1)
            ttk.Button(btn, text=label, command=callback).grid(
                row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)

        preview_box = ttk.LabelFrame(right, text="数据预览", padding=4)
        right.add(preview_box, weight=5)
        self.preview = TablePreview(preview_box)
        self.preview.pack(fill="both", expand=True)
        self.preview.on_header_click = self._on_header_click
        self.preview.on_row_click = self._on_preview_menu
        nav = ttk.Frame(preview_box)
        nav.pack(fill="x", pady=(4, 0))
        ttk.Button(nav, text="◀ 上一页", command=lambda: self._page(-1)).pack(side="left")
        ttk.Button(nav, text="下一页 ▶", command=lambda: self._page(1)).pack(side="left", padx=4)
        self.range_label = ttk.Label(nav, text="")
        self.range_label.pack(side="left", padx=8, fill="x", expand=True)

        nav2 = ttk.Frame(preview_box)
        nav2.pack(fill="x", pady=(2, 0))
        # 长表按 200 行翻页太慢，给一个直接跳行的入口
        ttk.Label(nav2, text=tf("跳到行", "Go to row")).pack(side="left", padx=(0, 2))
        self.jump_var = tk.StringVar()
        self.jump_entry = ttk.Entry(nav2, textvariable=self.jump_var, width=8)
        self.jump_entry.pack(side="left")
        self.jump_entry.bind("<Return>", self._jump_to_row)
        self.jump_entry.bind("<KP_Enter>", self._jump_to_row)
        ttk.Label(nav2, text=tf("每页", "Page")).pack(side="left", padx=(12, 2))
        self.page_var = tk.StringVar(value=str(PREVIEW_PAGE))
        page_box = ttk.Combobox(nav2, textvariable=self.page_var, width=6, state="readonly",
                                values=("200", "500", "1000", "2000"))
        page_box.pack(side="left")
        page_box.bind("<<ComboboxSelected>>", lambda _e: self._on_page_size())
        ttk.Button(nav2, text=tf("导出本页…", "Export page…"),
                   command=self.export_page).pack(side="right")
        # 统计单独一行，面板变窄时不会被挤掉
        self.stats_label = ttk.Label(preview_box, text="", foreground="#555", anchor="w")
        self.stats_label.pack(fill="x")

        # 拖动分隔条后别让任何一块被压成 0 高（等几何更新完成再校正）
        right.bind("<ButtonRelease-1>", self._on_sash_release, add="+")

        self._placeholder = "（请选择左侧对象）"
        self.preview.show(["—"], [])
        self._build_menus()

    # ---- 树操作 ----
    _PANE_MIN = 56

    def _on_sash_release(self, _event=None):
        self.after_idle(self._clamp_right_panes)

    def _zoom_info(self, delta: int):
        """调整对象信息字号（长属性文本需要放大才好读）。"""
        self._info_size = max(8, min(18, self._info_size + delta))
        self.info.configure(font=("Menlo", self._info_size))

    def _copy_info(self):
        text = self.info.get("1.0", "end").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.app.status.configure(
            text=tf("已复制对象信息到剪贴板", "Object information copied to the clipboard"))

    def _clamp_right_panes(self, _event=None):
        """拖动分隔条后保证每块都还有一点高度（TTK 允许把窗格拖成 0）。"""
        if getattr(self, "_clamping", False):
            return
        panes = self.right_panes.panes()
        if len(panes) < 2 or not self.right_panes.winfo_ismapped():
            return
        total = self.right_panes.winfo_height()
        last = len(panes) - 1
        if total < self._PANE_MIN * len(panes):
            return
        self._clamping = True
        try:
            for i in range(last):
                low = (self.right_panes.sashpos(i - 1) if i else 0) + self._PANE_MIN
                high = total - self._PANE_MIN * (last - i)
                pos = self.right_panes.sashpos(i)
                target = max(low, min(pos, high))
                if target != pos:
                    self.right_panes.sashpos(i, int(target))
        except tk.TclError:
            pass
        finally:
            self._clamping = False

    def clear(self):
        self.tree.delete(*self.tree.get_children())
        self.col_list.delete(0, "end")
        self.columns = []
        self.current_path = None
        self._set_info("")

    def populate(self):
        self.tree.delete(*self.tree.get_children())
        model = self.app.model
        if model is None:
            return
        self._add_children("", "")
        # 展开根的子组
        for iid in self.tree.get_children(""):
            self.tree.item(iid, open=False)

    def _add_children(self, parent_iid: str, parent_path: str):
        model = self.app.model
        for full, name, obj in model.children(parent_path):
            if isinstance(obj, h5py.Dataset):
                shape = str(tuple(obj.shape)) if obj.shape != () else "标量"
                iid = self.tree.insert(
                    parent_iid, "end", iid=full, text=name, values=(shape, str(obj.dtype))
                )
            else:
                iid = self.tree.insert(
                    parent_iid, "end", iid=full, text=name + "/", values=(f"{len(obj.keys())} 项", "")
                )
                self.tree.insert(iid, "end", iid=f"{full}||placeholder", text="…")

    def _on_open(self, _event=None):
        iid = self.tree.focus()
        if not iid or iid.endswith("||placeholder"):
            return
        children = self.tree.get_children(iid)
        if len(children) == 1 and children[0].endswith("||placeholder"):
            self.tree.delete(children[0])
            self._add_children(iid, iid)

    def _apply_filter(self):
        model = self.app.model
        if model is None:
            return
        text = self.filter_var.get().strip().lower()
        self.tree.delete(*self.tree.get_children())
        if not text:
            self.populate()
            return
        hits = [p for p in model.dataset_paths() if text in p.lower()]
        hits = hits[:600]
        for path in hits:
            ds = model.get(path)
            shape = str(tuple(ds.shape)) if ds.shape != () else "标量"
            self.tree.insert("", "end", iid=path, text=path, values=(shape, str(ds.dtype)))
        if not hits:
            self.tree.insert("", "end", text="（无匹配数据集）", values=("", ""))

    def _on_select(self, _event=None):
        """用户点击树节点：只负责显示，绝不回写 selection（否则事件自激）."""
        if self._selecting:
            return
        selection = self.tree.selection()
        if not selection:
            return
        iid = selection[0]
        if iid.endswith("||placeholder") or iid not in self.app.model.h5:
            return
        self._show_path(iid)

    def select_path(self, path: str):
        """外部调用：把树选中到某路径，并显示它。"""
        if self._selecting:
            return
        self._selecting = True
        try:
            try:
                if self.tree.exists(path) and path not in self.tree.selection():
                    self.tree.selection_set(path)
                    self.tree.focus(path)
                    self.tree.see(path)
            except Exception:
                pass
            self._show_path(path)
        finally:
            self._selecting = False

    def _show_path(self, path: str):
        model = self.app.model
        self.current_path = path
        self.offset = 0
        self._sort = None
        self._view = None
        self._set_info(model.info_text(path))
        self._fill_columns(path)
        self._render_preview()

    # ---- 右键菜单 ----
    def _build_menus(self):
        self.menu = tk.Menu(self, tearoff=0, postcommand=self._menu_post)
        self.menu.add_command(label=t("复制路径"), command=self._menu_copy_path)
        self.menu.add_separator()
        self.menu.add_command(label=t("设为 X"), command=lambda: self._assign_path("x"))
        self.menu.add_command(label=t("设为 Y"), command=lambda: self._assign_path("y"))
        self.menu.add_command(label=t("设为颜色"), command=lambda: self._assign_path("c"))
        self.menu.add_separator()
        self.menu.add_command(label=t("展开此节点"), command=lambda: self._menu_toggle(True))
        self.menu.add_command(label=t("折叠此节点"), command=lambda: self._menu_toggle(False))
        self.preview.menu = tk.Menu(self, tearoff=0)
        self.preview.menu.add_command(label=t("在「一维光谱」页查看这一行"),
                                      command=self._menu_show_row)
        self.preview.menu.add_command(label=t("复制整行"), command=self._menu_copy_row)

    def _menu_post(self):
        model = self.app.model
        path = self.menu_path
        state = "normal" if path and model and isinstance(model.get(path), h5py.Dataset) else "disabled"
        for i in range(2, 5):
            try:
                self.menu.entryconfigure(i, state=state)
            except tk.TclError:
                pass

    def _on_tree_menu(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid or iid.endswith("||placeholder"):
            return
        self.menu_path = iid
        try:
            self.tree.selection_set(iid)
            self.tree.focus(iid)
        except tk.TclError:
            pass
        self._show_path(iid)
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _menu_copy_path(self):
        if not self.menu_path:
            return
        self.clipboard_clear()
        self.clipboard_append("/" + self.menu_path)
        self.app.status.configure(text=tf("已复制路径 /{p}", "Path copied: /{p}", p=self.menu_path))

    def _menu_toggle(self, open_state: bool):
        if not self.menu_path:
            return
        try:
            self.tree.item(self.menu_path, open=open_state)
            if open_state:
                self.tree.focus(self.menu_path)
                self._on_open()
        except tk.TclError:
            pass

    def _menu_show_row(self):
        item = self.preview.tree.focus() or (
            self.preview.tree.selection()[0] if self.preview.tree.selection() else "")
        if not item:
            return
        values = self.preview.tree.item(item, "values")
        if not values:
            return
        try:
            row = int(float(values[0]))
        except (TypeError, ValueError):
            return
        model = self.app.model
        if model is None or not (0 <= row < model.nsrc):
            messagebox.showinfo(APP_TITLE, tf("这一行不是有效的源行号。",
                                              "This row is not a valid source row."), parent=self)
            return
        self.app.row_var.set(row)
        self.app.notebook.select(self.app.spec)

    def _menu_copy_row(self):
        item = self.preview.tree.focus() or (
            self.preview.tree.selection()[0] if self.preview.tree.selection() else "")
        if not item:
            return
        values = self.preview.tree.item(item, "values")
        text = "\t".join(str(v) for v in values)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.app.status.configure(text=t("已复制该行到剪贴板"))

    def _variable_for_path(self, path: str):
        """给右键菜单用：把整个数据集当作绘图变量。"""
        model = self.app.model
        obj = model.get(path)
        if not isinstance(obj, h5py.Dataset):
            return None
        if obj.ndim == 1:
            for spec in model.variable_map().values():
                if spec.path == path and spec.kind == "vec":
                    return spec
            return Variable(path, path, "vec", None, model.unit_of(path), int(obj.shape[0]))
        if obj.ndim == 2:
            selection = self.col_list.curselection()
            index = int(selection[0]) if selection else 0
            return model.column_variable(path, index)
        return None

    def _assign_path(self, axis: str):
        path = self.menu_path or self.current_path
        if not path:
            return
        spec = None
        if path == self.current_path:
            selected = self._selected_columns()
            if selected:
                spec = self._variable_for(selected[0])
        if spec is None:
            spec = self._variable_for_path(path)
        if spec is None:
            messagebox.showinfo(APP_TITLE, t("该对象不支持作为绘图变量。"), parent=self)
            return
        self.app.xy.adopt(spec, axis)
        self.app.notebook.select(self.app.xy)

    # ---- 列列表 ----
    def _fill_columns(self, path: str):
        model = self.app.model
        obj = model.get(path)
        self.columns = []
        default_index = 0
        self.col_list.delete(0, "end")
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
            if obj.dtype.names:
                for name in obj.dtype.names:
                    self.columns.append((f"{name}", "field", name))
            elif obj.ndim == 2:
                columns = model.table_columns(path)
                if columns:
                    for j, (name, unit) in enumerate(columns):
                        self.columns.append((f"[{j:2d}] {name}" + (f"  [{unit}]" if unit else ""), "col", j))
                else:
                    labels = model.pixel_labels(path, int(obj.shape[1]))
                    for j, label in enumerate(labels):
                        self.columns.append((f"[{j:3d}] {label}", "col", j))
                    # 光谱这类“像素列”首尾常有全 NaN 的冗余像元，默认选中间一列
                    default_index = len(self.columns) // 2
            for i, (label, _kind, _idx) in enumerate(self.columns):
                self.col_list.insert("end", label)
            if self.columns:
                self.col_list.selection_set(min(default_index, len(self.columns) - 1))

    def _on_column_select(self, _event=None):
        self._sort = None
        self._render_preview()

    def _selected_columns(self):
        return [self.columns[i] for i in self.col_list.curselection() if i < len(self.columns)]

    def _variable_for(self, entry) -> Variable | None:
        label, kind, index = entry
        path = self.current_path
        if path is None:
            return None
        if kind == "col":
            return self.app.model.column_variable(path, int(index))
        if kind == "field":
            text = f"{path} :: {index}"
            spec = self.app.model.variable_map().get(text)
            if spec is None:
                ds = self.app.model.get(path)
                spec = Variable(text, path, "field", index, "", int(ds.shape[0]))
            return spec
        return None

    def _on_preview_menu(self, event):
        try:
            self.preview.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.preview.menu.grab_release()

    # ---- 每页行数 / 排序 / 导出 ----
    def _page_size(self) -> int:
        try:
            return max(20, int(self.page_var.get()))
        except (TypeError, ValueError):
            return PREVIEW_PAGE

    def _on_page_size(self):
        if self.current_path is None:
            return
        size = self._page_size()
        self.offset = (self.offset // size) * size
        self._render_preview()

    def _on_header_click(self, index: int):
        """点击表头：按该列排序；再点一次反向；第 0 列取消排序。"""
        view = getattr(self, "_view", None)
        if not view or view[0] not in ("vector", "column", "matrix", "structured"):
            return
        if index <= 0:
            self._sort = None
            self.offset = 0
            self._render_preview()
            return
        kind = view[0]
        if kind in ("vector", "column"):
            key = ("value", None)
        elif kind == "matrix":
            key = ("col", index - 1)
        else:
            headers = self.preview.headers
            key = ("field", headers[index] if index < len(headers) else headers[-1])
        label = self.preview.headers[index] if index < len(self.preview.headers) else ""
        if self._sort is not None and self._sort[0] == key:
            self._sort = (key, not self._sort[1], label)
        else:
            self._sort = (key, True, label)
        self.offset = 0
        self._render_preview()

    def _sort_values(self, obj, key):
        name, index = key
        if name == "col":
            return np.asarray(obj[:, int(index)], dtype=float)
        if name == "field":
            if obj.dtype.names:
                return np.asarray(obj.fields(index)[:], dtype=float)
            return np.asarray(obj[:], dtype=float)
        # key == "value"：排序对象就是当前预览的那一列
        view = self._view or ("", None, None)
        if view[0] == "column" and view[2] is not None:
            extra = view[2]
            if isinstance(extra, str):
                return np.asarray(obj.fields(extra)[:], dtype=float)
            return np.asarray(obj[:, int(extra)], dtype=float)
        return np.asarray(obj[:], dtype=float)

    def _preview_sorted(self, path: str, obj):
        """把整列读进内存后排序，再按页显示（行号列仍是原始行号）。"""
        model = self.app.model
        n = int(obj.shape[0])
        if n > 5_000_000:
            messagebox.showinfo(
                APP_TITLE,
                tf("这个数据集太大（{n} 行），暂不支持排序。", "Dataset too large to sort ({n} rows).", n=n),
                parent=self)
            self._sort = None
            return False
        key, ascending, label = self._sort
        try:
            values = self._sort_values(obj, key)
        except Exception:
            self._sort = None
            return False
        if values.size != n:
            self._sort = None
            return False
        finite = np.isfinite(values)
        idx = np.flatnonzero(finite)
        ordered = idx[np.argsort(values[idx], kind="stable")]
        if not ascending:
            ordered = ordered[::-1]
        order = np.concatenate([ordered, np.flatnonzero(~finite)])
        start = min(self.offset, max(0, n - 1))
        stop = min(n, start + self._page_size())
        rows = [[int(order[i]), float(values[order[i]])] for i in range(start, stop)]
        self.preview.show([t("行"), t("值")], rows, widths=[70, 220])
        arrow = tf("升序", "asc") if ascending else tf("降序", "desc")
        self.range_label.configure(
            text=tf("{label} {arrow} · 行 {start}–{stop} / 共 {n}",
                    "{label} {arrow} · rows {start}–{stop} of {n}",
                    label=label, arrow=arrow, start=start, stop=stop - 1, n=n))
        self.stats_label.configure(text=self._stats_text(values))
        return True

    def export_page(self):
        """把预览当前页导出为 CSV（与「导出数据」同样的写入保护）。"""
        rows = getattr(self.preview, "rows", [])
        headers = getattr(self.preview, "headers", [])
        if not rows:
            messagebox.showinfo(APP_TITLE, tf("当前页没有数据。", "Nothing to export on this page."),
                                parent=self)
            return
        model = self.app.model
        stem = (self.current_path or "page").replace("/", "_")
        path = filedialog.asksaveasfilename(
            parent=self, title=t("导出预览页"),
            defaultextension=".csv",
            initialfile=f"{stem}.csv",
            filetypes=[("CSV", "*.csv"), ("文本", "*.txt"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            write_csv(path, [t("行"), t("值")] if self._sort else [t(h) for h in headers],
                      rows, model.path if model else None)
        except (OSError, ValueError) as exc:
            messagebox.showerror(APP_TITLE, tf("写入失败：{err}", "Write failed: {err}", err=exc),
                                 parent=self)
            return
        self.app.status.configure(text=tf("已导出 {n} 行 → {path}", "Exported {n} rows → {path}",
                                          n=len(rows), path=path))

    # ---- 预览 ----
    def _render_preview(self):
        model = self.app.model
        path = self.current_path
        if model is None or path is None:
            return
        obj = model.get(path)
        selected = self._selected_columns()
        try:
            if self._sort is not None and self._preview_sorted(path, obj):
                return
            if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and obj.dtype.names:
                self._preview_structured(obj)
            elif isinstance(obj, h5py.Group):
                self._preview_group(path, obj)
            elif selected:
                label, kind, index = selected[-1]
                self._preview_column(path, obj, kind, index, label)
            elif isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                self._preview_matrix(path, obj)
            elif isinstance(obj, h5py.Dataset) and obj.ndim == 1:
                self._preview_vector(path, obj)
            else:
                self.preview.clear()
                self.range_label.configure(text="")
                self.stats_label.configure(text="")
        except Exception as exc:  # 预览失败不应中断整个界面
            self.preview.show(["错误"], [[str(exc)]])
            self.stats_label.configure(text="读取失败")

    def _read_slice(self, path: str, key):
        try:
            return np.asarray(self.app.model.h5[path][key])
        except Exception:
            return np.array([])

    def _preview_vector(self, path: str, obj):
        self._view = ("vector", path, None)
        n = int(obj.shape[0])
        stop = min(n, self.offset + self._page_size())
        start = min(self.offset, max(0, n - 1))
        values = self._read_slice(path, slice(start, stop))
        rows = [[start + i, v] for i, v in enumerate(values)]
        self.preview.show(["行", "值"], rows, widths=[70, 220])
        self.range_label.configure(text=f"行 {start}–{stop - 1} / 共 {n}")
        self.stats_label.configure(text=self._stats_text(np.asarray(values)))

    def _preview_group(self, path: str, obj):
        """组：列出直接成员，当作“目录”来看。"""
        self._view = ("group", path, None)
        model = self.app.model
        names = sorted(obj.keys())[:200]
        rows = []
        nds = 0
        for name in names:
            sub = obj[name]
            child = f"{path}/{name}" if path else name
            if isinstance(sub, h5py.Dataset):
                nds += 1
                shape = str(tuple(sub.shape)) if sub.shape != () else "标量"
                rows.append([name, "数据集", shape, str(sub.dtype),
                             model.unit_of(child)])
            else:
                rows.append([name, "组", f"{len(sub.keys())} 项", "", ""])
        self.preview.show(["成员", "类型", "形状", "数据类型", "单位"], rows,
                          widths=[190, 60, 110, 110, 80])
        self.range_label.configure(text=f"直接成员 {len(obj.keys())} 个")
        self.stats_label.configure(text=f"其中数据集 {nds} 个")

    def _preview_structured(self, obj):
        self._view = ("structured", self.current_path, None)
        names = list(obj.dtype.names)
        n = int(obj.shape[0])
        stop = min(n, self.offset + self._page_size())
        start = min(self.offset, max(0, n - 1))
        selected = self._selected_columns()
        fields = [selected[-1][2]] if selected else names
        data = obj[start:stop]
        rows = [[i] + [data[field][i - start] for field in fields] for i in range(start, stop)]
        self.preview.show(["行"] + list(fields), rows)
        self.range_label.configure(text=f"行 {start}–{stop - 1} / 共 {n}")
        self.stats_label.configure(text=f"字段 {len(fields)} 个")

    def _preview_column(self, path: str, obj, kind, index, label):
        self._view = ("column", path, index)
        n = int(obj.shape[0])
        start = min(self.offset, max(0, n - 1))
        stop = min(n, start + self._page_size())
        if obj.dtype.names:
            chunk = np.asarray(obj.fields(index)[start:stop])
        elif obj.ndim == 2:
            chunk = self._read_slice(path, (slice(start, stop), int(index)))
        else:
            chunk = self._read_slice(path, slice(start, stop))
        rows = [[start + i, v] for i, v in enumerate(chunk)]
        self.preview.show(["行", "值"], rows, widths=[70, 220])
        self.range_label.configure(text=f"行 {start}–{stop - 1} / 共 {n}")
        self.stats_label.configure(text=self._stats_text(chunk))

    def _preview_matrix(self, path: str, obj):
        self._view = ("matrix", path, None)
        n, ncol = int(obj.shape[0]), int(obj.shape[1])
        stop = min(n, self.offset + self._page_size())
        start = min(self.offset, max(0, n - 1))
        show_cols = min(ncol, 12)
        block = self._read_slice(path, (slice(start, stop), slice(0, show_cols)))
        headers = ["行"] + [f"列{j}" for j in range(show_cols)]
        rows = [[start + i] + list(block[i]) for i in range(block.shape[0])]
        self.preview.show(headers, rows)
        note = "" if ncol <= show_cols else f"（仅显示前 {show_cols}/{ncol} 列）"
        self.range_label.configure(text=f"行 {start}–{stop - 1} / 共 {n} {note}")
        self.stats_label.configure(text=f"{n} × {ncol}")

    @staticmethod
    def _stats_text(data: np.ndarray) -> str:
        if data.size == 0:
            return ""
        if data.dtype.kind in "fiub":
            arr = data.astype(float, copy=False)
            finite = np.isfinite(arr)
            count = int(finite.sum())
            if count == 0:
                return f"N={arr.size}  全为 NaN"
            good = arr[finite]
            return (
                f"有效 {count}/{arr.size} · min {fmt_num(good.min())} · "
                f"中位 {fmt_num(np.median(good))} · max {fmt_num(good.max())}"
            )
        return f"N={data.size}"

    def _page(self, delta: int):
        if self.current_path is None:
            return
        obj = self.app.model.get(self.current_path)
        n = int(obj.shape[0]) if isinstance(obj, h5py.Dataset) and obj.ndim >= 1 else 0
        self.offset = max(0, min(self.offset + delta * self._page_size(), max(0, n - 1)))
        self._render_preview()

    def _jump_to_row(self, _event=None):
        """把预览跳到包含指定行号的那一页。"""
        if self.current_path is None:
            return
        obj = self.app.model.get(self.current_path)
        n = int(obj.shape[0]) if isinstance(obj, h5py.Dataset) and obj.ndim >= 1 else 0
        if n <= 0:
            return
        try:
            row = int(float(self.jump_var.get()))
        except (TypeError, ValueError):
            return
        row = max(0, min(row, n - 1))
        size = max(1, self._page_size())
        self.offset = (row // size) * size
        self._render_preview()
        self.jump_var.set("")

    def _set_info(self, text: str):
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", t(text))
        self.info.configure(state="disabled")

    # ---- 变量指派 ----
    def _assign(self, axis: str):
        selected = self._selected_columns()
        if not selected:
            messagebox.showinfo(APP_TITLE, "请先在「列 / 像素 / 字段」列表里选择一列。", parent=self)
            return
        spec = self._variable_for(selected[0])
        if spec is None:
            messagebox.showinfo(APP_TITLE, "该对象不支持作为绘图变量。", parent=self)
            return
        self.app.xy.adopt(spec, axis)
        self.app.notebook.select(self.app.xy)

    def _plot_two(self):
        selected = self._selected_columns()
        if len(selected) < 2:
            messagebox.showinfo(
                APP_TITLE,
                "请按住 ⌘ / Shift 在列表里选中两列（第 1 列为 X，第 2 列为 Y）。",
                parent=self,
            )
            return
        x_spec = self._variable_for(selected[0])
        y_spec = self._variable_for(selected[1])
        if x_spec is None or y_spec is None:
            return
        self.app.xy.plot_variables(x_spec, y_spec)
        self.app.notebook.select(self.app.xy)


# --------------------------------------------------------------------------
# 子图参数对话框（替代 matplotlib 自带的只有滑块的版本，可手输数字）
# --------------------------------------------------------------------------


class SubplotDialog(tk.Toplevel):
    """调整 left/bottom/right/top/wspace/hspace：滑块与数字框双向联动。"""

    PARAMS = ("left", "bottom", "right", "top", "wspace", "hspace")
    LABELS = {
        "left": "左边距 left",
        "bottom": "底边距 bottom",
        "right": "右边距 right",
        "top": "顶边距 top",
        "wspace": "水平间距 wspace",
        "hspace": "垂直间距 hspace",
    }

    def __init__(self, master, fig, canvas, owner=None):
        super().__init__(master)
        self.fig = fig
        self.canvas = canvas
        self.owner = owner
        self._busy = False
        self.title(t("子图参数"))
        self.resizable(False, False)
        self.transient(master.winfo_toplevel())

        current = self._current()
        self.vars: dict[str, tuple[tk.DoubleVar, tk.StringVar]] = {}
        for row, name in enumerate(self.PARAMS):
            ttk.Label(self, text=t(self.LABELS[name])).grid(
                row=row, column=0, sticky="w", padx=(8, 4), pady=3
            )
            svar = tk.DoubleVar(value=current[name])
            scale = ttk.Scale(
                self, from_=0.0, to=1.0, orient="horizontal", length=190,
                variable=svar, command=lambda value, n=name: self._from_scale(n, value),
            )
            scale.grid(row=row, column=1, padx=4, pady=3)
            evar = tk.StringVar(value=f"{current[name]:.3f}")
            entry = ttk.Entry(self, textvariable=evar, width=7, justify="right")
            entry.grid(row=row, column=2, sticky="w", padx=(4, 8), pady=3)
            entry.bind("<Return>", lambda _e, n=name: self._from_entry(n))
            entry.bind("<FocusOut>", lambda _e, n=name: self._from_entry(n))
            entry.bind("<KP_Enter>", lambda _e, n=name: self._from_entry(n))
            self.vars[name] = (svar, evar)

        bottom = ttk.Frame(self)
        bottom.grid(row=len(self.PARAMS), column=0, columnspan=3, sticky="ew", pady=(4, 8))
        ttk.Button(bottom, text=t("恢复默认"), command=self.reset).pack(side="left", padx=8)
        ttk.Button(bottom, text=t("关闭"), command=self.destroy).pack(side="right", padx=8)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.grab_set()

    # ---- 取值/应用 ----
    def _current(self) -> dict[str, float]:
        pars = self.fig.subplotpars
        return {
            "left": float(pars.left),
            "bottom": float(pars.bottom),
            "right": float(pars.right),
            "top": float(pars.top),
            "wspace": float(pars.wspace),
            "hspace": float(pars.hspace),
        }

    def _values(self) -> dict[str, float]:
        out = {}
        for name, (svar, evar) in self.vars.items():
            try:
                out[name] = float(evar.get())
            except (TypeError, ValueError):
                out[name] = float(svar.get())
        return out

    def _apply(self) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            values = self._values()
            if self.owner is not None:
                self.owner._subplot_override = dict(values)
            try:
                self.fig.subplots_adjust(**values)
            except Exception:
                return
            self.canvas.draw_idle()
        finally:
            self._busy = False

    def _from_scale(self, name: str, value) -> None:
        if self._busy:
            return
        try:
            val = float(value)
        except (TypeError, ValueError):
            return
        self.vars[name][1].set(f"{val:.3f}")
        self._apply()

    def _from_entry(self, name: str) -> None:
        if self._busy:
            return
        svar, evar = self.vars[name]
        try:
            val = float(evar.get())
        except (TypeError, ValueError):
            evar.set(f"{svar.get():.3f}")
            return
        val = max(0.0, min(1.0, val))
        evar.set(f"{val:.3f}")
        self._busy = True
        try:
            svar.set(val)
        finally:
            self._busy = False
        self._apply()

    def reset(self) -> None:
        defaults = {"left": 0.125, "bottom": 0.11, "right": 0.9, "top": 0.88,
                    "wspace": 0.2, "hspace": 0.2}
        self._busy = True
        try:
            for name, value in defaults.items():
                self.vars[name][0].set(value)
                self.vars[name][1].set(f"{value:.3f}")
        finally:
            self._busy = False
        if self.owner is not None:
            self.owner._subplot_override = None
            try:
                self.fig.subplots_adjust(**defaults)
            except Exception:
                pass
        self.canvas.draw_idle()


def attach_subplot_dialog(tab, fig, canvas, toolbar) -> None:
    """把工具栏上的「子图配置」按钮换成本工具自己的对话框。"""

    def _open(_event=None):
        existing = getattr(tab, "_subplot_dlg", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
        tab._subplot_dlg = SubplotDialog(tab, fig, canvas, owner=tab)

    for key in ("Subplots", "Configure subplots"):
        try:
            toolbar._buttons[key].configure(command=_open)
            return
        except Exception:
            continue
    toolbar.edit_parameters = _open


def apply_subplot_override(tab, fig) -> None:
    """把对话框里设定的边距在每次重绘后重新套用（tight_layout 会覆盖它）。"""
    values = getattr(tab, "_subplot_override", None)
    if not values:
        return
    try:
        fig.subplots_adjust(**values)
    except Exception:
        pass


# --------------------------------------------------------------------------
# 标签页 2：单源一维光谱
# --------------------------------------------------------------------------


class SpectrumTab(ttk.Frame):
    def __init__(self, master, app: "App"):
        super().__init__(master, padding=4)
        self.app = app
        self._pending = None
        self._syncing = False
        self._subplot_override = None
        self._subplot_dlg = None
        self.compare_rows: list[int] = []

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.panes = ttk.PanedWindow(self, orient="horizontal")
        self.panes.grid(row=0, column=0, sticky="nsew")
        self.controls = ScrollableFrame(self.panes, width=310)
        self.controls.canvas.configure(width=290)
        self.panes.add(self.controls, weight=0)
        left = self.controls.body

        # --- 源选择 ---
        src_box = ttk.LabelFrame(left, text="源选择", padding=6)
        src_box.pack(fill="x")
        row_line = ttk.Frame(src_box)
        row_line.pack(fill="x")
        ttk.Label(row_line, text="源行号").pack(side="left")
        self.row_spin = ttk.Spinbox(
            row_line, from_=0, to=0, increment=1, width=8,
            textvariable=app.row_var, command=self._on_spin,
        )
        self.row_spin.pack(side="left", padx=4)
        self.row_spin.bind("<Return>", lambda _e: self.redraw(force=True))
        ttk.Button(row_line, text="◀", width=3, command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(row_line, text="▶", width=3, command=lambda: self.step(1)).pack(side="left", padx=2)

        # 滑杆本身是连续的，用一个 DoubleVar 中转，取整后再写回 app.row_var，
        # 这样界面上的源行号永远是整数
        self._scale_var = tk.DoubleVar(value=0.0)
        self.row_scale = ttk.Scale(
            src_box, from_=0, to=0, orient="horizontal",
            variable=self._scale_var, command=self._on_scale,
        )
        self.row_scale.pack(fill="x", pady=(6, 4))
        self._scale_var.trace_add("write", self._on_scale_var)

        id_line = ttk.Frame(src_box)
        id_line.pack(fill="x")
        ttk.Label(id_line, text="ID").pack(side="left")
        self.id_var = tk.StringVar()
        id_entry = ttk.Entry(id_line, textvariable=self.id_var, width=14)
        id_entry.pack(side="left", padx=4, fill="x", expand=True)
        id_entry.bind("<Return>", lambda _e: self.find_id())
        ttk.Button(id_line, text="查找", width=6, command=self.find_id).pack(side="left")

        # 多源对比：把若干源同时画在一张图上比较
        cmp_box = ttk.LabelFrame(left, text=t("多源对比"), padding=6)
        cmp_box.pack(fill="x", pady=(6, 0))
        cmp_line = ttk.Frame(cmp_box)
        cmp_line.pack(fill="x")
        ttk.Button(cmp_line, text=tf("加入对比", "Add to compare"), width=10,
                   command=self.add_compare).pack(side="left")
        ttk.Button(cmp_line, text=tf("清空", "Clear"), width=6,
                   command=self.clear_compare).pack(side="left", padx=4)
        self.opt_compare = tk.BooleanVar(value=True)
        ttk.Checkbutton(cmp_line, text=tf("显示对比", "Show"), variable=self.opt_compare,
                        command=self.schedule_redraw).pack(side="left")
        self.compare_list = tk.Listbox(
            cmp_box, height=3, exportselection=False, font=("Menlo", 9),
            borderwidth=0, highlightthickness=0, background=PANEL, foreground=TEXT,
            activestyle="none",
        )
        self.compare_list.pack(fill="x", pady=(4, 0))
        self.compare_list.bind("<Double-Button-1>", self._remove_compare_at)

        cmp_opts = ttk.Frame(cmp_box)
        cmp_opts.pack(fill="x", pady=(4, 0))
        self.opt_cmp_dash = tk.BooleanVar(value=True)
        ttk.Checkbutton(cmp_opts, text=tf("虚线对比", "Dashed"), variable=self.opt_cmp_dash,
                        command=self.schedule_redraw).pack(side="left")
        ttk.Label(cmp_opts, text=tf("归一化", "Normalise")).pack(side="left", padx=(10, 3))
        self.norm_labels = [tf("不归一化", "None"), tf("按中位流量", "Median flux"),
                            tf("按 1σ", "Median 1σ")]
        self._norm_modes = ("none", "median", "sigma")
        self.norm_var = tk.StringVar(value=self.norm_labels[0])
        ttk.Combobox(cmp_opts, textvariable=self.norm_var, values=self.norm_labels,
                     state="readonly", width=11).pack(side="left")
        self.norm_var.trace_add("write", lambda *_a: self.schedule_redraw())

        # --- 显示选项 ---
        opt = ttk.LabelFrame(left, text="显示", padding=6)
        opt.pack(fill="x", pady=(6, 0))
        band_line = ttk.Frame(opt)
        band_line.pack(fill="x")
        ttk.Label(band_line, text="波段").pack(side="left")
        self.band_vars = {}
        for band in BANDS:
            var = tk.BooleanVar(value=True)
            self.band_vars[band] = var
            ttk.Checkbutton(
                band_line, text=band, variable=var, command=self.schedule_redraw
            ).pack(side="left", padx=2)

        q_line = ttk.Frame(opt)
        q_line.pack(fill="x", pady=(4, 0))
        ttk.Label(q_line, text="物理量").pack(side="left")
        self.q_var = tk.StringVar(value=QUANTITIES[0][0])
        friendly = {
            "flux_ujy_with_noise": tf("观测流量 · μJy", "Observed flux · μJy"),
            "flux_ujy_model": tf("模型流量 · μJy", "Model flux · μJy"),
            "flux_ujy": tf("入射流量 · μJy", "Incident flux · μJy"),
            "flux_elec": tf("电子计数", "Electron counts"),
            "source_electron_rate_pre_morph": tf("形态前电子率", "Pre-morph electron rate"),
            "snr": tf("信噪比", "Signal to noise"),
        }
        self.q_display = tk.StringVar(value=friendly[QUANTITY_DS[self.q_var.get()]])
        labels = {friendly[ds]: label for label, ds, _ in QUANTITIES}
        q_box = ttk.Combobox(q_line, textvariable=self.q_display, values=list(labels),
                            state="readonly", width=23)
        q_box.pack(side="left", padx=4, fill="x", expand=True)
        quantity_tip = ToolTip(q_box, self.q_var.get())
        def sync_quantity(*_):
            self.q_display.set(friendly[QUANTITY_DS[self.q_var.get()]])
            quantity_tip.text = self.q_var.get()
        self.q_var.trace_add("write", sync_quantity)
        def choose_quantity(_event=None):
            self.q_var.set(labels[self.q_display.get()])
            self.redraw(force=True)
        q_box.bind("<<ComboboxSelected>>", choose_quantity)

        self.opt_err = tk.BooleanVar(value=True)
        self.opt_model = tk.BooleanVar(value=True)
        self.opt_mask = tk.BooleanVar(value=True)
        self.opt_overlay = tk.BooleanVar(value=True)
        self.opt_lines = tk.BooleanVar(value=True)
        self.opt_0th = tk.BooleanVar(value=False)
        self.opt_logy = tk.BooleanVar(value=False)
        for text, var in (
            ("1σ 误差带", self.opt_err),
            ("模型曲线", self.opt_model),
            ("分析掩膜 (spec_mask)", self.opt_mask),
            ("三波段同轴叠画", self.opt_overlay),
            ("标注发射线", self.opt_lines),
            ("叠加 _0th 诊断代理（虚线）", self.opt_0th),
            ("纵向对数坐标", self.opt_logy),
        ):
            ttk.Checkbutton(opt, text=text, variable=var, command=self.schedule_redraw).pack(
                anchor="w"
            )

        # --- 源信息 ---
        info_box = ttk.LabelFrame(left, text="源信息", padding=6)
        info_box.pack(fill="both", expand=True, pady=(6, 0))
        self.info = tk.Text(info_box, height=8, wrap="word", font=("Menlo", 10), borderwidth=0, highlightthickness=0, background=PANEL, foreground=TEXT, padx=8, pady=6)
        info_scroll = ttk.Scrollbar(info_box, orient="vertical", command=self.info.yview)
        self.info.configure(yscrollcommand=info_scroll.set, state="disabled")
        info_scroll.pack(side="right", fill="y")
        self.info.pack(side="left", fill="both", expand=True)

        ttk.Button(left, text="导出该源光谱 (CSV)…", command=self.export_spectrum).pack(
            fill="x", pady=4
        )

        # --- 右：图 ---
        right = ttk.Frame(self.panes, padding=(10, 0, 0, 0))
        self.panes.add(right, weight=1)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(9.6, 7.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar_holder = ttk.Frame(right)
        toolbar_holder.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_holder)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x")
        attach_subplot_dialog(self, self.fig, self.canvas, self.toolbar)
        self.status = ttk.Label(right, text="", foreground="#666")
        self.status.grid(row=2, column=0, sticky="w")

    # ---- 多源对比 ----
    def add_compare(self):
        model = self.app.model
        if model is None:
            return
        row = max(0, min(int(self.app.row_var.get()), model.nsrc - 1))
        if row in self.compare_rows:
            return
        if len(self.compare_rows) >= len(COMPARE_COLORS):
            self.compare_rows.pop(0)
        self.compare_rows.append(row)
        self._refresh_compare_list()
        self.schedule_redraw()

    def clear_compare(self):
        if not self.compare_rows:
            return
        self.compare_rows = []
        self._refresh_compare_list()
        self.schedule_redraw()

    def _remove_compare_at(self, _event=None):
        selection = self.compare_list.curselection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self.compare_rows):
            self.compare_rows.pop(index)
            self._refresh_compare_list()
            self.schedule_redraw()

    def _refresh_compare_list(self):
        self.compare_list.delete(0, "end")
        model = self.app.model
        try:
            current = int(self.app.row_var.get())
        except Exception:
            current = -1
        for row in self.compare_rows:
            sid = ""
            if model is not None:
                try:
                    sid = model.ids_str()[row] or ""
                except Exception:
                    sid = ""
            label = tf("行 {row}", "Row {row}", row=row) + (f"  {sid}" if sid else "")
            if row == current:
                # 与主曲线是同一个源时不再叠画，列表里标一下免得看起来没生效
                label += tf("（当前源）", " (current)")
            self.compare_list.insert("end", t(label))

    # ---- 交互 ----
    def _on_spin(self):
        self.redraw()

    def _on_scale(self, _value):
        # 数值由 self._scale_var 的 trace 统一处理（取整后写回 row_var）
        pass

    def _on_scale_var(self, *_args):
        if getattr(self, "_syncing", False):
            return
        try:
            value = float(self._scale_var.get())
        except Exception:
            return
        target = int(round(value))
        try:
            if int(self.app.row_var.get()) != target:
                self.app.row_var.set(target)
        except Exception:
            self.app.row_var.set(target)

    def _sync_widgets(self):
        model = self.app.model
        if model is None:
            return
        self._syncing = True
        try:
            self.row_spin.configure(from_=0, to=max(0, model.nsrc - 1))
            self.row_scale.configure(from_=0, to=max(0, model.nsrc - 1))
            try:
                self._scale_var.set(float(self.app.row_var.get()))
            except Exception:
                self._scale_var.set(0.0)
        finally:
            self._syncing = False

    def step(self, delta: int):
        model = self.app.model
        if model is None:
            return
        new = int(self.app.row_var.get()) + delta
        new = max(0, min(new, model.nsrc - 1))
        self.app.row_var.set(new)

    def find_id(self):
        model = self.app.model
        if model is None:
            return
        text = self.id_var.get().strip()
        if not text:
            return
        row = model.row_of_id(text)
        if row is None:
            messagebox.showinfo(APP_TITLE, f"没有找到 ID 以 “{text}” 开头的源。", parent=self)
            return
        self.app.row_var.set(row)
        self.id_var.set(model.ids_str()[row])

    def on_row_changed(self):
        if self.compare_rows:
            self._refresh_compare_list()
        # 拖动滑杆时会被连续触发，用 schedule_redraw 合并重绘
        if not getattr(self, "_syncing", False):
            try:
                row = int(self.app.row_var.get())
                if abs(float(self._scale_var.get()) - row) > 1e-9:
                    self._syncing = True
                    try:
                        self._scale_var.set(float(row))
                    finally:
                        self._syncing = False
            except Exception:
                pass
        self.schedule_redraw()

    def schedule_redraw(self, *_args):
        if self._pending is not None:
            try:
                self.after_cancel(self._pending)
            except Exception:
                pass
        self._pending = self.after(60, self._draw_now)

    def _draw_now(self):
        self._pending = None
        self.redraw()

    def redraw(self, force: bool = False):
        model = self.app.model
        if model is None:
            self.fig.clear()
            self.canvas.draw_idle()
            return
        if not force and self._pending is not None:
            return
        row = int(self.app.row_var.get())
        row = max(0, min(row, model.nsrc - 1))
        quantity_label = self.q_var.get()
        dataset = QUANTITY_DS[quantity_label]
        unit = QUANTITY_UNIT[quantity_label]
        bands = [b for b in BANDS if self.band_vars[b].get() and model.band_available(b)]

        self.fig.clear()
        if not bands or model.nsrc == 0:
            ax = self.fig.add_subplot(111)
            ax.text(
                0.5, 0.55, t("没有可用波段（文件中缺少 /spectra）"),
                ha="center", va="center", transform=ax.transAxes, fontsize=11,
            )
            ax.text(
                0.5, 0.44,
                t("这个文件不是 CESS 光谱产品；可到「X-Y 绘图」页选「波长向量 + 二维光谱」，"
                  "用源行号画单个源的光谱。"),
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.35",
            )
            ax.set_axis_off()
            self.canvas.draw_idle()
            self._update_info(row)
            return

        overlay = bool(self.opt_overlay.get())
        if overlay:
            grid = self.fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
            flux_axes = [self.fig.add_subplot(grid[0])]
            snr_axes = [self.fig.add_subplot(grid[1], sharex=flux_axes[0])]
        else:
            n = len(bands)
            grid = self.fig.add_gridspec(n + 1, 1, height_ratios=[3.0] * n + [1.1], hspace=0.14)
            flux_axes = [self.fig.add_subplot(grid[i]) for i in range(n)]
            snr_axes = [self.fig.add_subplot(grid[n], sharex=flux_axes[0])]

        norm_mode = self._norm_mode()
        scale = self._row_scale(row, bands, dataset, norm_mode)
        shading_by_axes: dict = {}
        values_by_axes: dict = {}
        for band in bands:
            ax = flux_axes[0] if overlay else flux_axes[bands.index(band)]
            shadings, values = self._draw_band(ax, band, row, dataset, unit, scale)
            shading_by_axes.setdefault(ax, []).extend(shadings)
            if values is not None and values.size:
                values_by_axes.setdefault(ax, []).append(values)

        # 对比源的谱：细线、统一颜色，每源只加一个图例条目
        if self.compare_rows and self.opt_compare.get():
            dash = self.opt_cmp_dash.get()
            for i, crow in enumerate(self.compare_rows):
                if crow == row:
                    continue
                color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
                cscale = self._row_scale(crow, bands, dataset, norm_mode)
                labelled = False
                for band in bands:
                    ax = flux_axes[0] if overlay else flux_axes[bands.index(band)]
                    y = model.spectrum_row(band, dataset, crow)
                    if y is None:
                        continue
                    y = np.asarray(y, dtype=float)
                    good = np.isfinite(y)
                    if not good.any():
                        continue
                    label = None
                    if not labelled:
                        label = tf("对比 源行 {row}", "compare row {row}", row=crow)
                        labelled = True
                    ax.plot(model.band_wave(band), np.where(good, y / cscale, np.nan),
                            "--" if dash else "-", lw=0.9, color=color, alpha=0.7,
                            zorder=2, label=label)
                    if good.any():
                        values_by_axes.setdefault(ax, []).append(y[good])

        if norm_mode == "none":
            ylabel = f"{unit}" if unit else "value"
        elif norm_mode == "sigma":
            ylabel = tf("流量 / 中位 1σ", "flux / median 1σ")
        else:
            ylabel = tf("流量 / 各自中位", "flux / own median")
        clipped_axes = []
        for ax in flux_axes:
            ax.set_ylabel(ylabel)
            ax.tick_params(labelsize=8)
            # 个别源的坏值会把纵轴拉爆，这里用分位数收窄
            if ax in values_by_axes:
                vals = np.concatenate(values_by_axes[ax])
                if norm_mode == "none":
                    robust_ylim(ax, vals)
                else:
                    # 归一化后个别发射线/边沿噪声可达百倍，默认视图只看形状
                    if robust_ylim(ax, vals, qlo=1.0, qhi=99.0, ratio=1.5):
                        clipped_axes.append(ax)
            # 掩膜（不被分析支持的区域）用浅灰阴影标出
            y0, y1 = ax.get_ylim()
            for wave, mask in shading_by_axes.get(ax, []):
                if wave is None or mask is None or mask.all():
                    continue
                try:
                    ax.fill_between(
                        wave, y0, y1, where=~mask, color="0.35", alpha=0.16, lw=0,
                        step="mid", zorder=0,
                    )
                except Exception:
                    pass
            ax.set_ylim(y0, y1)

        if self.opt_logy.get():
            for ax in flux_axes:
                # 谱流常有负值（噪声），用 symlog 才能在“对数”下也显示完整曲线
                vals = np.concatenate(values_by_axes[ax]) if ax in values_by_axes else np.array([])
                vals = vals[np.isfinite(vals)]
                scale = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
                ax.set_yscale("symlog", linthresh=max(1e-3, 0.02 * scale))

        # SNR 面板
        for ax in snr_axes:
            for band in bands:
                snr = model.spectrum_row(band, "snr", row)
                if snr is None:
                    continue
                wave = model.band_wave(band)
                good = np.isfinite(snr) & (snr > 0)
                ax.plot(wave[good], snr[good], lw=1.0, color=BAND_COLORS[band],
                        label=band if overlay else None)
            ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.set_ylabel("SNR", fontsize=9)
            ax.set_xlabel(tf("观测波长 (Å)", "Observed wavelength (Å)"))
            ax.tick_params(labelsize=8)
            ax.set_xlim(2400, 10600)

        flux_axes[0].set_xlim(2400, 10600)
        for ax in flux_axes[:-1]:
            ax.tick_params(labelbottom=False)

        # 统一图例放到图底部，避免遮住曲线和波段标注
        handles, labels = [], []
        for ax in flux_axes + snr_axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label and label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            # 每列放一个波段的三条曲线，图例按波段分组更易读
            ncol = 3 if len(labels) >= 3 else len(labels)
            rows = int(np.ceil(len(labels) / ncol))
            self.fig.legend(
                handles, labels, loc="lower center", ncol=ncol,
                fontsize=7.5, frameon=False, handlelength=1.6, columnspacing=1.6,
                labelspacing=0.35, borderaxespad=0.1,
            )
            self.fig.subplots_adjust(bottom=0.075 + 0.028 * rows)

        sid = self._source_id(row)
        z = self._parameter(row, "parameters_desi", "z_best")
        title = tf("源行 {row}", "Source row {row}", row=row)
        if sid:
            title += f"   ID {sid}"
        if z is not None and np.isfinite(z):
            title += f"   z = {z:.4f}"
        title += f"    [{quantity_label.split()[0]}]"
        if norm_mode == "sigma":
            title += tf("    · 按 1σ 归一", "    · normalised by 1σ")
        elif norm_mode == "median":
            title += tf("    · 按中位流量归一", "    · normalised by median flux")
        self.fig.suptitle(title, fontsize=11)
        apply_subplot_override(self, self.fig)
        self.fig.canvas.draw_idle()
        self._update_info(row)
        status = tf(
            "源行 {row}/{last}   波段 {bands}   {mode}",
            "Source row {row}/{last}   bands {bands}   {mode}",
            row=row, last=model.nsrc - 1, bands=",".join(bands),
            mode=t("无缝") if overlay else t("分波段"),
        )
        if clipped_axes:
            status += tf("   纵轴截断至 1–99% 分位", "   y axis clipped to 1–99%")
        self.status.configure(text=status)

    def _norm_mode(self) -> str:
        try:
            index = self.norm_labels.index(self.norm_var.get())
        except ValueError:
            index = 0
        return self._norm_modes[index]

    def _row_scale(self, row, bands, dataset, mode) -> float:
        """对比用的归一分母：1σ 用误差中位，中位流量用 |flux| 中位。"""
        if mode == "none":
            return 1.0
        model = self.app.model
        if mode == "sigma":
            pools = []
            for band in bands:
                ferr = model.spectrum_row(band, "ferr", row)
                if ferr is not None:
                    pools.append(np.asarray(ferr, dtype=float))
            pool = np.concatenate(pools) if pools else np.array([])
            pool = pool[np.isfinite(pool) & (pool > 0)]
            if pool.size:
                factor = float(np.median(pool))
                if np.isfinite(factor) and factor > 0:
                    return factor
        pools = []
        for band in bands:
            y = model.spectrum_row(band, dataset, row)
            if y is not None:
                pools.append(np.asarray(y, dtype=float))
        if not pools:
            return 1.0
        pool = np.concatenate(pools)
        pool = pool[np.isfinite(pool)]
        if pool.size == 0:
            return 1.0
        factor = float(np.median(np.abs(pool)))
        return factor if np.isfinite(factor) and factor > 0 else 1.0

    def _draw_band(self, ax, band, row, dataset, unit, scale: float = 1.0):
        model = self.app.model
        wave = model.band_wave(band)
        color = BAND_COLORS[band]
        y = model.spectrum_row(band, dataset, row)
        if y is None:
            return [], None
        y = np.asarray(y, dtype=float)
        good = np.isfinite(y)
        if not good.any():
            return [], None
        if scale and scale != 1.0:
            y = y / scale
        ax.plot(wave, np.where(good, y, np.nan), "-", lw=1.15, color=color, label=band, zorder=3)

        if self.opt_0th.get():
            y0 = model.spectrum_row(band, dataset + "_0th", row)
            if y0 is not None:
                y0 = np.asarray(y0, dtype=float)
                if scale and scale != 1.0:
                    y0 = y0 / scale
                g0 = np.isfinite(y0)
                ax.plot(wave[g0], y0[g0], "--", lw=0.9, color=color, alpha=0.55,
                        label=f"{band} _0th", zorder=2)

        is_flux = dataset in ("flux_ujy_with_noise", "flux_ujy_model")
        if self.opt_err.get() and is_flux:
            ferr = model.spectrum_row(band, "ferr", row)
            if ferr is not None:
                ferr = np.asarray(ferr, dtype=float)
                if scale and scale != 1.0:
                    ferr = ferr / scale
                ok = np.isfinite(y) & np.isfinite(ferr) & (ferr > 0)
                if ok.any():
                    ax.fill_between(
                        wave, y - ferr, y + ferr, where=ok, color=color, alpha=0.18,
                        lw=0, label=f"{band} 1σ", zorder=1,
                    )

        # 模型曲线只在通量类物理量上叠加，避免不同单位混画
        if self.opt_model.get() and is_flux and dataset != "flux_ujy_model":
            ym = model.spectrum_row(band, "flux_ujy_model", row)
            if ym is not None:
                ym = np.asarray(ym, dtype=float)
                if scale and scale != 1.0:
                    ym = ym / scale
                gm = np.isfinite(ym)
                ax.plot(wave, np.where(gm, ym, np.nan), "-", lw=0.9, color="#111", alpha=0.55,
                        label=f"{band} 模型", zorder=2)

        shadings = []
        if self.opt_mask.get():
            spec_mask = model.spectrum_row(band, "spec_mask", row)
            if spec_mask is not None:
                shadings.append((wave, np.asarray(spec_mask).astype(bool)))

        if self.opt_lines.get():
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            waves_l, snrs_l = model.measured_lines(band, row)
            if waves_l.size == 0:
                waves_l = model.line_peaks(band, row, detected=False)
            for i, lam in enumerate(np.asarray(waves_l).ravel()):
                ax.axvline(lam, color=color, ls=":", lw=0.8, alpha=0.6, zorder=1,
                           label=f"{band} 参考峰" if i == 0 else None)
            detected = model.line_peaks(band, row, detected=True)
            if detected.size:
                for lam in detected:
                    ax.axvline(lam, color=color, ls="--", lw=1.0, alpha=0.9, zorder=1)
                ax.plot(
                    detected, np.full(detected.shape, 0.965), "v", ms=4,
                    color=color, transform=trans, clip_on=False,
                    label=f"{band} 匹配噪声峰", zorder=4,
                )

        # 波段标注放在该波段中心波长的位置，无缝模式下三个波段才不会叠在一起
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            float(np.mean(wave[good])) if good.any() else float(np.mean(wave)),
            0.985, BAND_LABELS[band], transform=trans, fontsize=8,
            color=color, va="top", ha="center", zorder=6,
        )
        return shadings, y[good]

    # ---- 信息面板 ----
    def _source_id(self, row: int):
        ids = self.app.model.ids_str()
        if ids is None or row >= ids.size:
            return None
        return ids[row]

    def _parameter(self, row: int, table: str, name: str):
        model = self.app.model
        path = model.first_path(
            table if table.startswith("catalog") else f"catalog/{table}", table
        )
        if path is None:
            return None
        columns = model.table_columns(path)
        if not columns:
            return None
        names = [c[0] for c in columns]
        if name not in names:
            return None
        try:
            return model.cell(path, row, name)
        except Exception:
            return None

    def _update_info(self, row: int):
        model = self.app.model
        if model is None or model.nsrc == 0:
            self.info.configure(state="normal")
            self.info.delete("1.0", "end")
            self.info.insert("1.0", tf("文件中没有源。", "No sources in this file."))
            self.info.configure(state="disabled")
            return
        lines = []
        sid = self._source_id(row)
        lines.append(f"ID        {sid if sid is not None else '—'}")
        lines.append(f"行号      {row}")

        def param(table, name, fmt="{:g}"):
            value = self._parameter(row, table, name)
            if value is None:
                return "—"
            return "—" if not np.isfinite(value) else fmt.format(value)

        lines.append(f"RA/Dec    {param('parameters_desi', 'RA', '{:.5f}')} / "
                     f"{param('parameters_desi', 'Dec', '{:.5f}')}")
        lines.append(f"z_best    {param('parameters_desi', 'z_best', '{:.4f}')}")
        lines.append(
            "MAG G/R/Z {} / {} / {}".format(
                param("parameters_desi", "MAG_G", "{:.2f}"),
                param("parameters_desi", "MAG_R", "{:.2f}"),
                param("parameters_desi", "MAG_Z", "{:.2f}"),
            )
        )
        lines.append(
            "Sérsic n  {}   Re {} pix   PA {}°".format(
                param("parameters_desi", "n", "{:.2f}"),
                param("parameters_desi", "Re", "{:.1f}"),
                param("parameters_desi", "PA", "{:.1f}"),
            )
        )
        lines.append(
            "baratio   {}   log M* {}".format(
                param("parameters_desi", "baratio", "{:.2f}"),
                param("parameters_desi", "str_mass", "{:.2f}"),
            )
        )

        # 测光
        phot_path = model.first_path("catalog/parameters_phot", "parameters_phot")
        phot = model.h5.get(phot_path) if phot_path else None
        columns = model.table_columns(phot_path) if phot_path else None
        if phot is not None and columns:
            names = [c[0] for c in columns]
            values = []
            for band in ("nuv", "u", "g", "r", "i", "z", "y"):
                key = f"mag_{band}"
                if key in names:
                    try:
                        v = model.cell(phot_path, row, key)
                    except Exception:
                        continue
                    values.append(f"{band}:{v:.2f}" if np.isfinite(v) else f"{band}:—")
            if values:
                lines.append("测光 mag " + " ".join(values))

        zo_path = model.first_path("catalog/zero_order_phot", "zero_order_phot")
        columns = model.table_columns(zo_path) if zo_path else None
        if zo_path is not None and columns:
            names = [c[0] for c in columns]
            values = []
            for key in ("mag_gu0", "mag_gv0", "mag_gi0"):
                if key in names:
                    try:
                        v = model.cell(zo_path, row, key)
                    except Exception:
                        continue
                    values.append(f"{key[4:]}:{v:.2f}" if np.isfinite(v) else f"{key[4:]}:—")
            if values:
                lines.append("零级 mag " + " ".join(values))

        for path, label in (
            (model.first_path("catalog/data_mask", "data_mask"), "data_mask"),
            (model.first_path("catalog/input_quality_flags", "input_quality_flags"), "质量标志"),
        ):
            if path and path in model.h5:
                try:
                    value = int(model.h5[path][row])
                    lines.append(f"{label:<9}  {value}")
                except Exception:
                    pass

        if "catalog/spectral_extraction/valid" in model.h5:
            try:
                valid = bool(model.h5["catalog/spectral_extraction/valid"][row])
                fraction = float(model.h5["catalog/spectral_extraction/source_fraction"][row])
                pixels = float(model.h5["catalog/spectral_extraction/spatial_pixels"][row])
                lines.append(
                    f"光谱提取  valid={valid}  fraction={fraction:.3f}  pixels={pixels:.2f}"
                )
            except Exception:
                pass

        line_bits = []
        for band in BANDS:
            n_int = model.line_peaks(band, row, detected=False).size
            n_det = model.line_peaks(band, row, detected=True).size
            if n_int or n_det:
                line_bits.append(tf("{band}: 匹配噪声峰 {nd} / 参考峰 {ni}",
                                    "{band}: matched noisy peaks {nd} / reference peaks {ni}",
                                    band=band, nd=n_det, ni=n_int))
        if line_bits:
            lines.append("发射线    " + "   ".join(line_bits))

        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", t("\n".join(lines)))
        self.info.configure(state="disabled")

    # ---- 导出 ----
    def current_table(self):
        """拼接当前源所有选中的波段，返回 (headers, rows)。"""
        model = self.app.model
        row = int(self.app.row_var.get())
        dataset = QUANTITY_DS[self.q_var.get()]
        headers = ["# band", "wave_A", dataset, "ferr", "flux_ujy_model",
                   "spec_mask", "snr"]
        rows = []
        for band in BANDS:
            if not (self.band_vars[band].get() and model.band_available(band)):
                continue
            wave = model.band_wave(band)
            flux = model.spectrum_row(band, dataset, row)
            if flux is None:
                continue
            ferr = model.spectrum_row(band, "ferr", row)
            model_flux = model.spectrum_row(band, "flux_ujy_model", row)
            spec_mask = model.spectrum_row(band, "spec_mask", row)
            snr = model.spectrum_row(band, "snr", row)
            for j in range(wave.size):
                rows.append([
                    band, wave[j],
                    flux[j] if flux is not None else np.nan,
                    ferr[j] if ferr is not None else np.nan,
                    model_flux[j] if model_flux is not None else np.nan,
                    bool(spec_mask[j]) if spec_mask is not None else "",
                    snr[j] if snr is not None else np.nan,
                ])
        return headers, rows, row

    def export_spectrum(self):
        model = self.app.model
        if model is None:
            return
        headers, rows, row = self.current_table()
        if not rows:
            messagebox.showinfo(APP_TITLE, "当前没有可导出的光谱数据。", parent=self)
            return
        sid = self._source_id(row) or f"row{row}"
        path = filedialog.asksaveasfilename(
            parent=self, title="导出光谱",
            defaultextension=".csv",
            initialfile=f"spectrum_row{row}_{sid}.csv",
            filetypes=[("CSV", "*.csv"), ("文本", "*.txt"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            write_csv(path, headers, rows, model.path)
        except (OSError, ValueError) as exc:
            messagebox.showerror(APP_TITLE, f"写入失败：{exc}", parent=self)
            return
        self.status.configure(text=f"已导出 {len(rows)} 行 → {path}")


# --------------------------------------------------------------------------
# 标签页 3：通用 X-Y 绘图
# --------------------------------------------------------------------------


class XYTab(ttk.Frame):
    def __init__(self, master, app: "App"):
        super().__init__(master, padding=4)
        self.app = app
        self.x_spec: Variable | None = None
        self.y_spec: Variable | None = None
        self.c_spec: Variable | None = None
        self.e_spec: Variable | None = None
        self._last = None
        self._uses_row = False
        self._default_hint = ""
        self._subplot_override = None
        self._subplot_dlg = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        ctrl = ttk.Frame(self, padding=10, style="Panel.TFrame")
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for col in (1, 3): ctrl.columnconfigure(col, weight=1)
        self.x_var, self.y_var = tk.StringVar(), tk.StringVar()
        self.c_var, self.e_var = tk.StringVar(), tk.StringVar()
        for col, axis, var in ((0, "x", self.x_var), (2, "y", self.y_var)):
            ttk.Label(ctrl, text=axis.upper(), style="Section.TLabel").grid(row=0, column=col, padx=(0, 8))
            box = ttk.Combobox(ctrl, textvariable=var, state="readonly", height=18, width=25)
            box.grid(row=0, column=col + 1, sticky="ew", padx=(0, 10))
            box.bind("<<ComboboxSelected>>", lambda _e, a=axis: self._on_select(a))
            setattr(self, axis + "_box", box)
        actions = ttk.Frame(ctrl)
        actions.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        self.kind_var = tk.StringVar(value=KINDS[0])
        self.kind_box = ttk.Combobox(actions, textvariable=self.kind_var, values=KINDS,
                                    state="readonly", width=18)
        self.kind_box.pack(side="left", padx=(0, 12))
        self.kind_box.bind("<<ComboboxSelected>>", self.schedule_redraw)
        ttk.Label(actions, text=tf("源行", "Row")).pack(side="left")
        self.row_spin = ttk.Spinbox(actions, from_=0, to=0, increment=1, width=6,
                                    textvariable=app.row_var, command=self.schedule_redraw)
        self.row_spin.pack(side="left", padx=(5, 12))
        ttk.Label(actions, text=tf("像素", "Pixel")).pack(side="left")
        self.pixel_var = tk.IntVar(value=0)
        self.pixel_spin = ttk.Spinbox(actions, from_=0, to=0, increment=1, width=6,
                                      textvariable=self.pixel_var, command=self.schedule_redraw)
        self.pixel_spin.pack(side="left", padx=5)
        ttk.Button(actions, text=tf("绘图", "Plot"), style="Accent.TButton", command=self.redraw).pack(side="right")
        self.advanced = CollapsibleFrame(ctrl, text=tf("筛选与显示选项", "Filters & display"))
        self.advanced.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        body = self.advanced.body
        body.columnconfigure(1, weight=1)
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self._refresh_choices())
        ttk.Label(body, text=tf("查找变量", "Find variable")).grid(row=0, column=0, sticky="w")
        ttk.Entry(body, textvariable=self.filter_var).grid(row=0, column=1, sticky="ew", padx=8)
        self.filter_expr = tk.StringVar()
        ttk.Label(body, text=tf("筛选表达式", "Filter expression")).grid(row=1, column=0, sticky="w", pady=5)
        expression = ttk.Entry(body, textvariable=self.filter_expr)
        expression.grid(row=1, column=1, sticky="ew", padx=8, pady=5)
        expression.bind("<Return>", lambda _e: self.redraw())
        ToolTip(expression, tf("例如 (x > 4000) & np.isfinite(y)", "Example: (x > 4000) & np.isfinite(y)"))
        ttk.Button(body, text=tf("清空", "Clear"), command=lambda: (self.filter_expr.set(""), self.redraw())).grid(row=1, column=2)
        for r, axis, variable, label in ((2, "c", self.c_var, tf("颜色", "Color")),
                                         (3, "e", self.e_var, tf("Y 误差", "Y error"))):
            ttk.Label(body, text=label).grid(row=r, column=0, sticky="w", pady=3)
            box = ttk.Combobox(body, textvariable=variable, state="readonly", height=18)
            box.grid(row=r, column=1, columnspan=2, sticky="ew", padx=8, pady=3)
            box.bind("<<ComboboxSelected>>", lambda _e, a=axis: self._on_select(a))
            setattr(self, axis + "_box", box)
        flags = ttk.Frame(body)
        flags.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        self.logx, self.logy = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.invert_y, self.grid_on = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.equal_xy, self.robust = tk.BooleanVar(value=False), tk.BooleanVar(value=True)
        for i, (label, var) in enumerate((("x 对数", self.logx), ("y 对数", self.logy),
            ("反转 y", self.invert_y), ("网格", self.grid_on), ("等比例", self.equal_xy),
            ("稳健范围 (0.2–99.8%)", self.robust))):
            ttk.Checkbutton(flags, text=label, variable=var, command=self.schedule_redraw).grid(
                row=i // 3, column=i % 3, sticky="w", padx=(0, 12))

        self.note = ttk.Label(
            self,
            text="源行号用于行向变量（二维谱的像素列会取这一行）。着色变量 / Y 误差选「（不使用）」即可清除。",
            foreground=MUTED, wraplength=700,
        )
        self.note.grid(row=1, column=0, sticky="w")

        plot_holder = ttk.Frame(self)
        plot_holder.grid(row=2, column=0, sticky="nsew")
        plot_holder.rowconfigure(0, weight=1)
        plot_holder.columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(9.6, 6.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_holder)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        tb_holder = ttk.Frame(plot_holder)
        tb_holder.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb_holder)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x")
        attach_subplot_dialog(self, self.fig, self.canvas, self.toolbar)

        self._all_choices: list[str] = []
        self._pending = None

    # ---- 选项同步 ----
    def refresh_variables(self):
        model = self.app.model
        if model is None:
            return
        labels = sorted(model.variable_map().keys(), key=str.lower)
        self._all_choices = labels
        self._refresh_choices()
        try:
            self.row_spin.configure(from_=0, to=max(0, model.nsrc - 1))
        except Exception:
            pass
        if not self.c_var.get():
            self.c_var.set(NONE_CHOICE)
        if not self.e_var.get():
            self.e_var.set(NONE_CHOICE)

    def plot_default(self):
        """打开文件后先画一张有意义的图（默认天区位置分布）。"""
        labels = self._all_choices
        if not labels:
            return

        def find(*needles):
            for label in labels:
                if all(needle in label for needle in needles):
                    return label
            return None

        x_label = find("parameters_desi", ":: RA")
        y_label = find("parameters_desi", ":: Dec")
        if x_label and y_label:
            self.adopt(self.app.model.variable_map()[x_label], "x")
            self.adopt(self.app.model.variable_map()[y_label], "y")
            self.kind_var.set(KINDS[0])
            # 星表里有 1e270 这类坏坐标，天区图必须用稳健范围
            self.robust.set(True)
            self.redraw()
            self.note.configure(text="默认显示：源在天球上的分布（可自行更换 X/Y）")
            return
        # 光谱立方体：一维波长向量 + 列数与之相等的二维数组 → 直接画当前源行的光谱
        model = self.app.model
        vmap = model.variable_map()
        wave_spec = None
        for label in labels:
            spec = vmap.get(label)
            if spec is None or spec.kind != "vec":
                continue
            name = spec.path.lower()
            if "wave" in name or "lambda" in name:
                try:
                    if model.get(spec.path).ndim == 1:
                        wave_spec = spec
                        break
                except Exception:
                    continue
        if wave_spec is not None:
            try:
                size = int(model.get(wave_spec.path).shape[0])
            except Exception:
                size = 0
            if size > 1:
                for label in labels:
                    spec = vmap.get(label)
                    if spec is None or spec.kind != "col2d":
                        continue
                    try:
                        shape = model.get(spec.path).shape
                    except Exception:
                        continue
                    if len(shape) == 2 and int(shape[1]) == size:
                        self.adopt(wave_spec, "x")
                        self.adopt(spec, "y")
                        self.kind_var.set(KINDS[1])
                        # 光谱要看到完整的峰，不做分位数裁剪
                        self.robust.set(False)
                        self._default_hint = t(
                            "默认显示：当前源行的一维光谱（用上面的「源行号」切换源）"
                        )
                        self.redraw()
                        return
        if len(labels) >= 2:
            self.adopt(self.app.model.variable_map()[labels[0]], "x")
            self.adopt(self.app.model.variable_map()[labels[1]], "y")
            self.redraw()

    def _refresh_choices(self):
        text = self.filter_var.get().strip().lower()
        values = [v for v in self._all_choices if text in v.lower()] if text else list(self._all_choices)
        for box in (self.x_box, self.y_box):
            box.configure(values=values)
        optional = [NONE_CHOICE] + values
        for box in (self.c_box, self.e_box):
            box.configure(values=optional)
        if self.c_var.get() and self.c_var.get() not in optional:
            self.c_var.set(NONE_CHOICE)
            self.c_spec = None
        if self.e_var.get() and self.e_var.get() not in optional:
            self.e_var.set(NONE_CHOICE)
            self.e_spec = None

    def _on_select(self, axis: str):
        model = self.app.model
        if model is None:
            return
        box_var = {"x": self.x_var, "y": self.y_var, "c": self.c_var, "e": self.e_var}[axis]
        label = box_var.get()
        if axis in ("c", "e") and (not label or label == NONE_CHOICE):
            # 选「（不使用）」= 清除该列
            if axis == "c":
                self.c_spec = None
            else:
                self.e_spec = None
            self.schedule_redraw()
            return
        spec = model.variable_map().get(label)
        if spec is None:
            return
        if axis == "x":
            self.x_spec = spec
        elif axis == "y":
            self.y_spec = spec
        elif axis == "c":
            self.c_spec = spec
        else:
            self.e_spec = spec
        if spec.kind == "col2d":
            ds = model.get(spec.path)
            self.pixel_spin.configure(from_=0, to=max(0, int(ds.shape[1]) - 1))
        self.schedule_redraw()

    def adopt(self, spec: Variable, axis: str):
        """由结构页调用：把一列放进 X / Y / 着色。"""
        self._default_hint = ""
        model = self.app.model
        if model is None:
            return
        if spec.label not in self._all_choices:
            self._all_choices.append(spec.label)
            self._all_choices.sort(key=str.lower)
            self._refresh_choices()
        target = {"x": self.x_var, "y": self.y_var, "c": self.c_var, "e": self.e_var}[axis]
        target.set(spec.label)
        self._on_select(axis)

    def plot_variables(self, x_spec: Variable, y_spec: Variable):
        self.adopt(x_spec, "x")
        self.adopt(y_spec, "y")
        self.redraw()

    def on_row_changed(self):
        """其它页改了源行号：只有当前这张图依赖行号时才重画。"""
        if getattr(self, "_uses_row", False):
            self.schedule_redraw()

    def schedule_redraw(self, *_args):
        if self._pending is not None:
            try:
                self.after_cancel(self._pending)
            except Exception:
                pass
        self._pending = self.after(60, self._tick)

    def _tick(self):
        self._pending = None
        self.redraw()

    # ---- 绘图 ----
    def redraw(self):
        model = self.app.model
        self.fig.clear()
        self._export_snapshot = None
        if model is None or self.x_spec is None or self.y_spec is None:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "请选择 X 与 Y 变量", ha="center", va="center",
                    transform=ax.transAxes, color="0.5")
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        pixel = int(self.pixel_var.get())
        expr = self.filter_expr.get().strip()
        try:
            x, y, row_used = model.resolve_xy(
                self.x_spec, self.y_spec, pixel=pixel, row=int(self.app.row_var.get()))
            mask = filter_mask(x, y, expr)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self._show_message(str(exc))
            return
        xv, yv = x[mask], y[mask]
        if xv.size == 0:
            self._show_message("筛选后没有数据点。")
            return
        kind = self.kind_var.get()
        self._uses_row = row_used is not None
        if kind == KINDS[4]:
            self._plot_hexbin(xv, yv)
            self._export_snapshot = ([self._label(self.x_spec, row_used),
                                      self._label(self.y_spec, row_used)], [xv.copy(), yv.copy()])
            return
        if kind == KINDS[3]:
            self._plot_hist(xv)
            self._uses_row = row_used is not None
            self._export_snapshot = ([self._label(self.x_spec, row_used)], [xv.copy()])
            return

        ax = self.fig.add_subplot(111)
        xv, yv = x[mask], y[mask]
        if xv.size == 0:
            self._show_message("筛选后没有数据点。")
            return

        color_spec = self.c_spec
        if color_spec is not None:
            try:
                cvals = np.asarray(model.resolve_for_axis(color_spec, pixel, row_used, x.size))
            except (ValueError, IndexError, KeyError, TypeError):
                cvals = np.array([])
            if cvals.size != x.size:
                color_spec = None
                cvals = None
            else:
                cvals = cvals[mask]
        else:
            cvals = None

        e_spec = self.e_spec
        yerr = None
        err_dropped = False
        if e_spec is not None:
            try:
                err = np.asarray(model.resolve_for_axis(e_spec, pixel, row_used, x.size), dtype=float)
            except (ValueError, IndexError, KeyError, TypeError):
                err = np.array([])
            if err.size == x.size:
                yerr = err[mask].copy()
                invalid_error = ~np.isfinite(yerr) | (yerr < 0)
                err_dropped = bool(invalid_error.any())
                yerr[invalid_error] = np.nan
            else:
                err_dropped = True

        label_x = self._label(self.x_spec, None if row_used is None else row_used)
        label_y = self._label(self.y_spec, None if row_used is None else row_used)

        if kind == KINDS[1]:
            order = np.argsort(xv)
            ax.plot(xv[order], yv[order], "-", lw=1.1, color="#1f6fb4")
        elif kind == KINDS[2]:
            order = np.argsort(xv)
            ax.step(xv[order], yv[order], where="mid", lw=1.1, color="#1f6fb4")
        else:  # 散点
            if cvals is not None and cvals.dtype.kind in "fiub":
                sc = ax.scatter(xv, yv, c=cvals.astype(float), s=8, cmap="viridis",
                                alpha=0.85, linewidths=0, zorder=2)
                cbar = self.fig.colorbar(sc, ax=ax, pad=0.01)
                cbar.set_label(self._label(color_spec), fontsize=9)
            else:
                ax.scatter(xv, yv, s=8, alpha=0.7, color="#1f6fb4", linewidths=0, zorder=2)

        if yerr is not None and np.isfinite(yerr).any():
            # zorder 低于数据点（2），不遮挡曲线本身
            ax.errorbar(xv, yv, yerr=yerr, fmt="none", ecolor="0.55", alpha=0.45,
                        elinewidth=0.7, capsize=0, zorder=1.5)

        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        if self.logx.get():
            ax.set_xscale("log", nonpositive="clip")
        if self.logy.get():
            ax.set_yscale("log", nonpositive="clip")
        if self.invert_y.get():
            ax.invert_yaxis()
        if self.grid_on.get():
            ax.grid(True, alpha=0.3, lw=0.5)
        else:
            ax.grid(False)
        if self.equal_xy.get():
            ax.set_aspect("equal", adjustable="datalim")

        # 稳健范围：忽略极端离群点（例如个别坏源里的 1e270）
        clipped = 0
        if self.robust.get():
            xl, yl = percentile_limits(xv, yv)
            if xl is not None:
                ax.set_xlim(*xl)
            if yl is not None:
                ax.set_ylim(*yl)
            if xl is not None and yl is not None:
                inside = (
                    (xv >= xl[0]) & (xv <= xl[1]) & (yv >= yl[0]) & (yv <= yl[1])
                )
                clipped = int((~inside).sum()) if inside.dtype == bool else 0
        else:
            self._guard_limits(ax, xv, yv)
        if expr:
            title = tf("{n} 点（筛选后，原 {m} 点）", "{n} points (filtered from {m})",
                       n=xv.size, m=x.size)
        else:
            title = tf("{n} 点", "{n} points", n=xv.size)
        if row_used is not None:
            title += tf("（源行 {row}）", " (source row {row})", row=row_used)
        if clipped:
            title += tf("；{k} 点在显示范围外", "; {k} points outside range", k=clipped)
        ax.set_title(title, fontsize=10)
        safe_tight_layout(self.fig)
        apply_subplot_override(self, self.fig)
        self.canvas.draw_idle()
        headers, columns = [label_x, label_y], [xv.copy(), yv.copy()]
        if cvals is not None and color_spec is not None:
            headers.append(self._label(color_spec, row_used))
            columns.append(cvals.copy())
        if yerr is not None:
            headers.append("Y_error")
            columns.append(yerr.copy())
        self._export_snapshot = (headers, columns)
        self._last = ("xy", self.x_spec, self.y_spec, pixel, mask)
        self._uses_row = row_used is not None
        note = tf("X 长度 {nx}，Y 长度 {ny}；有效点 {n}",
                  "X length {nx}, Y length {ny}; valid points {n}",
                  nx=x.size, ny=y.size, n=int(mask.sum()))
        if clipped:
            note += tf("；{k} 个离群点超出显示范围（可取消“稳健范围”）",
                       "; {k} outliers beyond display range (uncheck robust range)",
                       k=clipped)
        if err_dropped:
            note += tf("；不匹配或负/非有限 Y 误差已忽略",
                       "; mismatched or negative/nonfinite Y errors ignored")
        if self._default_hint:
            note += "；" + self._default_hint
        self.note.configure(text=note)

    def _plot_hist(self, x):
        ax = self.fig.add_subplot(111)
        data = x
        if data.dtype.kind in "fiub":
            data = data[np.isfinite(data.astype(float))]
        if data.size == 0:
            self._show_message("没有有效数据。")
            return
        ax.hist(data, bins=60, color="#1f6fb4", alpha=0.85)
        ax.set_xlabel(self._label(self.x_spec))
        ax.set_ylabel("计数")
        if self.logx.get():
            ax.set_xscale("log", nonpositive="clip")
        if self.logy.get():
            ax.set_yscale("log", nonpositive="clip")
        if self.grid_on.get():
            ax.grid(True, alpha=0.3, lw=0.5)
        else:
            ax.grid(False)
        ax.set_title(f"N = {data.size}", fontsize=10)
        safe_tight_layout(self.fig)
        apply_subplot_override(self, self.fig)
        self.canvas.draw_idle()
        self._last = ("hist", self.x_spec, None, 0, None)
        self._uses_row = False
        self.note.configure(text=f"X 长度 {x.size}")

    def _plot_hexbin(self, x, y):
        ax = self.fig.add_subplot(111)
        xf = x.astype(float) if x.dtype.kind in "fiub" else np.arange(x.size, dtype=float)
        yf = y.astype(float) if y.dtype.kind in "fiub" else np.arange(y.size, dtype=float)
        good = np.isfinite(xf) & np.isfinite(yf)
        hb = ax.hexbin(xf[good], yf[good], gridsize=60, bins="log", cmap="viridis", mincnt=1)
        cbar = self.fig.colorbar(hb, ax=ax, pad=0.01)
        cbar.set_label("log10(计数)", fontsize=9)
        ax.set_xlabel(self._label(self.x_spec))
        ax.set_ylabel(self._label(self.y_spec))
        if self.logx.get():
            ax.set_xscale("log", nonpositive="clip")
        if self.logy.get():
            ax.set_yscale("log", nonpositive="clip")
        if self.invert_y.get():
            ax.invert_yaxis()
        if self.grid_on.get():
            ax.grid(True, alpha=0.3, lw=0.5)
        else:
            ax.grid(False)
        if self.robust.get():
            xl, yl = percentile_limits(xf[good], yf[good])
            if xl is not None:
                ax.set_xlim(*xl)
            if yl is not None:
                ax.set_ylim(*yl)
        else:
            self._guard_limits(ax, xf[good], yf[good])
        safe_tight_layout(self.fig)
        apply_subplot_override(self, self.fig)
        self.canvas.draw_idle()
        self.note.configure(text=f"有效点 {int(good.sum())}")

    @staticmethod
    def _guard_limits(ax, xvals, yvals):
        """极端坏值（例如 1e270 的 RA）会让刻度定位器抛异常，这里兜底收窄。"""
        bad = False
        for lo, hi in (ax.get_xlim(), ax.get_ylim()):
            if not (np.isfinite(lo) and np.isfinite(hi)) or abs(float(lo)) > 1e100 or abs(float(hi)) > 1e100:
                bad = True
                break
        if not bad:
            return
        for axis, values in (("x", xvals), ("y", yvals)):
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            lo, hi = np.percentile(arr, [0.2, 99.8])
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                lo, hi = float(arr.min()), float(arr.max())
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                continue
            pad = (hi - lo) * 0.03 or 1.0
            getattr(ax, "set_%slim" % axis)(lo - pad, hi + pad)

    def _label(self, spec: Variable | None, row: int | None = None) -> str:
        """坐标轴标签：简短的列名 + 单位（完整路径在界面的下拉框里可见）。"""
        if spec is None:
            return ""
        if spec.kind == "col2d":
            name = _shorten(spec.path.rstrip("/").split("/")[-1], 24)
            if row is None:
                text = f"{name} · " + tf("像素 {k}", "pixel {k}", k=int(self.pixel_var.get()))
            else:
                text = f"{name} · " + tf("源行 {row}", "source row {row}", row=int(row))
        elif " :: " in spec.label:
            text = spec.label.split(" :: ", 1)[1]
        else:
            # 一维向量：用 "组/数据集" 两段，避免坐标轴写成长路径
            text = "/".join(spec.path.rstrip("/").split("/")[-2:])
        text = text.split(" (N=")[0].strip()
        unit = "" if spec.units in (None, "", "1") else spec.units
        unit_tail = f"[{unit}]" if unit else ""
        if unit_tail and text.endswith(unit_tail):
            text = text[: -len(unit_tail)].strip()
        return f"{text} {unit_tail}".strip()

    def _show_message(self, text: str):
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
                color="#b03030", wrap=True, fontsize=10)
        ax.set_axis_off()
        self.canvas.draw_idle()
        self.note.configure(text=text.splitlines()[0])
        self._last = None
        self._export_snapshot = None

    # ---- 导出 ----
    def export_data(self):
        snapshot = getattr(self, "_export_snapshot", None)
        if self.app.model is None or snapshot is None:
            messagebox.showinfo(APP_TITLE, "请先选择 X、Y 变量并绘图。", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="导出绘图数据", defaultextension=".csv",
            initialfile="xy_data.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        headers, columns = snapshot
        try:
            write_csv(path, headers, zip(*columns), self.app.model.path)
        except (OSError, ValueError) as exc:
            messagebox.showerror(APP_TITLE, f"写入失败：{exc}", parent=self)
            return
        self.note.configure(text=f"已导出 {len(columns[0])} 行 → {path}")


# --------------------------------------------------------------------------
# 标签页 4：说明
# --------------------------------------------------------------------------


HELP_TEXT = """\
CESS HDF5 只读浏览器
──────────────────
结构页：浏览全部组/数据集、属性、列定义；每页行数可选 200/500/1000/2000。
        右侧“对象信息/列/数据预览”之间的分隔条可上下拖动调高度；
        对象信息栏的 A−/A+ 调字号，「复制」把全文拷到剪贴板，「跳到行」直达某一行。
        点表头按该列排序（再点一次反向，点第 0 列取消）；「导出本页」存当前页 CSV。
        右键树节点：复制路径、设为 X/Y/颜色、展开或折叠；右键预览行可跳到该源的光谱。
光谱页：按源行或ID查看 GU/GV/GI；同轴叠画不会合并、重采样或校正波段。
        「加入对比」把最多 4 个源以细线叠在同一张图上，双击列表移除、清空按钮复位。
        「虚线对比」让对比曲线用虚线；「归一化」可把每条曲线除以自身中位流量或中位 1σ，
        便于比较不同亮度的谱形（不做平滑或平移，归一化因子显示在标题和纵轴上）；
        归一化时纵轴默认截断到 1–99% 分位，状态栏会提示，缩放或对称对数可看全。
X-Y页：选X/Y、颜色或Y误差；波长与二维谱组合时按当前源行读取。
窗口大小、分隔条位置、每页行数和上次打开的文件会被记住（~/.config/cess-viewer/）。
说明：更完整的使用方法见 docs/VIEWER_zh-CN.md。

科学量与产品
────────────
/catalog 保存ID、原星表行映射、质量标志和参数表。
/spectra/{GU,GV,GI} 保存谱、误差、掩膜和邻像元协方差。
flux_ujy 是光子等效入射bin fnu；flux_ujy_model 和 flux_ujy_with_noise
是响应标定后的提取谱。ferr仅对应后两者，不能作为入射量的误差。
source_electron_rate_pre_morph 单位为 electron s^-1 cm^-2。
/zero_order 是标准模式的波段积分零级测光；_0th数组只是历史诊断代理。
负噪声允许存在；symlog显示负值，S/N对数面板只显示正值。
/lines 中的 truth 窗口是参考中心测量，不等于确认检测或原子线鉴定；
detect_el_* 仅为匹配成功的噪声峰，每源可有多峰。
旧 intri_el_idx 在裁剪网格上，偏移取决于配置 *_init_idx；
优先直接读波长，或新测量表的完整网格 center_index。
paired-wce 光谱页显示基础（无WCE）分支；*_wce 用结构页/X-Y页查看。
EL-only文件没有父光谱，使用结构/X-Y页；通用HDF5的单位/轴需自行核实。

筛选与导出
──────────
筛选例：(x>22) & (y>20)、np.isfinite(x) & (np.abs(y)<10)。
仅支持白名单数组数学运算，不能执行任意Python。
所有图型应用筛选；CSV导出当前绘图的数据，稳健显示范围不会删除点。
CSV不含完整掩膜、协方差与provenance，不能代替科学HDF5。
导出不会改写当前输入HDF5；图片也不能保存到该输入文件。
快捷键：macOS为Cmd-O/S/E，其他平台为Ctrl-O/S/E；左右方向键换源。
"""

HELP_TEXT_EN = """\
CESS read-only HDF5 viewer
─────────────────────────
Structure: inspect groups, datasets, attributes and columns; pages hold 200/500/1000/2000 rows.
           Drag the dividers between the object information, column list and data preview
           to resize them; A-/A+ changes the text size, Copy grabs it all, Go to row jumps
           to any row. Click a header to sort by that column (again to reverse, row 0 to
           clear); Export page writes the visible page as CSV.
           Right-click a tree node for copy path / set X / set Y / set colour / expand;
           right-click a preview row to open that source in the Spectrum tab.
Spectrum: browse GU/GV/GI by source row or ID. Overlay does not merge, resample or calibrate bands.
          Add to compare overlays up to four sources as thin lines (double-click to remove).
          Dashed draws them dotted; Normalise divides every curve by its own median flux or
          median 1σ so spectra of different brightness can be compared by shape (the y axis
          is then clipped to 1–99% quantiles; the status bar says so).
X-Y: select X/Y, colour or Y error; a wavelength + 2-D spectrum uses the current source row.
Window size, divider positions, page size and the last opened file are remembered in
~/.config/cess-viewer/.
See docs/VIEWER.md for the complete guide.

Quantities and products
──────────────────────
/catalog holds IDs, original-row mappings, quality flags and parameter tables.
/spectra/{GU,GV,GI} holds spectra, errors, masks and adjacent-pixel covariance.
flux_ujy is incident photon-equivalent bin fnu; flux_ujy_model and flux_ujy_with_noise
are response-calibrated extracted spectra. ferr belongs only to the latter scale.
source_electron_rate_pre_morph has units electron s^-1 cm^-2.
/zero_order holds standard-mode passband-integrated zeroth-order photometry;
_0th arrays are historical diagnostic proxies.
Negative noise is allowed. Symlog preserves it; the logarithmic S/N panel shows positive values only.
The truth windows in /lines are measurements at reference centres, not confirmed detections
or atomic-line identifications. detect_el_* holds matched noisy peaks, possibly several per source.
Old intri_el_idx refers to a cropped grid with a configuration-dependent *_init_idx offset;
prefer wavelengths or the new window table's full-grid center_index.
The Spectrum tab displays the base (no-WCE) branch of paired-wce; inspect *_wce in Structure/X-Y.
EL-only products lack parent spectra; use Structure/X-Y. Verify units/axes for generic HDF5 files.

Filters and export
──────────────────
Examples: (x>22) & (y>20), np.isfinite(x) & (np.abs(y)<10).
Only allowlisted array mathematics is supported, not arbitrary Python.
All plot types apply filters. CSV exports the plotted data; robust limits only change the view.
CSV omits full masks, covariance and provenance and cannot replace the scientific HDF5.
Exports cannot overwrite the open HDF5, including aliases of that file.
Shortcuts: Cmd-O/S/E on macOS, Ctrl-O/S/E elsewhere; Left/Right changes source.
"""


def help_text() -> str:
    """当前语言下的说明文本。"""
    return HELP_TEXT_EN if LANG.startswith("en") else HELP_TEXT


class HelpTab(ttk.Frame):
    def __init__(self, master, app: "App"):
        super().__init__(master, padding=6)
        self.app = app
        text = tk.Text(self, wrap="none", font=("Menlo", 11))
        self.text = text
        yscroll = ttk.Scrollbar(self, orient="vertical", command=text.yview)
        xscroll = ttk.Scrollbar(self, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        text.insert("1.0", help_text())
        text.configure(state="disabled")
        text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)


# --------------------------------------------------------------------------
# 主窗口
# --------------------------------------------------------------------------


class App(tk.Tk):
    def __init__(self, path: str | None = None, lang: str | None = None):
        super().__init__()
        self.model: ProductModel | None = None
        self.row_var = tk.IntVar(value=0)
        self.row_var.trace_add("write", self._on_row_var)
        self._last_row = 0
        # 语言是模块级全局量，每个实例都要按自己的设置重建一次
        env_lang = os.environ.get("CESS_H5_VIEWER_LANG", "")
        self.lang = "en" if str(lang or env_lang or "en").lower().startswith("en") else "zh"
        configure_language(self.lang)
        apply_tk_fonts(self)
        self.settings = load_settings()
        self._build_ui()
        self._restore_layout()

        if path:
            self.after(50, lambda: self.open_file(path))
        else:
            last = str(self.settings.get("last_file") or "")
            if last and os.path.exists(last):
                self._pending_reopen = self.after(80, lambda: self.open_file(last))

    def _build_ui(self) -> None:
        self.title(APP_TITLE)
        self.ui_style = configure_theme(self)
        geometry, minimum = initial_geometry(self)
        self.geometry(geometry)
        self.minsize(*minimum)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(self, style="Sidebar.TFrame", width=170, padding=(14, 22))
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)
        ttk.Label(sidebar, text="CESS", style="Brand.TLabel").pack(anchor="w", padx=8)
        ttk.Label(sidebar, text="HDF5 WORKSPACE", style="Sidebar.Muted.TLabel").pack(anchor="w", padx=8, pady=(4, 24))
        self.open_button = RoundedButton(sidebar, text=tf("打开文件", "Open file"), command=self.ask_open,
                                        background="#171d28", width=142)
        self.open_button.pack(fill="x", pady=(0, 24))
        self.nav_buttons = []
        for index, label in enumerate((tf("数据浏览", "Data browser"), tf("一维光谱", "Spectrum"),
                                        tf("探索绘图", "Explore plots"), tf("使用说明", "Guide"))):
            button = ttk.Button(sidebar, text=label, style="Nav.TButton",
                                command=lambda i=index: self._navigate(i))
            button.pack(fill="x", pady=3)
            self.nav_buttons.append(button)
        bottom = ttk.Frame(sidebar, style="Sidebar.TFrame")
        bottom.pack(side="bottom", fill="x")
        ttk.Label(bottom, text=tf("只读浏览 · v0.9.0", "Read-only · v0.9.0"), style="Sidebar.Muted.TLabel").pack(anchor="w", padx=8, pady=(0, 12))
        for label, lang in (("EN", "en"), ("简中", "zh")):
            ttk.Button(bottom, text=label, width=5, style="Nav.TButton",
                       command=lambda value=lang: self._switch_language(value)).pack(side="left", padx=2)
        main = ttk.Frame(self, style="App.TFrame", padding=(18, 12, 18, 8))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        header = ttk.Frame(main, style="Header.TFrame", padding=(14, 10))
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        self.file_label = ttk.Label(header, text=tf("尚未打开文件", "No file open"),
                                     width=35, style="Header.Title.TLabel", anchor="w")
        self.file_label.grid(row=0, column=0, sticky="ew")
        self.file_tip = ToolTip(self.file_label, tf("打开 HDF5 后浏览数据。", "Open an HDF5 file to begin."))
        self.file_summary = ttk.Label(header, text=tf("选择一个 CESS 产品或 BayeSED 输入星表", "Choose a CESS product or a BayeSED catalogue"),
                                       style="Header.Muted.TLabel")
        self.file_summary.grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.save_button = ttk.Button(header, text=tf("保存图片", "Save figure"), command=self.save_figure)
        self.save_button.grid(row=0, column=1, rowspan=2, padx=(10, 6))
        self.export_button = ttk.Button(header, text=tf("导出数据", "Export data"), command=self.export_data)
        self.export_button.grid(row=0, column=2, rowspan=2)
        self.notebook = ttk.Notebook(main, style="Workspace.TNotebook")
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.structure = StructureTab(self.notebook, self)
        self.spec = SpectrumTab(self.notebook, self)
        self.xy = XYTab(self.notebook, self)
        self.help = HelpTab(self.notebook, self)
        for tab, label in ((self.structure, "结构 / 数据"), (self.spec, "一维光谱"),
                           (self.xy, "X-Y 绘图"), (self.help, "说明")):
            self.notebook.add(tab, text=label)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.welcome = ttk.Frame(main, padding=30, style="Panel.TFrame")
        self.welcome.grid(row=1, column=0, sticky="nsew")
        self.welcome.columnconfigure(0, weight=1)
        self.welcome.rowconfigure(0, weight=1)
        card = ttk.Frame(self.welcome)
        card.grid(row=0, column=0)
        ttk.Label(card, text=tf("从一个文件开始探索", "A closer look at your spectra"),
                  style="Welcome.TLabel").pack(pady=(0, 12))
        ttk.Label(card, text=tf("浏览数据结构、查看每个源的光谱，或探索变量之间的关系。", "Browse the data, inspect a source, or explore relationships."),
                  style="Section.Muted.TLabel", wraplength=460, justify="center").pack(pady=(0, 24))
        RoundedButton(card, text=tf("选择 HDF5 文件", "Choose HDF5 file"), command=self.ask_open,
                      width=190).pack()
        ttk.Label(card, text=tf("文件以只读方式打开，原始数据不会被修改。", "Files open read-only. Your original data stays intact."),
                  style="Section.Muted.TLabel").pack(pady=(18, 0))
        self.status = ttk.Label(main, text=tf("就绪", "Ready"), style="Muted.TLabel", anchor="w")
        self.status.grid(row=2, column=0, sticky="ew", pady=(7, 0))
        self._build_menu()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        localize_tree(self)
        self._on_tab_changed()

    def _navigate(self, index):
        self.notebook.select(index)
        if self.model is not None or index == 3:
            self.welcome.grid_remove()
        else:
            self.welcome.grid()
        self._on_tab_changed()

    def _on_tab_changed(self, _event=None):
        index = self.notebook.index("current")
        if hasattr(self, "welcome"):
            if self.model is not None or index == 3:
                self.welcome.grid_remove()
            else:
                self.welcome.grid()
        for i, button in enumerate(self.nav_buttons):
            button.configure(style="Selected.Nav.TButton" if i == index else "Nav.TButton")
        enabled = self.model is not None and index in (1, 2)
        for button in (self.save_button, self.export_button):
            button.configure(state="normal" if enabled else "disabled")
        if index == 1:
            # 光谱页第一次显示时才套用保存的分隔位置（未映射时 Tk 不认 sashpos）
            self.after(60, self._restore_spectrum_sash)

    def _restore_spectrum_sash(self):
        if getattr(self, "_spectrum_sash_done", False):
            return
        self._spectrum_sash_done = True
        sashes = (self.settings or {}).get("sashes") or {}
        try:
            position = int(sashes.get("spectrum") or 0)
            if position > 0 and self.spec.panes.winfo_ismapped():
                self.spec.panes.sashpos(0, position)
        except (tk.TclError, TypeError, ValueError):
            pass

    def _show_file_header(self):
        if self.model is None:
            self.file_label.configure(text=tf("尚未打开文件", "No file open"))
            self.file_summary.configure(text=tf("选择一个 CESS 产品或 BayeSED 输入星表", "Choose a CESS product or a BayeSED catalogue"))
            self.file_tip.text = tf("打开 HDF5 后浏览数据。", "Open an HDF5 file to begin.")
            self.welcome.grid()
        else:
            self.file_label.configure(text=_shorten(os.path.basename(self.model.path), 58))
            self.file_tip.text = self.model.path
            self.file_summary.configure(text=tf("{n:,} 个源   ·   {d} 个数据集   ·   {size}",
                "{n:,} sources   ·   {d} datasets   ·   {size}", n=self.model.nsrc,
                d=len(self.model.dataset_paths()), size=human_bytes(os.path.getsize(self.model.path))))
            self.welcome.grid_remove()
        self._on_tab_changed()

    def set_language(self, lang: str) -> None:
        """运行中切换中英文：重建界面，再重新打开当前文件。"""
        target = "en" if str(lang).lower().startswith("en") else "zh"
        if target == ("en" if LANG.startswith("en") else "zh"):
            return
        current_geometry = self.geometry()
        current_path = getattr(self.model, "path", None)
        current_row = int(self._last_row)
        self.close_file(quiet=True)
        try:
            current_tab = self.notebook.index("current")
        except Exception:
            current_tab = 0
        self.lang = target
        configure_language(target)
        apply_tk_fonts(self)
        for child in self.winfo_children():
            child.destroy()
        self.model = None
        self._build_ui()
        self.geometry(current_geometry)
        try:
            self.notebook.select(current_tab)
        except Exception:
            pass
        if current_path:
            self.open_file(current_path)
            try:
                self.row_var.set(current_row)
            except Exception:
                pass

    def _switch_language(self, lang: str) -> None:
        try:
            self.set_language(lang)
        except Exception as exc:  # 切换失败时至少不要炸掉主窗口
            messagebox.showerror(APP_TITLE, f"切换语言失败：{exc}", parent=self)

    # ---- 菜单与快捷键 ----
    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开 HDF5 文件…", accelerator=("Cmd+O" if sys.platform == "darwin" else "Ctrl+O"),
                              command=self.ask_open)
        file_menu.add_separator()
        file_menu.add_command(label="保存当前图…", accelerator=("Cmd+S" if sys.platform == "darwin" else "Ctrl+S"), command=self.save_figure)
        file_menu.add_command(label="导出当前数据…", accelerator=("Cmd+E" if sys.platform == "darwin" else "Ctrl+E"), command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="关闭文件", command=self.close_file)
        file_menu.add_command(label="退出", command=self.on_close)
        menubar.add_cascade(label="文件", menu=file_menu)
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="结构 / 数据", command=lambda: self.notebook.select(0))
        view_menu.add_command(label="一维光谱", command=lambda: self.notebook.select(1))
        view_menu.add_command(label="X-Y 绘图", command=lambda: self.notebook.select(2))
        view_menu.add_command(label="说明", command=lambda: self.notebook.select(3))
        menubar.add_cascade(label="视图", menu=view_menu)

        lang_menu = tk.Menu(menubar, tearoff=0)
        self._lang_var = tk.StringVar(value="en" if LANG.startswith("en") else "zh")
        lang_menu.add_radiobutton(label="中文", value="zh", variable=self._lang_var,
                                  command=lambda: self._switch_language("zh"))
        lang_menu.add_radiobutton(label="English", value="en", variable=self._lang_var,
                                  command=lambda: self._switch_language("en"))
        menubar.add_cascade(label="语言 / Language", menu=lang_menu)
        self.configure(menu=menubar)

    def _bind_keys(self):
        self.bind(("<Command-" if sys.platform == "darwin" else "<Control-") + "o>", lambda _e: self.ask_open())
        self.bind(("<Command-" if sys.platform == "darwin" else "<Control-") + "s>", lambda _e: self.save_figure())
        self.bind(("<Command-" if sys.platform == "darwin" else "<Control-") + "e>", lambda _e: self.export_data())
        self.bind("<Left>", self._key_prev)
        self.bind("<Right>", self._key_next)

    def _typing(self) -> bool:
        """焦点是否在文本输入类控件里（此时不该劫持方向键）。"""
        widget = self.focus_get()
        return isinstance(
            widget, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Text, ttk.Combobox)
        )

    def _key_prev(self, _event=None):
        if self._typing():
            return
        if self.notebook.index("current") == 1:
            self.spec.step(-1)

    def _key_next(self, _event=None):
        if self._typing():
            return
        if self.notebook.index("current") == 1:
            self.spec.step(1)

    # ---- 文件 ----
    def ask_open(self):
        path = filedialog.askopenfilename(
            parent=self, title="选择 CESS HDF5 产品",
            initialdir=os.path.dirname(self.model.path) if self.model is not None else os.getcwd(),
            filetypes=[("HDF5", "*.hdf5 *.h5 *.hdf"), ("所有文件", "*.*")],
        )
        if path:
            self.open_file(path)

    def open_file(self, path: str):
        pending = getattr(self, "_pending_reopen", None)
        if pending is not None:
            # 用户抢在“恢复上次文件”之前自己打开了别的文件，取消排队的那次
            try:
                self.after_cancel(pending)
            except Exception:
                pass
            self._pending_reopen = None
        if not os.path.exists(path):
            messagebox.showerror(APP_TITLE, f"文件不存在：\n{path}", parent=self)
            return
        self.close_file(quiet=True)
        try:
            self.model = ProductModel(path)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"无法打开文件：\n{exc}", parent=self)
            return
        self.title(f"{APP_TITLE} — {os.path.basename(path)}")
        self.structure.populate()
        self.xy.refresh_variables()
        self.row_var.set(0)
        self.spec.compare_rows = []
        self.spec._refresh_compare_list()
        self.spec._sync_widgets()
        self.xy.pixel_var.set(0)
        self.structure.select_path(self._default_structure_path())
        self.spec.redraw(force=True)
        self.xy.plot_default()
        self.status.configure(text=tf("只读模式 · 就绪", "Read-only · Ready"))
        self._show_file_header()
        try:
            save_settings(self._collect_layout())
        except Exception:
            pass

    def _default_structure_path(self) -> str:
        """打开文件后在结构页默认展示的对象。"""
        for cand in (
            "catalog/parameters_desi",
            "parameters_desi",
            "catalog/ID",
            "ID",
            "spectra/GV/flux_ujy_with_noise",
            "GV/flux_ujy_with_noise",
            "spectra/GV/wave",
            "GV/wave",
        ):
            if cand in self.model.h5:
                return cand
        keys = list(self.model.h5.keys())
        return keys[0] if keys else ""

    def close_file(self, quiet: bool = False):
        for tab in (self.spec, self.xy):
            pending = getattr(tab, "_pending", None)
            if pending is not None:
                tab.after_cancel(pending)
                tab._pending = None
        if self.model is not None:
            self.model.close()
            self.model = None
        self.structure.clear()
        self.spec.fig.clear()
        self.spec.canvas.draw_idle()
        self.xy.fig.clear()
        self.xy.canvas.draw_idle()
        self.xy._all_choices = []
        self.xy.x_spec = self.xy.y_spec = self.xy.c_spec = self.xy.e_spec = None
        self.xy._export_snapshot = None
        self.xy._last = None
        self.xy._uses_row = False
        self.xy._default_hint = ""
        for var in (self.xy.x_var, self.xy.y_var, self.xy.c_var, self.xy.e_var):
            var.set("")
        self._last_row = 0
        self._show_file_header()
        if not quiet:
            self.status.configure(text="已关闭文件")
            self.title(APP_TITLE)

    def _on_row_var(self, *_args):
        if self.model is None:
            return
        try:
            value = int(float(self.row_var.get()))
        except Exception:
            # IntVar 里被写进非整数文本（例如手输 12.5）：退回上一次有效值
            fallback = int(getattr(self, "_last_row", 0))
            try:
                current = int(self.row_var.get())
            except Exception:
                current = None
            if current != fallback:
                self.row_var.set(fallback)
            return
        value = max(0, min(value, self.model.nsrc - 1))
        if value != self.row_var.get():
            self.row_var.set(value)
            return
        self._last_row = value
        self.spec.on_row_changed()
        self.xy.on_row_changed()

    # ---- 导出 ----
    def _active_figure(self):
        index = self.notebook.index("current")
        if index == 1:
            return self.spec.fig
        if index == 2:
            return self.xy.fig
        return None

    def save_figure(self):
        fig = self._active_figure()
        if fig is None:
            messagebox.showinfo(APP_TITLE, "请先切到「一维光谱」或「X-Y 绘图」页再保存。", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="保存图", defaultextension=".png",
            initialfile="figure.png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            validate_export_path(path, self.model.path if self.model is not None else None)
            fig.savefig(path, dpi=150, bbox_inches="tight")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"保存失败：{exc}", parent=self)
            return
        self.status.configure(text=f"已保存图 → {path}")

    def export_data(self):
        index = self.notebook.index("current")
        if index == 1:
            self.spec.export_spectrum()
        elif index == 2:
            self.xy.export_data()

    # ---- 记住窗口布局 ----
    def _restore_layout(self):
        geometry = usable_geometry(self, self.settings.get("geometry", ""))
        if geometry:
            try:
                self.geometry(geometry)
            except tk.TclError:
                pass
        size = str(self.settings.get("page_size") or "")
        if size.isdigit() and 20 <= int(size) <= 5000:
            try:
                self.structure.page_var.set(size)
            except Exception:
                pass
        self.after(200, self._restore_sashes)

    def _restore_sashes(self):
        sashes = self.settings.get("sashes") or {}
        if not isinstance(sashes, dict):
            return
        try:
            split = int(sashes.get("split") or 0)
            if split > 0:
                self.structure.split_paned.sashpos(0, split)
        except (tk.TclError, TypeError, ValueError):
            pass
        try:
            for index, position in enumerate(sashes.get("right") or []):
                self.structure.right_panes.sashpos(index, int(position))
        except (tk.TclError, TypeError, ValueError):
            pass
        self.structure._clamp_right_panes()

    def _collect_layout(self) -> dict:
        data = dict(self.settings or {})
        data["version"] = SETTINGS_VERSION
        try:
            data["geometry"] = self.geometry()
        except tk.TclError:
            pass
        try:
            right_count = len(self.structure.right_panes.panes()) - 1
            data["sashes"] = {
                "split": self.structure.split_paned.sashpos(0),
                "right": [self.structure.right_panes.sashpos(i) for i in range(max(0, right_count))],
                "spectrum": self.spec.panes.sashpos(0),
            }
        except (tk.TclError, IndexError):
            pass
        try:
            data["page_size"] = self.structure.page_var.get()
        except Exception:
            pass
        if self.model is not None:
            data["last_file"] = self.model.path
        return data

    def on_close(self):
        try:
            save_settings(self._collect_layout())
        except Exception:
            pass
        self.close_file(quiet=True)
        self.destroy()


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def launch(path=None, lang="en"):
    """Start the optional desktop GUI after the lightweight CLI parsed arguments."""
    app = App(path=path, lang=lang)
    app.mainloop()
    return 0
