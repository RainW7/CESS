# 正值形态库与联合 strip 提取

[English](MGE_MODELS.md) | [简体中文](MGE_MODELS_zh-CN.md)

适用于CESS v0.9.0。这是可选的一维模型；默认运行使用随包旧库，无需先生成MGE库。运行与单位约定见[科学模型](SCIENCE_MODELS_zh-CN.md)。

## 1. 生成一个小型源基底

`cess-mge-library` 生成 `cess.positive_marginal_library` v1，每个新目录包含 `manifest.json` 和 `marginals.hdf5`。它采用未截断Sérsic的正值Gaussian近似，圆化Re及保面积椭圆变换；声明的PSF为圆Gaussian。它与旧pickle库及 `cess-profile-library` 生成的旧式bundle不是同一格式。

以下仅演示接口。在已激活CESS环境的源码根目录外，新建独立工作目录并执行：

```bash
mkdir -p /path/to/mge-work
cd /path/to/mge-work
python - <<'PY'
import json
from pathlib import Path
from cess.config.loader import load_config
config = load_config()
config['morphology_psf_fwhm'] = 0.2625  # arcsec，演示采用值，非仪器定标
Path('config.json').write_text(json.dumps(config, indent=2)+'\n')
grid = {'n': [1.0], 're_pixel': [4.0], 'pa_deg': [0.0, 45.0], 'axis_ratio': [0.5, 1.0]}
Path('grid.json').write_text(json.dumps(grid, indent=2)+'\n')
edges = {'x_edges_pixel': [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]}
Path('edges.json').write_text(json.dumps(edges, indent=2)+'\n')
PY
cess-mge-library --config config.json --grid grid.json --edges edges.json \
  --output basis --dry-run
```

检查计划后，去掉 `--dry-run` 才实际拟合和写出。输出父目录须存在，`basis` 须不存在。示例窄窗口和稀疏网格只用于小型接口检查，不能作为生产形态覆盖推荐。

四轴 `n/re_pixel/pa_deg/axis_ratio` 必须有限、非空且严格递增；0.5≤n≤5、Re>0、0<q≤1，PA单位为图像x轴角度。最多100000个笛卡尔profile，每方向1–2049个bin；边界相对中心0，可不等宽，缺省y边界复制x。像元尺度取 `arcsecperpix`，PSF取 `morphology_psf_fwhm`；后者未设时生成器采用0.2625 arcsec并记录，不能将其理解为CSST实测值。

默认数值控制为 `--l1-tolerance 1e-7 --tail-probability 1e-12 --max-components 160`。拟合不满足要求会失败，不自动放宽精度、截负值或重归一化有限窗口。失败staging保留以供检查，没有分片续跑机制。

只读核验使用当前Python包接口：

```bash
python - <<'PY'
from cess.morphology.bundle import verify_mge_bundle
print(verify_mge_bundle('/path/to/mge-work/basis'))
PY
```

核验检查字节SHA、结构、单位、行序、源系数和非负概率账目，不重新拟合，不打开manifest中记录的外部源路径，不认证真实PSF。完整格式见[库格式说明](MGE_BUNDLE_FORMAT.md)。

## 2. 显式启用 mge_strip

在完整运行配置的副本中删除旧 `widthlib_pkl`、`heightlib_pkl`，设置：

```json
{
  "positive_profile_library_manifest_path": "/path/to/mge-work/basis/manifest.json",
  "positive_profile_library_path": "/path/to/mge-work/basis/marginals.hdf5",
  "spectral_extraction": {
    "mode": "mge_strip",
    "spatial_pixels": 9,
    "kernel_radius_pixels": 16,
    "lsf_convention": "pre_psf"
  }
}
```

这只是需合入完整JSON的片段，不是可独立运行的配置。运行的PSF FWHM和像元尺度须与库声明严格相同；不同时提供旧 `profile_library_manifest_path`。支持空间像元数N=1–1001、色散半径K=0–1024的整数，核长2K+1；不再手动提供 `source_fraction`。

LSF角色必须显式选择：`pre_psf` 假设上游LSF未含点源PSF，在x/y都加指定PSF；`point_source` 假设色散PSF已含于上游LSF，只在y加PSF。示例的 `pre_psf` 是工作假设，不是软件自动判定的真实仪器属性。

先用少量源运行：

```bash
cess simulate --config /path/to/runtime.json -i /path/to/catalogue.hdf5 \
  -o /path/to/new_mge_results --source-stop 4 --all-effects --seed 1214
```

现代输出为schema3.0，逐源诊断位于 `/catalog/spectral_extraction` 和 `/catalog/mge_morphology`；legacy2.5对应根组。原始抽样形态参数保留，实际选中的网格参数另记。逐轴最近邻、端点选择及一维PA折叠均有明确记录，不把最近邻量化差异算入源fit误差。

## 3. 联合概率与电子预算

对圆Gaussian源分量尺度s，保面积椭圆变换后的协方差为S，det(S)=s⁴。令PSF标准差p=FWHM/(像元尺度×sqrt(8ln2))，两种角色使用：

$$
C=S+\mathrm{diag}(0,p^2)\quad(\mathrm{point\_source}),\qquad
C=S+\mathrm{diag}(p^2,p^2)\quad(\mathrm{pre\_psf}).
$$

对居中空间strip和全源归一化权重w，先合成整个mixture的概率，再条件化：

$$
F=\sum_k w_kP_k(y\in\mathrm{strip}),\quad
J_j=\sum_k w_kP_k(x\in\mathrm{bin}_j,y\in\mathrm{strip}),\quad
\mathrm{width}_j=J_j/F.
$$

不能逐分量条件化后仍沿用原权重。`height` 是积分全部x后的绝对空间bin概率，和约为F；两个独立marginal不能相乘充当这个二维矩形概率。

保留有限色散窗口质量M=sum(width)，不除以M重归一化。源电子期望应用F一次，再用未归一化width重分布；总保留比例为F×M。天空、暗电流和读出噪声使用N个空间像元，不乘F。关闭 `--morphology` 只将色散width替换为delta，仍保留F和噪声面积。

单分量用X|Y的Gaussian概率做正值积分。条件方差以正项det(C)/Cyy计算，避免近等数相减；可分极限解析处理，其他情况使用确定性积分。`quadrature_error_estimate` 是各分量max-bin误差估计之和，采用conditional单位；不是全核L1或总通量误差界。Gaussian尾遗漏另记，积分警告或不收敛显式失败。

## 4. 科学使用边界

假设SED与空间分布可分离、形态不随波长变化、strip居中、PSF为所声明的圆Gaussian、上游LSF可分离处理。没有速度场、任意二维孔径或完整弯曲trace。

整数源偏移bin对已经像元积分的光谱做离散重分布，是明确的量化近似；它不严格等于连续源和LSF卷积后只积分一次探测器像元。源fit L1和 `2*estimated_L1/F` 仅为数值误差指标，未包含网格选择差异，小F还会放大条件误差。生产应用需自行选择可追溯PSF/LSF、网格、strip和核窗口，开展相应科学验收。
