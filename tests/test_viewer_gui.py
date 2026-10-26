"""Headless checks of viewer labels and uncertainty-scale selection.

The real Tk window is validated separately; these tests require optional GUI
imports but do not create a window or require a display server.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_labels_error_scale_and_empty_source_information(tmp_path):
    if importlib.util.find_spec('_tkinter') is None or importlib.util.find_spec('matplotlib') is None:
        pytest.skip('Optional viewer dependencies are not installed')
    source = Path(__file__).resolve().parents[1] / 'src'
    code = '''
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
from matplotlib.figure import Figure
from cess.viewer import app as gui
from cess.viewer.data import ProductModel

gui.configure_language('en')
figure = Figure()
axes = figure.subplots()
axes.set_ylabel('计数')
assert axes.get_ylabel() == 'Count'
assert axes.text(0., 0., '筛选后没有数据点。').get_text() == 'No data points after filtering.'
gui.configure_language('zh')
axes.set_ylabel('计数')
assert axes.get_ylabel() == '计数'
gui.configure_language('en')

with h5py.File('product.hdf5', 'w') as handle:
    handle['catalog/ID'] = [b'one']
    handle['spectra/GV/wave'] = [4000., 5000., 6000.]
    for name in ('flux_ujy', 'flux_ujy_model', 'flux_ujy_with_noise'):
        handle['spectra/GV/' + name] = [[1., -1., 2.]]
    handle['spectra/GV/ferr'] = [[.1, .2, .3]]
yes = SimpleNamespace(get=lambda: True)
no = SimpleNamespace(get=lambda: False)
with ProductModel('product.hdf5') as model:
    tab = SimpleNamespace(app=SimpleNamespace(model=model), opt_0th=no, opt_err=yes,
                          opt_model=yes, opt_mask=no, opt_lines=no)
    incident = Figure().subplots()
    gui.SpectrumTab._draw_band(tab, incident, 'GV', 0, 'flux_ujy', 'uJy')
    assert len(incident.collections) == 0, 'calibrated ferr must not be attached to incident fnu'
    assert len(incident.lines) == 1, 'calibrated model must not be overplotted on incident fnu'
    observed = Figure().subplots()
    gui.SpectrumTab._draw_band(tab, observed, 'GV', 0, 'flux_ujy_with_noise', 'uJy')
    assert len(observed.collections) == 1
    assert len(observed.lines) == 2
    np.testing.assert_array_equal(observed.lines[0].get_ydata(), [1., -1., 2.])
assert dict((ds, unit) for _, ds, unit in gui.QUANTITIES)['source_electron_rate_pre_morph'] == 'electron s^-1 cm^-2'
class Information:
    text = ''
    def configure(self, **kwargs): pass
    def delete(self, *args): self.text = ''
    def insert(self, position, value): self.text += value
info = Information()
tab = SimpleNamespace(app=SimpleNamespace(model=SimpleNamespace(nsrc=0)), info=info)
gui.SpectrumTab._update_info(tab, 0)
assert info.text == 'No sources in this file.'
print('viewer label, scale and empty-source checks passed')
'''
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(tmp_path / 'mplconfig'))
    result = subprocess.run([sys.executable, '-s', '-B', '-c', code], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'checks passed' in result.stdout


@pytest.mark.parametrize("screen", [(1512, 982), (1024, 768), (800, 600)])
def test_initial_window_fits_screen(screen):
    pytest.importorskip('tkinter')
    from cess.viewer.ui import initial_geometry
    from types import SimpleNamespace
    import re
    root = SimpleNamespace(winfo_screenwidth=lambda: screen[0], winfo_screenheight=lambda: screen[1])
    geometry, minimum = initial_geometry(root)
    width, height, left, top = map(int, re.fullmatch(r'(\d+)x(\d+)\+(\d+)\+(\d+)', geometry).groups())
    assert 0 < minimum[0] <= width < screen[0]
    assert 0 < minimum[1] <= height < screen[1]
    assert left >= 0 and top >= 0
    assert left + width <= screen[0] and top + height <= screen[1]
