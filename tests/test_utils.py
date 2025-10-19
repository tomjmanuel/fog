import pytest

pytest.importorskip("numpy")
import numpy as np

from fog.probability import bin_edges_ems39, bin_edges_std11, bin_edges_tbias, planck_radiance


def test_planck_radiance_monotonic():
    temps = np.array([270.0, 280.0, 290.0])
    rad = planck_radiance(11.0, temps)
    assert np.all(np.diff(rad) > 0)


def test_bin_edges_bounds():
    values = np.array([-10.0, 0.0, 10.0])
    ems_idx = bin_edges_ems39(values)
    std_idx = bin_edges_std11(values)
    tbias_idx = bin_edges_tbias(values)
    assert ems_idx.min() >= 0 and ems_idx.max() <= 31
    assert std_idx.min() >= 0 and std_idx.max() <= 31
    assert tbias_idx.min() >= 0 and tbias_idx.max() <= 31
