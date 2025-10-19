"""Fog probability estimation following the provided pseudo code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.ndimage import generic_filter

from .fetch import (
    SectorDefinition,
    fetch_ABI_L1b,
    fetch_ABI_LWP,
    fetch_ABI_cloud_mask,
    fetch_ABI_cloud_phase,
    fetch_ABI_geolocation,
    fetch_NWP_surface_temperature,
    fetch_clear_sky_transmittance_and_radiance,
    fetch_surface_emissivity_maps,
)
from .config import GOESConfig


def _get_data_var(dataset: xr.Dataset, name: str) -> xr.DataArray | None:
    try:
        return dataset[name]
    except KeyError:
        return None


@dataclass
class FogProbabilityDiagnostics:
    BT11: np.ndarray
    EMS39: np.ndarray
    TBIAS: np.ndarray
    STD11: np.ndarray
    eligibility_day: np.ndarray
    eligibility_night: np.ndarray
    REF065: np.ndarray | None = None
    REF39: np.ndarray | None = None
    LWP_DAY: np.ndarray | None = None
    CLOUD_MASK: np.ndarray | None = None


class LookupTablePlaceholder:
    def __init__(self, shape: Tuple[int, int], fill: float = 0.2):
        self.table = np.full(shape, fill, dtype=np.float32)

    def __getitem__(self, key):
        return self.table[key]


def load_LUT(name: str) -> LookupTablePlaceholder:
    shape = (32, 32)
    fill = 0.35 if "DAY" in name.upper() else 0.25
    return LookupTablePlaceholder(shape, fill=fill)


def planck_radiance(wavelength_um: float, temperature_K: np.ndarray) -> np.ndarray:
    c1 = 1.191042e8
    c2 = 1.4387752e4
    wavelength = wavelength_um
    radiance = c1 / (wavelength**5 * (np.exp(c2 / (wavelength * temperature_K)) - 1.0))
    return radiance


def planck_inverse_temperature(wavelength_um: float, radiance: np.ndarray) -> np.ndarray:
    c1 = 1.191042e8
    c2 = 1.4387752e4
    wavelength = wavelength_um
    temperature = c2 / (wavelength * np.log((c1 / (wavelength**5 * radiance)) + 1.0))
    return temperature


def window_std(values: np.ndarray, size: int = 3) -> np.ndarray:
    def _std_filter(block):
        return np.nanstd(block)

    footprint = np.ones((size, size))
    std = generic_filter(values, _std_filter, footprint=footprint, mode="nearest")
    return std


def bin_edges_ems39(values: np.ndarray) -> np.ndarray:
    bins = np.linspace(0.0, 2.0, 33)
    return np.clip(np.digitize(values, bins) - 1, 0, len(bins) - 2)


def bin_edges_tbias(values: np.ndarray) -> np.ndarray:
    bins = np.linspace(-20.0, 5.0, 33)
    return np.clip(np.digitize(values, bins) - 1, 0, len(bins) - 2)


def bin_edges_std11(values: np.ndarray) -> np.ndarray:
    bins = np.linspace(0.0, 2.0, 33)
    return np.clip(np.digitize(values, bins) - 1, 0, len(bins) - 2)


def build_fog_probability(scene_time, sector_bbox, config: GOESConfig | None = None):
    config = config or GOESConfig()
    sector = sector_bbox if isinstance(sector_bbox, SectorDefinition) else SectorDefinition(*sector_bbox)

    ch_vis = fetch_ABI_L1b("C02", scene_time, sector, config)
    ch_swir = fetch_ABI_L1b("C07", scene_time, sector, config)
    ch_tir = fetch_ABI_L1b("C14", scene_time, sector, config)

    geo = fetch_ABI_geolocation(scene_time, sector, config)
    cloud_mask = fetch_ABI_cloud_mask(scene_time, sector, config)
    cloud_phase = fetch_ABI_cloud_phase(scene_time, sector, config)
    lwp_day = fetch_ABI_LWP(scene_time, sector, config)

    emis39, emis11 = fetch_surface_emissivity_maps(3.9, 11.0, sector)
    tsfc_nwp = fetch_NWP_surface_temperature(scene_time, sector)
    tau_atm_11, ratm_11 = fetch_clear_sky_transmittance_and_radiance(11.0, scene_time, sector)

    lut_night_low = load_LUT("night_emis_lt_0p90")
    lut_night_high = load_LUT("night_emis_ge_0p90")
    lut_day = load_LUT("day_uniformity_vs_tbias")

    if "Rad" not in ch_tir:
        raise KeyError("C14 dataset must contain 'Rad' radiance field")
    shape = ch_tir["Rad"].shape
    p_fog_day = np.full(shape, np.nan, dtype=np.float32)
    p_fog_night = np.full(shape, np.nan, dtype=np.float32)

    sza = geo["solar_zenith_angle"].values
    is_day = sza < 90.0
    is_night = ~is_day

    quality_tir = _get_data_var(ch_tir, "DQF")
    quality = np.ones(shape, dtype=bool) if quality_tir is None else quality_tir.values == 0

    quality_vis = _get_data_var(ch_vis, "DQF")
    quality_swir = _get_data_var(ch_swir, "DQF")

    valid_day = is_day & quality
    valid_night = is_night & quality
    if quality_vis is not None:
        valid_day &= quality_vis.values == 0
    if quality_swir is not None:
        valid_day &= quality_swir.values == 0
        valid_night &= quality_swir.values == 0
    cloud_mask_flag = cloud_mask.values == 0
    valid_day &= cloud_mask_flag
    valid_night &= cloud_mask_flag

    cloud_phase_data = cloud_phase.values
    ice_flag = cloud_phase_data == 3
    eligible_day = valid_day & ~ice_flag
    eligible_night = valid_night & ~ice_flag

    radiance_tir = ch_tir["Rad"].values
    bt11 = planck_inverse_temperature(11.0, radiance_tir)
    rad_swir = _get_data_var(ch_swir, "Rad")
    r39_obs = rad_swir.values if rad_swir is not None else np.zeros_like(bt11)
    b39_at_bt11 = planck_radiance(3.9, bt11)
    ems39 = np.clip(r39_obs / b39_at_bt11, 0.0, 2.0)

    r11_obs = radiance_tir
    r_sfc_11 = (r11_obs - ratm_11) / tau_atm_11
    tsfc_11 = planck_inverse_temperature(11.0, r_sfc_11 / emis11)
    tbias = tsfc_11 - tsfc_nwp

    std11 = window_std(bt11, size=3)

    rad_vis = _get_data_var(ch_vis, "Rad")
    ref065 = rad_vis.values if rad_vis is not None else None
    ref_swir = _get_data_var(ch_swir, "Ref")
    if ref_swir is not None:
        ref39 = ref_swir.values
    elif rad_swir is not None:
        ref39 = rad_swir.values
    else:
        ref39 = None

    if np.any(eligible_night):
        emis_bin_low = emis39 < 0.90
        emis_bin_high = ~emis_bin_low
        ems39_bin = bin_edges_ems39(ems39)
        tbias_bin = bin_edges_tbias(tbias)
        idx_low = eligible_night & emis_bin_low
        idx_high = eligible_night & emis_bin_high
        p_fog_night[idx_low] = lut_night_low[ems39_bin[idx_low], tbias_bin[idx_low]]
        p_fog_night[idx_high] = lut_night_high[ems39_bin[idx_high], tbias_bin[idx_high]]

    if np.any(eligible_day):
        std11_bin = bin_edges_std11(std11)
        tbias_bin_day = bin_edges_tbias(tbias)
        p_fog_day[eligible_day] = lut_day[std11_bin[eligible_day], tbias_bin_day[eligible_day]]

    p_fog = np.full(shape, np.nan, dtype=np.float32)
    p_fog[is_day] = p_fog_day[is_day]
    p_fog[is_night] = p_fog_night[is_night]

    diagnostics = FogProbabilityDiagnostics(
        BT11=bt11,
        EMS39=ems39,
        TBIAS=tbias,
        STD11=std11,
        eligibility_day=eligible_day,
        eligibility_night=eligible_night,
        REF065=ref065,
        REF39=ref39,
        LWP_DAY=lwp_day.values if hasattr(lwp_day, "values") else None,
        CLOUD_MASK=cloud_mask.values,
    )

    return p_fog, diagnostics


def build_fog_mask_with_objects(
    P_FOG: np.ndarray,
    GEO: xr.Dataset,
    REF065: np.ndarray | None,
    REF39: np.ndarray | None,
    BT11: np.ndarray,
    TBIAS: np.ndarray,
    CLOUDPH: xr.DataArray,
    IS_DAY: np.ndarray,
    IS_NIGHT: np.ndarray,
) -> np.ndarray:
    membership = (P_FOG >= 0.40) & (CLOUDPH.values != 3)
    labels, _ = ndimage.label(membership)
    keep = np.zeros_like(labels, dtype=bool)
    for label in np.unique(labels):
        if label == 0:
            continue
        pix = labels == label
        if np.sum(pix) < 4:
            continue
        if np.sum(IS_DAY[pix]) > np.sum(IS_NIGHT[pix]):
            cond1 = np.nanmedian(REF065[pix]) > 0.20 if REF065 is not None else True
            cond2 = np.nanstd(REF065[pix]) < 0.05 if REF065 is not None else True
            cond3 = np.nanmedian(REF39[pix]) > 0.05 if REF39 is not None else True
            cond4 = True
            cond5 = np.nanmedian(TBIAS[pix]) > -10.0
            if cond1 and cond2 and cond3 and cond4 and cond5:
                keep[pix] = True
        else:
            cond1 = np.nanstd(BT11[pix]) < 0.5
            cond2 = np.nanmedian(TBIAS[pix]) > -15.0
            if cond1 and cond2:
                keep[pix] = True
    fog_mask = (labels > 0) & keep
    return fog_mask


def estimate_fog_depth(IS_DAY, FOG_MASK, LWP_DAY, EMS39):
    depth = np.full_like(FOG_MASK, np.nan, dtype=np.float32)
    idx_day = FOG_MASK & IS_DAY & np.isfinite(LWP_DAY)
    depth[idx_day] = LWP_DAY[idx_day] / 0.3
    idx_ngt = FOG_MASK & (~IS_DAY)
    a, b = -1159.93, 1295.70
    depth[idx_ngt] = a * EMS39[idx_ngt] + b
    return depth


__all__ = [
    "FogProbabilityDiagnostics",
    "build_fog_probability",
    "build_fog_mask_with_objects",
    "estimate_fog_depth",
]
