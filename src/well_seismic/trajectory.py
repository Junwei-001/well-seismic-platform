from __future__ import annotations

import numpy as np


def minimum_curvature(md: np.ndarray, inclination_deg: np.ndarray, azimuth_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return TVD, east and north displacement using the minimum-curvature method."""
    md = np.asarray(md, dtype=float)
    inc = np.deg2rad(np.asarray(inclination_deg, dtype=float))
    azi = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    if not (md.size == inc.size == azi.size):
        raise ValueError("MD, inclination and azimuth must have equal lengths")
    if md.size == 0 or np.any(np.diff(md) < 0):
        raise ValueError("MD must be non-empty and monotonically increasing")
    tvd = np.zeros_like(md)
    east = np.zeros_like(md)
    north = np.zeros_like(md)
    for i in range(1, md.size):
        delta_md = md[i] - md[i - 1]
        cos_dogleg = np.cos(inc[i] - inc[i - 1]) - np.sin(inc[i - 1]) * np.sin(inc[i]) * (1 - np.cos(azi[i] - azi[i - 1]))
        dogleg = np.arccos(np.clip(cos_dogleg, -1.0, 1.0))
        ratio = 1.0 if abs(dogleg) < 1e-12 else 2.0 / dogleg * np.tan(dogleg / 2.0)
        north[i] = north[i - 1] + delta_md / 2 * (np.sin(inc[i - 1]) * np.cos(azi[i - 1]) + np.sin(inc[i]) * np.cos(azi[i])) * ratio
        east[i] = east[i - 1] + delta_md / 2 * (np.sin(inc[i - 1]) * np.sin(azi[i - 1]) + np.sin(inc[i]) * np.sin(azi[i])) * ratio
        tvd[i] = tvd[i - 1] + delta_md / 2 * (np.cos(inc[i - 1]) + np.cos(inc[i])) * ratio
    return tvd, east, north


def interpolate_trajectory(md_samples: np.ndarray, station_md: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.interp(md_samples, station_md, values, left=np.nan, right=np.nan)
    inside = (md_samples >= station_md[0]) & (md_samples <= station_md[-1])
    result[~inside] = np.nan
    return result

