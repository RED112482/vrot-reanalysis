"""Auditable, gate-based inbound/outbound couplet candidates.

These are exploratory measurements, not a tornado detection algorithm. Velocity
is assumed to be unfolded; folded/aliased data require manual review.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Lobe:
    xy: tuple[float, float]
    velocity: float
    peak: float
    gates: int


@dataclass
class Couplet:
    inbound: Lobe
    outbound: Lobe
    vrot: float
    diameter_nm: float
    midpoint_offset_nm: float
    radial_offset_nm: float
    raw_vrot: float


def _supported(data, eligible, tolerance=15.0):
    """Require 3 of the 8 neighboring gates to agree in sign and magnitude."""
    count = np.zeros(data.shape, dtype=np.uint8)
    for dr in (-1, 0, 1):
        for dg in (-1, 0, 1):
            if dr == dg == 0:
                continue
            neighbor = np.roll(np.roll(data, dr, axis=0), dg, axis=1)
            allowed = np.roll(np.roll(eligible, dr, axis=0), dg, axis=1)
            if dr == -1: allowed[-1, :] = False
            if dr == 1: allowed[0, :] = False
            if dg == -1: allowed[:, -1] = False
            if dg == 1: allowed[:, 0] = False
            count += allowed & (np.sign(neighbor) == np.sign(data)) & (np.abs(neighbor - data) <= tolerance)
    return eligible & (count >= 3)


def _lobes(data, x, y, eligible, polarity):
    magnitudes = polarity * data[eligible & (polarity * data > 0)]
    if magnitudes.size < 3:
        return []
    # Sign-specific threshold avoids one very large gate setting the cutoff.
    threshold = max(10.0, float(np.nanpercentile(magnitudes, 55)))
    mask = eligible & (polarity * data >= threshold)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    lobes = []
    for component in range(1, n + 1):
        where = labels == component
        if int(where.sum()) < 3:
            continue
        mag = polarity * data[where]
        # Top quartile of a spatially connected patch: resistant to one gate.
        core = where & (polarity * data >= np.nanpercentile(mag, 75))
        if int(core.sum()) < 2:
            core = where
        weights = np.maximum(polarity * data[core] - threshold + 1, 1)
        lobes.append(Lobe(
            xy=(float(np.average(x[core], weights=weights)),
                float(np.average(y[core], weights=weights))),
            velocity=float(np.median(data[core])),
            peak=float(np.max(data[where]) if polarity > 0 else np.min(data[where])),
            gates=int(where.sum()),
        ))
    return lobes


def detect_couplet(velocity_kt, x_nm, y_nm, target_xy_nm, radar_xy_nm=(0., 0.),
                   radius_nm=2.0, max_diameter_nm=1.5):
    """Return the strongest coherent pair in the 2 nmi target-centered circle.

    The lobes must each have >=3 connected gates with neighbor support. Their
    weighted centroids must be 0.15..max_diameter_nm apart, mostly separated
    across radar azimuth (similar radar range), and have a midpoint <=1.25 nmi
    from the target. Vrot uses the medians of the upper quartile of each lobe.
    """
    v = np.asarray(velocity_kt, dtype=float)
    x, y = np.asarray(x_nm, dtype=float), np.asarray(y_nm, dtype=float)
    tx, ty = target_xy_nm
    circle = np.isfinite(v) & (np.hypot(x - tx, y - ty) <= radius_nm)
    if np.count_nonzero(circle) < 8:
        return None
    raw = float((np.nanmax(v[circle]) - np.nanmin(v[circle])) / 2)
    eligible = _supported(v, circle)
    inbound = _lobes(v, x, y, eligible, -1)
    outbound = _lobes(v, x, y, eligible, +1)
    candidates = []
    rx, ry = radar_xy_nm
    for inn in inbound:
        for out in outbound:
            dx, dy = out.xy[0] - inn.xy[0], out.xy[1] - inn.xy[1]
            sep = float(np.hypot(dx, dy))
            if not 0.15 <= sep <= max_diameter_nm:
                continue
            midpoint = ((inn.xy[0] + out.xy[0]) / 2, (inn.xy[1] + out.xy[1]) / 2)
            offset = float(np.hypot(midpoint[0] - tx, midpoint[1] - ty))
            if offset > min(1.25, radius_nm):
                continue
            radial = np.array([midpoint[0] - rx, midpoint[1] - ry])
            radial /= max(float(np.linalg.norm(radial)), 1e-9)
            radial_offset = abs(float(np.dot([dx, dy], radial)))
            if radial_offset > min(.8, .7 * sep):
                continue
            vrot = (out.velocity - inn.velocity) / 2
            if vrot >= 10:
                candidates.append(Couplet(inn, out, float(vrot), sep, offset,
                                          radial_offset, raw))
    # Prefer stronger rotation, but penalize a large distance from track.
    return max(candidates, key=lambda c: c.vrot - 2 * c.midpoint_offset_nm,
               default=None)
