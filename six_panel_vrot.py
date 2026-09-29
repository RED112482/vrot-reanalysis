"""White Lake pilot: six-panel Level-II figures and centroid-based Vrot."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from couplet import detect_couplet
from white_lake_vrot import RADARS, KT, NM, archive_files, distance_nm, velocity_palette, xy

FIELDS = {
    "velocity": ("velocity", "VEL"),
    "reflectivity": ("reflectivity", "REF"),
    "spectrum_width": ("spectrum_width", "SW"),
    "cc": ("cross_correlation_ratio", "correlation_coefficient", "copol_correlation_coeff", "RHOHV"),
}


def field(radar, sweep, kind):
    name = next((n for n in FIELDS[kind] if n in radar.fields), None)
    if name is None:
        return None
    data = np.ma.filled(radar.fields[name]["data"][radar.get_slice(sweep)].astype(float), np.nan)
    if not np.isfinite(data).any():
        return None
    return data * KT if kind in ("velocity", "spectrum_width") else data


def scan_time(radar, sweep, pyart):
    stamp = pyart.util.datetime_from_radar(radar)
    utc = datetime(stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute,
                   stamp.second, stamp.microsecond, tzinfo=timezone.utc)
    return utc + timedelta(seconds=float(np.nanmedian(radar.time["data"][radar.get_slice(sweep)])
                                         - radar.time["data"][0]))


def geometry(radar, sweep, center):
    lat, lon = float(radar.latitude["data"][0]), float(radar.longitude["data"][0])
    ox, oy = xy(center[0], center[1], lat, lon)
    x, y, _ = radar.get_gate_x_y_z(sweep, edges=False)
    ex, ey, _ = radar.get_gate_x_y_z(sweep, edges=True)
    return ((x - ox) / NM, (y - oy) / NM, (ex - ox) / NM, (ey - oy) / NM,
            (-float(ox / NM), -float(oy / NM)))


def pair_for(radar, sweep, center, target, radius, diameter):
    values = field(radar, sweep, "velocity")
    if values is None:
        return None
    x, y, _, _, radar_pos = geometry(radar, sweep, center)
    nearby = np.hypot(x - target[0], y - target[1]) <= radius + .5
    rays, gates = np.where(nearby)
    if not len(rays):
        return None
    r0, r1 = max(int(rays.min())-1, 0), min(int(rays.max())+2, x.shape[0])
    g0, g1 = max(int(gates.min())-1, 0), min(int(gates.max())+2, x.shape[1])
    return detect_couplet(values[r0:r1, g0:g1], x[r0:r1, g0:g1],
                          y[r0:r1, g0:g1], target, radar_pos, radius, diameter)


def unfiltered_vrot(radar, sweep, center, target, radius):
    values = field(radar, sweep, "velocity")
    if values is None:
        return None
    x, y, _, _, _ = geometry(radar, sweep, center)
    within = np.isfinite(values) & (np.hypot(x-target[0], y-target[1]) <= radius)
    return float((np.max(values[within]) - np.min(values[within])) / 2) if within.any() else None


def matching_moment_sweep(radar, reference, kind, pyart, time_limit=150):
    """Find the nearest same-elevation moment, including legacy split cuts."""
    elevation = float(np.nanmedian(radar.elevation["data"][radar.get_slice(reference)]))
    when = scan_time(radar, reference, pyart)
    choices = []
    for sweep in range(radar.nsweeps):
        angle = float(np.nanmedian(radar.elevation["data"][radar.get_slice(sweep)]))
        if abs(angle - elevation) > .3 or field(radar, sweep, kind) is None:
            continue
        dt = abs((scan_time(radar, sweep, pyart) - when).total_seconds())
        if dt <= time_limit:
            choices.append((dt + abs(angle-elevation)*120, sweep))
    return min(choices)[1] if choices else None


def panel(ax, fig, radar, sweep, kind, title, center, initial, current, radius, palette,
          pair=None, note=None):
    ax.set_facecolor("black")
    ax.set_title(title, color="white", loc="left", fontsize=10)
    ax.set_aspect("equal")
    extent = radius + 1
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values(): spine.set_color("#777777")
    if sweep is not None:
        values = field(radar, sweep, kind)
        if values is None:
            note = "Not in this Level-II volume"
        else:
            x, y, ex, ey, _ = geometry(radar, sweep, center)
            near = np.hypot(x - current[0], y - current[1]) <= radius + .75
            rays, gates = np.where(near)
            if len(rays):
                r0, r1, g0, g1 = int(rays.min()), int(rays.max()) + 1, int(gates.min()), int(gates.max()) + 1
                crop = np.ma.masked_where(~near[r0:r1, g0:g1] |
                                          ~np.isfinite(values[r0:r1, g0:g1]), values[r0:r1, g0:g1])
                cmap, norm, units = {
                    "velocity": (*palette, "kt"),
                    "reflectivity": ("turbo", Normalize(-10, 75), "dBZ"),
                    "spectrum_width": ("viridis", Normalize(0, 30), "kt"),
                    "cc": ("plasma", Normalize(.5, 1.0), "ρhv"),
                }[kind]
                mesh = ax.pcolormesh(ex[r0:r1+1, g0:g1+1], ey[r0:r1+1, g0:g1+1], crop,
                                     shading="flat", cmap=cmap, norm=norm, rasterized=True)
                cb = fig.colorbar(mesh, ax=ax, shrink=.68, pad=.01)
                cb.ax.tick_params(colors="white", labelsize=7)
                cb.set_label(units, color="white", fontsize=8)
            else:
                note = "No nearby gates"
    if note:
        ax.text(.5, .5, note, ha="center", va="center", transform=ax.transAxes, color="white",
                bbox={"facecolor":"#222222", "edgecolor":"none"}, fontsize=9)
    ax.add_patch(plt.Circle(current, radius, fill=False, edgecolor="white", linewidth=.8))
    ax.plot([initial[0], current[0]], [initial[1], current[1]], color="#ffe743", linewidth=1.7, zorder=8)
    ax.plot(*initial, "o", color="#ffe743", markersize=3, zorder=9)
    ax.plot(*current, "x", color="#ffe743", markersize=8, markeredgewidth=2, zorder=9)
    if pair is not None and kind == "velocity":
        inn, out = pair.inbound.xy, pair.outbound.xy
        ax.plot([inn[0], out[0]], [inn[1], out[1]], color="white", linewidth=1.1, zorder=9)
        ax.scatter([inn[0], out[0]], [inn[1], out[1]], c=["cyan", "orange"],
                   edgecolors="black", s=60, zorder=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent /
                        "MOB's Comprehensive ROTATION Project - Reanalysis Master.csv")
    parser.add_argument("--radar-files", type=Path)
    parser.add_argument("--output", type=Path, default=Path("White_Lake_2011-03-09"))
    parser.add_argument("--radius-nm", type=float, default=2.0)
    parser.add_argument("--max-diameter-nm", type=float, default=1.5)
    args = parser.parse_args()
    import pyart
    palette = velocity_palette()
    with args.csv.open(newline="", encoding="utf-8-sig") as f:
        case = next(csv.DictReader(f))
    day = datetime.strptime(case["Tornado Date (Zulu)"], "%m/%d/%Y").replace(tzinfo=timezone.utc)
    start = datetime.combine(day.date(), datetime.strptime(case["Begin Time (Zulu)"], "%H:%M").time(), tzinfo=timezone.utc)
    end = datetime.combine(day.date(), datetime.strptime(case["End Time (Zulu)"], "%H:%M").time(), tzinfo=timezone.utc)
    a = (float(case["Begin Lat"]), float(case["Begn Lon"]))
    b = (float(case["End Lat"]), float(case["End Lon"]))
    center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    station = min(RADARS, key=lambda s: distance_nm(center, RADARS[s]))
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.radar_files.glob("*") if args.radar_files else
                   archive_files(station, day, start, end, args.output / "level2"))
    rows = []
    for path in files:
        if not path.is_file(): continue
        try:
            radar = pyart.io.read_nexrad_archive(str(path), station=station)
        except Exception as exc:
            print(f"Skipping unreadable {path.name}: {exc}", flush=True)
            continue
        if field(radar, 0, "velocity") is None: continue
        low, upper = [], []
        for sweep in range(radar.nsweeps):
            sl = radar.get_slice(sweep)
            if sl.stop - sl.start < 3: continue
            elev = float(np.nanmedian(radar.elevation["data"][sl]))
            t = scan_time(radar, sweep, pyart)
            if elev <= .75 and start <= t <= end and field(radar, sweep, "velocity") is not None:
                low.append((sweep, elev, t))
            elif .75 < elev <= 6.0 and field(radar, sweep, "velocity") is not None:
                upper.append((sweep, elev, t))
        for sweep, elev, t in low:
            fraction = float(np.clip((t-start).total_seconds() / max((end-start).total_seconds(), 1), 0, 1))
            target = (a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1]))
            lat0, lon0 = float(radar.latitude["data"][0]), float(radar.longitude["data"][0])
            ox, oy = xy(center[0], center[1], lat0, lon0)
            sx, sy = xy(a[0], a[1], lat0, lon0)
            tx, ty = xy(target[0], target[1], lat0, lon0)
            initial, current = ((float((sx-ox)/NM), float((sy-oy)/NM)),
                                (float((tx-ox)/NM), float((ty-oy)/NM)))
            found = pair_for(radar, sweep, center, current, args.radius_nm, args.max_diameter_nm)
            raw_vrot = unfiltered_vrot(radar, sweep, center, current, args.radius_nm)
            candidates = []
            for up, angle, up_time in upper:
                if abs((up_time-t).total_seconds()) > 300: continue
                candidate = pair_for(radar, up, center, current, args.radius_nm, args.max_diameter_nm)
                if candidate: candidates.append((candidate.vrot, up, angle, up_time, candidate))
            selected = max(candidates, default=None, key=lambda item: item[0])
            up, angle, up_time, up_pair = selected[1:] if selected else (None, None, None, None)
            refl = matching_moment_sweep(radar, sweep, "reflectivity", pyart)
            sw = matching_moment_sweep(radar, sweep, "spectrum_width", pyart)
            cc = matching_moment_sweep(radar, sweep, "cc", pyart)
            upper_refl = matching_moment_sweep(radar, up, "reflectivity", pyart) if up is not None else None
            fig, axes = plt.subplots(3, 2, figsize=(15, 18), layout="constrained")
            fig.patch.set_facecolor("black")
            panels = [
                (axes[0,0], sweep, "velocity", f"Lowest velocity {elev:.2f}°", found, None),
                (axes[0,1], refl, "reflectivity", f"Lowest reflectivity {elev:.2f}°", None,
                 None if refl is not None else "No matching reflectivity cut"),
                (axes[1,0], sw, "spectrum_width", f"Spectrum width {elev:.2f}°", None,
                 None if sw is not None else "No matching spectrum-width cut"),
                (axes[1,1], cc, "cc", f"Correlation coefficient {elev:.2f}°", None,
                 None if cc is not None else "CC unavailable in this volume"),
                (axes[2,0], up, "velocity", f"Upper velocity {angle:.2f}°" if selected else "Upper velocity", up_pair,
                 None if selected else "No qualifying upper-tilt couplet"),
                (axes[2,1], upper_refl, "reflectivity", f"Matching upper reflectivity {angle:.2f}°" if selected else "Matching upper reflectivity", None,
                 None if upper_refl is not None else "No matching upper reflectivity"),
            ]
            for axis, which, kind, label, pair, note in panels:
                panel(axis, fig, radar, which, kind, label, center, initial, current,
                      args.radius_nm, palette, pair, note)
            desc = lambda p: (f"Vin {p.inbound.velocity:+.1f} | Vout {p.outbound.velocity:+.1f} | "
                              f"Vrot {p.vrot:.1f} kt | diameter {p.diameter_nm:.2f} nmi"
                              if p else "No coherent centroid pair")
            fig.suptitle(f"{case['Tornado Name']} | {station} | {t:%Y-%m-%d %H:%M:%S} UTC\n"
                         f"Lowest: {desc(found)}\nUpper: {desc(up_pair)}", color="white", fontsize=15)
            name = f"{station}_{t:%Y%m%d_%H%M%S}_sweep{sweep:02d}_six_panel.png"
            fig.savefig(args.output / name, dpi=130, facecolor="black")
            plt.close(fig)
            rows.append({
                "tornado": case["Tornado Name"], "scan_utc": t.isoformat(), "radar": station,
                "source_file": path.name, "sweep": sweep, "elevation_deg": round(elev, 2),
                "track_lat": round(target[0], 5), "track_lon": round(target[1], 5),
                "radar_range_nm": round(distance_nm(target, RADARS[station]), 2),
                "raw_vrot_kt": round(raw_vrot, 1) if raw_vrot is not None else "",
                "vin_kt": round(found.inbound.velocity, 1) if found else "",
                "vout_kt": round(found.outbound.velocity, 1) if found else "",
                "vrot_kt": round(found.vrot, 1) if found else "",
                "diameter_nm": round(found.diameter_nm, 2) if found else "",
                "in_centroid_east_nm": round(found.inbound.xy[0], 3) if found else "",
                "in_centroid_north_nm": round(found.inbound.xy[1], 3) if found else "",
                "out_centroid_east_nm": round(found.outbound.xy[0], 3) if found else "",
                "out_centroid_north_nm": round(found.outbound.xy[1], 3) if found else "",
                "in_gates": found.inbound.gates if found else "",
                "out_gates": found.outbound.gates if found else "",
                "upper_tilt_deg": round(angle, 2) if selected else "",
                "upper_scan_utc": up_time.isoformat() if selected else "",
                "upper_vrot_kt": round(up_pair.vrot, 1) if selected else "",
                "upper_diameter_nm": round(up_pair.diameter_nm, 2) if selected else "",
                "cc_available": cc is not None,
                "qc": "CANDIDATE_REVIEW_ALIASING" if found else "NO_PAIR", "image": name,
            })
            print(f"Rendered {name}: {desc(found)}", flush=True)
    if not rows: raise SystemExit("No lowest-tilt scans during reported tornado interval")
    with (args.output / "measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} six-panel scans to {args.output.resolve()}")


if __name__ == "__main__": main()
