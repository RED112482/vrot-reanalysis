"""Pilot Vrot analysis for the first tornado in the supplied master CSV.

Requires: numpy, scipy, matplotlib, arm_pyart, boto3 (boto3 only for downloads).
Run: python white_lake_vrot.py
Offline: python white_lake_vrot.py --csv ... --radar-files path/to/LevelII/files
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np
from scipy.ndimage import median_filter

NM = 1852.0
KT = 1.94384449
RADARS = {"KMOB": (30.6794, -88.2397), "KEVX": (30.5646, -85.9215),
          "KLIX": (30.3367, -89.8254), "KDGX": (32.2800, -89.9840),
          "KEOX": (31.4605, -85.4594), "KBMX": (33.1723, -86.7697),
          "KTLH": (30.3975, -84.3289)}


def xy(lat, lon, lat0, lon0):
    return (np.asarray(lon) - lon0) * 111320 * np.cos(np.deg2rad(lat0)), (np.asarray(lat) - lat0) * 111120


def distance_nm(a, b):
    x, y = xy(a[0], a[1], b[0], b[1])
    return float(np.hypot(x, y) / NM)


def velocity_palette():
    """Read the user supplied -200 to +200 kt palette from its color strip."""
    strip = plt.imread(Path(__file__).resolve().parent / "velocity_palette.png")
    # The solid color strip occupies x=4..10, y=20..916. Image top is +200 kt.
    rgb = strip[20:917, 7, :3]
    if rgb.max() > 1:
        rgb = rgb / 255.0
    return ListedColormap(rgb[::-1], name="provided_velocity"), Normalize(-200, 200)


def archive_files(station, day, start, end, cache):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    # NOAA's former bucket stopped serving the archive in September 2025.
    bucket = "unidata-nexrad-level2"
    prefix = f"{day:%Y/%m/%d}/{station}/{station}{day:%Y%m%d}_"
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects = response.get("Contents", [])
    while response.get("IsTruncated"):
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=response["NextContinuationToken"])
        objects.extend(response.get("Contents", []))
    available = []
    for obj in objects:
        key = obj["Key"]
        match = re.search(r"_(\d{6})", key)
        if match and not key.endswith("MDM"):
            t = datetime.strptime(f"{day:%Y%m%d}{match[1]}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            if start - timedelta(minutes=12) <= t <= end + timedelta(minutes=3):
                available.append((t, key))
    # Include preceding volume because the first low sweep can predate the tornado.
    earlier = [(datetime.strptime(f"{day:%Y%m%d}{m[1]}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc), o["Key"])
               for o in objects if (m := re.search(r"_(\d{6})", o["Key"])) and not o["Key"].endswith("MDM")]
    before = [a for a in earlier if a[0] < start]
    if before:
        available.append(max(before))
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for _, key in sorted(set(available)):
        path = cache / Path(key).name
        if not path.exists():
            client.download_file(bucket, key, str(path))
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=Path(__file__).resolve().parent / "MOB's Comprehensive ROTATION Project - Reanalysis Master.csv",
                    help="Master CSV (defaults to the copy bundled beside this script)")
    ap.add_argument("--radar-files", type=Path, help="Directory of downloaded Level-II volumes; skips archive access")
    ap.add_argument("--output", type=Path, default=Path("White_Lake_2011-03-09"))
    ap.add_argument("--radius-nm", type=float, default=2.0)
    ap.add_argument("--max-diameter-nm", type=float, default=1.5)
    args = ap.parse_args()
    import pyart
    cmap, norm = velocity_palette()
    with args.csv.open(newline="", encoding="utf-8-sig") as f:
        case = next(csv.DictReader(f))
    day = datetime.strptime(case["Tornado Date (Zulu)"], "%m/%d/%Y").replace(tzinfo=timezone.utc)
    start = datetime.combine(day.date(), datetime.strptime(case["Begin Time (Zulu)"], "%H:%M").time(), tzinfo=timezone.utc)
    end = datetime.combine(day.date(), datetime.strptime(case["End Time (Zulu)"], "%H:%M").time(), tzinfo=timezone.utc)
    a = (float(case["Begin Lat"]), float(case["Begn Lon"]))
    b = (float(case["End Lat"]), float(case["End Lon"]))
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    station = min(RADARS, key=lambda s: distance_nm(mid, RADARS[s]))
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.radar_files.glob("*") if args.radar_files else archive_files(station, day, start, end, args.output / "level2"))
    if not files:
        raise SystemExit("No Level-II volumes found; provide --radar-files with downloaded volumes.")
    rows = []
    for path in files:
        if not path.is_file():
            continue
        try:
            radar = pyart.io.read_nexrad_archive(str(path), station=station)
        except Exception as exc:
            print(f"Skipping unreadable {path.name}: {exc}")
            continue
        if "velocity" not in radar.fields:
            continue
        for sweep in range(radar.nsweeps):
            sl = radar.get_slice(sweep)
            elev = float(np.nanmedian(radar.elevation["data"][sl]))
            # Lowest elevation only, including occasional supplemental sweeps.
            if elev > 0.75:
                continue
            radar_start = pyart.util.datetime_from_radar(radar)
            # Older Py-ART/cftime releases return a timezone-naive datetime.
            # Radar Level-II timestamps are UTC; reconstruct a standard aware
            # datetime so comparisons to the survey's UTC times are valid.
            radar_start_utc = datetime(radar_start.year, radar_start.month,
                                       radar_start.day, radar_start.hour,
                                       radar_start.minute, radar_start.second,
                                       radar_start.microsecond, tzinfo=timezone.utc)
            t = radar_start_utc + timedelta(seconds=float(
                np.nanmedian(radar.time["data"][sl]) - radar.time["data"][0]))
            if not (start <= t <= end):
                continue
            frac = np.clip((t - start).total_seconds() / max((end - start).total_seconds(), 1), 0, 1)
            target = (a[0] + frac * (b[0] - a[0]), a[1] + frac * (b[1] - a[1]))
            gx, gy, gz = radar.get_gate_x_y_z(sweep, edges=False)
            edge_x, edge_y, _ = radar.get_gate_x_y_z(sweep, edges=True)
            # Radar Cartesian coordinates: x east, y north. Convert tornado target likewise.
            tx, ty = xy(target[0], target[1], float(radar.latitude["data"][0]), float(radar.longitude["data"][0]))
            gx, gy = gx / NM, gy / NM
            edge_x, edge_y = edge_x / NM, edge_y / NM
            tx, ty = float(tx / NM), float(ty / NM)
            origin_x, origin_y = xy(mid[0], mid[1], float(radar.latitude["data"][0]), float(radar.longitude["data"][0]))
            origin_x, origin_y = float(origin_x / NM), float(origin_y / NM)
            start_x, start_y = xy(a[0], a[1], float(radar.latitude["data"][0]), float(radar.longitude["data"][0]))
            start_x, start_y = float(start_x / NM - origin_x), float(start_y / NM - origin_y)
            current_x, current_y = tx - origin_x, ty - origin_y
            d = np.hypot(gx - tx, gy - ty)
            raw = np.ma.filled(radar.fields["velocity"]["data"][sl], np.nan) * KT
            valid = np.isfinite(raw) & (d <= args.radius_nm)
            # Keep every available in-window low sweep, even if QC finds no pair.
            # A median gate is accepted only if the 3x3 local neighborhood contains
            # >=4 valid, same-sign gates within 12 kt of that median. No automatic
            # dealiasing: folded scans are flagged for human review.
            v = np.where(valid, raw, np.nan)
            med = median_filter(np.nan_to_num(v, nan=0), size=(3, 3), mode="nearest")
            support = np.zeros(v.shape, dtype=int)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    shifted = np.roll(np.roll(v, di, axis=0), dj, axis=1)
                    neighbor = np.isfinite(shifted) & (np.sign(shifted) == np.sign(med)) & (np.abs(shifted - med) <= 12)
                    if di == -1: neighbor[-1, :] = False
                    if di == 1: neighbor[0, :] = False
                    if dj == -1: neighbor[:, -1] = False
                    if dj == 1: neighbor[:, 0] = False
                    support += neighbor
            coherent = valid & (support >= 4) & (np.abs(v - med) <= 12)
            ins = np.argwhere(coherent & (v < 0))
            outs = np.argwhere(coherent & (v > 0))
            candidates = []
            for ii in ins:
                for oo in outs:
                    sep = float(np.hypot(gx[tuple(ii)] - gx[tuple(oo)], gy[tuple(ii)] - gy[tuple(oo)]))
                    if 0.15 <= sep <= args.max_diameter_nm:
                        candidates.append((float((v[tuple(oo)] - v[tuple(ii)]) / 2), ii, oo, sep))
            candidates.sort(key=lambda c: c[0], reverse=True)
            best = candidates[0] if candidates else None
            fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            # Actual polar gate boundaries, rather than screen-sized square dots.
            # Plot a small surrounding margin; keep analysis restricted to 2 nmi.
            nearby = d <= args.radius_nm + 0.75
            ray_indices, gate_indices = np.where(nearby)
            if not len(ray_indices):
                print(f"No plotted gates near track for {path.name} sweep {sweep}")
                plt.close(fig)
                continue
            r0, r1 = int(ray_indices.min()), int(ray_indices.max()) + 1
            g0, g1 = int(gate_indices.min()), int(gate_indices.max()) + 1
            display = np.ma.masked_where(~np.isfinite(raw[r0:r1, g0:g1]) |
                                         ~nearby[r0:r1, g0:g1], raw[r0:r1, g0:g1])
            plot = ax.pcolormesh(edge_x[r0:r1+1, g0:g1+1] - origin_x,
                                 edge_y[r0:r1+1, g0:g1+1] - origin_y, display,
                                 cmap=cmap, norm=norm, shading="flat", rasterized=True)
            colorbar = fig.colorbar(plot, ax=ax, label="Radial velocity (kt)",
                                    ticks=np.arange(-200, 201, 40), shrink=.77)
            colorbar.ax.tick_params(colors="white")
            colorbar.set_label("Radial velocity (kt)", color="white")
            ax.add_patch(plt.Circle((current_x, current_y), args.radius_nm,
                                    facecolor="none", edgecolor="white", linewidth=1.4))
            # Cumulative surveyed track: start to interpolated position at this scan.
            ax.plot([start_x, current_x], [start_y, current_y], color="#ffe743",
                    linewidth=2.3, zorder=6, label="Track through this scan")
            ax.plot(start_x, start_y, "o", color="#ffe743", markersize=6, zorder=7)
            ax.plot(current_x, current_y, "x", color="#ffe743", markersize=12,
                    markeredgewidth=2.5, zorder=8, label="Estimated tornado position")
            if best:
                _, ii, oo, sep = best
                ix, iy = gx[tuple(ii)] - origin_x, gy[tuple(ii)] - origin_y
                ox, oy = gx[tuple(oo)] - origin_x, gy[tuple(oo)] - origin_y
                ax.plot([ix, ox], [iy, oy], color="yellow", linewidth=1)
                ax.scatter([ix, ox], [iy, oy], c=["cyan", "orange"], edgecolors="black", s=95, zorder=5)
            extent = args.radius_nm + 1.0
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
                   xlabel="East of track midpoint (nmi)", ylabel="North of track midpoint (nmi)")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.grid(alpha=.15, color="white")
            ax.set_title(f"{case['Tornado Name']} • {station} • {t:%Y-%m-%d %H:%M:%S} UTC • {elev:.2f}°\n" +
                         (f"Vin {v[tuple(best[1])]:+.1f} kt | Vout {v[tuple(best[2])]:+.1f} kt | Vrot {best[0]:.1f} kt | {best[3]:.2f} nmi" if best else "No coherent pair within diameter limit"), color="white")
            outname = f"{station}_{t:%Y%m%d_%H%M%S}_sweep{sweep:02d}.png"
            fig.savefig(args.output / outname, dpi=170, facecolor=fig.get_facecolor())
            plt.close(fig)
            rows.append({"tornado": case["Tornado Name"], "scan_utc": t.isoformat(), "radar": station,
                         "source_file": path.name, "sweep": sweep, "elevation_deg": round(elev, 2),
                         "track_lat": round(target[0], 5), "track_lon": round(target[1], 5),
                         "radar_range_nm": round(distance_nm(target, RADARS[station]), 2),
                         "vin_kt": round(float(v[tuple(best[1])]), 1) if best else "",
                         "vout_kt": round(float(v[tuple(best[2])]), 1) if best else "",
                         "vrot_kt": round(best[0], 1) if best else "",
                         "diameter_nm": round(best[3], 2) if best else "",
                         "qc": "CANDIDATE_REVIEW_ALIASING" if best else "NO_PAIR",
                         "image": outname})
    if not rows:
        raise SystemExit("No lowest-tilt velocity sweeps found near tornado time.")
    with (args.output / "measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} scans and images to {args.output.resolve()}")


if __name__ == "__main__":
    main()
