# -*- coding: utf-8 -*-
#
# @File:   render_terreno_3d.py
# @Author: Mastermind (para David Antizar / Ntizar)
#
# Volumetria de un sitio SOBRE SU TERRENO REAL (no sobre un plano).
#
# Relieve: teselas Terrarium (S3 de AWS, sin clave, hasta zoom 15 -> ~3,5 m/px
# cerca de lat 43). Para precision catastral en Espana el dato bueno es el MDT02
# del IGN (LiDAR 2 m, por hojas del Centro de Descargas); esto es lo que hay sin
# friccion y sirve para ver pendientes y desniveles reales.
#
# Edificios: huellas .osm con su altura (height -> building:levels x 4,26 -> mediana),
# cada uno apoyado en la cota del terreno bajo su centroide.
#
# Salida:
#   <out>_hillshade.png  2D: sombreado del relieve + huellas (fiable, sin ordenacion 3D)
#   <out>_3d.png         3D: superficie del terreno + edificios extruidos

import argparse
import io
import math
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import utils.osm_helper

UA = {"User-Agent": "MastermindBot/1.0 (david@ntizar.com)"}
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/%d/%d/%d.png"


def _tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _tile_lonlat(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def fetch_dem(lat, lon, z=15, ntiles=3):
    xf, yf = _tile_xy(lon, lat, z)
    x0, y0 = int(xf) - ntiles // 2, int(yf) - ntiles // 2
    dem = np.full((256 * ntiles, 256 * ntiles), np.nan, np.float32)
    ok = 0
    for i in range(ntiles):
        for j in range(ntiles):
            url = TERRARIUM % (z, x0 + i, y0 + j)
            try:
                raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
                a = np.array(Image.open(io.BytesIO(raw)).convert("RGB")).astype(np.float32)
                elev = a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0
                dem[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = elev
                ok += 1
            except Exception as ex:  # noqa: BLE001
                print("  tesela fallida:", url, ex)
    if ok == 0:
        raise RuntimeError("No se pudo descargar ninguna tesela de relieve")
    bounds = (
        _tile_lonlat(x0, y0 + ntiles, z)[0],
        _tile_lonlat(x0 + ntiles, y0, z)[0],
        _tile_lonlat(x0, y0 + ntiles, z)[1],
        _tile_lonlat(x0, y0, z)[1],
    )  # minlon, maxlon, minlat, maxlat
    return dem, bounds, ok


def sample(dem, bounds, lon, lat):
    minlon, maxlon, minlat, maxlat = bounds
    h, w = dem.shape
    x = (lon - minlon) / (maxlon - minlon) * (w - 1)
    y = (maxlat - lat) / (maxlat - minlat) * (h - 1)
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    return float(dem[yi, xi])


def load_buildings(osm_file):
    _, footprints, nodes = utils.osm_helper.get_highways_and_footprints(osm_file)
    nodes = utils.osm_helper.get_nodes_lng_lat(osm_file, nodes)
    heights = [
        float(v["tags"]["height"])
        for v in footprints.values()
        if v["tags"].get("height") not in (None, "")
        and str(v["tags"]["height"]).replace(".", "", 1).isdigit()
    ]
    med = float(np.median(heights)) if heights else 12.0
    out = []
    for _, fp in footprints.items():
        pts, lons, lats = [], [], []
        for nid in fp["nodes"]:
            n = nodes.get(nid)
            if not n or "lng" not in n:
                continue
            lons.append(n["lng"])
            lats.append(n["lat"])
        if len(lons) < 3:
            continue
        hgt = fp["tags"].get("height")
        try:
            hgt = float(hgt)
        except (TypeError, ValueError):
            lv = fp["tags"].get("building:levels")
            try:
                hgt = float(lv) * 4.26
            except (TypeError, ValueError):
                hgt = med
        out.append((np.array(lons), np.array(lats), max(3.0, hgt)))
    return out


def hillshade(dem, out_png, buildings, bounds, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    gy, gx = np.gradient(d, 1.0, 1.0)
    az, alt = np.radians(315.0), np.radians(45.0)
    slp = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    asp = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.sin(slp) + np.cos(alt) * np.cos(slp) * np.cos(az - asp)
    shade = (shade - shade.min()) / (np.ptp(shade) + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=130)
    fig.patch.set_facecolor("#0b0f14")
    ax.set_facecolor("#0b0f14")
    # hipsometria tenue (verde valle -> gris alto) bajo el sombreado
    dmin, dmax = np.percentile(d, [2, 98])
    ext = (bounds[0], bounds[1], bounds[2], bounds[3])
    ax.imshow(d, cmap="terrain", vmin=dmin, vmax=dmax, extent=ext, alpha=0.9)
    ax.imshow(shade, cmap="gray", extent=ext, alpha=0.5)
    for lons, lats, _ in buildings:
        ax.fill(lons, lats, facecolor="#ffffff", edgecolor="#0b0f14", linewidth=0.35, alpha=0.55)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_axis_off()
    ax.set_title(
        "%s — relieve real + %d edificios\ncotas %.0f–%.0f m (desnivel %.0f m)"
        % (title, len(buildings), np.nanmin(dem), np.nanmax(dem), np.nanmax(dem) - np.nanmin(dem)),
        color="#c8d3e0",
        fontsize=13,
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Guardado:", out_png)


def view3d(dem, bounds, buildings, out_png, title, max_buildings=900):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    d = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    step = max(1, d.shape[0] // 220)
    dd = d[::step, ::step]
    h, w = dd.shape
    minlon, maxlon, minlat, maxlat = bounds
    LON, LAT = np.meshgrid(np.linspace(minlon, maxlon, w), np.linspace(maxlat, minlat, h))
    z0 = float(np.nanpercentile(d, 5))

    fig = plt.figure(figsize=(15, 11), dpi=130)
    fig.patch.set_facecolor("#0b0f14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b0f14")
    # Sin esto, la superficie del terreno se pinta por encima de los edificios
    # (matplotlib ordena colecciones por profundidad MEDIA y la malla es enorme).
    ax.computed_zorder = False
    ax.plot_surface(LON, LAT, dd, cmap="terrain", linewidth=0, antialiased=False,
                    alpha=0.95, zorder=1)

    sel = buildings[:max_buildings]
    walls, wc, roofs = [], [], []
    for lons, lats, hgt in sel:
        base = sample(dem, bounds, float(lons.mean()), float(lats.mean()))
        if not np.isfinite(base):
            continue
        n = len(lons)
        for i in range(n):
            a = (lons[i], lats[i], base)
            b = (lons[(i + 1) % n], lats[(i + 1) % n], base)
            walls.append([[a[0], a[1], a[2]], [b[0], b[1], b[2]], [b[0], b[1], b[2] + hgt], [a[0], a[1], a[2] + hgt]])
            wc.append((0.85, 0.86, 0.88, 1.0))
        roofs.append([[lo, la, base + hgt] for lo, la in zip(lons, lats)])
    if walls:
        ax.add_collection3d(Poly3DCollection(walls, facecolors=wc, edgecolor=(0, 0, 0, 0.25),
                                            linewidths=0.25, zorder=3))
        ax.add_collection3d(Poly3DCollection(roofs, facecolors=(0.62, 0.20, 0.16, 1.0),
                                            edgecolor=(0, 0, 0, 0.35), linewidths=0.35, zorder=4))
    ax.set_zlim(z0 - 5, z0 + np.nanpercentile(d, 95) - z0 + 120)
    ax.set_box_aspect((1, 1, 0.35))
    ax.view_init(elev=32, azim=-118)
    ax.set_axis_off()
    ax.set_title(
        "%s — edificios apoyados en el terreno (%.0f–%.0f m)\n%d edificios · malla de relieve %dx%d"
        % (title, np.nanmin(dem), np.nanmax(dem), len(sel), *dd.shape),
        color="#c8d3e0",
        fontsize=13,
    )
    fig.tight_layout(pad=0)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Guardado:", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--zoom", type=int, default=15)
    ap.add_argument("--tiles", type=int, default=3)
    ap.add_argument("--name", default="")
    ap.add_argument("--out-prefix", default="output/terreno")
    ap.add_argument("--max-buildings", type=int, default=900)
    args = ap.parse_args()

    dem, bounds, ok = fetch_dem(args.lat, args.lon, args.zoom, args.tiles)
    px_m = 156543.03 * math.cos(math.radians(args.lat)) / (2 ** args.zoom)
    print(
        "Relieve: %d/%d teselas | %s px | %.2f m/px | cotas %.0f–%.0f m (desnivel %.0f m)"
        % (ok, args.tiles ** 2, dem.shape, px_m, np.nanmin(dem), np.nanmax(dem), np.nanmax(dem) - np.nanmin(dem))
    )
    bl = load_buildings(args.osm)
    cotas = np.array([sample(dem, bounds, float(l.mean()), float(b.mean())) for l, b, _ in bl])
    cotas = cotas[np.isfinite(cotas)]
    print(
        "Edificios: %d | cota bajo los edificios %.0f–%.0f m (desnivel %.0f m) | sd %.1f m"
        % (len(bl), cotas.min(), cotas.max(), np.ptp(cotas), cotas.std())
    )
    hillshade(dem, args.out_prefix + "_hillshade.png", bl, bounds, args.name)
    view3d(dem, bounds, bl, args.out_prefix + "_3d.png", args.name, args.max_buildings)


if __name__ == "__main__":
    main()
