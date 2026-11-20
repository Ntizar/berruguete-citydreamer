# -*- coding: utf-8 -*-
#
# @File:   render_layout_3d.py
# @Author: Mastermind (para David Antizar / Ntizar)
#
# Vista 3D axonometrica del layout de un barrio, a partir del mismo .osm.
# Sirve para VER que el height field y las huellas son los del barrio real
# (Berruguete) antes de meterse con el render generativo de CityDreamer.
#
# Uso:
#   python scripts/render_layout_3d.py --osm data/xml/ES-Madrid-Berruguete.osm \
#       --crop-m 1100 --out output/berruguete_3d.png

import argparse
import math
import os
import sys

import numpy as np

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import utils.osm_helper


def load_footprints(osm_file, zoom):
    highways, footprints, nodes = utils.osm_helper.get_highways_and_footprints(osm_file)
    resolution = utils.osm_helper.get_map_resolution(
        {k: float(v) for k, v in utils.osm_helper.get_lnglat_bounds(osm_file).items()}, zoom
    )
    nodes = utils.osm_helper.get_nodes_xy_coordinates(nodes, resolution, zoom)
    bounds = utils.osm_helper.get_xy_bounds(nodes)
    footprints = utils.osm_helper.fix_missing_footprint_height(
        footprints, {"1/4": 12, "3/4": 26}  # Berruguete: manzana cerrada de 3-6 plantas
    )
    polys = []
    for _, fp in footprints.items():
        pts = []
        for nid in fp["nodes"]:
            n = nodes.get(nid)
            if not n or "x" not in n:
                continue
            pts.append((n["x"] - bounds["xmin"], n["y"] - bounds["ymin"]))
        if len(pts) < 3:
            continue
        h = float(fp["tags"].get("height") or 12)
        polys.append((np.array(pts, dtype=float), h, fp["tags"].get("building", "")))
    return polys, bounds


def render(polys, bounds, crop_m, out_png, max_buildings=1500, elev=38, azim=-118):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cx = (bounds["xmax"] - bounds["xmin"]) / 2
    cy = (bounds["ymax"] - bounds["ymin"]) / 2
    half = crop_m / 2
    sel = []
    for pts, h, kind in polys:
        c = pts.mean(axis=0)
        if abs(c[0] - cx) <= half and abs(c[1] - cy) <= half:
            sel.append((pts, h, kind))
    # Todos los edificios del recorte (si hay demasiados, los mas cercanos al centro)
    sel.sort(key=lambda t: float(np.hypot(t[0].mean(axis=0)[0] - cx, t[0].mean(axis=0)[1] - cy)))
    if len(sel) > max_buildings:
        sel = sel[:max_buildings]
    print("Edificios dibujados:", len(sel))

    fig = plt.figure(figsize=(16, 12), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b0f14")
    fig.patch.set_facecolor("#0b0f14")

    # NO se dibuja plano de suelo: matplotlib no ordena bien una losa grande frente
    # a miles de muros y la pinta ENCIMA, tapando la ciudad (verificado: con suelo,
    # solo se veian los edificios del borde). El fondo del eje hace de suelo.
    ax.set_zlim(bottom=0)

    # muros + cubierta
    walls, roofs, wall_c, roof_c = [], [], [], []
    for pts, h, kind in sel:
        h = max(4.0, h)
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            walls.append([[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], h], [a[0], a[1], h]])
            # sombreado por orientacion (luz desde el NO)
            d = math.atan2(b[1] - a[1], b[0] - a[0])
            shade = 0.42 + 0.30 * (0.5 + 0.5 * math.cos(d - 2.35))
            wall_c.append((0.62 * shade, 0.68 * shade, 0.75 * shade, 1.0))
        roofs.append([[p[0], p[1], h] for p in pts])
        roof_c.append((0.86, 0.88, 0.90, 1.0))

    ax.add_collection3d(
        Poly3DCollection(walls, facecolors=wall_c, edgecolor=(0, 0, 0, 0.25), linewidths=0.15)
    )
    ax.add_collection3d(
        Poly3DCollection(roofs, facecolors=roof_c, edgecolor=(0, 0, 0, 0.25), linewidths=0.2)
    )

    v = half * 0.78  # encuadre: recorta el margen vacio que deja la cubo de proyeccion
    ax.set_xlim(cx - v, cx + v)
    ax.set_ylim(cy - v, cy + v)
    ax.set_zlim(0, crop_m * 0.22)
    ax.set_box_aspect((1, 1, 0.22))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(
        "Berruguete (Tetuán, Madrid) — huellas y alturas de OpenStreetMap\n"
        "%d edificios · recorte %.0f × %.0f m · 1 px = 1 m (escala del layout de CityDreamer)"
        % (len(sel), crop_m, crop_m),
        color="#c8d3e0",
        fontsize=13,
        pad=6,
    )
    fig.tight_layout(pad=0)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    print("Guardado:", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm", default="data/xml/ES-Madrid-Berruguete.osm")
    ap.add_argument("--zoom", default=18, type=int)
    ap.add_argument("--crop-m", default=1100.0, type=float)
    ap.add_argument("--max-buildings", default=1500, type=int)
    ap.add_argument("--azim", default=-118, type=float)
    ap.add_argument("--elev", default=38, type=float)
    ap.add_argument("--out", default="output/berruguete_3d.png")
    args = ap.parse_args()
    polys, bounds = load_footprints(args.osm, args.zoom)
    print("Huellas totales en el extracto:", len(polys))
    render(polys, bounds, args.crop_m, args.out, args.max_buildings, args.elev, args.azim)


if __name__ == "__main__":
    main()
