# -*- coding: utf-8 -*-
#
# @File:   berruguete_layout.py
# @Author: Mastermind (para David Antizar / Ntizar)
#
# Genera el "city layout" (height field + semantic map) que consume CityDreamer,
# a partir de un extracto .osm de OpenStreetMap de un barrio real.
#
# Equivale a la parte OSM de scripts/dataset_generator.py:get_osm_images(), pero
# sin arrastrar las extensiones CUDA (que solo hacen falta para el volumen 3D y
# el renderizado, no para rasterizar el barrio).
#
# Salida (formato que espera scripts/inference.py:get_osm_city_layout()):
#   <out_dir>/hf.png        height field en uint16 (metros, techo 640)
#   <out_dir>/seg.png       mapa semantico en modo P con la paleta de utils.helpers
#   <out_dir>/metadata.json resolucion (m/px) y bounds
#
# Uso:
#   python scripts/berruguete_layout.py --osm data/xml/ES-Madrid-Berruguete.osm \
#       --out data/osm/ES-Madrid-Berruguete --zoom 18 --min-size 2880

import argparse
import json
import logging
import math
import os
import sys

import numpy as np
from PIL import Image

PROJECT_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(PROJECT_HOME)

import utils.osm_helper  # solo depende de cv2/lxml/numpy, no de torch


def _get_seg_map_palette():
    """Misma paleta que utils/helpers.py:get_seg_map_palette()."""
    palette = np.array([[i, i, i] for i in range(256)])
    palette[:7] = np.array(
        [
            [0, 0, 0],        # 0 empty        (solo volumen)
            [96, 0, 0],       # 1 highway
            [96, 96, 0],      # 2 building
            [0, 96, 0],       # 3 green lands
            [0, 96, 96],      # 4 construction
            [0, 0, 96],       # 5 water
            [128, 128, 128],  # 6 ground
        ]
    )
    return palette


def save_seg_map(seg_map, path):
    img = Image.fromarray(seg_map.astype(np.uint8))
    img.putpalette(_get_seg_map_palette().reshape(-1).tolist())
    img.save(path)

# Mismas constantes que scripts/dataset_generator.py
HEIGHTS = {
    "ROAD": 4,
    "GREEN_LANDS": 8,
    "CONSTRUCTION": 10,
    "COAST_ZONES": 0,
    "ROOF": 1,
}
CLASSES = {
    "NULL": 0,
    "ROAD": 1,
    "BLD_FACADE": 2,
    "GREEN_LANDS": 3,
    "CONSTRUCTION": 4,
    "COAST_ZONES": 5,
    "OTHERS": 6,
    "BLD_ROOF": 7,
}
MAX_LAYOUT_HEIGHT = 640


def _tag_equals(tags, key, values=None):
    if key not in tags:
        return False
    if values is None:
        return True
    return tags[key] in values


def _get_highway_color(map_name, highway_tags):
    if map_name == "height_field":
        return HEIGHTS["ROAD"]
    return (
        CLASSES["NULL"]
        if "layer" in highway_tags and highway_tags["layer"] < 0
        else CLASSES["ROAD"]
    )


def _get_footprint_color(map_name, footprint_tags):
    if map_name == "height_field":
        if _tag_equals(footprint_tags, "height"):
            return int(float(footprint_tags["height"]) + 0.5)
        if _tag_equals(footprint_tags, "role", ["inner"]):
            return None
        if _tag_equals(footprint_tags, "building", ["roof"]):
            return None
        if _tag_equals(footprint_tags, "landuse", ["construction"]):
            return HEIGHTS["CONSTRUCTION"]
        raise Exception("Unknown height for tag: %s" % footprint_tags)
    if _tag_equals(footprint_tags, "role", ["inner"]):
        return CLASSES["BLD_FACADE"]
    if _tag_equals(footprint_tags, "building") or _tag_equals(
        footprint_tags, "building:part"
    ):
        return CLASSES["BLD_FACADE"]
    if _tag_equals(footprint_tags, "landuse", ["construction"]):
        return CLASSES["CONSTRUCTION"]
    return CLASSES["NULL"]


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _fix_missing_footprint_height(footprints):
    """Rellena la altura de cada huella, tolerante a datos europeos.

    El helper original (utils/osm_helper.py) revienta si NINGUNA huella trae la
    etiqueta `height`: en Madrid lo normal es tener `building:levels` y no `height`.
    Aqui se usa `building:levels` cuando existe y, si no, la mediana del barrio.
    """
    heights = [
        float(v["tags"]["height"])
        for v in footprints.values()
        if _is_number(v["tags"].get("height"))
    ]
    levels = [
        float(v["tags"]["building:levels"])
        for v in footprints.values()
        if not _is_number(v["tags"].get("height"))
        and _is_number(v["tags"].get("building:levels"))
    ]
    logging.info(
        "Alturas OSM: %d con height, %d solo con building:levels, %d sin nada",
        len(heights),
        len(levels),
        len(footprints) - len(heights) - len(levels),
    )
    pool = heights + [lv * 4.26 for lv in levels]
    if len(pool) >= 2:
        q1, q3 = float(np.percentile(pool, 25)), float(np.percentile(pool, 75))
    else:
        q1, q3 = 12.0, 24.0  # manzana cerrada de 3-6 plantas
    for v in footprints.values():
        tags = v["tags"]
        if _is_number(tags.get("height")):
            continue
        if _is_number(tags.get("building:levels")):
            tags["height"] = int(float(tags["building:levels"]) * 4.26)
        else:
            tags["height"] = int(np.random.uniform(q1, q3))
    return footprints


def _read_all_nodes(osm_file):
    """id -> {lng, lat} de TODOS los nodos (las areas verdes traen nodos propios)."""
    import lxml.etree

    nodes = {}
    for node in lxml.etree.parse(osm_file).findall("node"):
        if node.get("lon") is None or node.get("lat") is None:
            continue
        nodes[node.get("id")] = {"lng": float(node.get("lon")), "lat": float(node.get("lat"))}
    return nodes


AREA_TAG_RULES = [
    # (etiqueta, valores, clase)
    ("leisure", ["park", "garden", "playground", "pitch", "dog_park", "nature_reserve", "common"], 3),
    ("landuse", ["grass", "forest", "meadow", "recreation_ground", "village_green", "allotments", "cemetery", "orchard"], 3),
    ("natural", ["wood", "scrub", "grassland", "heath"], 3),
    ("natural", ["water", "wetland"], 5),
    ("waterway", ["riverbank", "canal"], 5),
    ("landuse", ["reservoir", "basin"], 5),
]


def get_area_polygons(osm_file):
    """Zonas verdes y agua a partir de las etiquetas OSM.

    Sustituye al heurístico de color sobre tiles de Carto de
    dataset_generator.get_green_lands(): no depende de un servicio externo (Carto
    ya devuelve PNG de 0 bytes sin referer válido) y usa el dato real de OSM.
    """
    import lxml.etree

    root = lxml.etree.parse(osm_file)
    areas = []
    for way in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        cls = None
        for key, values, c in AREA_TAG_RULES:
            if tags.get(key) in values:
                cls = c
                break
        if cls is None:
            continue
        refs = [n.get("ref") for n in way.findall("nd")]
        if len(refs) >= 3:
            areas.append((refs, cls))
    return areas


def _fill_areas(areas, all_nodes, xy_bounds, resolution, zoom_level, seg_map, height_field):
    import cv2

    for refs, cls in areas:
        pts = []
        for r in refs:
            n = all_nodes.get(r)
            if n is None:
                continue
            x, y = utils.osm_helper.lnglat2xy(n["lng"], n["lat"], resolution, zoom_level)
            pts.append((x - xy_bounds["xmin"], y - xy_bounds["ymin"]))
        if len(pts) < 3:
            continue
        cv2.fillPoly(seg_map, [np.int32(pts)], color=cls)
        cv2.fillPoly(height_field, [np.int32(pts)], color=(
            HEIGHTS["COAST_ZONES"] if cls == CLASSES["COAST_ZONES"] else HEIGHTS["GREEN_LANDS"]
        ))
    return seg_map, height_field


def build_layout(osm_file, zoom_level):
    import cv2

    highways, footprints, nodes = utils.osm_helper.get_highways_and_footprints(osm_file)
    resolution = utils.osm_helper.get_map_resolution(
        {
            k: float(v)
            for k, v in utils.osm_helper.get_lnglat_bounds(osm_file).items()
        },
        zoom_level,
    )
    nodes = utils.osm_helper.get_nodes_xy_coordinates(nodes, resolution, zoom_level)
    xy_bounds = utils.osm_helper.get_xy_bounds(nodes)
    logging.info("Bounds XY: %s (resolucion %.3f m/px)", xy_bounds, resolution)

    highways = utils.osm_helper.fix_missing_highway_width(highways)
    footprints = _fix_missing_footprint_height(footprints)
    logging.info(
        "OSM leido: %d calles, %d huellas de edificio"
        % (len(highways), len(footprints))
    )

    # --- Mapa semantico + height field (mismo orden que dataset_generator) ---
    all_nodes = _read_all_nodes(osm_file)
    areas = get_area_polygons(osm_file)
    logging.info("Zonas verdes/agua en OSM: %d poligonos" % len(areas))

    seg_map = utils.osm_helper.get_empty_map(xy_bounds)
    seg_map = utils.osm_helper.plot_highways(
        "seg_map", _get_highway_color, seg_map, highways, nodes, xy_bounds, resolution
    )
    height_field = utils.osm_helper.get_empty_map(xy_bounds, dtype=np.uint16)
    height_field[height_field == 0] = HEIGHTS["ROAD"]
    seg_map, height_field = _fill_areas(
        areas, all_nodes, xy_bounds, resolution, zoom_level, seg_map, height_field
    )
    seg_map = utils.osm_helper.plot_footprints(
        "seg_map", _get_footprint_color, seg_map, footprints, nodes, xy_bounds
    )
    # Pixels sin etiqueta = suelo
    seg_map[seg_map == 0] = CLASSES["OTHERS"]
    height_field = utils.osm_helper.plot_footprints(
        "height_field", _get_footprint_color, height_field, footprints, nodes, xy_bounds
    )
    height_field = np.clip(height_field, 0, MAX_LAYOUT_HEIGHT - 1).astype(np.uint16)
    seg_map = seg_map.astype(np.uint8)

    # Estadisticas del barrio (para verificar que es Berruguete y no ruido)
    bld = seg_map == CLASSES["BLD_FACADE"]
    stats = {
        "resolution_m_per_px": round(resolution, 4),
        "size_px": [int(seg_map.shape[0]), int(seg_map.shape[1])],
        "size_m": [
            round(seg_map.shape[0] * resolution, 1),
            round(seg_map.shape[1] * resolution, 1),
        ],
        "bounds_xy_px": xy_bounds,
        "n_highways": len(highways),
        "n_footprints": len(footprints),
        "bld_pixels": int(bld.sum()),
        "bld_coverage": round(float(bld.mean()), 4),
        "road_coverage": round(float((seg_map == CLASSES["ROAD"]).mean()), 4),
        "green_coverage": round(float((seg_map == CLASSES["GREEN_LANDS"]).mean()), 4),
        "height_p95_m": float(np.percentile(height_field[bld], 95)) if bld.any() else 0,
        "height_max_m": int(height_field.max()),
    }
    return height_field, seg_map, {"resolution": resolution, "bounds": xy_bounds, "stats": stats}


def pad_to(img, min_size, fill):
    h, w = img.shape[:2]
    if h >= min_size and w >= min_size:
        return img
    th, tw = max(h, min_size), max(w, min_size)
    pt, pl = (th - h) // 2, (tw - w) // 2
    pad = [(pt, th - h - pt), (pl, tw - w - pl)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad, mode="constant", constant_values=fill)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--zoom", default=18, type=int)
    parser.add_argument("--min-size", default=2880, type=int)
    args = parser.parse_args()

    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    hf, seg, meta = build_layout(args.osm, args.zoom)

    os.makedirs(args.out, exist_ok=True)
    hf_p = pad_to(hf, args.min_size, HEIGHTS["ROAD"])
    seg_p = pad_to(seg, args.min_size, CLASSES["OTHERS"])
    meta["stats"]["padded_to"] = list(hf_p.shape)

    Image.fromarray(hf_p).save(os.path.join(args.out, "hf.png"))
    save_seg_map(seg_p, os.path.join(args.out, "seg.png"))
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta["stats"], indent=2, ensure_ascii=False))
    print("Salida:", args.out)


if __name__ == "__main__":
    main()
