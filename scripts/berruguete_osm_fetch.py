# -*- coding: utf-8 -*-
#
# @File:   berruguete_osm_fetch.py
# @Author: Mastermind (para David Antizar / Ntizar)
#
# Descarga el extracto OSM de un barrio (por defecto Berruguete, Tetuan, Madrid)
# con el area suficiente para llenar el "layout" de CityDreamer.
#
# Por que Overpass y no la API /map:  el layout que consume scripts/inference.py
# se recorta del centro de la imagen a EXTENDED_VOL_SIZE=2880 px, y en el
# pipeline de datos 1 px = 1 m. La API /map limita a 50.000 nodos, asi que no
# da para 2880x2880 m de ciudad densa. Overpass si.
#
# Overpass no emite el elemento <bounds>, que utils.osm_helper.get_lnglat_bounds
# necesita para calcular la resolucion: se inyecta a mano tras la descarga.

import argparse
import math
import os
import sys
import urllib.parse
import urllib.request

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def overpass_query(minlat, minlon, maxlat, maxlon):
    bbox = f"{minlat:.6f},{minlon:.6f},{maxlat:.6f},{maxlon:.6f}"
    return f"""
[out:xml][timeout:600];
(
  way["highway"]({bbox});
  way["building"]({bbox});
  way["building:part"]({bbox});
  way["landuse"="construction"]({bbox});
  way["leisure"~"^(park|garden|playground|pitch|dog_park|nature_reserve|common)$"]({bbox});
  way["landuse"~"^(grass|forest|meadow|recreation_ground|village_green|allotments|cemetery|orchard)$"]({bbox});
  way["natural"~"^(wood|scrub|grassland|heath|water|wetland)$"]({bbox});
  way["waterway"~"^(riverbank|canal)$"]({bbox});
  way["landuse"~"^(reservoir|basin)$"]({bbox});
  relation["building"]({bbox});
  relation["natural"="water"]({bbox});
  relation["landuse"="forest"]({bbox});
);
(._;>;);
out body;
"""


def fetch(minlat, minlon, maxlat, maxlon):
    q = overpass_query(minlat, minlon, maxlat, maxlon)
    last = None
    for ep in ENDPOINTS:
        try:
            print("Overpass:", ep, flush=True)
            req = urllib.request.Request(
                ep,
                data=urllib.parse.urlencode({"data": q}).encode(),
                headers={"User-Agent": "MastermindBot/1.0 (david@ntizar.com)"},
            )
            xml = urllib.request.urlopen(req, timeout=620).read().decode("utf-8")
            if "<osm" in xml and len(xml) > 100000:
                return xml
            last = "respuesta corta (%d bytes)" % len(xml)
        except Exception as ex:  # noqa: BLE001
            last = repr(ex)
            print("  fallo:", last, flush=True)
    raise RuntimeError("Overpass no respondio: %s" % last)


def inject_bounds(xml, minlat, minlon, maxlat, maxlon):
    if "<bounds" in xml:
        return xml
    tag = '<bounds minlat="%.7f" minlon="%.7f" maxlat="%.7f" maxlon="%.7f"/>' % (
        minlat,
        minlon,
        maxlat,
        maxlon,
    )
    i = xml.index(">", xml.index("<osm"))
    return xml[: i + 1] + "\n" + tag + xml[i + 1 :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=40.4589)
    ap.add_argument("--lon", type=float, default=-3.7049)
    ap.add_argument("--size-m", type=float, default=3200.0, help="lado del recuadro en metros")
    ap.add_argument("--name", default="ES-Madrid-Berruguete")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    half_lat = args.size_m / 2 / 111320.0
    half_lon = args.size_m / 2 / (111320.0 * math.cos(math.radians(args.lat)))
    minlat, maxlat = args.lat - half_lat, args.lat + half_lat
    minlon, maxlon = args.lon - half_lon, args.lon + half_lon
    print(
        "Recuadro: lat %.6f..%.6f lon %.6f..%.6f (%.0f x %.0f m)"
        % (minlat, maxlat, minlon, maxlon, (maxlon - minlon) * 111320 * math.cos(math.radians(args.lat)), (maxlat - minlat) * 111320)
    )

    xml = inject_bounds(fetch(minlat, minlon, maxlat, maxlon), minlat, minlon, maxlat, maxlon)
    out = args.out or os.path.join("data", "xml", args.name + ".osm")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(
        "%s  %.2f MB | nodos %d, ways %d, relations %d"
        % (out, len(xml) / 1e6, xml.count("<node "), xml.count("<way "), xml.count("<relation "))
    )


if __name__ == "__main__":
    main()
