"""w_track.py
--------------------------
Circuit w, imaginaire mais intéressant, généré par spline Catmull-Rom.
Difficulté 4 — circuit complexe avec courbes enchaînées, virage serrés.
"""

from tracks.Catmull_Rom_geometry import catmull_rom_spline
from tracks.track_geometry import generate_walls, compute_centerline
from tracks.track_geometry import AngularTrack

def get_name():
    return "w_track"

def get_difficulty():
    return 4

def get_spawn():
    return  9 * 31.400, 9 * 27.457, 0.0

def _get_centerline():
    line = [
    (9 * 31.400, 9 * 27.457),
    (9 * 44.000, 9 * 29.143),
    (9 * 48.500, 9 * 32.107),
    (9 * 54.400, 9 * 32.814),
    (9 * 58.400, 9 * 30.600),
    (9 * 62.700, 9 * 27.764),
    (9 * 68.800, 9 * 27.557),
    (9 * 71.700, 9 * 29.336),
    (9 * 74.000, 9 * 32.714),
    (9 * 72.100, 9 * 36.150),
    (9 * 69.000, 9 * 38.786),
    (9 * 64.600, 9 * 40.757),
    (9 * 59.000, 9 * 42.929),
    (9 * 52.000, 9 * 43.643),
    (9 * 45.000, 9 * 43.286),
    (9 * 42.000, 9 * 41.929),
    (9 * 38.800, 9 * 39.843),
    (9 * 33.800, 9 * 37.557),
    (9 * 29.800, 9 * 37.129),
    (9 * 25.200, 9 * 38.300),
    (9 * 19.600, 9 * 38.757),
    (9 * 14.800, 9 * 35.629),
    (9 * 12.000, 9 * 31.000),
    (9 * 7.5000, 9 * 24.250),
    (9 * 7.4000, 9 * 21.457),
    (9 * 11.800, 9 * 18.986),
    (9 * 19.200, 9 * 21.514),
    (9 * 24.400, 9 * 26.314)

    ]
    return line

def _build_walls_width():
    """Construction unique — partagée entre get_walls"""
    control_points = _get_centerline()
    centerline = catmull_rom_spline(control_points)
    outer, inner, width = generate_walls(centerline)
    return outer, inner, width

def get_width():
    _, _, width = _build_walls_width()
    return width

def get_walls():
    outer, inner, _ = _build_walls_width()
    walls  = [(inner[i], inner[i+1]) for i in range(len(inner) - 1)]
    walls += [(outer[i], outer[i+1]) for i in range(len(outer) - 1)]
    return walls

def get_checkpoints():
    outer, inner, _ = _build_walls_width()
    centerline = compute_centerline(outer, inner)
    return centerline[::20]

def get_track_geo():
    outer, inner, _ = _build_walls_width()
    outer, inner, _ = _build_walls_width()

    # outer et inner sont inversés sur ce circuit — on les swap
    outer_closed = inner + [inner[0]]
    inner_closed = outer + [outer[0]]

    return AngularTrack(outer_closed, inner_closed)

