"""circuit_fujispeedway.py
--------------------------
Circuit de Fujispeedway, réel, généré par spline Catmull-Rom.
Difficulté2— circuit complexe avec courbes enchaînées, virage serrés.
"""

from tracks.Catmull_Rom_geometry import catmull_rom_spline
from tracks.track_geometry import generate_walls, compute_centerline
from tracks.track_geometry import AngularTrack

def get_name():
    return "Fujispeedway"

def get_difficulty():
    return 0

def get_spawn():
    return   399, 66, 0.0

def _get_centerline():
    line = [
    (390, 66), (510, 66), (630, 69), (690, 75),
    (726, 105), (726, 150), (690, 195),
    (600, 255), (540, 330), (495, 390),
    (465, 435), (405, 450), (345, 420),
    (345, 345), (345, 285),
    (345, 225), (324, 192), (294, 204), (285, 240),
    (276, 300), (264, 360), (225, 396), (165, 405),
    (105, 396), (84, 375), (84, 330), (114, 315),
    (150, 300), (156, 255), (126, 225), (105, 195),
    (135, 174), (180, 174), (216, 186),
    (246, 180), (252, 150), (240, 114), (216, 90),
    (165, 78), (105, 84), (66, 105), (54, 75),
    (75, 75), (180, 66), (300, 66) 
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

