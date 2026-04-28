"""
circuit_rectangle.py
--------------------
Circuit rectangulaire simple.
Difficulté 2 — virages à 90°, plus exigeant que l'ovale.
""" 
 
def get_difficulty():
    return 2
  
def get_spawn():
    x = 400
    y = 50
    theta = 0.0
    return x, y, theta 
 
def get_walls():
    outer = [
        ((0, 0), (800, 0)),
        ((800, 0), (800, 600)),
        ((800, 600), (0, 600)),
        ((0, 600), (0, 0))
    ]

    inner = [
        ((80, 80), (720, 80)),
        ((720, 80), (720, 520)),
        ((720, 520), (80, 520)),
        ((80, 520), (80, 80))
    ]

    return outer + inner

def get_width():
    return 80.0
 
def get_checkpoints():

    return [(400, 80), (720, 300), (400, 520), (80, 300)]

def get_name():
    return "rectangle"