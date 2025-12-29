from envs.car_env_ray import SimpleCarEnv
from tracks.nascar_ring import get_walls as gw_nascar, get_spawn as gs_nascar
from tracks.simple_rectangle import get_walls as gw_rec, get_spawn as gs_rec
from tracks.high_speed_ring_gt import get_center_line as gcl1, get_spawn as gs_gt
from tracks.track_geometry import RectangularTrack, AngularTrack, generate_walls
from tracks.Catmull_Rom_geometry import catmull_rom_spline

# ---------- Create python env ---------
walls1 = gw_rec()
spawn1 = gs_rec()
track1 = RectangularTrack(walls1)
# --- Rectangle for the number 1 ---

outer1, inner1 = gw_nascar()
spawn2 = gs_nascar()
track2 = AngularTrack(outer1, inner1)
walls2 = [(outer1[i], outer1[i+1]) for i in range (len(outer1)-1)] + [(inner1[i], inner1[i+1]) for i in range (len(inner1)-1)]
# --- nascar for the number 2 ---

control_points = gcl1()
centerline = catmull_rom_spline(control_points)
outer2, inner2, _ = generate_walls (centerline)
track3 = AngularTrack(outer2, inner2)
walls3 = [(outer2[i], outer2[i+1]) for i in range (len(outer2)-1)] + [(inner2[i], inner2[i+1]) for i in range (len(inner2)-1)]
spawn3 = gs_gt()
# --- high speed ring in Gran Turismo for 3

env = SimpleCarEnv(spawn3, walls3, track3, 5, render_mode="human")
obs, _ = env.reset()


for _ in range(1000):
    action = env.action_space.sample() # Pour faire un peu de l'aléatoire et mieux tester le tout #1  # aucune dynamique
    env.step(action)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break