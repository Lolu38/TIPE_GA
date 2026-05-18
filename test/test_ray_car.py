from envs.car_env_ray import SimpleCarEnv
from tracks.circuit_nascar import get_walls as gw_nascar, get_spawn as gs_nascar, _create_walls
from tracks.circuit_rectangle import get_walls as gw_rec, get_spawn as gs_rec
from tracks.track_geometry import RectangularTrack, AngularTrack
from tracks.circuit_gp_chine import get_walls, get_spawn, get_width, _build_walls_width


# ---------- Create python env ---------
walls1 = gw_rec()
spawn1 = gs_rec()
track1 = RectangularTrack(walls1)
# --- Rectangle for the number 1 ---

outer, inner = _create_walls()
spawn2 = gs_nascar()
track2 = AngularTrack(outer, inner)
walls2 = [(outer[i], outer[i+1]) for i in range (len(outer)-1)] + [(inner[i], inner[i+1]) for i in range (len(inner)-1)]
# --- nascar for the number 2 ---

outer4, inner4, _ = _build_walls_width()
spawn4 = get_spawn()
track4 = AngularTrack(outer4, inner4)
walls4 = [(outer4[i], outer4[i+1]) for i in range (len(outer4)-1)] + [(inner4[i], inner4[i+1]) for i in range (len(inner4)-1)]
width4 = get_width
# --- suzuka for 4

env = SimpleCarEnv(spawn4, walls4, track4, nbr_rays=5, render_mode="human")
obs, _ = env.reset()


for _ in range(1000):
    action = env.action_space.sample() # Pour faire un peu de l'aléatoire et mieux tester le tout #1  # aucune dynamique
    env.step(action)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
