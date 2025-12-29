from envs.car_env_ray import SimpleCarEnv
from tracks.nascar_ring import get_walls as gw_nascar, get_spawn as gs_nascar
from tracks.simple_rectangle import get_walls as gw_rec, get_spawn as gs_rec
from tracks.track_geometry import RectangularTrack, AngularTrack

# ---------- Create python env ---------
walls1 = gw_rec()
spawn1 = gs_rec()
track1 = RectangularTrack(walls1)
# --- Rectangle for the number 1 ---

outer, inner = gw_nascar()
spawn2 = gs_nascar()
track2 = AngularTrack(outer, inner)
walls2 = [(outer[i], outer[i+1]) for i in range (len(outer)-1)] + [(inner[i], inner[i+1]) for i in range (len(inner)-1)]
# --- nascar for the number 2 ---

env = SimpleCarEnv(spawn1, walls1, track1, 5, render_mode="human")
obs, _ = env.reset()


for _ in range(1000):
    action = env.action_space.sample() # Pour faire un peu de l'aléatoire et mieux tester le tout #1  # aucune dynamique
    env.step(action)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
