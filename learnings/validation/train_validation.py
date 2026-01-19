from learnings.validation.q_agent import QAgent
from envs.car_env_reward import SimpleCarEnv
from tracks.nascar_ring import get_walls as gw_nascar, get_spawn as gs_nascar
from tracks.simple_rectangle import get_walls as gw_rec, get_spawn as gs_rec
from tracks.high_speed_ring_gt import get_center_line as gcl1, get_spawn as gs_gt
from tracks.track_geometry import RectangularTrack, AngularTrack, generate_walls, compute_centerline
from learnings.validation.mean_score import get_mean
from tracks.Catmull_Rom_geometry import catmull_rom_spline
import numpy as np



outer, inner = gw_nascar()
spawn = gs_nascar()
track = AngularTrack(outer, inner)
widths = [
    np.linalg.norm(np.array(outer[i]) - np.array(inner[i]))
    for i in range(min(len(outer), len(inner)))
]
track_width = np.mean(widths)
centerline = compute_centerline(outer, inner)
checkpoint = centerline[::5]
print(len(centerline), " -> ", len(checkpoint))
walls = [(outer[i], outer[i+1]) for i in range (len(outer)-1)] + [(inner[i], inner[i+1]) for i in range (len(inner)-1)]


"""control_points = gcl1()
centerline = catmull_rom_spline(control_points)
outer, inner, track_width = generate_walls (centerline)
track = AngularTrack(outer, inner,)
centerline = compute_centerline(outer, inner)
checkpoint = centerline[::5]
print(len(centerline), " -> ", len(checkpoint))
walls = [(outer[i], outer[i+1]) for i in range (len(outer)-1)] + [(inner[i], inner[i+1]) for i in range (len(inner)-1)]
spawn = gs_gt()"""

env = SimpleCarEnv(
    spawn=spawn,
    walls=walls,
    track=track,
    centerline=checkpoint,
    track_width=track_width,
    nbr_rays=5,
    render_mode=None
)

agent = QAgent(
    obs_space=env.observation_space,
    action_space=env.action_space,
    alpha=0.2,
    gamma=0.99,
    epsilon=1.0
)

N_EPISODES = 1500
MAX_STEPS = 500
tab_reward = []
slices = 200

for episode in range(N_EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        agent.update(state, action, reward, next_state, terminated)

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break
    
    #print(reward)

    agent.decay()
    tab_reward.append(total_reward)
    if episode > 0 and episode % 30 == 0:
        print(
            f"Episode {episode} | "
            f"Reward: {total_reward:.1f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Alpha: {agent.alpha:.3f}"
        )

    agent.save("q_agent.npy")

tab_max, tab_min, mean_slices, overall_max, overall_min, overall_mean = get_mean(tab_reward, slices)

for i in range(0,len(tab_max)):
    print(f"De {i*slices} à {i*slices + slices}: Max: {tab_max[i]} | Min: {tab_min[i]} | Moyenne: {mean_slices[i]}")
print(f"Maximum: {overall_max} | Minimum: {overall_min}, | Score moyen: {overall_mean}")
env.close()

env_show = SimpleCarEnv(
    spawn=spawn,
    walls=walls,
    track=track,
    centerline=centerline,
    track_width=track_width,
    nbr_rays=5,
    render_mode="human"
)

agent = QAgent(
    obs_space=env.observation_space,
    action_space=env.action_space,
    alpha=0.2,
    gamma=0.99,
    epsilon=0.0
)

reward_fn = 0
agent.load("q_agent.npy")
for step in range(MAX_STEPS):
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, _ = env_show.step(action)

    #agent.update(state, action, reward, next_state, terminated)

    state = next_state
    reward_fn += reward

    if terminated or truncated:
        break
    
print(reward_fn)
env_show.close()