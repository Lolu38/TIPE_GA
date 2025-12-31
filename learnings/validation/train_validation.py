from learnings.validation.q_agent import QAgent
from envs.car_env_reward import SimpleCarEnv
from tracks.nascar_ring import get_walls as gw_nascar, get_spawn as gs_nascar
from tracks.simple_rectangle import get_walls as gw_rec, get_spawn as gs_rec
from tracks.high_speed_ring_gt import get_center_line as gcl1, get_spawn as gs_gt
from tracks.track_geometry import RectangularTrack, AngularTrack, generate_walls
from learnings.validation.mean_score import get_mean

outer1, inner1 = gw_nascar()
spawn = gs_nascar()
track = AngularTrack(outer1, inner1)
walls = [(outer1[i], outer1[i+1]) for i in range (len(outer1)-1)] + [(inner1[i], inner1[i+1]) for i in range (len(inner1)-1)]

"""control_points = gcl1()
centerline = catmull_rom_spline(control_points)
outer, inner, track_width = generate_walls (centerline)
track = AngularTrack(outer, inner)
walls = [(outer[i], outer[i+1]) for i in range (len(outer)-1)] + [(inner[i], inner[i+1]) for i in range (len(inner)-1)]
spawn = gs_gt()"""

env = SimpleCarEnv(
    spawn=spawn,
    walls=walls,
    track=track,
    track_width=None,
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

N_EPISODES = 500
MAX_STEPS = 300
tab_reward = []
slices = 50

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

    agent.decay()
    tab_reward.append(total_reward)
    if episode > 0 and episode % 30 == 0:
        print(
            f"Episode {episode} | "
            f"Reward: {total_reward:.1f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Alpha: {agent.alpha:.3f}"
        )

tab_max, tab_min, mean_slices, overall_max, overall_min, overall_mean = get_mean(tab_reward, slices)

for i in range(0,len(tab_max)):
    print(f"De {i*slices} à {i*slices + slices}: Max: {tab_max[i]} | Min: {tab_min[i]} | Moyenne: {mean_slices[i]}")
print(f"Maximum: {overall_max} | Minimum: {overall_min}, | Score moyen: {overall_mean}")
env.close()

env_show = SimpleCarEnv(
    spawn=spawn,
    walls=walls,
    track=track,
    track_width=None,
    nbr_rays=5,
    render_mode="human"
)

agent_show = QAgent(
    obs_space=env.observation_space,
    action_space=env.action_space,
    alpha=0.2,
    gamma=0.99,
    epsilon=1.0
)

state, _ = env.reset()
for step in range(MAX_STEPS):
    action_show = agent_show.select_action(state)
    next_state, reward, terminated, truncated, _ = env_show.step(action)

    agent_show.update(state, action, reward, next_state, terminated)

    state = next_state

    if terminated or truncated:
        break

env_show.close()

