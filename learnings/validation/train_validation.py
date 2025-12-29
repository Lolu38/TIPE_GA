import gymnasium as gym
from learnings.validation.q_agent import QAgent
from env.car_env_ray import SimpleCarEnv
from envs.car_env_ray import SimpleCarEnv
from tracks.nascar_ring import get_walls as gw_nascar, get_spawn as gs_nascar
from tracks.simple_rectangle import get_walls as gw_rec, get_spawn as gs_rec
from tracks.high_speed_ring_gt import get_center_line as gcl1, get_spawn as gs_gt
from tracks.track_geometry import RectangularTrack, AngularTrack, generate_walls
from tracks.Catmull_Rom_geometry import catmull_rom_spline

outer1, inner1 = gw_nascar()
spawn = gs_nascar()
track = AngularTrack(outer1, inner1)
walls = [(outer1[i], outer1[i+1]) for i in range (len(outer1)-1)] + [(inner1[i], inner1[i+1]) for i in range (len(inner1)-1)]

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
MAX_STEPS = 500

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

    if episode % 50 == 0:
        print(
            f"Episode {episode} | "
            f"Reward: {total_reward:.1f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Alpha: {agent.alpha:.3f}"
        )

env.close()
