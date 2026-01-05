from .geometry import closest_point_index

class ProgressReward:
    def __init__(self, centerline):
        self.centerline = centerline
        self.prev_idx = None
        self.no_progress_steps = 0

    def reset(self, car_state):
        pos = car_state["position"]
        self.prev_idx = closest_point_index(pos, self.centerline)
        self.no_progress_steps = 0

    def step(self, car_state, collision):
        pos = car_state["position"]
        idx = closest_point_index(pos, self.centerline)
        N = len(self.centerline)

        delta = (idx - self.prev_idx) % N
        if delta > N // 2:
            delta = 0

        reward = delta * 1.0

        if delta == 0:
            self.no_progress_steps += 1
            if self.no_progress_steps > 50:
                reward -= 1.0
        else:
            self.no_progress_steps = 0

        if collision:
            reward -= 50.0

        self.prev_idx = idx
        return reward
