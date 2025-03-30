import numpy as np

from MarineEnv3 import MarineEnv, Robot
from random import choice

if __name__ == '__main__':
    agent = Robot()
    env = MarineEnv()
    env.reset()
    indexes = [0, 1, 2]
    for _ in range(150):
        actions = {}
        for i in range(6):
            a_idx = choice(indexes)
            w_idx = choice(indexes)
            a = a_idx * len(agent.w) + w_idx
            actions[f"agent_{i + 1}"] = a
        observations, rewards, dones, infos = env.step(actions)
        # print(np.array(observations).shape)
        env.render()