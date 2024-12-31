from typing import Tuple

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box
from ray.rllib.utils.typing import MultiAgentDict

from MARLlib.marllib import marl
from MARLlib.marllib.envs.base_env import ENV_REGISTRY
from MarineEnv3 import MarineEnv
import time

# register all scenario with env class
REGISTRY = {}
REGISTRY["MarineNav"] = MarineEnv

# provide detailed information of each scenario
# mostly for policy sharing
policy_mapping_dict = {
    "MarineNav": {
        "description": "cooperative navigation in a marine environment",
        "team_prefix": ("red_", "blue_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    }
}

# must inherit from MultiAgentEnv class
class MarineNavEnv(MultiAgentEnv):
    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        self.env = REGISTRY[map](**env_config)
        self.action_space = self.env.get_action_space_dimension()
        self.num_agents = self.env.num_cooperative

    def reset(self) -> MultiAgentDict:
        obs = self.env.reset()
        return obs

    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs, rewards, dones, infos = self.env.step(action_dict)
        return obs, rewards, dones, infos
    def render(self, mode='human'):
        self.env.render()

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 100,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info