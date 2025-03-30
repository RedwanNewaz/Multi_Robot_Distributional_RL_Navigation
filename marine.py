from typing import Tuple
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box
from ray.rllib.utils.typing import MultiAgentDict
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from MarineEnv3 import MarineEnv
import time
import gym
from random import choice

# Register all scenarios with env class
REGISTRY = {}
REGISTRY["MarineNav"] = MarineEnv

# Provide detailed information for each scenario
policy_mapping_dict = {
    "MarineNav": {
        "description": "Cooperative navigation in a marine environment",
        "team_prefix": ("agent_",),  # Changed to a single prefix
        "all_agents_one_policy": True,
        "one_agent_one_policy": False,  # Changed to False since we're using a shared policy
    }
}


class MarineNavEnv(MultiAgentEnv):
    def __init__(self, env_config):
        map_name = env_config.get("map_name")
        self.max_timesteps = env_config.get("max_timesteps")
        env_config.pop("map_name", None)
        env_config.pop("max_timesteps", None)

        if map_name not in REGISTRY:
            raise ValueError(f"Unknown map name: {map_name}")

        self.env = REGISTRY[map_name](**env_config)
        self.env.reset()
        self.env.max_timesteps = self.max_timesteps

        n = self.env.get_action_space_dimension()
        self.action_space = gym.spaces.Discrete(n)
        self.num_agents = self.env.num_cooperative  # Use the number from MarineEnv
        print(f"Number of agents: {self.num_agents}")
        # Define observation space
        self.observation_space = GymDict({
            "obs": Box(low=-100.0, high=100.0, shape=(39,), dtype=np.float64),
            "state": Box(low=-100.0, high=100.0, shape=(39,), dtype=np.float64)
        })



        env_config["map_name"] = map_name
        self.env_config = env_config
        self.agents = [f"agent_{i + 1}" for i in range(self.num_agents)]  # e.g., ["agent_1", "agent_2", ...]
        self.action_dict = {f"agent_{i + 1}": 0 for i in range(self.num_agents)}



    def reset(self) -> MultiAgentDict:
        observations, collisions, reach_goals = self.env.reset()
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            agent_obs = np.array(observations[i], dtype=np.float64)
            if agent_obs.shape != (39,):
                raise ValueError(f"Agent {agent} observation has shape {agent_obs.shape}, expected (39,)")
            obs_dict[agent] = {
                "obs": agent_obs,
                "state": agent_obs  # State is the same as obs for centralized critic
            }
        return obs_dict

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        # for key, val in action_dict.items():
        #     self.action_dict[key] = val
        for agent in self.action_dict:
            if agent in action_dict:
                self.action_dict[agent] = action_dict[agent]
            else:
                self.action_dict[agent] = 0

        raw_obs, rewards, done, info = self.env.step(self.action_dict)
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            agent_obs = np.array(raw_obs[i], dtype=np.float64)
            if agent_obs.shape != (39,):
                raise ValueError(f"Agent {agent} observation has shape {agent_obs.shape}, expected (39,)")
            obs_dict[agent] = {
                "obs": agent_obs,
                "state": agent_obs  # State is the same as obs for centralized critic
            }

        # Ensure that all agents have valid reward, done, and info
        rewards = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        done = {agent: done[i] for i, agent in enumerate(self.agents)}
        done["__all__"] = all(done.values())  # Mark if all agents are done
        info = {agent: info[i] for i, agent in enumerate(self.agents)}

        # reshape output
        # obs_dict = {agent: obs_dict[agent] for agent in action_dict}
        # rewards = {agent: rewards[agent] for agent in action_dict}
        # done = {agent: done[agent] for agent in action_dict}
        # done["__all__"] = all(done.values())
        # info = {agent: info[agent] for agent in action_dict}
        # agent = choice(list(action_dict.keys()))
        # obs_dict = {agent: obs_dict[agent]}
        # rewards = {agent: rewards[agent]}
        # done = {agent: done[agent] }
        # done["__all__"] = all(done.values())
        # info = {agent: info[agent]}

        return obs_dict, rewards, done, info

    def render(self, mode='human'):
        try:
            self.env.render()
        except Exception as e:
            print(f"Error in render: {e}")
            raise

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_timesteps,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

