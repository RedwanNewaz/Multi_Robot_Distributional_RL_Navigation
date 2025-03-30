from typing import Tuple, Dict, Any
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
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()

        # Initialize with defaults
        self.map_name = env_config.get("map_name", "MarineNav")
        self.max_timesteps = env_config.get("max_timesteps", 1000)

        # Initialize underlying environment
        env_config.pop("map_name", None)
        env_config.pop("max_timesteps", None)
        self.env = REGISTRY[self.map_name](**env_config)
        self.env.max_timesteps = self.max_timesteps

        # Initialize spaces
        self.action_space = gym.spaces.Discrete(self.env.get_action_space_dimension())
        self.observation_space = self._get_observation_space()

        # Initialize agents
        self._initialize_agents()

        env_config["map_name"] = self.map_name
        self.env_config = env_config

        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self._last_obs = None
        self._eps_id = None

    def _initialize_agents(self):
        """Initialize agent tracking structures."""
        self.num_cooperative = self.env.num_cooperative
        self.agents = [f"agent_{i + 1}" for i in range(self.num_cooperative)]
        self._agent_ids = set(self.agents)

    def _get_observation_space(self):
        """Determine observation space from a test reset."""
        test_obs = self.env.reset()[0]
        obs_shape = (len(test_obs[0]),) if test_obs and len(test_obs[0]) == 39 else (39,)
        return GymDict({
            "obs": Box(low=-100.0, high=100.0, shape=obs_shape, dtype=np.float64),
            "state": Box(low=-100.0, high=100.0, shape=obs_shape, dtype=np.float64)
        })

    def reset(self) -> MultiAgentDict:
        """Reset with strict episode boundary enforcement."""
        self.current_episode += 1
        self.current_step = 0
        self._eps_id = self.current_episode  # Unique ID for this episode

        # Reset underlying environment
        observations, _, _ = self.env.reset()

        # Reinitialize agents in case count changed
        self._initialize_agents()

        # Build observation dict
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            agent_obs = self._process_observation(observations[i])
            obs_dict[agent] = {
                "obs": agent_obs,
                "state": agent_obs.copy()
            }

        self._last_obs = obs_dict
        return obs_dict

    def _process_observation(self, obs):
        """Ensure observation has consistent shape."""
        obs_array = np.asarray(obs, dtype=np.float64).flatten()
        if len(obs_array) != self.observation_space["obs"].shape[0]:
            # Pad or truncate if necessary
            expected_len = self.observation_space["obs"].shape[0]
            if len(obs_array) > expected_len:
                obs_array = obs_array[:expected_len]
            else:
                obs_array = np.pad(obs_array, (0, expected_len - len(obs_array)))
        return obs_array

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Step with trajectory boundary enforcement."""
        self.current_step += 1

        # Convert actions to list format
        actions = [action_dict.get(agent, 0) for agent in self.agents]

        # Execute step
        observations, rewards, dones, infos = self.env.step(
            {f"agent_{i + 1}": act for i, act in enumerate(actions)}
        )

        # Process outputs
        obs_dict = {}
        rewards_dict = {}
        dones_dict = {}
        infos_dict = {}

        for i, agent in enumerate(self.agents):
            obs_dict[agent] = {
                "obs": self._process_observation(observations[i]),
                "state": self._process_observation(observations[i])
            }
            rewards_dict[agent] = float(rewards[i])
            # dones_dict[agent] = bool(dones[i])
            infos_dict[agent] = infos[i] if i < len(infos) else {}
            infos_dict[agent]["eps_id"] = self._eps_id  # Track episode ID

        # Enforce termination
        timeout = self.current_step >= self.max_timesteps
        dones_dict["__all__"] = all(dones_dict.values()) or timeout

        if dones_dict["__all__"]:
            for agent in self.agents:
                dones_dict[agent] = True
                if timeout:
                    infos_dict[agent]["timeout"] = True

        return obs_dict, rewards_dict, dones_dict, infos_dict

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_cooperative,
            "episode_limit": self.max_timesteps,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info