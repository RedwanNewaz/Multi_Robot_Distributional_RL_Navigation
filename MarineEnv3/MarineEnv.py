import numpy as np
import scipy.spatial
from . import robot
import gym
import json
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib as mpl
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class Core:
    """Vortex core with position and flow properties."""
    x: float
    y: float
    clockwise: bool
    Gamma: float  # circulation strength

@dataclass
class Obstacle:
    """Static obstacle with position and size."""
    x: float
    y: float
    r: float


class MarineEnv(gym.Env):
    init_display = False
    def __init__(self, seed: int = 0, schedule: dict = None):
        self.sd = seed
        self.rd = np.random.RandomState(seed)

        # Parameter initialization
        self.width = 50
        self.height = 50
        self.r = 0.5
        self.v_rel_max = 1.0
        self.p = 0.8
        self.v_range = [5, 10]
        self.obs_r_range = [1, 1]
        self.clear_r = 5.0
        self.timestep_penalty = -1.0
        self.collision_penalty = -50.0
        self.goal_reward = 100.0
        self.num_cores = 8
        self.num_obs = 8
        self.min_start_goal_dis = 30.0
        self.num_cooperative = 3
        self.num_non_cooperative = 3

        self.robots = [robot.Robot(cooperative=True) for _ in range(self.num_cooperative)] + \
                      [robot.Robot(cooperative=False) for _ in range(self.num_non_cooperative)]
        self.cores = []
        self.obstacles = []

        self.schedule = schedule
        self.episode_timesteps = 0
        self.total_timesteps = 0
        self.observation_in_robot_frame = True

    def get_action_space_dimension(self):
        return self.robot.compute_actions_dimension()

    def reset(self):
        if self.schedule:
            self.update_schedule()

        self.episode_timesteps = 0
        self.cores.clear()
        self.obstacles.clear()
        self.robots.clear()

        self.generate_robots()
        self.generate_vortex_cores()
        self.generate_obstacles()

        return self.get_observations()

    def update_schedule(self):
        steps = self.schedule["timesteps"]
        idx = np.searchsorted(steps, self.total_timesteps, side='right') - 1

        self.num_cooperative = self.schedule["num_cooperative"][idx]
        self.num_non_cooperative = self.schedule["num_non_cooperative"][idx]
        self.num_cores = self.schedule["num_cores"][idx]
        self.num_obs = self.schedule["num_obstacles"][idx]
        self.min_start_goal_dis = self.schedule["min_start_goal_dis"][idx]

    def generate_robots(self):
        robot_types = [True] * self.num_cooperative + [False] * self.num_non_cooperative
        for _ in range(500):
            if not robot_types:
                break
            start, goal = self.rd.uniform(2, self.width - 2, size=(2, 2))
            if self.check_start_and_goal(start, goal):
                rob = robot.Robot(robot_types.pop(0))
                rob.start = start
                rob.goal = goal
                self.reset_robot(rob)
                self.robots.append(rob)

    def generate_vortex_cores(self):
        centers = []
        for _ in range(500):
            if len(self.cores) >= self.num_cores:
                break
            center = self.rd.uniform(0, [self.width, self.height])
            direction = self.rd.binomial(1, 0.5)
            v_edge = self.rd.uniform(*self.v_range)
            Gamma = 2 * np.pi * self.r * v_edge
            core = Core(center[0], center[1], direction, Gamma)
            if self.check_core(core):
                self.cores.append(core)
                centers.append([core.x, core.y])
        if centers:
            self.core_centers = scipy.spatial.KDTree(centers)

    def generate_obstacles(self):
        for _ in range(500):
            if len(self.obstacles) >= self.num_obs:
                break
            center = self.rd.uniform(5, [self.width - 5, self.height - 5])
            r = self.rd.uniform(*self.obs_r_range)
            obs = Obstacle(center[0], center[1], r)
            if self.check_obstacle(obs):
                self.obstacles.append(obs)

    def reset_robot(self, rob):
        rob.reach_goal = False
        rob.collision = False
        rob.deactivated = False
        rob.init_theta = self.rd.uniform(0, 2 * np.pi)
        rob.init_speed = self.rd.uniform(0, rob.config.max_speed)
        rob.reset_state(current_velocity=self.get_velocity(rob.start[0], rob.start[1]))

    def check_start_and_goal(self, start, goal):
        if np.linalg.norm(goal - start) < self.min_start_goal_dis:
            return False
        if any(np.linalg.norm(rob.start - start) <= self.clear_r or np.linalg.norm(rob.goal - goal) <= self.clear_r
               for rob in self.robots):
            return False
        return True

    def check_core(self, core):
        if not (self.r <= core.x <= self.width - self.r and self.r <= core.y <= self.height - self.r):
            return False
        for rob in self.robots:
            if np.linalg.norm(core_pos := np.array([core.x, core.y]) - rob.start) < self.r + self.clear_r or \
                    np.linalg.norm(core_pos - rob.goal) < self.r + self.clear_r:
                return False
        return all(self.check_vortex_interaction(core, other_core) for other_core in self.cores)

    def check_vortex_interaction(self, core, other_core):
        dx, dy = other_core.x - core.x, other_core.y - core.y
        dis = np.hypot(dx, dy)
        if core.clockwise == other_core.clockwise:
            boundary_i = other_core.Gamma / (2 * np.pi * self.v_rel_max)
            boundary_j = core.Gamma / (2 * np.pi * self.v_rel_max)
            return dis >= boundary_i + boundary_j
        else:
            Gamma_l, Gamma_s = max(other_core.Gamma, core.Gamma), min(other_core.Gamma, core.Gamma)
            v_1, v_2 = Gamma_l / (2 * np.pi * (dis - 2 * self.r)), Gamma_s / (2 * np.pi * self.r)
            return v_1 <= self.p * v_2

    def check_obstacle(self, obs):
        if not (obs.r <= obs.x <= self.width - obs.r and obs.r <= obs.y <= self.height - obs.r):
            return False
        if any(np.linalg.norm(np.array([obs.x, obs.y]) - rob.start) < obs.r + self.clear_r or
               np.linalg.norm(np.array([obs.x, obs.y]) - rob.goal) < obs.r + self.clear_r for rob in self.robots):
            return False
        if any(np.hypot(obs.x - core.x, obs.y - core.y) <= self.r + obs.r for core in self.cores):
            return False
        return all(
            np.hypot(obs.x - other_obs.x, obs.y - other_obs.y) > obs.r + other_obs.r for other_obs in self.obstacles)

    def get_velocity(self, x: float, y: float):
        if not self.cores:
            return np.zeros(2)
        d, idx = self.core_centers.query([x, y], k=len(self.cores))
        if isinstance(idx, np.int64):
            idx = [idx]

        v_velocity = np.zeros(2)
        for i in idx:
            core = self.cores[i]
            dis = np.hypot(core.x - x, core.y - y)
            v_radial = np.array([core.x - x, core.y - y]) / dis
            rotation = np.array([[0, -1], [1, 0]]) if core.clockwise else np.array([[0, 1], [-1, 0]])
            v_tangent = rotation @ v_radial
            speed = self.compute_speed(core.Gamma, dis)
            v_velocity += v_tangent * speed

        return v_velocity

    def compute_speed(self, Gamma: float, d: float):
        return Gamma / (2 * np.pi * self.r * self.r) * d if d <= self.r else Gamma / (2 * np.pi * d)

    def get_observations(self):
        observations = [robot.perception_output(self.obstacles, self.robots, self.observation_in_robot_frame) for robot
                        in self.robots]
        return list(zip(*observations))

    def step(self, actions):
        rewards = [0] * len(self.robots)
        assert len(actions) == len(self.robots), "Number of actions not equal to number of robots!"
        assert not self.check_all_reach_goal(), "All robots reach goals, no actions are available!"

        # prev_robots = deepcopy(self.robots)
        for i, action in enumerate(actions):
            rob = self.robots[i]
            if rob.deactivated:
                continue
            rob.action_history.append(action)
            dis_before = rob.dist_to_goal()
            for _ in range(rob.config.N):
                rob.update_state(action, self.get_velocity(rob.x, rob.y))
            rob.trajectory.append([rob.x, rob.y, rob.theta, rob.speed, rob.velocity[0], rob.velocity[1]])
            dis_after = rob.dist_to_goal()
            rewards[i] += self.timestep_penalty + (dis_before - dis_after)

        # revert back robot position if it goes out of boundary
        # for i, outside in enumerate(self.out_of_boundary()):
        #     if outside:
        #         self.robots[i] = prev_robots[i]

        observations, collisions, reach_goals = self.get_observations()
        dones, infos = self.check_end_conditions(collisions, reach_goals)

        # update reward
        for idx, rob in enumerate(self.robots):
            if rob.deactivated:
                continue
            if rob.collision:
                rewards[idx] += self.collision_penalty
            elif rob.reach_goal:
                rewards[idx] += self.goal_reward

        self.episode_timesteps += 1
        self.total_timesteps += 1

        return observations, rewards, dones, infos

    def check_all_reach_goal(self):
        return all([rob.check_reach_goal() for rob in self.robots])

    def check_all_deactivated(self):
        return all([rob.deactivated for rob in self.robots])
    def check_end_conditions(self, collisions, reach_goals):
        dones = [False] * len(self.robots)
        infos = [{"state": "normal"}] * len(self.robots)

        for idx, rob in enumerate(self.robots):
            if rob.deactivated:
                dones[idx] = True
                infos[idx] = {
                    "state": "deactivated after collision" if rob.collision else "deactivated after reaching goal"}
                continue
            if self.episode_timesteps >= 1000:
                dones[idx] = True
                infos[idx] = {"state": "too long episode"}
            elif collisions[idx]:
                dones[idx] = True
                infos[idx] = {"state": "collision"}
            elif reach_goals[idx]:
                dones[idx] = True
                infos[idx] = {"state": "reach goal"}

        return dones, infos

    def out_of_boundary(self):
        return [not (rob.r <= rob.x <= self.width - rob.r and rob.r <= rob.y <= self.height - rob.r) for rob in self.robots]


    def reset_with_eval_config(self, eval_config):
        self.episode_timesteps = 0
        self.sd = eval_config["env"]["seed"]
        self.width = eval_config["env"]["width"]
        self.height = eval_config["env"]["height"]
        self.r = eval_config["env"]["r"]
        self.v_rel_max = eval_config["env"]["v_rel_max"]
        self.p = eval_config["env"]["p"]
        self.v_range = copy.deepcopy(eval_config["env"]["v_range"])
        self.obs_r_range = copy.deepcopy(eval_config["env"]["obs_r_range"])
        self.clear_r = eval_config["env"]["clear_r"]
        self.timestep_penalty = eval_config["env"]["timestep_penalty"]
        self.collision_penalty = eval_config["env"]["collision_penalty"]
        self.goal_reward = eval_config["env"]["goal_reward"]

        self.cores = [Core(*core) for core in zip(eval_config["env"]["cores"]["positions"],
                                                  eval_config["env"]["cores"]["clockwise"],
                                                  eval_config["env"]["cores"]["Gamma"])]
        self.obstacles = [Obstacle(*obs) for obs in zip(eval_config["env"]["obstacles"]["positions"],
                                                        eval_config["env"]["obstacles"]["r"])]
        self.robots = [self.create_robot_from_config(rob_config) for rob_config in eval_config["robots"]]

        centers = np.array([[core.x, core.y] for core in self.cores])
        if len(centers) > 0:
            self.core_centers = scipy.spatial.KDTree(centers)

        return self.get_observations()

    def create_robot_from_config(self, rob_config):
        rob = robot.Robot(rob_config["cooperative"])
        rob.dt = rob_config["dt"]
        rob.N = rob_config["N"]
        rob.length = rob_config["length"]
        rob.width = rob_config["width"]
        rob.r = rob_config["r"]
        rob.detect_r = rob_config["detect_r"]
        rob.goal_dis = rob_config["goal_dis"]
        rob.obs_dis = rob_config["obs_dis"]
        rob.max_speed = rob_config["max_speed"]
        rob.a = np.array(rob_config["a"])
        rob.w = np.array(rob_config["w"])
        rob.start = np.array(rob_config["start"])
        rob.goal = np.array(rob_config["goal"])
        rob.compute_k()
        rob.compute_actions()
        rob.init_theta = rob_config["init_theta"]
        rob.init_speed = rob_config["init_speed"]
        rob.perception.range = rob_config["perception"]["range"]
        rob.perception.angle = rob_config["perception"]["angle"]
        rob.reset_state(current_velocity=self.get_velocity(rob.start[0], rob.start[1]))
        return rob

    def episode_data(self):
        episode_data = {
            "env": {
                "seed": self.sd,
                "width": self.width,
                "height": self.height,
                "r": self.r,
                "v_rel_max": self.v_rel_max,
                "p": self.p,
                "v_range": list(self.v_range),
                "obs_r_range": list(self.obs_r_range),
                "clear_r": self.clear_r,
                "timestep_penalty": self.timestep_penalty,
                "collision_penalty": self.collision_penalty,
                "goal_reward": self.goal_reward,
                "cores": {
                    "positions": [[core.x, core.y] for core in self.cores],
                    "clockwise": [core.clockwise for core in self.cores],
                    "Gamma": [core.Gamma for core in self.cores]
                },
                "obstacles": {
                    "positions": [[obs.x, obs.y] for obs in self.obstacles],
                    "r": [obs.r for obs in self.obstacles]
                }
            },
            "robots": {
                "cooperative": [rob.cooperative for rob in self.robots],
                "dt": [rob.config.dt for rob in self.robots],
                "N": [rob.config.N for rob in self.robots],
                "length": [rob.config.length for rob in self.robots],
                "width": [rob.config.width for rob in self.robots],
                "r": [rob.r for rob in self.robots],
                "detect_r": [rob.detect_r for rob in self.robots],
                "goal_dis": [rob.config.goal_dis for rob in self.robots],
                "obs_dis": [rob.config.obs_dis for rob in self.robots],
                "max_speed": [rob.config.max_speed for rob in self.robots],
                "a": [list(rob.a) for rob in self.robots],
                "w": [list(rob.w) for rob in self.robots],
                "start": [list(rob.start) for rob in self.robots],
                "goal": [list(rob.goal) for rob in self.robots],
                "init_theta": [rob.config.init_theta for rob in self.robots],
                "init_speed": [rob.config.init_speed for rob in self.robots],
                "perception": {
                    "range": [rob.perception.range for rob in self.robots],
                    "angle": [rob.perception.angle for rob in self.robots]
                },
                "action_history": [copy.deepcopy(rob.action_history) for rob in self.robots],
                "trajectory": [copy.deepcopy(rob.trajectory) for rob in self.robots]
            }
        }
        return episode_data

    def save_episode(self, filename):
        with open(filename, "w") as file:
            json.dump(self.episode_data(), file)

    def render(self, mode='human'):
        if not self.init_display:
            self.initialize_env()
            self.init_display = True
        else:
            for i, rob in enumerate(self.robots):
                self.robot_plots[i].center = (rob.x, rob.y)
        plt.draw()
        plt.pause(0.001)

    def initialize_env(self):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw the environment boundary
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)

        # Draw vortex cores
        self.generate_background_image(ax)

        # Draw obstacles
        for obs in self.obstacles:
            circle = patches.Circle((obs.x, obs.y), obs.r, color='red', alpha=0.5)
            ax.add_patch(circle)

        # Draw robots
        self.robot_plots = []
        for rob in self.robots:
            circle = patches.Circle((rob.x, rob.y), rob.r, color='yellow' if rob.cooperative else 'orange', alpha=0.5)
            ax.add_patch(circle)
            self.robot_plots.append(circle)
            ax.plot([rob.x, rob.goal[0]], [rob.y, rob.goal[1]], 'k--')

        ax.set_aspect('equal', 'box')
        plt.tight_layout()


    def generate_background_image(self, axis):
        # plot current velocity in the map
        x_pos = list(np.linspace(0.0, self.width, 100))
        y_pos = list(np.linspace(0.0, self.height, 100))

        pos_x = []
        pos_y = []
        arrow_x = []
        arrow_y = []
        speeds = np.zeros((len(x_pos), len(y_pos)))
        for m, x in enumerate(x_pos):
            for n, y in enumerate(y_pos):
                v = self.get_velocity(x, y)
                speed = np.clip(np.linalg.norm(v), 0.1, 10)
                pos_x.append(x)
                pos_y.append(y)
                arrow_x.append(v[0])
                arrow_y.append(v[1])
                speeds[n, m] = np.log(speed)

        cmap = cm.Blues(np.linspace(0, 1, 20))
        cmap = mpl.colors.ListedColormap(cmap[10:, :-1])

        axis.contourf(x_pos, y_pos, speeds, cmap=cmap)
        axis.quiver(pos_x, pos_y, arrow_x, arrow_y, width=0.001, scale_units='xy', scale=2.0)