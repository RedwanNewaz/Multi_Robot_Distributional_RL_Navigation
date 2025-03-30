import scipy.spatial
from . import robot
import gym
import json
import copy
from dataclasses import dataclass
import cv2
import numpy as np

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
        self.max_timesteps = 1000

    def get_action_space_dimension(self):
        return self.robots[0].actions_dimension

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
        print("Generating robots... ", self.num_cooperative, self.num_non_cooperative)
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
        for i, (agent, action) in enumerate(actions.items()):
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
            if self.episode_timesteps >= self.max_timesteps:
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
        """
        Render the environment using OpenCV (cv2).

        Parameters:
        mode (str): Rendering mode, supports 'human' or 'rgb_array'

        Returns:
        numpy.ndarray: The rendered image if mode is 'rgb_array', otherwise None
        """


        # Create a blank canvas (white background)
        canvas = np.ones((int(self.height * 10), int(self.width * 10), 3), dtype=np.uint8) * 255

        # Define colors
        BLUE = (255, 144, 30)  # Cooperative robots (BGR format)
        ORANGE = (0, 165, 255)  # Non-cooperative robots
        RED = (0, 0, 255)  # Obstacles
        BLACK = (0, 0, 0)  # Lines, text
        GRAY = (200, 200, 200)  # Background flow field
        GREEN = (0, 255, 0)  # Goals
        YELLOW = (0, 255, 255)  # Collision state

        # Draw background flow field (velocity vectors)
        step = 3  # Grid spacing for flow field
        for x in range(0, int(self.width), step):
            for y in range(0, int(self.height), step):
                pos = (int(x * 10), int(y * 10))
                v = self.get_velocity(x, y)
                if np.linalg.norm(v) > 0.01:
                    v = v / np.linalg.norm(v) * 5  # Normalize and scale
                    end_pos = (int((x + v[0]) * 10), int((y + v[1]) * 10))
                    cv2.arrowedLine(canvas, pos, end_pos, GRAY, 1, tipLength=0.2)

        # Draw vortex cores
        for core in self.cores:
            pos = (int(core.x * 10), int(core.y * 10))
            radius = int(self.r * 10)
            color = (255, 0, 0) if core.clockwise else (0, 0, 255)  # Red for clockwise, blue for counter-clockwise
            cv2.circle(canvas, pos, radius, color, 2)
            # Draw circulation strength indicator
            # strength_radius = int(min(5, abs(core.Gamma) / 10) * 10)
            # cv2.circle(canvas, pos, strength_radius, color, -1, lineType=cv2.LINE_AA)
            # Draw rotation indicator
            if core.clockwise:
                cv2.ellipse(canvas, pos, (radius + 5, radius + 5), 0, 45, 315, color, 2, lineType=cv2.LINE_AA)
            else:
                cv2.ellipse(canvas, pos, (radius + 5, radius + 5), 0, 225, 135, color, 2, lineType=cv2.LINE_AA)

        # Draw obstacles
        for obs in self.obstacles:
            pos = (int(obs.x * 10), int(obs.y * 10))
            radius = int(obs.r * 10)
            cv2.circle(canvas, pos, radius, RED, -1, lineType=cv2.LINE_AA)

        # Draw robots and their goals
        for rob in self.robots:
            # Draw path to goal
            start_pos = (int(rob.x * 10), int(rob.y * 10))
            goal_pos = (int(rob.goal[0] * 10), int(rob.goal[1] * 10))
            cv2.line(canvas, start_pos, goal_pos, BLACK, 1, lineType=cv2.LINE_AA)

            # Draw goal
            cv2.circle(canvas, goal_pos, 5, GREEN, -1, lineType=cv2.LINE_AA)

            # Draw robot
            color = YELLOW if rob.collision else BLUE if rob.cooperative else ORANGE
            cv2.circle(canvas, start_pos, int(rob.r * 10), color, -1, lineType=cv2.LINE_AA)

            # Draw direction indicator
            direction_x = rob.x + rob.r * np.cos(rob.theta)
            direction_y = rob.y + rob.r * np.sin(rob.theta)
            direction_pos = (int(direction_x * 10), int(direction_y * 10))
            cv2.line(canvas, start_pos, direction_pos, BLACK, 2, lineType=cv2.LINE_AA)

            # # Draw perception field
            # if hasattr(rob, 'perception') and hasattr(rob.perception, 'range') and hasattr(rob.perception, 'angle'):
            #     perception_range = rob.perception.range
            #     perception_angle = rob.perception.angle
            #
            #     # Create arc for perception field
            #     start_angle = rob.theta - perception_angle / 2
            #     end_angle = rob.theta + perception_angle / 2
            #
            #     # Convert to degrees for cv2.ellipse
            #     start_angle_deg = int(np.degrees(start_angle))
            #     end_angle_deg = int(np.degrees(end_angle))
            #
            #     cv2.ellipse(canvas, start_pos,
            #                 (int(perception_range * 10), int(perception_range * 10)),
            #                 0, start_angle_deg, end_angle_deg,
            #                 (100, 100, 100), 1, lineType=cv2.LINE_AA)

            # Draw trajectory if available
            if hasattr(rob, 'trajectory') and len(rob.trajectory) > 1:
                traj_points = [(int(p[0] * 10), int(p[1] * 10)) for p in rob.trajectory]
                for i in range(1, len(traj_points)):
                    cv2.line(canvas, traj_points[i - 1], traj_points[i], color, 1, lineType=cv2.LINE_AA)

        # Add environment info
        info_text = [
            f"Steps: {self.episode_timesteps}",
            f"Cores: {len(self.cores)}",
            f"Obstacles: {len(self.obstacles)}",
            f"Robots: {len(self.robots)}"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(canvas, text, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

        # Display if mode is 'human'
        if mode == 'human':
            cv2.imshow('Marine Environment', canvas)
            cv2.waitKey(1)
            return None

        # Return the canvas if mode is 'rgb_array'
        elif mode == 'rgb_array':
            return canvas

        # Close any open windows when done
    def close(self):
        cv2.destroyAllWindows()