import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import heapq
from functools import lru_cache


class ObservationType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    SELF = "self"


@dataclass
class PerceptionConfig:
    range: float = 15.0
    angle: float = 2 * np.pi
    max_obs_num: int = 5
    max_obj_num: int = 5


@dataclass
class RobotConfig:
    length: float = 1.0
    width: float = 0.5
    dt: float = 0.05
    N: int = 10
    goal_dist: float = 2.0
    obs_dist: float = 5.0
    max_speed: float = 2.0
    init_theta: float = 0.0
    init_speed: float = 0.0


class Perception:
    def __init__(self, cooperative: bool = False, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        self.observation: Dict = {
            ObservationType.SELF.value: [],
            ObservationType.STATIC.value: []
        }
        if cooperative:
            self.observation[ObservationType.DYNAMIC.value] = []

        self.observed_obs: List[int] = []
        self.observed_objs: List[int] = []

    @property
    def range(self) -> float:
        return self.config.range

    @property
    def angle(self) -> float:
        return self.config.angle


class Robot:
    def __init__(self, cooperative: bool = False, config: Optional[RobotConfig] = None):
        self.config = config or RobotConfig()
        self.cooperative = cooperative
        self.perception = Perception(cooperative)

        # Computed properties
        self.r = 0.8
        self.detect_r = 0.5 * np.sqrt(self.config.length ** 2 + self.config.width ** 2)
        self.a = np.array([-0.4, 0.0, 0.4])
        self.w = np.array([-np.pi / 6, 0.0, np.pi / 6])
        self.k = np.max(self.a) / self.config.max_speed
        self.actions = self.compute_actions()

        # State variables
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.theta: Optional[float] = None
        self.speed: Optional[float] = None
        self.velocity: Optional[np.ndarray] = None
        self.start: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None

        # Status flags
        self.collision = False
        self.reach_goal = False
        self.deactivated = False

        # History
        self.action_history: List = []
        self.trajectory: List = []

    def compute_actions(self) -> List[Tuple[float, float]]:
        return [(acc, ang_v) for acc in self.a for ang_v in self.w]

    def compute_k(self) -> None:
        self.k = np.max(self.a) / self.config.max_speed

    @property
    def actions_dimension(self) -> int:
        return len(self.actions)

    @lru_cache(maxsize=128)
    def compute_dist_reward_scale(self) -> float:
        return 1 / (self.config.max_speed * self.config.N * self.config.dt)

    @lru_cache(maxsize=128)
    def compute_penalty_matrix(self) -> np.matrix:
        scale_a = 1 / (np.max(self.a) ** 2)
        scale_w = 1 / (np.max(self.w) ** 2)
        return -0.5 * np.matrix([[scale_a, 0.0], [0.0, scale_w]])

    def compute_action_energy_cost(self, action: int) -> float:
        a, w = self.actions[action]
        return np.abs(a / np.max(self.a)) + np.abs(w / np.max(self.w))

    def dist_to_goal(self) -> float:
        return np.linalg.norm(self.goal - np.array([self.x, self.y]))

    def check_reach_goal(self) -> bool:
        self.reach_goal = self.dist_to_goal() <= self.config.goal_dist
        return self.reach_goal

    def reset_state(self, current_velocity: np.ndarray = np.zeros(2)) -> None:
        self.action_history.clear()
        self.trajectory.clear()
        self.x, self.y = self.start
        self.theta = self.config.init_theta
        self.speed = self.config.init_speed
        self.update_velocity(current_velocity)
        self._record_state()

    def _record_state(self) -> None:
        self.trajectory.append([self.x, self.y, self.theta, self.speed, *self.velocity])

    def get_robot_transform(self) -> Tuple[np.matrix, np.matrix]:
        cos_theta, sin_theta = np.cos(self.theta), np.sin(self.theta)
        R_wr = np.matrix([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        t_wr = np.matrix([[self.x], [self.y]])
        return R_wr, t_wr

    def update_velocity(self, current_velocity: np.ndarray = np.zeros(2)) -> None:
        steer_velocity = self.speed * np.array([np.cos(self.theta), np.sin(self.theta)])
        self.velocity = steer_velocity + current_velocity

    def update_state(self, action: int, current_velocity: np.ndarray) -> None:
        self.update_velocity(current_velocity)
        self._update_position()
        self._update_speed_and_heading(action)
        self._record_state()

    def _update_position(self) -> None:
        dis = self.velocity * self.config.dt
        self.x += dis[0]
        self.y += dis[1]

    def _update_speed_and_heading(self, action: int) -> None:
        a, w = self.actions[action]
        self.speed += (a - self.k * self.speed) * self.config.dt
        self.speed = np.clip(self.speed, 0.0, self.config.max_speed)
        self.theta = (self.theta + w * self.config.dt) % (2 * np.pi)

    def check_collision(self, obj_x: float, obj_y: float, obj_r: float) -> None:
        if self.compute_distance(obj_x, obj_y, obj_r) <= 0.0:
            self.collision = True

    def compute_distance(self, x: float, y: float, r: float, in_robot_frame: bool = False) -> float:
        if in_robot_frame:
            return np.sqrt(x ** 2 + y ** 2) - r - self.r
        return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2) - r - self.r

    def check_detection(self, obj_x: float, obj_y: float, obj_r: float) -> bool:
        proj_pos = self.project_to_robot_frame(np.array([obj_x, obj_y]), False)
        if np.linalg.norm(proj_pos) > self.perception.range + obj_r:
            return False
        angle = np.arctan2(proj_pos[1], proj_pos[0])
        return -0.5 * self.perception.angle <= angle <= 0.5 * self.perception.angle

    def project_to_robot_frame(self, x: np.ndarray, is_vector: bool = True) -> np.ndarray:
        x_r = np.reshape(x, (2, 1))
        R_wr, t_wr = self.get_robot_transform()
        R_rw = np.transpose(R_wr)
        t_rw = -R_rw * t_wr
        x_r = R_rw * x_r + (0 if is_vector else t_rw)
        x_r.resize((2,))
        return np.array(x_r)

    def perception_output(self, obstacles: List, robots: List, in_robot_frame: bool = True) -> Tuple:
        if self.deactivated:
            return None, self.collision, self.reach_goal

        self._clear_perception()
        self._update_self_observation(in_robot_frame)
        self._process_static_obstacles(obstacles, in_robot_frame)

        if self.cooperative:
            self._process_dynamic_objects(robots, in_robot_frame)

        return self._prepare_output()

    def _clear_perception(self) -> None:
        self.perception.observation[ObservationType.STATIC.value].clear()
        if self.cooperative:
            self.perception.observation[ObservationType.DYNAMIC.value].clear()
        self.perception.observed_obs.clear()
        if self.cooperative:
            self.perception.observed_objs.clear()

    def _update_self_observation(self, in_robot_frame: bool) -> None:
        if in_robot_frame:
            abs_velocity_r = self.project_to_robot_frame(self.velocity)
            goal_r = self.project_to_robot_frame(self.goal, False)
            self.perception.observation[ObservationType.SELF.value] = list(np.concatenate((goal_r, abs_velocity_r)))
        else:
            self.perception.observation[ObservationType.SELF.value] = [
                self.x, self.y, self.theta, self.speed, *self.velocity, *self.goal
            ]

    def _process_static_obstacles(self, obstacles: List, in_robot_frame: bool) -> None:
        for i, obs in enumerate(obstacles):
            if not self.check_detection(obs.x, obs.y, obs.r):
                continue

            self.perception.observed_obs.append(i)
            if not self.collision:
                self.check_collision(obs.x, obs.y, obs.r)

            if in_robot_frame:
                pos_r = self.project_to_robot_frame(np.array([obs.x, obs.y]), False)
                self.perception.observation[ObservationType.STATIC.value].append([pos_r[0], pos_r[1], obs.r])
            else:
                self.perception.observation[ObservationType.STATIC.value].append([obs.x, obs.y, obs.r])

    def _process_dynamic_objects(self, robots: List, in_robot_frame: bool) -> None:
        for j, robot in enumerate(robots):
            if robot is self or robot.deactivated:
                continue
            if not self.check_detection(robot.x, robot.y, robot.detect_r):
                continue

            self.perception.observed_objs.append(j)
            if not self.collision:
                self.check_collision(robot.x, robot.y, robot.r)

            if in_robot_frame:
                pos_r = self.project_to_robot_frame(np.array([robot.x, robot.y]), False)
                v_r = self.project_to_robot_frame(robot.velocity)
                self.perception.observation[ObservationType.DYNAMIC.value].append(list(np.concatenate((pos_r, v_r))))
            else:
                self.perception.observation[ObservationType.DYNAMIC.value].append(
                    [robot.x, robot.y, robot.velocity[0], robot.velocity[1]]
                )

    def _prepare_output(self) -> Tuple:
        self.check_reach_goal()
        self_state = self.perception.observation[ObservationType.SELF.value].copy()

        static_observations = heapq.nsmallest(
            self.perception.config.max_obs_num,
            self.perception.observation[ObservationType.STATIC.value],
            key=lambda obs: self.compute_distance(obs[0], obs[1], obs[2], True)
        )

        static_states = self._pad_observations(
            static_observations,
            self.perception.config.max_obs_num,
            3
        )

        if self.cooperative:
            dynamic_observations = heapq.nsmallest(
                self.perception.config.max_obj_num,
                self.perception.observation[ObservationType.DYNAMIC.value],
                key=lambda obj: self.compute_distance(obj[0], obj[1], self.r, True)
            )
        else:
            dynamic_observations = []

        dynamic_states = self._pad_observations(
            dynamic_observations,
            self.perception.config.max_obj_num,
            4
        )

        return self_state + static_states + dynamic_states, self.collision, self.reach_goal

    @staticmethod
    def _pad_observations(observations: List, max_size: int, feature_size: int) -> List:
        flattened = [item for obs in observations for item in obs]
        padding = [0.0] * (max_size * feature_size - len(flattened))
        return flattened + padding