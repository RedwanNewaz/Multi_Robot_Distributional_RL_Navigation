import json
import numpy as np
import os
import copy
import time
from tqdm import tqdm


class Trainer:
    def __init__(self, train_env, eval_env, eval_schedule, non_cooperative_agent=None, cooperative_agent=None,
                 UPDATE_EVERY=4, learning_starts=2000, target_update_interval=10000,
                 exploration_fraction=0.25, initial_eps=0.6, final_eps=0.05):
        self.train_env = train_env
        self.eval_env = eval_env
        self.cooperative_agent = cooperative_agent
        self.noncooperative_agent = non_cooperative_agent
        self.eval_config = []
        self.create_eval_configs(eval_schedule)
        self.UPDATE_EVERY = UPDATE_EVERY
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.current_timestep = 0
        self.eval_data = {
            'timesteps': [], 'actions': [], 'trajectories': [], 'rewards': [],
            'successes': [], 'times': [], 'energies': [], 'obs': [], 'objs': []
        }

    def create_eval_configs(self, eval_schedule):
        for i, num_episode in enumerate(eval_schedule["num_episodes"]):
            for _ in range(num_episode):
                self.eval_env.num_cooperative = eval_schedule["num_cooperative"][i]
                self.eval_env.num_non_cooperative = eval_schedule["num_non_cooperative"][i]
                self.eval_env.num_cores = eval_schedule["num_cores"][i]
                self.eval_env.num_obs = eval_schedule["num_obstacles"][i]
                self.eval_env.min_start_goal_dis = eval_schedule["min_start_goal_dis"][i]
                self.eval_env.reset()
                self.eval_config.append(self.eval_env.episode_data())

    def save_eval_config(self, directory):
        with open(os.path.join(directory, "eval_configs.json"), "w+") as f:
            json.dump(self.eval_config, f)

    def learn(self, total_timesteps, eval_freq, eval_log_path, verbose=True):
        states, _, _ = self.train_env.reset()
        ep_rewards = np.zeros(len(self.train_env.robots))
        ep_deactivated_t = [-1] * len(self.train_env.robots)
        ep_length = 0
        ep_num = 0

        pbar = tqdm(total=total_timesteps, desc="Training Progress")
        while self.current_timestep <= total_timesteps:
            eps = self.linear_eps(total_timesteps)
            actions = self.get_actions(states, eps)
            next_states, rewards, dones, infos = self.train_env.step(actions)
            self.save_experience(states, actions, rewards, next_states, dones)
            ep_rewards, ep_deactivated_t = self.update_episode_data(ep_rewards, ep_deactivated_t, rewards, ep_length)

            end_episode = (ep_length >= 1000) or self.train_env.check_all_deactivated()

            if self.current_timestep >= self.learning_starts:
                self.learn_and_update_models()
                if self.current_timestep % eval_freq == 0:
                    self.evaluation()
                    self.save_evaluation(eval_log_path)
                    self.save_latest_models(eval_log_path)

            if end_episode:
                if verbose:
                    self.print_episode_info(ep_length, ep_num, eps, infos, ep_rewards, ep_deactivated_t)
                states, _, _ = self.train_env.reset()
                ep_rewards = np.zeros(len(self.train_env.robots))
                ep_deactivated_t = [-1] * len(self.train_env.robots)
                ep_length = 0
                ep_num += 1
            else:
                states = next_states
                ep_length += 1

            self.current_timestep += 1
            pbar.update(1)

        pbar.close()

    def get_actions(self, states, eps):
        actions = []
        for i, rob in enumerate(self.train_env.robots):
            if rob.deactivated:
                actions.append(None)
                continue
            agent = self.cooperative_agent if rob.cooperative else self.noncooperative_agent
            if agent.use_iqn:
                action, _, _ = agent.act(states[i], eps)
            else:
                action, _ = agent.act_dqn(states[i], eps)
            actions.append(action)
        return actions

    def save_experience(self, states, actions, rewards, next_states, dones):
        for i, rob in enumerate(self.train_env.robots):
            if rob.deactivated:
                continue
            agent = self.cooperative_agent if rob.cooperative else self.noncooperative_agent
            if agent.training:
                agent.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))

    def update_episode_data(self, ep_rewards, ep_deactivated_t, rewards, ep_length):
        for i, rob in enumerate(self.train_env.robots):
            if rob.deactivated:
                continue
            agent = self.cooperative_agent if rob.cooperative else self.noncooperative_agent
            ep_rewards[i] += agent.GAMMA ** ep_length * rewards[i]
            if rob.collision or rob.reach_goal:
                rob.deactivated = True
                ep_deactivated_t[i] = ep_length
        return ep_rewards, ep_deactivated_t

    def learn_and_update_models(self):
        for agent in [self.cooperative_agent, self.noncooperative_agent]:
            if agent is None or not agent.training:
                continue
            if self.current_timestep % self.UPDATE_EVERY == 0:
                if agent.memory.size() > agent.BATCH_SIZE:
                    agent.train()
            if self.current_timestep % self.target_update_interval == 0:
                agent.soft_update()

    def linear_eps(self, total_timesteps):
        progress = self.current_timestep / total_timesteps
        if progress < self.exploration_fraction:
            r = progress / self.exploration_fraction
            return self.initial_eps + r * (self.final_eps - self.initial_eps)
        else:
            return self.final_eps

    def evaluation(self):
        for idx, config in enumerate(self.eval_config):
            print(f"Evaluating episode {idx}")
            state, _, _ = self.eval_env.reset_with_eval_config(config)
            obs = [[copy.deepcopy(rob.perception.observed_obs)] for rob in self.eval_env.robots]
            objs = [[copy.deepcopy(rob.perception.observed_objs)] for rob in self.eval_env.robots]
            rob_num = len(self.eval_env.robots)
            rewards = [0.0] * rob_num
            times = [0.0] * rob_num
            energies = [0.0] * rob_num
            length = 0

            while True:
                action = self.get_actions(state, 0)  # Use epsilon=0 for evaluation
                state, reward, done, info = self.eval_env.step(action)
                self.update_eval_data(action, reward, length, obs, objs)

                if self.eval_env.check_any_collision() or self.eval_env.check_all_deactivated() or length >= 1000:
                    break
                length += 1

            self.save_eval_episode_data(rewards, times, energies)

        self.print_eval_summary()

    def update_eval_data(self, action, reward, length, obs, objs):
        for i, rob in enumerate(self.eval_env.robots):
            if rob.deactivated:
                continue
            agent = self.cooperative_agent if rob.cooperative else self.noncooperative_agent
            rewards[i] += agent.GAMMA ** length * reward[i]
            times[i] += rob.dt * rob.N
            energies[i] += rob.compute_action_energy_cost(action[i])
            obs[i].append(copy.deepcopy(rob.perception.observed_obs))
            objs[i].append(copy.deepcopy(rob.perception.observed_objs))

    def save_eval_episode_data(self, rewards, times, energies):
        actions = [rob.action_history for rob in self.eval_env.robots]
        trajectories = [rob.trajectory for rob in self.eval_env.robots]
        success = self.eval_env.check_all_reach_goal()

        self.eval_data['actions'].append(actions)
        self.eval_data['trajectories'].append(trajectories)
        self.eval_data['rewards'].append(np.mean(rewards))
        self.eval_data['successes'].append(success)
        self.eval_data['times'].append(np.mean(times))
        self.eval_data['energies'].append(np.mean(energies))
        self.eval_data['obs'].append(obs)
        self.eval_data['objs'].append(objs)

    def print_eval_summary(self):
        avg_r = np.mean(self.eval_data['rewards'])
        success_rate = np.mean(self.eval_data['successes'])
        successful_episodes = np.where(np.array(self.eval_data['successes']) == 1)[0]
        avg_t = np.mean(np.array(self.eval_data['times'])[successful_episodes]) if len(
            successful_episodes) > 0 else None
        avg_e = np.mean(np.array(self.eval_data['energies'])[successful_episodes]) if len(
            successful_episodes) > 0 else None

        print(f"++++++++ Evaluation Summary ++++++++")
        print(f"Avg cumulative reward: {avg_r:.2f}")
        print(f"Success rate: {success_rate:.2f}")
        if avg_t is not None:
            print(f"Avg time: {avg_t:.2f}")
        if avg_e is not None:
            print(f"Avg energy: {avg_e:.2f}")
        print(f"++++++++ Evaluation Summary ++++++++\n")

    def save_evaluation(self, eval_log_path):
        filename = "evaluations.npz"
        np.savez(os.path.join(eval_log_path, filename), **self.eval_data)

    def save_latest_models(self, eval_log_path):
        for agent in [self.cooperative_agent, self.noncooperative_agent]:
            if agent is not None and agent.training:
                agent.save_latest_model(eval_log_path)

    def print_episode_info(self, ep_length, ep_num, eps, infos, ep_rewards, ep_deactivated_t):
        print("======== Episode Info ========")
        print(f"current ep_length: {ep_length}")
        print(f"current ep_num: {ep_num}")
        print(f"current exploration rate: {eps}")
        print(f"current timesteps: {self.current_timestep}")
        print("======== Episode Info ========\n")
        print("======== Robots Info ========")
        for i, rob in enumerate(self.train_env.robots):
            info = infos[i]["state"]
            if info in ["deactivated after collision", "deactivated after reaching goal"]:
                print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info} at step {ep_deactivated_t[i]}")
            else:
                print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info}")
        print("======== Robots Info ========\n")
