import hydra
from omegaconf import DictConfig
import marinenav_env.envs.marinenav_env as marinenav_env
from MarineEnv3 import MarineEnv
from policy.agent import Agent
from run_experiments import evaluation, exp_setup
import numpy as np

import sys
sys.path.insert(0,"./thirdparty")
import RVO
from thirdparty import APF
from random import choice
def run_experiment(cfg:DictConfig, env:marinenav_env, agent: Agent, eval_schedule: dict):
    idx = choice(range(len(eval_schedule["num_cooperative"])))
    env.num_cooperative = eval_schedule["num_cooperative"][idx]
    env.num_non_cooperative = eval_schedule["num_non_cooperative"][idx]
    env.num_cores = eval_schedule["num_cores"][idx]
    env.num_obs = eval_schedule["num_obstacles"][idx]
    env.min_start_goal_dis = eval_schedule["min_start_goal_dis"][idx]
    state,_,_ = env.reset()
    end_episode = False
    length = 0
    rewards = [0.0] * len(env.robots)
    while not end_episode:
        action = []
        for i, rob in enumerate(env.robots):
            if rob.deactivated:
                action.append(None)
                continue
            assert rob.cooperative, "Every robot must be cooperative!"
            # a, _, _ = agent.act(state[i])
            if cfg.agent.use_rl:
                if cfg.agent.use_iqn:
                    a,_,_ = agent.act(state[i])
                else:
                    a,_ = agent.act_dqn(state[i])
            else:
                a = agent.act(state[i])
            action.append(a)
        # execute actions in the training environment

        state, reward, done, info = env.step(action)
        env.render()
        for i,rob in enumerate(env.robots):
            if rob.deactivated:
                continue
            rewards[i] += agent.GAMMA ** length * reward[i]
            if rob.collision or rob.reach_goal:
                rob.deactivated = True

        end_episode = (length >= 360) or env.check_all_deactivated()

        length += 1

    success = env.check_all_reach_goal()
    print(f"[{cfg.agent.name}]: success = {success} | average reward = {np.average(rewards):.3f}")
@hydra.main(version_base=None, config_path="config", config_name="main")
def my_app(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # env = marinenav_env.MarineNavEnv2(cfg.seed)
    env = MarineEnv(cfg.seed)
    agent = None
    if cfg.agent.use_rl:
        agent = Agent(cooperative=True, device=cfg.device, use_iqn=cfg.agent.use_iqn)
        agent.load_model(cfg.agent.save_dir, cfg.agent.type, cfg.device)
    elif cfg.agent.name == "APF":
        agent = APF.APF_agent(env.robots[0].a, env.robots[0].w)
    elif cfg.agent.name == "RVO":
        agent = RVO.RVO_agent(env.robots[0].a,env.robots[0].w, env.robots[0].config.max_speed)
    else:
        raise NotImplementedError("The agent is not implemented")

    # convert the cfg.exp to dictionary
    schedules = dict(cfg.exp)
    run_experiment(cfg, env, agent, schedules)

if __name__ == "__main__":
    my_app()

