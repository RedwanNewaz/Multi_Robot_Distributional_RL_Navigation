import hydra
from omegaconf import DictConfig
import marinenav_env.envs.marinenav_env as marinenav_env
from policy.agent import Agent
from run_experiments import evaluation, exp_setup
import numpy as np

import sys
import os
from datetime import datetime
from policy.trainer import Trainer
from multiprocessing import Pool


def run_trial(cfg):
    train_env = marinenav_env.MarineNavEnv2(cfg.seed, schedule=dict(cfg.exp.train))
    eval_env = marinenav_env.MarineNavEnv2(seed=253)
    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")

    agent = None
    if cfg.agent.use_rl:
        agent = Agent(cooperative=cfg.agent.cooperative, device=cfg.device, use_iqn=cfg.agent.use_iqn,
                      seed=cfg.seed + 100)
    else:
        raise NotImplementedError("The agent is not implemented")

    exp_dir = os.path.join(os.getcwd(), cfg.agent.save_dir,
                           "training_" + timestamp,
                           "seed_" + str(cfg.seed))
    os.makedirs(exp_dir, exist_ok=True)

    trainer = Trainer(train_env=train_env,
                      eval_env=eval_env,
                      eval_schedule=dict(cfg.exp.eval),
                      cooperative_agent=agent,
                      non_cooperative_agent=None,
                      )
    trainer.save_eval_config(exp_dir)
    trainer.learn(total_timesteps=cfg.exp.total_timesteps,
                  eval_freq=cfg.exp.eval_freq,
                  eval_log_path=exp_dir,
                  verbose=False
                  )
@hydra.main(version_base=None, config_path="config", config_name="main_train")
def my_app(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # run_trial(cfg)

    with Pool(processes=cfg.num_procs) as pool:
        pool.apply_async(run_trial, (cfg, ))

        pool.close()
        pool.join()

        # convert the cfg.exp to dictionary



if __name__ == "__main__":
    my_app()

