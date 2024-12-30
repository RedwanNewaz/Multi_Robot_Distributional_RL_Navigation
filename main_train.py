import hydra
from omegaconf import DictConfig, OmegaConf
import marinenav_env.envs.marinenav_env as marinenav_env
from policy.agent import Agent
from policy.trainer import Trainer
from multiprocessing import Pool
import os
from datetime import datetime

def run_trial(cfg):
    train_env = marinenav_env.MarineNavEnv2(cfg.seed, schedule=dict(cfg.exp.train))
    eval_env = marinenav_env.MarineNavEnv2(seed=253)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if cfg.agent.use_rl:
        agent = Agent(cooperative=cfg.agent.cooperative, device=cfg.device, use_iqn=cfg.agent.use_iqn,
                      seed=cfg.seed + 100)
    else:
        raise NotImplementedError("The agent is not implemented")

    exp_dir = os.path.join(os.getcwd(), cfg.agent.save_dir,
                           f"training_{timestamp}",
                           f"seed_{cfg.seed}")
    os.makedirs(exp_dir, exist_ok=True)

    trainer = Trainer(train_env=train_env,
                      eval_env=eval_env,
                      eval_schedule=dict(cfg.exp.eval),
                      cooperative_agent=agent,
                      non_cooperative_agent=None)

    trainer.save_eval_config(exp_dir)
    trainer.learn(total_timesteps=cfg.exp.total_timesteps,
                  eval_freq=cfg.exp.eval_freq,
                  eval_log_path=exp_dir,
                  verbose=cfg.verbose)

@hydra.main(version_base=None, config_path="config", config_name="main_train")
def main(cfg: DictConfig) -> None:
    if cfg.num_procs == 1:
        run_trial(cfg)
    else:
        configs = [cfg for _ in range(cfg.num_procs)]
        with Pool(processes=cfg.num_procs) as pool:
            pool.map(run_trial, configs)

if __name__ == "__main__":
    main()
