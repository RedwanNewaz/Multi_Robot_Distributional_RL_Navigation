#!/home/redwan/anaconda3/envs/Multi_Robot_Distributional_RL_Navigation/bin/python
from marllib import marl
from marine import MarineNavEnv  # Make sure this import path is correct
from marllib.envs.base_env import ENV_REGISTRY
import os
# Properly register the environment
ENV_REGISTRY["marine_env"] = MarineNavEnv  # Changed key to match error message
# Also register using the alternate method
marl.register_env("marine_env", MarineNavEnv)  # Uncommented this line

if __name__ == '__main__':
    # initialize env
    config = os.path.join(os.getcwd(), "config/marine_env.yaml")
    env = marl.make_env(environment_name="marine_env", map_name="MarineNav", abs_path=config)  # Changed environment_name

    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="test")

    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})

    mappo.fit(env, model, local_mode=False, stop={'timesteps_total': 6000000}, checkpoint_freq=10,
              config={"log_level": "INFO", "tensorboard_log": "./marl_tensorboard"}
              )

