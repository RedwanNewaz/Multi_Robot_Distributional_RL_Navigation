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

    # start learning
    # mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000},
    #           local_mode=True, num_gpus=1, num_workers=2, share_policy='all', checkpoint_freq=50)

    # mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000},
    #           local_mode=False, num_gpus=1, num_workers=2, share_policy='all', checkpoint_freq=50)

    # mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000},
    #           local_mode=False, num_gpus=0, num_workers=0, share_policy='all', checkpoint_freq=50)

    mappo.fit(env, model, local_mode=False, stop={'timesteps_total': 10000}, checkpoint_freq=10)

