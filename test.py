from marllib import marl
from marine import MarineNavEnv  # Make sure this import path is correct
from marllib.envs.base_env import ENV_REGISTRY
import os
# Properly register the environment
ENV_REGISTRY["marine_env"] = MarineNavEnv  # Changed key to match error message
# Also register using the alternate method
marl.register_env("marine_env", MarineNavEnv)  # Uncommented this line


if __name__ == '__main__':
    config = os.path.join(os.getcwd(), "config/marine_env.yaml")
    env = marl.make_env(environment_name="marine_env", map_name="MarineNav", abs_path=config)  # Changed environment_name

    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="test")


    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})

    # rendering
    mappo.render(env, model,
                 stop={'timesteps_total': 5000000},
                 restore_path={'params_path': "exp_results/mappo_mlp_MarineNav/MAPPOTrainer_marine_env_MarineNav_2bbc2_00000_0_2025-03-30_15-04-21/params_test.json",  # experiment configuration
                               'model_path': "exp_results/mappo_mlp_MarineNav/MAPPOTrainer_marine_env_MarineNav_2bbc2_00000_0_2025-03-30_15-04-21/checkpoint_000290/checkpoint-290",
                               'render': True
                               },  # checkpoint path
                 num_workers=10,
                 num_gpus=0,
                 num_gpus_per_worker=0,
                 local_mode=False,
                 # share_policy="all",
                 checkpoint_end=True)