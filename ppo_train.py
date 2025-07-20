import numpy as np
from stable_baselines3.ppo import PPO
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.util.util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium.envs.registration import register
import subprocess
import threading
from reward_net import feature_extractor, Bayesian_reward_net
from pretrain_ppo import pretrain
from stable_baselines3.common.callbacks import BaseCallback

SEED = 43


def start_external_program():
    subprocess.Popen(['D:\\zhechen\\baseline\\Easy\\Blocks.exe', '-windowed', '-ResX=1280', '-ResY=720'])


class DistanceCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_episode_distances = []
        self.episode_mean_distances = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "distance" in info:
                self.current_episode_distances.append(info["distance"])
            if "episode" in info:
                # episode 结束
                if self.current_episode_distances:
                    mean_dist = sum(self.current_episode_distances) / len(self.current_episode_distances)
                    self.episode_mean_distances.append(mean_dist)
                    self.logger.record("rollout/ep_mean_distance", mean_dist)
                    self.current_episode_distances = []
        return True


def set_seed(random_seed):
    set_random_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    thread = threading.Thread(target=start_external_program)
    thread.start()
    SEED = 43
    set_seed(SEED)

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 250000,
        "env_id": "airsim",
    }

    register(
        id='AirSim/AirSimEnv',
        entry_point='airsim_test_env:AirSimEnv_v3',  # 指向自定义环境的模块和类
        max_episode_steps=150,
    )

    env = make_vec_env(
        'AirSim/AirSimEnv',
        rng=np.random.default_rng(SEED),
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    )

    env = VecNormalize(env, norm_reward=True, norm_obs=False, clip_reward=3)
    learner = PPO.load("./model/airl/policy/ppo_policy_bairl15.zip", env=env)

    # policy_kwargs = dict(
    #     features_extractor_class=feature_extractor,
    #     features_extractor_kwargs=dict(features_dim=96),
    #     share_features_extractor=False,
    #     log_std_init=-1.5
    # )
    #
    # learner = PPO(
    #     env=env,
    #     policy=ActorCriticPolicy,
    #     gae_lambda=0.98,
    #     ent_coef=0.0001,
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=1e-4,
    #     n_steps=2048,
    #     batch_size=512,
    #     gamma=0.995,
    #     clip_range=0.2,
    #     clip_range_vf=0.1,
    #     normalize_advantage=True,
    #     vf_coef=0.2,
    #     max_grad_norm=0.9,
    #     n_epochs=10,
    #     seed=SEED,
    #     device="cuda",
    #     verbose=2,
    #     tensorboard_log="./test_logs/"
    # )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./model/airl/test',
        name_prefix='ppo_model'
    )
    # pretrain(learner, env, 20, figure=False)
    distance_callback = DistanceCallback()
    learner.learn(total_timesteps=260000, callback=[checkpoint_callback, distance_callback])
    learner.save('test_train.zip')
