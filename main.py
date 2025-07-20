from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.sac import policies as sac_policies
from imitation.rewards.reward_nets import BasicShapedRewardNet
from stable_baselines3.ppo import MlpPolicy
import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import BaseCallback
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym
import torch
from pretrain_ppo import pretrain
from imitation.data import serialize
import os
from gymnasium.envs.registration import register
import subprocess
import threading
import matplotlib
import matplotlib.pyplot as plt
from reward_net import feature_extractor, Bayesian_reward_net

from airl import custom_AIRL

matplotlib.use("Agg")
STOCHASTIC_POLICIES = (sac_policies.SACPolicy, policies.ActorCriticPolicy)


def start_external_program():
    subprocess.Popen(['D:\\zhechen\\baseline\\Easy\\Blocks.exe', '-windowed', '-ResX=1280', '-ResY=720'])


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
    # env = make_vec_env(
    #     "MountainCarContinuous-v0",
    #     rng=np.random.default_rng(SEED),
    #     n_envs=1,
    #     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    # )
    #
    # env = VecNormalize(env, norm_obs=True, norm_reward=False)
    # expert = load_policy(
    #     "ppo-huggingface",
    #     organization="sb3",
    #     env_name="MountainCarContinuous-v0",
    #     venv=env,
    # )
    # env = make_vec_env(
    #     "seals:seals/CartPole-v0",
    #     rng=np.random.default_rng(SEED),
    #     n_envs=8,
    #     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    # )
    # expert = load_policy(
    #     "ppo-huggingface",
    #     organization="HumanCompatibleAI",
    #     env_name="seals-CartPole-v0",
    #     venv=env,
    # )
    # rollouts = rollout.rollout(
    #     expert,
    #     env,
    #     rollout.make_sample_until(min_episodes=60),
    #     rng=np.random.default_rng(SEED),
    # )
    # 3. 收集专家示范
    # rollouts = rollout.rollout(
    #     expert,
    #     env,
    #     rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    #     rng=np.random.default_rng(SEED),
    # )

    # policy_kwags = {
    #     "log_std_init": 0
    # }
    #
    # learner = PPO(
    #     env=env,
    #     policy=MlpPolicy,
    #     batch_size=256,
    #     ent_coef=0.004,
    #     learning_rate=0.0005,
    #     gamma=0.95,
    #     clip_range=0.1,
    #     vf_coef=0.1,
    #     n_epochs=10,
    #     seed=SEED,
    #     policy_kwargs=policy_kwags,
    #     max_grad_norm=5
    # )

    # reward_net = BasicShapedRewardNet(
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     normalize_input_layer=RunningNorm,
    # )
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 250000,
        "env_id": "airsim",
    }

    register(
        id='AirSim/AirSimEnv',
        entry_point='airsim_env:AirSimEnv_v3',  # 指向自定义环境的模块和类
        max_episode_steps=150,
    )

    env = make_vec_env(
        'AirSim/AirSimEnv',
        rng=np.random.default_rng(SEED),
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    )
    env = VecNormalize(env, gamma=0.999, norm_reward=True, norm_obs=False)
    b_reward_net = Bayesian_reward_net(observation_space=env.observation_space,
                                       action_space=env.action_space,
                                       use_state=True,
                                       use_action=False,
                                       use_next_state=True,
                                       use_done=True,
                                       feature_extractor=feature_extractor
                                       )
    file_path = os.path.join("model", "expert_data_all.pkl")
    if os.path.exists(file_path):
        print("Loading expert rollouts...")
        rollouts = serialize.load(file_path)
        print("success loading expert rollouts...")
    else:
        raise ValueError("without expert data")

    pretrain_path = "./model/airl/policy/ppo_policy_bairl111.zip"
    if os.path.exists(pretrain_path):
        print("loading pretrain model")
        learner = PPO.load(pretrain_path)
        b_reward_net = torch.load("./model/airl/reward/reward_net_bairl1.pth", weights_only=False)
    else:
        policy_kwargs = dict(
            features_extractor_class=feature_extractor,
            features_extractor_kwargs=dict(features_dim=96),
            share_features_extractor=False,
            log_std_init=-1
        )

        learner = PPO(
            env=env,
            policy=ActorCriticPolicy,
            gae_lambda=0.92,
            ent_coef=2.0745206045994986e-05,
            policy_kwargs=policy_kwargs,
            learning_rate=2.0309225666232827e-05,
            n_steps=2048,
            batch_size=256,
            gamma=0.999,
            clip_range=0.2,
            # clip_range_vf=0.1,
            normalize_advantage=True,
            vf_coef=0.819262464558427,
            max_grad_norm=0.5,
            n_epochs=20,
            seed=SEED,
            device="cuda",
            tensorboard_log="./log/",
            verbose=1,
        )

    airl_trainer = custom_AIRL(
        demonstrations=rollouts,
        demo_batch_size=4096,
        gen_replay_buffer_capacity=4096,
        n_disc_updates_per_round=4,
        venv=env,
        gen_algo=learner,
        reward_net=b_reward_net,
        reg="B",  # Adversarial_Augmentation:AA, Gradient penalty: GP, Bayesian
        lambda_reg=0.01,
        init_tensorboard=True,
        log_dir="./log/",
        init_tensorboard_graph=True,
        allow_variable_horizon=True
    )
    pretrain(learner, env, 20, figure=False)
    # airl_trainer.train_gen()
    # opt = torch.load("./model/airl/policy/ppo_policy_bairl_opt2.pth", weights_only=False)
    # airl_trainer._disc_opt.load_state_dict(opt)
    learner.policy.log_std.requires_grad = True
    # for n in range(1, 10):
    #     print("warmup reward net")
    #     airl_trainer.train_disc()
    # pretrain(learner, env, data=rollouts, n_epoch=5, figure=False, testing=False)
    for n in range(1, 26):
        # if n % 2 == 0:
        #     pretrain(learner, env, data=rollouts, n_epoch=2, figure=False, testing=False)
        if n > 2:
            airl_trainer.lambda_reg *= 0.2
        learner.policy.log_std.requires_grad = True
        print(f"This is the No. {n}")
        airl_trainer.train(20000)
        torch.save(b_reward_net, f"./model/airl/reward/reward_net_bairl_1_{n}.pth")
        learner.save(f"./model/airl/policy/ppo_policy_bairl_1_{n}.zip")
        torch.save(airl_trainer._disc_opt.state_dict(), f"./model/airl/policy/ppo_policy_bairl_opt_1_{n}.pth")
    # env.seed(SEED)
    # learner_rewards_before_training, _ = evaluate_policy(
    #     learner, env, 100, return_episode_rewards=True,
    # )
    # airl_trainer.train(1000000)
    # env.seed(SEED)
    # learner_rewards_after_training, _ = evaluate_policy(
    #     learner, env, 100, return_episode_rewards=True,
    # )
    # print("mean reward after training:", np.mean(learner_rewards_after_training))
    # print("mean reward before training:", np.mean(learner_rewards_before_training))
    # graph_reward_box(airl_trainer.reward_exp, airl_trainer.reward_steps, "expert-mountain")
    # graph_reward_box(airl_trainer.reward_gen, airl_trainer.reward_steps, "gen-mountain")
