import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from imitation.util.networks import BaseNorm, RunningNorm
from stable_baselines3.common.torch_layers import CombinedExtractor, BaseFeaturesExtractor
from imitation.rewards.reward_nets import BasicShapedRewardNet
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, cast
from imitation.rewards.reward_nets import ShapedRewardNet, BasicPotentialMLP, BasicRewardNet
from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial.airl import AIRL
from typing import Optional, Mapping
import torch
from imitation.rewards.reward_nets import BasicShapedRewardNet, NormalizedRewardNet, RewardNet
from imitation.util.networks import RunningNorm


def obs_split(combined):
    batch_size = combined.shape[0]
    img = combined[:, :, :, :128]
    vel = combined[:, :, :, 128:]
    vel = vel.reshape(batch_size, 1, 72, 3)
    vel = vel[:, 0, 0, :]

    return img, vel


class feature_extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces, normalize_velocity_layer=None,
                 normalize_image_layer=None,
                 features_dim=64,
                 normalize=RunningNorm):
        super().__init__(observation_space, features_dim=features_dim)
        self.normalize_velocity_layer = normalize_velocity_layer
        self.normalize_image_layer = normalize_image_layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv1[0].weight.data.normal_(0, 0.5)  # initialization

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2[0].weight.data.normal_(0, 0.5)  # initialization

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv3[0].weight.data.normal_(0, 0.5)  # initialization
        self.image_fc1 = nn.Linear(256, 64, bias=False)
        self.image_fc2 = nn.Linear(64, int(features_dim / 2), bias=False)
        self.image_fc1.weight.data.normal_(0, 0.5)
        self.image_fc2.weight.data.normal_(0, 0.5)

        self.vel_fc1 = nn.Linear(3, int(features_dim / 2), bias=False)
        self.vel_fc1.weight.data.normal_(0, 0.5)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(inplace=True)

        self.normalize = normalize(self.features_dim)

    def forward(self, obs):
        image, vel = obs_split(obs)

        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.image_fc1(x)
        x = self.relu(x)
        x = self.image_fc2(x)

        vel = self.vel_fc1(vel)

        state_process = torch.cat((x, vel), dim=1)
        state_process = self.tanh(state_process)

        state_process = self.normalize(state_process)

        return state_process


class Bayesian_reward_net(RewardNet):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            *,
            use_state: bool = True,
            use_action: bool = True,
            use_next_state: bool = False,
            use_done: bool = False,
            discount_factor: float = 0.99,
            feature_extractor: Type[BaseFeaturesExtractor] = None,
            normalize_output_layer=RunningNorm,
            **kwargs,
    ):

        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
        )
        self.observation_space = observation_space
        self.action_space = action_space
        self.use_state = use_state
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.use_action = use_action
        self.discount_factor = discount_factor
        self.normalize_output_layer = normalize_output_layer(1)
        # Compute observation dimension
        if hasattr(self.observation_space, 'shape'):
            self.obs_dim = int(np.prod(self.observation_space.shape))
        elif hasattr(self.observation_space, 'n'):
            self.obs_dim = self.observation_space.n
        else:
            raise ValueError('Unsupported observation space type')
        # Compute action dimension
        if hasattr(self.action_space, 'n'):
            self.act_is_discrete = True
            self.act_dim = self.action_space.n
        elif hasattr(self.action_space, 'shape'):
            self.act_is_discrete = False
            self.act_dim = int(np.prod(self.action_space.shape))
        else:
            raise ValueError('Unsupported action space type')

        if feature_extractor is None:
            self.feature_extractor = None
            if use_action:
                self.mlp = BayesianNet(self.obs_dim + self.act_dim)
            else:
                self.mlp = BayesianNet(self.obs_dim)
            self.potential = BayesianNet(self.obs_dim)
        else:
            self.feature_extractor = feature_extractor(observation_space)

            if use_action:
                self.action_transform = nn.Linear(
                    3, int(self.feature_extractor.features_dim)  # Match input size of the first layer
                )
                self.mlp = BayesianNet(self.feature_extractor.features_dim * 2)
            else:
                self.mlp = BayesianNet(self.feature_extractor.features_dim)
            self.potential = BayesianNet(self.feature_extractor.features_dim)

    def forward(self, obs, action, next_obs, done, **kwargs):
        inputs = []
        if self.feature_extractor is not None:
            obs = self.feature_extractor(obs)
            next_obs = self.feature_extractor(next_obs)

        if self.use_state:
            inputs.append(torch.flatten(obs, 1))
        if self.use_action:
            action = self.action_transform(action)
            inputs.append(torch.flatten(action, 1))
        combined_inputs = torch.cat(inputs, dim=-1)
        reward = self.mlp(combined_inputs)
        if self.use_next_state:
            new_shaping_output = self.potential(next_obs).flatten()
            old_shaping_output = self.potential(obs).flatten()
            reward = reward.squeeze(-1)
            old_shaping_output = old_shaping_output.squeeze(-1)
            new_shaping_output = new_shaping_output.squeeze(-1)
            new_shapping = (1 - done.float()) * new_shaping_output
            shaped = self.discount_factor * new_shapping - old_shaping_output
            final_reward = (reward + shaped)
        else:
            final_reward = reward.squeeze(-1)

        final_reward = self.normalize_output_layer(final_reward)
        with torch.no_grad():
            self.normalize_output_layer.update_stats(final_reward)
        assert final_reward.shape == obs.shape[:1]
        return final_reward

    def predict_processed(self, state, action, next_state, done, update_stats=True, **kwargs):
        with torch.no_grad():
            if self.act_is_discrete:
                # 转 one-hot
                if torch.is_tensor(action):
                    action = torch.nn.functional.one_hot(action.long(), num_classes=self.act_dim).float()
                else:
                    # numpy
                    action = np.array(action).reshape(-1)
                    action = torch.tensor(np.eye(self.act_dim)[action], dtype=torch.float32)
            rew_th = self.forward(
                torch.as_tensor(state, device=self.device),
                torch.as_tensor(action, device=self.device),
                torch.as_tensor(next_state, device=self.device),
                torch.as_tensor(done, device=self.device)
            )
            rew = self.normalize_output_layer(rew_th).detach().cpu().numpy().flatten()
            if update_stats:
                self.normalize_output_layer.update_stats(rew_th)
        assert rew.shape == state.shape[:1]
        return rew


class BayesianNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_p=0.2):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_p),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_p),
        #     nn.Linear(hidden_dim, 1),
        # )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.mlp(x)

    def sample_reward(self, x, n_samples=20):
        self.train()
        rewards = []
        for _ in range(n_samples):
            rewards.append(self.forward(x).detach())
        rewards = torch.stack(rewards, dim=0)
        mean = rewards.mean(dim=0).squeeze(-1)
        std = rewards.std(dim=0).squeeze(-1)
        return mean, std
