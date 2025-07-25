# Drone-BAIRL
A PyTorch-based implementation of Adversarial Inverse Reinforcement Learning with Bayesian Network (BAIRL) for vision-based continuous-control drone navigation.
By introducing MC-dropout into the update process of the discriminator, the initial oscillations can be reduced.This repository provides training, evaluation, and reward visualization tools for UAV navigation tasks in AirSim environments.

## 🚀 Requirement
unreal engine = 4.27<br>
airsim = 1.5.0<br>
gymnasium >= 0.29.1<br>
stablebaseline3 = 2.2.1<br>
torch >= 2.6.0<br>
imitation = 1.0.1<br>

## 💻 Equirement
Windows 10<br>
Python 3.10.13<br>

## 🔌 env and model
Download：https://drive.google.com/file/d/1PP7fZ1aoXN4u-jsJjv2CFm5q5xAD4mbV/view?usp=sharing <br>

## 🔥 Training

run main.py<br>
Hyperparameter selection: <br>
PPO: <br> https://huggingface.co/HumanCompatibleAI/ppo-seals-Humanoid-v0<br> n_step=2048 batch_size=512 <br>
reward net: <br> demo_batch_size=2048, gen_replay_buffer_capacity=512, n_disc_updates_per_round=4, lambda_reg=0.01<br>

## 🔥 model

there is the trained policy and reward model in ./model success rate: 80 average step: 91.25

## 😄 Reference & acknowledgment

Reference environment and part of baseline reward function idea for my training **https://github.com/sunghoonhong/AirsimDRL **
