import os
import torch
import pygame

import numpy as np
import gymnasium as gym

from dqn_agent import DQNAgent

class DQNTrainer:
    def __init__(self,
                 episodes=30000,
                 initial_epsilon=1.,
                 min_epsilon=0.1,
                 exploration_ratio=0.5,
                 max_steps=2000,
                 render_freq=500,
                 enable_render=True,
                 render_fps=20,
                 save_dir='checkpoints',
                 enable_save=True,
                 save_freq=500,
                 gamma=0.99,
                 batch_size=64,
                 min_replay_memory_size=1000,
                 replay_memory_size=100000,
                 target_update_freq=5,
                 seed=42
                 ):

        # set random seed with seed

        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.exploration_ratio = exploration_ratio
        self.render_freq = render_freq
        self.enable_render = enable_render
        self.render_fps = render_fps
        self.save_dir = save_dir
        self.enable_save = enable_save
        self.save_freq = save_freq

        if enable_save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.env = gym.make("CartPole-v1")
        self.current_episode = 0
        self.max_average_length = 0

        self.epsilon_decay = (initial_epsilon-min_epsilon)/(exploration_ratio*episodes)

        self.agent = DQNAgent(
            gamma=gamma,
            batch_size=batch_size,
            min_replay_memory_size=min_replay_memory_size,
            replay_memory_size=replay_memory_size,
            target_update_freq=target_update_freq
        )

    def train(self):
        current_state = self.env.reset()
        current_state = current_state[0]

        print(f"current_state : {current_state}")
        action = np.argmax(self.agent.get_q_values(current_state))
        print(f"action : {action}")
        next_state, reward, done, truncated, info = self.env.step(action)

        print(f"next_state : {next_state}, reward : {reward}, done :{done}")
