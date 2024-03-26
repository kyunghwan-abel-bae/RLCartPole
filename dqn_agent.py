import torch
import random

import numpy as np

from torch import nn
from collections import deque

NUM_CHANNELS = 4
NUM_ACTIONS = 2


class DQNAgentModel(nn.Module):
    def __init__(self, num_channels, num_actions):
        super().__init__()

        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(num_channels, 256)
        self.linear2 = nn.Linear(256, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class DQNAgent:
    def __init__(self, gamma, batch_size, min_replay_memory_size, replay_memory_size, target_update_freq):
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_memory_size = min_replay_memory_size
        self.target_update_freq = target_update_freq

        self.model = DQNAgentModel(NUM_CHANNELS, NUM_ACTIONS)
        self.target_model = DQNAgentModel(NUM_CHANNELS, NUM_ACTIONS)
        self.target_model.load_state_dict(self.model.state_dict())

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters())

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.target_update_counter = 0

    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))

    def get_q_values(self, x):
        q_values = self.model(x)
        return q_values.detach().numpy()

    def train(self):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        samples = random.sample(self.replay_memory, self.batch_size)
        current_input = np.stack([sample[0] for sample in samples])
        current_q_values = self.model(current_input)
        current_q_values_clone = current_q_values.clone()

        next_input = np.stack([sample[3] for sample in samples])
        next_q_values = self.target_model(next_input)

        for i, (current_state, action, reward, _, done) in enumerate(samples):
            if done:
                next_q_value = reward
            else:
                # print(f"next_q_values : {next_q_values}")
                next_q_value = reward + self.gamma * torch.max(next_q_values[i])
            current_q_values[i, action] = next_q_value

        pred_tensor = current_q_values_clone
        target_tensor = current_q_values

        loss = self.loss_fn(pred_tensor, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def increase_target_update_encounter(self):
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        self.target_model.load_state_dict(torch.load(file_name))


