import torch

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
        print("init")

    def train(self):
        print("train")