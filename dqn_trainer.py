import os
import torch
import pygame
import random

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
        max_score = 0
        while self.current_episode < self.episodes:
            current_state = self.env.reset()
            current_state = current_state[0]
            # print(f"current_state : {current_state}")

            done = False
            steps = 0
            score = 0
            while not done and steps < self.max_steps:
                if random.random() > self.epsilon:
                    action = np.argmax(self.agent.get_q_values(current_state))
                else:
                    action = np.random.randint(2)

                # print(f"action : {action}")
                next_state, reward, done, truncated, info = self.env.step(action)
                # print(f"next_state : {next_state}, reward : {reward}, done :{done}")

                score += reward

                self.agent.update_replay_memory(current_state, action, reward, next_state, done)

                self.agent.train()

                current_state = next_state
                steps += 1

            self.agent.increase_target_update_encounter()

            self.epsilon = max(self.epsilon-self.epsilon_decay, self.min_epsilon)

            self.current_episode += 1

            if self.enable_save and self.current_episode % self.save_freq == 0:
                self.save("model_" + str(self.current_episode) + ".pth")

                if score > max_score:
                    max_score = score
                    self.save("best.pth")

                print(f"Current Episode : {self.current_episode}, Epsilon : {self.epsilon}, Max score : {max_score}, score : {score}")

    def save(self, file_name):
        str_name_save = self.save_dir + "/" + file_name
        torch.save(self.agent.model.state_dict(), str_name_save)

    def load(self, file_name):
        str_name_load = self.save_dir + "/" + file_name + ".pth"
        self.agent.load(str_name_load)

    def play(self, render_fps):
        pygame.init()

        SCREEN_WIDTH, SCREEN_HEIGHT = 1400, 400
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        pygame.display.set_caption('CartPole')
        font = pygame.font.SysFont(None, 36)  # 기본 시스템 폰트, 크기 36

        running = True
        clock = pygame.time.Clock()
        while running:
            current_state = self.env.reset()
            current_state = current_state[0]

            done = False
            steps = 0
            score = 0
            while not done and steps < self.max_steps:
                screen.fill((255, 255, 255))

                cart_x = SCREEN_WIDTH // 2 + int(current_state[0] * SCREEN_WIDTH / 2)
                pole_top = SCREEN_HEIGHT // 2 - 100
                pole_end_x = cart_x + int(np.sin(current_state[2]) * 100)
                pole_end_y = pole_top - int(np.cos(current_state[2]) * 100)
                pygame.draw.rect(screen, (0, 255, 0), [cart_x - 25, pole_top, 50, 10])  # 카트 그리기
                pygame.draw.line(screen, (0, 0, 0), (cart_x, pole_top), (pole_end_x, pole_end_y), 2)  # 막대 그리기

                with torch.no_grad():
                    action = np.argmax(self.agent.get_q_values(current_state))

                next_state, reward, done, truncated, info = self.env.step(action)

                score += reward

                current_state = next_state
                steps += 1

                # Rendering score part
                score_text = font.render("Score: " + str(score), True, (0, 0, 0))  # 흰색 텍스트
                score_rect = score_text.get_rect()
                score_rect.bottomright = (SCREEN_WIDTH - 10, SCREEN_HEIGHT - 10)  # 오른쪽 하단에 위치

                # 화면에 스코어 텍스트 그리기
                screen.blit(score_text, score_rect)

                pygame.display.flip()  # 화면 업데이트

                clock.tick(render_fps)  # arg frames per second

                if done:
                    pygame.display.set_caption("DONE")
                    running = False

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()



