import gym
import pygame
import numpy as np

# 게임 윈도우 초기화
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('CartPole')

# CartPole 환경 생성
env = gym.make('CartPole-v1')

# 게임 루프
running = True
while running:
    screen.fill((255, 255, 255))  # 화면을 흰색으로 채움

    # CartPole 상태 업데이트
    observation = env.reset()
    # print(f"observation : {observation}")
    # print(f"observation[0] : {observation[0]}")
    # print(f"observation[2] : {observation[2]}")
    observation = observation[0]
    for t in range(1000):  # 최대 1000번 반복
        # CartPole을 화면에 그리기
        cart_x = SCREEN_WIDTH // 2 + int(observation[0] * SCREEN_WIDTH / 2)
        pole_top = SCREEN_HEIGHT // 2 - 100
        pole_end_x = cart_x + int(np.sin(observation[2]) * 100)
        pole_end_y = pole_top - int(np.cos(observation[2]) * 100)
        pygame.draw.rect(screen, (0, 255, 0), [cart_x - 25, pole_top, 50, 10])  # 카트 그리기
        pygame.draw.line(screen, (0, 0, 0), (cart_x, pole_top), (pole_end_x, pole_end_y), 2)  # 막대 그리기

        pygame.display.flip()  # 화면 업데이트

        action = env.action_space.sample()  # 무작위로 행동 선택
        # observation, reward, done, info = env.step(action)
        print(env.step(action))

        # if done:
        #     break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

pygame.quit()  # pygame 종료
