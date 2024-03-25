import gymnasium as gym
import pygame
import numpy as np

render = False
fps = 5

if render is True:
    # 게임 윈도우 초기화
    pygame.init()

    SCREEN_WIDTH, SCREEN_HEIGHT = 600, 400
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    pygame.display.set_caption('CartPole')
    font = pygame.font.SysFont(None, 36)  # 기본 시스템 폰트, 크기 36

# CartPole 환경 생성
env = gym.make('CartPole-v1')

# 게임 루프
running = True
clock = pygame.time.Clock()

while running:
    # CartPole 상태 업데이트
    observation = env.reset()
    observation = observation[0]

    score = 0
    for t in range(1000):  # 최대 1000번 반복
        if render is True:
            screen.fill((255, 255, 255))  # 화면을 흰색으로 채움

            cart_x = SCREEN_WIDTH // 2 + int(observation[0] * SCREEN_WIDTH / 2)
            pole_top = SCREEN_HEIGHT // 2 - 100
            pole_end_x = cart_x + int(np.sin(observation[2]) * 100)
            pole_end_y = pole_top - int(np.cos(observation[2]) * 100)
            pygame.draw.rect(screen, (0, 255, 0), [cart_x - 25, pole_top, 50, 10])  # 카트 그리기
            pygame.draw.line(screen, (0, 0, 0), (cart_x, pole_top), (pole_end_x, pole_end_y), 2)  # 막대 그리기

        action = env.action_space.sample()  # 무작위로 행동 선택

        observation, reward, terminated, truncated, info = env.step(action)

        score += reward
        print(f"score : {score}")

        if render is True:
            # Rendering score part
            score_text = font.render("Score: " + str(score), True, (0, 0, 0))  # 흰색 텍스트
            score_rect = score_text.get_rect()
            score_rect.bottomright = (SCREEN_WIDTH - 10, SCREEN_HEIGHT- 10)  # 오른쪽 하단에 위치

            # 화면에 스코어 텍스트 그리기
            screen.blit(score_text, score_rect)

            pygame.display.flip()  # 화면 업데이트

            clock.tick(fps) # arg frames per second

        if terminated:
            print("DONE")
            pygame.display.set_caption("DONE")
            running = False
            break


while True and render is True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
