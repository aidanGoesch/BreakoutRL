import pygame as pg
from time import time
import math

from agent import Agent
from state import State


HEIGHT, WIDTH, OFFSET = 50, 100, 5

UP_VECTOR = pg.Vector2(1, 0)
DOWN_VECTOR = pg.Vector2(-1, 0)
LEFT_VECTOR = pg.Vector2(0, -1)
RIGHT_VECTOR = pg.Vector2(0, 1)


class BreakoutSquare:
    def __init__(self, y: int, x: int) -> None:
        self.coordinate = (y, x)
        self.color = (255, 255, 255)
        self.rect = pg.Rect(x * WIDTH, y * HEIGHT, WIDTH, HEIGHT)


class Paddle:
    def __init__(self):
        self.x = 400

    y_loc = 550

    def right(self):
        if self.x < 750:
            self.x += 5

    def left(self):
        if self.x > 50:
            self.x -= 5

    def get_coordinate(self):
        return Paddle.y_loc, self.x


class Ball:
    def __init__(self):
        self.coordinate = 425, 400
        self.color = (255, 255, 255)
        self.velocity = pg.Vector2(1, 1)
        self.radius = 10

        self.paddle_hit = False


    def get_velocity(self):
        return self.velocity[0], self.velocity[1]


    def update(self, squares, paddle, start_time):
        # update the coordinate
        self.coordinate = (self.coordinate[0] + self.velocity[0], self.coordinate[1] + self.velocity[1])

        alpha = 1.00005

        self.velocity = pg.Vector2(alpha * self.velocity[0], alpha * self.velocity[1])
        # print(self.velocity, alpha)

        # check collisions
        y, x = self.coordinate

        # check for collisions with the edge of the map
        if x + self.radius > 800:
            self.velocity = self.velocity.reflect(LEFT_VECTOR)

        if x - self.radius < 0:
            self.velocity = self.velocity.reflect(RIGHT_VECTOR)

        if y - self.radius < 0:
            self.velocity = self.velocity.reflect(DOWN_VECTOR)

        # check for collisions with the paddle
        if time() - self.paddle_hit > 1:  # cool down for paddle hits so that it doesn't get stuck in the paddle
            self.paddle_hit = -1

        if (paddle.y_loc - 12.5 < y + self.radius < paddle.y_loc and        # make sure that the ball can't be hit by
                (paddle.x - 50 < x < paddle.x + 50) and self.paddle_hit == -1):  # using the under side of the paddle
            self.velocity = self.velocity.reflect(UP_VECTOR)
            self.paddle_hit = time()

        ball_rect = pg.Rect(0, 0, 2 * self.radius, 2 * self.radius)
        ball_rect.center = x, y

        # check for collision with squares
        to_remove = []
        for i, row in enumerate(squares):
            for j, square in enumerate(row):
                if ball_rect.colliderect(square.rect):
                    bounce = False

                    if square.rect.bottom > y - self.radius and square.rect.left < x - self.radius and square.rect.right > x + self.radius:
                        bounce = True
                        self.velocity = self.velocity.reflect(DOWN_VECTOR)

                    elif square.rect.right > x - self.radius:
                        bounce = True
                        self.velocity = self.velocity.reflect(RIGHT_VECTOR)

                    elif square.rect.left < x + self.radius:
                        bounce = True
                        self.velocity = self.velocity.reflect(LEFT_VECTOR)

                    elif square.rect.top < y + self.radius:
                        bounce = True
                        self.velocity = self.velocity.reflect(UP_VECTOR)

                    if bounce:
                        to_remove.append((i, j))

        for i, j in to_remove:
            try:
                squares[i].pop(j)
            except IndexError:
                print("whoops")

        return squares



class Breakout:
    def __init__(self):
        pg.init()

        self.screen = pg.display.set_mode((800, 600))
        pg.display.set_caption('Breakout')
        self.clock = pg.time.Clock()
        self.fps = 1200

        self.running = True

        self.board = [[BreakoutSquare(y, x) for x in range(8)] for y in range(8)]
        self.paddle = Paddle()
        self.ball = Ball()

        self.paddle_right = False
        self.paddle_left = False

        self.start_time = time()

        self.agent = Agent()
        self.iterations = 0

    def draw(self, paddle_rect):
        self.screen.fill((0, 0, 0))

        # draw the paddle rect
        pg.draw.rect(self.screen, (255, 255, 255), paddle_rect)

        # draw the ball
        pg.draw.circle(self.screen, self.ball.color, (self.ball.coordinate[1], self.ball.coordinate[0]), self.ball.radius)

        # draw all the squares
        for row in self.board:
            for square in row:
                pg.draw.rect(self.screen, square.color, square.rect)  # draw the actual rect
                pg.draw.rect(self.screen, (0, 0, 0), square.rect, 2)  # draw the border rect

        pg.display.flip()

    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_RIGHT:
                    self.paddle_right = True
                elif event.key == pg.K_LEFT:
                    self.paddle_left = True

            elif event.type == pg.KEYUP:
                if event.key == pg.K_RIGHT:
                    self.paddle_right = False
                elif event.key == pg.K_LEFT:
                    self.paddle_left = False

        if self.paddle_left:
            self.paddle.left()
        if self.paddle_right:
            self.paddle.right()

    def win(self):
        self.running = False

        # use this to change some sort of reward signal


    def lose(self):
        self.running = False


    def reset(self):
        print(f"iterations: {self.iterations}")
        self.iterations += 1
        self.clock = pg.time.Clock()
        # self.fps = 120

        self.running = True

        self.board = [[BreakoutSquare(y, x) for x in range(8)] for y in range(8)]
        self.paddle = Paddle()
        self.ball = Ball()

        self.paddle_right = False
        self.paddle_left = False

        self.start_time = time()

        # self.agent.reset()


    def check_win_lose(self, state : State, action):
        if all(len(row) == 0 for row in self.board):  # win condition
            self.agent.update(state, action, reward=50)
            self.reset()

        if self.ball.coordinate[0] > 600:  # lose condition
            r = self.calculate_reward()     # less punishment if the ball is close to the paddle during loss
            print(r)
            self.agent.update(state, action, reward=r)
            self.reset()

    def calculate_reward(self):
        return - math.sqrt(((self.ball.coordinate[0] - self.paddle.y_loc) ** 2 + (self.ball.coordinate[1] - self.paddle.x) ** 2)) / 85

    def run(self):
        prev_action = "RIGHT"
        while self.running:
            self.clock.tick(self.fps)

            py, px = self.paddle.get_coordinate()
            paddle_rect = pg.Rect(0, 0, WIDTH, 25)
            paddle_rect.center = px, py

            state = State(self.ball.get_velocity(), self.ball.coordinate[0], self.ball.coordinate[1], px)

            y, x = self.ball.coordinate
            if (self.paddle.y_loc - 12.5 < y + 10 < self.paddle.y_loc and  # make sure that the ball can't be hit by
                    (self.paddle.x - 50 < x < self.paddle.x + 50)):
                self.agent.update(state, prev_action, 2 * (1 - abs(self.paddle.x - x) / 50))  # make it so that you get more reward for hitting it in the cetner of the paddle
            else:
                self.agent.update(state, prev_action)
            prev_action = self.agent.act(state)

            if prev_action == "LEFT":
                self.paddle.left()
            elif prev_action == "RIGHT":
                self.paddle.right()

            self.board = self.ball.update(self.board, self.paddle, self.start_time)

            # self.agent.path.append(state)   # add it to the eligibility trace path

            self.check_win_lose(state, prev_action)

            self.draw(paddle_rect)
            self.handle_events()


if __name__ == "__main__":
    breakout = Breakout()

    breakout.run()