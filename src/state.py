
class State:
    def __init__(self, velocity, ball_y, ball_x, paddle_x, paddle_y=550):
        self.velocity = round(velocity[0], 5), round(velocity[1], 5)
        self.ball_y = round(ball_y)
        self.ball_x = round(ball_x)
        self.paddle_x = round(paddle_x)
        self.paddle_y = paddle_y


    def __repr__(self):
        return f"State(velocity={self.velocity}, ball_y={self.ball_y}, ball_x={self.ball_x}, self.paddle_x={self.paddle_x})"
    def __eq__(self, other):
        return (self.velocity == other.velocity and self.ball_y == other.ball_y and self.ball_x == other.ball_x
                and self.paddle_x == other.paddle_x and self.paddle_y == other.paddle_y)


    def __hash__(self):
        return hash((self.velocity, self.ball_y, self.ball_x, self.paddle_x, self.paddle_y))
