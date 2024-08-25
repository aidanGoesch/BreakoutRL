from collections import defaultdict
from state import State
import random

GAMMA = 0.9
ALPHA = 0.01
EPSILON = 0.1

LAMBDA = 0.9

INITIAL_ACTION_VALUE = 3.0


class Agent:
    def __init__(self):
        self.q = defaultdict(lambda: dict({"LEFT": [INITIAL_ACTION_VALUE, 0], "RIGHT": [INITIAL_ACTION_VALUE, 0]}))  # initializes a value function where every state can be assigned a value
                                                                    #      - init value = 0
        self.prev_state = State((1, 1), 425, 400, 400)     # null initialize for the start of the episode

        self.to_update = []


    def update(self, state : State, action : str, reward = 0):
        # tmp_q = self.q[self.prev_state][action] +\
        #                                    ALPHA * (reward + GAMMA * max(self.q[state].values()) - self.q[self.prev_state][action])

        sigma = reward + GAMMA * self.q[state][action][0] - self.q[self.prev_state][action][0]    # TD Error
        self.q[self.prev_state][action][1] = 1    # set ET
        self.to_update.append(self.prev_state)

        to_remove = []
        for i, key in enumerate(self.to_update):   # update every eligible state
            if self.q[key][action][1] < 0.005:
                to_remove.append(i)

            self.q[key][action][0] = self.q[key][action][0] + ALPHA * sigma * self.q[key][action][1]
            self.q[key][action][1] = LAMBDA * GAMMA * self.q[key][action][1]     # decrement ET

        for i in reversed(to_remove):   # remove ineligible states
            self.to_update.pop(i)

        # self.q[self.prev_state][action][0] = tmp_q

        self.prev_state = state

    def act(self, state : State):
        if self.q[state]["LEFT"] == self.q[state]["RIGHT"]:
            return random.choice(["RIGHT", "LEFT"])

        if random.random() < EPSILON:
            action = min(self.q[state].items(), key=lambda x: x[1])[0]
        else:
            action = max(self.q[state].items(), key=lambda x: x[1])[0]

        return action

    # def reset(self):
    #     self.prev_state = State((1, 1), 425, 400, 400)
    #     self.path = [self.prev_state]


if __name__ == '__main__':
    a = Agent()

    print(a.q["B"])

