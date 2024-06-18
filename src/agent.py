from collections import defaultdict
from state import State
import random

GAMMA = 0.9
ALPHA = 0.01
EPSILON = 0.1

INITIAL_ACTION_VALUE = 0.0


class Agent:
    def __init__(self):
        self.q = defaultdict(lambda: dict({"LEFT": INITIAL_ACTION_VALUE, "RIGHT": INITIAL_ACTION_VALUE}))  # initializes a value function where every state can be assigned a value
                                                                    #      - init value = 0
        self.prev_state = State((1, 1), 425, 400, 400)     # null initialize for the start of the episode
        self.iterations_since_new_reward = 0
        self.total_reward_this_iteration = 0
        self.prev_rewards = []

        self.e = EPSILON

    def update(self, state : State, action : str, reward = 0):
        # print(list(self.q[state].values()))
        if reward != 0:
            if reward == -5:  # end via losing
                if self.prev_rewards[-1] == self.total_reward_this_iteration:
                    self.prev_rewards.append(self.total_reward_this_iteration - 5)

                    if len(self.prev_rewards) > 30:
                        self.e = 0.4
                else:
                    self.prev_rewards = []
                    self.e = EPSILON

                # self.total_reward_this_iteration = 0
            else:
                self.total_reward_this_iteration += reward
            self.iterations_since_new_reward = 0
        else:
            self.iterations_since_new_reward += 1
        self.q[self.prev_state][action] = self.q[self.prev_state][action] +\
                                           ALPHA * (reward + GAMMA * max(self.q[state].values()) - self.q[self.prev_state][action])

        self.prev_state = state

    def act(self, state : State):
        # action = max(self.q[state].values, key=lambda x: x[1])[0]

        if self.q[state]["LEFT"] == self.q[state]["RIGHT"]:
            return random.choice(["RIGHT", "LEFT"])
        e = EPSILON
        if self.iterations_since_new_reward > 100:
            e = 0.5
        if random.random() < e:
            action = min(self.q[state].items(), key=lambda x: x[1])[0]
        else:
            action = max(self.q[state].items(), key=lambda x: x[1])[0]

        return action


if __name__ == '__main__':
    a = Agent()

    print(a.q["B"])

