import numpy as np

class TemporalDifference:
    def __init__(self, num_states, episodes, terminal_states, alpha=0.1, gamma=0.99):
        self.num_states = num_states
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.terminal_states = terminal_states


    def td_zero(self):
        V = np.zeros(self.num_states)

        for episode in self.episodes:
            state = episode[0]
            while state not in self.terminal_states:
                reward = episode[1]
                next_state = episode[2]

                target = reward + self.gamma * V[next_state]
                V[state] += self.alpha * (target - V[state])

                state = next_state

        return V