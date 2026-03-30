import numpy as np
from enum import Enum

class ControlAlgorithm(Enum):
   MonteCarlo = 1
   QLearning = 2
   SARSA = 3

class ModelFreeControl:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma

    def monte_carlo(self, episodes):
        Q = np.zeros((self.num_states, self.num_actions))

        for episode in episodes:
            G = 0
            for state, action, reward in reversed(episode):
                G = reward + self.gamma * G
                Q[state, action] += self.alpha * (G - Q[state, action])

        return Q

    def sarsa(self, episodes):
        Q = np.zeros((self.num_states, self.num_actions))

        for episode in episodes:
            for state, action, reward, next_state, next_action in episode:
                target = reward + self.gamma * Q[next_state, next_action]
                Q[state, action] += self.alpha * (target - Q[state, action])
                
        return Q
    
    def q_learning(self, episodes):
        Q = np.zeros((self.num_states, self.num_actions))
        
        for episode in episodes:
            for state, action, reward, next_state in episode:
                next_action = np.argmax(Q[next_state])
                target = reward + self.gamma * next_action
                Q[state, action] += self.alpha * (target - Q[state, action])
        
        return Q