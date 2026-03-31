import numpy as np

class MonteCarlo:
    def __init__(self, num_states, episodes, alpha=0.1, gamma=0.99):
        self.num_states = num_states
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma

    def first_visit(self):
        V = np.zeros(self.num_states)
        g_sums = np.zeros(self.num_states)
        g_counts = np.zeros(self.num_states)

        for episode in self.episodes:
            G = 0
            states_in_episode = [step[0] for step in episode]
            for t in reversed(range(len(episode))):
                s, r = episode[t]
                G = r + self.gamma * G
                if s not in states_in_episode[:t]:
                    g_sums[s] += G
                    g_counts[s] += 1

        V = np.divide(g_sums, g_counts, out=np.zeros_like(g_sums), where=g_counts!=0)
        return V
    
    def every_visit(self):
        V = np.zeros(self.num_states)
        g_sums = np.zeros(self.num_states)
        g_counts = np.zeros(self.num_states)

        for episode in self.episodes:
            G = 0 
            for t in reversed(range(len(episode))):
                s, r = episode[t]
                G = r + self.gamma * G
                g_sums[s] += G
                g_counts[s] += 1

        V = np.divide(g_sums, g_counts, out=np.zeros_like(g_sums), where=g_counts!=0)
        return V

    def incremental(self):
        V = np.zeros(self.num_states)

        for episode in self.episodes:
            G = 0
            for t in reversed(range(len(episode))):
                s, r = episode[t]
                G = r + self.gamma * G
                V[s] += self.alpha * (G - V[s]) 

        return V