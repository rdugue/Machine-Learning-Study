import numpy as np


class DynamicMDP:
    def __init__(self, num_states, num_actions, P, R, gamma=0.9):
        self.rng = np.random.default_rng(42)
        self.S = num_states
        self.A = num_actions
        self.gamma = gamma
        # Transitions: P[state, action, next_state]
        self.P = P
        # Rewards: R[state, action]
        self.R = R

    def policy_evaluation(self, policy, threshold=1e-6):
        V = np.zeros(self.S)
        while True:
            delta = 0
            for s in range(self.S):
                v_old = V[s]
                # Use the deterministic policy to pick the action
                a = policy[s]

                # V[s] = R(s,a) + gamma * sum_{s'} (P(s'|s,a) * V[s'])
                new_v = self.R[s, a] + self.gamma * np.dot(self.P[s, a, :], V)

                V[s] = new_v
                delta = max(delta, abs(v_old - V[s]))

            if delta < threshold:
                break
        return V

    def policy_iteration(self):
        # Initialize a random deterministic policy
        policy = self.rng.integers(0, self.A, self.S)

        while True:
            # 1. Evaluation
            V = self.policy_evaluation(policy)

            policy_stable = True
            for s in range(self.S):
                old_action = policy[s]

                # Calculate Q(s, a) for all actions and take the argmax
                q_values = [
                    self.R[s, a] + self.gamma * np.dot(self.P[s, a, :], V)
                    for a in range(self.A)
                ]

                policy[s] = np.argmax(q_values)

                if old_action != policy[s]:
                    policy_stable = False

            # If policy doesn't change, it will never change again
            if policy_stable:
                break

        return policy, V

    def value_iteration(self, threshold=1e-6):
        V = np.zeros(self.S)  # Initialize values to zero
        while True:
            delta = 0
            for s in range(self.S):
                v_old = V[s]

                # V[s] = max_a [ R(s,a) + gamma * sum_{s'} (P(s'|s,a) * V[s']) ]
                q_values = [
                    self.R[s, a] + self.gamma * np.dot(self.P[s, a, :], V)
                    for a in range(self.A)
                ]
                V[s] = max(q_values)

                delta = max(delta, abs(v_old - V[s]))

            # Check convergence using L-infinity norm
            if delta < threshold:
                break

        # Extract the final policy using argmax
        optimal_policy = np.zeros(self.S, dtype=int)
        for s in range(self.S):
            q_values = [
                self.R[s, a] + self.gamma * np.dot(self.P[s, a, :], V)
                for a in range(self.A)
            ]
            optimal_policy[s] = np.argmax(q_values)

        return optimal_policy, V
