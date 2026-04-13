import numpy as np


def bandit(num_actions, R, epsilon):
    Q = np.zeros(num_actions)
    N = np.zeros(num_actions)
    rng = np.random.default_rng(42)

    while True:
        if rng.uniform() < epsilon:
            a = np.argmax(Q)
        else:
            a = rng.integers(low=0, high=Q.shape[0])

        old_qa = Q[a]
        N[a] += 1
        Q[a] += (R[a] - Q[a]) / N[a]

        if old_qa == Q[a]:
            break

    return Q
