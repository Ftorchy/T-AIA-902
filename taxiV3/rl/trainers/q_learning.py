import numpy as np
import gymnasium as gym
from dataclasses import dataclass

@dataclass
class QLParams:
    episodes: int
    alpha: float
    gamma: float
    eps0: float
    eps_min: float
    eps_decay: float
    max_steps: int = 200

def train(params: QLParams):
    env = gym.make("Taxi-v3")
    q = np.zeros((env.observation_space.n, env.action_space.n), np.float32)

    eps_curve = np.maximum(
        params.eps0 - params.eps_decay * np.arange(params.episodes),
        params.eps_min,
    )

    rewards, steps, success = [], [], []
    for ep in range(params.episodes):
        s, _ = env.reset()
        tot_r = n = 0
        for _ in range(params.max_steps):
            n += 1
            a = env.action_space.sample() if np.random.rand() < eps_curve[ep] else int(q[s].argmax())
            ns, r, term, trunc, _ = env.step(a)
            if term and r == 20:
                success.append(1)
            q[s, a] += params.alpha * (r + params.gamma * q[ns].max() - q[s, a])
            tot_r += r
            s = ns
            if term or trunc:
                break
        rewards.append(tot_r)
        steps.append(n)

    env.close()
    return {
        "algo": "q_learning",
        "q": q,
        "metrics": {
            "reward_mean": np.mean(rewards[-100:]),
            "success_pct": 100 * np.mean(success[-100:]),
            "steps_mean":  np.mean(steps[-100:]),
        },
    }