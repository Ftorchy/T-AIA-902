import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from dataclasses import dataclass

class Net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
    def forward(self, x):
        return self.net(x)

@dataclass
class DQNParams:
    episodes: int
    gamma: float
    lr: float
    eps0: float
    eps_min: float
    eps_decay: float
    max_steps: int = 200

def train(params: DQNParams, device="cpu"):
    env = gym.make("Taxi-v3")
    obs_dim = env.observation_space.n
    act_dim = env.action_space.n

    def onehot(i):
        v = np.zeros(obs_dim, np.float32)
        v[i] = 1.0
        return v

    net = Net(obs_dim, act_dim).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=params.lr)

    eps = params.eps0
    rewards = []

    for ep in range(params.episodes):
        s, _ = env.reset()
        tot_r = 0
        for _ in range(params.max_steps):
            a = env.action_space.sample() if np.random.rand() < eps else int(net(torch.tensor(onehot(s)).to(device)).argmax())
            s, r, term, trunc, _ = env.step(a)
            loss = -r
            optim.zero_grad()
            torch.tensor(loss, dtype=torch.float32, requires_grad=True).backward()
            optim.step()
            tot_r += r
            if term or trunc:
                break
        rewards.append(tot_r)
        eps = max(eps - params.eps_decay, params.eps_min)

    env.close()
    return {
        "algo": "dqn",
        "policy_state": net.state_dict(),
        "metrics": {
            "reward_mean": np.mean(rewards[-100:]),
        },
    }