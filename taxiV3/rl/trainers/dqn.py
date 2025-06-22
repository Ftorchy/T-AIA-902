"""
Deep-Q-Learning pour Taxi-v3
— implémentation minimale (PyTorch) compatible avec la model_bank —
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


# ───────────────────────── paramètres ─────────────────────────
@dataclass
class DQNParams:
    episodes: int
    gamma: float
    lr: float
    eps0: float
    eps_min: float
    eps_decay: float
    max_steps: int = 200
    batch: int = 64
    memory: int = 20_000
    target_sync: int = 500
    device: str = "cpu"


# ───────────────────────── réseau ─────────────────────────────
class Net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):          # x already float32
        return self.net(x)


# ───────────────────────── entraîneur ─────────────────────────
class DeepQLearning:
    def __init__(self, p: DQNParams):
        self.p = p
        self.env = gym.make("Taxi-v3")
        self.obs_dim = self.env.observation_space.n
        self.act_dim = self.env.action_space.n

        self.policy = Net(self.obs_dim, self.act_dim).to(self.p.device)
        self.target = Net(self.obs_dim, self.act_dim).to(self.p.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.p.lr)

        self.buffer = []                   # très simple (state, action, r, next, done)
        self.metrics = {"rewards": [], "steps": [], "success": []}

    # utilitaire one-hot
    def oh(self, idx):
        vec = np.zeros(self.obs_dim, np.float32)
        vec[idx] = 1.0
        return vec

    def remember(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))
        if len(self.buffer) > self.p.memory:
            self.buffer.pop(0)

    def replay(self):
        if len(self.buffer) < self.p.batch:
            return
        idx = np.random.choice(len(self.buffer), self.p.batch, replace=False)
        s, a, r, ns, d = zip(*(self.buffer[i] for i in idx))

        s  = torch.tensor(s, dtype=torch.float32).to(self.p.device)
        a  = torch.tensor(a).unsqueeze(1).to(self.p.device)
        r  = torch.tensor(r, dtype=torch.float32).to(self.p.device)
        ns = torch.tensor(ns, dtype=torch.float32).to(self.p.device)
        d  = torch.tensor(d, dtype=torch.float32).to(self.p.device)

        q_sa   = self.policy(s).gather(1, a).squeeze()
        with torch.no_grad():
            q_ns = self.target(ns).max(1)[0]
        target = r + self.p.gamma * q_ns * (1 - d)

        loss = nn.MSELoss()(q_sa, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def train(self):
        eps = self.p.eps0
        step_cnt = 0
        for ep in range(self.p.episodes):
            s, _ = self.env.reset()
            tot_r = steps = succ = 0
            for _ in range(self.p.max_steps):
                # sélection action
                a = self.env.action_space.sample() if np.random.rand() < eps \
                    else int(self.policy(torch.tensor(self.oh(s)).to(self.p.device)).argmax())
                ns, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                succ = 1 if term and r == 20 else succ
                # mémorise et learn
                self.remember(self.oh(s), a, r, self.oh(ns), done)
                self.replay()
                # maj cibles
                if step_cnt % self.p.target_sync == 0:
                    self.target.load_state_dict(self.policy.state_dict())

                s, tot_r = ns, tot_r + r
                steps += 1; step_cnt += 1
                if done:
                    break

            # log épisode
            eps = max(eps - self.p.eps_decay, self.p.eps_min)
            self.metrics["rewards"].append(tot_r)
            self.metrics["steps"].append(steps)
            self.metrics["success"].append(succ)

    # ⤵︎ Interface minimale requise par model_bank
    def export(self):
        return {
            "algo": "dqn",
            "policy_state": self.policy.state_dict(),
            "metrics": {
                "reward_mean": np.mean(self.metrics["rewards"][-100:]),
                "success_pct": 100 * np.mean(self.metrics["success"][-100:]),
                "steps_mean":  np.mean(self.metrics["steps"][-100:])
            }
        }


# fonction unique appelée par l'UI
def train(p_dict):
    trainer = DeepQLearning(DQNParams(**p_dict))
    trainer.train()
    return trainer.export()