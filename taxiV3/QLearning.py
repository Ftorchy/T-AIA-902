"""Tabular Q-learning pour l’environnement *Taxi-v3* (Gymnasium).

Fonctions exportées
-------------------
train_tabular
    Entraîne un agent tabulaire prêt à l’emploi ; retourne un dictionnaire de
    statistiques (récompenses, steps, temps CPU/Wall, etc.).
TabularQLearning
    Classe agent (méthodes ``train()``, ``act()``, ``play_greedy()``).
QLearningAgent
    Alias public vers :class:`TabularQLearning` pour simplifier l’API.

Ce module est import-safe : tout dépendant lourd (Gymnasium) est déjà mocké
dans ``docs/source/conf.py`` afin que Sphinx puisse générer la doc sur CI.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from time import perf_counter, process_time

__all__: list[str] = ["train_tabular", "TabularQLearning", "QLearningAgent"]  # ✨

def train_tabular(
    env_name: str = "Taxi-v3",
    episodes: int = 5000,
    alpha: float = 0.2,
    gamma: float = 0.95,
    eps0: float = 1.0,
    eps_min: float = 0.01,
    decay: float = 0.0005,
    max_steps: int = 22,
    progress_cb = None,
)-> dict[str, object]:
    """Fonction utilitaire haut-niveau : entraîne et renvoie les stats."""
    learner = TabularQLearning(
        env_name=env_name,
        alpha=alpha,
        gamma=gamma,
        eps0=eps0,
        eps_min=eps_min,
        decay=decay,
        max_steps=max_steps,
    )
    stats = learner.train(episodes, progress_cb)
    return stats

class TabularQLearning:
    """Agent Q-learning tabulaire pour *Taxi-v3*."""
     
    def __init__(
        self,
        env_name: str = "Taxi-v3",
        alpha: float = 0.2,
        gamma: float = 0.95,
        eps0: float = 1.0,
        eps_min: float = 0.01,
        decay: float = 0.0005,
        max_steps: int = 22,
    ):
        self.env_name = env_name
        self.alpha = alpha
        self.gamma = gamma
        self.eps0 = eps0
        self.eps_min = eps_min
        self.decay = decay
        self.max_steps = max_steps

        self.env = gym.make(self.env_name)
        self.qtable = np.zeros(
            (self.env.observation_space.n, self.env.action_space.n), dtype=np.float32
        )
        self.rewards = None
        self.steps = None
        self.epsilons = None
        self.wall_times = None
        self.cpu_times = None
        self.total_wall = 0.0
        self.total_cpu = 0.0

    def train(self, episodes: int = 5000, progress_cb=None):
        rng = np.random.default_rng()
        self.epsilons = np.maximum(
            self.eps0 - self.decay * np.arange(episodes), self.eps_min
        )

        self.rewards = np.zeros(episodes, np.int16)
        self.steps = np.zeros(episodes, np.int16)
        self.wall_times = np.zeros(episodes, np.float32)
        self.cpu_times = np.zeros(episodes, np.float32)
        self.success = np.zeros(episodes, np.int8)

        tic_global_wall = perf_counter()
        tic_global_cpu = process_time()

        for ep in range(episodes):
            tic_w = perf_counter()
            tic_c = process_time()

            s, _ = self.env.reset(seed=ep)
            tot_r = n_s = 0
            for _ in range(self.max_steps):
                n_s += 1
                # ε-greedy
                if rng.random() < self.epsilons[ep]:
                    a = rng.integers(self.env.action_space.n)
                else:
                    a = int(self.qtable[s].argmax())

                ns, r, term, trunc, _ = self.env.step(a)
                # Q‑Learning update
                self.qtable[s, a] += self.alpha * (
                    r + self.gamma * self.qtable[ns].max() - self.qtable[s, a]
                )
                tot_r += r
                s = ns
                
                if term and r == 20:          # épisode réussi
                    succ_flag = 1
                else:
                    succ_flag = 0
                
                if term or trunc:
                    break

            self.rewards[ep] = tot_r
            self.steps[ep] = n_s
            self.wall_times[ep] = perf_counter() - tic_w
            self.cpu_times[ep] = process_time() - tic_c
            self.success[ep] = succ_flag

            if progress_cb is not None:
                if callable(progress_cb):
                    progress_cb((ep + 1) / episodes)
                elif hasattr(progress_cb, "progress"):
                    progress_cb.progress((ep + 1) / episodes,
                             text=f"Episode {ep + 1}/{episodes}")

        self.total_wall = perf_counter() - tic_global_wall
        self.total_cpu = process_time() - tic_global_cpu
        

        stats = {
            "qtable": self.qtable,
            "rewards": self.rewards,
            "steps": self.steps,
            "epsilons": self.epsilons,
            "wall_times": self.wall_times,
            "cpu_times": self.cpu_times,
            "smooth_rewards": pd.Series(self.rewards).rolling(100).mean(),
            "smooth_steps": pd.Series(self.steps).rolling(100).mean(),
            "total_wall": self.total_wall,
            "total_cpu": self.total_cpu,
            "success": self.success,
            "smooth_success": pd.Series(self.success).rolling(100).mean(),
        }
        return stats

    def act(self, state: int) -> int:
        return int(self.qtable[state].argmax())

    def play_greedy(self, render_mode: str = "ansi"):
        env = gym.make(self.env_name, render_mode=render_mode)
        s, _ = env.reset()
        done = False
        frames = []
        while not done:
            frames.append(env.render())
            s, _, term, trunc, _ = env.step(self.act(s))
            done = term or trunc
        env.close()
        return "".join(frames)
