"""Package TaxiV3 – expose une API propre pour Sphinx et pour les utilisateurs."""

from .QLearning import TabularQLearning, train_tabular

QLearningAgent = TabularQLearning

from .DeepQLearning import DeepQLearning

__all__ = [
    "train_tabular",
    "TabularQLearning",
    "QLearningAgent",
    "DeepQLearning",
]