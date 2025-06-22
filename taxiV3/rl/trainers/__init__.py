from .q_learning import QLParams, train as train_ql
try:
    from .dqn import DQNParams, train as train_dqn
except ImportError:
    DQNParams = None
    train_dqn = None