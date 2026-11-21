from dataclasses import dataclass


@dataclass
class Hyperparameter:
    default: float
    interval: tuple[float, float]


min_freq = Hyperparameter(default=20, interval=(0, 5000))
max_freq = Hyperparameter(default=20000, interval=(0, 7000))
split = Hyperparameter(default=0.25, interval=(0.07, 0.4))
