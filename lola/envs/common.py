import gym
import numpy as np

from gym.utils import seeding


class OneHot(gym.Space):
    """
    One-hot space. Used as the observation space.
    """
    def __init__(self, n, seed=None):
        self.n = n
        self.np_random, _ = seeding.np_random(seed)
        # Optionally initialize the parent with shape and dtype info.
        super().__init__(shape=(n,), dtype=np.int32)

    def sample(self):
        return self.np_random.multinomial(1, [1. / self.n] * self.n)

    def contains(self, x):
        return isinstance(x, np.ndarray) and \
               x.shape == (self.n,) and \
               np.all(np.logical_or(x == 0, x == 1)) and \
               np.sum(x) == 1

    @property
    def shape(self):
        return (self.n,)

    def __repr__(self):
        return "OneHot(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, OneHot) and self.n == other.n
