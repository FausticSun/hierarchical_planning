import numpy as np


class Tracker:
    def __init__(self, N) -> None:
        self.grid = np.zeros([N, N])
        self.grid[1, 1] = 1

    def observe(self, observations, rewards):
        for k, v in observations.items():
            if k == "global":
                continue
            location = tuple(v["location"])
            if rewards[k] == 1:
                self.grid[location] = 2
            elif self.grid[location] == 0:
                self.grid[location] = 1
