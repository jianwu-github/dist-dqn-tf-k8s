from gym import Env
from gym import spaces

import numpy as np


class SampleSimulator(Env):
    """
    Simulating Sample Environment
    """

    def __init__(self):
        # reset env to start new simulation
        self.reset()
