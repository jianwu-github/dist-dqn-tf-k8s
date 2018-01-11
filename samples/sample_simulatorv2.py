import csv
import sys
import numpy as np

from collections import deque
from json_tricks import load
from gym import Env


_MAX_EPISODE_LENGTH = 20


def _parse_state(state):
    if state.startswith("[") and state.endswith("]"):
        state_str_vals = state[1:-1].split(",") if state.find(",") >= 0 else state[1:-1].split()

        state_value = []
        for val in state_str_vals:
            state_value.append(float(val.strip()))

        return np.array(state_value)
    else:
        raise ValueError("Invalid State Value: " + state)


class SampleSimulator(Env):
    """
    Simulating Sample Environment
    """

    def __init__(self, norm_stats_csv_file, cluster_center_csv_file, reward_dist_json_file):
        # reset env to start new simulation
        self._read_norm_stats_csv_file(norm_stats_csv_file)
        self._read_cluster_center_csv_file(cluster_center_csv_file)
        self._cluster_reward_dist = load(reward_dist_json_file, preserve_order=True)

        self._state = None
        self._action = None
        self._reward = None

        self._prev_states = deque(maxlen=_MAX_EPISODE_LENGTH)
        self._prev_actions = deque(maxlen=_MAX_EPISODE_LENGTH)
        self._prev_rewards = deque(maxlen=_MAX_EPISODE_LENGTH)

    def _read_norm_stats_csv_file(self, norm_stats_csv_file):
        csv_reader = csv.DictReader(open(norm_stats_csv_file, 'r'))
        norm_stats_row = csv_reader.next()

        self._state_mean = _parse_state(norm_stats_row['state_mean'])
        self._state_std = _parse_state(norm_stats_row['state_std'])

    def _read_cluster_center_csv_file(self, cluster_center_csv_file):
        csv_reader = csv.DictReader(open(cluster_center_csv_file, 'r'))

        self._cluster_centers = {}
        for row in csv_reader:
            cluster_id = int(row['label'])
            centroid = _parse_state(row['centroid'])
            self._cluster_centers[cluster_id] = centroid

    def reset(self):
        self._state = None
        self._action = None
        self._reward = None
        
        self._prev_states = deque(maxlen=_MAX_EPISODE_LENGTH)
        self._prev_actions = deque(maxlen=_MAX_EPISODE_LENGTH)
        self._prev_rewards = deque(maxlen=_MAX_EPISODE_LENGTH)

    def set_state(self, state):
        self.reset()
        self._state = state

    def step(self, action):
        if action == 0:
            self._action = 0
            self._reward = 0.0
        elif action == 1:
            self._action = 1

            # compute rewards for the action
            normalized_state = (self._state - self._state_mean) / self._state_std

            # find closest cluster
            min_distance = sys.float_info[0]
            cluster_id = -1




        else:
            raise ValueError("Invalid Action Value: " + str(action))