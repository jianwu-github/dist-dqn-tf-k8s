import csv
import random
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


def _distance(x, y):
    return np.sqrt(np.sum((x - y) * (x - y)))


class SampleSimulator(Env):
    """
    Simulating Sample Environment
    """

    def __init__(self, norm_stats_csv_file, cluster_center_csv_file, reward_dist_json_file):
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

            for key, value in self._cluster_centers.items():
                curr_centroid = _parse_state(value)
                distance = _distance(normalized_state - curr_centroid)

                if distance < min_distance:
                    min_distance = distance
                    cluster_id = key

            cluster_reward_dist = self._cluster_reward_dist[cluster_id]

            # sample the reward
            sample_val = random.random()
            sample_reward = 0.0

            for key, value in cluster_reward_dist.items():
                if sample_val < value:
                    sample_reward = key
                    break

            self._reward = sample_reward
        else:
            raise ValueError("Invalid Action Value: " + str(action))

        # compute next state
        next_state = []
        next_state[0] = self._state[0]
        next_state[1] = self._state[1]

        # ngiftall including the current campaign
        if self._action > 0 and self._reward > 0:
            next_state[2] = self._state[2] + 1
        else:
            next_state[2] = self._state[2]

        # numprom including current campaign
        if self._action > 0:
            next_state[3] = self._state[3] + 1
        else:
            next_state[3] = self._state[3]

        # frequency
        if float(next_state[3]) > 0:
            next_state[4] = float(next_state[2]) / float(next_state[3])
        else:
            next_state[4] = self._state[4]

        months = 0
        recency = 0
        lastgift = 0
        for prev_reward in self._prev_rewards:
            months += 1
            if prev_reward > 0:
                recency = months
                lastgift = prev_reward
                break

        next_state[5] = recency
        next_state[6] = lastgift

        # ramntall including current campaign
        if self._action > 0 and self._reward > 0:
            next_state[7] = self._state[7] + self._reward
        else:
            next_state[7] = self._state[7]

        months = 0
        nrecproms = 1 if self._action > 0 else 0
        for prev_action in self._prev_actions:
            months += 1
            if months <= 6:
                if prev_action > 0:
                    nrecproms += 1
            else:
                break

        next_state[8] = nrecproms

        months = 0
        nrecgifts = 1 if self._reward > 0 else 0
        for prev_reward in self._prev_rewards:
            months += 1
            if months <- 6:
                if prev_reward > 0:
                    nrecgifts += 1
            else:
                break

        next_state[9] = nrecgifts

        months = 0
        totrecamt = self._reward if self._reward > 0 else 0
        for prev_reward in self._prev_rewards:
            months += 1
            if months <- 6:
                if prev_reward > 0:
                    totrecamt += prev_reward
            else:
                break

        next_state[10] = totrecamt

        # 12. recamtpergift:    recent gift amount per gift(6mo.)
        # 13. recamtperprom:    recent gift amount per prom(6mo.)
        # 14. promrecency:      num. of months since last promotion
        # 15. timelag:          num. of mo’s from first prom to gift
        # 16. recencyratio:     recency / timelag
        # 17. promrecratio:     promrecency / timelag
        # 18. respondedbit[1]:  whether responded last month
        # 19. respondedbit[2]:  whether responded 2 months ago
        # 20. respondedbit[3]:  whether responded 3 months ago
        # 21. mailedbit[1]:     whether promotion mailed last month
        # 22. mailedbit[2]:     whether promotion mailed 2 mo’s ago
        # 23. mailedbit[3]:     whether promotion mailed 3 mo’s ago