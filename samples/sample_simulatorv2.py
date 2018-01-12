import csv
import random
import sys
import numpy as np

from collections import deque
from json_tricks import load
from gym import Env


_MAX_EPISODE_LENGTH = 17


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

        recamtpergift = 0.0
        if totrecamt > 0 and nrecgifts > 0:
            recamtpergift = float(totrecamt) / nrecgifts

        next_state[11] = recamtpergift

        recamtperprom = 0.0
        if totrecamt > 0 and nrecproms > 0:
            recamtperprom = float(totrecamt) / nrecproms

        next_state[12] = recamtperprom

        months = 0
        promrecency = 0
        for prev_action in self._prev_actions:
            months += 1
            if prev_action > 0:
                promrecency = months
                break

        next_state[13] = promrecency

        prom_months = 0
        first_prom = 0
        for prev_action in self._prev_actions:
            prom_months += 1
            if prev_action > 0:
                first_prom = prom_months

        gift_months = 0
        first_gift = 0
        for prev_reward in self._prev_rewards:
            gift_months += 1
            if prev_reward > 0:
                first_gift = gift_months

        timelag = 0
        if first_prom > 0 and first_gift > 0:
            timelag = first_gift - first_prom
            if timelag == 0:
                # For timelag happened within a month in the same campaign,
                # using 0.5 as estimated value to generate non-zero recencyratio
                # and promrecratio
                timelag = 0.5
            elif timelag < 0:
                timelag = 0

        next_state[14] = timelag

        recencyratio = float(recency) / timelag if timelag > 0 else 0.0
        promrecratio = float(promrecency) / timelag if timelag > 0 else 0.0

        next_state[15] = recencyratio
        next_state[16] = promrecratio

        respondedbit1 = 0
        respondedbit2 = 0
        respondedbit3 = 0
        gift_months = 0
        for prev_reward in self._prev_rewards:
            gift_months += 1
            if prev_reward > 0:
                if gift_months == 1:
                    respondedbit1 = 1
                elif gift_months == 2:
                    respondedbit2 = 1
                elif gift_months == 3:
                    respondedbit3 = 1

        next_state[17] = respondedbit1
        next_state[18] = respondedbit2
        next_state[19] = respondedbit3

        mailedbit1 = 0
        mailedbit2 = 0
        mailedbit3 = 0
        prom_months = 0
        for prev_action in self._prev_actions:
            prom_months += 1
            if prev_action > 0:
                if prom_months == 1:
                    mailedbit1 = 1
                elif prom_months == 2:
                    mailedbit2 = 1
                elif prom_months == 3:
                    mailedbit3 = 1

        next_state[20] = mailedbit1
        next_state[21] = mailedbit2
        next_state[22] = mailedbit3

        self._prev_actions.appendleft(self._action)
        self._prev_rewards.appendleft(self._reward)
        self._prev_states.appendleft(self._state)

        curr_state = self._state
        self._state = next_state

        return (next_state, self._reward, len(self._prev_states) == _MAX_EPISODE_LENGTH, {})