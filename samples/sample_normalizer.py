import ast
import csv
import sys

import pprint

import numpy as np


NUM_OF_FEATURES = 23
NORM_STATS_FIELDS = ['row_counts', 'reward_max', 'reward_mean', 'state_mean', 'state_std']


def _parse_state(state):
    if state.startswith("[") and state.endswith("]"):
        state_str_vals = state[1:-1].split(",")

        state_value = []
        for val in state_str_vals:
            state_value.append(float(val.strip()))

        return np.array(state_value)
    else:
        raise ValueError("Invalid State Value: " + state)


def process_data(input_csv_file, output_stats_file):
    with open(input_csv_file, 'r') as input_csv, open(output_stats_file, 'w') as output_csv:
        csv_reader = csv.DictReader(input_csv)

        csv_writer = csv.DictWriter(output_csv, fieldnames=NORM_STATS_FIELDS)
        csv_writer.writeheader()

        row_counts = 0
        max_reward = -1
        reward_sum = 0

        state_counts = 0
        state_sum1 = np.zeros(NUM_OF_FEATURES)
        state_sum2 = np.zeros(NUM_OF_FEATURES)

        for row in csv_reader:
            raw_state = row['state']
            raw_reward = row['reward']
            raw_next_state = row['next_state']
            raw_done = row['done']

            state_val = _parse_state(raw_state)
            #pprint.pprint(state_val)

            state_sum1 = state_sum1 + state_val
            state_sum2 = state_sum2 + (state_val * state_val)
            state_counts += 1

            if ast.literal_eval(raw_done):
                next_state_val = _parse_state(raw_next_state)
                state_sum1 = state_sum1 + next_state_val
                state_sum2 = state_sum2 + (next_state_val * next_state_val)
                state_counts += 1

            reward_val = float(raw_reward)
            if reward_val > max_reward:
                max_reward = reward_val

            reward_sum = reward_sum + reward_val

            row_counts += 1

        reward_mean = reward_sum / row_counts

        state_mean = state_sum1 / state_counts
        state_std = np.sqrt((state_sum2 / state_counts) - (state_mean * state_mean))

        csv_writer.writerow({'row_counts':   str(row_counts),
                     'reward_max':   str(max_reward),
                     'reward_mean':  str(reward_mean),
                     'state_mean':   str(state_mean),
                     'state_std':    str(state_std)
                     })


def main(args):
    input_data_file = "data/training_data.csv"
    output_data_file = "data/training_data_norm_stats.csv"

    process_data(input_data_file, output_data_file)


if __name__ == '__main__':
    main(sys.argv)

    sys.exit(0)