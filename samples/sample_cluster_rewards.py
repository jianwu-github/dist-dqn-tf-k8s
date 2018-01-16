import csv
import sys

from collections import OrderedDict
from contextlib import ExitStack
from operator import itemgetter

from json_tricks import dump

import pprint

import numpy as np

from sklearn.cluster import KMeans
from sklearn.externals import joblib


def _parse_state(state):
    if state.startswith("[") and state.endswith("]"):
        state_str_vals = state[1:-1].split(",") if state.find(",") >= 0 else state[1:-1].split()

        state_value = []
        for val in state_str_vals:
            state_value.append(float(val.strip()))

        return np.array(state_value)
    else:
        raise ValueError("Invalid State Value: " + state)


def _parse_cluster(cluster):
    if cluster.startswith("[") and cluster.endswith("]"):
        cluster_str_vals = cluster[1:-1].split(",") if cluster.find(",") >= 0 else cluster[1:-1].split()

        cluster_value = []
        for val in cluster_str_vals:
            cluster_value.append(float(val.strip()))

        return np.array(cluster_value)
    else:
        raise ValueError("Invalid Cluster Value: " + cluster)


def count_cluster_rewards_dist(input_data_file, input_stats_file, input_cluster_file,
                               input_model_file, num_of_clusters, output_data_file):
    kmeans_model = joblib.load(input_model_file)

    with ExitStack() as ctx_stack:
        input_csv = ctx_stack.enter_context(open(input_data_file, 'r'))
        input_csv_reader = csv.DictReader(input_csv)

        stats_csv = ctx_stack.enter_context(open(input_stats_file, 'r'))
        stats_csv_reader = csv.DictReader(stats_csv)
        stats_row = next(stats_csv_reader)

        state_mean = _parse_state(stats_row['state_mean'])
        state_std = _parse_state(stats_row['state_std'])

        print("\nState Mean: ==================================================")
        pprint.pprint(state_mean)
        print("\nState Std: ===================================================")
        pprint.pprint(state_std)

        cluster_csv = ctx_stack.enter_context(open(input_cluster_file, 'r'))
        cluster_csv_reader = csv.DictReader(cluster_csv)
        cluster_centroids = []

        for cluster_row in cluster_csv_reader:
            cluster_centroids.append({cluster_row['label']: _parse_cluster(cluster_row['centroid'])})

        cluster_no_actions = {}
        cluster_actions = {}
        cluster_rewards = {}
        cluster_reward_dist = OrderedDict()
        for cluster_id in range(num_of_clusters):
            cluster_no_actions[cluster_id] = 0
            cluster_actions[cluster_id] = 0
            cluster_rewards[cluster_id] = {}

        for row in input_csv_reader:
            raw_state = row['state']
            state_val = _parse_state(raw_state)
            norm_state = (state_val - state_mean) / state_std

            cluster_id = kmeans_model.predict(np.array([norm_state]))[0]

            action = row['action']
            if int(action) == 1:
                cluster_actions[cluster_id] += 1

                reward = row['reward']
                reward_val = float(reward)

                if reward_val in cluster_rewards[cluster_id]:
                    cluster_rewards[cluster_id][reward_val] += 1
                else:
                    cluster_rewards[cluster_id][reward_val] = 1
            else:
                cluster_no_actions[cluster_id] += 1

        # normalize reward distribution per cluster
        for cluster_id in range(num_of_clusters):
            curr_cluster_actions = cluster_actions[cluster_id]

            if curr_cluster_actions > 0:
                curr_cluster_rewards = cluster_rewards[cluster_id]
                reward_list = [(k, v) for (k, v) in curr_cluster_rewards.items()]
                reward_list = sorted(reward_list, key=itemgetter(0))
                normalized_reward_list = [(k, float(v) / float(curr_cluster_actions)) for (k, v) in reward_list]

                reward_dist = OrderedDict()
                reward_dist[normalized_reward_list[0][0]] = normalized_reward_list[0][1]
                for i in range(1, len(normalized_reward_list)):
                    reward_dist[normalized_reward_list[i][0]] = normalized_reward_list[i - 1][1] + normalized_reward_list[i][1]

                cluster_reward_dist[cluster_id] = reward_dist

        # write out cluster reward distribution to a json file with preserved order
        output_json = ctx_stack.enter_context(open(output_data_file, 'w'))
        dump(cluster_reward_dist, output_json, force_flush=True, sort_keys=False)


def main(args):
    num_of_clusters = 550

    input_data_file = 'data/training_data_v2.csv'
    input_stats_file = 'data/training_data_norm_stats_v2.csv'
    input_cluster_file = 'data/state_cluster_centers_v2.csv'
    input_model_file = 'data/kmeans_' + str(num_of_clusters) + "_model_v2.pkl"

    output_data_file = 'data/state_cluster_rewards_v2.json'

    count_cluster_rewards_dist(input_data_file, input_stats_file, input_cluster_file,
                               input_model_file, num_of_clusters, output_data_file)


if __name__ == '__main__':
    main(sys.argv)

    sys.exit(0)