import ast
import csv
import sys
import time

from contextlib import ExitStack

import pprint

import numpy as np

from sklearn.cluster import KMeans
from sklearn.externals import joblib


CLUSTER_FIELDS = ['label', 'centroid']


def _parse_state(state):
    if state.startswith("[") and state.endswith("]"):
        state_str_vals = state[1:-1].split(",") if state.find(",") >= 0 else state[1:-1].split()

        state_value = []
        for val in state_str_vals:
            state_value.append(float(val.strip()))

        return np.array(state_value)
    else:
        raise ValueError("Invalid State Value: " + state)


def build_state_cluster(input_csv_file, stats_csv_file, output_csv_file, output_model_file, num_of_clusters):
    with ExitStack() as ctx_stack:
        input_csv = ctx_stack.enter_context(open(input_csv_file, 'r'))
        input_csv_reader = csv.DictReader(input_csv)

        stats_csv = ctx_stack.enter_context(open(stats_csv_file, 'r'))
        stats_csv_reader = csv.DictReader(stats_csv)
        stats_row = next(stats_csv_reader)

        state_mean = _parse_state(stats_row['state_mean'])
        state_std = _parse_state(stats_row['state_std'])

        print("\nState Mean: ==================================================")
        pprint.pprint(state_mean)
        print("\nState Std: ===================================================")
        pprint.pprint(state_std)

        output_csv = ctx_stack.enter_context(open(output_csv_file, 'w'))
        output_csv_writer = csv.DictWriter(output_csv, fieldnames=CLUSTER_FIELDS)
        output_csv_writer.writeheader()

        state_list = []

        for row in input_csv_reader:
            raw_state = row['state']
            raw_next_state = row['next_state']
            raw_done = row['done']

            state_val = _parse_state(raw_state)
            norm_state = (state_val - state_mean) / state_std
            state_list.append(norm_state)

            if ast.literal_eval(raw_done):
                next_state_val = _parse_state(raw_next_state)
                next_norm_state = (next_state_val - state_mean) / state_std
                state_list.append(next_norm_state)

        # building cluster
        print("Building {} clusters for total of {} states ...".format(num_of_clusters, len(state_list)))
        start = time.process_time()
        kmeans = KMeans(n_clusters=num_of_clusters)
        kmeans.fit(state_list)
        elapsed_time = time.process_time() - start
        print("Done! The building time is {} seconds".format(elapsed_time))

        labels = kmeans.labels_
        print("num of labels: {}".format(len(labels)))

        centroids = kmeans.cluster_centers_
        for l in range(num_of_clusters):
            output_csv_writer.writerow({'label': l, 'centroid': str(centroids[l])})

        # persist KMeans Cluster model to disk
        joblib.dump(kmeans, output_model_file)


def main(args):
    num_of_clusters = 550

    input_data_file = 'data/training_data_v2.csv'
    input_stats_file = 'data/training_data_norm_stats_v2.csv'

    output_data_file = 'data/state_cluster_centers_v2.csv'
    output_model_file = 'data/kmeans_' + str(num_of_clusters) + "_model_v2.pkl"


    build_state_cluster(input_data_file, input_stats_file, output_data_file, output_model_file, num_of_clusters)


if __name__ == '__main__':
    main(sys.argv)

    sys.exit(0)
