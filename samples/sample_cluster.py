import ast
import csv
import sys

import pprint

import numpy as np

from sklearn.cluster import KMeans


def main(args):
    input_data_file = 'data/training_data.csv'
    input_stats_file = 'data/training_data_norm_stats.csv'

    output_data_file = 'data/state_cluster_centers.csv'


if __name__ == '__main__':
    main(sys.argv)

    sys.exit(0)