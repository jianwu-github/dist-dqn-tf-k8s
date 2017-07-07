import math
import os
import numpy as np

import tensorflow as tf

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epoch_number', None, 'Number of epochs to run trainer.')
flags.DEFINE_string("output_dir", "./tensorboard/", "indicates training output")

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("log_path", "/tmp/train", "Log path")
tf.app.flags.DEFINE_string("data_dir", "/data", "Data dir path")

state_dim = 20
num_actions = 2

def q_network(states):
    W1 = tf.get_variable("W1", [state_dim, 20],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [20],
                         initializer=tf.constant_initializer(0))
    z1 = tf.matmul(states, W1) + b1
    h1 = tf.nn.elu(z1)

    W2 = tf.get_variable("W2", [20, 20],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [20],
                         initializer=tf.constant_initializer(0))
    z2 = tf.matmul(h1, W2) + b2
    h2 = tf.nn.elu(z2)

    W3 = tf.get_variable("W3", [20, num_actions],
                         initializer=tf.random_normal_initializer())
    b3 = tf.get_variable("b3", [num_actions],
                         initializer=tf.constant_initializer(0))
    q = tf.matmul(h2, W3) + b3

    return q

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # create input placeholders
            with tf.name_scope("inputs"):
                states = tf.placeholder(tf.float32, (None, self.state_dim), "states")
                actions = tf.placeholder(tf.int32, (None,), "actions")
                rewards = tf.placeholder(tf.float32, (None,), "rewards")
                next_states = tf.placeholder(tf.float32, (None, self.state_dim), "next_states")
                dones = tf.placeholder(tf.bool, (None,), "dones")
                one_hot_actions = tf.one_hot(actions, num_actions, axis=-1)

            # create variables for q-network
            with tf.name_scope("action_values"):
                with tf.variable_scope("q_network"):
                    q_values = q_network(states)
            with tf.name_scope("action_scores"):
                action_scores = tf.reduce_sum(tf.mul(q_values, one_hot_actions), reduction_indices=1)


if __name__ == "__main__":
    tf.app.run()
