import csv
import random

from collections import deque

import numpy as np

import tensorflow as tf

class Sampler(object):
    """
    Sampler Class is adopted from PGQ Repository(https://github.com/abhishm/PGQ) by Abhishek Mishra
    """
    def __init__(self,
                 csv_file,
                 num_episodes=10,
                 max_step=20):
        self._num_episodes = num_episodes
        self._max_step = max_step
        self._csv_file = csv_file

        self._field_names = ("state", "action", "reward", "next_state", "done")
        self._csv_file_reader = self._get_csv_file_reader(self._field_names, self._csv_file)

    def _get_csv_file_reader(self, field_names, csv_file):
        with open(csv_file, 'rU') as data:
            reader = csv.DictReader(data, fieldnames=field_names)
            for row in reader:
                yield row

    def collect_one_episode(self):
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for t in xrange(self._max_step):
            try:
                data = next(self._csv_file_reader)
            except StopIteration:
                print("Reaching the end of file {}, reading from the beginning again".format(self._csv_file))
                self._csv_file_reader = self._get_csv_file_reader(self._field_names, self._csv_file)
                data = next(self._csv_file_reader)

            states.append(np.array(map(lambda x: float(x.strip()), data["state"][1:-1].split(","))))
            actions.append(1 if data["action"].strip() == "True" else 0)
            rewards.append(0.0 if data["reward"].strip() == "None" else float(data["reward"].strip()))
            next_states.append(np.array(map(lambda x: float(x.strip()), data["next_state"][1:-1].split(","))))
            dones.append(data["done"].lower() == "true")

        return dict(states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones
                )

    def collect_one_batch(self):
        episodes = []
        for i_episode in xrange(self._num_episodes):
            episodes.append(self.collect_one_episode())
        # prepare input
        states = np.concatenate([episode["states"] for episode in episodes])
        actions = np.concatenate([episode["actions"] for episode in episodes])
        rewards = np.concatenate([episode["rewards"] for episode in episodes])
        next_states = np.concatenate([episode["next_states"] for episode in episodes])
        dones = np.concatenate([episode["dones"] for episode in episodes])

        return dict(states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones
                )


class ReplayBuffer(object):
    """
    ReplayBuffer is adopted from PGQ Repository(https://github.com/abhishm/PGQ) by Abhishek Mishra
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_items = 0
        self.buffer = deque()

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def add(self, item):
        if self.num_items < self.buffer_size:
            self.buffer.append(item)
            self.num_items += 1
        else:
            self.buffer.popleft()
            self.buffer.append(item)

    def add_items(self, items):
        for item in items:
            self.add(item)

    def add_batch(self, batch):
        keys = ["states", "actions", "rewards", "next_states", "dones"]
        items = []
        for i in range(len(batch["states"])):
            item = []
            for key in keys:
                item.append(batch[key][i])
            items.append(item)
        self.add_items(items)

    def sample_batch(self, batch_size):
        keys = ["states", "actions", "rewards", "next_states", "dones"]
        samples = self.sample(batch_size)
        samples = zip(*samples)
        batch = {key: np.array(value) for key, value in zip(keys, samples)}
        return batch

    def count(self):
        return self.num_items

    def erase(self):
        self.buffer = deque()
        self.num_items = 0

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
tf.app.flags.DEFINE_string("task_index", "0", "Index of task within the job")
tf.app.flags.DEFINE_string("log_path", "/tmp/train", "Log path")
tf.app.flags.DEFINE_string("data_dir", "/data", "Data dir path")

# Hyperparameters
state_dim = 23
num_actions = 2

discount = 0.9
learning_rate = 0.00001
target_update_rate = 0.5

sample_size = 32
num_of_episodes_for_batch = 10
replay_buffer_size = 10000

synchronized_training = True

sample_csv_file = "/dqn-training-data/dqn_training_samples.csv"

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

def getDataReader():
    pass

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    task_index = int(FLAGS.task_index)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        is_chief = task_index == 0
        checkpointDir = "/dqn-training-data/train_logs/worker-" + str(task_index)

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):

            global_step = tf.Variable(0, name='global_step', trainable=False)

            adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            if synchronized_training:
                # synchronized training across multiple workers
                optimizer = tf.train.SyncReplicasOptimizer(adam_optimizer,
                                                           replicas_to_aggregate=len(worker_hosts),
                                                           total_num_replicas=len(worker_hosts),
                                                           use_locking=True
                                                           )
            else:
                # or no synchronized training
                optimizer = adam_optimizer

            # create input placeholders
            with tf.name_scope("inputs"):
                states = tf.placeholder(tf.float32, (None, state_dim), "states")
                actions = tf.placeholder(tf.int32, (None,), "actions")
                rewards = tf.placeholder(tf.float32, (None,), "rewards")
                next_states = tf.placeholder(tf.float32, (None, state_dim), "next_states")
                dones = tf.placeholder(tf.bool, (None,), "dones")
                one_hot_actions = tf.one_hot(actions, num_actions, axis=-1)

            # create variables for q-network
            with tf.name_scope("action_values"):
                with tf.variable_scope("q_network"):
                    q_values = q_network(states)
            with tf.name_scope("action_scores"):
                action_scores = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), reduction_indices=1)

            # create variables for target-network
            with tf.name_scope("target_values"):
                not_the_end_of_an_episode = 1.0 - tf.cast(dones, tf.float32)
                with tf.variable_scope("target_network"):
                    target_q_values = q_network(next_states)
                max_target_q_values = tf.reduce_max(target_q_values, reduction_indices=1)
                max_target_q_values = tf.multiply(max_target_q_values, not_the_end_of_an_episode)
                target_values = rewards + discount * max_target_q_values

            # create variables for optimization
            with tf.name_scope("optimization"):
                loss = tf.reduce_mean(tf.square(action_scores - target_values))
                loss = tf.cond(loss <= 0.5, lambda: 0.5 * tf.pow(loss, 2), lambda: 0.5 * loss - 0.125, name="huber_loss")
                trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
                gradients = optimizer.compute_gradients(loss, var_list=trainable_variables)
                train_op = optimizer.apply_gradients(gradients, global_step)

            # create variables for target network update
            with tf.name_scope("target_network_update"):
                target_ops = []
                q_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_network")
                target_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_network")
                for v_source, v_target in zip(q_network_variables, target_network_variables):
                    target_op = v_target.assign_sub(target_update_rate * (v_target - v_source))
                    target_ops.append(target_op)
                target_update = tf.group(*target_ops)

            # The StopAtStepHook handles stopping after running given steps.
            session_hooks = [tf.train.StopAtStepHook(last_step=100)]

            # Create the hook which handles initialization and queues.
            if synchronized_training:
                make_session_hook = optimizer.make_session_run_hook(is_chief)
                session_hooks.append(make_session_hook)
                session_hooks.reverse()

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(task_index == 0),
                                                   checkpoint_dir=checkpointDir,
                                                   hooks=session_hooks
                                                   ) as mon_sess:

                # Sampler (collect trajectories recorded in sample_csv_file)
                sampler = Sampler(sample_csv_file, num_of_episodes_for_batch, sample_size)

                # Initializing ReplayBuffer
                replay_buffer = ReplayBuffer(replay_buffer_size)

                i = 0
                while not mon_sess.should_stop():
                    print("Entering step {} ...".format(i))
                    batch = sampler.collect_one_batch()
                    replay_buffer.add_batch(batch)

                    random_batch = replay_buffer.sample_batch(sample_size)  # replay buffer

                    # mon_sess.run handles AbortedError in case of preempted PS.
                    gs_val, loss_val, _ = mon_sess.run([global_step, loss, train_op],
                                                        {states: random_batch["states"],
                                                        actions: random_batch["actions"],
                                                        rewards: random_batch["rewards"],
                                                        next_states: random_batch['next_states'],
                                                        dones: random_batch["dones"]})

                    mon_sess.run(target_update)

                    print("The loss and global step at worker {} local step {} is {} and {}".format(task_index, i, loss_val, gs_val))
                    i += 1

            print("Training on Worker {} has finished!".format(task_index))

if __name__ == "__main__":
    tf.app.run()
