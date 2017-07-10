import csv
import numpy as np

def _get_csv_file_reader(field_names, csv_file):
    with open(csv_file, 'rU') as data:
        reader = csv.DictReader(data, fieldnames=field_names)
        for row in reader:
            yield row


class Sampler(object):
    """
    Sampler Class is adopted from PGQ Repository (https://github.com/abhishm/PGQ)
    by Abhishek Mishra
    """
    def __init__(self,
                 csv_file,
                 num_episodes=10,
                 max_step=20):
        self._num_episodes = num_episodes
        self._max_step = max_step
        self._csv_file = csv_file

        self._field_names = ("state", "action", "reward", "next_state", "done")
        self._csv_file_reader = _get_csv_file_reader(self._field_names, self._csv_file)

    def collect_one_episode(self):
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for t in xrange(self._max_step):
            try:
                data = next(self._csv_file_reader)
            except StopIteration:
                print("Reaching the end of file {}, reading from the beginning again".format(self._csv_file))
                self._csv_file_reader = _get_csv_file_reader(self._field_names, self._csv_file)
                data = next(self._csv_file_reader)

            states.append(np.array(map(lambda x: float(x.strip()), data["state"][1:-1].split(","))))
            actions.append(float(data["action"].strip()))
            rewards.append(float(data["reward"].strip()))
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
        for i_episode in xrange(self.num_episodes):
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


