
import numpy as np

#================================================================
class SumTree:
    '''### Sum Tree data structure for Prioritized Experience Replay
    Args:
        capacity (int): The capacity of the tree
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    #----------------------------------------
    def add(self, priority, data):
        '''### Add a new priority and data to the tree
        Args:
            priority (float): The priority of the data
            data (object): The data to store in the tree
        '''
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    #----------------------------------------
    def update(self, tree_index, priority):
        '''### Update the priority of a data in the tree
        Args:
            tree_index (int): The index of the data in the tree
            priority (float): The new priority of the data
        '''
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    #----------------------------------------
    def _propagate(self, tree_index, change):
        '''### Propagate the change in priority up the tree
        Args:
            tree_index (int): The index of the data in the tree
            change (float): The change in priority
        '''
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    #----------------------------------------
    def get_leaf(self, v):
        '''### Get the leaf node index, priority, and data for a given value
        Args:
            v (float): The value to search for
        Returns:
            leaf (int): The leaf node index
            priority (float): The priority of the leaf node
            data (object): The data stored in the leaf node
        '''
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            else:
                if v <= self.tree[left]:
                    parent = left
                else:
                    v -= self.tree[left]
                    parent = right
        data_index = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[data_index]

    #----------------------------------------
    def total_priority(self):
        '''### Get the total priority of the tree
        Returns:
            total_priority (float): The total priority of the tree
        '''
        return self.tree[0]

#================================================================
class PrioritizedReplayBuffer:
    '''### Prioritized Replay Buffer for Prioritized Experience Replay
    Args:
        capacity (int): The capacity of the replay buffer
        alpha (float): The alpha parameter for the prioritization
    '''
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.alpha = alpha

    #----------------------------------------
    def add(self, error, sample):
        '''### Add a new sample to the replay buffer
        Args:
            error (float): The error of the sample
            sample (tuple): The sample to store
        '''
        priority = (error + 1e-5) ** self.alpha
        self.tree.add(priority, sample)

    #----------------------------------------
    def sample(self, batch_size):
        '''### Sample a batch of data from the replay buffer
        Args:
            batch_size (int): The size of the batch to sample
        Returns:
            batch (list): The batch of data
            indices (list): The indices of the data in the tree
            sampling_probabilities (list): The sampling probabilities of the data
        '''
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority() / batch_size

        for i in range(batch_size):
            v = np.random.uniform(segment * i, segment * (i + 1))
            index, priority, data = self.tree.get_leaf(v)
            batch.append(data)
            indices.append(index)
            priorities.append(priority)
        sampling_probabilities = priorities / self.tree.total_priority()
        return batch, indices, sampling_probabilities

    #----------------------------------------
    def update(self, index, error):
        '''### Update the priority of a sample in the replay buffer
        Args:
            index (int): The index of the sample in the tree
            error (float): The error of the sample
        '''
        priority = (error + 1e-5) ** self.alpha
        self.tree.update(index, priority)
