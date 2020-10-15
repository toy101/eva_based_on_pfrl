import torch
from pfrl.collections.random_access_queue import RandomAccessQueue

from lsm_knn_buffer import LSMKNNBuffer as lkb

class ValueBuffer:

    def __init__(self, capacity:int, n_dim:int, dtype=torch.float32):
        self.capacity = capacity
        self.n_dim = n_dim
        self.memory = RandomAccessQueue(maxlen=capacity)
        self.h_memory = lkb(capacity=capacity, n_dim=n_dim)

    def store(self, features, q_values):
        pass

    def get_non_param_q(self):
        pass

    def _lookup(self):
        pass