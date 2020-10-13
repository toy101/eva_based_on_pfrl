import torch
import torch.nn as nn
import torch.nn.functional as F
from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default

from pfrl.action_value import DiscreteActionValue

class EVANetwork(nn.Module):

    def __init__(self, n_actions, n_output_channels=512, ration_lambda=0.5):
        super().__init__()
        self.embedding_size = n_output_channels
        self.ration_lambda=ration_lambda

        self.cnn = pnn.LargeAtariCNN(n_output_channels=self.embedding_size)
        self.output = nn.Linear(self.embedding_size, n_actions)
        self.apply(init_chainer_default)

    def forward(self, x, eva_flag=False):
        h = self.cnn(x)
        q = self.output(h)
        param_q = DiscreteActionValue(q)

        # TODO:Value buffer
        non_param_q = None
        if eva_flag:
            return self.ration_lambda*param_q + (1-self.ration_lambda)*non_param_q, h
        else:
            return param_q, h

