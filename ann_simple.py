import torch.nn as nn
from ann_base import ANNBase


class ANNSimple(ANNBase):
    def __init__(self, train_x, train_y, test_x, test_y, validation_x, validation_y):
        super().__init__(train_x, train_y, test_x, test_y, validation_x, validation_y)
        self.linear = nn.Sequential(
            nn.Linear(11,6),
            nn.LeakyReLU(),
            nn.Linear(6,1)
        )

    def forward(self,x):
        return self.linear(x)

