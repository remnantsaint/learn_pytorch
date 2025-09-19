from torch.utils.data import DataLoader
import torchvision
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear

# 搭建神经网络
class Remsait(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

# 被调用时不运行，运行该代码时才运行
if __name__ == '__main__':
    remsait = Remsait()
    input = torch.ones(64, 3, 32, 32)
    output = remsait(input)
    print(output.shape)