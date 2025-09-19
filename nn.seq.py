import torch
from torch import nn
from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, Sequential
from torch.utils.tensorboard.writer import SummaryWriter

class Remsait(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 卷积前后尺寸不变的话，其他参数默认，padding = (kernel_size - 1) / 2
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x
remsait = Remsait()
print(remsait)
input = torch.ones(64, 3, 32, 32) # batch_size channels H W
output = remsait(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(remsait, input) # 画一个模型图
writer.close()