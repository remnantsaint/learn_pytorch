import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets
import torchvision

dataset = datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False, num_workers=0)
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# input = torch.reshape(input, (-1, 1, 5, 5)) # -1被计算成 1
# print(input.shape)

class Conv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True) # stride 默认等于 kernel_size
        # 池化操作只支持浮点类型（float32/float64）或者半精度（float16/bfloat16）
    def forward(self, x):
        output = self.maxpool1(x)
        return output

conv = Conv()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = conv(imgs) # 池化操作通道数不变
    writer.add_images("output", output, step)
    print(imgs.shape, " ", output.shape)
    step += 1

writer.close()
