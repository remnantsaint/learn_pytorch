from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets
import torchvision
import torch

dataset = datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False, num_workers=0)

class Conv(nn.Module):
    def __init__(self) -> None:
        super().__init__() # 调用父类的构造函数
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0) # 进行一次卷积变换
    
    def forward(self, x):
        x = self.conv1(x)
        return x

conv = Conv()

writer = SummaryWriter("logs")

step = 0
# 输入图片像素是 32×32
for data in dataloader:
    imgs, targets = data
    output = conv(imgs) # __call__ 方法自动调用前向
    # print(imgs.shape)
    # print(output.shape) # [图片数量，通道数，卷积操作后的高宽]
    writer.add_images("input", imgs, step)
    output = torch.reshape(output,(-1, 3, 30, 30)) # 因为压缩了通道数，-1被计算成128
    writer.add_images("output", output, step)
    step += 1

writer.close()