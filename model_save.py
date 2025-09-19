import torch
from torch import nn
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=True)

# 保存方式 1，会保存所有的结构和参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式 2，只存参数，如权重偏置，更灵活，官方推荐
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 陷阱1
class Remsait(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
    def forward(self, x):
        x = self.conv1(x)
        return x
remsait = Remsait()
torch.save(remsait, "remsait_method1.pth")