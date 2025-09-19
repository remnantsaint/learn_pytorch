import torch
from torch import nn
import torchvision
from model_save import *

# 方式 1 保存方式1加载模型
model1 = torch.load("vgg16_method1.pth")
# print(model1)

# 方式 2 给模型加载保存的参数
model2 = torchvision.models.vgg16(pretrained=False)
model2.load_state_dict(torch.load("vgg16_method2.pth"))
# print(model2)

# 陷阱1
# class Remsait(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
model = torch.load('remsait_method1.pth')
print(model) # 直接输出会报错，需要把模型那个类复制过来，或者添加头文件，但是不需要创建对象