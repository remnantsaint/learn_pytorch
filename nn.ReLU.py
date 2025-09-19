import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets
import torchvision

input = torch.tensor([[1.0, -0.5],[-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))

dataset = datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False, num_workers=0)

class Nen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()
    
    def forward(self, x):
        output = self.sigmoid1(x)
        return output

nen = Nen()
# output = relu(input) # 正数不变，负数为0
# print(output)

step = 0
writer = SummaryWriter("logs")
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = nen(imgs)
    writer.add_images("output", output, step)
    step += 1
writer.close()