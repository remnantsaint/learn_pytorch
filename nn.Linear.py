import torch
from torch.nn import Linear
from torch.utils.data import DataLoader
import torchvision

dataset = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Remsait(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(196608, 10)
    def forward(self, x):
        output = self.linear1(x)
        return output

remsait = Remsait()
for data in dataloader:
    imgs, targets = data
    # print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs) # 线性层要求是二维向量，所以要先展平（正确应该是保留batchsize
    # output = torch.flatten(imgs, start_dim=1)  # 从第1维开始展平，保留 batch 维度
    # print(output.shape)
    output = remsait(output)
    print(output.shape)
    

