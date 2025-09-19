from torch import nn, optim
from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)
class Remsait(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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

loss = nn.CrossEntropyLoss()
remsait = Remsait()
# writer = SummaryWriter("logs")
for data in dataloader:
    imgs, targets = data
    outputs = remsait(imgs)
    # print(outputs)
    # print(targets)
    result_loss = loss(outputs, targets)
    # result_loss.backward()
    print(result_loss)