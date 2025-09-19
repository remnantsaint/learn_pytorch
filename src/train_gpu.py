from torch import optim, nn
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
import torch
from torch.nn import CrossEntropyLoss
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear
import time

# 准备数据集 注意要在终端进入 src 目录下，才能正常找到数据集
train_data = torchvision.datasets.CIFAR10('../dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10('../dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 获取数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入网络模型
remsait = Remsait()
# if torch.cuda.is_available():
#     remsait = remsait.cuda()
remsait.to(device)

# 损失函数
loss_fn = CrossEntropyLoss()
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()
loss_fn.to(device)

# 优化器（随机梯度下降
learning_rate = 1e-2
optimizer = optim.SGD(remsait.parameters(), lr=learning_rate)


# 设置训练网络的一些参数
total_train_step = 0 # 记录训练次数
total_test_step = 0 # 记录测试次数
epoch = 10 # 训练轮数

# 添加 tensorboard
writer = SummaryWriter("../logs")

time_start = time.time()

# 训练
for i in range(epoch):
    print("-----第 {} 轮训练开始-----".format(i + 1))
    # 训练步骤开始
    remsait.train()
    for data in train_dataloader:
        imgs, targets = data
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        imgs.to(device)
        targets.to(device)
        outputs = remsait(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item())) # a.item()会打印数值
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 每轮训练完成后，应该进行测试
    remsait.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 关闭梯度计算
        for data in test_dataloader:
            imgs, targets = data
            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            imgs.to(device)
            targets.to(device)
            outputs = remsait(imgs) # 参数存在模型中
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum() # 沿着类别(列)维度比较不同样本
            total_accuracy += accuracy
    print("整体测试集上的 Loss: {}".format(total_test_loss))
    print(f"整体测试集上的正确率: {total_accuracy / test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # 保存每轮模型
    torch.save(remsait, "remsait_{}.pth".format(i))
    # torch.save(remsait.state_dict(), f"remsait_{i}.pth") # 官方推荐
    print(f"remsait_{i}.pth 模型已保存")

time_end = time.time()
print(f"运行时间: {time_end - time_start}")
writer.close()