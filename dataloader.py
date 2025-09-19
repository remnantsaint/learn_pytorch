from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms

# 准备的测试集
test_data = datasets.CIFAR10("./dataset", train=False, download=True, transform=transforms.ToTensor())

# num_workers=0单进程 drop_last=False最后数据不够batch数量是否丢弃
# shuffle=True打乱每轮 Epoch 顺序
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片及target\
img, target = test_data[0] # getitem() 方法直接返回 img 和 target
print(img.shape)
print(target)

writer = SummaryWriter("logs")
# test_loader设置批数量后，imgs targets 会进行打包
# TensorBoard 不会显示所有的 step，为了节省显存和前端性能，它会进行采样，只显示部分 step
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape) # 第一个值是 batch_size 大小
        # print(targets)
        # imgs 形状为 (N, C, H, W)，用于批量可视化应使用 add_images
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step += 1

writer.close()