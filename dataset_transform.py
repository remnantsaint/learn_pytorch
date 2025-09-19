from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

dataset_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])
# train=True代表划分的是训练集，false代表是测试集   transform参数是用到的变换
train_set = datasets.CIFAR10("./dataset", train=True, download=True, transform=dataset_transform)
test_set = datasets.CIFAR10("./dataset", train=False, download=True, transform=dataset_transform)

# img, target = test_set[0]
# print(test_set.classes)
# print(target) # 标签
# print(test_set.classes[target])
# img.show()

writer = SummaryWriter("logs")
# tensorboard --logdir=logs
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i) # img已经转换为了张量

writer.close()