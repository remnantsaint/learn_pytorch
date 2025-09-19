import torch
import torchvision
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

# train_data = ImageNet("./imagenet", split='train', download=True, transform=torchvision.transforms.ToTensor())
vgg16_False = torchvision.models.vgg16(pretrained=False)
vgg16_True = torchvision.models.vgg16(pretrained=True)
train_data = torchvision.datasets.CIFAR10('./dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader('train_data', batch_size=1)
# CIFAR10 是10分类，vgg最后是1000分类，所以我们要改动现有的模型，添加一层线性层即可
vgg16_True.classifier.add_module('relu', torch.nn.ReLU(inplace=True))
vgg16_True.classifier.add_module('drop', torch.nn.Dropout(p=0.5, inplace=False))
vgg16_True.classifier.add_module('add_linear', torch.nn.Linear(1000, 10, bias=True)) # 添加层
vgg16_False.classifier[6] = torch.nn.Linear(4096, 10, bias=True) # 修改模型
print(vgg16_True)