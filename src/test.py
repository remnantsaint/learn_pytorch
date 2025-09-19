from PIL import Image
import torchvision
from torchvision.transforms import Resize, ToTensor
from model import *
import torch

image_path = 'dog.png'
image = Image.open(image_path)
print(image)
image = image.convert('RGB') # png是四通道，多一个透明度通道

# 截图保留的图片像素宽高不同, Reize成模型输入像素大小
transform = torchvision.transforms.Compose([Resize((32, 32)), ToTensor()])

image = transform(image)
image = torch.reshape(image, (1, 3, 32, 32))
# print(image.shape)

model = torch.load("remsait_9.pth", map_location=torch.device('cpu'))
# print(model)

model.eval()
with torch.no_grad():
    output = model(image)
    print(output)
    print(output.argmax(1)) # 输出预测值最大的标签下标，下标从 0 开始