from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("E:\\tobebetter\\learn_pytorch\\data\\train\\ants_image\\0013035.jpg")
# print(img)   size = 768 × 512 宽 高

"""
oooo = transforms.xxxx()都是创建一个xxxx实例 并用__init__初始化
yyyy = oooo(kk)就是调用 __call__ 方法
用 ctrl+左键 来查看官方文档的说明 输入的参数看__init__默认的不用输 变量需要输入
"""

# ToTensor() 使用方法 转换为张量
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
# print(img_tensor.shape) shape = 3 512 768 通道 高 宽
# print(img_tensor)
writer.add_image("totensor", img_tensor)
print("——————————————")

# Normalize() 使用方法 标准化
# output[channel] = (input[channel] - mean[channel]) / std[channel]  三通道对应分别计算
# mean 是均值， std 是标准差，Normalize()的参数就是均值和标准差
trans_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_norm = trans_norm(img_tensor)
# print(img_norm)
# img_tensor 是图像的张量，数值分布是 [0, 1]
# img_norm 是图像的张量标准化后，数值分布是 均值为0，标准差为1 
print(img_tensor[0][0][0])
print(img_norm[0][0][0])
writer.add_image("norm", img_norm)
# 未标准化前的tensor图像因为比例相同，图像没啥变化
# 标准化后每个像素都改变了，所以肉眼看颜色变化了，但是方便计算机识别
print("——————————————")

# Resize() 使用方法 改变高和宽
print(img.size)
trans_resize = transforms.Resize((768, 512)) # 高 宽
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize tens-or
img_resize = trans_totensor(img_resize)
print(img_resize.shape)
print("——————————————")

# Compose() 使用方法，合并功能，按写的transforms变换0列表执行流程
# compose() 需要一个包含 transforms类型的列表
trans_resize_2 = transforms.Resize((800,400)) # 只有一个参数x,将图片短边缩放至x,长宽比不变
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2= trans_compose(img)
print(img_resize_2.shape)
writer.add_image("Compose", img_resize_2, 1)

# RandomCrop() 使用方法 随机裁剪
trans_random = transforms.RandomCrop(50) # 随机裁剪 50×50 像素的正方形区域，如果指定两个值(50,100)就是高和宽
trans_compose_2 = transforms.Compose([trans_random, trans_totensor]) # 裁剪完变成张量
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("Random", img_crop, i)


writer.close()