from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

"""
python 的用法 -> tensor数据类型
通过 transforms.ToTensor 去解决两个问题

2. 为什么需要 Tensor 数据类型
深度学习模型无法直接处理 PIL 图像/python 数组，必须输入torch.Tensor类型
tensor 支持GPU加速、自动求导,并且"C H W"格式更符合计算逻辑
"""
# 绝对路径 E:\tobebetter\learn_pytorch\data\train\ants_image\0013035.jpg
# 相对路径 data\train\ants_image\0013035.jpg
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1. transforms 应该如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# 输出图片的张量，三通道，每个通道每个元素是一个像素的归一化值

writer.add_image("Tensor_img", tensor_img)

writer.close()

