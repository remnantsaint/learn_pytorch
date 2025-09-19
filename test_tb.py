from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import os

writer = SummaryWriter("logs") # 创建日志文件夹
# 在终端用 tensorboard --logdir=logs --port=6007 运行查看

image_path = os.path.join("data", "train", "ants_image", "0013035.jpg")
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
writer.add_image("test", img_array, 1, dataformats='HWC') # 添加图像
# 用 HWC 是因为 array 的 shape 中通道是最后一个参数
# 如果换张图片再运行，网页就能看到 step2 是第二张图片了

# for i in range(100):
#     writer.add_scalar("y=3x", 3*i, i) # 第一个参数是标签，第二个参数是 y，第三个参数是 x

# 左上角切换 scalars 和 images

writer.close()

# 观察训练时候，给 model 用了哪些数据，或者对 model 测试时看每一阶段的输出结果