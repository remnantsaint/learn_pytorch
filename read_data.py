from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir): # 初始化定义一些变量
        # root_dir = "dataset/train"  label_dir = "ants"
        # 这样组合是为了更好的分开蚂蚁和蜜蜂，也就是标签
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path) # 从文件夹内容获取列表 img_path

    # __开头和结尾的都是魔法方法，不用调用，在初始化时会自动调用
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) # 图像的文件路径
        img= Image.open(img_item_path) # 打开图像
        label= self.label_dir
        return img, label  # 返回单个图像和其标签

    def __len__(self):
        return len(self.img_path) # len(列表) 返回列表的长度

root_dir = "data/hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir) # 创建一个蚂蚁数据集
bees_dataset = MyData(root_dir, bees_label_dir) # 创建一个蜜蜂数据集
'''
ants_dataset[0][0].show() # 第一个0是索引，指第一个文件；第二个0是自动返回的img，
print(ants_dataset[0][1]) # [0][1]就是label
'''
img, label = ants_dataset[0]
img.show()

train_dataset = ants_dataset + bees_dataset 
# 只有定义了__len__方法，才能相加
# 顺序是前后两个数据集的拼接
print(len(train_dataset))