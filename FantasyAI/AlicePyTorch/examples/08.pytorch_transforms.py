from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms

class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)
        with open(label_item_path, 'r') as f:
            label = f.readline()

        #  进行过进行transform进行处理
        if self.transform:
            img = transform(img)


        return img, label

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)

#  transforms 的作用是把图片缩放到400*400，然后把图片转换成张量
#  transforms的用法：
#  Compose:     顺序执行
#  ToTensor:    把图片转换成张量
#  Resize:      把图片缩放到指定大小
#  PILToTensor: 把PIL图片转换成张量
transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor()])
root_dir = "data/hymenoptera_data/train"
image_ants = "ants"
label_ants = "ants_label"
ants_dataset = MyData(root_dir, image_ants, label_ants, transform=transform)
image_bees = "bees_image"
label_bees = "bees_label"
bees_dataset = MyData(root_dir, image_bees, label_bees, transform=transform)




