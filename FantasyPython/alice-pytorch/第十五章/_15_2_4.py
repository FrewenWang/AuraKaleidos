from torch.utils.data import DataLoader, Dataset
import random
import linecache
import torch
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self,path_file,transform=None,should_invert=False):	# path_file是所有人脸图片的地址，每行地址是一个图片
        self.transform=transform
        self.should_invert=should_invert
        self.path_file = path_file

    def __getitem__(self, index):
        line=linecache.getline(self.path_file,random.randint(1,self.__len__()))
        img0_list=line.split("\\")
        #若为0，取得不同人的图片
        shouled_get_same_class=random.randint(0,1)
        if shouled_get_same_class:
            while True:
                img1_list=linecache.getline(self.path_file,random.randint(1,self.__len__())).split('\\')
                if img0_list[-1]==img1_list[-1]:
                   break

        else:
            while True:
                img1_list=linecache.getline(self.path_file,random.randint(1,self.__len__())).split('\\')
                if img0_list[-1]!=img1_list[-1]:
                    break

        img0_path = "/".join(img0_list).replace("\n","")
        img1_path = "/".join(img1_list).replace("\n","")

        im0=Image.open(img0_path).convert('L')
        im1=Image.open(img1_path).convert('L')


        im0 = torch.tensor(np.array(im0))
        im1 = torch.tensor(np.array(im1))

        return im0,im1,torch.tensor(shouled_get_same_class,dtype=torch.float32)

    def __len__(self):
        fh=open(self.path_file,'r')
        num=len(fh.readlines())
        fh.close()
        return num




if __name__ == '__main__':

    path_file = "./dataset/lfw-path_file.txt"
    ds = MyDataset(path_file)
    for _ in range(1024):
        a,b,l = ds.__getitem__(0)
        print(a.shape)
        print(b.shape)
        print(l)
        print("----------------")