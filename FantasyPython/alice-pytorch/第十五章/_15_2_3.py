import numpy as np
import dlib
import matplotlib.image as mpimg
import cv2
import imageio
from pathlib import Path
import os
from tqdm import tqdm
shape = 144

def clip_image(image, boundary):
    top = np.clip(boundary.top(), 0, np.Inf).astype(np.int16)
    bottom = np.clip(boundary.bottom(), 0, np.Inf).astype(np.int16)
    left = np.clip(boundary.left(), 0, np.Inf).astype(np.int16)
    right = np.clip(boundary.right(), 0, np.Inf).astype(np.int16)
    image = cv2.resize(image[top:bottom, left:right],(128,128))
    return image

def fun(file_dirs):

    for file_dir in tqdm(file_dirs):
        image_path_list = list(file_dir.glob('*.jpg'))
        for image_path in image_path_list:
            image = np.array(mpimg.imread(image_path))
            boundarys = detector(image, 2)
            if len(boundarys) == 1:
                image_new = clip_image(image, boundarys[0])
                os.remove(image_path)
                image_path_new = image_path #这里可以对保存的地点调整路径
                imageio.imsave(image_path_new, image_new)
            else:
                os.remove(image_path)

import os
# 这个是列出所有目录下文件夹的函数
def list_folders(path):
    """
    列出指定路径下的所有文件夹名
    """
    folders = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            folders.append(os.path.join(root, dir))
    return folders

def list_files(path):
    files = []
    for item in os.listdir(path):
        file = os.path.join(path, item)
        if os.path.isfile(file):
            files.append(file)
    return files

if __name__ == '__main__':

    detector = dlib.get_frontal_face_detector() #切割器
    path="./dataset/lfw-deepfunneled"
    path = Path(path)
    file_dirs = [x for x in path.iterdir() if x.is_dir()]

    print(len(file_dirs))
    fun(file_dirs)

    folders = list_folders(path)
    path_file_collect = []
    cutoff = 10
    for folder in folders:
        files = list_files(folder)
        if len(files) >= cutoff:
            path_file_collect += (files)

    path_file = "./dataset/lfw-path_file.txt"
    file2 = open(path_file, 'w+')
    for line in path_file_collect:
        file2.write(line)
        file2.write("\n")
    file2.close()

